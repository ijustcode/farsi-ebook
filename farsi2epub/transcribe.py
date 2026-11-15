"""Vision transcription orchestrator: fast pass, validation, escalation, sidecars.

Per page:
  1. Transcribe with the chosen model and resolution. The default is the
     strong model (Sonnet) on the high-res render (2576px) — measured best:
     the std-res pass made character-level errors (bidi-reversed dotted
     abbreviations like ه.ق -> ق.ه, word transpositions, lost diacritics)
     that hi-res avoided.
  2. Score it with deterministic validators (embedded-text cross-check for
     digital PDFs, script sanity for all pages) plus model confidence.
  3. On failure, escalate to the strong model on the hi-res render — unless
     the first pass was already strong + hi-res; keep whichever result scores
     better. This keeps the economical path (--res std, or --model haiku)
     safe: its failing pages get the full-fidelity retry.
  4. Repair headings the model declared but omitted from text_md: promote an
     exact-matching body/bold line in place (free), else retry once with a
     hint listing the missing headings; keep whichever result scores better.
  5. A page with no embedded-text oracle that still trips the review gate is
     read again by the hard-page reader (hard_reader.py: Gemini by default, Opus by choice, on
     native-resolution bands). Its words replace the first pass; paragraph
     breaks and headings are carried over from the first pass, whose text is
     kept as NNNN.first.md. The page stays marked for review. After the
     page loop a book-level rule (hard_book_decision) re-reads the remaining
     no-oracle pages when at least a fifth of them tripped the gate: hard pages
     come in whole documents and the first reader's confidence is unstable.
  6. Only pages that still fail are marked needs_review for the human pass.

Pages are independent (cross-page paragraph joins are resolved at build time
from the starts_mid_paragraph/continues_next fields), so transcription runs
fully parallel.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from dataclasses import dataclass, field

import click

from . import hard_reader, llm, qc, validators
from .config import (HARD_BOOK_MIN_PAGES, HARD_BOOK_SHARE, HARD_READER_DEFAULT, HARD_READERS, LONG_EDGE_BY_RES, LONG_EDGE_STD,
                     MODEL_FAST, MODEL_FIRST_DEFAULT, MODEL_STRONG, RES_HI, RES_STD, hard_page_cost)
from .render import extract_embedded_text, extract_heading_candidates, render_page_to
from .workspace import Workspace

_BOLD_WHOLE_LINE = re.compile(r"^\*\*(.+)\*\*$")


@dataclass
class RunState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    total_cost: float = 0.0
    ok: int = 0
    escalated: int = 0
    hard_read: int = 0
    hard_reader: str = HARD_READER_DEFAULT      # "gemini" or "opus"
    flagged: int = 0
    failed: list = field(default_factory=list)
    budget_exceeded: bool = False

    def add_cost(self, cost: float, max_cost: float | None) -> None:
        with self.lock:
            self.total_cost += cost
            if max_cost is not None and self.total_cost >= max_cost:
                self.budget_exceeded = True


def _ensure_metadata(ws: Workspace, client, state: RunState) -> None:
    """Propose title/author from the opening pages if book.yaml has none yet."""
    meta = ws.meta
    if meta.get("title_fa") or meta.get("metadata_checked"):
        return
    if client is None:        # Gemini-only run without an Anthropic key: the proposal uses Haiku
        click.echo("No Anthropic key: skipping the title/author proposal (edit book.yaml to set them).")
        return
    click.echo("Proposing book metadata from the opening pages ...")
    pngs = []
    for n in (1, 2):
        if n <= meta.get("page_count", 1):
            path = ws.page_image_path(n)
            if not path.is_file():
                render_page_to(ws.pdf_path, n, path, long_edge=LONG_EDGE_STD)
            pngs.append(path.read_bytes())
    try:
        proposed, cost = llm.propose_metadata(client, pngs, model=MODEL_FAST)
        state.add_cost(cost, None)
    except Exception as exc:  # metadata is a nicety; never block transcription on it
        click.echo(f"  (metadata proposal failed: {exc})")
        return
    if proposed is None:
        return
    meta.update(
        {
            "title_fa": proposed.title_fa,
            "author_fa": proposed.author_fa,
            "title_en": proposed.title_en or meta.get("title_en"),
            "publisher_fa": proposed.publisher_fa,
            "metadata_checked": True,
        }
    )
    ws.save_meta(meta)
    if proposed.title_fa:
        click.echo(f"  title: {proposed.title_fa}   author: {proposed.author_fa}")
    else:
        click.echo("  no title page found; leaving metadata blank (edit book.yaml to set it).")


def _auto_promote_headings(text_md: str, declared_headings: list[str]) -> tuple[str, list[str]]:
    """Promote body lines that exactly match a declared heading into heading
    lines, at no API cost. A `**heading**` line becomes `### heading`; a plain
    line becomes `## heading`. Returns (new_text_md, promoted_heading_strings).
    """
    if not declared_headings:
        return text_md, []
    present = set(validators.heading_lines(text_md))
    lines = text_md.split("\n")
    promoted: list[str] = []
    for heading in declared_headings:
        heading = heading.strip()
        if not heading or heading in present:
            continue
        for i, line in enumerate(lines):
            stripped = line.strip()
            bold_match = _BOLD_WHOLE_LINE.match(stripped)
            if bold_match and bold_match.group(1).strip() == heading:
                lines[i] = f"### {heading}"
                promoted.append(heading)
                break
            if stripped == heading:
                lines[i] = f"## {heading}"
                promoted.append(heading)
                break
    return "\n".join(lines), promoted


def _page_image(ws: Workspace, n: int, resolution: str) -> bytes:
    """The page render at the requested resolution, rendering it if missing."""
    path = ws.page_hires_path(n) if resolution == RES_HI else ws.page_image_path(n)
    if not path.is_file():
        render_page_to(ws.pdf_path, n, path, long_edge=LONG_EDGE_BY_RES[resolution])
    return path.read_bytes()


def _hard_read(ws: Workspace, client, n: int, page_count: int, result, state: RunState) -> tuple[dict, float, dict]:
    """Re-read a hard page with the band reader and put its words into `result`.

    Returns (sidecar record, cost, usage). On failure the first pass stands
    and the record says why; a credit or access error is not hidden.
    """
    first_md = result.text_md
    try:
        lines, info, usage, cost = hard_reader.read_page(
            client, ws.pdf_path, n, page_count, ws.meta.get("title_fa"), _page_image(ws, n, RES_HI),
            reader=state.hard_reader)
    except Exception as exc:
        billed = float(getattr(exc, "cost", 0.0) or 0.0)      # attempts that answered are billed even when unusable
        state.add_cost(billed, None)
        return ({"adopted": False, "reader": state.hard_reader, "reason": f"{type(exc).__name__}: {exc}"[:300]},
                billed, dict(getattr(exc, "usage", {}) or {}))
    state.add_cost(cost, None)
    plain, guessed, unread = hard_reader.plain_lines(lines)
    text_md, headings, report = hard_reader.lines_to_markdown(plain, first_md)
    record = {"model": HARD_READERS[state.hard_reader], "reader": state.hard_reader, "reader_version": hard_reader.READER_VERSION, **info, **report,
              "printed_lines": len(plain), "guessed_words": guessed, "unreadable_words": unread,
              "first_pass": {"confidence": result.confidence, "flags": list(result.flags),
                             "text": f"{n:04d}.first.md"}}
    if not text_md.strip():
        return {**record, "adopted": False, "reason": "reader returned no lines"}, cost, usage
    ws.page_md_path(n).with_suffix(".first.md").write_text(first_md, encoding="utf-8")
    # the reader's printed lines as returned, markers included (for review and re-derivation at no cost)
    ws.page_md_path(n).with_suffix(".reader.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result.text_md = text_md
    result.headings = headings
    if unread and "illegible" not in result.flags:
        result.flags = list(result.flags) + ["illegible"]
    return {**record, "adopted": True}, cost, usage


def _transcribe_one(
    ws: Workspace, client, n: int, page_count: int, model: str, resolution: str, state: RunState,
    hard_pages: bool | str = True,
) -> str:
    """Process one page end to end. Returns a one-line status string."""
    title_hint = ws.meta.get("title_fa")

    png = _page_image(ws, n, resolution)

    fallback = None
    try:
        result, usage, cost = llm.transcribe_page(client, png, model, n, page_count, title_hint)
    except hard_reader.ReaderUnavailable:
        raise
    except RuntimeError as exc:
        # Google refuses some pages outright (seen: "blocked due to copyright/recitation content").
        # With an Anthropic key the page is read by the strong Anthropic model instead of being lost.
        if not model.startswith("gemini-") or client is None:
            raise
        state.add_cost(float(getattr(exc, "cost", 0.0) or 0.0), None)
        fallback = {"from": model, "reason": str(exc)[:200]}
        model = MODEL_STRONG
        result, usage, cost = llm.transcribe_page(client, png, model, n, page_count, title_hint)
    state.add_cost(cost, None)

    embedded = extract_embedded_text(ws.pdf_path, n)
    heading_candidates = extract_heading_candidates(ws.pdf_path, n) if embedded else []
    checks = validators.evaluate(
        result.text_md, result.confidence, result.flags, embedded,
        headings=result.headings, heading_candidates=heading_candidates,
    )

    escalated = False
    model_used = model
    res_used = resolution
    used_png = png
    strong = model if model.startswith("gemini-") else MODEL_STRONG      # escalation stays with the same vendor
    at_max_fidelity = model == strong and resolution == RES_HI
    if not at_max_fidelity and validators.needs_escalation(
        result.confidence, result.flags, checks, result.is_blank
    ):
        escalated = True
        hi_bytes = _page_image(ws, n, RES_HI)
        result2, usage2, cost2 = llm.transcribe_page(
            client, hi_bytes, strong, n, page_count, title_hint
        )
        state.add_cost(cost2, None)
        checks2 = validators.evaluate(
            result2.text_md, result2.confidence, result2.flags, embedded,
            headings=result2.headings, heading_candidates=heading_candidates,
        )
        cost += cost2
        usage = {k: usage[k] + usage2[k] for k in usage}
        if checks2["quality_score"] >= checks["quality_score"]:
            result, checks, model_used, used_png = result2, checks2, strong, hi_bytes
            res_used = RES_HI

    # Safe auto-promotion (no API cost) -----------------------------------
    auto_promoted: list[str] = []
    if not result.is_blank and result.headings:
        promoted_md, promoted = _auto_promote_headings(result.text_md, result.headings)
        if promoted:
            result.text_md = promoted_md
            auto_promoted = promoted
            checks = validators.evaluate(
                result.text_md, result.confidence, result.flags, embedded,
                headings=result.headings, heading_candidates=heading_candidates,
            )

    # Retry-with-hint: one more attempt if heading issues remain -----------
    heading_retry = False
    if not result.is_blank and (
        "missing_heading" in checks["issues"] or "embedded_heading_missing" in checks["issues"]
    ):
        hint_headings = list(dict.fromkeys(
            validators.missing_headings(result.headings, result.text_md)
            + validators.missing_embedded_headings(heading_candidates, result.text_md)
        ))
        if hint_headings:
            heading_retry = True
            extra_hint = (
                "IMPORTANT: your previous transcription omitted these headings from text_md. "
                "Each MUST appear as a #, ##, or ### line at its correct position: "
                + ", ".join(hint_headings)
            )
            result3, usage3, cost3 = llm.transcribe_page(
                client, used_png, model_used, n, page_count, title_hint, extra_hint=extra_hint
            )
            state.add_cost(cost3, None)
            promoted_md3, promoted3 = _auto_promote_headings(result3.text_md, result3.headings)
            result3.text_md = promoted_md3
            checks3 = validators.evaluate(
                result3.text_md, result3.confidence, result3.flags, embedded,
                headings=result3.headings, heading_candidates=heading_candidates,
            )
            cost += cost3
            usage = {k: usage[k] + usage3[k] for k in usage}
            if checks3["quality_score"] >= checks["quality_score"]:
                result, checks, auto_promoted = result3, checks3, promoted3

    review = validators.needs_review(result.confidence, result.flags, checks, result.is_blank)
    first_review = review       # the gate's verdict on the first pass (book-level rule)

    # Hard page with no oracle: second reading on native bands ---------------
    hard = None
    stale_first = ws.page_md_path(n).with_suffix(".first.md")
    for stale in (stale_first, stale_first.with_name(stale_first.name.replace(".first.md", ".reader.txt"))):
        if stale.is_file():
            stale.unlink()
    # hard_pages: True = pages the review gate flags; "all" = every page without an oracle
    if hard_pages and (review or hard_pages == "all") and not result.is_blank \
            and checks.get("embedded_similarity") is None:
        if hard_reader.structure_blocks(result.text_md) is None:
            hard = {"adopted": False, "reason": "first pass has verse or a table; not re-read"}
        else:
            hard, cost_h, usage_h = _hard_read(ws, client, n, page_count, result, state)
            hard["trigger"] = "page" if first_review else "all"
            cost += cost_h
            usage = {k: usage[k] + usage_h.get(k, 0) for k in usage}
            if hard["adopted"]:
                model_used = HARD_READERS[state.hard_reader]
                checks = validators.evaluate(
                    result.text_md, result.confidence, result.flags, embedded,
                    headings=result.headings, heading_candidates=heading_candidates,
                )
                review = True      # a hard page stays in front of the reviewer

    ws.page_md_path(n).write_text(result.text_md, encoding="utf-8")
    sidecar = {
        "page": n,
        "model_used": model_used,
        "resolution": res_used,
        "escalated": escalated,
        "heading_retry": heading_retry,
        "hard_reader": hard,
        "first_reader_fallback": fallback,
        "is_blank": result.is_blank,
        "confidence": result.confidence,
        "quality_score": checks["quality_score"],
        "needs_review": review,
        "first_pass_review": first_review,
        "flags": result.flags,
        "headings": result.headings,
        "auto_promoted": auto_promoted,
        "starts_mid_paragraph": result.starts_mid_paragraph,
        "continues_next": result.continues_next,
        "validators": {k: v for k, v in checks.items() if k != "quality_score"},
        "usage": usage,
        "cost_usd": round(cost, 6),
    }
    ws.page_meta_path(n).write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for issue in checks["issues"]:
        try:
            qc.record_event(
                book=ws.slug,
                page=n,
                detected_by="validator",
                issue_type=issue,
                source_type=ws.meta.get("source_type"),
                model_used=model_used,
                flags=result.flags,
                confidence=result.confidence,
                char_signals=["heading"] if "heading" in issue else None,
            )
        except Exception:
            pass

    with state.lock:
        state.ok += 1
        if escalated:
            state.escalated += 1
        if hard and hard["adopted"]:
            state.hard_read += 1
        if review:
            state.flagged += 1

    marks = []
    if fallback:
        marks.append(f"{fallback['from']} refused, read by {MODEL_STRONG}")
    if escalated:
        marks.append("escalated")
    if hard:
        marks.append("hard-page reader" if hard["adopted"] else "hard-page reader not used")
    if review:
        marks.append("REVIEW")
    suffix = f"  [{', '.join(marks)}]" if marks else ""
    return f"page {n:>4}  conf={result.confidence:.2f}  q={checks['quality_score']:.2f}  ${cost:.4f}{suffix}"


def _no_oracle(side: dict) -> bool:
    return not side.get("is_blank") and (side.get("validators") or {}).get("embedded_similarity") is None


def _first_gate(side: dict) -> bool:
    """Did the review gate fire on this page's first pass? Sidecars written
    before the field existed are recomputed from the stored values."""
    if "first_pass_review" in side:
        return bool(side["first_pass_review"])
    first = (side.get("hard_reader") or {}).get("first_pass") or {}
    checks = dict(side.get("validators") or {})
    checks.setdefault("issues", [])
    return validators.needs_review(side.get("confidence", 1.0), first.get("flags", side.get("flags") or []),
                                   checks, bool(side.get("is_blank")))


def hard_book_decision(sidecars: dict[int, dict]) -> dict:
    """Book-level rule over page sidecars: when at least HARD_BOOK_SHARE of the
    non-blank pages without embedded text tripped the review gate on their first
    pass, the book is hard and the rest of those pages should be re-read too.
    `candidates` are the pages not yet handed to the hard-page reader."""
    pages = {n: s for n, s in sidecars.items() if _no_oracle(s)}
    fired = sum(1 for s in pages.values() if _first_gate(s))
    hard = len(pages) >= HARD_BOOK_MIN_PAGES and fired >= HARD_BOOK_SHARE * len(pages)
    return {"hard": hard, "fired": fired, "counted": len(pages),
            "candidates": sorted(n for n, s in pages.items() if not s.get("hard_reader")) if hard else []}


def _book_reread_one(ws: Workspace, client, n: int, page_count: int, state: RunState) -> str:
    """Hard-page reader on a page whose first pass is already stored (book rule)."""
    side = json.loads(ws.page_meta_path(n).read_text(encoding="utf-8"))
    result = SimpleNamespace(text_md=ws.page_md_path(n).read_text(encoding="utf-8"),
                             confidence=side["confidence"], flags=list(side.get("flags") or []),
                             headings=list(side.get("headings") or []))
    if hard_reader.structure_blocks(result.text_md) is None:
        hard, cost, usage = {"adopted": False, "reason": "first pass has verse or a table; not re-read"}, 0.0, {}
    else:
        hard, cost, usage = _hard_read(ws, client, n, page_count, result, state)
    hard["trigger"] = "book"
    side["hard_reader"] = hard
    side.setdefault("first_pass_review", False)
    if hard["adopted"]:
        checks = validators.evaluate(result.text_md, result.confidence, result.flags, None,
                                     headings=result.headings, heading_candidates=[])
        ws.page_md_path(n).write_text(result.text_md, encoding="utf-8")
        was_flagged = bool(side.get("needs_review"))
        side.update({"model_used": HARD_READERS[state.hard_reader], "needs_review": True, "flags": result.flags,
                     "headings": result.headings, "quality_score": checks["quality_score"],
                     "validators": {k: v for k, v in checks.items() if k != "quality_score"}})
        side["usage"] = {k: v + usage.get(k, 0) for k, v in (side.get("usage") or {}).items()}
        side["cost_usd"] = round(side.get("cost_usd", 0.0) + cost, 6)
        with state.lock:
            state.hard_read += 1
            if not was_flagged:
                state.flagged += 1
    ws.page_meta_path(n).write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
    mark = "hard-page reader, book rule" if hard["adopted"] else f"hard-page reader not used: {hard.get('reason', '')}"
    return f"page {n:>4}  ${cost:.4f}  [{mark}]"


def _book_rule(ws: Workspace, client, run_pages: list[int], page_count: int, state: RunState,
               max_cost: float | None, concurrency: int) -> None:
    """After the page loop: apply the book-level rule to this run's pages."""
    sides = {}
    for n in ws.pages_done():
        try:
            sides[n] = json.loads(ws.page_meta_path(n).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    decision = hard_book_decision(sides)
    if not decision["candidates"]:
        return
    todo = [n for n in decision["candidates"] if n in set(run_pages) and not sides[n].get("reviewed")]
    other = [n for n in decision["candidates"] if n not in todo]
    click.echo("")
    click.echo(f"Hard book: {decision['fired']} of {decision['counted']} pages without embedded text tripped the review gate.")
    if other:
        click.echo(f"  Not re-read (outside this run, or already reviewed): {','.join(map(str, other))}. "
                   f"To re-read: transcribe --pages <list> --force --hard-pages all")
    if not todo:
        return
    each = hard_page_cost(state.hard_reader)
    click.echo(f"  Re-reading {len(todo)} more page(s) with {HARD_READERS[state.hard_reader]}, about ${each:.2f} each ...")
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {}
        for n in todo:
            if max_cost is not None and state.total_cost + each * len(futures) >= max_cost:
                state.budget_exceeded = True
                break
            futures[pool.submit(_book_reread_one, ws, client, n, page_count, state)] = n
        for fut in as_completed(futures):
            try:
                click.echo(f"  {fut.result()}")
            except Exception as exc:
                click.echo(f"  page {futures[fut]:>4}  hard-page reader FAILED: {exc}")


def transcribe_pages(
    ws: Workspace,
    pages: list[int],
    model: str | None = None,
    max_cost: float | None = None,
    concurrency: int = 4,
    resolution: str = RES_HI,
    hard_pages: bool | str = True,
    hard_reader_name: str = HARD_READER_DEFAULT,
) -> None:
    # Measured escalation rates under the strict validators: 22/23 scanned
    # pages and 16/17 digital pages failed the Haiku pass, so the two-pass
    # model ladder costs more than starting on the strong model (break-even
    # ~72%). Haiku remains available via an explicit --model claude-haiku-4-5.
    # Hi-res input is likewise the measured default (see module docstring);
    # --res std is the economical path, with hi-res escalation on failure.
    if model is None:
        model = MODEL_FIRST_DEFAULT
        if model.startswith("gemini-") and not hard_reader.gemini_key():
            click.echo(f"No GEMINI_API_KEY in the environment or .env: transcribing with {MODEL_STRONG} instead.")
            model = MODEL_STRONG
    if resolution not in (RES_HI, RES_STD):
        raise ValueError(f"unknown resolution {resolution!r} (expected {RES_HI!r} or {RES_STD!r})")
    try:
        client = llm.get_client()
    except RuntimeError:
        if not model.startswith("gemini-"):
            raise
        client = None         # every call of this run goes to Google
    page_count = ws.meta.get("page_count", max(pages))
    state = RunState()
    if hard_reader_name not in HARD_READERS:
        raise ValueError(f"unknown hard-page reader {hard_reader_name!r} (expected one of {sorted(HARD_READERS)})")
    if hard_pages and hard_reader_name == "gemini" and not hard_reader.gemini_key():
        click.echo("No GEMINI_API_KEY in the environment or .env: hard pages will be read by "
                   f"{HARD_READERS['opus']} instead (--hard-reader opus).")
        hard_reader_name = "opus"
    if hard_pages and hard_reader_name == "opus" and client is None:
        raise RuntimeError("--hard-reader opus needs ANTHROPIC_API_KEY")
    state.hard_reader = hard_reader_name

    _ensure_metadata(ws, client, state)

    click.echo(
        f"Transcribing {len(pages)} page(s) with {model} at {resolution}-res "
        f"(escalation: {model if model.startswith('gemini-') else MODEL_STRONG} + hi-res"
        + (f"; {'every page' if hard_pages == 'all' else 'flagged pages'} without embedded text: {HARD_READERS[state.hard_reader]} on bands, about ${hard_page_cost(state.hard_reader):.2f} each" if hard_pages else "")
        + ") ..."
    )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {}
        for n in pages:
            if state.budget_exceeded:
                break
            futures[pool.submit(_transcribe_one, ws, client, n, page_count, model, resolution, state, hard_pages)] = n
        for fut in as_completed(futures):
            n = futures[fut]
            if max_cost is not None and state.total_cost >= max_cost:
                state.budget_exceeded = True
            try:
                click.echo(f"  {fut.result()}")
            except Exception as exc:
                with state.lock:
                    state.failed.append(n)
                click.echo(f"  page {n:>4}  FAILED: {exc}")

    if hard_pages is True and not state.budget_exceeded:
        _book_rule(ws, client, [n for n in pages if n not in state.failed], page_count, state, max_cost, concurrency)

    click.echo("")
    click.echo(f"Done: {state.ok} page(s) transcribed, {state.escalated} escalated, {state.hard_read} re-read by the hard-page reader, {state.flagged} flagged for review.")
    if state.failed:
        click.echo(f"Failed pages (rerun with --pages): {','.join(str(n) for n in sorted(state.failed))}")
    if state.budget_exceeded:
        click.echo(f"Stopped early: --max-cost {max_cost} reached.")
    click.echo(f"Transcription cost this run: ${state.total_cost:.4f}")
