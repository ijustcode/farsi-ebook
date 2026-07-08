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
  5. Only pages that still fail are marked needs_review for the human pass.

Pages are independent (cross-page paragraph joins are resolved at build time
from the starts_mid_paragraph/continues_next fields), so transcription runs
fully parallel.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import click

from . import llm, qc, validators
from .config import LONG_EDGE_BY_RES, LONG_EDGE_STD, MODEL_FAST, MODEL_STRONG, RES_HI, RES_STD
from .render import extract_embedded_text, extract_heading_candidates, render_page_to
from .workspace import Workspace

_BOLD_WHOLE_LINE = re.compile(r"^\*\*(.+)\*\*$")


@dataclass
class RunState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    total_cost: float = 0.0
    ok: int = 0
    escalated: int = 0
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


def _transcribe_one(
    ws: Workspace, client, n: int, page_count: int, model: str, resolution: str, state: RunState
) -> str:
    """Process one page end to end. Returns a one-line status string."""
    title_hint = ws.meta.get("title_fa")

    png = _page_image(ws, n, resolution)

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
    at_max_fidelity = model == MODEL_STRONG and resolution == RES_HI
    if not at_max_fidelity and validators.needs_escalation(
        result.confidence, result.flags, checks, result.is_blank
    ):
        escalated = True
        hi_bytes = _page_image(ws, n, RES_HI)
        result2, usage2, cost2 = llm.transcribe_page(
            client, hi_bytes, MODEL_STRONG, n, page_count, title_hint
        )
        state.add_cost(cost2, None)
        checks2 = validators.evaluate(
            result2.text_md, result2.confidence, result2.flags, embedded,
            headings=result2.headings, heading_candidates=heading_candidates,
        )
        cost += cost2
        usage = {k: usage[k] + usage2[k] for k in usage}
        if checks2["quality_score"] >= checks["quality_score"]:
            result, checks, model_used, used_png = result2, checks2, MODEL_STRONG, hi_bytes
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

    ws.page_md_path(n).write_text(result.text_md, encoding="utf-8")
    sidecar = {
        "page": n,
        "model_used": model_used,
        "resolution": res_used,
        "escalated": escalated,
        "heading_retry": heading_retry,
        "is_blank": result.is_blank,
        "confidence": result.confidence,
        "quality_score": checks["quality_score"],
        "needs_review": review,
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
        if review:
            state.flagged += 1

    marks = []
    if escalated:
        marks.append("escalated")
    if review:
        marks.append("REVIEW")
    suffix = f"  [{', '.join(marks)}]" if marks else ""
    return f"page {n:>4}  conf={result.confidence:.2f}  q={checks['quality_score']:.2f}  ${cost:.4f}{suffix}"


def transcribe_pages(
    ws: Workspace,
    pages: list[int],
    model: str | None = None,
    max_cost: float | None = None,
    concurrency: int = 4,
    resolution: str = RES_HI,
) -> None:
    # Measured escalation rates under the strict validators: 22/23 scanned
    # pages and 16/17 digital pages failed the Haiku pass, so the two-pass
    # model ladder costs more than starting on the strong model (break-even
    # ~72%). Haiku remains available via an explicit --model claude-haiku-4-5.
    # Hi-res input is likewise the measured default (see module docstring);
    # --res std is the economical path, with hi-res escalation on failure.
    model = model or MODEL_STRONG
    if resolution not in (RES_HI, RES_STD):
        raise ValueError(f"unknown resolution {resolution!r} (expected {RES_HI!r} or {RES_STD!r})")
    client = llm.get_client()
    page_count = ws.meta.get("page_count", max(pages))
    state = RunState()

    _ensure_metadata(ws, client, state)

    click.echo(
        f"Transcribing {len(pages)} page(s) with {model} at {resolution}-res "
        f"(escalation: {MODEL_STRONG} + hi-res) ..."
    )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {}
        for n in pages:
            if state.budget_exceeded:
                break
            futures[pool.submit(_transcribe_one, ws, client, n, page_count, model, resolution, state)] = n
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

    click.echo("")
    click.echo(f"Done: {state.ok} page(s) transcribed, {state.escalated} escalated, {state.flagged} flagged for review.")
    if state.failed:
        click.echo(f"Failed pages (rerun with --pages): {sorted(state.failed)}")
    if state.budget_exceeded:
        click.echo(f"Stopped early: --max-cost {max_cost} reached.")
    click.echo(f"Total cost this run: ${state.total_cost:.4f}")
