"""Vision transcription orchestrator: fast pass, validation, escalation, sidecars.

Per page:
  1. Transcribe the standard-res render with the fast model (Haiku).
  2. Score it with deterministic validators (embedded-text cross-check for
     digital PDFs, script sanity for all pages) plus model confidence.
  3. On failure, re-render at high resolution and retry on the strong model
     (Sonnet supports high-res vision); keep whichever result scores better.
  4. Only pages that still fail are marked needs_review for the human pass.

Pages are independent (cross-page paragraph joins are resolved at build time
from the starts_mid_paragraph/continues_next fields), so transcription runs
fully parallel.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import click

from . import llm, validators
from .config import LONG_EDGE_HI, LONG_EDGE_STD, MODEL_FAST, MODEL_STRONG
from .render import extract_embedded_text, render_page, render_page_to
from .workspace import Workspace


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


def _transcribe_one(ws: Workspace, client, n: int, page_count: int, model: str, state: RunState) -> str:
    """Process one page end to end. Returns a one-line status string."""
    title_hint = ws.meta.get("title_fa")

    std_path = ws.page_image_path(n)
    if not std_path.is_file():
        render_page_to(ws.pdf_path, n, std_path, long_edge=LONG_EDGE_STD)
    png = std_path.read_bytes()

    result, usage, cost = llm.transcribe_page(client, png, model, n, page_count, title_hint)
    state.add_cost(cost, None)

    embedded = extract_embedded_text(ws.pdf_path, n)
    checks = validators.evaluate(result.text_md, result.confidence, result.flags, embedded)

    escalated = False
    model_used = model
    if model != MODEL_STRONG and validators.needs_escalation(
        result.confidence, result.flags, checks, result.is_blank
    ):
        escalated = True
        hi_path = ws.page_hires_path(n)
        if not hi_path.is_file():
            hi_path.write_bytes(render_page(ws.pdf_path, n, long_edge=LONG_EDGE_HI))
        result2, usage2, cost2 = llm.transcribe_page(
            client, hi_path.read_bytes(), MODEL_STRONG, n, page_count, title_hint
        )
        state.add_cost(cost2, None)
        checks2 = validators.evaluate(result2.text_md, result2.confidence, result2.flags, embedded)
        cost += cost2
        usage = {k: usage[k] + usage2[k] for k in usage}
        if checks2["quality_score"] >= checks["quality_score"]:
            result, checks, model_used = result2, checks2, MODEL_STRONG

    review = validators.needs_review(result.confidence, result.flags, checks, result.is_blank)

    ws.page_md_path(n).write_text(result.text_md, encoding="utf-8")
    sidecar = {
        "page": n,
        "model_used": model_used,
        "escalated": escalated,
        "is_blank": result.is_blank,
        "confidence": result.confidence,
        "quality_score": checks["quality_score"],
        "needs_review": review,
        "flags": result.flags,
        "headings": result.headings,
        "starts_mid_paragraph": result.starts_mid_paragraph,
        "continues_next": result.continues_next,
        "validators": {k: v for k, v in checks.items() if k != "quality_score"},
        "usage": usage,
        "cost_usd": round(cost, 6),
    }
    ws.page_meta_path(n).write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
) -> None:
    # Measured escalation rates under the strict validators: 22/23 scanned
    # pages and 16/17 digital pages failed the Haiku pass, so the two-pass
    # ladder costs more than starting on the strong model (break-even ~72%).
    # Haiku remains available via an explicit --model claude-haiku-4-5.
    model = model or MODEL_STRONG
    client = llm.get_client()
    page_count = ws.meta.get("page_count", max(pages))
    state = RunState()

    _ensure_metadata(ws, client, state)

    click.echo(f"Transcribing {len(pages)} page(s) with {model} (escalation: {MODEL_STRONG}) ...")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {}
        for n in pages:
            if state.budget_exceeded:
                break
            futures[pool.submit(_transcribe_one, ws, client, n, page_count, model, state)] = n
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
