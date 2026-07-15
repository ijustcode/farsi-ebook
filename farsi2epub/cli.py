"""Command-line interface for farsi2epub."""

from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import click

from . import epub, qc, review, transcribe
from .config import LONG_EDGE_BY_RES, LONG_EDGE_STD, MODEL_STRONG, RES_HI, RES_STD, estimate_cost
from .render import classify_pdf, render_page_to
from .workspace import Workspace, parse_pages_spec


def _slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", stem)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "book"


@click.group()
def main():
    """farsi2epub: convert Farsi PDF books to EPUB via a vision-LLM pipeline."""


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--slug", "slug_opt", default=None, help="Workspace slug (default: derived from filename).")
@click.option("--pages", "pages_spec", default=None, help='Page range, e.g. "5", "3-10", "3-", "-20", "1,3-5".')
@click.option("--force", is_flag=True, help="Overwrite an existing workspace with the same slug.")
def analyze(pdf_path: Path, slug_opt: str | None, pages_spec: str | None, force: bool):
    """Create a workspace for PDF_PATH, classify it, and estimate cost."""
    slug = slug_opt or _slugify(pdf_path.name)

    existing_root = Workspace(slug).root
    if existing_root.is_dir():
        if not force:
            click.echo(
                f"Error: workspace '{slug}' already exists at {existing_root}. "
                f"Use --force to overwrite, or pick a different --slug.",
                err=True,
            )
            sys.exit(1)
        shutil.rmtree(existing_root)

    ws = Workspace.create(pdf_path, slug)

    click.echo(f"Analyzing {pdf_path} ...")
    info = classify_pdf(ws.pdf_path)
    page_count = info["page_count"]
    source_type = info["kind"]

    # Render page 1 at standard resolution so there's something to look at.
    render_page_to(ws.pdf_path, 1, ws.page_image_path(1), long_edge=LONG_EDGE_STD)

    page_range = pages_spec if pages_spec else None

    meta = {
        "slug": slug,
        "title_fa": None,
        "author_fa": None,
        "title_en": None,
        "language": "fa",
        "source_type": source_type,
        "page_count": page_count,
        "page_range": page_range,
        "chapters": None,
        "created": date.today().isoformat(),
    }
    ws.save_meta(meta)

    if page_range:
        est_pages = len(parse_pages_spec(page_range, page_count))
    else:
        est_pages = page_count
    cost_hi = estimate_cost(est_pages, resolution=RES_HI)
    cost_std = estimate_cost(est_pages, resolution=RES_STD)

    click.echo("")
    click.echo(f"Workspace created: {ws.root}")
    click.echo(f"Classification:    {source_type}")
    click.echo(f"Page count:        {page_count}")
    if page_range:
        click.echo(f"Page range:        {page_range} ({est_pages} pages)")
    click.echo(f"Estimated cost:    ${cost_hi:.2f} (default: {MODEL_STRONG} + hi-res input)")
    click.echo(f"Economy estimate:  ${cost_std:.2f} (transcribe --res std; failing pages escalate to hi-res)")
    click.echo("")
    click.echo(f"Next: farsi2epub transcribe {slug}")


@main.command()
@click.argument("slug")
@click.option("--pages", "pages_spec", default=None, help="Page range to transcribe (default: full book.yaml page_range or all pages).")
@click.option("--force", is_flag=True, help="Re-transcribe pages that already have output.")
@click.option("--max-cost", "max_cost", type=float, default=None, help="Abort if the estimated cost exceeds this many dollars.")
@click.option("--concurrency", type=int, default=4, help="Number of pages to transcribe concurrently.")
@click.option("--model", "model", default=None, help="Model to use for transcription (default: Sonnet; failing pages escalate to Sonnet + hi-res).")
@click.option("--res", "resolution", type=click.Choice([RES_HI, RES_STD]), default=RES_HI, help="Page-image resolution: hi (2576px, default — best character accuracy) or std (1568px, ~30% cheaper; use for crisp large-print sources or very long books; failing pages escalate to hi-res).")
@click.option("--qc", "qc_mode", type=click.Choice(["auto", "manual", "skip", "ask"]), default="ask", help="Run a QC pass after transcription: auto (automated), manual (review UI), skip (none), or ask (prompt if TTY).")
@click.option("--yes", "assume_yes", is_flag=True, help="Skip the auto-QC cost confirmation prompt.")
def transcribe_cmd(slug: str, pages_spec: str | None, force: bool, max_cost: float | None, concurrency: int, model: str, resolution: str, qc_mode: str, assume_yes: bool):
    """Transcribe pages of workspace SLUG using the vision-LLM pipeline."""
    ws = Workspace.load(slug)
    meta = ws.meta
    page_count = meta.get("page_count")
    if page_count is None:
        click.echo("Error: book.yaml is missing page_count; run `analyze` again.", err=True)
        sys.exit(1)

    spec = pages_spec or meta.get("page_range")
    pages = parse_pages_spec(spec, page_count)

    if not force:
        done = set(ws.pages_done())
        pages = [p for p in pages if p not in done]

    if not pages:
        click.echo("Nothing to do: all requested pages are already transcribed (use --force to redo).")
        return

    click.echo(f"Resolved {len(pages)} page(s) to transcribe: {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"Resolved 1 page to transcribe: {pages[0]}")

    # Ensure page images exist at the chosen resolution, rendering any missing.
    for n in pages:
        img_path = ws.page_hires_path(n) if resolution == RES_HI else ws.page_image_path(n)
        if not img_path.is_file():
            click.echo(f"Rendering page {n} ({resolution}-res) ...")
            render_page_to(ws.pdf_path, n, img_path, long_edge=LONG_EDGE_BY_RES[resolution])

    try:
        transcribe.transcribe_pages(
            ws, pages, model=model, max_cost=max_cost, concurrency=concurrency, resolution=resolution
        )
    except NotImplementedError:
        click.echo("")
        click.echo("Transcription module not yet implemented (coming in a later task).")
        return

    # Handle post-transcription QC logic
    resolved_qc_mode = qc_mode
    if qc_mode == "ask":
        if sys.stdin.isatty():
            click.echo("")
            click.echo("QC options:")
            click.echo("  auto   - LLM verifier checks risky pages and attaches suggested corrections")
            click.echo("  manual - open the review web UI to correct flagged pages yourself")
            click.echo("  skip   - do nothing now (you can run `farsi2epub qc` later)")
            resolved_qc_mode = click.prompt(
                "Run QC now?",
                type=click.Choice(["auto", "manual", "skip"]),
                default="auto",
                show_choices=True,
            )
        else:
            resolved_qc_mode = "skip"

    if resolved_qc_mode in ("auto", "manual"):
        try:
            qc.run_qc(ws, resolved_qc_mode, assume_yes=assume_yes, pages=pages)
        except NotImplementedError:
            click.echo("QC module not yet implemented (coming in a later task).")


# `transcribe` is a reserved-looking name; expose it as the `transcribe` command.
main.add_command(transcribe_cmd, name="transcribe")


@main.command()
@click.argument("slug")
@click.option("--mode", "qc_mode", type=click.Choice(["auto", "manual"]), default="auto", help="QC mode: auto (automated verification) or manual (review UI).")
@click.option("--all", "all_pages", is_flag=True, help="Verify/review every transcribed page instead of risk-selected ones.")
@click.option("--yes", "assume_yes", is_flag=True, help="Skip cost confirmation and proceed with QC.")
@click.option("--force", is_flag=True, help="Also re-verify pages whose previous QC suggestion is still pending (replaces the old suggestion).")
@click.option("--pages", "pages_spec", default=None, help='Restrict QC to this page range, e.g. "5", "3-10", "1,3-5" (default: risk-select across all transcribed pages).')
def qc_cmd(slug: str, qc_mode: str, all_pages: bool, assume_yes: bool, force: bool, pages_spec: str | None):
    """Run quality control checks for workspace SLUG."""
    ws = Workspace.load(slug)
    pages = parse_pages_spec(pages_spec, ws.meta.get("page_count")) if pages_spec else None
    try:
        qc.run_qc(ws, qc_mode, all_pages=all_pages, assume_yes=assume_yes, force=force, pages=pages)
    except NotImplementedError:
        click.echo("QC module not yet implemented (coming in a later task).")


main.add_command(qc_cmd, name="qc")


@main.command()
@click.argument("slug")
@click.option("--all", "all_pages", is_flag=True, help="Surface every flagged page, ignoring the review budget.")
@click.option("--reset", "reset", is_flag=True, help="Undo all human review decisions (keeps text edits and .orig.md backups) and exit.")
@click.option("--background", "-b", "background", is_flag=True, help="Start the review server detached and return immediately.")
@click.option("--status", "status", is_flag=True, help="Report whether a review server is running for this workspace.")
@click.option("--stop", "stop_server", is_flag=True, help="Stop a running review server for this workspace.")
@click.option("--_child", "is_child", is_flag=True, hidden=True, help="Internal: re-entry point for a detached background server.")
def review_cmd(slug: str, all_pages: bool, reset: bool, background: bool, status: bool, stop_server: bool, is_child: bool):
    """Launch the review workflow for workspace SLUG."""
    ws = Workspace.load(slug)
    if reset:
        if all_pages:
            click.echo("Note: --all is ignored with --reset.")
        stats = review.reset_reviews(ws)
        click.echo(
            f"Reset {stats['pages_reset']} page(s), removed {stats['events_removed']} "
            f"review event(s). Run: farsi2epub review {slug}"
        )
        return

    if status:
        state = review.read_server_state(ws)
        if state:
            click.echo(f"Review server running: {state['url']} (pid {state['pid']})")
        else:
            click.echo(f"No review server running for '{slug}'.")
        return

    if stop_server:
        state = review.read_server_state(ws)
        if not state:
            click.echo(f"No review server running for '{slug}'.")
            return
        # POST /quit with a short timeout; reuse stdlib (urllib.request), no new deps.
        import urllib.request

        try:
            urllib.request.urlopen(
                urllib.request.Request(state["url"] + "quit", method="POST"), timeout=3
            )
            click.echo(f"Stopped review server for '{slug}'.")
        except Exception:
            # Fall back to SIGTERM if /quit is unreachable.
            import signal

            try:
                os.kill(state["pid"], signal.SIGTERM)
                click.echo(
                    f"Review server for '{slug}' was unresponsive; sent SIGTERM (pid {state['pid']})."
                )
            except ProcessLookupError:
                click.echo(f"No review server running for '{slug}'.")
        return

    if is_child:
        # This is the detached child process spawned by launch_review_background
        # — just run normally.
        try:
            review.run_review(ws, budget_all=all_pages)
        except NotImplementedError:
            click.echo("Review module not yet implemented (coming in a later task).")
        return

    # Plain `review <slug>` or `--background`: never start a duplicate server.
    existing = review.read_server_state(ws)
    if existing:
        click.echo(f"Review server already running: {existing['url']} (pid {existing['pid']})")
        if not background:
            try:
                import webbrowser

                webbrowser.open(existing["url"])
            except Exception:
                pass
        return

    if background:
        try:
            url = review.launch_review_background(ws, budget_all=all_pages)
        except RuntimeError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        click.echo(f"Review server started in background: {url}")
        click.echo(f"Logs: {ws.review_dir / 'server.log'}")
        click.echo(f"Run 'farsi2epub review {slug} --stop' when done.")
        return

    try:
        review.run_review(ws, budget_all=all_pages)
    except NotImplementedError:
        click.echo("Review module not yet implemented (coming in a later task).")


main.add_command(review_cmd, name="review")


@main.command()
@click.argument("slug")
def build(slug: str):
    """Build the final EPUB for workspace SLUG."""
    ws = Workspace.load(slug)
    try:
        epub.build_epub(ws)
    except NotImplementedError:
        click.echo("EPUB builder not yet implemented (coming in a later task).")


if __name__ == "__main__":
    main()
