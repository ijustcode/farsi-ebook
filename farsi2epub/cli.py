"""Command-line interface for farsi2epub."""

from __future__ import annotations

import re
import shutil
import sys
from datetime import date
from pathlib import Path

import click

from . import epub, review, transcribe
from .config import LONG_EDGE_STD, MODEL_FAST, estimate_cost
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
    cost = estimate_cost(est_pages)

    click.echo("")
    click.echo(f"Workspace created: {ws.root}")
    click.echo(f"Classification:    {source_type}")
    click.echo(f"Page count:        {page_count}")
    if page_range:
        click.echo(f"Page range:        {page_range} ({est_pages} pages)")
    click.echo(f"Estimated cost:    ${cost:.2f}")
    click.echo("")
    click.echo(f"Next: farsi2epub transcribe {slug}")


@main.command()
@click.argument("slug")
@click.option("--pages", "pages_spec", default=None, help="Page range to transcribe (default: full book.yaml page_range or all pages).")
@click.option("--force", is_flag=True, help="Re-transcribe pages that already have output.")
@click.option("--max-cost", "max_cost", type=float, default=None, help="Abort if the estimated cost exceeds this many dollars.")
@click.option("--concurrency", type=int, default=4, help="Number of pages to transcribe concurrently.")
@click.option("--model", "model", default=None, help="Model to use for transcription (default: Sonnet; escalation ladder applies only when starting on Haiku).")
def transcribe_cmd(slug: str, pages_spec: str | None, force: bool, max_cost: float | None, concurrency: int, model: str):
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

    # Ensure page images exist, rendering any that are missing.
    for n in pages:
        img_path = ws.page_image_path(n)
        if not img_path.is_file():
            click.echo(f"Rendering page {n} ...")
            render_page_to(ws.pdf_path, n, img_path, long_edge=LONG_EDGE_STD)

    try:
        transcribe.transcribe_pages(ws, pages, model=model, max_cost=max_cost, concurrency=concurrency)
    except NotImplementedError:
        click.echo("")
        click.echo("Transcription module not yet implemented (coming in a later task).")
        return


# `transcribe` is a reserved-looking name; expose it as the `transcribe` command.
main.add_command(transcribe_cmd, name="transcribe")


@main.command()
@click.argument("slug")
def review_cmd(slug: str):
    """Launch the review workflow for workspace SLUG."""
    ws = Workspace.load(slug)
    try:
        review.run_review(ws)
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
