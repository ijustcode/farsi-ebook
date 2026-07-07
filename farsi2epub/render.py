"""PDF inspection and rasterization helpers, built on PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .config import LONG_EDGE_STD

DIGITAL_THRESHOLD = 0.80
SCANNED_THRESHOLD = 0.20
CHARS_PER_PAGE_MIN = 200


def classify_pdf(pdf_path: str | Path) -> dict:
    """Inspect a PDF and classify it as "digital", "scanned", or "mixed".

    Returns {page_count, per_page_chars, kind}.
    - "digital" if >= 80% of pages have more than 200 extracted characters
    - "scanned" if <= 20% of pages have more than 200 extracted characters
    - "mixed" otherwise
    """
    doc = fitz.open(str(pdf_path))
    try:
        page_count = doc.page_count
        per_page_chars = []
        for i in range(page_count):
            text = doc[i].get_text()
            per_page_chars.append(len(text))
    finally:
        doc.close()

    if page_count == 0:
        return {"page_count": 0, "per_page_chars": [], "kind": "scanned"}

    frac_text_pages = sum(1 for c in per_page_chars if c > CHARS_PER_PAGE_MIN) / page_count

    if frac_text_pages >= DIGITAL_THRESHOLD:
        kind = "digital"
    elif frac_text_pages <= SCANNED_THRESHOLD:
        kind = "scanned"
    else:
        kind = "mixed"

    return {"page_count": page_count, "per_page_chars": per_page_chars, "kind": kind}


def render_page(pdf_path: str | Path, page_number_1based: int, long_edge: int = LONG_EDGE_STD) -> bytes:
    """Render a single PDF page to PNG bytes, scaled so its longest edge is
    approximately `long_edge` pixels."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number_1based - 1]
        rect = page.rect
        max_dim = max(rect.width, rect.height)
        zoom = long_edge / max_dim if max_dim else 1.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_page_to(
    pdf_path: str | Path,
    page_number_1based: int,
    out_path: str | Path,
    long_edge: int = LONG_EDGE_STD,
) -> Path:
    """Render a page and write the PNG bytes to `out_path`."""
    data = render_page(pdf_path, page_number_1based, long_edge=long_edge)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def extract_embedded_text(pdf_path: str | Path, page_number_1based: int) -> str:
    """Extract the embedded text layer (if any) for a single page."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number_1based - 1]
        return page.get_text()
    finally:
        doc.close()
