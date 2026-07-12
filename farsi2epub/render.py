"""PDF inspection and rasterization helpers, built on PyMuPDF (fitz)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from .config import LONG_EDGE_STD

DIGITAL_THRESHOLD = 0.80
SCANNED_THRESHOLD = 0.20
CHARS_PER_PAGE_MIN = 200

BOLD_FLAG = 16  # fitz span flags bit for bold
HEADING_SIZE_RATIO = 1.15  # span size over this multiple of body size reads as a heading
HEADING_LEN_BOUNDS = (2, 60)
_DIGIT_PUNCT_ONLY = re.compile(r"^[\d۰-۹.,:;!?()\[\]{}«»،؛؟\-–—_*#>`^\"'|/\\\s]+$")


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


# --- text-layer gate --------------------------------------------------------
# NB: the QC-box snippet locator that used to live here has moved to
# `farsi2epub.locate` (tiered word-match / line-layout tiers). Only the
# document-level text-layer gate remains, still used elsewhere.

_ARABIC_BLOCK_MIN_FRAC = 0.30
_USABLE_CHARS_MIN = 200


def _is_arabic_block(ch: str) -> bool:
    return "؀" <= ch <= "ۿ"


def text_layer_usable(pdf_path: str | Path, sample_pages: int = 5) -> bool:
    """True when the PDF's embedded text layer looks like real Unicode Persian.

    Samples up to `sample_pages` pages whose extracted text exceeds 200 chars;
    returns True if at least one has >= 30% Arabic-block characters among its
    non-space characters. Glyph-soup layers (non-Unicode CID gibberish) and
    scanned books (no layer at all) return False.
    """
    doc = fitz.open(str(pdf_path))
    try:
        sampled = 0
        for page in doc:
            text = page.get_text()
            if len(text) <= _USABLE_CHARS_MIN:
                continue
            sampled += 1
            non_space = [c for c in text if not c.isspace()]
            if non_space:
                arabic = sum(1 for c in non_space if _is_arabic_block(c))
                if arabic / len(non_space) >= _ARABIC_BLOCK_MIN_FRAC:
                    return True
            if sampled >= sample_pages:
                break
        return False
    finally:
        doc.close()


def extract_heading_candidates(pdf_path: str | Path, page_number_1based: int) -> list[str]:
    """Detect probable heading lines from the PDF text layer's font weight/size.

    Body size is the span size (rounded to 1 decimal) with the most characters
    on the page. A line is a heading candidate when every non-empty span in it
    is bold, or its largest span exceeds 1.15x the body size, and its joined
    text is 2-60 chars and not purely digits/punctuation (page numbers).
    Returns lines in reading order; [] for pages with no text layer.
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number_1based - 1]
        data = page.get_text("dict")
    finally:
        doc.close()

    size_chars: Counter = Counter()
    lines_info = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            lines_info.append((spans, text))
            for s in spans:
                size_chars[round(s["size"], 1)] += len(s["text"])

    if not size_chars:
        return []
    body_size = size_chars.most_common(1)[0][0]

    candidates = []
    for spans, text in lines_info:
        if not (HEADING_LEN_BOUNDS[0] <= len(text) <= HEADING_LEN_BOUNDS[1]):
            continue
        if _DIGIT_PUNCT_ONLY.match(text):
            continue
        all_bold = all(s["flags"] & BOLD_FLAG for s in spans)
        max_size = max(s["size"] for s in spans)
        if all_bold or max_size > HEADING_SIZE_RATIO * body_size:
            candidates.append(text)
    return candidates
