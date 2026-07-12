"""PDF inspection and rasterization helpers, built on PyMuPDF (fitz)."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional

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


# --- text-layer gate + snippet locator (QC box overlays) --------------------

_ZWNJ = "‌"
_ARABIC_BLOCK_MIN_FRAC = 0.30
_USABLE_CHARS_MIN = 200

# Markdown noise stripped from snippet queries before searching the text layer.
_MD_LINE_LEAD = re.compile(r"^\s*(?:#{1,6}|>+|\|)\s*")
_MD_LIST_LEAD = re.compile(r"^\s*(?:[-*+]|[\d۰-۹]+[.)])\s+")
_MD_FOOTNOTE = re.compile(r"\[\^[^\]]{1,6}\]:?")


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


def _clean_query(query: str, max_words: int = 10) -> Optional[str]:
    """Reduce a snippet/hunk query to a searchable plain-text phrase.

    Takes the longest single line, strips markdown markers (heading/quote/table
    leads, list markers, footnote markers, emphasis/code characters), collapses
    whitespace, and caps the result at `max_words` words. Returns None when
    fewer than 3 non-space characters remain (too short to locate reliably).
    """
    if not query:
        return None
    line = max(query.splitlines(), default="", key=lambda s: len(s.strip()))
    line = _MD_LINE_LEAD.sub("", line)
    line = _MD_LIST_LEAD.sub("", line)
    line = _MD_FOOTNOTE.sub(" ", line)
    for ch in "*_`":
        line = line.replace(ch, "")
    words = line.split()[:max_words]
    clean = " ".join(words)
    if len(clean.replace(" ", "")) < 3:
        return None
    return clean


# Encoding differences between our transcriptions and embedded text layers:
# layers often use U+06C0 (ۀ) for هٔ, spaces instead of ZWNJ, and detached /
# reordered punctuation. Variants keep the same strict acceptance rules; they
# only give search_for a chance to match the layer's encoding.
_PUNCT_TO_SPACE = re.compile(r"[:.,;!?،؛؟«»()\[\]{}\-–—…]+")
_HEH_HAMZA = "هٔ"  # هٔ as two codepoints


def _query_variants(q: str) -> list[str]:
    """Ordered, de-duplicated search variants for a phrase or word (original
    first): ZWNJ as space / removed, هٔ -> ۀ, punctuation as space."""
    variants = [q]
    transforms = (
        lambda s: s.replace(_ZWNJ, " "),
        lambda s: s.replace(_ZWNJ, ""),
        lambda s: s.replace(_HEH_HAMZA, "ۀ"),
        lambda s: _PUNCT_TO_SPACE.sub(" ", s),
    )
    for tr in transforms:
        for v in list(variants):
            w = " ".join(tr(v).split())
            if len(w.replace(" ", "")) >= 2 and w not in variants:
                variants.append(w)
    return variants


def _search_word(page: fitz.Page, word: str) -> list[fitz.Rect]:
    """search_for a single word, retrying encoding variants until one hits."""
    for v in _query_variants(word):
        rects = page.search_for(v)
        if rects:
            return rects
    return []


def _rect_to_fracs(rect: fitz.Rect, page_rect: fitz.Rect) -> dict:
    """Normalize a fitz rect to 0-1 fractions of the page, clamped."""
    w = page_rect.width or 1.0
    h = page_rect.height or 1.0

    def _cl(v: float) -> float:
        return max(0.0, min(1.0, v))

    return {
        "x0": _cl((rect.x0 - page_rect.x0) / w),
        "y0": _cl((rect.y0 - page_rect.y0) / h),
        "x1": _cl((rect.x1 - page_rect.x0) / w),
        "y1": _cl((rect.y1 - page_rect.y0) / h),
    }


def _union_rects(rects: list[fitz.Rect]) -> fitz.Rect:
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u


def _line_height(rects: list[fitz.Rect]) -> float:
    """Median hit-rect height; ~12pt fallback when no rects are available."""
    heights = [r.height for r in rects if r.height > 0]
    if not heights:
        return 12.0
    return statistics.median(heights)


def _cluster_occurrences(rects: list[fitz.Rect], lh: float) -> list[fitz.Rect]:
    """Merge hit rects into visual occurrences. RTL text layers often return
    several fragment rects for ONE printed occurrence of a word; rects on the
    same line (vertical centers within 0.6 line-heights) and horizontally
    within one line-height of each other are one occurrence. Returns the
    union rect per occurrence.
    """
    clusters: list[fitz.Rect] = []
    for r in sorted(rects, key=lambda r: ((r.y0 + r.y1) / 2.0, r.x0)):
        merged = False
        for i, u in enumerate(clusters):
            same_line = abs((r.y0 + r.y1) / 2.0 - (u.y0 + u.y1) / 2.0) <= 0.6 * lh
            near_x = r.x0 <= u.x1 + lh and r.x1 >= u.x0 - lh
            if same_line and near_x:
                clusters[i] = u | r
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(r))
    return clusters


def _locate_one(page: fitz.Page, query: str) -> Optional[dict]:
    """Confident-only location of one query on a page. See locate_snippets."""
    clean = _clean_query(query)
    if clean is None:
        return None

    # (2) full-phrase search: accept a single (possibly line-wrapped)
    # occurrence, i.e. all hit rects within one vertical band of ~3 lines.
    # Encoding variants are tried in order; the band check guards each one.
    for phrase in _query_variants(clean):
        phrase_rects = page.search_for(phrase)
        if not phrase_rects:
            continue
        lh = _line_height(phrase_rects)
        y0 = min(r.y0 for r in phrase_rects)
        y1 = max(r.y1 for r in phrase_rects)
        if (y1 - y0) <= 3.0 * lh:
            return _rect_to_fracs(_union_rects(phrase_rects), page.rect)
        # scattered hits (multiple occurrences) -> try next variant / words

    # (3) word fallback: anchor on the rarest word, extend to nearby words.
    # Counting is per occurrence *cluster*, not per raw rect: RTL layers
    # fragment one printed occurrence into several rects.
    words = clean.split()
    raw_hits: list[tuple[str, list[fitz.Rect]]] = [(w, _search_word(page, w)) for w in words]
    all_rects = [r for _w, rects in raw_hits for r in rects]
    lh = _line_height(all_rects)
    hits: list[tuple[str, list[fitz.Rect]]] = [
        (w, _cluster_occurrences(rects, lh)) for w, rects in raw_hits
    ]

    anchor_idx: Optional[int] = None
    best_count = 3  # only words with <= 2 occurrences may anchor
    for i, (_w, rects) in enumerate(hits):
        if rects and len(rects) < best_count:
            best_count = len(rects)
            anchor_idx = i
            if best_count == 1:
                break
    if anchor_idx is None:
        return None  # every word is missing or too common -> ambiguous

    anchor_rects = hits[anchor_idx][1]

    def _support(anchor: fitz.Rect) -> list[fitz.Rect]:
        band_top = anchor.y0 - 1.5 * lh
        band_bot = anchor.y1 + 1.5 * lh
        near = []
        for j, (_w, rects) in enumerate(hits):
            if j == anchor_idx:
                continue
            for r in rects:
                if r.y0 >= band_top and r.y1 <= band_bot:
                    near.append(r)
        return near

    if len(anchor_rects) == 1:
        anchor = anchor_rects[0]
    else:
        # Two occurrences: only trust the one with strictly more support.
        supports = [_support(r) for r in anchor_rects]
        counts = [len(s) for s in supports]
        if counts[0] == counts[1]:
            return None
        anchor = anchor_rects[counts.index(max(counts))]

    return _rect_to_fracs(_union_rects([anchor] + _support(anchor)), page.rect)


def locate_snippets(pdf_path: str | Path, page_no: int, queries: list[str]) -> list[Optional[dict]]:
    """Locate each query string on page `page_no` (1-based) via the embedded
    text layer. Returns, per query, {"x0","y0","x1","y1"} as 0-1 fractions of
    the page, or None. Confident-only: ambiguous or unfound queries yield None
    rather than a guessed box.
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_no - 1]
        return [_locate_one(page, q) for q in queries]
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
