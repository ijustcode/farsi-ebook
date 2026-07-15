"""Tiered snippet locator for the review UI's page-image boxes.

Given a page's Markdown transcription and a set of query snippets (QC issue
text or correction-hunk old text), locate each on the source PDF page and
return a 0-1-fraction box. Three deterministic tiers, each self-gating per
page:

  - Tier A "match": fuzzy normalized word/fragment-window scoring against
    ``page.get_text('words')``. Immune to RTL intra-word character reordering
    (char-sorted equality), Arabic/Persian letterform variation, presentation-
    form glyphs, and legacy text layers that split words into several tokens.
    Requires a decodable Unicode text layer.
  - Tier B "layout": maps the snippet's character offset in the page Markdown
    onto the PDF's cumulative per-line character counts. The line *rects* are
    real geometry even when the text layer is a non-Unicode cipher, so this
    works where Tier A cannot.
  - Tier C "scan": detects printed lines and word gaps directly from page
    pixels, then maps the Markdown character offset onto that image geometry.
    It requires no PDF text layer and is insensitive to font age or OCR support.

Flow per query: Tier A (if the layer decodes) else Tier B else Tier C else
None. review.py then falls back to the QC verifier's model bbox (Tier D) when
this returns None.

Coordinates are 0-1 fractions of the page (x0,y0,x1,y1) plus a "source" of
"match", "layout", or "scan"; review.py scales them to CSS percentages.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np


# Tier A acceptance / tuning constants (measured; see plan).
_MATCH_ACCEPT = 0.72
_MATCH_TIE = 0.05
_DRIFT_PENALTY = 0.02

# Tier B page gate: PDF/MD countable-char ratio must stay in this band.
_LAYOUT_RATIO_LO = 0.5
_LAYOUT_RATIO_HI = 2.0

# Tier A page gate: at least this fraction of normalized page words must carry
# an Arabic-block character.
_ARABIC_WORD_FRAC = 0.30

# Fuzzy snippet-span acceptance (for QC snippets not verbatim in the markdown).
_FUZZY_ACCEPT = 0.65

# Tier C render / segmentation tuning. The scan is deliberately modest
# resolution: enough to resolve line and word gaps while keeping review-page
# startup fast even for dozens of flagged pages.
_SCAN_SCALE = 1.5
_SCAN_X_MARGIN = 0.04
_SCAN_MAX_BLANK_ROWS = 1


@dataclass(frozen=True)
class Query:
    """A snippet to locate. `span` is the exact [start, end) char span in the
    page Markdown when the caller already knows it (correction hunks); None
    means locate.py must find it. Frozen so instances are hashable and usable
    as lru_cache keys.
    """

    text: str
    span: Optional[tuple[int, int]] = None


@dataclass
class _ScanLine:
    """One printed line detected from page pixels, in PDF page coordinates."""

    rect: fitz.Rect
    words: list[fitz.Rect]
    weight: float = 0.0


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

# Arabic->Persian letterform folding + drop ZWNJ (U+200C) and kashida (U+0640).
_FOLD = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ؤ": "و",
        "ئ": "ی",
        "‌": "",
        "ـ": "",
    }
)

_NONWORD_RE = re.compile(r"[^؀-ۿ0-9a-zA-Z]")


def _fold_word(w: str) -> str:
    """Normalize a single word: fold letterforms, strip combining diacritics,
    drop everything that is not an Arabic-block or ASCII alphanumeric char."""
    # Older Persian PDFs commonly encode visible glyphs with the Arabic
    # Presentation Forms blocks (for example ``ﺧ`` instead of ``خ``).
    # NFKC converts those compatibility glyphs back to ordinary Arabic
    # codepoints before _NONWORD_RE gets a chance to discard them.
    w = unicodedata.normalize("NFKC", w)
    w = w.translate(_FOLD)
    w = "".join(c for c in w if not unicodedata.combining(c))
    return _NONWORD_RE.sub("", w)


def _norm_words(text: str) -> list[str]:
    """Fold every whitespace-token of `text`, dropping ones that fold empty."""
    out = []
    for tok in text.split():
        w = _fold_word(tok)
        if w:
            out.append(w)
    return out


# Markdown syntax stripped before counting characters so the Markdown char
# offsets line up with the PDF text layer (which has no markdown).
_FOOTNOTE_RE = re.compile(r"\[\^[^\]]{1,10}\]")
_MD_SYNTAX_RE = re.compile(r"[#>*_`|~\[\]]")


def _countable(s: str) -> int:
    """Count characters that also appear in the printed page: drop markdown
    syntax/footnote markers and all whitespace."""
    s = _FOOTNOTE_RE.sub("", s)
    s = _MD_SYNTAX_RE.sub("", s)
    return sum(1 for c in s if not c.isspace())


def _has_arabic(w: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in w)


# ---------------------------------------------------------------------------
# geometry helpers (copied from render.py, which no longer owns them)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# snippet-span resolution (markdown side)
# ---------------------------------------------------------------------------


def _fuzzy_span(md: str, text: str) -> Optional[tuple[int, int]]:
    """Locate `text` in `md` approximately when an exact find failed.

    Coarse SequenceMatcher window scan (step len//3, quick_ratio prefilter)
    then a fine ±step refine at step 1. Accept the best offset with
    ratio >= _FUZZY_ACCEPT, else None. Covers the ~18% of QC snippets that are
    not verbatim in the transcription.
    """
    tlen = len(text)
    if tlen < 4 or len(md) < tlen:
        return None
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(text)
    step = max(1, tlen // 3)
    best_ratio = 0.0
    best_start = -1
    for start in range(0, len(md) - tlen + 1, step):
        sm.set_seq1(md[start : start + tlen])
        if sm.quick_ratio() < 0.5:
            continue
        r = sm.ratio()
        if r > best_ratio:
            best_ratio = r
            best_start = start
    if best_start < 0:
        return None
    lo = max(0, best_start - step)
    hi = min(len(md) - tlen, best_start + step)
    for start in range(lo, hi + 1):
        sm.set_seq1(md[start : start + tlen])
        r = sm.ratio()
        if r > best_ratio:
            best_ratio = r
            best_start = start
    if best_ratio >= _FUZZY_ACCEPT:
        return (best_start, best_start + tlen)
    return None


def _find_span(md: str, text: str) -> Optional[tuple[int, int]]:
    """Exact find first, then fuzzy. None if neither locates `text`."""
    if not text:
        return None
    i = md.find(text)
    if i != -1:
        return (i, i + len(text))
    return _fuzzy_span(md, text)


def _resolve_span(md: str, q: Query) -> Optional[tuple[int, int]]:
    if q.span is not None:
        return q.span
    return _find_span(md, q.text)


# ---------------------------------------------------------------------------
# Tier B: char-offset prior on PDF line geometry
# ---------------------------------------------------------------------------


def _pdf_lines(page: fitz.Page) -> list[tuple[fitz.Rect, int]]:
    """(rect, non-space-char-count) for every type-0 text line with >= 2
    non-space characters, in reading order (top-to-bottom, then left)."""
    data = page.get_text("dict")
    lines: list[tuple[fitz.Rect, int]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            txt = "".join(s.get("text", "") for s in line.get("spans", []))
            n = len(re.sub(r"\s", "", txt))
            if n >= 2:
                lines.append((fitz.Rect(line["bbox"]), n))
    lines.sort(key=lambda t: (round(t[0].y0, 1), t[0].x0))
    return lines


def _line_word_extent(
    pwords: list[tuple[fitz.Rect, str]],
    line_rect: fitz.Rect,
    u0: float,
    u1: float,
) -> Optional[fitz.Rect]:
    """Narrow a single hit line to the word sub-run covering within-line char
    fractions [u0, u1] (RTL: fraction 0 = right edge = reading start), using the
    line's real word rectangles so the extent tracks actual glyph positions
    rather than a linear char-width guess. Returns None (keep the full line) when
    no words sit on the line.
    """
    yc = (line_rect.y0 + line_rect.y1) / 2
    # per-word char counts, in RTL reading order (rightmost word first)
    line_words = [
        (r, len(t.strip()))
        for r, t in pwords
        if r.y0 - 1.0 <= yc <= r.y1 + 1.0
    ]
    line_words.sort(key=lambda t: -t[0].x1)
    total = sum(c for _r, c in line_words)
    if not line_words or total == 0:
        return None
    lo = max(0.0, min(1.0, u0))
    hi = max(0.0, min(1.0, u1))
    if hi <= lo:
        hi = min(1.0, lo + 1e-6)
    selected: list[fitz.Rect] = []
    cum = 0
    for r, c in line_words:
        w0 = cum / total
        w1 = (cum + c) / total
        if w1 > lo and w0 < hi:
            selected.append(r)
        cum += c
    if not selected:
        idx = min(int(lo * len(line_words)), len(line_words) - 1)
        selected = [line_words[idx][0]]
    return _union_rects(selected)


def _locate_layout(
    page: fitz.Page,
    md: str,
    span: tuple[int, int],
    lines: list[tuple[fitz.Rect, int]],
    pwords: list[tuple[fitz.Rect, str]],
) -> Optional[dict]:
    """Map the markdown char span onto PDF line geometry. See module docstring."""
    if not lines:
        return None
    total_pdf = sum(n for _, n in lines)
    total_md = _countable(md)
    if total_pdf == 0 or total_md == 0:
        return None
    ratio = total_pdf / total_md
    if not (_LAYOUT_RATIO_LO <= ratio <= _LAYOUT_RATIO_HI):
        return None

    pos, end = span
    a = _countable(md[:pos])
    b = a + _countable(md[pos:end])
    fa = a / total_md
    fb = b / total_md
    if fb < fa:
        fa, fb = fb, fa

    # Cumulative fraction ranges of each line on the PDF side.
    line_fracs: list[tuple[fitz.Rect, float, float]] = []
    cum = 0
    for rect, n in lines:
        line_fracs.append((rect, cum / total_pdf, (cum + n) / total_pdf))
        cum += n

    hit_idx = [i for i, (_r, ls, le) in enumerate(line_fracs) if le > fa and ls < fb]
    if not hit_idx:
        # Zero-width / boundary span: take the line containing fa, else nearest.
        contained = [i for i, (_r, ls, le) in enumerate(line_fracs) if ls <= fa < le]
        if contained:
            hit_idx = [contained[0]]
        else:
            hit_idx = [
                min(
                    range(len(line_fracs)),
                    key=lambda i: abs((line_fracs[i][1] + line_fracs[i][2]) / 2 - fa),
                )
            ]

    # Narrow each hit line to the word sub-run the span actually overlaps on
    # that line, then union — so the box is a zoomable fraction of the line(s)
    # rather than the full line width. (No adjacent-line padding: the word
    # extent already tracks where the span lands.)
    rects: list[fitz.Rect] = []
    for i in hit_idx:
        rect, ls, le = line_fracs[i]
        width_f = le - ls
        refined = None
        if width_f > 0:
            u0 = (max(fa, ls) - ls) / width_f
            u1 = (min(fb, le) - ls) / width_f
            refined = _line_word_extent(pwords, rect, u0, u1)
        rects.append(refined if refined is not None else rect)

    box = _rect_to_fracs(_union_rects(rects), page.rect)
    box["source"] = "layout"
    return box


# ---------------------------------------------------------------------------
# Tier C: image-only scan layout
# ---------------------------------------------------------------------------


def _otsu_threshold(gray: np.ndarray) -> int:
    """Return a conservative global ink threshold for a grayscale page."""
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(gray.size)
    values = np.arange(256, dtype=np.float64)
    cumulative_n = np.cumsum(hist)
    cumulative_sum = np.cumsum(hist * values)
    total_sum = cumulative_sum[-1]
    denom = cumulative_n * (total - cumulative_n)
    between = np.zeros(256, dtype=np.float64)
    valid = denom > 0
    delta = total_sum * cumulative_n - cumulative_sum * total
    between[valid] = (delta[valid] ** 2) / denom[valid]
    # Avoid turning paper texture into ink or erasing thin old type.
    return max(70, min(190, int(np.argmax(between))))


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs where the one-dimensional bool array is true."""
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_nearby_runs(
    runs: list[tuple[int, int]], max_gap: int
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        old_start, old_end = merged[-1]
        if start - old_end <= max_gap:
            merged[-1] = (old_start, end)
        else:
            merged.append((start, end))
    return merged


def _scan_page_lines(page: fitz.Page) -> list[_ScanLine]:
    """Detect text-line and visual-word rectangles directly from page pixels.

    This is script-agnostic image geometry, not OCR. Horizontal ink projection
    finds lines despite old fonts or soft scans; vertical gaps inside each line
    provide word-boundary snapping for tighter RTL boxes.
    """
    scale = _SCAN_SCALE
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False
    )
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = gray < _otsu_threshold(gray)
    _h, w = ink.shape
    xlo = int(w * _SCAN_X_MARGIN)
    xhi = int(w * (1.0 - _SCAN_X_MARGIN))
    min_row_ink = max(3, int(w * 0.003))
    active_rows = ink[:, xlo:xhi].sum(axis=1) >= min_row_ink
    row_runs = _merge_nearby_runs(
        _true_runs(active_rows), max_gap=_SCAN_MAX_BLANK_ROWS
    )

    detected: list[_ScanLine] = []
    for y0, y1 in row_runs:
        sub = ink[y0:y1, xlo:xhi]
        _ys, xs = np.nonzero(sub)
        if not len(xs):
            continue
        px0 = xlo + int(xs.min())
        px1 = xlo + int(xs.max()) + 1
        line_h = y1 - y0
        line_w = px1 - px0
        if line_h < 3 or line_w < 10:
            continue
        # Rules and border noise are wide but only a pixel or two tall.
        if line_w / max(1, line_h) > 50:
            continue

        rect = fitz.Rect(px0 / scale, y0 / scale, px1 / scale, y1 / scale)
        # Running page numbers are omitted from Markdown; discard isolated,
        # narrow marks at the extreme page edges so they do not shift offsets.
        yfrac = ((rect.y0 + rect.y1) / 2 - page.rect.y0) / (page.rect.height or 1)
        if (yfrac < 0.04 or yfrac > 0.92) and rect.width / page.rect.width < 0.15:
            continue

        col_active = sub.sum(axis=0) > 0
        glyph_runs = _true_runs(col_active)
        # A real inter-word space scales with font height; smaller gaps are
        # disconnected letters or dots within one Persian word.
        word_gap = max(3, int(round(line_h * 0.20)))
        word_runs = _merge_nearby_runs(glyph_runs, max_gap=word_gap)
        words = [
            fitz.Rect(
                (xlo + start) / scale,
                y0 / scale,
                (xlo + end) / scale,
                y1 / scale,
            )
            for start, end in word_runs
            if end > start
        ]
        words.sort(key=lambda r: -r.x1)  # RTL reading order
        detected.append(_ScanLine(rect=rect, words=words))

    detected.sort(key=lambda line: (line.rect.y0, -line.rect.x1))
    if not detected:
        return []

    median_h = float(np.median([line.rect.height for line in detected])) or 1.0
    for line in detected:
        # Width / font-height approximates character capacity across headings,
        # body text, and smaller footnotes better than raw line width alone.
        effective_h = min(max(line.rect.height, median_h * 0.65), median_h * 1.5)
        line.weight = max(1e-6, line.rect.width / effective_h)
    return detected


def _scan_line_extent(line: _ScanLine, u0: float, u1: float) -> fitz.Rect:
    """Snap an estimated RTL within-line range to detected visual words."""
    lo = max(0.0, min(1.0, u0))
    hi = max(lo + 1e-6, min(1.0, u1))
    right = line.rect.x1 - lo * line.rect.width
    left = line.rect.x1 - hi * line.rect.width
    selected = [word for word in line.words if word.x1 > left and word.x0 < right]
    if not selected and line.words:
        center = (left + right) / 2
        selected = [
            min(line.words, key=lambda word: abs((word.x0 + word.x1) / 2 - center))
        ]
    return (
        _union_rects(selected)
        if selected
        else fitz.Rect(left, line.rect.y0, right, line.rect.y1)
    )


def _locate_scan(
    page: fitz.Page,
    md: str,
    span: tuple[int, int],
    lines: list[_ScanLine],
) -> Optional[dict]:
    """Map a Markdown span onto image-detected line/word geometry."""
    total_md = _countable(md)
    total_layout = sum(line.weight for line in lines)
    if total_md == 0 or total_layout == 0:
        return None

    pos, end = span
    prefix_n = _countable(md[:pos])
    fa = prefix_n / total_md
    fb = (prefix_n + _countable(md[pos:end])) / total_md
    if fb < fa:
        fa, fb = fb, fa

    candidates: list[tuple[float, fitz.Rect]] = []
    cumulative = 0.0
    for line in lines:
        ls = cumulative / total_layout
        le = (cumulative + line.weight) / total_layout
        cumulative += line.weight
        if le <= fa or ls >= fb:
            continue
        width_f = le - ls
        u0 = (max(fa, ls) - ls) / width_f
        u1 = (min(fb, le) - ls) / width_f
        overlap = min(fb, le) - max(fa, ls)
        candidates.append((overlap, _scan_line_extent(line, u0, u1)))

    if not candidates:
        return None
    # A short word cannot genuinely wrap. If proportional offset estimation
    # straddles a line boundary, keep the line carrying most of the estimated
    # span instead of drawing a huge diagonal union across both lines.
    span_chars = _countable(md[pos:end])
    if span_chars <= 14 and len(candidates) > 1:
        rects = [max(candidates, key=lambda item: item[0])[1]]
    else:
        rects = [rect for _overlap, rect in candidates]
    box = _rect_to_fracs(_union_rects(rects), page.rect)
    box["source"] = "scan"
    return box


# ---------------------------------------------------------------------------
# Tier A: fuzzy word-window match
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=200000)
def _wsim(a: str, b: str) -> float:
    """Similarity of two normalized words. 1.0 equal; 0.95 char-sorted-equal
    (immune to RTL intra-word reordering); else difflib ratio. Memoized per
    pair — required for the sliding window to be affordable."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if sorted(a) == sorted(b):
        return 0.95
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _page_words(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """(rect, normalized-word) for every non-empty page word."""
    out: list[tuple[fitz.Rect, str]] = []
    for w in page.get_text("words"):
        nw = _fold_word(w[4])
        if nw:
            out.append((fitz.Rect(w[0], w[1], w[2], w[3]), nw))
    return out


def _tier_a_usable(pwords: list[tuple[fitz.Rect, str]]) -> bool:
    """Per-page Tier A gate: >= 30% of normalized page words carry an
    Arabic-block character (a decodable Persian text layer)."""
    if not pwords:
        return False
    arabic = sum(1 for _r, nw in pwords if _has_arabic(nw))
    return arabic / len(pwords) >= _ARABIC_WORD_FRAC


def _locate_match(
    page: fitz.Page,
    pwords: list[tuple[fitz.Rect, str]],
    query_text: str,
    expected_y: Optional[float],
) -> Optional[dict]:
    """Slide a fuzzy word window over the page words scoring against the
    normalized query words. See module docstring / plan for scoring."""
    qwords = _norm_words(query_text)
    n = len(qwords)
    if n == 0:
        return None
    words = [nw for _r, nw in pwords]
    m = len(words)
    if m == 0:
        return None

    # (score, start, L) for every candidate window.
    candidates: list[tuple[float, int, int]] = []
    for L in range(max(1, n - 1), n + 3):
        penalty = _DRIFT_PENALTY * abs(L - n)
        for start in range(0, m - L + 1):
            window = words[start : start + L]
            total = 0.0
            for k in range(n):
                best = 0.0
                for j in (k - 1, k, k + 1):
                    if 0 <= j < L:
                        s = _wsim(qwords[k], window[j])
                        if s > best:
                            best = s
                total += best
            candidates.append((total / n - penalty, start, L))

    # Some legacy RTL text layers split one visible word into many extraction
    # tokens (occasionally one token per joined-glyph run). Word-for-word
    # alignment cannot match those pages: a three-word query may correspond to
    # seven PDF tokens. Compare contiguous same-line fragments as a joined
    # character stream as well. _wsim's char-sorted equality deliberately
    # tolerates the intra-fragment reversal produced by these PDFs, while the
    # returned start/L still gives us the exact union of the source glyph
    # rectangles.
    qjoined = "".join(qwords)
    qchars = len(qjoined)
    if qchars:
        min_chars = max(1, int(qchars * 0.65))
        max_chars = max(min_chars, int(qchars * 1.35))
        for start in range(m):
            first_rect = pwords[start][0]
            joined = ""
            for stop in range(start, m):
                rect, word = pwords[stop]
                # Never let a fragment candidate spill onto another printed
                # line, even when the PDF's extraction order is unusual.
                if abs(rect.y0 - first_rect.y0) > 1.0:
                    break
                joined += word
                nchars = len(joined)
                if nchars > max_chars:
                    break
                if nchars >= min_chars:
                    length_drift = abs(nchars - qchars) / qchars
                    score = _wsim(qjoined, joined) - _DRIFT_PENALTY * length_drift
                    candidates.append((score, start, stop - start + 1))

    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    top_score, top_start, top_L = candidates[0]
    if top_score < _MATCH_ACCEPT:
        return None

    chosen_start, chosen_L = top_start, top_L
    if expected_y is not None:
        page_h = page.rect.height or 1.0

        def _yfrac(start: int, L: int) -> float:
            rects = [pwords[start + i][0] for i in range(L)]
            cy = sum((r.y0 + r.y1) / 2 for r in rects) / L
            return (cy - page.rect.y0) / page_h

        near = [c for c in candidates if c[0] >= top_score - _MATCH_TIE]
        best_c = min(near, key=lambda c: abs(_yfrac(c[1], c[2]) - expected_y))
        chosen_start, chosen_L = best_c[1], best_c[2]

    rects = [pwords[chosen_start + i][0] for i in range(chosen_L)]
    box = _rect_to_fracs(_union_rects(rects), page.rect)
    box["source"] = "match"
    return box


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def locate_queries(
    pdf_path: str | Path,
    page_no: int,
    page_md: str,
    queries: list[Query],
) -> list[Optional[dict]]:
    """Locate each query on page `page_no` (1-based). Returns, per query, a box
    dict {"x0","y0","x1","y1","source"} (source "match", "layout", or
    "scan") in 0-1 page fractions, or None. Tiers self-gate per page; a query
    that no deterministic tier can place yields None (review.py then uses the
    model bbox).
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_no - 1]
        pwords = _page_words(page)
        a_usable = _tier_a_usable(pwords)
        lines = _pdf_lines(page)

        results: list[Optional[dict]] = []
        spans: list[Optional[tuple[int, int]]] = []
        for q in queries:
            span = _resolve_span(page_md, q)
            spans.append(span)
            b_box = _locate_layout(page, page_md, span, lines, pwords) if span else None
            expected_y = ((b_box["y0"] + b_box["y1"]) / 2) if b_box else None

            box: Optional[dict] = None
            if a_usable:
                box = _locate_match(page, pwords, q.text, expected_y)
            if box is None and b_box is not None:
                box = b_box
            results.append(box)

        # Image analysis is the expensive fallback, so render/segment at most
        # once and only when a text-layer tier left a resolvable query unplaced.
        if any(
            box is None and span is not None
            for box, span in zip(results, spans)
        ):
            scan_lines = _scan_page_lines(page)
            for i, (box, span) in enumerate(zip(results, spans)):
                if box is None and span is not None:
                    results[i] = _locate_scan(page, page_md, span, scan_lines)
        return results
    finally:
        doc.close()
