"""Embedded-word matching and correction-independent image geometry.

Proportional placement algorithms exist only in tests/historical.
"""
from __future__ import annotations
import re
import functools
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
import fitz
import numpy as np
_ARABIC_WORD_FRAC = 0.3
_SCAN_SCALE = 1.5
_SCAN_X_MARGIN = 0.04
_SCAN_MAX_BLANK_ROWS = 1
_LINE_CLUSTER_FRAC = 0.6
_LINE_CLUSTER_MIN = 1.0
_SCAN_VERTICAL_ASPECT = 3.0

@dataclass(frozen=True)
class Query:
    """Full correction text, exact Markdown span, alternatives and edit kind.

    Both evidence paths use all alternatives and surrounding context. Geometry
    never selects the occurrence. Frozen instances serve as cache identities.
    """
    text: str
    span: Optional[tuple[int, int]] = None
    alts: tuple[str, ...] = ()
    kind: str = 'replacement'

@dataclass
class _ScanLine:
    """One printed line detected from page pixels, in PDF page coordinates."""
    rect: fitz.Rect
    words: list[fitz.Rect]
    weight: float = 0.0
_FOLD = str.maketrans({'ي': 'ی', 'ك': 'ک', 'ۀ': 'ه', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ی', '\u200c': '', 'ـ': ''})
_NONWORD_RE = re.compile('[^\u0600-ۿ0-9a-zA-Z]')
_ARABIC_PUNCT_TABLE = {cp: None for cp in range(1536, 1792) if unicodedata.category(chr(cp)) in {'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps', 'Cf'}}

def _fold_word(w: str) -> str:
    """Normalize a single word: fold letterforms, strip combining diacritics,
    drop everything that is not an Arabic-block or ASCII alphanumeric char,
    and drop Arabic-block punctuation (see _ARABIC_PUNCT_TABLE)."""
    w = unicodedata.normalize('NFKC', w)
    w = w.translate(_FOLD)
    w = ''.join((c for c in w if not unicodedata.combining(c)))
    return _NONWORD_RE.sub('', w).translate(_ARABIC_PUNCT_TABLE).translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))

def _norm_words(text: str) -> list[str]:
    """Fold every whitespace-token of `text`, dropping ones that fold empty."""
    out = []
    for tok in text.split():
        w = _fold_word(tok)
        if w:
            out.append(w)
    return out

def _has_arabic(w: str) -> bool:
    return any(('\u0600' <= c <= 'ۿ' for c in w))

def _rect_to_fracs(rect: fitz.Rect, page_rect: fitz.Rect) -> dict:
    """Normalize a fitz rect to 0-1 fractions of the page, clamped."""
    w = page_rect.width or 1.0
    h = page_rect.height or 1.0

    def _cl(v: float) -> float:
        return max(0.0, min(1.0, v))
    return {'x0': _cl((rect.x0 - page_rect.x0) / w), 'y0': _cl((rect.y0 - page_rect.y0) / h), 'x1': _cl((rect.x1 - page_rect.x0) / w), 'y1': _cl((rect.y1 - page_rect.y0) / h)}

def _union_rects(rects: list[fitz.Rect]) -> fitz.Rect:
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u

def _median_height(rects: list[fitz.Rect]) -> float:
    if not rects:
        return 1.0
    return float(np.median([r.height for r in rects])) or 1.0

def _cluster_rect_lines(rects: list[fitz.Rect]) -> list[list[fitz.Rect]]:
    """Group rects into printed lines by y-center, top to bottom.

    Tolerance is 0.6 x the median rect height, floored at the +-1.0pt band
    ``_line_word_extent`` uses, so mixed-size glyphs on one baseline still
    cluster together while separate printed lines stay apart.
    """
    if not rects:
        return []
    tol = max(_LINE_CLUSTER_MIN, _LINE_CLUSTER_FRAC * _median_height(rects))
    ordered = sorted(rects, key=lambda r: ((r.y0 + r.y1) / 2, -r.x1))
    groups: list[list[fitz.Rect]] = [[ordered[0]]]
    center = (ordered[0].y0 + ordered[0].y1) / 2
    for r in ordered[1:]:
        c = (r.y0 + r.y1) / 2
        if abs(c - center) <= tol:
            groups[-1].append(r)
            center = sum(((x.y0 + x.y1) / 2 for x in groups[-1])) / len(groups[-1])
        else:
            groups.append([r])
            center = c
    return groups

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
    between[valid] = delta[valid] ** 2 / denom[valid]
    return max(70, min(190, int(np.argmax(between))))

def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs where the one-dimensional bool array is true."""
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))

def _merge_nearby_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
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

def _split_tall_runs(runs: list[tuple[int, int]], row_ink: np.ndarray, h0: float, depth: int=3) -> list[tuple[int, int]]:
    """Split row runs that are tall enough to be two printed lines fused.

    MEASURED (bachehaye_ghali p64): two adjacent printed lines whose ascenders
    and descenders touch produce ONE 62px row run against a 27px reference, so
    the detector returns 28 lines where the page prints 29. The judge's
    Needleman-Wunsch line alignment reconciles that deficit by DROPPING a line
    in the middle of the page, which shifts every reading line between the drop
    and the merge one printed line up — the box is then read out against the
    wrong ink and a perfectly-placed box grades as catastrophically wrong.

    `h0` must come from the UNMERGED runs so a merge cannot inflate the
    reference it is being measured against. A run is split at the minimum-ink
    row of its middle third, but only when that row is a genuine VALLEY (at
    most half the run's median ink) and both halves are still plausible lines
    (>= 0.5 * h0). A heading, a drop cap or a rule is legitimately tall and has
    no valley, so it is left alone.
    """
    if h0 <= 0 or depth <= 0:
        return list(runs)
    out: list[tuple[int, int]] = []
    for y0, y1 in runs:
        h = y1 - y0
        if h <= 1.6 * h0:
            out.append((y0, y1))
            continue
        band0 = y0 + h // 3
        band1 = y0 + 2 * h // 3
        if band1 <= band0:
            out.append((y0, y1))
            continue
        band = row_ink[band0:band1]
        cut = band0 + int(np.argmin(band))
        median_ink = float(np.median(row_ink[y0:y1]))
        if float(row_ink[cut]) > 0.5 * median_ink or cut - y0 < 0.5 * h0 or y1 - cut < 0.5 * h0:
            out.append((y0, y1))
            continue
        out.extend(_split_tall_runs([(y0, cut), (cut, y1)], row_ink, h0, depth - 1))
    return out

def _scan_page_lines(page: fitz.Page) -> list[_ScanLine]:
    """Detect text-line and visual-word rectangles directly from page pixels.

    This is script-agnostic image geometry, not OCR. Horizontal ink projection
    finds lines despite old fonts or soft scans; vertical gaps inside each line
    provide word-boundary snapping for tighter RTL boxes.
    """
    scale = _SCAN_SCALE
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = gray < _otsu_threshold(gray)
    _h, w = ink.shape
    xlo = int(w * _SCAN_X_MARGIN)
    xhi = int(w * (1.0 - _SCAN_X_MARGIN))
    min_row_ink = max(3, int(w * 0.003))
    row_ink = ink[:, xlo:xhi].sum(axis=1)
    active_rows = row_ink >= min_row_ink
    raw_runs = _true_runs(active_rows)
    h0 = float(np.median([e - s for s, e in raw_runs])) if raw_runs else 0.0
    row_runs = _split_tall_runs(_merge_nearby_runs(raw_runs, max_gap=_SCAN_MAX_BLANK_ROWS), row_ink, h0)
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
        if line_w / max(1, line_h) > 50:
            continue
        if line_h / max(1, line_w) > _SCAN_VERTICAL_ASPECT:
            continue
        rect = fitz.Rect(px0 / scale, y0 / scale, px1 / scale, y1 / scale)
        yfrac = ((rect.y0 + rect.y1) / 2 - page.rect.y0) / (page.rect.height or 1)
        if (yfrac < 0.04 or yfrac > 0.92) and rect.width / page.rect.width < 0.15:
            continue
        col_active = sub.sum(axis=0) > 0
        glyph_runs = _true_runs(col_active)
        word_gap = max(3, int(round(line_h * 0.2)))
        word_runs = _merge_nearby_runs(glyph_runs, max_gap=word_gap)
        words = [fitz.Rect((xlo + start) / scale, y0 / scale, (xlo + end) / scale, y1 / scale) for start, end in word_runs if end > start]
        words.sort(key=lambda r: -r.x1)
        detected.append(_ScanLine(rect=rect, words=words))
    detected.sort(key=lambda line: (line.rect.y0, -line.rect.x1))
    if not detected:
        return []
    median_h = float(np.median([line.rect.height for line in detected])) or 1.0
    for line in detected:
        effective_h = min(max(line.rect.height, median_h * 0.65), median_h * 1.5)
        line.weight = max(1e-06, line.rect.width / effective_h)
    return detected

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
    for w in page.get_text('words'):
        nw = _fold_word(w[4])
        if nw:
            out.append((fitz.Rect(w[0], w[1], w[2], w[3]), nw))
    return out

def _tier_a_usable(pwords: list[tuple[fitz.Rect, str]]) -> bool:
    """Per-page Tier A gate: >= 30% of normalized page words carry an
    Arabic-block character (a decodable Persian text layer)."""
    if not pwords:
        return False
    arabic = sum((1 for _r, nw in pwords if _has_arabic(nw)))
    return arabic / len(pwords) >= _ARABIC_WORD_FRAC

def locate_queries(pdf_path: str | Path, page_no: int, page_md: str, queries: list[Query]) -> list[Optional[dict]]:
    """Embedded-text placement; misses require the unified scan service."""
    from .page_map import PrintedWord, place
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_no - 1]
        pwords = _page_words(page)
        if not _tier_a_usable(pwords):
            return [None] * len(queries)
        groups = _cluster_rect_lines([r for r, _ in pwords])
        printed = []
        for line_no, group in enumerate(groups):
            for rect in sorted(group, key=lambda r: -r.x1):
                text = next((t for r, t in pwords if r == rect))
                nr = _rect_to_fracs(rect * page.rotation_matrix, page.rect)
                printed.append(PrintedWord(text, [nr[k] for k in ('x0', 'y0', 'x1', 'y1')], line_no, ['pdf_words']))
        return [place(page_md, q, printed, source='match').box for q in queries]
