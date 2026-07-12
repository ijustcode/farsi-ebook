"""Tiered snippet locator for the review UI's page-image boxes.

Given a page's Markdown transcription and a set of query snippets (QC issue
text or correction-hunk old text), locate each on the source PDF page and
return a 0-1-fraction box. Two tiers, each self-gating per page:

  - Tier A "match": fuzzy normalized word-window scoring against
    ``page.get_text('words')``. Immune to RTL intra-word character reordering
    (char-sorted equality) and Arabic/Persian letterform variation (folding).
    Requires a decodable Unicode text layer.
  - Tier B "layout": maps the snippet's character offset in the page Markdown
    onto the PDF's cumulative per-line character counts. The line *rects* are
    real geometry even when the text layer is a non-Unicode cipher, so this
    works where Tier A cannot.

Flow per query: Tier A (if the layer decodes) else Tier B else None. review.py
then falls back to the QC verifier's model bbox (Tier C) when this returns
None.

Coordinates are 0-1 fractions of the page (x0,y0,x1,y1) plus a "source" of
"match" or "layout"; review.py scales them to CSS percentages.
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

# Toggle for the optional single-line x-extent refinement (Tier B). Disabled:
# harness inspection showed within-line x offsets sit below the char-fraction
# method's noise floor (correct line, skewed segment), and a tight box in the
# wrong spot misleads where a full-line box is honestly right.
_X_REFINE = False

# Tier A acceptance / tuning constants (measured; see plan).
_MATCH_ACCEPT = 0.72
_MATCH_TIE = 0.05
_DRIFT_PENALTY = 0.02

# Tier B page gate: PDF/MD countable-char ratio must stay in this band.
_LAYOUT_RATIO_LO = 0.5
_LAYOUT_RATIO_HI = 2.0
# Neighbor line inclusion: span boundary within this fraction of a hit line's
# own width of the shared edge pulls in the adjacent line.
_NEIGHBOR_FRAC = 0.10

# Tier A page gate: at least this fraction of normalized page words must carry
# an Arabic-block character.
_ARABIC_WORD_FRAC = 0.30

# Fuzzy snippet-span acceptance (for QC snippets not verbatim in the markdown).
_FUZZY_ACCEPT = 0.65


@dataclass(frozen=True)
class Query:
    """A snippet to locate. `span` is the exact [start, end) char span in the
    page Markdown when the caller already knows it (correction hunks); None
    means locate.py must find it. Frozen so instances are hashable and usable
    as lru_cache keys.
    """

    text: str
    span: Optional[tuple[int, int]] = None


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


def _refine_x(rect: fitz.Rect, local_start: float, local_end: float) -> Optional[fitz.Rect]:
    """Refine a single hit line's x-extent to the span's within-line fractions,
    measured RTL from the right edge. Returns None (keep the full line) when the
    refined width would be < 15% of the line width.
    """
    w = rect.width
    if w <= 0:
        return None
    ls = max(0.0, min(1.0, local_start))
    le = max(0.0, min(1.0, local_end))
    if le <= ls:
        return None
    # RTL: fraction 0 is at the right edge (x1), growing leftward.
    x_right = rect.x1 - ls * w
    x_left = rect.x1 - le * w
    if (x_right - x_left) < 0.15 * w:
        return None
    return fitz.Rect(x_left, rect.y0, x_right, rect.y1)


def _locate_layout(
    page: fitz.Page,
    md: str,
    span: tuple[int, int],
    lines: list[tuple[fitz.Rect, int]],
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

    # Neighbor rule: pull in an adjacent line only when a span boundary lands
    # within _NEIGHBOR_FRAC of the outermost hit line's width of the shared edge.
    first, last = hit_idx[0], hit_idx[-1]
    _r, lsf, lef = line_fracs[first]
    if first > 0 and (fa - lsf) <= _NEIGHBOR_FRAC * (lef - lsf):
        hit_idx.insert(0, first - 1)
    _r, lsl, lel = line_fracs[last]
    if last < len(line_fracs) - 1 and (lel - fb) <= _NEIGHBOR_FRAC * (lel - lsl):
        hit_idx.append(last + 1)

    rects = [line_fracs[i][0] for i in hit_idx]

    # Optional x-refinement for a single-line hit.
    if _X_REFINE and len(rects) == 1:
        rect, ls, le = line_fracs[hit_idx[0]]
        width_f = le - ls
        if width_f > 0:
            local_start = (fa - ls) / width_f
            local_end = (fb - ls) / width_f
            refined = _refine_x(rect, local_start, local_end)
            if refined is not None:
                rects = [refined]

    box = _rect_to_fracs(_union_rects(rects), page.rect)
    box["source"] = "layout"
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
    dict {"x0","y0","x1","y1","source"} (source "match" or "layout") in 0-1
    page fractions, or None. Tiers self-gate per page; a query that neither tier
    can place yields None (review.py then uses the model bbox).
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_no - 1]
        pwords = _page_words(page)
        a_usable = _tier_a_usable(pwords)
        lines = _pdf_lines(page)

        results: list[Optional[dict]] = []
        for q in queries:
            span = _resolve_span(page_md, q)
            b_box = _locate_layout(page, page_md, span, lines) if span else None
            expected_y = ((b_box["y0"] + b_box["y1"]) / 2) if b_box else None

            box: Optional[dict] = None
            if a_usable:
                box = _locate_match(page, pwords, q.text, expected_y)
            if box is None and b_box is not None:
                box = b_box
            results.append(box)
        return results
    finally:
        doc.close()
