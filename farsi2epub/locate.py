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
this returns None. Scan-sourced boxes can afterwards be upgraded by
``refine_scan_boxes`` (source "scan_vlm"): a caller-injected VLM reader
transcribes clean strip crops of the hit line(s) and the query is re-aligned
onto the detected word rectangles. Refined results preserve an ordered
``segments`` list (one rectangle per printed line) as well as the legacy union
envelope — locate.py itself never imports an LLM client.

Every tier is vertically coherent: a located run may occupy one printed line or
wrap onto vertically *adjacent* lines (in which case it too carries an ordered
``segments`` list plus the union envelope), but a run whose members straddle
non-consecutive lines is rejected rather than unioned into a tall empty box.

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
from typing import Callable, Optional

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

# Scan-box VLM refinement: acceptance for aligning a query inside a strip
# transcription (same scale as _MATCH_ACCEPT), plus strip crop padding so
# dots/diacritics at the crop edges are not clipped.
_REFINE_ACCEPT = 0.72
_STRIP_Y_PAD = 0.25  # fraction of median detected line height
_STRIP_X_PAD = 0.02  # fraction of page width

# Vertical coherence of a located run (Tiers A and B). A phrase may wrap onto
# the next printed line, but never skips lines: consecutive lines of one hit
# must sit within this many median line heights of each other (center to
# center), and their line indices must be consecutive on the page.
_LINE_GAP_MAX = 2.5
# y-center clustering tolerance as a fraction of median line height, floored at
# the +-1.0pt band _line_word_extent already uses.
_LINE_CLUSTER_FRAC = 0.6
_LINE_CLUSTER_MIN = 1.0

# A PDF text line whose writing direction is not (approximately) left-to-right
# horizontal is rotated/vertical decorative text: it is not part of the
# horizontal reading flow, so it must not consume any of Tier B's cumulative
# character budget. Fallback shape test for lines with no usable `dir`.
_DIR_TOL = 0.1
_VERTICAL_ASPECT = 2.0  # pdf line: height > 2.0 x width => vertical column
_SCAN_VERTICAL_ASPECT = 3.0  # detected scan line: height/width > 3.0

# Near-tie band used by the refinement window scorer (same value Tier A uses).
# Exported so scoring harnesses can reason about the candidate pool without
# reaching for a private name.
BEST_WINDOW_TIE = _MATCH_TIE


@dataclass(frozen=True)
class Query:
    """A snippet to locate. `span` is the exact [start, end) char span in the
    page Markdown when the caller already knows it (correction hunks); None
    means locate.py must find it. `alts` carries alternate texts refinement
    may also try (for wrong-word findings the query is the incorrect
    transcription while the image shows the corrected text); the deterministic
    tiers ignore it. Frozen so instances are hashable and usable as lru_cache
    keys.
    """

    text: str
    span: Optional[tuple[int, int]] = None
    alts: tuple[str, ...] = ()


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

# _NONWORD_RE keeps the WHOLE Arabic block, and that block carries Persian
# PUNCTUATION as well as letters and digits — ، ؛ ؟ ٪ ۔ and friends. So the
# fold used to be asymmetric in a way nobody intended:
#
#     _fold_word("متن.") == "متن"    # ASCII full stop stripped
#     _fold_word("زد،")  == "زد،"    # Persian comma survives
#
# and every window comparison in this module — Tier A's word-multiset match,
# _best_window / _window_candidates scoring, _align_strip — then treated a
# comma-terminated word as a different token from the bare word. Persian prose
# is comma-dense, so any query whose first or last word abuts punctuation was
# systematically penalized against a page that punctuates it differently (or a
# reader that spaces the comma off).
#
# Only the Arabic block needs filtering: everything else _NONWORD_RE admits is
# ASCII alphanumeric. Categories P* are punctuation and Cf is invisible format
# (U+0600-0605 number signs, U+06DD). Arabic-Indic digits (Nd) and the letters
# are untouched, and so is tatweel U+0640 — kashida is already dropped by
# _FOLD, and widening this to letter modifiers would be a separate change with
# its own measurement.
_ARABIC_PUNCT_TABLE = {
    cp: None
    for cp in range(0x0600, 0x0700)
    if unicodedata.category(chr(cp)) in {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Cf"}
}


def _fold_word(w: str) -> str:
    """Normalize a single word: fold letterforms, strip combining diacritics,
    drop everything that is not an Arabic-block or ASCII alphanumeric char,
    and drop Arabic-block punctuation (see _ARABIC_PUNCT_TABLE)."""
    # Older Persian PDFs commonly encode visible glyphs with the Arabic
    # Presentation Forms blocks (for example ``ﺧ`` instead of ``خ``).
    # NFKC converts those compatibility glyphs back to ordinary Arabic
    # codepoints before _NONWORD_RE gets a chance to discard them.
    w = unicodedata.normalize("NFKC", w)
    w = w.translate(_FOLD)
    w = "".join(c for c in w if not unicodedata.combining(c))
    return _NONWORD_RE.sub("", w).translate(_ARABIC_PUNCT_TABLE)


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

# Whole lines that carry no character into the page's reading flow. Tiers B and
# C map a Markdown character offset onto page geometry proportionally, so a line
# counted here but not printed there (or printed somewhere else entirely) skews
# every box below it on the page.
#   fence/hr/table-delimiter — markdown-only, never printed at all.
#
# Footnote *definition* lines are deliberately NOT excluded. They look like a
# candidate — out of reading-order position at the page foot — but both
# geometry sides detect them (they are real printed lines to _pdf_lines and
# real ink to _scan_page_lines), and they are last in the Markdown as well as
# last on the page, so the two sequences already correspond. Dropping them from
# the Markdown side only is a measured regression: on bachehaye_ghali p8 it
# widened the located box from 0.43 to 0.74 of the page.
_FENCE_LINE_RE = re.compile(r"^\s*```")
_HR_LINE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_DELIM_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

# The ` --- ` hemistich separator inside a verse block (see TRANSCRIBE_SYSTEM):
# three characters in the Markdown, printed as whitespace between hemistichs.
_HEMISTICH_RE = re.compile(r"(?<=\s)-{3}(?=\s)")

# Block segmentation (see "zone-anchored offset mapping" below).
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_FOOTNOTE_DEF_RE = re.compile(r"^\s{0,3}\[\^[^\]]{1,10}\]:")


def _out_of_flow_line(line: str) -> bool:
    """True when none of the line's characters belong to the reading flow."""
    return bool(
        _FENCE_LINE_RE.match(line)
        or _HR_LINE_RE.match(line)
        or _TABLE_DELIM_RE.match(line)
    )


@functools.lru_cache(maxsize=64)
def _countable_prefix(md: str) -> tuple[int, ...]:
    """prefix[i] = countable characters in md[:i], line-aware.

    Memoized per page: the callers need many prefix lookups into the same
    Markdown, and rebuilding the mask per lookup made offset mapping O(n^2).
    """
    counts = bytearray(len(md))  # 1 where the character reaches the page
    pos = 0
    for line in md.splitlines(keepends=True):
        end = pos + len(line)
        if not _out_of_flow_line(line):
            drop = bytearray(len(line))
            for rx in (_FOOTNOTE_RE, _MD_SYNTAX_RE, _HEMISTICH_RE):
                for m in rx.finditer(line):
                    for i in range(m.start(), m.end()):
                        drop[i] = 1
            for i, c in enumerate(line):
                if not drop[i] and not c.isspace():
                    counts[pos + i] = 1
        pos = end
    prefix = [0] * (len(md) + 1)
    total = 0
    for i, v in enumerate(counts):
        total += v
        prefix[i + 1] = total
    return tuple(prefix)


def _countable_upto(md: str, i: int) -> int:
    """Countable characters in md[:i] (line-aware)."""
    p = _countable_prefix(md)
    return p[max(0, min(i, len(md)))]


def _countable_span(md: str, a: int, b: int) -> int:
    """Countable characters in md[a:b] (line-aware)."""
    return _countable_upto(md, b) - _countable_upto(md, a)


def _countable(s: str) -> int:
    """Count characters that also appear in the printed page: drop markdown
    syntax/footnote markers and all whitespace.

    Substring-only form, kept for callers that genuinely have no page context.
    Prefer _countable_span/_countable_upto, which additionally drop whole
    out-of-flow lines — a decision that cannot be made from a substring.
    """
    s = _FOOTNOTE_RE.sub("", s)
    s = _MD_SYNTAX_RE.sub("", s)
    return sum(1 for c in s if not c.isspace())


def _has_arabic(w: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in w)


# ---------------------------------------------------------------------------
# zone-anchored offset mapping (shared by Tiers B and C)
# ---------------------------------------------------------------------------
#
# Tiers B and C both turn a Markdown character offset into a position in the
# page's reading flow. Done as a single cumulative proportion over the whole
# page, *any* local discrepancy — a dropped word, a heading whose printed size
# bears no relation to its character count, a footnote block set in smaller
# type — displaces every box below it on the page. The error accumulates
# downward, which is exactly the observed signature (boxes land systematically
# early, and a quarter of them miss the region entirely).
#
# So instead: partition both sides into blocks (Markdown paragraphs/headings/
# footnote definitions; printed paragraphs recovered from line geometry), align
# the two block sequences monotonically by relative size, and interpolate
# *within* the aligned pair. Errors then stay inside one paragraph instead of
# propagating to the foot of the page. When the alignment is low-confidence the
# map degrades to the previous global proportion.
#
# Printed-paragraph recovery is RTL-specific and deliberately geometric: in
# justified Persian body text a paragraph's first line is indented on the
# *right* (the reading start) and its last line falls short on the *left*.
# Neither signal needs a text layer, so the same partitioner serves the PDF
# line rects of Tier B and the ink rects of Tier C.

# A vertical gap this many median line heights above the median gap starts a
# new printed block.
_BLOCK_GAP_H = 0.6
# Right-indent / left-shortfall tolerance, as a fraction of the body measure.
_BLOCK_INDENT_FRAC = 0.03
# Monotone alignment: how many blocks one side may collapse into the other,
# the per-merge and per-drop preference penalties, and the residual mass above
# which the alignment is judged untrustworthy and the global map is used.
_ALIGN_MAX_SPAN = 3
_ALIGN_MERGE_PENALTY = 0.01
_ALIGN_DROP_PENALTY = 0.02
_ALIGN_MAX_RESIDUAL = 0.35


@functools.lru_cache(maxsize=64)
def _md_blocks(md: str) -> tuple[tuple[int, int], ...]:
    """Markdown blocks as [start, end) char spans, in document order.

    Blank lines separate blocks; headings, fenced verse lines and footnote
    *definition* lines each form their own block, because each is printed as
    its own physically separate run of lines. Blocks with no countable
    character are dropped — they have no counterpart in the page geometry.
    """
    blocks: list[tuple[int, int]] = []
    start: Optional[int] = None
    end = 0
    in_fence = False

    def _flush() -> None:
        nonlocal start
        if start is not None and _countable_span(md, start, end) > 0:
            blocks.append((start, end))
        start = None

    pos = 0
    for line in md.splitlines(keepends=True):
        stop = pos + len(line)
        if _FENCE_LINE_RE.match(line):
            _flush()
            in_fence = not in_fence
        elif not line.strip():
            _flush()
        elif (
            in_fence
            or _HEADING_LINE_RE.match(line)
            or _FOOTNOTE_DEF_RE.match(line)
        ):
            _flush()
            start, end = pos, stop
            _flush()
        else:
            if start is None:
                start = pos
            end = stop
        pos = stop
    _flush()
    return tuple(blocks)


def _geom_blocks(rects: list[fitz.Rect]) -> list[tuple[int, int]]:
    """Group printed lines into blocks as inclusive [lo, hi] index ranges.

    A new block starts at line i when any of these hold:
      * the vertical gap before it is materially larger than the page's median
        line gap (headings, footnote rules, section breaks);
      * its right edge is indented relative to the body's right margin — in RTL
        text that is a paragraph's indented first line;
      * the previous line fell short of the body's left margin — in RTL text
        that is a paragraph's last line.
    """
    n = len(rects)
    if n <= 1:
        return [(0, 0)] if n else []
    widths = np.array([r.width for r in rects], dtype=float)
    measure = float(np.percentile(widths, 90)) or 1.0
    right = float(np.percentile([r.x1 for r in rects], 75))
    left = float(np.percentile([r.x0 for r in rects], 25))
    med_h = float(np.median([r.height for r in rects])) or 1.0
    gaps = [max(0.0, rects[i + 1].y0 - rects[i].y1) for i in range(n - 1)]
    med_gap = float(np.median(gaps))
    tol = _BLOCK_INDENT_FRAC * measure

    starts = [0]
    for i in range(1, n):
        prev, cur = rects[i - 1], rects[i]
        gap = max(0.0, cur.y0 - prev.y1)
        if (
            gap > med_gap + _BLOCK_GAP_H * med_h
            or cur.x1 < right - tol
            or prev.x0 > left + tol
        ):
            starts.append(i)
    return [
        (s, (starts[k + 1] - 1) if k + 1 < len(starts) else n - 1)
        for k, s in enumerate(starts)
    ]


def _align_blocks(
    mw: list[float], gw: list[float]
) -> Optional[list[tuple[int, int, int, int]]]:
    """Monotonically align two block-weight sequences by relative size.

    Gale-Church-shaped DP over normalized weights: each step consumes 1..N
    Markdown blocks against 1..N geometry blocks (only one side may exceed 1,
    so blocks merge and split but never cross), or drops a block on one side.
    Cost is the absolute weight mismatch of the pairing plus small structural
    penalties, so a 1:1 pairing wins ties. Returns the groups as
    (md_lo, md_hi, geom_lo, geom_hi) half-open pairs, or None when the best
    alignment still misallocates more than _ALIGN_MAX_RESIDUAL of the page.
    """
    ni, nj = len(mw), len(gw)
    if ni == 0 or nj == 0:
        return None
    pm = [0.0] * (ni + 1)
    for i, v in enumerate(mw):
        pm[i + 1] = pm[i] + v
    pg = [0.0] * (nj + 1)
    for j, v in enumerate(gw):
        pg[j + 1] = pg[j] + v

    inf = float("inf")
    dp = [[inf] * (nj + 1) for _ in range(ni + 1)]
    back: list[list[Optional[tuple[int, int]]]] = [
        [None] * (nj + 1) for _ in range(ni + 1)
    ]
    dp[0][0] = 0.0
    moves = [
        (a, b)
        for a in range(0, _ALIGN_MAX_SPAN + 1)
        for b in range(0, _ALIGN_MAX_SPAN + 1)
        # Merges and splits, plus the classic 2:2 (both sides segmented the
        # same run differently). Wider many-to-many steps are excluded so the
        # DP cannot buy a low cost by blobbing the page into one group.
        if (a or b) and (min(a, b) <= 1 or a == b == 2)
    ]
    for i in range(ni + 1):
        for j in range(nj + 1):
            if dp[i][j] == inf:
                continue
            base = dp[i][j]
            for a, b in moves:
                if i + a > ni or j + b > nj:
                    continue
                dm = pm[i + a] - pm[i]
                dg = pg[j + b] - pg[j]
                if a == 0 or b == 0:
                    cost = dm + dg + _ALIGN_DROP_PENALTY
                else:
                    cost = abs(dm - dg) + _ALIGN_MERGE_PENALTY * (a + b - 2)
                if base + cost < dp[i + a][j + b]:
                    dp[i + a][j + b] = base + cost
                    back[i + a][j + b] = (a, b)
    if dp[ni][nj] == inf:
        return None

    groups: list[tuple[int, int, int, int]] = []
    i, j = ni, nj
    residual = 0.0
    while i or j:
        step = back[i][j]
        if step is None:
            return None
        a, b = step
        groups.append((i - a, i, j - b, j))
        dm = pm[i] - pm[i - a]
        dg = pg[j] - pg[j - b]
        residual += (dm + dg) if (a == 0 or b == 0) else abs(dm - dg)
        i, j = i - a, j - b
    if residual > _ALIGN_MAX_RESIDUAL:
        return None
    groups.reverse()
    return groups


def _geom_key(
    rects: list[fitz.Rect], weights: list[float]
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Hashable, cache-stable signature of a page's line geometry."""
    return tuple(
        (
            round(r.x0, 1),
            round(r.y0, 1),
            round(r.x1, 1),
            round(r.y1, 1),
            round(float(w), 3),
        )
        for r, w in zip(rects, weights)
    )


@functools.lru_cache(maxsize=32)
def _zone_points(
    md: str, key: tuple[tuple[float, float, float, float, float], ...]
) -> tuple[tuple[float, float], ...]:
    """Control points (md_fraction, geometry_weight_fraction) of the anchored
    map for one page, or () to mean "use the global proportional map".

    Memoized per (page Markdown, page geometry) — every query on a page shares
    one map.
    """
    if len(key) < 2:
        return ()
    rects = [fitz.Rect(k[0], k[1], k[2], k[3]) for k in key]
    weights = [k[4] for k in key]
    total_md = _countable_upto(md, len(md))
    total_g = sum(weights)
    if total_md <= 0 or total_g <= 0:
        return ()

    mblocks = _md_blocks(md)
    gblocks = _geom_blocks(rects)
    if len(mblocks) < 2 and len(gblocks) < 2:
        return ()

    cum_g = [0.0] * (len(rects) + 1)
    for i, w in enumerate(weights):
        cum_g[i + 1] = cum_g[i] + w

    mw = [_countable_span(md, a, b) / total_md for a, b in mblocks]
    gw = [(cum_g[hi + 1] - cum_g[lo]) / total_g for lo, hi in gblocks]
    groups = _align_blocks(mw, gw)
    if not groups:
        return ()

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for _mi, mj, _gi, gj in groups:
        if mj == 0 or gj == 0:
            continue
        x = _countable_upto(md, mblocks[mj - 1][1]) / total_md
        y = cum_g[gblocks[gj - 1][1] + 1] / total_g
        if x > points[-1][0] and y > points[-1][1]:
            points.append((x, y))
    points.append((1.0, 1.0))
    # Fewer than one interior anchor is the global map by another name.
    if len(points) < 3:
        return ()
    return tuple(points)


def _apply_zone(points: tuple[tuple[float, float], ...], f: float) -> float:
    """Piecewise-linear, monotone map of a Markdown fraction onto the page's
    geometry-weight fraction. Identity when no anchors were established."""
    if not points:
        return f
    if f <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if f <= x1:
            if x1 <= x0:
                return y1
            return y0 + (y1 - y0) * (f - x0) / (x1 - x0)
    return points[-1][1]


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
            center = sum((x.y0 + x.y1) / 2 for x in groups[-1]) / len(groups[-1])
        else:
            groups.append([r])
            center = c
    return groups


def _rect_center_y(rects: list[fitz.Rect]) -> float:
    return sum((r.y0 + r.y1) / 2 for r in rects) / len(rects)


def _coherent_segments(
    rects: list[fitz.Rect],
    page_line_centers: list[float],
    median_h: float,
) -> Optional[list[fitz.Rect]]:
    """Split a located run into one rect per printed line, or reject it.

    A genuine phrase occupies one line or wraps onto vertically *adjacent*
    lines. A window whose members straddle non-consecutive printed lines (for
    example a decorative heading plus a body line two lines below) is spurious:
    unioning it yields a tall mostly-empty rectangle. Returns the per-line
    rects in top-to-bottom order, or None when the run is not vertically
    coherent.
    """
    groups = _cluster_rect_lines(rects)
    if not groups:
        return None
    if len(groups) == 1:
        return [_union_rects(groups[0])]

    centers = [_rect_center_y(g) for g in groups]
    for prev, cur in zip(centers, centers[1:]):
        if abs(cur - prev) > _LINE_GAP_MAX * median_h:
            return None
    if page_line_centers:
        idxs = sorted(
            min(
                range(len(page_line_centers)),
                key=lambda i: abs(page_line_centers[i] - c),
            )
            for c in centers
        )
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            return None
    return [_union_rects(g) for g in groups]


def _attach_segments(box: dict, rects: list[fitz.Rect], page_rect: fitz.Rect) -> dict:
    """Add a per-printed-line ``segments`` list (same shape refine_scan_boxes
    emits) when a located run wrapped across lines. Single-line runs are left
    as a plain box."""
    if len(rects) > 1:
        box["segments"] = [_rect_to_fracs(r, page_rect) for r in rects]
    return box


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


def _is_horizontal_line(line: dict, rect: fitz.Rect, n_chars: int) -> bool:
    """True when a PDF text line belongs to the horizontal reading flow.

    Rotated/vertical text (decorative sidebars, spine labels) is real geometry
    but its position in the reading order is not recoverable, so Tier B must
    not let it consume any of the cumulative character budget — doing so shifts
    the offset->line mapping for every line after it. PyMuPDF reports a writing
    direction on the line (and on its spans); ``(1, 0)`` is normal horizontal
    text. When no direction is available, fall back to the bounding shape: a
    tall narrow rect holding several characters is a vertical column.
    """
    direction = line.get("dir")
    if not direction:
        for span in line.get("spans", []):
            if span.get("dir"):
                direction = span["dir"]
                break
    if direction and len(direction) >= 2:
        dx, dy = float(direction[0]), float(direction[1])
        return abs(dx - 1.0) <= _DIR_TOL and abs(dy) <= _DIR_TOL
    return not (n_chars >= 3 and rect.height > _VERTICAL_ASPECT * max(rect.width, 1e-6))


def _pdf_lines(page: fitz.Page) -> list[tuple[fitz.Rect, int]]:
    """(rect, non-space-char-count) for every horizontal type-0 text line with
    >= 2 non-space characters, in reading order (top-to-bottom, then left).
    Rotated/vertical lines are excluded — see _is_horizontal_line."""
    data = page.get_text("dict")
    lines: list[tuple[fitz.Rect, int]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            txt = "".join(s.get("text", "") for s in line.get("spans", []))
            n = len(re.sub(r"\s", "", txt))
            if n < 2:
                continue
            rect = fitz.Rect(line["bbox"])
            if not _is_horizontal_line(line, rect, n):
                continue
            lines.append((rect, n))
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
    total_md = _countable_upto(md, len(md))
    if total_pdf == 0 or total_md == 0:
        return None
    ratio = total_pdf / total_md
    if not (_LAYOUT_RATIO_LO <= ratio <= _LAYOUT_RATIO_HI):
        return None

    pos, end = span
    # Line-aware counting on both sides: the substring form of _countable
    # cannot drop whole out-of-flow lines, so mixing it with the line-aware
    # total below made numerator and denominator disagree.
    a = _countable_upto(md, pos)
    b = a + _countable_span(md, pos, end)
    zone = _zone_points(
        md, _geom_key([r for r, _n in lines], [float(n) for _r, n in lines])
    )
    fa = _apply_zone(zone, a / total_md)
    fb = _apply_zone(zone, b / total_md)
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
    per_line: list[tuple[fitz.Rect, float]] = []  # (rect, overlap fraction)
    for i in hit_idx:
        rect, ls, le = line_fracs[i]
        width_f = le - ls
        refined = None
        if width_f > 0:
            u0 = (max(fa, ls) - ls) / width_f
            u1 = (min(fb, le) - ls) / width_f
            refined = _line_word_extent(pwords, rect, u0, u1)
        per_line.append(
            (refined if refined is not None else rect, max(0.0, min(fb, le) - max(fa, ls)))
        )

    # Same vertical-coherence discipline as Tier A: a span may wrap onto the
    # next printed line but cannot jump a gap. Keep the longest run of
    # vertically adjacent hit lines (the one carrying most of the span) and
    # emit one segment per line instead of a single tall union.
    median_h = _median_height([r for r, _n in lines])
    runs: list[list[tuple[fitz.Rect, float]]] = [[per_line[0]]]
    for prev, cur in zip(per_line, per_line[1:]):
        gap = abs(_rect_center_y([cur[0]]) - _rect_center_y([prev[0]]))
        if gap > _LINE_GAP_MAX * median_h:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    best = max(runs, key=lambda run: sum(o for _r, o in run))
    rects = [r for r, _o in best]

    box = _rect_to_fracs(_union_rects(rects), page.rect)
    box["source"] = "layout"
    return _attach_segments(box, rects, page.rect)


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
        # The inverse shape is a rotated/vertical text column or a vertical
        # rule: real ink, but not a printed line of the reading flow, so it
        # must not consume any of the cumulative character budget.
        if line_h / max(1, line_w) > _SCAN_VERTICAL_ASPECT:
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


def _scan_hits(
    md: str,
    span: tuple[int, int],
    lines: list[_ScanLine],
) -> list[tuple[int, float, float, float]]:
    """Proportionally map a Markdown char span onto detected line weights.

    Returns, per overlapped line in reading order, (line_index, u0, u1,
    overlap) where [u0, u1] is the within-line RTL char range and `overlap`
    the span fraction the line carries; [] when the page or span has no
    countable geometry. Shared by _locate_scan and refine_scan_boxes.
    """
    total_md = _countable_upto(md, len(md))
    total_layout = sum(line.weight for line in lines)
    if total_md == 0 or total_layout == 0:
        return []

    pos, end = span
    prefix_n = _countable_upto(md, pos)
    zone = _zone_points(
        md, _geom_key([ln.rect for ln in lines], [ln.weight for ln in lines])
    )
    fa = _apply_zone(zone, prefix_n / total_md)
    fb = _apply_zone(
        zone, (prefix_n + _countable_span(md, pos, end)) / total_md
    )
    if fb < fa:
        fa, fb = fb, fa

    hits: list[tuple[int, float, float, float]] = []
    cumulative = 0.0
    for i, line in enumerate(lines):
        ls = cumulative / total_layout
        le = (cumulative + line.weight) / total_layout
        cumulative += line.weight
        if le <= fa or ls >= fb:
            continue
        width_f = le - ls
        u0 = (max(fa, ls) - ls) / width_f
        u1 = (min(fb, le) - ls) / width_f
        overlap = min(fb, le) - max(fa, ls)
        hits.append((i, u0, u1, overlap))
    return hits


def _locate_scan(
    page: fitz.Page,
    md: str,
    span: tuple[int, int],
    lines: list[_ScanLine],
) -> Optional[dict]:
    """Map a Markdown span onto image-detected line/word geometry."""
    hits = _scan_hits(md, span, lines)
    if not hits:
        return None
    pos, end = span
    # A short word cannot genuinely wrap. If proportional offset estimation
    # straddles a line boundary, keep the line carrying most of the estimated
    # span instead of drawing a huge diagonal union across both lines.
    span_chars = _countable_span(md, pos, end)
    if span_chars <= 14 and len(hits) > 1:
        hits = [max(hits, key=lambda h: h[3])]
    rects = [_scan_line_extent(lines[i], u0, u1) for i, u0, u1, _overlap in hits]
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
    top_score = candidates[0][0]
    if top_score < _MATCH_ACCEPT:
        return None

    # A scored window is a contiguous run of extraction tokens, which says
    # nothing about where those tokens are *printed*: a window can straddle a
    # decorative heading and the body line below it, whose union is a tall
    # mostly-empty rectangle. Prefer, inside the near-tie band, a window that
    # sits on a single printed line — a genuine wrap only ever has half the
    # query on each line, which scores far below the band — then take the first
    # vertically coherent candidate; wrapped runs keep one segment per line.
    page_line_centers = [
        _rect_center_y(group)
        for group in _cluster_rect_lines([r for r, _nw in pwords])
    ]
    median_h = _median_height([r for r, _nw in pwords])
    page_h = page.rect.height or 1.0

    def _rects_of(start: int, L: int) -> list[fitz.Rect]:
        return [pwords[start + i][0] for i in range(L)]

    def _near_key(c: tuple[float, int, int]) -> tuple:
        rects = _rects_of(c[1], c[2])
        multi = len(_cluster_rect_lines(rects)) > 1
        if expected_y is None:
            return (multi,)
        cy = sum((r.y0 + r.y1) / 2 for r in rects) / len(rects)
        return (multi, abs((cy - page.rect.y0) / page_h - expected_y))

    near = [c for c in candidates if c[0] >= top_score - _MATCH_TIE]
    near.sort(key=_near_key)
    rest = [
        c
        for c in candidates
        if c[0] < top_score - _MATCH_TIE and c[0] >= _MATCH_ACCEPT
    ]

    for _score, start, L in near + rest:
        segments = _coherent_segments(
            _rects_of(start, L), page_line_centers, median_h
        )
        if segments is None:
            continue
        box = _rect_to_fracs(_union_rects(segments), page.rect)
        box["source"] = "match"
        return _attach_segments(box, segments, page.rect)
    return None


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
        # Tier B boxes held back on cipher pages — see the precedence note below.
        layout_fallback: list[Optional[dict]] = []
        for q in queries:
            span = _resolve_span(page_md, q)
            spans.append(span)
            b_box = _locate_layout(page, page_md, span, lines, pwords) if span else None
            expected_y = ((b_box["y0"] + b_box["y1"]) / 2) if b_box else None

            box: Optional[dict] = None
            if a_usable:
                box = _locate_match(page, pwords, q.text, expected_y)
                # Tier A failed but the layer decodes, so B's line geometry is
                # as trustworthy as it gets on this page: take it now.
                if box is None and b_box is not None:
                    box = b_box
            layout_fallback.append(b_box)
            results.append(box)

        # TIER PRECEDENCE ON CIPHER PAGES. Tier B used to be accepted here
        # unconditionally, and since it has no_box_rate 0.0 it always returned
        # something — so on a glyph-cipher page (a_usable False: real line
        # rects, undecodable characters) Tier C was structurally NEVER reached,
        # and neither was refine_scan_boxes, which only accepts source "scan".
        # That is why haaji-agha, 50 of the 54 layout cases, is the worst book.
        # Measured, unrefined, Tier C already beats Tier B 0.5748 to 0.4231, and
        # refinement takes its cases to 0.8430 — so a cipher page is better
        # served by ink geometry that can then be content-checked than by exact
        # line rects that never can.
        #
        # Tier B stays as the FALLBACK rather than being dropped: it is the only
        # tier here that always produces a box, and letting a scan miss fall
        # through to no box at all would trade a mediocre box for the model's
        # bbox estimate (measured 0.0).
        if any(
            box is None and span is not None
            for box, span in zip(results, spans)
        ):
            scan_lines = _scan_page_lines(page)
            for i, (box, span) in enumerate(zip(results, spans)):
                if box is None and span is not None:
                    results[i] = _locate_scan(page, page_md, span, scan_lines)
                    if results[i] is None:
                        results[i] = layout_fallback[i]
        return results
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# scan-box refinement: VLM strip oracle ("scan_vlm")
# ---------------------------------------------------------------------------


def _strip_rect(
    page: fitz.Page, lines: list[_ScanLine], lo: int, hi: int, median_h: float
) -> fitz.Rect:
    """Padded page-coordinate crop rect covering detected lines lo..hi
    (inclusive), full line width, clamped to the page. Padding keeps dots and
    diacritics at the crop edges from being clipped."""
    u = _union_rects([line.rect for line in lines[lo : hi + 1]])
    return fitz.Rect(
        max(page.rect.x0, u.x0 - _STRIP_X_PAD * page.rect.width),
        max(page.rect.y0, u.y0 - _STRIP_Y_PAD * median_h),
        min(page.rect.x1, u.x1 + _STRIP_X_PAD * page.rect.width),
        min(page.rect.y1, u.y1 + _STRIP_Y_PAD * median_h),
    )


def _render_strip(page: fitz.Page, rect: fitz.Rect) -> Optional[bytes]:
    """Clean PNG crop of `rect` — page pixels only, nothing occluding glyphs.
    Scale targets ~1600px of strip width, clamped to 2-4x so narrow strips
    stay legible without ballooning the payload."""
    if rect.width < 1.0 or rect.height < 1.0:
        return None
    s = max(2.0, min(4.0, 1600 / rect.width))
    pix = page.get_pixmap(matrix=fitz.Matrix(s, s), clip=rect)
    return pix.tobytes("png")


def _window_candidates(
    variants: list[list[str]], flat_words: list[str]
) -> list[tuple[float, int, int]]:
    """Every (score, start, L) candidate window, in generation order.

    Same scoring shape as _locate_match: for each variant of length n and each
    window length L in [max(1, n-1), n+1], the mean over query words of the best
    _wsim against the window word at the same index +-1, minus a length-drift
    penalty. Tier A deliberately uses a wider L range (up to n+2); the two are
    not unified.
    """
    m = len(flat_words)
    candidates: list[tuple[float, int, int]] = []
    if m == 0:
        return candidates
    for qwords in variants:
        n = len(qwords)
        if n == 0:
            continue
        for L in range(max(1, n - 1), n + 2):
            penalty = _DRIFT_PENALTY * abs(L - n)
            for start in range(0, m - L + 1):
                total = 0.0
                for k in range(n):
                    best = 0.0
                    for j in (k - 1, k, k + 1):
                        if 0 <= j < L:
                            s = _wsim(qwords[k], flat_words[start + j])
                            if s > best:
                                best = s
                    total += best
                candidates.append((total / n - penalty, start, L))
    return candidates


def _best_window(
    variants: list[list[str]], flat_words: list[str]
) -> tuple[float, int, int]:
    """Best-scoring (score, start_index, length) window over all variants.

    Pure and fitz-free so scoring harnesses can call it directly. Returns
    (-1.0, -1, 0) when no candidate window exists (empty words or variants).
    """
    candidates = _window_candidates(variants, flat_words)
    if not candidates:
        return (-1.0, -1, 0)
    candidates.sort(key=lambda t: -t[0])
    return candidates[0]


def _align_strip(
    query_words_variants: list[list[str]],
    vlm_lines: list[str],
    strip_lines: list[_ScanLine],
    prior_center: Optional[fitz.Point],
) -> Optional[tuple[list[fitz.Rect], dict]]:
    """Align a query (any of its normalized-word variants) inside a VLM strip
    transcription and map the winning window back onto detected word rects.

    Pure scoring/geometry — no PDF, no API — so it is unit-testable. VLM lines
    arrive top-to-bottom; VLM words within a line are in reading order and
    `_ScanLine.words` is already RTL reading order (rightmost first), so word
    index i of a VLM line corresponds to `words[i]` whenever counts agree.
    Returns (rects, meta) — page-coordinate rectangles in printed reading
    order, one unioned rectangle per covered line, plus alignment diagnostics
    {"score", "counts_agree", "n_near_tie", "vlm_lines", "strip_lines"} — or
    None when no window reaches _REFINE_ACCEPT. Keeping line-local geometry
    here prevents a wrapped RTL phrase from collapsing into a page-width
    diagonal envelope.
    """
    vwords = [_norm_words(t) for t in vlm_lines]
    flat: list[tuple[int, int, str]] = []  # (vlm line, in-line index, word)
    for j, ws in enumerate(vwords):
        for i, w in enumerate(ws):
            flat.append((j, i, w))
    m = len(flat)
    if m == 0:
        return None

    # Same scoring shape as _locate_match: mean best _wsim per query word
    # against the window word at the same index +-1, length-drift penalized.
    candidates = _window_candidates(query_words_variants, [w for _j, _i, w in flat])
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    top_score = candidates[0][0]
    if top_score < _REFINE_ACCEPT:
        return None

    counts_agree = len(vlm_lines) == len(strip_lines)
    meta = {
        "score": float(top_score),
        "counts_agree": counts_agree,
        "n_near_tie": sum(1 for c in candidates if c[0] >= top_score - _MATCH_TIE),
        "vlm_lines": len(vlm_lines),
        "strip_lines": len(strip_lines),
    }

    def _window_rects(start: int, L: int) -> Optional[list[fitz.Rect]]:
        window = flat[start : start + L]
        if not counts_agree:
            # The VLM segmented the strip into a different number of lines
            # than were detected: no per-line index mapping exists, so place
            # the window char-proportionally over all strip word rects in
            # reading order (width-weighted).
            total_chars = sum(len(w) for _j, _i, w in flat)
            if total_chars == 0:
                return None
            c0 = sum(len(w) for _j, _i, w in flat[:start])
            c1 = c0 + sum(len(w) for _j, _i, w in window)
            u0, u1 = c0 / total_chars, c1 / total_chars
            rects_ro = [
                (line_idx, r)
                for line_idx, line in enumerate(strip_lines)
                for r in line.words
            ]
            total_w = sum(r.width for _line_idx, r in rects_ro)
            if not rects_ro or total_w <= 0:
                return None
            selected: list[tuple[int, fitz.Rect]] = []
            cum = 0.0
            for line_idx, r in rects_ro:
                w0 = cum / total_w
                w1 = (cum + r.width) / total_w
                if w1 > u0 and w0 < u1:
                    selected.append((line_idx, r))
                cum += r.width
            if not selected:
                idx = min(int(u0 * len(rects_ro)), len(rects_ro) - 1)
                selected = [rects_ro[idx]]
            by_detected_line: dict[int, list[fitz.Rect]] = {}
            for line_idx, rect in selected:
                by_detected_line.setdefault(line_idx, []).append(rect)
            return [
                _union_rects(by_detected_line[line_idx])
                for line_idx in sorted(by_detected_line)
            ]

        rects: list[fitz.Rect] = []
        by_line: dict[int, list[int]] = {}  # window words are contiguous per line
        for j, i, _w in window:
            by_line.setdefault(j, []).append(i)
        for j, idxs in sorted(by_line.items()):
            line = strip_lines[j]
            ws = vwords[j]
            if line.words and len(ws) == len(line.words):
                rects.append(_union_rects([line.words[i] for i in idxs]))
            else:
                # Word segmentation disagrees on this line only: place the
                # covered folded-char run proportionally within the line.
                total = sum(len(w) for w in ws)
                i0, i1 = min(idxs), max(idxs)
                c0 = sum(len(w) for w in ws[:i0])
                c1 = sum(len(w) for w in ws[: i1 + 1])
                if total:
                    rects.append(_scan_line_extent(line, c0 / total, c1 / total))
        return rects or None

    best_start, best_L = candidates[0][1], candidates[0][2]
    if prior_center is not None:
        # Mirror _MATCH_TIE handling: among near-best windows prefer the one
        # whose center is closest to the original scan placement.
        best_d: Optional[float] = None
        best_rects: Optional[list[fitz.Rect]] = None
        for score, start, L in candidates:
            if score < top_score - _MATCH_TIE:
                break
            rects = _window_rects(start, L)
            if rects is None:
                continue
            r = _union_rects(rects)
            d = (
                ((r.x0 + r.x1) / 2 - prior_center.x) ** 2
                + ((r.y0 + r.y1) / 2 - prior_center.y) ** 2
            )
            if best_d is None or d < best_d:
                best_d = d
                best_rects = rects
        if best_rects is not None:
            return best_rects, meta
    rects = _window_rects(best_start, best_L)
    return (rects, meta) if rects is not None else None


def refine_scan_boxes(
    pdf_path: str | Path,
    page_no: int,
    page_md: str,
    queries: list[Query],
    boxes: list[Optional[dict]],
    read_strips: Callable[[list[bytes]], list[list[str]]],
) -> list[Optional[dict]]:
    """Refine Tier C boxes using a VLM as a local transcription oracle.

    For each query whose `boxes[i]` came from the scan tier, render a clean
    strip crop of the initially-hit detected line(s) +-1, have `read_strips`
    (caller-injected — locate.py never talks to an LLM) transcribe every
    unique strip in ONE batched call, then deterministically re-align the
    query (and any Query.alts) inside the reading and snap it onto the
    detected word rectangles: line identification becomes exact, within-line
    placement a word-index lookup. Failures widen to +-2 lines for one more
    batched call. Returns per query a box with source "scan_vlm", an ordered
    per-line ``segments`` list, the segments' union as the legacy envelope, and
    a ``debug`` dict of alignment diagnostics (the _align_strip meta plus the
    ``radius`` of the pass that succeeded — diagnostics only, never sent to the
    browser); or None (non-scan position, or refinement failed — the caller keeps the
    plain scan box).
    """
    results: list[Optional[dict]] = [None] * len(queries)
    todo = [
        i
        for i, box in enumerate(boxes[: len(queries)])
        if box is not None and box.get("source") == "scan"
    ]
    if not todo:
        return results

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_no - 1]
        lines = _scan_page_lines(page)
        if not lines:
            return results
        median_h = float(np.median([line.rect.height for line in lines])) or 1.0

        ranges: dict[int, tuple[int, int]] = {}  # hit-line index range
        priors: dict[int, fitz.Point] = {}  # original placement center
        variants: dict[int, list[list[str]]] = {}  # normalized query + alts
        for i in todo:
            span = _resolve_span(page_md, queries[i])
            hits = _scan_hits(page_md, span, lines) if span else []
            if not hits:
                continue
            ranges[i] = (hits[0][0], hits[-1][0])
            box = boxes[i]
            priors[i] = fitz.Point(
                page.rect.x0 + (box["x0"] + box["x1"]) / 2 * page.rect.width,
                page.rect.y0 + (box["y0"] + box["y1"]) / 2 * page.rect.height,
            )
            qv = [_norm_words(queries[i].text)]
            qv.extend(_norm_words(alt) for alt in queries[i].alts)
            variants[i] = [v for v in qv if v]

        pending = [i for i in ranges if variants.get(i)]
        for radius in (1, 2):  # first pass +-1 line; failures retry once at +-2
            if not pending:
                break
            strips: list[bytes] = []
            strip_range: list[tuple[int, int]] = []
            by_key: dict[tuple[int, int], int] = {}  # dedupe identical crops
            strip_of: dict[int, int] = {}
            for i in pending:
                lo = max(0, ranges[i][0] - radius)
                hi = min(len(lines) - 1, ranges[i][1] + radius)
                key = (lo, hi)
                if key not in by_key:
                    png = _render_strip(page, _strip_rect(page, lines, lo, hi, median_h))
                    if png is None:
                        continue
                    by_key[key] = len(strips)
                    strips.append(png)
                    strip_range.append(key)
                if key in by_key:
                    strip_of[i] = by_key[key]
            if not strips:
                break
            readings = read_strips(strips)
            failed: list[int] = []
            for i in pending:
                si = strip_of.get(i)
                reading = readings[si] if si is not None and si < len(readings) else []
                aligned = None
                if reading:
                    lo, hi = strip_range[si]
                    aligned = _align_strip(
                        variants[i], reading, lines[lo : hi + 1], priors[i]
                    )
                if not aligned or not aligned[0]:
                    failed.append(i)
                    continue
                rects, meta = aligned
                box = _rect_to_fracs(_union_rects(rects), page.rect)
                box["source"] = "scan_vlm"
                box["segments"] = [
                    _rect_to_fracs(rect, page.rect) for rect in rects
                ]
                box["debug"] = {**meta, "radius": radius}
                results[i] = box
            pending = failed
        return results
    finally:
        doc.close()
