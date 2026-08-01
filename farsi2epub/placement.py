"""Box-placement geometry and the word-miss-distance metric.

This module is the shared core between the offline accuracy harness
(``tests/bbox_score.py``) and the live review UI. It was extracted OUT of the
harness rather than copied into production, because a duplicated copy would
drift and the offline number would stop describing shipped behaviour — which is
the exact failure mode the accuracy loop exists to prevent. The harness now
imports every geometry primitive from here and keeps only what is
judging-specific (truth-window search against the frozen answer key, verdicts,
degenerate routing, report aggregation, ``compare``, the CLI).

Two questions live here, and they are not the same question:

* **Which words does this rectangle cover?** — ``cover``. A reading (a list of
  text lines, top to bottom) is aligned monotonically onto detected ink lines,
  every word gets a rectangle, and the words whose horizontal span the box
  covers by >=50% are the covered run. Purely geometric; no model is ever asked.
* **How far is that from the words it should have covered?** —
  ``word_miss_distances``, the metric defined in the project owner's METRIC.md:
  per target word, the reading-order index distance to the nearest covered word,
  0 when the target itself is covered. Index space, never screen space: the text
  is RTL.

LIVE vs OFFLINE — the one substantive difference
------------------------------------------------
The harness grades against a **frozen, independent answer key** (a whole-page
reading in ``books/<slug>/locate_page_read.json``, sometimes billed). A book a
human is reviewing for the first time has no such key, so ``score_box`` builds
its reference from what is already on disk. That makes the live number a
**placement** check —

    "is the box on the words this phrase is printed at?"

— which catches the line-alignment slips and geometry-source mis-routing that
motivated this work, and CANNOT catch a wrong transcription. Callers must not
present it as accuracy or confidence.

Two references, in order of preference (see ``score_box`` for the measurements):

* ``text_layer`` — the page's own PDF word lines. These ARE printed lines, so
  they pair with the detected ink lines exactly, and the query is found in them
  by content. Independent of the transcription as well as of the locator.
* ``markdown`` — the page's ``text/NNNN.md``, used when no usable text layer
  exists. A transcription reflows printed lines into paragraphs, so the line
  pairing usually fails and the answer is an honest "cannot verify": an
  image-only scan cannot be checked without someone reading the ink, and this
  scorer never bills for that.

Zero API calls, ever. Everything here is PyMuPDF plus arithmetic.

Zero API calls, ever. Everything here is PyMuPDF plus arithmetic.
"""

from __future__ import annotations

import bisect
import statistics
from collections import Counter
from typing import NamedTuple, Optional

import fitz

from . import locate

# One printed line: (rectangle, folded word), in RTL reading order.
Line = list[tuple[fitz.Rect, str]]

# Text-layer routing gate: fraction of the folded page-layer words that also
# appear (multiset) in the folded page markdown. Two probes, both required.
#
# MEASURED (mean over every case page of each book):
#
#     book              raw recall   merged recall   layer kind
#     Hossein_shenasi       0.72         0.84        clean Unicode
#     boof-e-koor           0.17         0.58        cipher (presentation
#                                                    forms, fragmented tokens)
#     review-test           0.16         0.55        cipher
#     haaji-agha            0.00         0.00        glyph cipher (#/&)
#     bachehaye_ghali       0.00         0.00        image-only scan
#
# "Merged" is the fragment-merged reader tokenization (page_reader_lines);
# "raw" is PyMuPDF's own tokens. The merge heuristic pulls the cipher books up
# to 0.55-0.58, which is why the original 0.40 merged-only gate routed them
# deterministic — and that was WRONG. Measured agreement between the
# deterministic reader and the VLM on 24 matched boof-e-koor cases: only 71%
# (17/24), and 4 of the 7 disagreements were the deterministic reader claiming
# wrong_region on boxes the VLM read correctly (confirmed by eye on the crops).
# On the clean-Unicode book agreement was 9/11. Conclusion: the deterministic
# reader is a trustworthy oracle on clean Unicode text layers ONLY.
#
# The RAW recall is what separates the two classes cleanly (0.72 vs 0.17, a 4x
# gap); the merged recall does not (0.84 vs 0.58). So the raw probe is the
# primary gate at 0.45 — the midpoint of the measured gap — and the merged gate
# is kept as a weak sanity floor.
#
# Do not lower these without re-running the agreement measurement above.
LAYER_RAW_RECALL_MIN = 0.45
LAYER_RECALL_MIN = 0.70

# Candidate fragment-merge thresholds, as a fraction of the line height. 0.0
# means "trust the extractor's own tokens"; the rest recover printed words from
# legacy layers that split one word into many glyph-run tokens.
MERGE_RATIOS = (0.0, 0.015, 0.03, 0.05, 0.08, 0.12)


# ---------------------------------------------------------------------------
# small geometry primitives
# ---------------------------------------------------------------------------


def overlap_frac(a: fitz.Rect, b: fitz.Rect) -> float:
    """Fraction of `a`'s area that lies inside `b`."""
    inter = fitz.Rect(a) & b
    area = a.width * a.height
    if area <= 0:
        return 0.0
    if inter.is_empty:
        return 0.0
    return (inter.width * inter.height) / area


def box_rects(page: fitz.Page, box: dict) -> list[fitz.Rect]:
    """Page-coordinate rectangles for a box: one per segment for a segmented
    scan_vlm box, else the single envelope."""
    pr = page.rect

    def _mk(d: dict) -> fitz.Rect:
        return fitz.Rect(
            pr.x0 + d["x0"] * pr.width,
            pr.y0 + d["y0"] * pr.height,
            pr.x0 + d["x1"] * pr.width,
            pr.y0 + d["y1"] * pr.height,
        )

    segments = box.get("segments")
    if box.get("source") == "scan_vlm" and isinstance(segments, list) and segments:
        return [_mk(s) for s in segments if isinstance(s, dict)]
    return [_mk(box)]


def layer_geometry_usable(page: fitz.Page) -> bool:
    """True when the PDF's own word rects may stand in for the scan detector.

    This is a WEAKER and genuinely different question from `layer_usable`,
    which asks whether PyMuPDF may stand in for the VLM *reader*. That one
    rightly gates on token recall against the Markdown, because it decides what
    becomes TRUTH. Geometry needs no such thing: the caller already holds the
    reference text and needs the layer only for WHERE words sit.

    Measured, conflating the two silently mis-graded a whole book. Every
    sampled `boof-e-koor` page has a clean, correctly-positioned Unicode layer
    but folds against its Markdown at raw recall 0.115-0.264, far under
    `LAYER_RAW_RECALL_MIN` (0.45). All 34 pages were therefore forced onto scan
    geometry, which found no words inside the box at all, and 10 correctly
    placed boxes were reported as `no_covered_words` — pixel-verified: the box
    on `boof-e-koor:p20:h1` sits exactly on میلولیدند, its own target.

    `locate._tier_a_usable` is the whole predicate, and it is sufficient because
    it already excludes both failure modes the scan fallback exists for:
      * an image-only scan has no text layer at all (`bachehaye_ghali`: 0/14
        sampled pages pass);
      * a GLYPH CIPHER decodes to non-Persian, so its rects tokenize
        differently from the printed words and would mis-attribute the covered
        run (`haaji-agha`: 0/14 pass — it substitutes literal '#'/'&' for
        Persian glyphs; see `crop_word_lines`'s docstring for the measured
        damage).
    Both keep the scan path. Only a real, decodable Persian layer passes.
    """
    try:
        return bool(locate._tier_a_usable(locate._page_words(page)))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# the deterministic reader: PyMuPDF word rects grouped into printed lines
# ---------------------------------------------------------------------------


def merge_line(line: Line, ratio: float) -> Line:
    """Join neighbouring tokens on one RTL-ordered line whose horizontal gap is
    below `ratio` * the line's median glyph height."""
    if not line or ratio <= 0.0:
        return list(line)
    heights = [r.height for r, _w in line if r.height > 0]
    h = statistics.median(heights) if heights else 1.0
    max_gap = ratio * h
    out: Line = [line[0]]
    for rect, word in line[1:]:
        prev_rect, prev_word = out[-1]
        if prev_rect.x0 - rect.x1 <= max_gap:
            out[-1] = (prev_rect | rect, prev_word + word)
        else:
            out.append((rect, word))
    return out


def raw_page_lines(page: fitz.Page) -> list[Line]:
    """Every folded page word, grouped into printed lines (y0 within +-1.0) and
    sorted by -x1 within a line — the RTL reading-order convention
    ``locate._line_word_extent`` uses."""
    kept = [
        (fitz.Rect(w[0], w[1], w[2], w[3]), locate._fold_word(w[4]))
        for w in page.get_text("words")
    ]
    kept = [(r, f) for r, f in kept if f]
    lines: list[Line] = []
    for rect, folded in sorted(kept, key=lambda t: (t[0].y0, -t[0].x1)):
        if lines and abs(lines[-1][0][0].y0 - rect.y0) <= 1.0:
            lines[-1].append((rect, folded))
        else:
            lines.append([(rect, folded)])
    return [sorted(line, key=lambda t: -t[0].x1) for line in lines]


def recall_of(lines: list[Line], md_counts: Counter) -> float:
    have = Counter(w for line in lines for _r, w in line)
    total = sum(have.values())
    if not total:
        return 0.0
    return sum(min(c, md_counts.get(w, 0)) for w, c in have.items()) / total


def page_reader_lines(
    page: fitz.Page, page_md: str
) -> tuple[list[Line], float, float]:
    """Printed-word lines for the deterministic reader, plus (merge ratio,
    recall).

    Several of these books' legacy RTL layers split one visible word into many
    extraction tokens (``شالمهٔ هندی`` comes out as ``ش الم ه هن د ی``), and
    their inter-word gaps are far narrower than a normal space. A reader —
    human or VLM — sees printed words, so the deterministic reader must too;
    otherwise every box on such a page grades wrong_region purely because the
    word counts disagree. The merge threshold is not guessed: each candidate is
    scored by how much of the resulting word multiset actually occurs in the
    page's own transcription, and the best one wins. Pages that need no merging
    keep ratio 0.0 because merging can only lower their recall.
    """
    raw = raw_page_lines(page)
    md_counts = Counter(locate._norm_words(page_md))
    best_lines, best_ratio, best_recall = raw, 0.0, recall_of(raw, md_counts)
    for ratio in MERGE_RATIOS[1:]:
        cand = [merge_line(line, ratio) for line in raw]
        recall = recall_of(cand, md_counts)
        if recall > best_recall + 1e-9:
            best_lines, best_ratio, best_recall = cand, ratio, recall
    return best_lines, best_ratio, best_recall


def layer_recall(pwords: list[tuple[fitz.Rect, str]], page_md: str) -> float:
    """Multiset recall of folded page-layer words inside the folded markdown.

    A glyph-cipher text layer (haaji-agha substitutes literal '#' and '&' for
    Persian glyphs) still passes the Arabic-fraction gate but its words do not
    occur in the transcription, so this probe is what actually separates
    "PyMuPDF can act as the reader" from "only a VLM can".
    """
    if not pwords:
        return 0.0
    have = Counter(nw for _r, nw in pwords)
    md = Counter(locate._norm_words(page_md))
    hit = sum(min(c, md.get(w, 0)) for w, c in have.items())
    total = sum(have.values())
    return hit / total if total else 0.0


def layer_usable(
    page: fitz.Page,
    page_md: str,
    recall: Optional[float] = None,
    raw_recall: Optional[float] = None,
) -> bool:
    """True when PyMuPDF can stand in for the VLM reader on this page.

    Three gates: locate._tier_a_usable (a decodable Persian text layer at all),
    the RAW token recall (the probe that actually separates a clean Unicode
    layer from a cipher one — see the constants above), and the fragment-merged
    reader recall as a floor.
    """
    pwords = locate._page_words(page)
    if not locate._tier_a_usable(pwords):
        return False
    if recall is None:
        _lines, _ratio, recall = page_reader_lines(page, page_md)
    if raw_recall is None:
        raw_recall = layer_recall(pwords, page_md)
    return raw_recall >= LAYER_RAW_RECALL_MIN and recall >= LAYER_RECALL_MIN


# ---------------------------------------------------------------------------
# reading -> detected ink line alignment
# ---------------------------------------------------------------------------


def flatten(reading: dict) -> tuple[list[str], list[int]]:
    """(folded words in reading order, their line indices)."""
    words: list[str] = []
    line_of: list[int] = []
    for li, line in enumerate(reading.get("line_text_full") or []):
        for w in locate._norm_words(line or ""):
            words.append(w)
            line_of.append(li)
    return words, line_of


def crop_word_lines(
    page_lines: list[Line], scan_lines: list, crop: fitz.Rect,
    prefer_layer: bool = True,
) -> tuple[list[list[fitz.Rect]], str]:
    """Detected word rectangles inside `crop`, grouped into printed lines
    (top-to-bottom) with each line in RTL reading order.

    Clean-Unicode pages use the PDF's own word rects (already fragment-merged
    and RTL-sorted by page_reader_lines); image-only scans — and CIPHER text
    layers — fall back to locate._scan_page_lines, whose binarization is reused,
    never reimplemented. A word belongs to the crop when at least half its own
    area is inside.

    `prefer_layer` is the caller's `layer_geometry_usable` verdict for this page
    and it is not optional in spirit: a glyph-cipher layer (haaji-agha
    substitutes literal '#'/'&' for Persian glyphs) yields word rects that
    TOKENIZE DIFFERENTLY from the printed words — measured, a line a reader
    transcribes as 23 words came back as 16 rects, and another page read
    [10, 9, 9] against [1, 5, 4] rects. Those rects are not the printed words,
    so aligning a reading onto them mis-attributes the covered run
    (haaji-agha:p50:h0 read out three words for a rectangle covering one).
    Emptiness is NOT the right probe for "can the layer stand in for the
    detector": a cipher layer is full, it is just wrong.
    """
    lines = (
        [[r for r, _w in line if overlap_frac(r, crop) >= 0.5] for line in page_lines]
        if prefer_layer
        else []
    )
    lines = [ln for ln in lines if ln]
    if lines:
        return lines, "text_layer"
    out: list[list[fitz.Rect]] = []
    for sl in scan_lines:
        kept = [r for r in sl.words if overlap_frac(r, crop) >= 0.5]
        if kept:
            out.append(sorted(kept, key=lambda r: -r.x1))
    out.sort(key=lambda ln: min(r.y0 for r in ln))
    return out, ("scan" if out else "none")


def counts_close(n_read: int, n_det: int) -> bool:
    """Is a per-line word-count difference small enough for the monotone
    alignment below to be trustworthy?

    MEASURED on the 23-case calibration sample (bachehaye_ghali + haaji-agha):
    the LINE counts agree on 22/23 crops, but the per-line WORD counts agree
    exactly on only 1/23 — typical readings are `[13, 17, 15]` against detected
    `[15, 14, 16]`. Word segmentation on a scan splits or merges a couple of
    tokens per line; that is normal, not a failure. Demanding equality (the
    original gate) therefore reported geom_confident=False on 100% of cases,
    which made the flag useless as a health metric. Within ~20% or +-2 the
    alignment still lands every word on the right ink.
    """
    return abs(n_read - n_det) <= max(2, int(0.2 * max(n_read, n_det)))


def align_line_spans(
    words: list[str], rects: list[fitz.Rect]
) -> list[tuple[float, float]]:
    """Monotone alignment of one printed line's reader words onto its detected
    word rectangles. Returns each word's (x0, x1) span in page coordinates.

    Both sequences are in RTL reading order, so the correspondence is monotone
    even when the counts differ — a merged or split token shifts the pairing
    locally, it never reorders it. Each detected rect is assigned to the reader
    word whose cumulative CHARACTER range contains the rect's cumulative INK
    range midpoint (character count is the best available proxy for printed
    width, and matching ink-to-ink keeps inter-word gaps out of both scales).
    The pointer only ever advances, which is what makes the mapping monotone.

    Words that no rect lands on (the reader split a word the detector merged)
    are interpolated char-proportionally into the RTL gap between their
    neighbours' assigned edges, so they still get a plausible, non-overlapping
    span. The superseded char-proportional placement snapped each word out to
    the union of every detected rect it grazed, which produced heavily
    overlapping, wildly oversized word rects.
    """
    if not words or not rects:
        return []
    if len(words) == len(rects):
        return [(r.x0, r.x1) for r in rects]

    env = locate._union_rects(rects)
    widths = [max(r.width, 1e-6) for r in rects]
    total_w = sum(widths)
    mids: list[float] = []
    cum_w = 0.0
    for w in widths:
        mids.append((cum_w + w / 2.0) / total_w)
        cum_w += w

    lens = [max(1, len(w)) for w in words]
    total_c = sum(lens)
    ends: list[float] = []
    cum_c = 0
    for length in lens:
        cum_c += length
        ends.append(cum_c / total_c)

    assigned: list[list[fitz.Rect]] = [[] for _ in words]
    i = 0
    for j, mid in enumerate(mids):
        while i < len(words) - 1 and ends[i] <= mid:
            i += 1
        assigned[i].append(rects[j])

    spans: list[Optional[tuple[float, float]]] = [None] * len(words)
    for k, group in enumerate(assigned):
        if group:
            u = locate._union_rects(group)
            spans[k] = (u.x0, u.x1)

    k = 0
    while k < len(words):
        if spans[k] is not None:
            k += 1
            continue
        end = k
        while end < len(words) and spans[end] is None:
            end += 1
        # RTL: reading order runs from high x to low x, so the previous word's
        # x0 is this run's right edge and the next word's x1 is its left edge.
        right = spans[k - 1][0] if k > 0 else env.x1  # type: ignore[index]
        left = spans[end][1] if end < len(words) else env.x0  # type: ignore[index]
        if right <= left:
            right = left = (right + left) / 2.0
        run = list(range(k, end))
        tot = sum(lens[t] for t in run) or 1
        width = right - left
        cum = 0
        for t in run:
            u0 = cum / tot
            cum += lens[t]
            u1 = cum / tot
            spans[t] = (right - u1 * width, right - u0 * width)
        k = end
    return [s for s in spans if s is not None]


def align_line_sequences(
    vwords: list[list[str]], det_lines: list[list[fitz.Rect]]
) -> Optional[list[Optional[int]]]:
    """Monotone alignment of READING lines onto DETECTED ink lines.

    Returns one detected-line index (or None) per reading line, or None when
    the two sequences are too dissimilar to align at all.

    WHY THIS EXISTS. The old code required ``len(vwords) == len(det_lines)``
    and otherwise spread the whole reading char-proportionally across every
    rect. Over a 3-line crop that fallback was survivable; over a WHOLE PAGE it
    is not — one undetected line (measured: the reading has a page number or a
    footnote rule the binarizer merges, so 31 reading lines meet 30 detected
    ones on most bachehaye_ghali pages) threw away the per-line correspondence
    for all ~900 words at once. Page-scoped truth therefore scored 0.20 against
    the crop route's 0.366 on identical boxes, with geom_confident=0.0: a
    defect in the measurement, not in the locator.

    Both sequences are strictly top-to-bottom, so the correspondence is
    monotone and a Needleman-Wunsch pass recovers it. The similarity of a
    candidate pair is how close their word counts are, plus how close their
    relative positions down the page are; a skip on either side costs a fixed
    gap penalty. n is ~30 per side, so the O(n*m) table is free.
    """
    n, m = len(vwords), len(det_lines)
    if not n or not m:
        return None
    gap = -0.6

    def sim(i: int, j: int) -> float:
        a, b = len(vwords[i]), len(det_lines[j])
        if not a and not b:
            count = 1.0
        else:
            count = 1.0 - abs(a - b) / float(max(a, b, 1))
        pos = 1.0 - abs(i / max(n - 1, 1) - j / max(m - 1, 1))
        return 1.4 * count + 0.6 * pos - 1.0

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(
                score[i - 1][j - 1] + sim(i - 1, j - 1),
                score[i - 1][j] + gap,
                score[i][j - 1] + gap,
            )

    out: list[Optional[int]] = [None] * n
    i, j = n, m
    n_matched = 0
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j - 1] + sim(i - 1, j - 1):
            out[i - 1] = j - 1
            n_matched += 1
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    # Too few lines paired means these are not the same page's lines at all;
    # let the caller keep its unconfident char-proportional fallback.
    if n_matched < 0.6 * min(n, m):
        return None
    return out


class Alignment(NamedTuple):
    """Result of mapping a reading onto detected word rects.

    Deliberately keeps the three trust signals APART rather than collapsing
    them into one boolean, because they mean different things and only one of
    them is evidence of the mis-grading this exists to catch:

    * ``word_conf`` — per word, local. Its line's word count is close AND the
      word is not downstream of a pairing discontinuity.
    * ``structural`` — page-wide and genuinely page-wide: the two line
      sequences could be aligned at all, and few enough reading lines went
      unpaired. NOT "every line's word count was close": at page scale that
      global reduction measures page size, not alignment quality (CLAUDE.md).
    * ``drift_start`` — flat word index of the first drift-tainted word, or
      None. This is the discriminator for the p64 class of failure, and the
      only signal `geom_unconfident` gates on.
    """

    rects: list[Optional[fitz.Rect]]
    confident: bool
    word_conf: list[bool]
    drift_lines: int
    structural: bool
    drift_start: Optional[int]


def assign_word_rects(
    vlm_lines: list[str], det_lines: list[list[fitz.Rect]]
) -> "Alignment":
    """Map every word of the reading (in `flatten`'s exact index order) onto a
    detected word rectangle.

    Same discipline as locate._align_strip: when the LINE counts agree each
    line is aligned monotonically by `align_line_spans` (exact word-count
    equality is NOT required — see `counts_close`), and every word takes the
    full y-band of its printed line, which is the band the boxes themselves are
    drawn on. When the line counts disagree there is no per-line
    correspondence, so the reading is spread char-proportionally over the whole
    crop and confidence is cleared.

    `confident` means "this alignment is trustworthy": the line counts agree
    AND every line's word-count delta is small. It does NOT mean the counts
    were exactly equal.

    DRIFT. The pairing is monotone but not necessarily contiguous: when the
    detector finds fewer ink lines than the reading has (a fused pair — see
    locate._split_tall_runs), the aligner balances the books by dropping a
    reading line in the MIDDLE of the page, and every line after the drop is
    then mapped one printed line off. Measured on bachehaye_ghali p64 that
    silently graded three correctly-placed boxes as catastrophically wrong,
    with every per-line word count still "close" so `confident` stayed True.
    So any discontinuity in the pairing (`dj is None`, or a jump on either
    side) LATCHES a drift flag, and every word from that point on is marked
    unconfident. It latches forward only, positionally: a drop on the last line
    (the page number, routinely absent from the Markdown) must not blank out
    the words above it — measured, that trailing drop is the COMMON case on
    bachehaye_ghali (124 of 136 sampled cases), so a page-wide latch would
    have excluded the whole book. `drift_lines` / `drift_start` are reported so
    the report can discriminate "was mis-graded, now correct" from a genuine
    change.
    """
    vwords = [locate._norm_words(t or "") for t in vlm_lines]
    flat_n = sum(len(ws) for ws in vwords)
    if flat_n == 0 or not det_lines:
        return Alignment(
            [None] * flat_n, False, [False] * flat_n, 0, False, 0 if flat_n else None
        )

    rects: list[Optional[fitz.Rect]] = []
    # Per-word alignment trust, so a caller can ask about the words it actually
    # used instead of the whole page (see `cover`).
    word_conf: list[bool] = []
    # Equal counts are the easy case; unequal ones are aligned monotonically
    # rather than abandoned (see align_line_sequences).
    pairing: Optional[list[Optional[int]]]
    if len(vwords) == len(det_lines):
        pairing = list(range(len(vwords)))
    else:
        pairing = align_line_sequences(vwords, det_lines)
    if pairing is not None:
        confident = True
        n_unpaired = 0
        drifted = False
        drift_lines = 0
        drift_start: Optional[int] = None
        dj_prev: Optional[int] = None
        for li, ws in enumerate(vwords):
            dj = pairing[li]
            was_drifted = drifted
            if dj is not None and dj_prev is not None and dj != dj_prev + 1:
                drifted = True  # a detected line no reading line claims
            if dj is None:
                drifted = True
            if drifted:
                drift_lines += 1
                if not was_drifted:
                    drift_start = len(word_conf)
            if dj is not None:
                dj_prev = dj
            if not ws:
                continue
            if dj is None:
                # A line the detector never found: its words get no geometry,
                # which keeps them out of the covered run without disturbing the
                # index alignment of every other line.
                n_unpaired += 1
                rects.extend([None] * len(ws))
                word_conf.extend([False] * len(ws))
                continue
            dl = det_lines[dj]
            counts_ok = counts_close(len(ws), len(dl))
            # Page-level confidence keeps its old, count-based meaning; drift
            # is applied POSITIONALLY through word_conf so a discontinuity at
            # the foot of the page cannot invalidate the lines above it.
            if not counts_ok:
                confident = False
            line_ok = counts_ok and not drifted
            env = locate._union_rects(dl)
            spans = align_line_spans(ws, dl)
            if len(spans) != len(ws):  # hard fallback: keep index alignment
                confident = False
                rects.extend([None] * len(ws))
                word_conf.extend([False] * len(ws))
                continue
            rects.extend(fitz.Rect(x0, env.y0, x1, env.y1) for x0, x1 in spans)
            word_conf.extend([line_ok] * len(ws))
        structural = n_unpaired <= 0.15 * max(len(vwords), 1)
        if not structural:
            confident = False
        return Alignment(
            rects, confident, word_conf, drift_lines, structural, drift_start
        )

    # Line counts disagree: no per-line correspondence exists, so spread the
    # reading char-proportionally over every detected rect in reading order,
    # width-weighted (locate._align_strip's own !counts_agree branch).
    flat_det = [r for ln in det_lines for r in ln]
    widths = [r.width for r in flat_det]
    total_w = sum(widths) or 1.0
    starts: list[float] = []
    cum_w = 0.0
    for w in widths:
        starts.append(cum_w / total_w)
        cum_w += w
    ends = starts[1:] + [1.0]
    flat_words = [w for ws in vwords for w in ws]
    total_c = sum(len(w) for w in flat_words) or 1
    cum_c = 0
    for w in flat_words:
        u0, u1 = cum_c / total_c, (cum_c + len(w)) / total_c
        cum_c += len(w)
        hit = [flat_det[i] for i in range(len(flat_det)) if ends[i] > u0 and starts[i] < u1]
        rects.append(locate._union_rects(hit) if hit else None)
    # No per-line correspondence at all: the whole page is drift, by definition.
    return Alignment(
        rects, False, [False] * len(rects), len(vwords), False, 0
    )


# ---------------------------------------------------------------------------
# which words does the rectangle cover?
# ---------------------------------------------------------------------------


class Coverage(NamedTuple):
    """Which flat reading-word indices a box covers, and how much to trust it.

    ``has_geometry`` False means no word rectangle could be detected at all —
    the caller has nothing to reduce over and must fall back (the harness keeps
    the model's own claim; the live scorer emits no score).
    """

    inside: list[int]
    rects: list[Optional[fitz.Rect]]
    alignment: Alignment
    source: str
    has_geometry: bool


def cover(
    page: fitz.Page,
    box: dict,
    crop: fitz.Rect,
    reading_lines: list[str],
    page_lines: list[Line],
    scan_lines: list,
    prefer_layer: bool = True,
    claim: str = "",
    _retry: bool = False,
) -> Coverage:
    """The covered run: flat reading-word indices whose printed span the box
    covers.

    A word is inside the box when the box covers >=50% of the word's HORIZONTAL
    span along its own printed line (see `_inside`). Segmented scan_vlm boxes
    union across their segments.

    `claim` is the reading's own pre-existing covered-text claim, used only by
    the empty-covered-set retry below to decide whether the alternative
    geometry source's (geometry-less) answer is worth preferring.
    """
    det_lines, src = crop_word_lines(page_lines, scan_lines, crop, prefer_layer)
    al = assign_word_rects(reading_lines, det_lines)
    rects = al.rects
    if src == "none" or not any(r is not None for r in rects):
        return Coverage([], [], al, src, False)

    brects = box_rects(page, box)

    def _inside(r: fitz.Rect) -> bool:
        """1-D horizontal coverage along the word's own line.

        These runs are single-line and a box spans a horizontal SUB-RANGE of a
        line, so the axis that carries the information is x. The old test asked
        for >=50% of the word rect's 2-D AREA, which on a proportionally-placed
        rect let a 2-word box read out 5 words (bachehaye_ghali:p26:h0). Here y
        is only a gate — "is this box on this word's line at all" — and the
        verdict rests on how much of the word's x-span the box actually covers.
        """
        span = r.width
        segs: list[tuple[float, float]] = []
        for br in brects:
            oy = min(r.y1, br.y1) - max(r.y0, br.y0)
            if oy <= 0 or oy < 0.5 * min(r.height, br.height):
                continue  # a different printed line
            if span <= 0:
                # A word the alignment could only collapse to a point (the
                # reader split a token at a line edge with no gap left to place
                # it in). Dropping it silently punches a hole in the middle of
                # an otherwise contiguous run, which grading reports as
                # judge_unusable; decide it by containment instead.
                if br.x0 <= r.x0 <= br.x1:
                    return True
                continue
            x0, x1 = max(r.x0, br.x0), min(r.x1, br.x1)
            if x1 > x0:
                segs.append((x0, x1))
        if span <= 0:
            return False
        if not segs:
            return False
        segs.sort()
        covered = 0.0
        cur0, cur1 = segs[0]
        for a, b in segs[1:]:
            if a > cur1:
                covered += cur1 - cur0
                cur0, cur1 = a, b
            else:
                cur1 = max(cur1, b)
        covered += cur1 - cur0
        return (covered / span) >= 0.5

    inside = [i for i, r in enumerate(rects) if r is not None and _inside(r)]
    if not inside and not _retry:
        # A box drawn on printed text covers SOMETHING. An empty covered set is
        # far more often a geometry-SOURCE error than a real "covers nothing":
        # measured, all 10 `no_covered_words` cases in the frozen sample were
        # correctly placed boxes whose page had been routed to the wrong source
        # (pixel-verified on boof-e-koor p14/p20/p56). Try the other source
        # before declaring the box empty; keep `no_covered_words` for the case
        # where BOTH sources come up empty, which is a real failure.
        alt = cover(
            page, box, crop, reading_lines, page_lines, scan_lines,
            not prefer_layer, claim, _retry=True,
        )
        # Mirrors the superseded `if alt[0].get("boxed_text")` test: a
        # geometry-less alt still "answers" with the reading's own claim.
        if alt.inside or (not alt.has_geometry and claim):
            return alt
    return Coverage(inside, rects, al, src, True)


def coverage_confidence(cov: Coverage) -> tuple[bool, bool]:
    """(confident, drift_tainted) for a Coverage.

    Confidence is about the words this case actually rests on, not the whole
    image. Over a 3-line crop those were nearly the same thing; over a whole
    page, demanding that all ~31 lines align well reported geom_confident on
    7.7% of cases where the crop route reported 66% — measuring page size, not
    alignment quality. Reduce over the covered run when there is one.

    AND, never REPLACE — but only with the STRUCTURAL page-level signal. The
    local reduction used to overwrite the page verdict outright, so a box
    resting on drift-tainted words could report geom_confident=True purely
    because its own line's word counts happened to be close, which is exactly
    how the p64 mis-grading passed unflagged. Restoring the FULL page-level AND
    was measured and rejected: it flags 76.2% of the frozen sample, because
    over ~30 lines some line's word count is always off — the page size effect
    CLAUDE.md already documents. `structural` is the part of the page verdict
    that is genuinely page-wide.
    """
    al = cov.alignment
    confident = al.confident
    if cov.inside and al.word_conf:
        confident = al.structural and all(
            al.word_conf[i] for i in cov.inside if i < len(al.word_conf)
        )
    drift_tainted = al.drift_start is not None and any(
        i >= al.drift_start for i in cov.inside
    )
    return bool(confident), bool(drift_tainted)


# ---------------------------------------------------------------------------
# the metric (METRIC.md)
# ---------------------------------------------------------------------------


def word_miss_distances(t_set: set[int], b_set: set[int]) -> list[int]:
    """Per-target-word distance to the nearest covered word, in reading-order
    index space (crosses line boundaries by design — METRIC.md).

    0 when the target word itself is covered. `b_set` must be non-empty;
    callers route the empty case through the degenerate-case policy instead,
    since "distance to the nearest covered word" is undefined with no covered
    words and must not silently become 0 or infinity here.
    """
    bs = sorted(b_set)
    out: list[int] = []
    for t in sorted(t_set):
        if t in b_set:
            out.append(0)
            continue
        i = bisect.bisect_left(bs, t)
        best = None
        if i < len(bs):
            best = bs[i] - t
        if i > 0:
            cand = t - bs[i - 1]
            if best is None or cand < best:
                best = cand
        out.append(best)
    return out


# ---------------------------------------------------------------------------
# the live scorer
# ---------------------------------------------------------------------------


class PlacementScore(NamedTuple):
    """A live placement measurement for one box.

    `score` is the word-miss SUM over the target words — 0 is perfect. It is a
    PLACEMENT check against the page's own Markdown, not an accuracy or
    confidence check: see this module's docstring.
    """

    score: int
    mean: float
    worst: int
    target_words: int
    covered_words: int
    confident: bool
    source: str        # where the WORD GEOMETRY came from: text_layer / scan
    reference: str     # what the target was resolved against: text_layer / markdown


def _md_lines(page_md: str) -> list[str]:
    """The Markdown's own lines, kept 1:1 with their character offsets.

    Blank lines are preserved as empty strings so `flatten`'s word indices stay
    in the Markdown's reading order and the char-offset -> word-index map below
    stays exact; `assign_word_rects` skips empty reading lines anyway.
    """
    return page_md.split("\n")


def _word_index_at(page_md: str, char_pos: int) -> int:
    """Number of folded words strictly before `char_pos` in `page_md`."""
    return len(locate._norm_words(page_md[:char_pos]))


def _query_variants(query: locate.Query) -> list[list[str]]:
    out = [locate._norm_words(query.text)]
    out.extend(locate._norm_words(alt) for alt in query.alts)
    return [v for v in out if v]


def _layer_target(
    page_lines: list[Line], query: locate.Query
) -> Optional[set[int]]:
    """Target word indices inside a PDF-word-line reading.

    The reading's words are the PRINTED words, so the query has to be found in
    them by content — `locate._best_window` is the locator's own pure window
    scorer, reused rather than reimplemented, and the same `_MATCH_ACCEPT`
    threshold Tier A trusts gates the answer.
    """
    variants = _query_variants(query)
    flat = [w for line in page_lines for _r, w in line]
    if not flat or not variants:
        return None
    score, start, length = locate._best_window(variants, flat)
    if start < 0 or length <= 0 or score < locate._MATCH_ACCEPT:
        return None
    return set(range(start, start + length))


def _target_indices(page_md: str, query: locate.Query) -> Optional[set[int]]:
    """Reading-order word indices the box is supposed to cover.

    Preferred route is `locate._resolve_span`, the exact same span resolution
    the locator itself used, so the target is definitionally "where the Markdown
    says this phrase is". When the query carries no span and the text cannot be
    found verbatim, fall back to the locator's own fuzzy window search over the
    folded word sequence.
    """
    span = locate._resolve_span(page_md, query)
    if span is not None:
        a, b = span
        start = _word_index_at(page_md, a)
        length = len(locate._norm_words(page_md[a:b]))
        if length > 0:
            return set(range(start, start + length))
    words = locate._norm_words(page_md)
    variants = [locate._norm_words(query.text)]
    variants.extend(locate._norm_words(alt) for alt in query.alts)
    variants = [v for v in variants if v]
    if not words or not variants:
        return None
    score, start, length = locate._best_window(variants, words)
    if start < 0 or length <= 0 or score < locate._MATCH_ACCEPT:
        return None
    return set(range(start, start + length))


def score_box(
    page: fitz.Page,
    box: Optional[dict],
    page_md: str,
    query: locate.Query,
    *,
    page_lines: Optional[list[Line]] = None,
    scan_lines: Optional[list] = None,
    prefer_layer: Optional[bool] = None,
) -> Optional[PlacementScore]:
    """Live placement score for one located box. Never raises; None means
    "cannot be measured here" and the caller must show no score.

    The reference is the page's own Markdown (`text/NNNN.md`), which every
    reviewed book already has on disk — so this costs no API call and works on
    a book that has never been near the offline harness. The price is that a
    wrong transcription makes the score wrong with it; this measures PLACEMENT,
    not accuracy.

    `page_lines` / `scan_lines` / `prefer_layer` are the per-page work
    (`page_reader_lines`, `locate._scan_page_lines`, `layer_geometry_usable`);
    pass them in when scoring several boxes on one page — `locate._scan_page_lines`
    rasterizes, so calling it per box would be far too slow for the render path.
    """
    try:
        if box is None or not page_md:
            return None
        if prefer_layer is None:
            prefer_layer = layer_geometry_usable(page)
        if page_lines is None:
            page_lines = page_reader_lines(page, page_md)[0] if prefer_layer else []
        if scan_lines is None:
            scan_lines = locate._scan_page_lines(page)

        # REFERENCE CHOICE, and the measured reason for it.
        #
        # The straightforward reading of "is the box where the Markdown says
        # this phrase is" is to use the Markdown's own lines as the reading.
        # Measured, that is nearly useless: a transcription reflows printed
        # lines into PARAGRAPHS, so the Markdown has ~5 lines where the page
        # has ~30, `align_line_sequences` cannot pair them, and the alignment
        # degrades to the char-proportional fallback that clears confidence.
        # On Hossein_shenasi 10 of 11 corrections and on boof-e-koor 101 of 110
        # came back unscored.
        #
        # When the page has a usable text layer its OWN word lines are real
        # printed lines, so they pair with the detected ink lines exactly. The
        # query is then found in them by content (`_layer_target`), which is a
        # STRONGER check than the Markdown route — it is independent of the
        # transcription rather than merely independent of the locator.
        # The Markdown route stays as the fallback for pages with no usable
        # layer; on an image-only scan it will almost always come back
        # unconfident, which is the honest answer: verifying a scan's placement
        # needs someone to read the ink, and this scorer never bills for that.
        reading_lines: list[str]
        t_set: Optional[set[int]]
        if prefer_layer and page_lines:
            reference = "text_layer"
            reading_lines = [" ".join(w for _r, w in line) for line in page_lines]
            t_set = _layer_target(page_lines, query)
        else:
            reference = "markdown"
            reading_lines = _md_lines(page_md)
            t_set = _target_indices(page_md, query)
        if not t_set:
            return None
        cov = cover(
            page,
            box,
            fitz.Rect(page.rect),
            reading_lines,
            page_lines,
            scan_lines,
            prefer_layer,
        )
        if not cov.has_geometry or not cov.inside:
            return None
        confident, drift_tainted = coverage_confidence(cov)
        distances = word_miss_distances(t_set, set(cov.inside))
        if not distances:
            return None
        return PlacementScore(
            score=int(sum(distances)),
            mean=round(sum(distances) / len(distances), 3),
            worst=int(max(distances)),
            target_words=len(t_set),
            covered_words=len(cov.inside),
            confident=bool(confident and not drift_tainted),
            source=cov.source,
            reference=reference,
        )
    except Exception:
        return None
