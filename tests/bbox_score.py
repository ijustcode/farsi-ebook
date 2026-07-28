"""Automatic accuracy scorer for the review UI's bounding boxes.

Design principle: **the VLM transcribes, Python grades.** The judge model is
never asked "is this box right?". It is asked to READ a crop of the page — its
printed lines, verbatim — and this module then decides the verdict
deterministically by aligning the query against that reading and comparing word
indices. Books with a clean Unicode text layer skip the model entirely:
PyMuPDF's own word rectangles play the reader's part, emitting the identical
reading shape so the grading code is shared byte-for-byte between the routes.

The model is also asked which words the magenta rectangle covers, but that
answer is NOT what the verdict rests on — deciding which glyphs fall inside a
rectangle is a geometric judgement, so Python does it (`_geometric_boxed`):
the reader's words are aligned monotonically onto the detected word rectangles
of their printed line, and the words whose HORIZONTAL span the box covers by
>=50% ARE the boxed run.
The model's claim is retained per case as `boxed_text_vlm` and surfaced as the
`boxed_readout_disagreement` health metric. See the block comment above
`_geometric_boxed` for the measured failures that motivated this.

Cost: the only billed artifact is the model's reading of a crop image, and it
is cached on the crop rectangle alone (`_ReadCache`), never on the box or the
grading rules. Re-scoring after a grading change, a query change or a box nudge
that does not move the crop is therefore free.

RTL SIGN CONVENTION (enforced everywhere in this file)
------------------------------------------------------
``shift_words`` is measured in **reading-order index space**, never in screen
space. It is ``box_start - truth_start`` over the flattened reading-order word
list of the crop. A **positive** value means the box must move **later in
reading order** to be correct; a negative value means it must move **earlier**.
The words "left" and "right" never appear in a report field, an HTML caption,
or a prompt — on an RTL page they are ambiguous and actively misleading.

Cases are produced from the real production path
(``review._page_box_inputs`` -> ``review._box_specs`` ->
``review._locate_queries_cached`` -> ``review._ScanBoxRefiner.refine`` ->
``review._build_boxes``); nothing here reimplements the locator.

Usage
-----
    ./venv/bin/python tests/bbox_score.py cases --replay all \\
        --out tests/data/bbox_cases.json
    ./venv/bin/python tests/bbox_score.py score --cases tests/data/bbox_cases.json \\
        --judge deterministic --out out/bbox_score.json --html out/bbox_score.html
    ./venv/bin/python tests/bbox_score.py compare --baseline A.json --candidate B.json --gate
    ./venv/bin/python tests/bbox_score.py golden --from out/bbox_score.json \\
        --out tests/data/bbox_golden.json

    ./venv/bin/python tests/bbox_score.py crops --cases tests/data/bbox_cases.json \\
        --sample 240 --out-dir out/judgebatch      # PNGs + BATCHES/ work orders
    # ... a Claude Code subagent reads each batch and writes readings JSON ...
    ./venv/bin/python tests/bbox_score.py ingest-readings \\
        --manifest out/judgebatch/manifest.json --readings out/judgebatch/readings/
    ./venv/bin/python tests/bbox_score.py score --judge cached ...

``--judge deterministic`` never touches the network. ``--judge auto`` routes per
page: text-layer books grade deterministically, cipher/scan books call the VLM.
``--judge cached`` routes the same way but takes every VLM reading from the
cache — see the ``crops``/``ingest-readings`` block below for why.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html as html_mod
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farsi2epub import locate, review  # noqa: E402


def _llm():
    """Import farsi2epub.llm (and the anthropic SDK) only when an API path is
    actually taken. --judge cached / --judge deterministic never call the API,
    and importing the SDK costs minutes of wall clock when the filesystem is
    under pressure, so the scorer must not pay it just to grade from cache."""
    from farsi2epub import llm as _m

    return _m
from farsi2epub.config import MODEL_STRONG  # noqa: E402
from farsi2epub.workspace import (  # noqa: E402
    DEFAULT_BOOKS_ROOT,
    Workspace,
    parse_pages_spec,
)

import bbox_eval  # noqa: E402  (same directory; only defines helpers at import)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Bump when _judge_crop's geometry/marking changes: it is part of the READING
# cache key, so every cached reading taken with the old crop is invalidated.
_CROP_GEOMETRY_VERSION = 1
# Bump when the grading semantics change. Since the reading cache was split out
# (see _ReadCache) this constant no longer gates any billed result: grades are
# recomputed for free from cached readings on every run, so a grading change
# costs nothing. It is still recorded in the report for provenance.
# 2 = boxed run derived geometrically instead of from the model's boxed_text.
# 3 = monotone within-line alignment + 1-D horizontal box coverage.
_SCORE_CACHE_VERSION = 3
# Bump when the cached reading's value shape changes.
_READ_CACHE_VERSION = 1

# Marking colour: pure magenta, a hue no book ink has.
_MARK_RGB = (0xFF, 0x00, 0xE5)
_MARK_PX = 3

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
# "Merged" is the fragment-merged reader tokenization (_page_reader_lines);
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
# is kept as a weak sanity floor. Cost consequence, accepted deliberately:
# boof-e-koor's ~155 cases now bill the VLM (~$0.80 at the measured
# $0.0053/crop). Accuracy of the metric wins over its price.
#
# Do not lower these without re-running the agreement measurement above.
_LAYER_RAW_RECALL_MIN = 0.45
_LAYER_RECALL_MIN = 0.70

TIERS = ["match", "layout", "scan", "scan_vlm", "model", "none"]
KINDS = ["hunk", "issue", "finding-edit"]

# Worst-first ordering for the HTML sheet and for "most severe" reporting.
VERDICT_RANK = {
    "wrong_region": 0,
    "wrong_line": 1,
    "no_box": 2,
    "shifted": 3,
    "partial": 4,
    "phrase_absent": 5,
    "judge_unusable": 6,
    "unmeasurable": 7,
    "exact": 8,
}
# Verdicts that carry no information about the locator and are therefore kept
# OUT of every denominator (they are still counted and reported).
_EXCLUDED_VERDICTS = ("stale", "uncached", "unmeasurable", "judge_failed")
VERDICTS = list(VERDICT_RANK)

_MAX_CROPS_PER_CALL = 6
_WORKERS = 4

# Measured price of one crop reading through llm.read_box_crops (Sonnet, the
# batched 6-per-call path). Used only to report the API spend a subagent batch
# avoids — never to bill anything.
_CROP_READ_USD = 0.00633
# Crops per subagent work order. Sized so one agent turn can hold the images.
_SUBAGENT_BATCH = 20

# ---------------------------------------------------------------------------
# PAGE-LEVEL TRUTH (the answer key)
# ---------------------------------------------------------------------------
#
# WHY THIS REPLACES THE CROP-KEYED READING CACHE. `_read_key` is keyed on the
# crop rectangle, and `_crop_clip` derives that rectangle from THE BOX UNDER
# TEST. So the ground truth was a function of the artifact being measured: any
# structural change to locate.py moved the boxes, moved the crops, missed the
# cache, and turned those cases into `uncached` — silently dropped from every
# denominator. Iteration 2 lost 125 of 240 cases that way, and the surviving
# 111 were biased towards the cases the change did not touch, which is exactly
# the wrong denominator.
#
# A page's printed words do not move when the locator changes. So the reading
# is taken PER PAGE, once, and frozen: `_page_read_key` is keyed on
# (slug, page, tile, render geometry, model) and mentions no box at all.
# Grading then re-derives the truth window and the boxed run from that frozen
# reading for free, on every run, for any box any future locator produces.
# `uncached` becomes structurally impossible for a locator change; only a page
# that has never been read at all can be missing.
#
# MEASURED CHOICE OF WHOLE PAGE OVER TILES. The reader's acuity limit is the
# vision long-edge ceiling (2576 px). Rendered whole-page at that ceiling, all
# three VLM-routed books are comfortably legible by inspection (bachehaye_ghali
# 577x841pt -> 3.06x, haaji-agha 595x842pt -> 3.06x, boof-e-koor 420x595pt ->
# 4.33x). One image per page is ~4x cheaper to acquire than the 3-tile
# alternative, so whole-page is the default and tiles are the ESCALATION, not
# the norm: `ingest-pages` scores every reading against the page's own
# independent transcription (`text/NNNN.md`) and any page below
# `_PAGE_RECALL_MIN` is reported for re-reading at tile resolution. The answer
# key therefore verifies itself on acquisition instead of being trusted.
_PAGE_GEOMETRY_VERSION = 1
_PAGE_READ_CACHE_VERSION = 1
# Vision long-edge ceiling; also the render target for a whole-page image.
_PAGE_LONG_EDGE = 2576.0
# Fraction of the page transcription's words a reading must recall to be
# trusted as truth. Deliberately not 1.0: the reference is itself a VLM
# transcription with its own errors, and markdown carries words (headings,
# footnote bodies) whose print the reader legitimately orders differently.
_PAGE_RECALL_MIN = 0.85
# Pages per subagent work order. Smaller than _SUBAGENT_BATCH because a
# whole-page image is far larger than a crop.
_PAGE_SUBAGENT_BATCH = 8
# Fraction of the sample that may go ungraded before `score` refuses to write a
# report. 0.10 is well under the 52% that made iteration 2's number meaningless
# and well over the handful of genuinely unmeasurable cases a healthy run has.
_MAX_EXCLUDED_FRACTION = 0.10


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _git_state() -> tuple[str, bool]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                args, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return sha, dirty


def _mean(xs: list[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _p90(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(xs)
    k = max(0, min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1))
    return ordered[k]


# ---------------------------------------------------------------------------
# case replay: production hunk derivation without the "pending" gate
# ---------------------------------------------------------------------------


def _replay_text(ws: Workspace, n: int) -> str:
    """The markdown the QC suggestion was computed against: the pre-edit
    ``.orig.md`` backup when a human already accepted the page, else the
    current markdown."""
    orig = ws.page_orig_path(n)
    if orig.is_file():
        return orig.read_text(encoding="utf-8")
    md = ws.page_md_path(n)
    return md.read_text(encoding="utf-8") if md.is_file() else ""


def _replay_box_inputs(
    ws: Workspace, n: int
) -> tuple[dict, list[dict], list[dict], str, Optional[str]]:
    """Mirror of ``review._page_box_inputs`` that drops the
    ``suggestion_status == "pending"`` gate and replays against the pre-edit
    markdown when one exists. Same 5-tuple shape; the hunk/issue derivation is
    the production helpers themselves, never a copy.

    With the pending gate the corpus is only a few hundred cases on 70 pages
    (haaji-agha contributes 3); replaying every verdict-fail page — including
    ones a human already resolved — recovers the whole measured corpus.
    """
    sidecar = review._read_sidecar(ws, n)
    text = _replay_text(ws, n)

    issues: list[dict] = []
    hunks: list[dict] = []
    panel_kind: Optional[str] = None
    qc_data = sidecar.get("qc")
    if qc_data and qc_data.get("verdict") == "fail":
        issues = [
            {
                "type": i.get("type"),
                "description": i.get("description"),
                "snippet": i.get("snippet"),
                "bbox": i.get("bbox"),
            }
            for i in qc_data.get("issues") or []
        ]
        suggested = qc_data.get("suggested_text_md")
        if suggested is None:
            panel_kind = "no_suggestion"
        elif suggested == text:
            panel_kind = "identical"
        else:
            panel_kind = "hunks"
            hunks = review._derive_hunks(text, suggested)
            review._link_issues_to_hunks(issues, hunks)
        hunks += review._derive_finding_hunks(text, issues, hunks)

    return sidecar, issues, hunks, text, panel_kind


def _page_inputs(ws: Workspace, n: int, replay: str):
    if replay == "all":
        return _replay_box_inputs(ws, n)
    return review._page_box_inputs(ws, n)


# ---------------------------------------------------------------------------
# case features
# ---------------------------------------------------------------------------

_VERSE_RE = re.compile(r"^.*\s---\s.*$", re.MULTILINE)
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^[^\]]{1,10}\]:", re.MULTILINE)


def _page_features(text: str) -> dict:
    return {
        "page_has_verse": bool(_VERSE_RE.search(text)),
        "page_has_footnote_def": bool(_FOOTNOTE_DEF_RE.search(text)),
    }


def _y_bucket(box: Optional[dict]) -> str:
    if not box:
        return "unknown"
    yc = (box["y0"] + box["y1"]) / 2.0
    return "y_top" if yc < 1 / 3 else ("y_mid" if yc < 2 / 3 else "y_bottom")


def _spec_kind(hunk: Optional[dict], issue: Optional[dict]) -> str:
    if hunk is not None:
        return "finding-edit" if hunk.get("edit_only") else "hunk"
    return "issue"


def _spec_issue_type(
    hunk: Optional[dict], issue: Optional[dict], issues: list[dict]
) -> str:
    if issue is not None:
        return issue.get("type") or "unknown"
    if hunk is not None:
        idx = hunk.get("issue_idx")
        if idx is not None and idx < len(issues):
            return issues[idx].get("type") or "unknown"
    return "unlinked"


# ---------------------------------------------------------------------------
# case enumeration
# ---------------------------------------------------------------------------


def _iter_book_pages(ws: Workspace, pages_spec: Optional[str]) -> list[int]:
    done = ws.pages_done()
    if pages_spec:
        page_count = int(ws.meta.get("page_count") or (max(done) if done else 0))
        wanted = set(parse_pages_spec(pages_spec, page_count))
        done = [n for n in done if n in wanted]
    return done


def build_cases(
    books: list[str], pages_spec: Optional[str], replay: str
) -> list[dict]:
    cases: list[dict] = []
    for slug in books:
        ws = Workspace.load(slug)
        for n in _iter_book_pages(ws, pages_spec):
            _sidecar, issues, hunks, text, panel_kind = _page_inputs(ws, n, replay)
            if panel_kind is None or (not issues and not hunks):
                continue
            specs = review._box_specs(issues, hunks, text)
            feats = _page_features(text)
            md_sha = _sha1(text)
            for key, query, _fallback, hunk, issue in specs:
                cases.append(
                    {
                        "id": f"{slug}:p{n}:{key}",
                        "slug": slug,
                        "page": n,
                        "key": key,
                        "md_sha1": md_sha,
                        "replay": replay == "all",
                        "query": {
                            "text": query.text,
                            "span": list(query.span) if query.span else None,
                            "alts": list(query.alts),
                        },
                        "kind": _spec_kind(hunk, issue),
                        "issue_type": _spec_issue_type(hunk, issue, issues),
                        "source_kind": panel_kind,
                        "features": {
                            **feats,
                            "short_query_le14": locate._countable(query.text) <= 14,
                        },
                    }
                )
    return cases


# ---------------------------------------------------------------------------
# judge routing
# ---------------------------------------------------------------------------


def _layer_recall(pwords: list[tuple[fitz.Rect, str]], page_md: str) -> float:
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


def _layer_usable(
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
        _lines, _ratio, recall = _page_reader_lines(page, page_md)
    if raw_recall is None:
        raw_recall = _layer_recall(pwords, page_md)
    return raw_recall >= _LAYER_RAW_RECALL_MIN and recall >= _LAYER_RECALL_MIN


# ---------------------------------------------------------------------------
# crop geometry
# ---------------------------------------------------------------------------


def _median_line_h(page: fitz.Page) -> float:
    """Median printed line height in page units."""
    try:
        scan = locate._scan_page_lines(page)
    except Exception:
        scan = []
    if scan:
        return float(statistics.median([ln.rect.height for ln in scan])) or 1.0
    pdf_lines = locate._pdf_lines(page)
    if pdf_lines:
        return float(statistics.median([r.height for r, _n in pdf_lines])) or 1.0
    return 0.03 * (page.rect.height or 1.0)


def _box_rects(page: fitz.Page, box: dict) -> list[fitz.Rect]:
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


def _crop_clip(page: fitz.Page, box: dict, line_h: float) -> fitz.Rect:
    """The crop rectangle alone, without rasterizing anything.

    Split out of _judge_crop because it is the READING CACHE KEY: a cache
    lookup must be able to identify the crop without paying to render it.
    """
    pr = page.rect
    env = locate._union_rects(_box_rects(page, box))
    pad_x = max(2.0 * max(env.width, 1e-6), 0.25 * pr.width)
    pad_y = 1.6 * line_h
    clip = fitz.Rect(
        max(pr.x0, env.x0 - pad_x),
        max(pr.y0, env.y0 - pad_y),
        min(pr.x1, env.x1 + pad_x),
        min(pr.y1, env.y1 + pad_y),
    )
    if clip.width < 1.0 or clip.height < 1.0:
        clip = fitz.Rect(pr)
    return clip


def _judge_crop(
    page: fitz.Page, box: dict, line_h: float, scale_boost: float = 1.0
) -> tuple[bytes, fitz.Rect]:
    """Marked PNG crop around `box`, plus the crop rect in page coordinates.

    Deliberately wider than bbox_eval._crop_png: the box under test may be
    wrong, so the phrase's true position must still be inside the crop for the
    reader to find it. Horizontal padding is the larger of twice the box width
    and a quarter of the page; vertical padding is about one printed line.
    The mark is a plain magenta OUTLINE (never a fill, never the review UI's
    ripped polygons) so it occludes no glyph the reader must transcribe.
    """
    rects = _box_rects(page, box)
    clip = _crop_clip(page, box, line_h)

    scale = max(2.0, min(4.0, 1900.0 / max(clip.width, 1.0))) * scale_boost
    long_edge = max(clip.width, clip.height) * scale
    if long_edge > 2000.0:
        scale *= 2000.0 / long_edge

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    # A CLIPPED pixmap does not start at (0, 0): pix.set_rect works in the
    # pixmap's own device coordinates, so every rectangle must be shifted by
    # pix.x / pix.y or the call silently returns False and draws nothing.
    ox, oy = pix.x, pix.y

    def _fill(x0: int, y0: int, x1: int, y1: int) -> None:
        r = fitz.IRect(
            ox + max(0, x0),
            oy + max(0, y0),
            ox + min(pix.width, x1),
            oy + min(pix.height, y1),
        )
        if not r.is_empty:
            pix.set_rect(r, _MARK_RGB)

    for rect in rects:
        x0 = int(round((rect.x0 - clip.x0) * scale))
        y0 = int(round((rect.y0 - clip.y0) * scale))
        x1 = int(round((rect.x1 - clip.x0) * scale))
        y1 = int(round((rect.y1 - clip.y0) * scale))
        t = _MARK_PX
        _fill(x0 - t, y0 - t, x1 + t, y0)          # top
        _fill(x0 - t, y1, x1 + t, y1 + t)          # bottom
        _fill(x0 - t, y0, x0, y1)                  # start edge
        _fill(x1, y0, x1 + t, y1)                  # end edge
    return pix.tobytes("png"), clip


# ---------------------------------------------------------------------------
# readers: deterministic (PyMuPDF) and VLM — identical output shape
# ---------------------------------------------------------------------------


def _reading_dict(
    line_text_full: list[str],
    boxed_line_index: int,
    boxed_text: str,
    boxed_spans_lines: int,
    legible: bool,
    note: str,
) -> dict:
    return {
        "line_text_full": list(line_text_full),
        "boxed_line_index": int(boxed_line_index),
        "boxed_text": boxed_text,
        "boxed_spans_lines": int(boxed_spans_lines),
        "legible": bool(legible),
        "note": note,
    }


def _overlap_frac(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = fitz.Rect(a) & b
    area = a.width * a.height
    if area <= 0:
        return 0.0
    if inter.is_empty:
        return 0.0
    return (inter.width * inter.height) / area


# Candidate fragment-merge thresholds, as a fraction of the line height. 0.0
# means "trust the extractor's own tokens"; the rest recover printed words from
# legacy layers that split one word into many glyph-run tokens.
_MERGE_RATIOS = (0.0, 0.015, 0.03, 0.05, 0.08, 0.12)

_Line = list[tuple[fitz.Rect, str]]


def _merge_line(line: _Line, ratio: float) -> _Line:
    """Join neighbouring tokens on one RTL-ordered line whose horizontal gap is
    below `ratio` * the line's median glyph height."""
    if not line or ratio <= 0.0:
        return list(line)
    heights = [r.height for r, _w in line if r.height > 0]
    h = statistics.median(heights) if heights else 1.0
    max_gap = ratio * h
    out: _Line = [line[0]]
    for rect, word in line[1:]:
        prev_rect, prev_word = out[-1]
        if prev_rect.x0 - rect.x1 <= max_gap:
            out[-1] = (prev_rect | rect, prev_word + word)
        else:
            out.append((rect, word))
    return out


def _raw_page_lines(page: fitz.Page) -> list[_Line]:
    """Every folded page word, grouped into printed lines (y0 within +-1.0) and
    sorted by -x1 within a line — the RTL reading-order convention
    ``locate._line_word_extent`` uses."""
    kept = [
        (fitz.Rect(w[0], w[1], w[2], w[3]), locate._fold_word(w[4]))
        for w in page.get_text("words")
    ]
    kept = [(r, f) for r, f in kept if f]
    lines: list[_Line] = []
    for rect, folded in sorted(kept, key=lambda t: (t[0].y0, -t[0].x1)):
        if lines and abs(lines[-1][0][0].y0 - rect.y0) <= 1.0:
            lines[-1].append((rect, folded))
        else:
            lines.append([(rect, folded)])
    return [sorted(line, key=lambda t: -t[0].x1) for line in lines]


def _recall_of(lines: list[_Line], md_counts: Counter) -> float:
    have = Counter(w for line in lines for _r, w in line)
    total = sum(have.values())
    if not total:
        return 0.0
    return sum(min(c, md_counts.get(w, 0)) for w, c in have.items()) / total


def _page_reader_lines(page: fitz.Page, page_md: str) -> tuple[list[_Line], float, float]:
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
    raw = _raw_page_lines(page)
    md_counts = Counter(locate._norm_words(page_md))
    best_lines, best_ratio, best_recall = raw, 0.0, _recall_of(raw, md_counts)
    for ratio in _MERGE_RATIOS[1:]:
        cand = [_merge_line(line, ratio) for line in raw]
        recall = _recall_of(cand, md_counts)
        if recall > best_recall + 1e-9:
            best_lines, best_ratio, best_recall = cand, ratio, recall
    return best_lines, best_ratio, best_recall


def _det_reading(
    page_lines: list[_Line], page: fitz.Page, box: dict, crop: fitz.Rect
) -> tuple[dict, list[fitz.Rect], Optional[int]]:
    """Read `crop` from the PDF's own word rectangles, emitting the same dict
    shape ``llm.read_box_crops`` produces so grading is shared.

    Only words that fold to a non-empty token are kept, so
    ``locate._norm_words`` of an emitted line yields exactly the words whose
    rectangles are returned alongside it.
    Returns (reading, flat word rects, flat index where the boxed run starts).
    """
    lines = [
        [(r, w) for r, w in line if _overlap_frac(r, crop) >= 0.5]
        for line in page_lines
    ]
    lines = [line for line in lines if line]

    line_text: list[str] = []
    flat_rects: list[fitz.Rect] = []
    flat_words: list[str] = []
    flat_line_of: list[int] = []
    for li, line in enumerate(lines):
        # Every kept word folds non-empty, so locate._norm_words of this joined
        # line reproduces exactly `flat_words` for the line, in the same order.
        line_text.append(" ".join(w for _r, w in line))
        for rect, folded in line:
            flat_rects.append(rect)
            flat_words.append(folded)
            flat_line_of.append(li)

    box_rects = _box_rects(page, box)
    inside = [
        i
        for i, rect in enumerate(flat_rects)
        if max(_overlap_frac(rect, br) for br in box_rects) >= 0.5
    ]
    if inside:
        boxed_text = " ".join(flat_words[i] for i in inside)
        boxed_lines = len({flat_line_of[i] for i in inside})
        boxed_line_index = flat_line_of[inside[0]]
        contiguous = inside == list(range(inside[0], inside[0] + len(inside)))
        # A non-contiguous covered set is a real signal (the rectangle straddles
        # unrelated words), so drop the hint and let grading fail the run search.
        start_hint: Optional[int] = inside[0] if contiguous else None
    else:
        boxed_text = ""
        boxed_lines = 0
        boxed_line_index = -1
        start_hint = None

    reading = _reading_dict(
        line_text,
        boxed_line_index,
        boxed_text,
        boxed_lines,
        True,
        "deterministic text-layer read",
    )
    return reading, flat_rects, start_hint


def _subagent_reading(line_text_full: list[str], legible: bool, note: str) -> dict:
    """A reading supplied by a Claude Code subagent instead of the API.

    The subagent does exactly the half of the job the API model is good at and
    that costs money — transcribing the printed lines of a crop. It never
    reports which words the magenta rectangle covers: that is a geometric
    judgement Python already makes from the word rects (`_geometric_boxed`),
    and accepting a claim about it would quietly reintroduce the very failure
    mode the geometric readout was built to remove.

    So the three boxed_* fields get the neutral "unknown readout" values the
    deterministic-none branch of _geometric_boxed already uses. `boxed_text` is
    overwritten from geometry before grading; the empty string that survives
    when no geometry exists is read as "no VLM claim" by _grade_case.
    """
    return _reading_dict(list(line_text_full), -1, "", 0, legible, note)


def _vlm_reading(r) -> Optional[dict]:
    if r is None:
        return None
    return _reading_dict(
        list(r.line_text_full),
        r.boxed_line_index,
        r.boxed_text,
        r.boxed_spans_lines,
        r.legible,
        r.note or "",
    )


# ---------------------------------------------------------------------------
# geometric readout: which words does the rectangle actually cover?
# ---------------------------------------------------------------------------
#
# MEASURED FAILURE THIS REPLACES. The model's `boxed_text` — its claim about
# which words fall inside the drawn rectangle — carries routine +-1 word noise
# and occasional +-3:
#
#   boof-e-koor:p48:h0   rectangle plainly covers 5 words; model said 2
#                        -> bogus "partial, shift_words=3"
#   boof-e-koor:p30:h1   over-read by one word    -> bogus "partial"
#   boof-e-koor:p32:h100002  under-read by one at the start -> bogus "partial"
#
# Deciding which glyphs fall inside a rectangle is a GEOMETRIC judgement, so it
# belongs to Python, exactly like every other grading decision in this file.
# The model keeps the job it is good at (`line_text_full`, verified correct on
# every case inspected by hand); its `boxed_text` is retained only as a
# diagnostic (`boxed_text_vlm`) and as the last-resort fallback when no word
# rectangles can be detected in the crop at all.


def _crop_word_lines(
    page_lines: list[_Line], scan_lines: list, crop: fitz.Rect,
    prefer_layer: bool = True,
) -> tuple[list[list[fitz.Rect]], str]:
    """Detected word rectangles inside `crop`, grouped into printed lines
    (top-to-bottom) with each line in RTL reading order.

    Clean-Unicode pages use the PDF's own word rects (already fragment-merged
    and RTL-sorted by _page_reader_lines); image-only scans — and CIPHER text
    layers — fall back to locate._scan_page_lines, whose binarization is reused,
    never reimplemented. A word belongs to the crop when at least half its own
    area is inside.

    `prefer_layer` is the caller's `_layer_usable` verdict for this page and it
    is not optional in spirit: a glyph-cipher layer (haaji-agha substitutes
    literal '#'/'&' for Persian glyphs) yields word rects that TOKENIZE
    DIFFERENTLY from the printed words — measured, a line a reader transcribes
    as 23 words came back as 16 rects, and another page read [10, 9, 9] against
    [1, 5, 4] rects. Those rects are not the printed words, so aligning a
    reading onto them mis-attributes the boxed run (haaji-agha:p50:h0 read out
    three words for a rectangle covering one). Emptiness is NOT the right probe
    for "can the layer stand in for the detector": a cipher layer is full, it is
    just wrong. Same gate as the judge routing above, by construction.
    """
    lines = (
        [[r for r, _w in line if _overlap_frac(r, crop) >= 0.5] for line in page_lines]
        if prefer_layer
        else []
    )
    lines = [ln for ln in lines if ln]
    if lines:
        return lines, "text_layer"
    out: list[list[fitz.Rect]] = []
    for sl in scan_lines:
        kept = [r for r in sl.words if _overlap_frac(r, crop) >= 0.5]
        if kept:
            out.append(sorted(kept, key=lambda r: -r.x1))
    out.sort(key=lambda ln: min(r.y0 for r in ln))
    return out, ("scan" if out else "none")


def _counts_close(n_read: int, n_det: int) -> bool:
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


def _align_line_spans(
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


def _align_line_sequences(
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


def _assign_word_rects(
    vlm_lines: list[str], det_lines: list[list[fitz.Rect]]
) -> tuple[list[Optional[fitz.Rect]], bool]:
    """Map every word of the VLM reading (in _flatten's exact index order) onto
    a detected word rectangle.

    Same discipline as locate._align_strip: when the LINE counts agree each
    line is aligned monotonically by `_align_line_spans` (exact word-count
    equality is NOT required — see `_counts_close`), and every word takes the
    full y-band of its printed line, which is the band the boxes themselves are
    drawn on. When the line counts disagree there is no per-line
    correspondence, so the reading is spread char-proportionally over the whole
    crop and confidence is cleared.

    `confident` means "this alignment is trustworthy": the line counts agree
    AND every line's word-count delta is small. It does NOT mean the counts
    were exactly equal.
    """
    vwords = [locate._norm_words(t or "") for t in vlm_lines]
    flat_n = sum(len(ws) for ws in vwords)
    if flat_n == 0 or not det_lines:
        return [None] * flat_n, False, [False] * flat_n

    rects: list[Optional[fitz.Rect]] = []
    # Per-word alignment trust, so a caller can ask about the words it actually
    # used instead of the whole page (see _geometric_boxed).
    word_conf: list[bool] = []
    # Equal counts are the easy case; unequal ones are aligned monotonically
    # rather than abandoned (see _align_line_sequences).
    pairing: Optional[list[Optional[int]]]
    if len(vwords) == len(det_lines):
        pairing = list(range(len(vwords)))
    else:
        pairing = _align_line_sequences(vwords, det_lines)
    if pairing is not None:
        confident = True
        n_unpaired = 0
        for li, ws in enumerate(vwords):
            if not ws:
                continue
            dj = pairing[li]
            if dj is None:
                # A line the detector never found: its words get no geometry,
                # which keeps them out of the boxed run without disturbing the
                # index alignment of every other line.
                n_unpaired += 1
                rects.extend([None] * len(ws))
                word_conf.extend([False] * len(ws))
                continue
            dl = det_lines[dj]
            line_ok = _counts_close(len(ws), len(dl))
            if not line_ok:
                confident = False
            env = locate._union_rects(dl)
            spans = _align_line_spans(ws, dl)
            if len(spans) != len(ws):  # hard fallback: keep index alignment
                confident = False
                rects.extend([None] * len(ws))
                word_conf.extend([False] * len(ws))
                continue
            rects.extend(fitz.Rect(x0, env.y0, x1, env.y1) for x0, x1 in spans)
            word_conf.extend([line_ok] * len(ws))
        if n_unpaired > 0.15 * max(len(vwords), 1):
            confident = False
        return rects, confident, word_conf

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
    return rects, False, [False] * len(rects)


def _geometric_boxed(
    page: fitz.Page,
    box: dict,
    crop: fitz.Rect,
    reading: dict,
    page_lines: list[_Line],
    scan_lines: list,
    prefer_layer: bool = True,
) -> tuple[dict, list[Optional[fitz.Rect]], Optional[int], bool, str]:
    """Recompute the reading's boxed run from geometry.

    Returns (reading with a geometric ``boxed_text``/``boxed_line_index``, the
    per-word rects for IoU, the flat start index of the boxed run when it is
    contiguous, geom_confident, rect source). A word is inside the box when the
    box covers >=50% of the word's HORIZONTAL span along its own printed line
    (see `_inside`). Segmented scan_vlm boxes union across their segments.
    """
    det_lines, src = _crop_word_lines(page_lines, scan_lines, crop, prefer_layer)
    words, line_of = _flatten(reading)
    rects, confident, word_conf = _assign_word_rects(
        reading.get("line_text_full") or [], det_lines
    )
    if src == "none" or not any(r is not None for r in rects):
        # No word geometry at all: keep the model's claim, flagged unconfident.
        return dict(reading), [], None, False, src

    box_rects = _box_rects(page, box)

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
        for br in box_rects:
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
    out = dict(reading)
    if inside:
        out["boxed_text"] = " ".join(words[i] for i in inside if i < len(words))
        out["boxed_line_index"] = line_of[inside[0]] if inside[0] < len(line_of) else -1
        out["boxed_spans_lines"] = len(
            {line_of[i] for i in inside if i < len(line_of)}
        )
        contiguous = inside == list(range(inside[0], inside[0] + len(inside)))
        start = inside[0] if contiguous else None
    else:
        out["boxed_text"] = ""
        out["boxed_line_index"] = -1
        out["boxed_spans_lines"] = 0
        start = None
    # Confidence is about the words this case actually rests on, not the whole
    # image. Over a 3-line crop those were nearly the same thing; over a whole
    # page, demanding that all ~31 lines align well reported geom_confident on
    # 7.7% of cases where the crop route reported 66% — measuring page size,
    # not alignment quality. Reduce over the boxed run when there is one.
    if inside and word_conf:
        confident = all(word_conf[i] for i in inside if i < len(word_conf))
    return out, rects, start, confident, src


def _readout_disagreement(a: str, b: str) -> int:
    """Multiset word distance between two boxed readouts: how many words one
    has that the other does not, both directions. 0 = identical; 1 = one side
    read exactly one extra/missing word (the model's known +-1 noise floor)."""
    ca, cb = Counter(locate._norm_words(a or "")), Counter(locate._norm_words(b or ""))
    common = sum((ca & cb).values())
    return sum(ca.values()) + sum(cb.values()) - 2 * common


# ---------------------------------------------------------------------------
# grading (pure python — the heart)
# ---------------------------------------------------------------------------


def _flatten(reading: dict) -> tuple[list[str], list[int]]:
    """(folded words in reading order, their line indices)."""
    words: list[str] = []
    line_of: list[int] = []
    for li, line in enumerate(reading.get("line_text_full") or []):
        for w in locate._norm_words(line or ""):
            words.append(w)
            line_of.append(li)
    return words, line_of


def _find_contiguous(
    words: list[str], needle: list[str], line_of: list[int], prefer_line: int
) -> Optional[int]:
    if not needle:
        return None
    hits = [
        s
        for s in range(len(words) - len(needle) + 1)
        if words[s : s + len(needle)] == needle
    ]
    if not hits:
        return None
    if prefer_line >= 0:
        on_line = [s for s in hits if line_of[s] == prefer_line]
        if on_line:
            return on_line[0]
    return hits[0]


def _best_truth_window(
    variants: list[list[str]], words: list[str], anchor: Optional[int]
) -> tuple[float, int, int, int]:
    """(score, start, length, near-tie count) for the query's true position.

    A phrase printed twice in the crop produces two windows with identical
    scores, and an arbitrary tie-break would report a correctly-placed box on
    the second occurrence as `wrong_line`. Among windows within
    ``locate._MATCH_TIE`` of the best, the one nearest the rectangle wins —
    the same tie discipline ``locate._locate_match`` applies via ``expected_y``
    and ``_align_strip`` via ``prior_center``. ``n_near_tie`` records how
    ambiguous the choice was so the report can surface it.
    """
    cands_fn = getattr(locate, "_window_candidates", None)
    if cands_fn is None or anchor is None:
        score, start, length = locate._best_window(variants, words)
        return score, start, length, 1
    try:
        candidates = list(cands_fn(variants, words))
    except Exception:
        candidates = []
    if not candidates:
        score, start, length = locate._best_window(variants, words)
        return score, start, length, 1
    candidates.sort(key=lambda t: -t[0])
    top = candidates[0][0]
    near = [c for c in candidates if c[0] >= top - locate._MATCH_TIE]
    best = min(near, key=lambda c: (abs(c[1] - anchor), c[1]))
    return best[0], best[1], best[2], len(near)


def _iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = fitz.Rect(a) & b
    ia = 0.0 if inter.is_empty else inter.width * inter.height
    union = a.width * a.height + b.width * b.height - ia
    return (ia / union) if union > 0 else 0.0


def _grade(
    reading: dict,
    variants: list[list[str]],
    page: Optional[fitz.Page] = None,
    box: Optional[dict] = None,
    flat_rects: Optional[list[fitz.Rect]] = None,
    boxed_start_hint: Optional[int] = None,
    insertion: bool = False,
) -> dict:
    """Grade one crop reading. Pure given the reading; PDF arguments are used
    only for the optional IoU."""
    words, line_of = _flatten(reading)
    legible = bool(reading.get("legible", True))
    grade: dict = {
        "verdict": "judge_unusable",
        "shift_words": None,
        "line_delta": None,
        "iou": None,
        "truth_text": "",
        "boxed_text": reading.get("boxed_text") or "",
        "align_score": None,
        "legible": legible,
        "needs_retry": False,
    }
    if not words:
        grade["note"] = "reader returned no words"
        return grade
    if not variants or insertion:
        # A pure-insertion / whitespace-only hunk has no printed phrase to
        # locate, so there is no truth for a box to be compared against.
        # Unmeasurable, not wrong.
        grade["verdict"] = "unmeasurable"
        grade["note"] = "insertion hunk — no printed truth to place a box on"
        return grade

    # Where the rectangle actually landed, first — it anchors the tie-break.
    boxed_words = locate._norm_words(reading.get("boxed_text") or "")
    b_start = boxed_start_hint
    if b_start is None and boxed_words:
        b_start = _find_contiguous(
            words, boxed_words, line_of, int(reading.get("boxed_line_index", -1))
        )

    score, t_start, t_len, n_tie = _best_truth_window(variants, words, b_start)
    grade["n_near_tie"] = n_tie
    grade["align_score"] = round(score, 4) if score is not None else None
    if t_start < 0 or score < locate._MATCH_ACCEPT:
        # The phrase is not in the crop at all. When the crop is legible that
        # is a locator failure (the box is nowhere near the phrase); when it is
        # illegible we cannot tell, so it stays a judge failure.
        grade["verdict"] = "wrong_region" if legible else "phrase_absent"
        return grade
    grade["truth_text"] = " ".join(words[t_start : t_start + t_len])

    if not boxed_words:
        grade["verdict"] = "wrong_region" if legible else "phrase_absent"
        grade["note"] = "reader found no words inside the rectangle"
        return grade
    if b_start is None:
        grade["needs_retry"] = True
        grade["note"] = "boxed words are not a contiguous run of the reading"
        return grade
    b_len = len(boxed_words)

    grade["shift_words"] = b_start - t_start
    grade["line_delta"] = line_of[b_start] - line_of[t_start]

    t_set = set(range(t_start, t_start + t_len))
    b_set = set(range(b_start, b_start + b_len))
    if t_set == b_set:
        verdict = "exact"
    elif t_set & b_set:
        verdict = "partial"
    elif grade["line_delta"] == 0:
        verdict = "shifted"
    else:
        verdict = "wrong_line"
    grade["verdict"] = verdict

    if flat_rects and page is not None and box is not None and t_start + t_len <= len(
        flat_rects
    ):
        truth_rects = [r for r in flat_rects[t_start : t_start + t_len] if r is not None]
        if truth_rects:
            truth_rect = locate._union_rects(truth_rects)
            box_rect = locate._union_rects(_box_rects(page, box))
            grade["iou"] = round(_iou(truth_rect, box_rect), 4)
    return grade


def _acc1(grade: dict) -> bool:
    """Headline metric: exact, or a near-miss on the same printed line."""
    v = grade.get("verdict")
    if v == "exact":
        return True
    if v in ("partial", "shifted"):
        return (
            grade.get("line_delta") == 0
            and grade.get("shift_words") is not None
            and abs(grade["shift_words"]) <= 1
        )
    return False


# ---------------------------------------------------------------------------
# judgement cache (mirrors review._ScanBoxRefiner's discipline)
# ---------------------------------------------------------------------------


class _ReadCache:
    """books/<slug>/locate_read.json — one entry per billed crop READING.

    Deliberately NOT a cache of verdicts. The only thing that costs money is
    the model's transcription of a crop image, and that depends solely on the
    crop's pixels: the page, the rounded crop rectangle, the render scale and
    the model. Grading is pure Python over the cached reading, so changing the
    grading semantics — or the query, or anything else about the case — is
    free. A box only re-bills when it moves enough to change the crop
    rectangle. Same lock discipline and atomic temp-file + os.replace save as
    review._ScanBoxRefiner.
    """

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.path = ws.root / "locate_read.json"
        self.lock = threading.Lock()
        self._cache: Optional[dict] = None
        self._dirty = False

    def _load(self) -> dict:
        if self._cache is None:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._cache = {}
        return self._cache

    def get(self, key: str) -> Optional[dict]:
        with self.lock:
            return self._load().get(key)

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self._load()[key] = value
            self._dirty = True

    def flush(self) -> None:
        with self.lock:
            if not self._dirty or self._cache is None:
                return
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False


def _read_key(
    slug: str, page: int, crop: fitz.Rect, scale_boost: float, model: str
) -> str:
    """Cache key for one billed crop reading.

    Keyed on the CROP, not the box: the crop is what the model sees. Two
    different boxes that happen to produce the same padded crop share one
    reading, and — the point of the split — a box that moves within its crop,
    or a grading change, costs nothing at all.
    """
    raw_key = json.dumps(
        [
            _READ_CACHE_VERSION,
            _CROP_GEOMETRY_VERSION,
            slug,
            page,
            [round(float(v), 3) for v in (crop.x0, crop.y0, crop.x1, crop.y1)],
            round(float(scale_boost), 3),
            model,
        ],
        ensure_ascii=False,
        sort_keys=False,
    )
    return _sha1(raw_key)


# ---------------------------------------------------------------------------
# page-level reading: acquisition geometry, cache, quality gate
# ---------------------------------------------------------------------------


class _PageReadCache(_ReadCache):
    """books/<slug>/locate_page_read.json — one entry per PAGE reading.

    Same discipline as _ReadCache (lock, atomic replace) but keyed on the page,
    never on a box or a crop. This is the file that makes an iteration free:
    nothing a locator change can do will miss a key here.
    """

    def __init__(self, ws: Workspace):
        super().__init__(ws)
        self.path = ws.root / "locate_page_read.json"


def _page_read_key(
    slug: str, page: int, model: str, tile_i: int = 0, n_tiles: int = 1
) -> str:
    """Cache key for one page reading. Mentions no box, by construction."""
    raw_key = json.dumps(
        [
            _PAGE_READ_CACHE_VERSION,
            _PAGE_GEOMETRY_VERSION,
            slug,
            page,
            int(tile_i),
            int(n_tiles),
            round(float(_PAGE_LONG_EDGE), 1),
            model,
        ],
        ensure_ascii=False,
        sort_keys=False,
    )
    return _sha1(raw_key)


def _page_tiles(page: fitz.Page, n_tiles: int) -> list[fitz.Rect]:
    """`n_tiles` full-width horizontal bands covering the page, overlapping by
    ~4% of page height so no printed line is truncated at a tile boundary and
    every line appears whole in at least one tile."""
    r = page.rect
    if n_tiles <= 1:
        return [fitz.Rect(r)]
    band = r.height / n_tiles
    pad = 0.04 * r.height
    out: list[fitz.Rect] = []
    for i in range(n_tiles):
        y0 = max(r.y0, r.y0 + i * band - (pad if i else 0.0))
        y1 = min(r.y1, r.y0 + (i + 1) * band + (pad if i < n_tiles - 1 else 0.0))
        out.append(fitz.Rect(r.x0, y0, r.x1, y1))
    return out


def _page_image(page: fitz.Page, clip: Optional[fitz.Rect] = None) -> bytes:
    """PNG of the page (or one tile of it) rendered so its long edge hits the
    vision ceiling — the most pixels per glyph a reader can actually use.

    Unmarked, deliberately: a page reading is truth about what is PRINTED, and
    must not be biased by where any box happens to sit. The crop path draws a
    magenta rectangle because it asks a box-local question; this one never does.
    """
    rect = fitz.Rect(clip) if clip is not None else fitz.Rect(page.rect)
    scale = _PAGE_LONG_EDGE / max(rect.width, rect.height, 1.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png")


# locate._fold_word keeps the WHOLE Arabic block (its _NONWORD_RE is
# `[^؀-ۿ0-9a-zA-Z]`), and that block contains Persian punctuation: ، U+060C,
# ؛ U+061B, ؟ U+061F, ٪ U+066A, ٫ U+066B and the ornate ﴾ ﴿. So `زد،` folds to
# `زد،` and does NOT equal `زد`, while ASCII `متن.` correctly folds to `متن`.
# For the acquisition gate that is pure noise: whether the reader attached a
# comma to the preceding word or spaced it off says nothing about whether it
# read the page. Measured on the pilot, ignoring this cost ~7 recall points and
# would have put the threshold on top of the noise floor.
#
# NOTE, deliberately not acted on here: this asymmetry is in the PRODUCTION
# word folding, so the locator's own matching sees `زد،` and `زد` as different
# words too. That is a candidate locator fix, and it must be measured as one
# — with a before and after against a frozen answer key — not smuggled in as
# part of the measurement harness.
_ARABIC_PUNCT = "؀؁؂؃؄؅؆؇؈؉؊؋،؍؎؏؛؜؝؞؟٪٫٬٭۔﴾﴿"
_PUNCT_TABLE = {ord(c): None for c in _ARABIC_PUNCT}


def _recall_words(text: str) -> list[str]:
    """`locate._norm_words` with Persian punctuation stripped — words only."""
    out: list[str] = []
    for w in locate._norm_words(text or ""):
        w = w.translate(_PUNCT_TABLE)
        if w:
            out.append(w)
    return out


def _page_recall(reading: dict, page_md: str) -> float:
    """Fraction of the page transcription's words the reading also contains.

    The acquisition quality gate. `text/NNNN.md` is an INDEPENDENT high-res
    transcription of the same page produced by the pipeline itself, so it is a
    fair reference for "did the reader actually read this page" without being
    the same artifact. Multiset recall over folded words, so RTL ordering and
    Arabic/Persian letterform variance do not count against a good reading.
    """
    want = Counter(_recall_words(page_md))
    if not want:
        return 1.0
    have = Counter(
        w
        for line in (reading.get("line_text_full") or [])
        for w in _recall_words(line)
    )
    hit = sum(min(c, have.get(w, 0)) for w, c in want.items())
    return hit / sum(want.values())


def _merge_tile_readings(readings: list[dict]) -> dict:
    """Stitch tile readings into one page reading, dropping the lines the
    overlap duplicates.

    Tiles overlap by design, so the last lines of tile i and the first lines of
    tile i+1 are the same print. A line is a duplicate when its folded word
    sequence equals one already emitted from the previous tile; only the tail
    of the previous tile is consulted, so a genuinely repeated refrain
    elsewhere on the page survives.
    """
    out: list[str] = []
    prev_tail: list[tuple[str, ...]] = []
    for reading in readings:
        lines = list(reading.get("line_text_full") or [])
        emitted: list[tuple[str, ...]] = []
        for line in lines:
            folded = tuple(locate._norm_words(line or ""))
            if folded and folded in prev_tail:
                continue
            out.append(line)
            emitted.append(folded)
        prev_tail = emitted[-6:]
    legible = all(bool(r.get("legible", True)) for r in readings) if readings else False
    notes = [r.get("note") or "" for r in readings]
    return _reading_dict(out, -1, "", 0, legible, "; ".join(n for n in notes if n))


# ---------------------------------------------------------------------------
# locating phase: rebuild the production boxes for every case
# ---------------------------------------------------------------------------


class _CostGuard:
    def __init__(self, limit: float):
        self.limit = limit
        self.spent = 0.0
        self.aborted = False
        self.lock = threading.Lock()

    def add(self, amount: float) -> None:
        with self.lock:
            self.spent += amount
            if self.spent >= self.limit:
                self.aborted = True

    def ok(self) -> bool:
        with self.lock:
            return not self.aborted


def _locate_cases(
    cases: list[dict],
    replay: str,
    refine: bool,
    model: str,
    warnings: list[str],
) -> tuple[list[dict], list[dict]]:
    """Rebuild each case's production box. Returns (live, stale).

    Fidelity: review._box_specs -> review._locate_queries_cached ->
    _ScanBoxRefiner.refine (when enabled) -> review._build_boxes, i.e. the same
    sequence tests/bbox_eval.py drives and the review server runs.
    """
    by_book: dict[str, list[dict]] = {}
    for c in cases:
        by_book.setdefault(c["slug"], []).append(c)

    live: list[dict] = []
    stale: list[dict] = []
    for slug, book_cases in by_book.items():
        ws = Workspace.load(slug)
        refiner = None
        if refine:
            refiner = review._ScanBoxRefiner(ws, model)
        review._REFINER = refiner
        by_page: dict[int, list[dict]] = {}
        for c in book_cases:
            by_page.setdefault(c["page"], []).append(c)
        for n, page_cases in sorted(by_page.items()):
            _sc, issues, hunks, text, panel_kind = _page_inputs(ws, n, replay)
            md_sha = _sha1(text)
            fresh = [c for c in page_cases if c["md_sha1"] == md_sha]
            stale.extend(c for c in page_cases if c["md_sha1"] != md_sha)
            if not fresh:
                continue
            if panel_kind is None or (not issues and not hunks):
                stale.extend(fresh)
                continue
            specs = review._box_specs(issues, hunks, text)
            queries = [s[1] for s in specs]
            if refiner is not None:
                located = list(
                    review._locate_queries_cached(
                        str(ws.pdf_path), n, text, tuple(queries)
                    )
                )
                try:
                    refiner.refine(n, text, queries, located)
                except Exception as exc:  # never let refinement abort scoring
                    warnings.append(f"refine {slug} p{n}: {exc}")
            review._build_boxes(ws, n, issues, hunks, text)
            spec_by_key = {s[0]: s for s in specs}
            wanted = {c["key"]: c for c in fresh}
            for key, case in wanted.items():
                spec = spec_by_key.get(key)
                if spec is None:
                    stale.append(case)
                    continue
                _k, query, _fb, hunk, issue = spec
                box = (hunk or issue).get("box")
                case = dict(case)
                case["_ws"] = ws
                case["_box"] = box
                case["_query"] = query
                case["_page_md"] = text
                case["tier"] = box["source"] if box else "none"
                live.append(case)
        review._REFINER = None
    return live, stale


# ---------------------------------------------------------------------------
# judging phase
# ---------------------------------------------------------------------------


def _variants_for(case: dict) -> tuple[list[list[str]], bool]:
    """(non-empty folded word variants, primary-query-is-empty).

    A hunk whose `old` text is empty is a pure INSERTION: there is no printed
    phrase for a box to sit on, so nothing about its box can be graded. Its
    `alts` (the corrected text) would otherwise be scored as if it were printed
    and every such case would be reported as a locator failure.
    """
    q = case["query"]
    primary = locate._norm_words(q["text"])
    out = [primary] + [locate._norm_words(a) for a in q["alts"]]
    return [v for v in out if v], not primary


def _grade_case(
    reading: dict,
    case: dict,
    page: fitz.Page,
    crop: fitz.Rect,
    page_lines: list[_Line],
    scan_lines: list,
    prefer_layer: bool = True,
) -> dict:
    """Grade a VLM reading for `case`, deriving the boxed run geometrically.

    The model supplies only `line_text_full`; Python decides which of those
    words the rectangle covers (see _geometric_boxed). The model's own
    `boxed_text` is kept in the grade as `boxed_text_vlm` for the health
    metric, and is used as the verdict's basis only when the crop yields no
    word geometry at all.
    """
    variants, insertion = _variants_for(case)
    box = case["_box"]
    vlm_claim = reading.get("boxed_text") or ""
    geom_reading, rects, start, confident, src = _geometric_boxed(
        page, box, crop, reading, page_lines, scan_lines, prefer_layer
    )
    grade = _grade(
        geom_reading,
        variants,
        page=page,
        box=box,
        flat_rects=rects or None,
        boxed_start_hint=start,
        insertion=insertion,
    )
    # An EMPTY claim is "no claim", not "the reader saw nothing boxed": a
    # subagent-supplied reading only ever carries line_text_full (the boxed
    # fields are Python's job, see _reading_dict's neutral values), so scoring
    # its blank claim against the geometric run would report a disagreement of
    # exactly len(boxed_text) on every single such case and poison the
    # boxed_readout_disagreement health metric. None means "not comparable",
    # which _metrics already filters out.
    grade["boxed_text_vlm"] = vlm_claim or None
    grade["geom_confident"] = bool(confident)
    grade["geom_source"] = src
    grade["boxed_disagreement"] = (
        _readout_disagreement(grade.get("boxed_text") or "", vlm_claim)
        if vlm_claim
        else None
    )
    return grade


class _Judge:
    """Per-book judging: routing, crops, batching, caching."""

    def __init__(
        self,
        ws: Workspace,
        judge_mode: str,
        model: str,
        guard: _CostGuard,
        use_cache: bool,
        refresh: bool,
        offline: bool,
        batch: int,
        warnings: list[str],
    ):
        self.ws = ws
        self.judge_mode = judge_mode
        self.model = model
        self.guard = guard
        self.use_cache = use_cache
        self.refresh = refresh
        self.offline = offline
        self.batch = batch
        self.warnings = warnings
        self.n_failed = 0  # crops whose judge call raised (API/network), not data
        self.cache = _ReadCache(ws)
        self.page_cache = _PageReadCache(ws)
        # (slug, page) -> sha1 of the reading actually used, for truth_sha1.
        self.truth_used: dict[int, str] = {}
        self.doc = fitz.open(str(ws.pdf_path))
        self._client = None
        self._client_lock = threading.Lock()
        self._line_h: dict[int, float] = {}
        self._route: dict[int, str] = {}
        self._reader: dict[int, tuple[list[_Line], float, float]] = {}
        self._reader_lock = threading.Lock()
        self._scan: dict[int, list] = {}
        self._scan_lock = threading.Lock()
        self._layer_usable: dict[int, bool] = {}
        self._layer_lock = threading.Lock()

    def close(self) -> None:
        self.cache.flush()
        self.doc.close()

    def client(self):
        with self._client_lock:
            if self._client is None:
                self._client = _llm().get_client()
            return self._client

    def line_h(self, n: int) -> float:
        if n not in self._line_h:
            self._line_h[n] = _median_line_h(self.doc[n - 1])
        return self._line_h[n]

    def reader(self, n: int, page_md: str) -> tuple[list[_Line], float, float]:
        with self._reader_lock:
            if n not in self._reader:
                try:
                    self._reader[n] = _page_reader_lines(self.doc[n - 1], page_md)
                except Exception:
                    self._reader[n] = ([], 0.0, 0.0)
            return self._reader[n]

    def scan(self, n: int) -> list:
        """locate._scan_page_lines for page `n`, memoized (it rasterizes the
        page, so it is far too expensive to call per case)."""
        with self._scan_lock:
            if n not in self._scan:
                try:
                    self._scan[n] = locate._scan_page_lines(self.doc[n - 1])
                except Exception:
                    self._scan[n] = []
            return self._scan[n]

    def layer_usable(self, n: int, page_md: str) -> bool:
        """Memoized `_layer_usable` for page `n` — the SINGLE probe behind two
        decisions that must agree: whether PyMuPDF may stand in for the VLM
        reader (judge routing) and whether its word rects may stand in for the
        scan detector (`_crop_word_lines`). A layer too corrupt to read is
        equally too corrupt to measure geometry with, so they share this. The
        probe re-derives the page's reader lines and folds every word, so it is
        cached: it is consulted once per case otherwise.
        """
        with self._layer_lock:
            if n not in self._layer_usable:
                try:
                    _lines, _ratio, recall = self.reader(n, page_md)
                    usable = _layer_usable(self.doc[n - 1], page_md, recall)
                except Exception:
                    usable = False
                self._layer_usable[n] = usable
            return self._layer_usable[n]

    # -- page-level truth ----------------------------------------------

    def page_reading(self, n: int, page_md: str) -> tuple[Optional[dict], str]:
        """(reading, source) for the whole of page `n` — the answer key.

        Text-layer pages rebuild it from PyMuPDF's own word rects on every run
        (free, exact, never cached — there is nothing to bill). Every other
        page takes the frozen reading from locate_page_read.json. Returning
        None means the page has never been read, which is the ONLY remaining
        way a case can be missing a grade.
        """
        if self.layer_usable(n, page_md):
            page_lines, _ratio, _recall = self.reader(n, page_md)
            reading, _rects, _hint = _det_reading(
                page_lines, self.doc[n - 1], {"x0": 0, "y0": 0, "x1": 0, "y1": 0},
                fitz.Rect(self.doc[n - 1].rect),
            )
            return reading, "pdf_words"
        hit = self.page_cache.get(_page_read_key(self.ws.slug, n, self.model))
        if hit is None or hit.get("reading") is None:
            return None, "missing"
        return hit["reading"], hit.get("source") or "vlm_page"

    def route(self, n: int, page_md: str) -> str:
        if self.judge_mode == "page":
            return "page"
        if self.judge_mode in ("vlm", "deterministic"):
            return "det" if self.judge_mode == "deterministic" else "vlm"
        # "cached" routes exactly like "auto" — text-layer pages still grade
        # deterministically (free, no API, no cache entry needed) and only the
        # pages that genuinely need a reading consult locate_read.json. It is
        # `offline` that makes the mode cache-only. Use `--judge vlm --offline`
        # for the stricter "every case must come from the cache" variant.
        if n not in self._route:
            self._route[n] = "det" if self.layer_usable(n, page_md) else "vlm"
        return self._route[n]

    # -- per page ------------------------------------------------------

    def judge_page(self, n: int, cases: list[dict]) -> None:
        """Fill case["_result"] for every case on page `n`."""
        page = self.doc[n - 1]
        page_md = cases[0]["_page_md"]
        route = self.route(n, page_md)
        model_id = "deterministic" if route == "det" else self.model
        line_h = self.line_h(n)

        if route == "page":
            self._judge_page_truth(n, page, cases, page_md)
            return

        todo: list[dict] = []
        for case in cases:
            box = case["_box"]
            if box is None:
                case["_result"] = {
                    "grade": {
                        "verdict": "no_box",
                        "shift_words": None,
                        "line_delta": None,
                        "iou": None,
                        "truth_text": "",
                        "boxed_text": "",
                        "align_score": None,
                    },
                    "judge": route,
                    "cached": False,
                    "cost": 0.0,
                }
                continue
            if route == "det":
                todo.append(case)
                continue
            # VLM route: the cache holds the READING, keyed on the crop. A hit
            # is re-graded here for free, so a grading change never re-bills.
            crop = _crop_clip(page, box, line_h)
            case["_crop"] = crop
            case["_key"] = _read_key(self.ws.slug, n, crop, 1.0, model_id)
            if self.use_cache and not self.refresh:
                hit = self.cache.get(case["_key"])
                if hit is not None and hit.get("reading") is not None:
                    grade = self._grade_vlm(page, n, case, hit["reading"], page_md)
                    if grade.get("needs_retry"):
                        # This crop was retried at 1.5x last time; reuse that
                        # reading too rather than re-billing it.
                        hit2 = self.cache.get(
                            _read_key(self.ws.slug, n, crop, 1.5, model_id)
                        )
                        if hit2 is not None and hit2.get("reading") is not None:
                            grade = self._grade_vlm(
                                page, n, case, hit2["reading"], page_md
                            )
                        if grade.get("needs_retry"):
                            grade["verdict"] = "judge_unusable"
                    case["_result"] = {
                        "grade": grade, "judge": "vlm", "cached": True, "cost": 0.0,
                    }
                    continue
            todo.append(case)

        if not todo:
            return
        if route == "det":
            page_lines, _ratio, _recall = self.reader(n, page_md)
            for case in todo:
                self._judge_det(page, page_lines, case, line_h)
            return
        if self.offline:
            for case in todo:
                case["_result"] = {"grade": None, "judge": "vlm", "cached": False,
                                   "cost": 0.0, "uncached": True}
            return
        self._judge_vlm(page, n, todo, line_h)

    def _judge_page_truth(
        self, n: int, page: fitz.Page, cases: list[dict], page_md: str
    ) -> None:
        """Grade every case on page `n` against the page's frozen reading.

        Wholly offline and wholly free. The crop passed to grading is the PAGE
        rectangle, so the same `_grade_case` the crop route uses applies
        unchanged — the truth window search, the anchor tie-break that resolves
        a phrase printed twice, and the geometric boxed run all operate over
        the whole page instead of a box-shaped window around the answer.
        """
        reading, source = self.page_reading(n, page_md)
        prefer_layer = self.layer_usable(n, page_md)
        page_lines, _ratio, _recall = self.reader(n, page_md)
        scan_lines: list = []
        if reading is not None:
            self.truth_used[n] = _sha1(
                json.dumps(reading.get("line_text_full") or [], ensure_ascii=False)
            )
            if not prefer_layer:
                scan_lines = self.scan(n)
        for case in cases:
            if case["_box"] is None:
                case["_result"] = {
                    "grade": {
                        "verdict": "no_box",
                        "shift_words": None,
                        "line_delta": None,
                        "iou": None,
                        "truth_text": "",
                        "boxed_text": "",
                        "align_score": None,
                    },
                    "judge": "page",
                    "cached": True,
                    "cost": 0.0,
                }
                continue
            if reading is None:
                # The page has no answer key yet. Not a locator failure.
                case["_result"] = {
                    "grade": None, "judge": "page", "cached": False,
                    "cost": 0.0, "uncached": True,
                }
                continue
            grade = _grade_case(
                reading, case, page, fitz.Rect(page.rect), page_lines, scan_lines,
                prefer_layer,
            )
            grade["truth_source"] = source
            case["_result"] = {
                "grade": grade, "judge": "page", "cached": True, "cost": 0.0,
            }

    def _grade_vlm(
        self, page: fitz.Page, n: int, case: dict, reading: dict, page_md: str
    ) -> dict:
        page_lines, _ratio, _recall = self.reader(n, page_md)
        crop = case.get("_crop") or _crop_clip(page, case["_box"], self.line_h(n))
        # A page reaching the VLM route has, by the shared probe, either an
        # unusable layer (cipher/scan -> must rasterize) or a usable one that
        # judge_mode forced past the deterministic route. Only in the latter
        # case may the layer supply geometry, and only then is the rasterizer
        # avoidable when it already does.
        prefer_layer = self.layer_usable(n, page_md)
        probe, _src = (
            _crop_word_lines(page_lines, [], crop) if prefer_layer else ([], "none")
        )
        scan_lines = [] if probe else self.scan(n)
        return _grade_case(
            reading, case, page, crop, page_lines, scan_lines, prefer_layer
        )

    def _store_reading(self, case: dict, key: Optional[str], reading: Optional[dict],
                       cost: float) -> None:
        if self.use_cache and key and reading is not None:
            self.cache.put(
                key,
                {
                    "reading": reading,
                    "cost": cost,
                    "model": self.model,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )

    def _store(self, case: dict, grade: dict, judge: str, cost: float,
               reading: Optional[dict]) -> None:
        case["_result"] = {"grade": grade, "judge": judge, "cached": False, "cost": cost}

    def _judge_det(
        self, page: fitz.Page, page_lines: list[_Line], case: dict, line_h: float
    ) -> None:
        box = case["_box"]
        variants, insertion = _variants_for(case)
        _png, crop = _judge_crop(page, box, line_h)
        reading, flat_rects, hint = _det_reading(page_lines, page, box, crop)
        grade = _grade(
            reading, variants, page=page, box=box, flat_rects=flat_rects,
            boxed_start_hint=hint, insertion=insertion,
        )
        if grade.get("needs_retry"):
            # Non-contiguous run: retry once with a wider crop before giving up.
            _png2, crop2 = _judge_crop(page, box, line_h * 1.5)
            reading, flat_rects, hint = _det_reading(page_lines, page, box, crop2)
            grade = _grade(
                reading, variants, page=page, box=box, flat_rects=flat_rects,
                boxed_start_hint=hint, insertion=insertion,
            )
            if grade.get("needs_retry"):
                grade["verdict"] = "judge_unusable"
        # The deterministic reader's boxed run is already geometric (it IS the
        # word rects), so it is confident by construction and has no model
        # claim to disagree with.
        grade["boxed_text_vlm"] = None
        grade["geom_confident"] = True
        grade["geom_source"] = "text_layer"
        grade["boxed_disagreement"] = None
        self._store(case, grade, "det", 0.0, reading)

    def _judge_vlm(
        self, page: fitz.Page, n: int, cases: list[dict], line_h: float
    ) -> None:
        page_md = cases[0]["_page_md"]
        for i in range(0, len(cases), self.batch):
            chunk = cases[i : i + self.batch]
            if not self.guard.ok():
                for case in chunk:
                    case["_result"] = {"grade": None, "judge": "vlm", "cached": False,
                                       "cost": 0.0, "aborted": True}
                continue
            crops = [_judge_crop(page, c["_box"], line_h)[0] for c in chunk]
            try:
                readings, _usage, cost = _llm().read_box_crops(
                    self.client(), crops, self.model, n
                )
            except Exception as exc:
                # A failed API call is NOT a measurement. Grading it as
                # judge_unusable (which means "the crop was illegible") would
                # write fiction into the report — an exhausted credit balance
                # once produced a 240-case run reporting judged=240,
                # cost=$0.00, judge_unusable=95%. Mark it excluded instead and
                # let the caller abort on the failure rate.
                self.warnings.append(f"judge {self.ws.slug} p{n}: {exc}")
                self.n_failed += len(chunk)
                for case in chunk:
                    case["_result"] = {"grade": None, "judge": "vlm",
                                       "cached": False, "cost": 0.0,
                                       "failed": True}
                continue
            self.guard.add(cost)
            per = cost / max(1, len(chunk))
            retry: list[dict] = []
            for case, raw in zip(chunk, readings):
                reading = _vlm_reading(raw)
                if reading is None:
                    grade = {"verdict": "judge_unusable", "shift_words": None,
                             "line_delta": None, "iou": None, "truth_text": "",
                             "boxed_text": "", "align_score": None}
                    self._store(case, grade, "vlm", per, None)
                    continue
                self._store_reading(case, case.get("_key"), reading, per)
                grade = self._grade_vlm(page, n, case, reading, page_md)
                if grade.get("needs_retry"):
                    case["_pending_reading"] = reading
                    case["_pending_cost"] = per
                    retry.append(case)
                    continue
                self._store(case, grade, "vlm", per, reading)
            if retry and self.guard.ok():
                self._retry_vlm(page, n, retry, line_h)
            else:
                for case in retry:
                    grade = self._grade_vlm(
                        page, n, case, case["_pending_reading"], page_md
                    )
                    grade["verdict"] = "judge_unusable"
                    self._store(case, grade, "vlm", case["_pending_cost"],
                                case["_pending_reading"])

    def _retry_vlm(
        self, page: fitz.Page, n: int, cases: list[dict], line_h: float
    ) -> None:
        """One retry at 1.5x crop scale for crops whose boxed words did not
        come back as a contiguous run of the reading."""
        page_md = cases[0]["_page_md"]
        crops = [_judge_crop(page, c["_box"], line_h, scale_boost=1.5)[0] for c in cases]
        try:
            readings, _usage, cost = _llm().read_box_crops(
                self.client(), crops, self.model, n
            )
        except Exception as exc:
            self.warnings.append(f"judge-retry {self.ws.slug} p{n}: {exc}")
            readings, cost = [None] * len(cases), 0.0
        self.guard.add(cost)
        per = cost / max(1, len(cases))
        for case, raw in zip(cases, readings):
            fresh = _vlm_reading(raw)
            reading = fresh or case["_pending_reading"]
            if fresh is not None and case.get("_crop") is not None:
                self._store_reading(
                    case,
                    _read_key(self.ws.slug, n, case["_crop"], 1.5, self.model),
                    fresh,
                    per,
                )
            grade = self._grade_vlm(page, n, case, reading, page_md)
            if grade.get("needs_retry"):
                grade["verdict"] = "judge_unusable"
            self._store(case, grade, "vlm", case["_pending_cost"] + per, reading)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def _stratified_sample(
    cases: list[dict], n: int, seed: int, dims: list[str]
) -> list[dict]:
    if n >= len(cases):
        return cases
    rng = random.Random(seed)
    if not dims:
        return sorted(rng.sample(cases, n), key=lambda c: c["id"])
    groups: dict[tuple, list[dict]] = {}
    for c in cases:
        key = tuple(str(c.get(d, "")) for d in dims)
        groups.setdefault(key, []).append(c)
    total = len(cases)
    quotas: list[tuple[float, tuple, int]] = []
    for key, members in groups.items():
        exact = n * len(members) / total
        quotas.append((exact - int(exact), key, int(exact)))
    assigned = {key: base for _frac, key, base in quotas}
    left = n - sum(assigned.values())
    for _frac, key, _base in sorted(quotas, key=lambda t: -t[0])[: max(0, left)]:
        assigned[key] += 1
    out: list[dict] = []
    for key, members in groups.items():
        k = min(assigned.get(key, 0), len(members))
        out.extend(rng.sample(members, k))
    return sorted(out, key=lambda c: c["id"])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    out: dict = {"n": n}
    if n == 0:
        return out
    graded = [r for r in rows if r["verdict"] not in _EXCLUDED_VERDICTS]
    d = len(graded) or 1
    out["n_denominator"] = len(graded)
    excluded = Counter(
        r["verdict"] for r in rows if r["verdict"] in _EXCLUDED_VERDICTS
    )
    if excluded:
        out["excluded"] = dict(excluded)
    out["acc@1"] = round(sum(1 for r in graded if r["acc1"]) / d, 4)
    out["acc@0"] = round(sum(1 for r in graded if r["verdict"] == "exact") / d, 4)
    counts = Counter(r["verdict"] for r in graded)
    out["verdicts"] = {v: counts.get(v, 0) for v in VERDICTS if counts.get(v)}
    out["verdict_rates"] = {
        v: round(counts.get(v, 0) / d, 4) for v in VERDICTS if counts.get(v)
    }
    shifts = [r["shift_words"] for r in graded if r["shift_words"] is not None]
    abs_shifts = [abs(s) for s in shifts]
    out["mean_abs_shift"] = round(_mean(abs_shifts), 3) if abs_shifts else None
    out["median_abs_shift"] = (
        round(statistics.median(abs_shifts), 3) if abs_shifts else None
    )
    out["p90_abs_shift"] = round(_p90(abs_shifts), 3) if abs_shifts else None
    # Signed drift: a non-zero mean is the signature of an accumulating offset
    # and says which way (later/earlier in reading order) to push the locator.
    out["signed_drift"] = round(_mean(shifts), 3) if shifts else None
    out["wrong_line_rate"] = round(counts.get("wrong_line", 0) / d, 4)
    out["no_box_rate"] = round(counts.get("no_box", 0) / d, 4)
    out["judge_unusable_rate"] = round(
        (counts.get("judge_unusable", 0) + counts.get("phrase_absent", 0)) / d, 4
    )
    ious = [r["iou"] for r in graded if r["iou"] is not None]
    out["mean_iou"] = round(_mean(ious), 4) if ious else None
    fallbacks = [r for r in graded if r.get("align_fallback")]
    out["align_fallback_rate"] = round(len(fallbacks) / d, 4)

    # Judge-health metrics for the geometric boxed readout.
    #  * boxed_readout_disagreement: how often Python's geometric boxed run and
    #    the model's own boxed_text claim differ by MORE than one word. A ±1
    #    difference is the model's known noise floor and is not counted.
    #    Read it together with geom_confident_rate: high disagreement WITH high
    #    confidence points at the geometry, not the model.
    #  * geom_confident_rate: fraction of VLM-judged cases where the reading's
    #    line and word counts matched the detected rects exactly (direct index
    #    mapping), rather than needing char-proportional placement.
    #  * geom_fallback_rate: cases where no word rects could be detected in the
    #    crop at all and the model's boxed_text had to stand in.
    comparable = [r for r in graded if r.get("boxed_disagreement") is not None]
    if comparable:
        out["boxed_readout_disagreement"] = round(
            sum(1 for r in comparable if r["boxed_disagreement"] > 1) / len(comparable),
            4,
        )
        out["boxed_readout_n_compared"] = len(comparable)
        out["mean_boxed_disagreement"] = round(
            _mean([float(r["boxed_disagreement"]) for r in comparable]), 3
        )
    with_geom = [r for r in graded if r.get("geom_source") is not None]
    if with_geom:
        out["geom_confident_rate"] = round(
            sum(1 for r in with_geom if r.get("geom_confident")) / len(with_geom), 4
        )
        out["geom_fallback_rate"] = round(
            sum(1 for r in with_geom if r.get("geom_source") == "none")
            / len(with_geom),
            4,
        )
        out["geom_sources"] = dict(Counter(r["geom_source"] for r in with_geom))
    return out


def _slice(rows: list[dict], keyfn) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        for kk in k if isinstance(k, list) else [k]:
            groups.setdefault(str(kk), []).append(r)
    return {k: _metrics(v) for k, v in sorted(groups.items())}


def _feature_keys(row: dict) -> list[str]:
    keys: list[str] = []
    for name in ("page_has_verse", "page_has_footnote_def", "short_query_le14"):
        if row["features"].get(name):
            keys.append(name)
    if row.get("align_fallback"):
        keys.append("align_fallback")
    if row.get("geom_source") is not None and not row.get("geom_confident"):
        # Sliceable: these are the cases whose geometric boxed run needed a
        # proportional fallback, so their verdicts are the least trustworthy.
        keys.append("geom_unconfident")
    keys.append(row.get("y_bucket", "unknown"))
    return keys


# ---------------------------------------------------------------------------
# score command
# ---------------------------------------------------------------------------


def _select_and_locate(
    args: argparse.Namespace, warnings: list[str]
) -> tuple[list[dict], list[dict], Optional[str], Optional[str]]:
    """Case selection + sampling + the locate pass, shared by `score` and
    `crops` so both judge exactly the same boxes.

    Returns (live, stale, cases_sha1, cases_file). Raises ValueError when the
    selection is empty.
    """
    if args.cases:
        cases_path = Path(args.cases)
        blob = json.loads(cases_path.read_text(encoding="utf-8"))
        cases = blob["cases"]
        replay = "all" if blob.get("replay") else "pending"
        cases_sha1 = _sha1(cases_path.read_text(encoding="utf-8"))
        cases_file = str(cases_path)
    else:
        books = _resolve_books(args.books)
        replay = args.replay
        cases = build_cases(books, args.pages, replay)
        cases_sha1 = None
        cases_file = None

    if args.books:
        wanted = set(_resolve_books(args.books))
        cases = [c for c in cases if c["slug"] in wanted]
    if args.pages and args.cases:
        wanted_pages: dict[str, set[int]] = {}
        for c in cases:
            if c["slug"] not in wanted_pages:
                ws = Workspace.load(c["slug"])
                pc = int(ws.meta.get("page_count") or 0)
                wanted_pages[c["slug"]] = set(parse_pages_spec(args.pages, pc))
        cases = [c for c in cases if c["page"] in wanted_pages[c["slug"]]]

    if not cases:
        raise ValueError("no cases selected")

    if args.sample:
        dims = [d.strip() for d in (args.stratify or "").split(",") if d.strip()]
        # `tier` is only known after locating, so a tier-stratified sample needs
        # the locate pass first. Locating is cheap (memoized) and free.
        if "tier" in dims:
            located, stale_pre = _locate_cases(
                cases, replay, args.refine, args.model, warnings
            )
            cases = _stratified_sample(located, args.sample, args.seed, dims)
            live, stale = cases, stale_pre
        else:
            cases = _stratified_sample(cases, args.sample, args.seed, dims)
            live, stale = _locate_cases(
                cases, replay, args.refine, args.model, warnings
            )
    else:
        live, stale = _locate_cases(cases, replay, args.refine, args.model, warnings)

    if stale:
        warnings.append(
            f"{len(stale)} case(s) are stale (page markdown changed since the "
            f"case set was frozen) and were excluded"
        )
    return live, stale, cases_sha1, cases_file


def cmd_score(args: argparse.Namespace) -> int:
    warnings: list[str] = []
    if args.judge == "cached":
        # Cache-only grading: no client is ever constructed.
        args.offline = True
    if args.offline and args.refine:
        # --offline promises zero network traffic; scan-box refinement is an
        # API call, so it is switched off rather than silently violating that.
        args.refine = False
        warnings.append("--offline implies --no-refine (scan boxes stay unrefined)")
    try:
        live, stale, cases_sha1, cases_file = _select_and_locate(args, warnings)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    guard = _CostGuard(args.max_cost)
    by_book: dict[str, list[dict]] = {}
    for c in live:
        by_book.setdefault(c["slug"], []).append(c)

    judges: dict[str, _Judge] = {}
    routing: dict[str, dict] = {}
    truth_index: dict[str, str] = {}   # "slug:pN" -> sha1 of the reading used
    try:
        for slug, book_cases in by_book.items():
            ws = book_cases[0]["_ws"]
            judge = _Judge(
                ws,
                args.judge,
                args.model,
                guard,
                use_cache=not args.no_cache,
                refresh=args.refresh,
                offline=args.offline,
                batch=args.batch,
                warnings=warnings,
            )
            judges[slug] = judge
            by_page: dict[int, list[dict]] = {}
            for c in book_cases:
                by_page.setdefault(c["page"], []).append(c)
            # Routing telemetry: recall probe per book (measured, reported).
            raw_recalls: list[float] = []
            reader_recalls: list[float] = []
            ratios: list[float] = []
            for n in sorted(by_page):
                md = by_page[n][0]["_page_md"]
                try:
                    raw_recalls.append(
                        _layer_recall(locate._page_words(judge.doc[n - 1]), md)
                    )
                    _lines, ratio, recall = judge.reader(n, md)
                    reader_recalls.append(recall)
                    ratios.append(ratio)
                except Exception:
                    pass
            routes = [judge.route(n, by_page[n][0]["_page_md"]) for n in sorted(by_page)]
            routing[slug] = {
                "mean_raw_layer_recall": (
                    round(_mean(raw_recalls), 4) if raw_recalls else None
                ),
                "mean_reader_recall": (
                    round(_mean(reader_recalls), 4) if reader_recalls else None
                ),
                "mean_merge_ratio": round(_mean(ratios), 4) if ratios else None,
                "routes": dict(Counter(routes)),
            }
            with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
                list(
                    pool.map(
                        lambda item: judge.judge_page(item[0], item[1]),
                        sorted(by_page.items()),
                    )
                )
        # The identity of the answer key this run graded against. Two reports
        # sharing a truth_sha1 were scored on the same frozen readings, which
        # is what makes them comparable; `compare` refuses to diff two that do
        # not (see cmd_compare).
        for slug, judge in judges.items():
            for n, h in judge.truth_used.items():
                truth_index[f"{slug}:p{n}"] = h
    finally:
        for judge in judges.values():
            judge.close()

    if guard.aborted:
        warnings.append(
            f"--max-cost ${args.max_cost:.2f} reached; remaining cases were not judged"
        )

    rows: list[dict] = []
    n_uncached = 0
    n_failed = 0
    for c in live:
        res = c.get("_result") or {}
        grade = res.get("grade")
        if grade is None and res.get("failed"):
            n_failed += 1
            verdict = "judge_failed"
            grade = {}
        elif grade is None:
            n_uncached += 1
            verdict = "uncached"
            grade = {}
        else:
            verdict = grade.get("verdict", "judge_unusable")
        box = c["_box"]
        debug = (box or {}).get("debug") or {}
        row = {
            "id": c["id"],
            "slug": c["slug"],
            "page": c["page"],
            "md_sha1": c["md_sha1"],
            "tier": c.get("tier", "none"),
            "kind": c["kind"],
            "issue_type": c.get("issue_type", "unknown"),
            "verdict": verdict,
            "shift_words": grade.get("shift_words"),
            "line_delta": grade.get("line_delta"),
            "iou": grade.get("iou"),
            # boxed_text is now DERIVED FROM GEOMETRY; boxed_text_vlm is the
            # model's own (noisy) claim, kept as a diagnostic only.
            "boxed_text": grade.get("boxed_text", ""),
            "boxed_text_vlm": grade.get("boxed_text_vlm"),
            "geom_confident": grade.get("geom_confident"),
            "geom_source": grade.get("geom_source"),
            "boxed_disagreement": grade.get("boxed_disagreement"),
            "truth_text": grade.get("truth_text", ""),
            "truth_source": grade.get("truth_source"),
            "align_score": grade.get("align_score"),
            "judge": res.get("judge", "det"),
            "cached": bool(res.get("cached")),
            "cost": round(float(res.get("cost", 0.0)), 6),
            "query": c["query"],
            "features": c["features"],
            "y_bucket": _y_bucket(box),
            "align_fallback": debug.get("counts_agree") is False,
            "debug": debug or None,
            "box": box,
        }
        row["acc1"] = _acc1(grade) if verdict not in _EXCLUDED_VERDICTS else False
        rows.append(row)
    for c in stale:
        rows.append(
            {
                "id": c["id"], "slug": c["slug"], "page": c["page"], "tier": "none",
                "md_sha1": c["md_sha1"],
                "kind": c["kind"], "issue_type": c.get("issue_type", "unknown"),
                "verdict": "stale", "shift_words": None, "line_delta": None,
                "iou": None, "boxed_text": "", "truth_text": "", "align_score": None,
                "boxed_text_vlm": None, "geom_confident": None,
                "geom_source": None, "boxed_disagreement": None,
                "judge": "-", "cached": False, "cost": 0.0,
                "query": c["query"], "features": c.get("features", {}),
                "y_bucket": "unknown", "align_fallback": False, "debug": None,
                "box": None, "acc1": False,
            }
        )

    if n_uncached:
        warnings.append(
            f"OFFLINE: {n_uncached} case(s) had no cached judgement and were "
            f"EXCLUDED from every denominator — the numbers below cover only "
            f"the cached subset"
        )

    # A THIN DENOMINATOR IS NOT A RESULT. Iteration 2 reported acc@1=0.4955 on
    # 111 of 240 cases because a structural change moved every box out of its
    # cached crop, and the 129 excluded cases were exactly the ones the change
    # touched. A warning was not enough — the number was quoted anyway. So the
    # run now refuses to produce a report at all when too much of the sample is
    # missing, on the same principle as the judge-failure refusal below.
    excluded_total = n_uncached + len(stale)
    if len(rows) and excluded_total / len(rows) > _MAX_EXCLUDED_FRACTION:
        print(
            f"ERROR: {excluded_total}/{len(rows)} case(s) "
            f"({excluded_total / len(rows):.0%}) have no grade "
            f"(uncached={n_uncached}, stale={len(stale)}), above the "
            f"{_MAX_EXCLUDED_FRACTION:.0%} limit. The surviving subset is "
            f"biased towards the cases your change did NOT affect, so its "
            f"acc@1 is not comparable to anything. Acquire the missing page "
            f"readings first:\n"
            f"  bbox_score.py pages --cases <cases> --out-dir <dir>\n"
            f"  bbox_score.py ingest-pages --manifest <dir>/manifest.json "
            f"--readings <dir>/readings/\n"
            f"Pass --allow-thin-denominator to override (the result is not a "
            f"valid baseline).",
            file=sys.stderr,
        )
        if not getattr(args, "allow_thin_denominator", False):
            return 2

    if n_failed:
        warnings.append(
            f"JUDGE FAILED on {n_failed} case(s) — excluded from every "
            f"denominator. See the warnings above for the underlying error."
        )

    graded_rows = [r for r in rows if r["verdict"] not in _EXCLUDED_VERDICTS]

    # A run where most judge calls errored carries no signal. Refuse to present
    # it as a result: an exhausted API credit balance would otherwise yield a
    # confident-looking report built entirely from failures.
    attempted = n_failed + len(graded_rows)
    if attempted and n_failed / attempted > 0.20:
        print(
            f"ERROR: {n_failed}/{attempted} judge calls failed "
            f"({n_failed / attempted:.0%}). Refusing to write a report from a "
            f"failed run — fix the underlying error and re-run.",
            file=sys.stderr,
        )
        for w in warnings[:3]:
            print(f"  {w}", file=sys.stderr)
        return 2

    sha, dirty = _git_state()
    report = {
        "run": {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sha,
            "dirty": dirty,
            "model": args.model,
            "judge_mode": args.judge,
            "cases_file": cases_file,
            "cases_sha1": cases_sha1,
            "n_cases": len(rows),
            "n_judged": len(graded_rows),
            "n_cached": sum(1 for r in graded_rows if r["cached"]),
            "n_stale": len(stale),
            "n_uncached": n_uncached,
            "truth_sha1": _sha1(
                json.dumps(truth_index, ensure_ascii=False, sort_keys=True)
            ) if truth_index else None,
            "truth_pages": len(truth_index),
            "cost_usd": round(sum(r["cost"] for r in rows), 6),
            "crop_geometry_version": _CROP_GEOMETRY_VERSION,
            "score_cache_version": _SCORE_CACHE_VERSION,
            "refine": bool(args.refine),
            "routing": routing,
            "warnings": warnings,
            "sign_convention": (
                "shift_words is reading-order index space; positive means the "
                "box must move LATER in reading order"
            ),
        },
        "overall": _metrics(rows),
        "by_tier": _slice(rows, lambda r: r["tier"]),
        "by_book": _slice(rows, lambda r: r["slug"]),
        "by_kind": _slice(rows, lambda r: r["kind"]),
        "by_issue_type": _slice(rows, lambda r: r["issue_type"]),
        "by_feature": _slice(rows, _feature_keys),
        "cases": rows,
    }

    if args.bias_audit and not args.offline:
        report["bias_audit"] = _bias_audit(live, args, guard, warnings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_path}")

    if args.html:
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        _write_html(rows, report, html_path)
        print(f"wrote {html_path} ({html_path.stat().st_size / 1024:.0f} KiB)")

    _print_summary(report)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


def _print_summary(report: dict) -> None:
    run, overall = report["run"], report["overall"]
    print(
        f"cases={run['n_cases']} judged={run['n_judged']} cached={run['n_cached']} "
        f"stale={run['n_stale']} uncached={run['n_uncached']} cost=${run['cost_usd']:.4f}"
    )
    print(
        f"acc@1={overall.get('acc@1')} acc@0={overall.get('acc@0')} "
        f"signed_drift={overall.get('signed_drift')} "
        f"wrong_line={overall.get('wrong_line_rate')} "
        f"no_box={overall.get('no_box_rate')} "
        f"judge_unusable={overall.get('judge_unusable_rate')} "
        f"mean_iou={overall.get('mean_iou')}"
    )
    print("verdicts: " + json.dumps(overall.get("verdicts", {})))
    print(
        f"judge health: boxed_readout_disagreement="
        f"{overall.get('boxed_readout_disagreement')} "
        f"(n={overall.get('boxed_readout_n_compared')}, "
        f"mean={overall.get('mean_boxed_disagreement')}) "
        f"geom_confident={overall.get('geom_confident_rate')} "
        f"geom_fallback={overall.get('geom_fallback_rate')}"
    )
    for name in ("by_tier", "by_book", "by_kind"):
        parts = [
            f"{k}={v.get('acc@1')}({v.get('n')})" for k, v in report[name].items()
        ]
        print(f"{name}: " + "  ".join(parts))


# ---------------------------------------------------------------------------
# bias audit
# ---------------------------------------------------------------------------


def _bias_audit(
    live: list[dict], args: argparse.Namespace, guard: _CostGuard, warnings: list[str]
) -> dict:
    """Send the CLEAN crop and the MARKED crop of the same region as two images
    in one message and diff the two line_text_full readings. A high divergence
    rate would mean drawing the rectangle biases the transcription — which
    would invalidate the whole "the VLM transcribes, Python grades" design."""
    pool = [c for c in live if c["_box"] is not None]
    sample = _stratified_sample(pool, min(args.bias_audit, len(pool)), args.seed,
                                ["slug", "tier"])
    diverged = 0
    checked = 0
    cost = 0.0
    by_book: dict[str, list[dict]] = {}
    for c in sample:
        by_book.setdefault(c["slug"], []).append(c)
    for slug, cases in by_book.items():
        ws = cases[0]["_ws"]
        doc = fitz.open(str(ws.pdf_path))
        try:
            client = _llm().get_client()
            for case in cases:
                if not guard.ok():
                    break
                page = doc[case["page"] - 1]
                line_h = _median_line_h(page)
                marked, clip = _judge_crop(page, case["_box"], line_h)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        max(2.0, min(4.0, 1900.0 / max(clip.width, 1.0))),
                        max(2.0, min(4.0, 1900.0 / max(clip.width, 1.0))),
                    ),
                    clip=clip,
                )
                clean = pix.tobytes("png")
                try:
                    readings, _u, c_cost = _llm().read_box_crops(
                        client, [clean, marked], args.model, case["page"]
                    )
                except Exception as exc:
                    warnings.append(f"bias-audit {slug} p{case['page']}: {exc}")
                    continue
                guard.add(c_cost)
                cost += c_cost
                if readings[0] is None or readings[1] is None:
                    continue
                checked += 1
                a = [locate._norm_words(x) for x in readings[0].line_text_full]
                b = [locate._norm_words(x) for x in readings[1].line_text_full]
                if a != b:
                    diverged += 1
        finally:
            doc.close()
    return {
        "n_checked": checked,
        "n_diverged": diverged,
        "divergence_rate": round(diverged / checked, 4) if checked else None,
        "cost_usd": round(cost, 6),
    }


# ---------------------------------------------------------------------------
# HTML sheet
# ---------------------------------------------------------------------------


_VERDICT_COLORS = {
    "exact": "#2fa35a",
    "partial": "#7ac65c",
    "shifted": "#f0c24a",
    "wrong_line": "#f0a03a",
    "wrong_region": "#d9534f",
    "phrase_absent": "#b05ad9",
    "no_box": "#8a8b93",
    "judge_unusable": "#55565e",
    "unmeasurable": "#3a3b42",
    "stale": "#3a3b42",
    "uncached": "#3a3b42",
}
_MAX_HTML_ROWS = 200


def _shift_caption(row: dict) -> str:
    s = row.get("shift_words")
    if s is None:
        return "no reading-order offset measured"
    if s == 0:
        return "0 words offset in reading order"
    direction = "later" if s > 0 else "earlier"
    return f"{abs(s)} word(s) {direction} in reading order"


def _write_html(rows: list[dict], report: dict, path: Path) -> None:
    shown = sorted(
        rows,
        key=lambda r: (
            VERDICT_RANK.get(r["verdict"], 9),
            -abs(r["shift_words"] or 0),
            r["id"],
        ),
    )[:_MAX_HTML_ROWS]

    imgs: dict[str, str] = {}
    by_book: dict[str, list[dict]] = {}
    for r in shown:
        if r.get("box"):
            by_book.setdefault(r["slug"], []).append(r)
    for slug, book_rows in by_book.items():
        try:
            ws = Workspace.load(slug)
            doc = fitz.open(str(ws.pdf_path))
        except Exception:
            continue
        try:
            for r in book_rows:
                try:
                    page = doc[r["page"] - 1]
                    # The judge's own crop, magenta mark and all: the sheet
                    # should show exactly the image the verdict was derived
                    # from. (tests/bbox_eval._crop_png is imported for its tier
                    # palette only — its own drawing is a no-op on clipped
                    # pixmaps, see the pix.x/pix.y note in _judge_crop.)
                    png, _clip = _judge_crop(page, r["box"], _median_line_h(page))
                    imgs[r["id"]] = base64.standard_b64encode(png).decode("ascii")
                except Exception:
                    pass
        finally:
            doc.close()

    overall = report["overall"]
    chips = "\n".join(
        f".v-{v} {{ background: {c}; }}" for v, c in _VERDICT_COLORS.items()
    )
    tier_chips = "\n".join(
        f".t-{t} {{ background: {c}; }}" for t, c in bbox_eval.TIER_COLORS.items()
    )
    parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>bbox score — acc@1 {overall.get('acc@1')}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; background: #16161a; color: #e8e8ea; }}
header {{ padding: 1rem 1.5rem; border-bottom: 1px solid #333; position: sticky; top: 0; background: #16161aee; backdrop-filter: blur(4px); z-index: 5; }}
header h1 {{ font-size: 1.1rem; margin: 0 0 .4rem; }}
header p {{ margin: .15rem 0; font-size: .9rem; color: #b8b8c0; }}
.chip {{ display:inline-block; padding:.1rem .55rem; border-radius:999px; color:#0d0d0f; font-size:.78rem; font-weight:700; vertical-align:middle; }}
{chips}
{tier_chips}
.row {{ display: grid; grid-template-columns: 420px 1fr; gap: 1rem; padding: 1rem 1.5rem; border-bottom: 1px solid #2a2a30; align-items: center; }}
.meta {{ font-size: .85rem; color: #b8b8c0; }}
.meta .id {{ font-family: "SF Mono", monospace; font-size: .8rem; color: #8a8b93; display:block; margin-bottom:.35rem; }}
.caption {{ margin-top:.4rem; font-size:.9rem; color:#e8e8ea; }}
.texts {{ margin-top:.6rem; display:grid; grid-template-columns:1fr; gap:.35rem; }}
.texts div {{ direction: rtl; text-align: right; font-family:"Vazirmatn","Tahoma",sans-serif; font-size:1.05rem; line-height:1.9; background:#1d1d22; border-radius:4px; padding:.3rem .5rem; }}
.texts .lbl {{ direction: ltr; text-align: left; font-size:.72rem; color:#7a7a85; background:none; padding:0; font-family:inherit; }}
.row img {{ max-width: 100%; height: auto; border: 1px solid #333; border-radius: 4px; background: #fff; }}
</style>
</head>
<body>
<header>
<h1>bbox score — acc@1 {overall.get('acc@1')} · acc@0 {overall.get('acc@0')} · n={overall.get('n_denominator')}</h1>
<p>signed_drift {overall.get('signed_drift')} (positive = boxes sit LATER in reading order than the truth) ·
mean|shift| {overall.get('mean_abs_shift')} · p90 {overall.get('p90_abs_shift')} ·
wrong_line {overall.get('wrong_line_rate')} · no_box {overall.get('no_box_rate')} ·
mean_iou {overall.get('mean_iou')}</p>
<p>verdicts: {html_mod.escape(json.dumps(overall.get('verdicts', {})))}</p>
<p>Sorted worst first. Offsets are reading-order word indices, never screen positions.</p>
</header>
"""
    ]
    for r in shown:
        img = imgs.get(r["id"])
        parts.append(
            f"""<div class="row">
<div class="meta">
  <span class="id">{html_mod.escape(r['id'])}</span>
  <span class="chip v-{html_mod.escape(r['verdict'])}">{html_mod.escape(r['verdict'])}</span>
  <span class="chip t-{html_mod.escape(r['tier'])}">{html_mod.escape(r['tier'])}</span>
  <span>align {r['align_score']} · iou {r['iou']} · {html_mod.escape(r['kind'])} · {html_mod.escape(r['judge'])}</span>
  <div class="caption">{html_mod.escape(_shift_caption(r))}{' · line_delta ' + str(r['line_delta']) if r['line_delta'] is not None else ''}</div>
  <div class="texts">
    <div class="lbl">truth (what the page says at the query's position)</div>
    <div dir="rtl">{html_mod.escape(r['truth_text'] or '—')}</div>
    <div class="lbl">boxed — GEOMETRIC (words the rectangle covers &ge;50% of horizontally){'' if r.get('geom_confident') is not False else ' · low confidence: alignment not trustworthy'}</div>
    <div dir="rtl">{html_mod.escape(r['boxed_text'] or '—')}</div>
    {f'''<div class="lbl">boxed — model's claim (diagnostic only, not used for the verdict)</div>
    <div dir="rtl">{html_mod.escape(r['boxed_text_vlm'] or '—')}</div>''' if r.get('boxed_text_vlm') is not None else ''}
  </div>
</div>
<div>{f'<img src="data:image/png;base64,{img}" alt="crop">' if img else '<em>no box drawn</em>'}</div>
</div>
"""
        )
    parts.append("</body>\n</html>\n")
    path.write_text("".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# subagent judging: crops -> (Claude Code subagent) -> ingest-readings
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. The only billed artifact in this file is the model's
# transcription of a crop image (see _ReadCache), and the API credit balance is
# the bottleneck on the improvement loop: ~2000 crops at the measured
# $0.00633/crop. But "transcribe the printed lines in this crop" is a plain
# perception task with no privileged access to anything — a Claude Code
# subagent can do it just as well, billed to the consumer plan instead.
#
# The cache split makes that a drop-in substitution rather than a second code
# path: `crops` renders exactly the images the API judge would have sent, named
# by their _read_key; `ingest-readings` writes the returned transcriptions into
# locate_read.json under those same keys; `score --judge cached` then grades
# them with the identical pure-Python code the API route uses. Nothing
# downstream can distinguish the two except by the entry's `judge` field.
#
# NOT INCLUDED, deliberately: review._ScanBoxRefiner (production box placement,
# not measurement) stays on the API, and the subagent is never asked which
# words the rectangle covers — see _subagent_reading.

_SUBAGENT_INSTRUCTIONS = (
    "For each crop PNG listed below: transcribe EVERY printed text line fully "
    "visible in the crop, top to bottom, one list entry per printed line, "
    "verbatim. The crops come from printed or scanned Persian (Farsi) book "
    "pages. Each crop has one vivid magenta rectangle drawn on it by a "
    "program; IGNORE it completely — transcribe as if it were not there, "
    "including text outside it, and do NOT report which words it covers "
    "(Python derives that from geometry). Never merge two printed lines into "
    "one entry and never split one printed line across entries. Plain text "
    "only: no Markdown, no headings, no corrections, no modernized spelling. "
    "Use standard Persian codepoints: always ی (U+06CC) and ک "
    "(U+06A9), never Arabic ي or ك; keep ZWNJ (U+200C) where the "
    "print shows joined-boundary compounds. A crop may clip letters or dots at "
    "its edges; transcribe the line anyway. Guess rather than omit, but never "
    "invent words that are not printed. Set legible=false when a crop is too "
    "blurry or dark to read. note: at most 15 words of English, or \"\"."
)

_READINGS_SCHEMA = {
    "readings": [
        {
            "read_key": "<the read_key of the crop, exactly as listed>",
            "line_text_full": ["<printed line 1>", "<printed line 2>"],
            "legible": True,
            "note": "",
        }
    ]
}

# Fields the subagent must NOT supply: they are geometric judgements Python
# makes from the detected word rects, and accepting them would undo the whole
# "the VLM transcribes, Python grades" design.
_FORBIDDEN_READING_FIELDS = ("boxed_text", "boxed_line_index", "boxed_spans_lines")


def _page_needs_vlm(page: fitz.Page, page_md: str) -> bool:
    """Same routing decision `_Judge.route` makes in auto/cached mode: pages
    whose text layer can stand in for the reader cost nothing and need no
    crop."""
    try:
        _lines, _ratio, recall = _page_reader_lines(page, page_md)
        return not _layer_usable(page, page_md, recall)
    except Exception:
        return True


_PAGE_SUBAGENT_INSTRUCTIONS = (
    "For each page PNG listed below: transcribe EVERY printed text line on the "
    "page, top to bottom, one list entry per printed line, verbatim. The pages "
    "come from printed or scanned Persian (Farsi) books. There are NO marks or "
    "rectangles on these images — transcribe the whole page as printed. "
    "Never merge two printed lines into one entry and never split one printed "
    "line across entries; the line structure is the measurement, so a merged "
    "or split line corrupts it. Include headings, page numbers, footnotes and "
    "running headers as their own lines, in the position they are printed. "
    "Plain text only: no Markdown, no corrections, no modernized spelling. "
    "Use standard Persian codepoints: always ی (U+06CC) and ک (U+06A9), never "
    "Arabic ي or ك; keep ZWNJ (U+200C) where the print shows joined-boundary "
    "compounds. Guess rather than omit, but never invent words that are not "
    "printed. Set legible=false only when the page is too blurry or dark to "
    "read at all. note: at most 15 words of English, or \"\"."
)

_PAGE_READINGS_SCHEMA = {
    "readings": [
        {
            "read_key": "<the read_key of the page, exactly as listed>",
            "line_text_full": ["<printed line 1>", "<printed line 2>"],
            "legible": True,
            "note": "",
        }
    ]
}


def cmd_pages(args: argparse.Namespace) -> int:
    """Render one image per page needing an answer key, plus subagent batches.

    This is the ONLY acquisition step that ever costs anything in the new loop,
    and it is per PAGE, not per case or per box: 146 sampled pages instead of
    ~700 box crops, acquired once and reused by every future iteration.
    """
    warnings: list[str] = []
    try:
        live, _stale, cases_sha1, cases_file = _select_and_locate(args, warnings)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    batches_dir = out_dir / "BATCHES"
    pages_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    by_book: dict[str, list[dict]] = {}
    for c in live:
        by_book.setdefault(c["slug"], []).append(c)

    entries: dict[str, dict] = {}
    n_det_pages = 0
    n_cached_pages = 0

    for slug in sorted(by_book):
        book_cases = by_book[slug]
        ws = book_cases[0]["_ws"]
        cache = _PageReadCache(ws)
        doc = fitz.open(str(ws.pdf_path))
        try:
            by_page: dict[int, list[dict]] = {}
            for c in book_cases:
                by_page.setdefault(c["page"], []).append(c)
            for n in sorted(by_page):
                page = doc[n - 1]
                page_md = by_page[n][0]["_page_md"]
                # Text-layer pages need no reading at all: PyMuPDF's own word
                # rects rebuild the answer key for free on every run.
                if not args.include_deterministic and not _page_needs_vlm(
                    page, page_md
                ):
                    n_det_pages += 1
                    continue
                key = _page_read_key(slug, n, args.model)
                if args.only_uncached:
                    hit = cache.get(key)
                    if hit is not None and hit.get("reading") is not None:
                        n_cached_pages += 1
                        continue
                png = _page_image(page)
                (pages_dir / f"{key}.png").write_bytes(png)
                entries[key] = {
                    "read_key": key,
                    "png": f"pages/{key}.png",
                    "slug": slug,
                    "page": n,
                    "n_cases": len(by_page[n]),
                }
        finally:
            doc.close()

    rows = [entries[k] for k in sorted(entries, key=lambda k: (entries[k]["slug"],
                                                              entries[k]["page"]))]
    manifest = {
        "version": 1,
        "kind": "pages",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "page_geometry_version": _PAGE_GEOMETRY_VERSION,
        "page_read_cache_version": _PAGE_READ_CACHE_VERSION,
        "long_edge": _PAGE_LONG_EDGE,
        "cases_file": cases_file,
        "cases_sha1": cases_sha1,
        "n_cases_covered": sum(e["n_cases"] for e in rows),
        "instructions": _PAGE_SUBAGENT_INSTRUCTIONS,
        "readings_schema": _PAGE_READINGS_SCHEMA,
        "pages": rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_batches = 0
    for i in range(0, len(rows), _PAGE_SUBAGENT_BATCH):
        chunk = rows[i : i + _PAGE_SUBAGENT_BATCH]
        n_batches += 1
        (batches_dir / f"batch_{n_batches:02d}.json").write_text(
            json.dumps(
                {
                    "batch": n_batches,
                    "manifest": str((out_dir / "manifest.json").resolve()),
                    "root": str(out_dir.resolve()),
                    "instructions": _PAGE_SUBAGENT_INSTRUCTIONS,
                    "output_schema": _PAGE_READINGS_SCHEMA,
                    "output_note": (
                        "Write ONE JSON file with a `readings` entry per page "
                        "below, then ingest it with: bbox_score.py "
                        "ingest-pages --manifest <manifest> --readings <file>"
                    ),
                    "pages": [
                        {
                            "read_key": e["read_key"],
                            "png": str((out_dir / e["png"]).resolve()),
                        }
                        for e in chunk
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"wrote {out_dir}: {len(rows)} page(s) covering "
        f"{manifest['n_cases_covered']} case(s) in {n_batches} batch(es) of "
        f"{_PAGE_SUBAGENT_BATCH}"
    )
    print(
        f"skipped: {n_cached_pages} already-read, {n_det_pages} text-layer "
        f"(graded free from the PDF's own word rects)"
    )
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


def cmd_ingest_pages(args: argparse.Namespace) -> int:
    """Validate subagent page readings, gate them on quality, and freeze them.

    THE ANSWER KEY VERIFIES ITSELF HERE. Each reading is scored against the
    page's own independent transcription (`text/NNNN.md`, produced by the
    pipeline's hi-res vision pass) with `_page_recall`. A whole-page render is
    the cheap acquisition shape but it is also the lowest-acuity one, so a page
    the reader fumbled must be caught at ingest rather than silently becoming
    wrong truth for every future iteration. Below `_PAGE_RECALL_MIN` the
    reading is REJECTED and the page is reported for re-reading at tile
    resolution (`pages --tiles 3`).
    """
    manifest_path = Path(args.manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read manifest: {exc}", file=sys.stderr)
        return 1
    if manifest.get("kind") != "pages":
        print(
            "ERROR: this manifest was written by `crops`, not `pages`. Use "
            "ingest-readings for box-crop readings.",
            file=sys.stderr,
        )
        return 1
    if manifest.get("page_geometry_version") != _PAGE_GEOMETRY_VERSION:
        print(
            f"ERROR: manifest page_geometry_version "
            f"{manifest.get('page_geometry_version')} != "
            f"{_PAGE_GEOMETRY_VERSION}; re-render the pages.",
            file=sys.stderr,
        )
        return 1

    by_key = {e["read_key"]: e for e in manifest.get("pages", [])}
    model = manifest.get("model") or MODEL_STRONG

    rows: list[dict] = []
    for path in _readings_files(args.readings):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            return 1
        got = blob.get("readings") if isinstance(blob, dict) else blob
        if not isinstance(got, list):
            print(f"ERROR: {path}: no `readings` list", file=sys.stderr)
            return 1
        rows.extend(got)

    caches: dict[str, _PageReadCache] = {}
    workspaces: dict[str, Workspace] = {}
    cached_keys: dict[str, set[str]] = {}
    seen: set[str] = set()
    n_ok = 0
    rejected: list[tuple[str, str]] = []
    low_recall: list[tuple[str, int, float]] = []

    for row in rows:
        slug_page = "?"
        try:
            entry = by_key.get((row or {}).get("read_key"))
        except AttributeError:
            entry = None
        if entry is None:
            rejected.append(("?", "read_key not in manifest"))
            continue
        slug, n = entry["slug"], entry["page"]
        slug_page = f"{slug}:p{n}"
        if slug not in caches:
            workspaces[slug] = Workspace.load(slug)
            caches[slug] = _PageReadCache(workspaces[slug])
            cached_keys[slug] = {
                k for k, e in by_key.items()
                if e["slug"] == slug and caches[slug].get(k) is not None
            }
        cache = caches[slug]
        norm, why = _validate_row(
            row, by_key, seen, cached_keys[slug], args.force
        )
        if norm is None:
            rejected.append((slug_page, why or "invalid"))
            continue
        seen.add(norm["read_key"])

        reading = _subagent_reading(
            norm["line_text_full"], norm["legible"], norm["note"]
        )
        try:
            page_md = (workspaces[slug].root / "text" / f"{n:04d}.md").read_text(
                encoding="utf-8"
            )
        except OSError:
            page_md = ""
        recall = _page_recall(reading, page_md)
        if page_md and recall < _PAGE_RECALL_MIN:
            low_recall.append((slug_page, len(reading["line_text_full"]), recall))
            rejected.append(
                (slug_page, f"recall {recall:.2f} < {_PAGE_RECALL_MIN} vs the "
                            f"page's own transcription")
            )
            continue
        if args.dry_run:
            n_ok += 1
            continue
        cache.put(
            norm["read_key"],
            {
                "reading": reading,
                "model": model,
                "judge": "subagent",
                "source": "vlm_page",
                "recall": round(recall, 4),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "cost": 0.0,
            },
        )
        n_ok += 1

    for cache in caches.values():
        cache.flush()

    print(
        f"ingested {n_ok} page reading(s)"
        + (" (dry run, nothing written)" if args.dry_run else "")
    )
    if rejected:
        print(f"rejected {len(rejected)}:", file=sys.stderr)
        for sp, why in rejected[:20]:
            print(f"  {sp}: {why}", file=sys.stderr)
    if low_recall:
        print(
            f"\n{len(low_recall)} page(s) read below the quality gate — re-read "
            f"these at tile resolution:",
            file=sys.stderr,
        )
        for sp, n_lines, rec in sorted(low_recall, key=lambda t: t[2]):
            print(f"  {sp}: recall={rec:.2f} lines={n_lines}", file=sys.stderr)
    return 0 if n_ok else 1


def cmd_crops(args: argparse.Namespace) -> int:
    warnings: list[str] = []
    try:
        live, _stale, cases_sha1, cases_file = _select_and_locate(args, warnings)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    batches_dir = out_dir / "BATCHES"
    crops_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    by_book: dict[str, list[dict]] = {}
    for c in live:
        by_book.setdefault(c["slug"], []).append(c)

    entries: dict[str, dict] = {}   # read_key -> manifest row (deduped)
    n_no_box = 0
    n_det = 0
    n_cached_skipped = 0            # cases whose crop already has a reading
    cached_keys: set[str] = set()

    for slug in sorted(by_book):
        book_cases = by_book[slug]
        ws = book_cases[0]["_ws"]
        cache = _ReadCache(ws)
        doc = fitz.open(str(ws.pdf_path))
        try:
            by_page: dict[int, list[dict]] = {}
            for c in book_cases:
                by_page.setdefault(c["page"], []).append(c)
            for n in sorted(by_page):
                page = doc[n - 1]
                page_cases = by_page[n]
                if not args.include_deterministic and not _page_needs_vlm(
                    page, page_cases[0]["_page_md"]
                ):
                    n_det += len(page_cases)
                    continue
                line_h = _median_line_h(page)
                for case in page_cases:
                    box = case["_box"]
                    if box is None:
                        n_no_box += 1     # no_box is graded without a reading
                        continue
                    crop = _crop_clip(page, box, line_h)
                    key = _read_key(slug, n, crop, 1.0, args.model)
                    if key in entries:
                        entries[key]["n_cases"] += 1
                        continue
                    if key in cached_keys:
                        n_cached_skipped += 1
                        continue
                    if args.only_uncached:
                        hit = cache.get(key)
                        if hit is not None and hit.get("reading") is not None:
                            cached_keys.add(key)
                            n_cached_skipped += 1
                            continue
                    png, _clip = _judge_crop(page, box, line_h)
                    (crops_dir / f"{key}.png").write_bytes(png)
                    entries[key] = {
                        "read_key": key,
                        "png": f"crops/{key}.png",
                        "slug": slug,
                        "page": n,
                        "scale_boost": 1.0,
                        "n_cases": 1,
                    }
        finally:
            doc.close()

    crops = [entries[k] for k in sorted(entries)]
    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "crop_geometry_version": _CROP_GEOMETRY_VERSION,
        "read_cache_version": _READ_CACHE_VERSION,
        "cases_file": cases_file,
        "cases_sha1": cases_sha1,
        "n_cases_covered": sum(e["n_cases"] for e in crops),
        "instructions": _SUBAGENT_INSTRUCTIONS,
        "readings_schema": _READINGS_SCHEMA,
        "crops": crops,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_batches = 0
    for i in range(0, len(crops), _SUBAGENT_BATCH):
        chunk = crops[i : i + _SUBAGENT_BATCH]
        n_batches += 1
        batch_path = batches_dir / f"batch_{n_batches:02d}.json"
        batch_path.write_text(
            json.dumps(
                {
                    "batch": n_batches,
                    "manifest": str((out_dir / "manifest.json").resolve()),
                    "root": str(out_dir.resolve()),
                    "instructions": _SUBAGENT_INSTRUCTIONS,
                    "output_schema": _READINGS_SCHEMA,
                    "output_note": (
                        "Write ONE JSON file with a `readings` entry per crop "
                        "below, then ingest it with: bbox_score.py "
                        "ingest-readings --manifest <manifest> --readings <file>"
                    ),
                    "crops": [
                        {
                            "read_key": e["read_key"],
                            "png": str((out_dir / e["png"]).resolve()),
                        }
                        for e in chunk
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"wrote {out_dir}: {len(crops)} crop(s) covering "
        f"{manifest['n_cases_covered']} case(s) in {n_batches} batch(es) of "
        f"{_SUBAGENT_BATCH}"
    )
    print(
        f"skipped: {n_cached_skipped} already-cached, {n_det} deterministic-route, "
        f"{n_no_box} no-box"
    )
    print(
        f"estimated API cost avoided: ${len(crops) * _CROP_READ_USD:.2f} "
        f"({len(crops)} x ${_CROP_READ_USD})"
    )
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


# -- ingest ----------------------------------------------------------------


def _validate_row(
    row, by_key: dict[str, dict], seen: set[str], cached: set[str], force: bool
) -> tuple[Optional[dict], Optional[str]]:
    """(normalized row, rejection reason). Strict by construction: there is no
    schema enforcement on the subagent side, so this is the only integrity
    boundary between a hand-written JSON file and the scored corpus."""
    if not isinstance(row, dict):
        return None, "row is not a JSON object"
    key = row.get("read_key")
    if not isinstance(key, str) or not key:
        return None, "missing or non-string read_key"
    if key not in by_key:
        return None, f"read_key not in manifest: {key[:12]}…"
    for f in _FORBIDDEN_READING_FIELDS:
        if f in row:
            return None, (
                f"supplied `{f}` — the boxed run is derived geometrically by "
                f"Python and must not be claimed by the reader"
            )
    if key in seen:
        return None, "duplicate read_key within this ingest run"
    if key in cached and not force:
        return None, "already has a cached reading (pass --force to overwrite)"
    lines = row.get("line_text_full")
    if lines is None:
        return None, "missing line_text_full"
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        return None, "line_text_full is not a list of strings"
    legible = row.get("legible", True)
    if not isinstance(legible, bool):
        return None, "legible is not a boolean"
    if legible and not [x for x in lines if x.strip()]:
        return None, "line_text_full is empty but legible is true"
    note = row.get("note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        return None, "note is not a string"
    return {
        "read_key": key,
        "line_text_full": lines,
        "legible": legible,
        "note": note,
    }, None


def _readings_files(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        return sorted(p.glob("*.json"))
    return [p]


def cmd_ingest_readings(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read manifest {manifest_path}: {exc}", file=sys.stderr)
        return 2
    if manifest.get("read_cache_version") != _READ_CACHE_VERSION or manifest.get(
        "crop_geometry_version"
    ) != _CROP_GEOMETRY_VERSION:
        print(
            "ERROR: manifest was written under a different crop geometry / read "
            "cache version; its read_keys no longer address these crops. "
            "Re-run `crops`.",
            file=sys.stderr,
        )
        return 2
    by_key = {c["read_key"]: c for c in manifest.get("crops") or []}
    if not by_key:
        print("ERROR: manifest lists no crops", file=sys.stderr)
        return 2

    files = _readings_files(args.readings)
    if not files:
        print(f"ERROR: no readings JSON found at {args.readings}", file=sys.stderr)
        return 2

    caches: dict[str, _ReadCache] = {}

    def _cache_for(slug: str) -> _ReadCache:
        if slug not in caches:
            caches[slug] = _ReadCache(Workspace.load(slug))
        return caches[slug]

    # Pre-read the existing cache state so "already cached" is decided against
    # the file on disk, not against rows this run just wrote.
    cached: set[str] = set()
    for key, entry in by_key.items():
        hit = _cache_for(entry["slug"]).get(key)
        if hit is not None and hit.get("reading") is not None:
            cached.add(key)

    seen: set[str] = set()
    n_accepted = 0
    n_rejected = 0
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model = manifest.get("model")

    for path in files:
        f_ok = 0
        reasons: list[str] = []
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            n_rejected += 1
            print(f"{path.name}: 0 accepted, 1 rejected")
            print(f"  REJECT (whole file): unreadable JSON: {exc}")
            continue
        rows = blob.get("readings") if isinstance(blob, dict) else None
        if not isinstance(rows, list):
            n_rejected += 1
            print(f"{path.name}: 0 accepted, 1 rejected")
            print('  REJECT (whole file): expected an object with a "readings" list')
            continue
        for i, row in enumerate(rows):
            norm, reason = _validate_row(row, by_key, seen, cached, args.force)
            if norm is None:
                n_rejected += 1
                reasons.append(f"  REJECT row {i}: {reason}")
                continue
            key = norm["read_key"]
            seen.add(key)
            entry = by_key[key]
            reading = _subagent_reading(
                norm["line_text_full"], norm["legible"], norm["note"]
            )
            if not args.dry_run:
                _cache_for(entry["slug"]).put(
                    key,
                    {
                        "reading": reading,
                        "cost": 0.0,
                        "model": model,
                        "ts": ts,
                        "judge": "subagent",
                    },
                )
            f_ok += 1
            n_accepted += 1
        print(f"{path.name}: {f_ok} accepted, {len(reasons)} rejected")
        for r in reasons:
            print(r)

    if args.dry_run:
        print(
            f"DRY RUN — nothing written. total: {n_accepted} accepted, "
            f"{n_rejected} rejected"
        )
        return 0
    written: list[str] = []
    for slug, cache in caches.items():
        # A cache is only touched when a row for that book was accepted; the
        # rest were opened purely to probe for existing readings.
        if cache._dirty:
            written.append(str(cache.path))
        cache.flush()
    print(
        f"total: {n_accepted} accepted, {n_rejected} rejected; wrote "
        + (", ".join(sorted(written)) or "nothing")
    )
    if n_rejected:
        print(
            f"ERROR: {n_rejected} row(s)/file(s) were rejected (accepted rows were "
            f"still written)",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# compare / golden
# ---------------------------------------------------------------------------


def _mcnemar_exact_p(n_fixed: int, n_broken: int) -> Optional[float]:
    """Two-sided exact McNemar p-value over the DISCORDANT pairs.

    The cases are paired (same case, before and after), so the only evidence a
    change carries is the cases whose verdict flipped; the hundreds that agree
    are not evidence either way. Under the null "a flip is equally likely in
    either direction", the fixed count is Binomial(discordant, 0.5).

    This exists because "0.4685 -> 0.4955, 3 fixed, 0 broken" was read as a
    real improvement: 3 discordant pairs give p=0.25, which is noise. Detecting
    a true 3-point shift at 80% power needs roughly 15-25 discordant pairs.
    """
    n = n_fixed + n_broken
    if n == 0:
        return None
    k = min(n_fixed, n_broken)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def cmd_compare(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8"))

    # COMPARABILITY IS CHECKED, NOT ASSUMED. Two reports are only comparable if
    # they were graded against the same frozen answer key; otherwise a moved
    # denominator or a re-read page can masquerade as a locator improvement.
    b_truth = base["run"].get("truth_sha1")
    c_truth = cand["run"].get("truth_sha1")
    if b_truth != c_truth and not args.allow_truth_drift:
        print(
            f"ERROR: these reports were graded against DIFFERENT answer keys "
            f"(baseline truth_sha1={str(b_truth)[:12]}, "
            f"candidate={str(c_truth)[:12]}). Any difference below could come "
            f"from the truth changing rather than the locator. Re-score both "
            f"against the same page readings, or pass --allow-truth-drift if "
            f"you know why they differ.",
            file=sys.stderr,
        )
        return 2

    b_cases = {c["id"]: c for c in base["cases"]}
    c_cases = {c["id"]: c for c in cand["cases"]}
    shared = sorted(set(b_cases) & set(c_cases))
    # Only cases GRADED in both runs are paired evidence.
    scored = [
        i for i in shared
        if b_cases[i]["verdict"] not in _EXCLUDED_VERDICTS
        and c_cases[i]["verdict"] not in _EXCLUDED_VERDICTS
    ]

    fixed = [i for i in scored if not b_cases[i]["acc1"] and c_cases[i]["acc1"]]
    broken = [i for i in scored if b_cases[i]["acc1"] and not c_cases[i]["acc1"]]

    b_acc = base["overall"].get("acc@1")
    c_acc = cand["overall"].get("acc@1")
    print(f"overall acc@1: {b_acc} -> {c_acc}  (n shared {len(shared)}, "
          f"graded in both {len(scored)})")
    print(f"fixed ({len(fixed)}): " + (", ".join(fixed) or "—"))
    print(f"broken ({len(broken)}): " + (", ".join(broken) or "—"))
    p = _mcnemar_exact_p(len(fixed), len(broken))
    if p is None:
        print("McNemar: no discordant pairs — this change moved nothing")
    else:
        print(
            f"McNemar exact p={p:.4f} on {len(fixed) + len(broken)} discordant "
            f"pair(s)" + ("" if p < 0.05 else "  ← NOT significant")
        )
    for tier in sorted(set(base["by_tier"]) | set(cand["by_tier"])):
        ba = base["by_tier"].get(tier, {}).get("acc@1")
        ca = cand["by_tier"].get(tier, {}).get("acc@1")
        print(f"  tier {tier}: {ba} -> {ca}")

    if not args.gate:
        return 0

    failures: list[str] = []
    if c_acc is None or b_acc is None or c_acc <= b_acc:
        failures.append(f"overall acc@1 did not improve ({b_acc} -> {c_acc})")
    for tier in sorted(set(base["by_tier"]) & set(cand["by_tier"])):
        ba = base["by_tier"][tier].get("acc@1")
        ca = cand["by_tier"][tier].get("acc@1")
        if ba is not None and ca is not None and (ba - ca) * 100.0 > 2.0:
            failures.append(f"tier {tier} acc@1 regressed {ba} -> {ca} (>2.0 points)")
    b_nb = base["overall"].get("no_box_rate") or 0.0
    c_nb = cand["overall"].get("no_box_rate") or 0.0
    if c_nb > b_nb:
        failures.append(f"no_box_rate increased {b_nb} -> {c_nb}")
    ju = cand["overall"].get("judge_unusable_rate") or 0.0
    if ju > 0.03:
        failures.append(f"judge_unusable rate {ju} exceeds 3%")
    golden_path = PROJECT_ROOT / "tests" / "data" / "bbox_golden.json"
    if golden_path.is_file():
        golden_ids = {
            g["id"] for g in json.loads(golden_path.read_text(encoding="utf-8"))["cases"]
        }
        hit = sorted(golden_ids & set(broken))
        if hit:
            failures.append(f"golden case(s) broken: {', '.join(hit)}")
    for f in failures:
        print(f"GATE FAIL: {f}", file=sys.stderr)
    return 1 if failures else 0


def cmd_golden(args: argparse.Namespace) -> int:
    report = json.loads(Path(getattr(args, "from")).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    existing: dict[str, dict] = {}
    if out_path.is_file():
        existing = {
            g["id"]: g
            for g in json.loads(out_path.read_text(encoding="utf-8"))["cases"]
        }

    by_id = {c["id"]: c for c in report["cases"]}
    frozen: list[dict] = []
    promoted: list[str] = []
    for c in report["cases"]:
        if c["verdict"] != "exact":
            continue
        if c.get("align_score") is None or c["align_score"] < locate._MATCH_ACCEPT:
            continue  # not a self-consistent reading
        entry = {
            "id": c["id"],
            "slug": c["slug"],
            "page": c["page"],
            "md_sha1": c.get("md_sha1"),
            "query": c.get("query"),
            "expect": {
                "source": c["tier"],
                "box": c.get("box"),
                "segments": (c.get("box") or {}).get("segments"),
                "min_iou": 0.60,
                "tol": {"x": 0.02, "y": 0.01},
            },
            "evidence": {
                "judge": c["judge"],
                "boxed_text": c["boxed_text"],
                "align_score": c["align_score"],
                "judged_at": report["run"]["ts"],
                "cost": c["cost"],
            },
        }
        old = existing.get(c["id"])
        if old and old["expect"].get("box") != entry["expect"].get("box"):
            if not args.promote:
                frozen.append(old)
                continue
            promoted.append(c["id"])
            print(
                f"promote {c['id']}: box "
                f"{json.dumps(old['expect'].get('box'), ensure_ascii=False)} -> "
                f"{json.dumps(entry['expect'].get('box'), ensure_ascii=False)}"
            )
        frozen.append(entry)

    # Keep previously-golden cases that this report did not cover.
    covered = {e["id"] for e in frozen}
    for gid, g in existing.items():
        if gid not in covered and gid not in by_id:
            frozen.append(g)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "from": str(getattr(args, "from")),
                "crop_geometry_version": _CROP_GEOMETRY_VERSION,
                "cases": sorted(frozen, key=lambda g: g["id"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(frozen)} golden case(s), {len(promoted)} promoted)")
    return 0


# ---------------------------------------------------------------------------
# cases command / CLI
# ---------------------------------------------------------------------------


def _resolve_books(spec: Optional[str]) -> list[str]:
    if spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    return sorted(
        p.name
        for p in DEFAULT_BOOKS_ROOT.iterdir()
        if (p / "book.yaml").is_file() and (p / "source.pdf").is_file()
    )


def cmd_cases(args: argparse.Namespace) -> int:
    books = _resolve_books(args.books)
    cases = build_cases(books, args.pages, args.replay)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "books": books,
                "pages": args.pages,
                "replay": args.replay == "all",
                "n_cases": len(cases),
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    per_book = Counter(c["slug"] for c in cases)
    per_kind = Counter(c["kind"] for c in cases)
    pages = len({(c["slug"], c["page"]) for c in cases})
    print(f"wrote {out_path}: {len(cases)} case(s) over {pages} page(s)")
    print("by book: " + ", ".join(f"{k}={v}" for k, v in sorted(per_book.items())))
    print("by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(per_kind.items())))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cases = sub.add_parser("cases", help="enumerate locate queries into a case set")
    p_cases.add_argument("--books", default=None, help="comma-separated slugs")
    p_cases.add_argument("--pages", default=None)
    p_cases.add_argument("--replay", choices=["all", "pending"], default="all")
    p_cases.add_argument("--out", default="tests/data/bbox_cases.json")
    p_cases.set_defaults(func=cmd_cases)

    p_score = sub.add_parser("score", help="judge boxes and write a score report")
    p_score.add_argument("--cases", default=None)
    p_score.add_argument("--books", default=None)
    p_score.add_argument("--pages", default=None)
    p_score.add_argument("--replay", choices=["all", "pending"], default="all")
    p_score.add_argument("--sample", type=int, default=None)
    p_score.add_argument("--seed", type=int, default=0)
    p_score.add_argument("--stratify", default="book,tier,kind")
    p_score.add_argument(
        "--judge",
        choices=["auto", "vlm", "deterministic", "cached", "page"],
        default="page",
        help="page (default) = grade against the frozen per-page answer key in "
             "locate_page_read.json; fully offline, free, and immune to box "
             "movement. cached = the superseded box-crop cache.",
    )
    p_score.add_argument(
        "--allow-thin-denominator",
        action="store_true",
        help="write a report even when most of the sample has no grade (the "
             "result is not a valid baseline; see _MAX_EXCLUDED_FRACTION)",
    )
    p_score.add_argument("--model", default=MODEL_STRONG)
    p_score.add_argument("--batch", type=int, default=_MAX_CROPS_PER_CALL)
    p_score.add_argument("--refine", dest="refine", action="store_true", default=True)
    p_score.add_argument("--no-refine", dest="refine", action="store_false")
    p_score.add_argument("--refresh", action="store_true", help="ignore cached reads")
    p_score.add_argument("--no-cache", action="store_true", help="neither read nor write")
    p_score.add_argument("--offline", action="store_true", help="cache-only; no API")
    p_score.add_argument("--bias-audit", type=int, default=0)
    p_score.add_argument(
        "--max-cost", type=float, default=2.0, help="abort mid-run above this (USD)"
    )
    p_score.add_argument("--out", default="out/bbox_score.json")
    p_score.add_argument("--html", default=None)
    p_score.set_defaults(func=cmd_score)

    p_pages = sub.add_parser(
        "pages",
        help="render one page image per sampled page for a subagent to read "
             "(the answer key; acquired once, never invalidated by a locator "
             "change)",
    )
    p_pages.add_argument("--cases", default=None)
    p_pages.add_argument("--books", default=None)
    p_pages.add_argument("--pages", default=None)
    p_pages.add_argument("--replay", choices=["all", "pending"], default="all")
    p_pages.add_argument("--sample", type=int, default=None)
    p_pages.add_argument("--seed", type=int, default=0)
    p_pages.add_argument("--stratify", default="book,tier,kind")
    # Boxes are irrelevant to page acquisition, but _select_and_locate builds
    # them; --no-refine keeps that from billing the refinement API.
    p_pages.add_argument("--refine", dest="refine", action="store_true", default=False)
    p_pages.add_argument("--no-refine", dest="refine", action="store_false")
    p_pages.add_argument("--model", default=MODEL_STRONG,
                         help="must match the --model `score` will use: it is "
                              "part of the page read cache key")
    p_pages.add_argument("--out-dir", required=True)
    p_pages.add_argument(
        "--only-uncached", dest="only_uncached", action="store_true", default=True,
        help="skip pages already in locate_page_read.json (default)",
    )
    p_pages.add_argument(
        "--all-pages", dest="only_uncached", action="store_false",
        help="re-emit pages that already have a reading",
    )
    p_pages.add_argument(
        "--include-deterministic", action="store_true",
        help="also emit pages whose text layer grades them for free",
    )
    p_pages.set_defaults(func=cmd_pages)

    p_ingp = sub.add_parser(
        "ingest-pages", help="load subagent PAGE readings into the answer key"
    )
    p_ingp.add_argument("--manifest", required=True)
    p_ingp.add_argument("--readings", required=True, help="a JSON file or a directory")
    p_ingp.add_argument("--dry-run", action="store_true")
    p_ingp.add_argument("--force", action="store_true",
                        help="overwrite pages that already have a reading")
    p_ingp.set_defaults(func=cmd_ingest_pages)

    p_crops = sub.add_parser(
        "crops", help="render judge crops for a Claude Code subagent to read"
    )
    p_crops.add_argument("--cases", default=None)
    p_crops.add_argument("--books", default=None)
    p_crops.add_argument("--pages", default=None)
    p_crops.add_argument("--replay", choices=["all", "pending"], default="all")
    p_crops.add_argument("--sample", type=int, default=None)
    p_crops.add_argument("--seed", type=int, default=0)
    p_crops.add_argument("--stratify", default="book,tier,kind")
    p_crops.add_argument("--refine", dest="refine", action="store_true", default=True)
    p_crops.add_argument("--no-refine", dest="refine", action="store_false")
    p_crops.add_argument("--model", default=MODEL_STRONG,
                         help="must match the --model `score` will use: it is "
                              "part of the read cache key")
    p_crops.add_argument("--out-dir", required=True)
    p_crops.add_argument(
        "--only-uncached", dest="only_uncached", action="store_true", default=True,
        help="skip crops already in locate_read.json (default; the cost control)",
    )
    p_crops.add_argument(
        "--all-crops", dest="only_uncached", action="store_false",
        help="re-emit crops that already have a cached reading",
    )
    p_crops.add_argument(
        "--include-deterministic", action="store_true",
        help="also emit crops for pages the text-layer reader can grade for free",
    )
    p_crops.set_defaults(func=cmd_crops)

    p_ing = sub.add_parser(
        "ingest-readings", help="load subagent crop readings into the read cache"
    )
    p_ing.add_argument("--manifest", required=True)
    p_ing.add_argument("--readings", required=True, help="a JSON file or a directory")
    p_ing.add_argument("--dry-run", action="store_true")
    p_ing.add_argument("--force", action="store_true",
                       help="overwrite keys that already have a cached reading")
    p_ing.set_defaults(func=cmd_ingest_readings)

    p_cmp = sub.add_parser("compare", help="diff two score reports")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--candidate", required=True)
    p_cmp.add_argument("--gate", action="store_true")
    p_cmp.add_argument(
        "--allow-truth-drift", action="store_true",
        help="compare reports graded against different answer keys (refused by "
             "default: the difference may be the truth, not the locator)",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_gold = sub.add_parser("golden", help="freeze exact cases into a golden set")
    p_gold.add_argument("--from", dest="from", required=True)
    p_gold.add_argument("--out", default="tests/data/bbox_golden.json")
    p_gold.add_argument("--promote", action="store_true")
    p_gold.set_defaults(func=cmd_golden)

    args = ap.parse_args(argv)
    if getattr(args, "offline", False) and getattr(args, "judge", "") == "vlm":
        print("note: --offline with --judge vlm reports uncached misses only")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
