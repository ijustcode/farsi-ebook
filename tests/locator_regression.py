"""Regression checks for legacy RTL PDF text-layer phrase location.

Usage: source venv/bin/activate && python tests/locator_regression.py

Includes synthetic unit checks for the VLM strip-alignment helper
(`_align_strip` — pure geometry, no PDF/API), plumbing checks for
`refine_scan_boxes` with a fake strip reader on the real scanned book, and
one API-gated live check of `llm.read_strips` (skipped without a key).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farsi2epub.locate import (  # noqa: E402
    Query,
    _align_strip,
    _fold_word,
    _locate_match,
    _rect_to_fracs,
    _resolve_span,
    _scan_hits,
    _scan_page_lines,
    _ScanLine,
    _strip_rect,
    _render_strip,
    _union_rects,
    locate_queries,
    refine_scan_boxes,
)
from farsi2epub.review import (  # noqa: E402
    _LOCATE_VLM_CACHE_VERSION,
    _PAGE_TEMPLATE,
    _ScanBoxRefiner,
    _attach_boxes,
    _ripped_segment_points,
    _tear_profile,
)


class _Page:
    rect = fitz.Rect(0, 0, 420, 595)


def _mk_line(y0: float, y1: float, words_text: list[str]) -> _ScanLine:
    """Synthetic detected line: fixed-width word boxes laid out right-to-left
    from x=500, so words[0] is the RIGHTMOST rect (RTL reading order), exactly
    like _scan_page_lines produces."""
    words: list[fitz.Rect] = []
    x1 = 500.0
    for _ in words_text:
        words.append(fitz.Rect(x1 - 50.0, y0, x1, y1))
        x1 -= 60.0
    rect = fitz.Rect(words[-1].x0, y0, words[0].x1, y1)
    return _ScanLine(rect=rect, words=words)


def _rects_close(a: fitz.Rect, b: fitz.Rect, tol: float = 1e-9) -> bool:
    return (
        abs(a.x0 - b.x0) < tol
        and abs(a.y0 - b.y0) < tol
        and abs(a.x1 - b.x1) < tol
        and abs(a.y1 - b.y1) < tol
    )


def _check_align_strip_synthetic() -> None:
    """Pure _align_strip checks on hand-built strips: no PDF, no API."""
    l0_words = ["درخت", "سنگ", "ابر", "ماه", "ستاره"]
    l1_words = ["کتاب", "قلم", "دفتر", "مدرسه", "معلم", "کلاس"]
    l2_words = ["نان", "پنیر", "چای", "شیر", "عسل"]
    strip = [
        _mk_line(100, 120, l0_words),
        _mk_line(130, 150, l1_words),
        _mk_line(160, 180, l2_words),
    ]
    vlm = [" ".join(l0_words), " ".join(l1_words), " ".join(l2_words)]

    # 1. Exact word-index mapping: line counts agree, per-line word counts
    # agree, query = words 2-4 (0-based) of line 1 -> rect must equal the
    # union of exactly those detected word rects.
    rects = _align_strip([["دفتر", "مدرسه", "معلم"]], vlm, strip, None)
    assert rects is not None and len(rects) == 1
    expected = _union_rects(strip[1].words[2:5])
    assert _rects_close(rects[0], expected), f"{rects[0]} != {expected}"

    # 2. RTL orientation: the FIRST word of a VLM line (reading order) maps to
    # the RIGHTMOST detected rect of that line.
    rects = _align_strip([["کتاب"]], vlm, strip, None)
    assert rects is not None and len(rects) == 1
    rightmost = max(strip[1].words, key=lambda r: r.x1)
    assert _rects_close(rects[0], strip[1].words[0])
    assert _rects_close(rects[0], rightmost)

    # 3. Count-mismatch fallback: VLM says 4 words on a line where only 3
    # visual words were detected -> char-proportional placement, box must
    # still land inside that line's rect.
    line = _mk_line(200, 220, ["ا", "ب", "پ"])  # 3 detected word rects
    rects = _align_strip([["دو", "سه"]], ["یک دو سه چهار"], [line], None)
    assert rects is not None and len(rects) == 1
    rect = rects[0]
    assert rect.x0 >= line.rect.x0 - 1e-9 and rect.x1 <= line.rect.x1 + 1e-9
    assert rect.y0 >= line.rect.y0 - 1e-9 and rect.y1 <= line.rect.y1 + 1e-9
    assert rect.width < line.rect.width  # narrowed, not the whole line

    # 4. Alt rescue: the main query text is garbage (scores below the accept
    # threshold) but an alternate (corrected) text matches -> accepted.
    assert _align_strip([["قظفغصضچجح"]], vlm, strip, None) is None
    rects = _align_strip([["قظفغصضچجح"], ["مدرسه", "معلم"]], vlm, strip, None)
    assert rects is not None and len(rects) == 1
    assert _rects_close(rects[0], _union_rects(strip[1].words[3:5]))

    # 5. Rejection: garbage VLM lines never reach the accept threshold.
    assert (
        _align_strip([["خورشید", "دریاچه"]], ["تش کج غس", "لم خت وی"], strip, None)
        is None
    )

    # 6. Tie-break: the same phrase printed twice in the strip -> the window
    # nearer prior_center wins.
    twice = [
        _mk_line(100, 120, ["گل", "بلبل", "باغ"]),
        _mk_line(130, 150, ["میز", "صندلی", "فرش"]),
        _mk_line(160, 180, ["گل", "بلبل", "باغ"]),
    ]
    vlm2 = ["گل بلبل باغ", "میز صندلی فرش", "گل بلبل باغ"]
    q = [["گل", "بلبل", "باغ"]]
    top = _align_strip(q, vlm2, twice, fitz.Point(300, 110))
    assert top is not None and len(top) == 1
    assert _rects_close(top[0], _union_rects(twice[0].words))
    bottom = _align_strip(q, vlm2, twice, fitz.Point(300, 170))
    assert bottom is not None and len(bottom) == 1
    assert _rects_close(bottom[0], _union_rects(twice[2].words))

    # 7. Two-line RTL wrap: preserve one tight rectangle per printed line in
    # top-to-bottom reading order instead of returning their page-wide union.
    wrapped = _align_strip(
        [["ماه", "ستاره", "کتاب", "قلم"]], vlm, strip, None
    )
    assert wrapped is not None and len(wrapped) == 2
    assert _rects_close(wrapped[0], _union_rects(strip[0].words[3:5]))
    assert _rects_close(wrapped[1], _union_rects(strip[1].words[0:2]))

    # 8. Three-line wrap: intermediate lines remain independent segments.
    three_lines = _align_strip(
        [["ستاره", "کتاب", "قلم", "دفتر", "مدرسه", "معلم", "کلاس", "نان"]],
        vlm,
        strip,
        None,
    )
    assert three_lines is not None and len(three_lines) == 3
    assert _rects_close(three_lines[0], strip[0].words[4])
    assert _rects_close(three_lines[1], _union_rects(strip[1].words))
    assert _rects_close(three_lines[2], strip[2].words[0])

    # 9. When VLM and detected line counts disagree, the proportional fallback
    # still groups selected visual words by detected line.
    mismatch_strip = [
        _mk_line(230, 250, ["ا", "ب", "پ"]),
        _mk_line(260, 280, ["ت", "ث", "ج"]),
    ]
    mismatch = _align_strip(
        [["پ", "ت"]], ["ا ب پ ت ث ج"], mismatch_strip, None
    )
    assert mismatch is not None and len(mismatch) == 2
    assert mismatch[0].y0 == mismatch_strip[0].rect.y0
    assert mismatch[1].y0 == mismatch_strip[1].rect.y0


def _check_review_segment_plumbing() -> None:
    """Segment payload, torn presentation geometry, zoom contract, and cache."""
    hunk: dict = {}
    located = {
        "x0": 0.10,
        "y0": 0.20,
        "x1": 0.90,
        "y1": 0.36,
        "source": "scan_vlm",
        "segments": [
            {"x0": 0.10, "y0": 0.20, "x1": 0.28, "y1": 0.24},
            {"x0": 0.72, "y0": 0.26, "x1": 0.90, "y1": 0.30},
            {"x0": 0.12, "y0": 0.32, "x1": 0.30, "y1": 0.36},
        ],
    }
    specs = [("h7", Query("الف"), None, hunk, None)]
    view = _attach_boxes(specs, [located])[0]
    assert view["x0"] == 10.0 and view["x1"] == 90.0  # union compatibility envelope
    assert len(view["segments"]) == 3 and len(view["paths"]) == 3
    assert hunk["box"] is located  # payload keeps 0-1 source geometry

    first, middle, last = view["segments"]
    first_pts = _ripped_segment_points(first, 0, 3, "h7")
    middle_pts = _ripped_segment_points(middle, 1, 3, "h7")
    last_pts = _ripped_segment_points(last, 2, 3, "h7")

    def interior(
        points: list[tuple[float, float]], segment: dict
    ) -> list[tuple[float, float]]:
        return [p for p in points if segment["y0"] < p[1] < segment["y1"]]

    first_inner = interior(first_pts, first)
    middle_inner = interior(middle_pts, middle)
    last_inner = interior(last_pts, last)
    assert first_inner and all(x < first["x0"] + 1.0 for x, _y in first_inner)
    assert any(x > first["x0"] for x, _y in first_inner)  # left edge torn inward
    assert any(x < middle["x1"] for x, _y in middle_inner)  # right edge torn
    assert any(x > middle["x0"] for x, _y in middle_inner)  # left edge torn
    assert last_inner and all(x > last["x1"] - 1.0 for x, _y in last_inner)
    assert any(x < last["x1"] for x, _y in last_inner)  # right edge torn inward
    assert _tear_profile("h7", 0) == _tear_profile("h7", 0)  # stable pair profile

    # Browser renderer contract: no connector, stable logical box IDs, and
    # multipart focus set from the first ordered segment.
    assert "qc-box-connector" not in _PAGE_TEMPLATE
    assert "'box-' + page + '-' + b.key" in _PAGE_TEMPLATE
    assert "setFocusGeometry(svg, segments[0])" in _PAGE_TEMPLATE
    assert "data-focus-x0" in _PAGE_TEMPLATE

    # Versioned cache keys make union-only v1 entries miss once. Negative v2
    # entries count as resolved; positive segmented geometry round-trips.
    with tempfile.TemporaryDirectory() as tmp:
        refiner = _ScanBoxRefiner(SimpleNamespace(root=Path(tmp)), "test-model")
        q = Query("الف", (0, 3))
        page_md = "الف"
        current_key = refiner._key(1, page_md, q)
        md_sha = hashlib.sha1(page_md.encode("utf-8")).hexdigest()
        legacy_raw = json.dumps(
            [1, md_sha, q.text, list(q.alts), list(q.span)], ensure_ascii=False
        )
        legacy_key = hashlib.sha1(legacy_raw.encode("utf-8")).hexdigest()
        assert _LOCATE_VLM_CACHE_VERSION == 2 and current_key != legacy_key

        scan_box = {"x0": 0.1, "y0": 0.2, "x1": 0.2, "y1": 0.3, "source": "scan"}
        refiner.cache_path.write_text(
            json.dumps({current_key: {"box": None, "cost": 0.0}}), encoding="utf-8"
        )
        assert not refiner.pending(1, page_md, [q], [scan_box])
        assert refiner.apply_cached(1, page_md, [q], [scan_box]) == [scan_box]

        refined_box = dict(located)
        refiner._cache = {current_key: {"box": refined_box, "cost": 0.0}}
        applied = refiner.apply_cached(1, page_md, [q], [scan_box])[0]
        assert applied is not None and len(applied["segments"]) == 3


def _check_refine_plumbing(root: Path) -> None:
    """refine_scan_boxes with fake readers on the real scanned book: position
    pass-through, successful refinement into the pinned y-band, and the
    widen-then-give-up path."""
    md = (root / "text" / "0008.md").read_text(encoding="utf-8")
    q = Query("بداالله دنبال خر بود")
    scan_box = locate_queries(root / "source.pdf", 8, md, [q])[0]
    assert scan_box is not None and scan_box["source"] == "scan"

    doc = fitz.open(root / "source.pdf")
    try:
        page = doc[7]
        lines = _scan_page_lines(page)
        span = _resolve_span(md, q)
        assert span is not None
        hits = _scan_hits(md, span, lines)
        assert hits
        lo = max(0, hits[0][0] - 1)
        hi = min(len(lines) - 1, hits[-1][0] + 1)
        target = hits[0][0]  # detected line the fake reading puts the query on
        nwords = len(lines[target].words)
        qwords = q.text.split()
        assert nwords >= len(qwords)
        off = min(3, nwords - len(qwords))

        filler = [
            "سیب", "انار", "هلو", "گردو", "بادام", "پسته",
            "آلبالو", "زردآلو", "توت", "خرما", "انجیر", "لیمو",
        ]
        target_line_words = [filler[i % len(filler)] for i in range(nwords)]
        target_line_words[off : off + len(qwords)] = qwords
        reading: list[str] = []
        for i in range(lo, hi + 1):
            if i == target:
                reading.append(" ".join(target_line_words))
            else:
                reading.append("میز صندلی")

        calls: list[int] = []

        def _good_reader(strips: list[bytes]) -> list[list[str]]:
            calls.append(len(strips))
            return [reading for _ in strips]

        # Non-scan and None positions pass through as None (caller keeps the
        # original box); the scan position gets a refined scan_vlm box snapped
        # onto the target detected line's word rects, inside the y-band the
        # plain-scan regression above pins. (Near-best tie windows may shift
        # the x-extent by one word, so pin the line, not the exact word run.)
        match_box = {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.12, "source": "match"}
        refined = refine_scan_boxes(
            root / "source.pdf",
            8,
            md,
            [Query("الف"), Query("ب"), q],
            [None, match_box, scan_box],
            _good_reader,
        )
        assert refined[0] is None
        assert refined[1] is None
        box = refined[2]
        assert box is not None and box["source"] == "scan_vlm"
        assert len(box["segments"]) == 1
        assert all(
            abs(box[k] - box["segments"][0][k]) < 1e-9
            for k in ("x0", "y0", "x1", "y1")
        )
        assert len(calls) == 1  # aligned on the first (radius 1) pass
        assert 0.44 < box["y0"] < box["y1"] < 0.51
        line_frac = _rect_to_fracs(lines[target].rect, page.rect)
        assert abs(box["y0"] - line_frac["y0"]) < 1e-6  # exact target line
        assert abs(box["y1"] - line_frac["y1"]) < 1e-6
        assert box["x0"] >= line_frac["x0"] - 1e-6
        assert box["x1"] <= line_frac["x1"] + 1e-6
        # Word-level snapping: a fraction of the line, not the whole line.
        assert (box["x1"] - box["x0"]) < 0.8 * (line_frac["x1"] - line_frac["x0"])

        # Garbage reader: alignment fails, widens once (radius 2), then gives
        # up -> exactly two reader invocations and a None result.
        garbage_calls: list[int] = []

        def _garbage_reader(strips: list[bytes]) -> list[list[str]]:
            garbage_calls.append(len(strips))
            return [["خط بی ربط"] for _ in strips]

        failed = refine_scan_boxes(
            root / "source.pdf", 8, md, [q], [scan_box], _garbage_reader
        )
        assert failed[0] is None
        assert len(garbage_calls) == 2, garbage_calls
    finally:
        doc.close()


def _check_wrapped_real_case(root: Path) -> None:
    """The reported page-44 RTL wrap yields two narrow refined segments."""
    md = (root / "text" / "0044.md").read_text(encoding="utf-8")
    q = Query("نخ‌های عمودی سایید ه بودند.")
    scan_box = locate_queries(root / "source.pdf", 44, md, [q])[0]
    assert scan_box is not None and scan_box["source"] == "scan"

    doc = fitz.open(root / "source.pdf")
    try:
        page = doc[43]
        lines = _scan_page_lines(page)
        span = _resolve_span(md, q)
        assert span is not None
        hits = _scan_hits(md, span, lines)
        assert len(hits) == 2
        lo = max(0, hits[0][0] - 1)
        hi = min(len(lines) - 1, hits[-1][0] + 1)
        reading = ["میز صندلی" for _ in range(lo, hi + 1)]
        upper_n = len(lines[hits[0][0]].words)
        lower_n = len(lines[hits[1][0]].words)
        assert upper_n >= 2 and lower_n >= 3
        reading[hits[0][0] - lo] = " ".join(
            ["بالا"] * (upper_n - 2) + ["نخ‌های", "عمودی"]
        )
        reading[hits[1][0] - lo] = " ".join(
            ["سایید", "ه", "بودند"] + ["پایین"] * (lower_n - 3)
        )

        def _reader(strips: list[bytes]) -> list[list[str]]:
            return [list(reading) for _ in strips]

        refined = refine_scan_boxes(
            root / "source.pdf", 44, md, [q], [scan_box], _reader
        )[0]
        assert refined is not None and refined["source"] == "scan_vlm"
        segments = refined["segments"]
        assert len(segments) == 2
        assert segments[0]["y0"] < segments[1]["y0"]
        assert segments[0]["x1"] - segments[0]["x0"] < 0.5
        assert segments[1]["x1"] - segments[1]["x0"] < 0.5
        assert refined["x1"] - refined["x0"] > 0.7  # legacy envelope remains wide
    finally:
        doc.close()


def _check_read_strips_live(root: Path) -> None:
    """API-gated live check: one real llm.read_strips call on one real strip
    from page 8, then refine_scan_boxes reusing that reading. Skips (with a
    note) when no API key resolves via llm.load_env()."""
    from farsi2epub import llm
    from farsi2epub.config import MODEL_STRONG

    llm.load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("live read_strips check: SKIPPED (no API key)")
        return

    import numpy as np

    md = (root / "text" / "0008.md").read_text(encoding="utf-8")
    q = Query("بداالله دنبال خر بود")
    scan_box = locate_queries(root / "source.pdf", 8, md, [q])[0]
    assert scan_box is not None and scan_box["source"] == "scan"

    doc = fitz.open(root / "source.pdf")
    try:
        page = doc[7]
        lines = _scan_page_lines(page)
        span = _resolve_span(md, q)
        hits = _scan_hits(md, span, lines)
        lo = max(0, hits[0][0] - 1)
        hi = min(len(lines) - 1, hits[-1][0] + 1)
        median_h = float(np.median([ln.rect.height for ln in lines])) or 1.0
        png = _render_strip(page, _strip_rect(page, lines, lo, hi, median_h))
        assert png is not None

        client = llm.get_client()
        readings, _usage, cost = llm.read_strips(client, [png], MODEL_STRONG, 8)
        assert len(readings) == 1
        assert any(line.strip() for line in readings[0]), readings
        assert cost > 0
        total_cost = cost
        n_calls = 1

        # Reuse the live reading for the identical strip refine renders; only
        # an unexpected (e.g. widened) strip triggers another real call.
        def _reader(strips: list[bytes]) -> list[list[str]]:
            nonlocal total_cost, n_calls
            if all(s == png for s in strips):
                return [list(readings[0]) for _ in strips]
            more, _u, c = llm.read_strips(client, strips, MODEL_STRONG, 8)
            total_cost += c
            n_calls += 1
            return more

        refined = refine_scan_boxes(
            root / "source.pdf", 8, md, [q], [scan_box], _reader
        )[0]
        assert refined is not None, "live refinement returned no box"
        assert refined["source"] == "scan_vlm"
        assert 0.44 < refined["y0"] < 0.51, refined
        print(
            f"live read_strips check: PASSED "
            f"({n_calls} API call(s), ${total_cost:.4f})"
        )
    finally:
        doc.close()


def main() -> int:
    # Presentation-form glyphs must survive normalization. Previously they
    # folded to an empty string, leaving punctuation as the only line token.
    assert _fold_word("ﺧﻮﺍﻫﺮﺑﺮﺍﺩﺭ") == "خواهربرادر"

    # Pure synthetic checks for the VLM strip-alignment helper.
    _check_align_strip_synthetic()
    _check_review_segment_plumbing()

    # This mirrors the problematic PDF's extraction: three visible words are
    # seven fragments, and one fragment has RTL-reversed character order.
    fragments = [
        (fitz.Rect(238, 82, 246, 102), "خ"),
        (fitz.Rect(196, 82, 236, 102), "واهربرادر"),
        (fitz.Rect(168, 82, 194, 102), "یریش"),
        (fitz.Rect(162, 82, 166, 102), "ب"),
        (fitz.Rect(151, 82, 160, 102), "ود"),
        (fitz.Rect(146, 82, 151, 102), "ی"),
        (fitz.Rect(141, 82, 146, 102), "م"),
        (fitz.Rect(136, 82, 141, 102), "،"),
    ]
    box = _locate_match(_Page(), fragments, "خواهربرادر شیری بودیم", None)
    assert box is not None
    assert box["source"] == "match"
    assert abs(box["x0"] - 141 / 420) < 1e-9
    assert abs(box["x1"] - 246 / 420) < 1e-9

    # Real image-only scan regression. This page has zero PDF text words; the
    # phrase appears on the first short body line beneath the opening paragraph.
    root = Path(__file__).resolve().parent.parent / "books" / "bachehaye_ghali"
    if (root / "source.pdf").is_file():
        md = (root / "text" / "0008.md").read_text(encoding="utf-8")
        scan_box = locate_queries(
            root / "source.pdf", 8, md, [Query("بداالله دنبال خر بود")]
        )[0]
        assert scan_box is not None
        assert scan_box["source"] == "scan"
        assert 0.44 < scan_box["y0"] < 0.51
        assert scan_box["x1"] - scan_box["x0"] < 0.60

        # A finding with no verifier bbox still gets deterministic geometry.
        md21 = (root / "text" / "0021.md").read_text(encoding="utf-8")
        no_model_box = locate_queries(
            root / "source.pdf", 21, md21, [Query("کشیدبه طرف کاهدان")]
        )[0]
        assert no_model_box is not None and no_model_box["source"] == "scan"

        # Fake-reader plumbing for the scan_vlm refinement path.
        _check_refine_plumbing(root)
        _check_wrapped_real_case(root)

        # One real strip transcription (API-gated; skips without a key).
        _check_read_strips_live(root)

    # Keep the original digital-PDF failure fixed as the scan tier evolves.
    digital_root = Path(__file__).resolve().parent.parent / "books" / "boof-e-koor"
    if (digital_root / "source.pdf").is_file():
        md46 = (digital_root / "text" / "0046.md").read_text(encoding="utf-8")
        digital_box = locate_queries(
            digital_root / "source.pdf",
            46,
            md46,
            [Query("خواهربرادر شیری بودیم")],
        )[0]
        assert digital_box is not None and digital_box["source"] == "match"
        assert 0.33 < digital_box["x0"] < 0.35
        assert 0.58 < digital_box["x1"] < 0.60
    print("ALL LOCATOR REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
