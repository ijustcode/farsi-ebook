#!/usr/bin/env python3
"""Regression test for the judge's line-alignment fix (JUDGE_FIX.md).

THE BUG. `tests/bbox_score.py` graded a box by mapping the frozen page reading
onto the detected ink lines with a monotone Needleman-Wunsch pass. On
bachehaye_ghali p64 two printed lines fused into ONE 62px row run against a
27px reference, so the detector returned 28 lines where the page prints 29.
The aligner balanced the deficit by DROPPING a reading line in the middle of
the page (L4), which shifted every line between the drop and the merge one
printed line up. Three correctly-placed boxes were then read out against the
wrong ink and graded catastrophically wrong — while `geom_confident` reported
the broken mapping as trustworthy, and was never gated on anyway.

The pixel-verified truth (JUDGE_FIX.md, corroborated from the rendered
overlay) is that all three boxes sit on their target words. So:

  p64:h100001  word_miss_sum == 0
  p64:h100007  word_miss_sum == 0
  p64:h2       word_miss_sum <= 1

The `geom_confident is True` assertion is LOAD-BEARING: a "fix" that merely
marks everything unconfident would satisfy the score assertions by exclusion.
It must not pass.

Everything here is offline and free ($0): it re-grades the frozen page truth.

Run:  ./venv/bin/python tests/bbox_judge_regression.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import fitz  # noqa: E402

import bbox_score  # noqa: E402
from farsi2epub import locate  # noqa: E402
from farsi2epub.workspace import Workspace  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN = PROJECT_ROOT / "tests" / "data" / "bbox_sample240.json"
SLUG = "bachehaye_ghali"
PAGE = 64

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


# ---------------------------------------------------------------------------
# (a) Layer 1 — the tall-line split, at the row-run level
# ---------------------------------------------------------------------------


def section_a() -> None:
    print("\n== (a) Layer 1: over-tall row runs are split ==")

    # A synthetic profile: two 20px lines of ink separated by a 6px near-blank
    # valley, detected as one 46px run.
    prof = np.array([0] * 5 + [50] * 20 + [1] * 6 + [50] * 20 + [0] * 5)
    out = locate._split_tall_runs([(5, 51)], prof, 20.0)
    check(
        "fused pair with a valley is split in two",
        len(out) == 2 and out[0][1] == out[1][0],
        str(out),
    )
    # A drop cap / heading: tall, but solid ink all the way down. No valley, so
    # it must survive intact.
    solid = np.array([0] * 5 + [50] * 46 + [0] * 5)
    check(
        "valley-less tall run is left alone",
        locate._split_tall_runs([(5, 51)], solid, 20.0) == [(5, 51)],
        str(locate._split_tall_runs([(5, 51)], solid, 20.0)),
    )
    # A normal line is never touched.
    check(
        "a normal-height run is untouched",
        locate._split_tall_runs([(5, 25)], prof, 20.0) == [(5, 25)],
    )
    # Depth cap: a 4x run splits at most 3 deep, so it cannot loop forever.
    long_prof = np.array(([50] * 18 + [1] * 4) * 4)
    deep = locate._split_tall_runs([(0, 88)], long_prof, 20.0)
    check("split is depth-capped", 1 < len(deep) <= 8, f"{len(deep)} pieces")


# ---------------------------------------------------------------------------
# (b) the page-64 pairing — the root-cause story itself
# ---------------------------------------------------------------------------


def section_b() -> None:
    print("\n== (b) page-64 reading lines pair with their own printed lines ==")
    ws = Workspace.load(SLUG)
    doc = fitz.open(str(ws.pdf_path))
    page = doc[PAGE - 1]
    page_md = bbox_score._replay_text(ws, PAGE)
    page_lines, _ratio, recall = bbox_score._page_reader_lines(page, page_md)
    prefer_layer = bbox_score._layer_usable(page, page_md, recall)
    check("p64 routes to the scan detector", not prefer_layer, f"recall={recall}")

    scan_lines = locate._scan_page_lines(page)
    det_lines, src = bbox_score._crop_word_lines(
        page_lines, scan_lines, fitz.Rect(page.rect), prefer_layer
    )
    check("geometry source is scan", src == "scan", src)

    hit = bbox_score._PageReadCache(ws).get(
        bbox_score._page_read_key(SLUG, PAGE, bbox_score.MODEL_STRONG)
    )
    if hit is None or not hit.get("reading"):
        check("frozen page reading present", False, "no locate_page_read.json entry")
        doc.close()
        return
    vwords = [
        locate._norm_words(t or "")
        for t in (hit["reading"].get("line_text_full") or [])
    ]
    pairing = (
        list(range(len(vwords)))
        if len(vwords) == len(det_lines)
        else bbox_score._align_line_sequences(vwords, det_lines)
    )
    # The reading has 30 lines; the last is the printed folio "63", which the
    # Markdown omits, so 29 detected lines is the correct count.
    check("p64 detects 29 ink lines", len(det_lines) == 29, str(len(det_lines)))
    check(
        "no reading line is dropped before the folio",
        pairing is not None and all(p is not None for p in pairing[:-1]),
        str(pairing),
    )
    # The spec's falsification check: these two pairings are the bug.
    check("L6 -> det6", pairing[6] == 6, str(pairing[6]))
    check("L7 -> det7", pairing[7] == 7, str(pairing[7]))
    doc.close()


# ---------------------------------------------------------------------------
# (c) the three pixel-verified cases, through cmd_score's own path
# ---------------------------------------------------------------------------

EXPECTED = {
    f"{SLUG}:p{PAGE}:h100001": 0,
    f"{SLUG}:p{PAGE}:h100007": 0,
    f"{SLUG}:p{PAGE}:h2": 1,
}


def section_c() -> None:
    print("\n== (c) the three pixel-verified p64 cases ==")
    tmp = Path(tempfile.mkdtemp(prefix="bbox_judge_regression_")) / "p64.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "bbox_score.py"),
        "score",
        "--cases", str(FROZEN),
        "--books", SLUG,
        "--pages", str(PAGE),
        "--out", str(tmp),
    ]
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.is_file():
        check(
            "p64 re-score ran",
            False,
            f"rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}",
        )
        return
    report = json.loads(tmp.read_text(encoding="utf-8"))
    check("p64 re-score ran", True)
    check(
        "cost was $0",
        float(report["run"].get("cost_usd") or 0.0) == 0.0,
        str(report["run"].get("cost_usd")),
    )
    rows = {r["id"]: r for r in report["cases"]}
    for cid, limit in EXPECTED.items():
        r = rows.get(cid)
        if r is None:
            check(f"{cid} present", False, "case not in the report")
            continue
        check(
            f"{cid} degenerate is None",
            r.get("degenerate") is None,
            str(r.get("degenerate")),
        )
        wms = r.get("word_miss_sum")
        check(
            f"{cid} word_miss_sum <= {limit}",
            wms is not None and wms <= limit,
            str(wms),
        )
        # LOAD-BEARING: a fix by exclusion must not pass.
        check(
            f"{cid} geom_confident is True",
            r.get("geom_confident") is True,
            str(r.get("geom_confident")),
        )



# ---------------------------------------------------------------------------
# (f) geometry-source routing — pixel-verified boof-e-koor cases
#
# A layer too unreliable to READ can still be perfectly POSITIONED. Every
# sampled boof-e-koor page folds against its Markdown at raw recall 0.115-0.264
# (threshold 0.45), so the old single probe forced all 34 onto scan geometry,
# which found nothing inside the box. Ten correctly placed boxes were reported
# as `no_covered_words`. Verified from the page pixels: the box on p20:h1 sits
# exactly on میلولیدند, its own target.
# ---------------------------------------------------------------------------

ROUTING_SLUG = "boof-e-koor"
ROUTING_EXPECTED = {
    f"{ROUTING_SLUG}:p14:h0": 0,   # box sits exactly on راروشن
    f"{ROUTING_SLUG}:p20:h1": 0,   # box sits exactly on میلولیدند
    f"{ROUTING_SLUG}:p56:h0": 1,   # correctly on a printed word; کوه recurs
}


def section_f() -> None:
    print("\n== (f) geometry-source routing (boof-e-koor) ==")
    doc = fitz.open(PROJECT_ROOT / "books" / ROUTING_SLUG / "source.pdf")
    try:
        page = doc[19]  # p20
        check(
            "a real Unicode layer is geometry-usable",
            bbox_score._layer_geometry_usable(page) is True,
            "boof-e-koor p20",
        )
    finally:
        doc.close()
    # The two failure modes the scan fallback exists for must still take it.
    for slug, pno, label in (
        ("bachehaye_ghali", 63, "image scan"),
        ("haaji-agha", 29, "glyph cipher"),
    ):
        d2 = fitz.open(PROJECT_ROOT / "books" / slug / "source.pdf")
        try:
            check(
                f"{label} is NOT geometry-usable ({slug})",
                bbox_score._layer_geometry_usable(d2[pno]) is False,
            )
        finally:
            d2.close()

    for cid, limit in ROUTING_EXPECTED.items():
        page_no = int(cid.split(":p")[1].split(":")[0])
        tmp = Path(tempfile.mkdtemp(prefix="bbox_routing_")) / "r.json"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "tests" / "bbox_score.py"), "score",
            "--cases", str(FROZEN), "--books", ROUTING_SLUG,
            "--pages", str(page_no), "--out", str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.is_file():
            check(f"{cid} re-score ran", False, f"rc={proc.returncode}")
            continue
        rows = {r["id"]: r for r in json.loads(tmp.read_text(encoding="utf-8"))["cases"]}
        r = rows.get(cid)
        if r is None:
            check(f"{cid} present", False, "case not in the report")
            continue
        check(f"{cid} degenerate is None", r.get("degenerate") is None,
              str(r.get("degenerate")))
        wms = r.get("word_miss_sum")
        check(f"{cid} word_miss_sum <= {limit}", wms is not None and wms <= limit,
              str(wms))

# ---------------------------------------------------------------------------
# (d) Layer 2 — drift latches positionally, and it gates
# ---------------------------------------------------------------------------


def section_d() -> None:
    print("\n== (d) Layer 2: drift latch and gating ==")
    R = fitz.Rect

    def line(n: int, y: float) -> list[fitz.Rect]:
        return [R(10 * i, y, 10 * i + 8, y + 8) for i in range(n)]

    # Four reading lines, four detected lines: no drift anywhere.
    lines = ["a b c", "d e f", "g h i", "j k l"]
    al = bbox_score._assign_word_rects(lines, [line(3, y) for y in (0, 10, 20, 30)])
    check(
        "clean pairing has no drift",
        al.drift_lines == 0 and al.drift_start is None and all(al.word_conf),
        f"drift_lines={al.drift_lines} start={al.drift_start}",
    )

    # A dropped LAST line (the printed folio, which the Markdown omits) must
    # not taint the words above it — this is the bachehaye_ghali common case.
    folio = ["a b c", "d e f", "g h i", "63"]
    al = bbox_score._assign_word_rects(folio, [line(3, y) for y in (0, 10, 20)])
    tail_ok = al.drift_start is None or al.drift_start >= 9
    check(
        "a drop on the last line does not taint the page",
        tail_ok and all(al.word_conf[:9]),
        f"start={al.drift_start} conf={al.word_conf}",
    )

    # A dropped MIDDLE line taints everything after it and nothing before it.
    al = bbox_score._assign_word_rects(
        ["a b c", "d e f", "g h i"],
        [line(3, 0), line(3, 20)],  # the reading's middle line has no ink line
    )
    if al.drift_start is None:
        check("a mid-page drop latches drift", False, "no drift recorded")
    else:
        check(
            "a mid-page drop latches drift",
            0 < al.drift_start < len(al.word_conf),
            f"start={al.drift_start}",
        )
        check(
            "words before the drop stay confident",
            all(al.word_conf[: al.drift_start]),
            str(al.word_conf),
        )
        check(
            "words after the drop are unconfident",
            not any(al.word_conf[al.drift_start :]),
            str(al.word_conf),
        )

    check(
        "geom_unconfident is routed as NEUTRAL, not penalized",
        "geom_unconfident" in bbox_score._NEUTRAL_DEGENERATE
        and "geom_unconfident" not in bbox_score._PENALIZED_DEGENERATE
        and "geom_unconfident" not in bbox_score._EXCLUDED_VERDICTS,
    )


# ---------------------------------------------------------------------------
# (e) the exclusion bucket must not become a hiding place
# ---------------------------------------------------------------------------

BASELINE = PROJECT_ROOT / "out" / "score_word_miss_baseline_v3.json"


def section_e() -> None:
    print("\n== (e) geom_unconfident stays a rare, gated bucket ==")
    if not BASELINE.is_file():
        check("v2 baseline present", False, f"missing {BASELINE}")
        return
    o = json.loads(BASELINE.read_text(encoding="utf-8"))["overall"]
    rate = o.get("geom_unconfident_rate")
    check(
        "geom_unconfident_rate <= 3% on the frozen sample",
        rate is not None and rate <= 0.03,
        str(rate),
    )
    check(
        "perfect_rate is computed on the FULL graded denominator",
        o["word_miss_n"] <= o["n_denominator"],
        f"measured={o['word_miss_n']} denominator={o['n_denominator']}",
    )


def main() -> int:
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} check(s): " + ", ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
