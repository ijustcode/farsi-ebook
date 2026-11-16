#!/usr/bin/env python3
"""Regression checks for `hard_reader.line_rows` (plan
docs/line-finder-integration-plan.md 3.2). Synthetic pages only; no books/,
no API.

  ./venv/bin/python tests/line_rows_regression.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from farsi2epub import hard_reader as HR  # noqa: E402


def synth_page(H=3000, W=1800, n_lines=34, pitch=78, top=140, seed=7, slope=0.0):
    """White page with rows of glyph-like dark blobs; returns (gray, [(y0, y1)])
    with each line's exact ink rows (y1 exclusive)."""
    rng = np.random.default_rng(seed)
    g = np.full((H, W), 245.0, np.float32)
    spans = []
    for i in range(n_lines):
        base = top + i * pitch
        lo, hi = H, 0
        x = W - 150
        x_end = 150 if i % 7 != 6 else W // 2          # a short last line of a "paragraph" now and then
        while x > x_end:
            wl = int(rng.integers(3, 8))                 # letters per word
            for _ in range(wl):
                w = int(rng.integers(6, 16))
                h = int(rng.integers(14, 30))
                dy = int(slope * (W - x))
                y1 = base + dy + int(rng.integers(0, 4))
                y0 = y1 - h
                if rng.random() < 0.2:                   # descender
                    y1 += int(rng.integers(4, 10))
                g[y0:y1, x - w:x] = 30.0
                lo, hi = min(lo, y0), max(hi, y1)
                x -= w + int(rng.integers(1, 3))
            x -= int(rng.integers(14, 26))               # word gap
        spans.append((lo, hi))
    return g, spans


def covers(rows, H):
    return (bool(rows) and rows[0][0] == 0 and rows[-1][1] == H
            and all(rows[i - 1][0] < rows[i][0] <= rows[i - 1][1] < rows[i][1] for i in range(1, len(rows))))


def check(name, cond):
    print(f"  {'ok ' if cond else 'FAIL'} {name}")
    if not cond:
        check.failed += 1


check.failed = 0


def main() -> int:
    g, spans = synth_page()
    H, W = g.shape
    fixed = HR.band_rows(H, W)
    check("fixed bands cover the page", covers(fixed, H))
    for lpb in (1, 3, "same", 10):
        rows, info = HR.line_rows(g, lpb)
        check(f"lpb={lpb}: line cut used (reason {info['reason']})", info["cut"] == "lines")
        check(f"lpb={lpb}: bands cover every row in order", covers(rows, H))
        check(f"lpb={lpb}: no band over the no-resize size", all(HR._band_fits(b - a, W) for a, b in rows))
        cuts = {y for a, b in rows for y in (a, b) if 0 < y < H}
        check(f"lpb={lpb}: no cut inside a line's ink", not any(lo < y < hi for y in cuts for lo, hi in spans))
        check(f"lpb={lpb}: every line whole inside a band", all(any(a <= lo and hi <= b for a, b in rows)
                                                               for lo, hi in spans))
        rows2, info2 = HR.line_rows(g, lpb)
        check(f"lpb={lpb}: deterministic", rows == rows2 and info2["cut"] == info["cut"])
        if lpb == "same":
            check(f"lpb=same: band count equals the fixed cut ({len(rows)} vs {len(fixed)})", len(rows) == len(fixed))
        if lpb == 1:
            # the first and last line's bands lie inside their neighbour's (which reaches the page edge) and are dropped
            check(f"lpb=1: one band per found text line but the first and last ({len(rows)} vs {info['text_lines']})",
                  len(rows) == info["text_lines"] - 2 and info.get("dropped_contained") == 2)

    # rule 3: each band holds its lines plus one whole neighbour above and below
    rows, info = HR.line_rows(g, 3)
    inside = [[i for i, (lo, hi) in enumerate(spans) if a <= lo and hi <= b] for a, b in rows]
    check("lpb=3: inner bands hold 3 lines + 1 neighbour each side (5 whole lines)",
          all(len(x) == 5 for x in inside[1:-1]))

    # a sloped page: still no cut through ink
    gs, ss = synth_page(seed=11, slope=0.012)
    rows, info = HR.line_rows(gs, 3)
    cuts = {y for a, b in rows for y in (a, b) if 0 < y < gs.shape[0]}
    check(f"sloped page: cut={info['cut']}, covers, no cut through ink",
          covers(rows, gs.shape[0]) and not any(lo < y < hi for y in cuts for lo, hi in ss))

    # size rule: a page much taller than one band allows -> bands split, all fit
    gt, st = synth_page(H=7000, n_lines=86, seed=3)
    rows, info = HR.line_rows(gt, 40)
    check(f"tall page, 40 lines a band: split ({info['splits']}), all fit, covers",
          info["cut"] == "lines" and info["splits"] > 0 and covers(rows, 7000)
          and all(HR._band_fits(b - a, gt.shape[1]) for a, b in rows))

    # fallbacks
    blank = np.full((2600, 1800), 250.0, np.float32)
    rows, info = HR.line_rows(blank, 3)
    check(f"blank page falls back to fixed bands ({info['reason']})",
          info["cut"] == "fixed" and rows == HR.band_rows(2600, 1800) and info["reason"])

    def boom(_g):
        raise RuntimeError("finder broke")
    rows, info = HR.line_rows(g, 3, finder=boom)
    check("finder exception falls back to fixed bands", info["cut"] == "fixed" and rows == fixed
          and info["reason"].startswith("finder error"))

    two = synth_page(H=1200, n_lines=2, seed=5)[0]
    rows, info = HR.line_rows(two, 1)
    check("two text lines fall back (fewer than 3)", info["cut"] == "fixed" and "fewer than 3" in info["reason"])

    rows, info = HR.line_rows(np.zeros((0, 0)), 3)
    check("empty array does not raise", isinstance(rows, list) and info["cut"] == "fixed")
    rows, info = HR.line_rows(g, 0)
    check("lines_per_band 0 falls back", info["cut"] == "fixed" and rows == fixed)
    rows, info = HR.line_rows(np.dstack([g, g, g]), 3)
    check("RGB input is accepted", info["cut"] == "lines" and covers(rows, H))

    # production path does not use it
    src = inspect.getsource(HR.build_request) + inspect.getsource(HR.band_images) + inspect.getsource(HR.read_page)
    check("build_request / band_images / read_page do not call line_rows", "line_rows" not in src)
    for f in ("transcribe.py", "cli.py"):
        check(f"farsi2epub/{f} does not reference line_rows or line_finder",
              not any(s in (ROOT / "farsi2epub" / f).read_text(encoding="utf-8") for s in ("line_rows", "line_finder")))

    print(f"\n{'FAILED: ' + str(check.failed) if check.failed else 'all line_rows checks passed'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
