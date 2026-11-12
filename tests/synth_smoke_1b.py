#!/usr/bin/env python3
"""Smoke test 1b (plan §7 Step 1): sim-to-real sanity of the degradations.

Runs the CURRENT production line detector, ``farsi2epub.locate._scan_page_lines``,
on synthetic pages at increasing warp, wrapping each generated image in a
one-page A4 PDF exactly like a scanned page. Reported per level: detected vs
true lines, tallest detected block (PDF units), number of merged blocks
(detected blocks containing >= 2 truth lines, by warped baseline midpoint),
"tall" blocks (> 1.6x the detector's median block height on the clean page;
a single sloped line also gets tall, so tall != merged) and the most truth
lines inside one block.

Expected (plan E9/E17): correct count on clean pages; merged multi-line blocks
once the slope exceeds ~1 deg, as on books/hadpi-articles (p2: 22 detected
"lines", blocks up to 193 units vs ~12 for a text line). The real hadpi pages
are run alongside for comparison (read-only; nothing under books/ is written).

PASS = the clean page is detected correctly AND the detector merges lines on
the warped synthetic pages from ~1 deg on, like the real pages.

Run: ANTHROPIC_BASE_URL=http://127.0.0.1:9 ./venv/bin/python tests/synth_smoke_1b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import synth_pages as S  # noqa: E402
from farsi2epub import locate  # noqa: E402

LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.3]
MERGE_FACTOR = 1.6


def detect(u8: np.ndarray):
    H, W = u8.shape
    doc = fitz.open()
    pg = doc.new_page(width=S.PAGE_W, height=S.PAGE_H)
    pg.insert_image(pg.rect, pixmap=fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(u8).tobytes(), False))
    lines = locate._scan_page_lines(pg)
    doc.close()
    return lines


def summarise(lines, truth: dict, ref_h: float | None) -> dict:
    hs = [ln.rect.height for ln in lines]
    tl = truth["lines"]
    per_block = []
    for ln in lines:
        r = ln.rect
        n = 0
        for t in tl:
            pts = t["baseline"]
            x, y = pts[len(pts) // 2]
            if r.x0 <= x * S.PAGE_W <= r.x1 and r.y0 <= y * S.PAGE_H <= r.y1 + 2:
                n += 1
        per_block.append(n)
    ref = ref_h or (float(np.median(hs)) if hs else 0.0)
    return dict(detected=len(lines), true=len(tl), tallest=round(max(hs), 1) if hs else 0.0,
                median_h=round(float(np.median(hs)), 1) if hs else 0.0,
                tall_blocks=sum(1 for h in hs if h > MERGE_FACTOR * ref),
                merged_blocks=sum(1 for n in per_block if n >= 2),
                max_lines_in_block=max(per_block) if per_block else 0,
                lines_in_merged=sum(n for n in per_block if n >= 2))


def run_font(cls: str, rows: list, overlay_photo: bool = True) -> None:
    base = dict(size=12.5, leading=1.75)
    ref_h = None
    for mode in ("ramp", "rotate"):
        for lv in LEVELS:
            if mode == "rotate" and lv == 0.0:
                continue
            deg = ({"warp_top_deg": 0.1 if lv else 0.0, "warp_bottom_deg": lv} if mode == "ramp"
                   else {"rotate_deg": lv})
            spec = S.make_spec(cls, layout="articles", preset="clean", seed=5,
                               name=f"1b_{cls}_{mode}_{lv}", degrade=deg, **base)
            res = S.generate(spec)
            s = summarise(detect(res["image"]), res["truth"], ref_h)
            if mode == "ramp" and lv == 0.0:
                ref_h = s["median_h"]
                s = summarise(detect(res["image"]), res["truth"], ref_h)
            rows.append(dict(font_class=cls, mode=mode, slope_bottom_deg=lv, **s))
            S.save_page(res, S.OUT_DIR / "1b")
    # E18's measured hadpi p2 profile (0.1 / 2.2 / 3.3 deg, top/middle/bottom),
    # geometry only.
    spec = S.make_spec(cls, layout="articles", preset="clean", seed=5, name=f"1b_{cls}_e18",
                       degrade={"warp_top_deg": 0.1, "warp_mid_deg": 2.2, "warp_bottom_deg": 3.3}, **base)
    res = S.generate(spec)
    rows.append(dict(font_class=cls, mode="e18(0.1/2.2/3.3)", slope_bottom_deg=3.3,
                     **summarise(detect(res["image"]), res["truth"], ref_h)))
    S.save_page(res, S.OUT_DIR / "1b")
    if overlay_photo:
        spec = S.make_spec(cls, layout="articles", preset="photo", seed=5, name=f"1b_{cls}_photo", **base)
        res = S.generate(spec)
        rows.append(dict(font_class=cls, mode="photo(e18+all)", slope_bottom_deg=3.3,
                         **summarise(detect(res["image"]), res["truth"], ref_h)))
        S.save_page(res, S.OUT_DIR / "1b")
    if cls == "typewriter":
        # pipeline v2 presets (memo layout, invented pool text, their own
        # leading); reported beside the old photo row, not part of the verdict
        for preset, bottom in (("tw_photo_medium", 3.3), ("tw_photo_hard", 3.6)):
            for sd in (5, 6):
                spec = S.make_spec(cls, preset=preset, seed=sd, name=f"1b_{cls}_{preset}_s{sd}")
                res = S.generate(spec)
                rows.append(dict(font_class=cls, mode=f"{preset}_s{sd}", slope_bottom_deg=bottom,
                                 **summarise(detect(res["image"]), res["truth"], ref_h)))
                S.save_page(res, S.OUT_DIR / "1b")


REAL_PAGES = (1, 2)   # hadpi p3-p5 are held out (plan §6.2): never read here


def real_hadpi() -> list:
    """Same numbers on the real photographed pages p1-p2 (read-only)."""
    pdf = ROOT / "books" / "hadpi-articles" / "source.pdf"
    out = []
    if not pdf.exists():
        return out
    with fitz.open(str(pdf)) as d:
        for pn in [p - 1 for p in REAL_PAGES if p <= len(d)]:
            L = locate._scan_page_lines(d[pn])
            hs = sorted(ln.rect.height for ln in L)
            out.append(dict(page=pn + 1, detected=len(L), tallest=round(hs[-1], 1),
                            median_h=round(float(np.median(hs)), 1),
                            blocks_over_30=sum(1 for h in hs if h > 30)))
    return out


def overlay_png(cls: str, mode_lv: str) -> None:
    """Draw detector blocks on one synthetic page for eyeballing."""
    png = S.OUT_DIR / "1b" / f"1b_{cls}_{mode_lv}.png"
    if not png.exists():
        return
    doc = fitz.open()
    pg = doc.new_page(width=S.PAGE_W, height=S.PAGE_H)
    pg.insert_image(pg.rect, filename=str(png))
    for ln in locate._scan_page_lines(pg):
        pg.draw_rect(ln.rect, color=(1, 0, 0), width=0.8)
    pg.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(str(S.OUT_DIR / f"detector_1b_{cls}_{mode_lv}.png"))
    doc.close()


def main() -> int:
    rows: list = []
    for cls in ("modern_naskh", "typewriter"):
        if S.resolve_class_font(cls):
            run_font(cls, rows)
    print(f"{'class':13s} {'mode':22s} {'slope':>5s} {'det':>4s} {'true':>4s} {'tallest':>7s} "
          f"{'median':>6s} {'tall':>4s} {'merged':>6s} {'maxL/blk':>8s} {'inMerged':>8s}")
    for r in rows:
        print(f"{r['font_class']:13s} {r['mode']:22s} {r['slope_bottom_deg']:5.1f} {r['detected']:4d} {r['true']:4d} "
              f"{r['tallest']:7.1f} {r['median_h']:6.1f} {r['tall_blocks']:4d} {r['merged_blocks']:6d} {r['max_lines_in_block']:8d} {r['lines_in_merged']:8d}")
    real = real_hadpi()
    if real:
        print("\nreal books/hadpi-articles (same detector):")
        for r in real:
            print(f"  p{r['page']}: detected={r['detected']:3d} tallest={r['tallest']:6.1f} "
                  f"median={r['median_h']:5.1f} blocks>30pt={r['blocks_over_30']}")
    for cls in ("modern_naskh", "typewriter"):
        for m in ("ramp_0.0", "ramp_3.3", "rotate_2.0", "e18", "photo", "tw_photo_hard_s5"):
            overlay_png(cls, m)
    # Verdict, three separate criteria (all reported):
    #  (a) clean page: detected == true, no multi-line block;
    #  (b) under E18's measured hadpi p2 slope profile the synthetic page also
    #      produces multi-line blocks (the real page does: E9/E17);
    #  (c) onset: a uniform slope of 1.5 deg already produces multi-line
    #      blocks (the brief's "once slope exceeds roughly 1 deg").
    crit = {}
    for cls in sorted({r["font_class"] for r in rows}):
        mine = [r for r in rows if r["font_class"] == cls]
        clean = next(r for r in mine if r["mode"] == "ramp" and r["slope_bottom_deg"] == 0.0)
        e18 = next(r for r in mine if r["mode"].startswith("e18"))
        on = next(r for r in mine if r["mode"] == "rotate" and r["slope_bottom_deg"] == 1.5)
        crit[cls] = dict(a_clean=clean["detected"] == clean["true"] and clean["merged_blocks"] == 0,
                         b_e18_merges=e18["merged_blocks"] > 0,
                         c_onset_1p5deg=on["merged_blocks"] > 0)
        print(f"  {cls}: " + ", ".join(f"{k}={'ok' if v else 'NO'}" for k, v in crit[cls].items()))
    ok = all(all(c.values()) for c in crit.values())
    (S.OUT_DIR / "smoke_1b.json").write_text(json.dumps(dict(rows=rows, real_hadpi=real, criteria=crit), indent=1), encoding="utf-8")
    print(f"\nreport: {S.OUT_DIR / 'smoke_1b.json'}")
    print("SMOKE 1b:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
