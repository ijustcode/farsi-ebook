#!/usr/bin/env python3
"""Smoke test 1a (plan §7 Step 1): synthetic feasibility.

For every font class with a usable local font, generate a clean page and a
degraded ("photo") page and check that the truth is right:

* every laid-out word has a truth rectangle;
* word-ink agreement, measured against the per-pixel word-id ink map (the
  identity render, see tests/synth_pages.py): a word AGREES when
    - cov2  >= COV_MIN : share of the word's own ink inside its rectangle
                         expanded by EXPAND_PX (anti-aliasing tolerance), and
    - swallow <= SWALLOW_MAX : largest share of the word's ink that falls
                         inside any OTHER word's (unexpanded) rectangle;
  cov_eroded (rectangle shrunk by EXPAND_PX) is reported to show tightness;
* image consistency: dark pixels of the actual page image are explained by
  word ink (or by drawn non-text ink);
* reading order: word ids (logical order) never go back up a line, and within
  a printed line run right-to-left except inside embedded LTR runs, where they
  run left-to-right.

PASS = >= 99% of words agree, per class, on the clean page. The degraded page
is checked the same way with the warped id map against the warped polygons
(geometric consistency of the truth under the warp).

Writes out/synth/smoke_1a.json and contact sheets out/synth/contact_1a_<class>.png
(clean left, degraded right, truth drawn) plus out/synth/contact_1a_index.png.

Run: ANTHROPIC_BASE_URL=http://127.0.0.1:9 ./venv/bin/python tests/synth_smoke_1a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_pages as S  # noqa: E402

COV_MIN = 0.95
SWALLOW_MAX = 0.10
EXPAND_PX = 2
PASS_FRAC = 0.99
SEED = 11


def poly_mask(poly: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Even-odd point-in-polygon for pixel centres of [x0,x1) x [y0,y1)."""
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    xs += 0.5; ys += 0.5
    inside = np.zeros(xs.shape, bool)
    n = len(poly)
    for i in range(n):
        xa, ya = poly[i]; xb, yb = poly[(i + 1) % n]
        if ya == yb:
            continue
        cond = (ya > ys) != (yb > ys)
        xint = xa + (ys - ya) * (xb - xa) / (yb - ya)
        inside ^= cond & (xs < xint)
    return inside


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        o = out.copy()
        o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        out = o
    return out


def _ink_index(idmap: np.ndarray, ntok: int):
    """token -> (ys, xs) of its ink pixels."""
    ys, xs = np.nonzero(idmap)
    ids = idmap[ys, xs] - 1
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    out = {}
    if len(ids):
        cuts = np.flatnonzero(np.diff(ids)) + 1
        for a, b, c in zip(np.split(ids, cuts), np.split(ys, cuts), np.split(xs, cuts)):
            out[int(a[0])] = (b, c)
    return out


def check_clean(res: dict) -> dict:
    t = res["truth"]
    idmap = res["idmap_clean"]
    H, W = idmap.shape
    words = t["words"]
    ink = _ink_index(idmap, len(words))
    rects = {}
    for w in words:
        if not w["missing"]:
            rects[w["id"]] = [[v * S.ZOOM for v in sg["layout_rect_pt"]] for sg in w["segments"]]
    per = []
    for w in words:
        wid = w["id"]
        row = dict(id=wid, text=w["text"], line=w["line"], kind=w["kind"])
        if w["missing"] or wid not in ink:
            row.update(agree=False, reason="no rectangle" if w["missing"] else "no ink")
            per.append(row); continue
        ys, xs = ink[wid]
        n = len(xs)
        cx, cy = xs + 0.5, ys + 0.5
        e = EXPAND_PX
        inexp = np.zeros(n, bool); iner = np.zeros(n, bool)
        for x0, y0, x1, y1 in rects[wid]:
            inexp |= (cx >= x0 - e) & (cx <= x1 + e) & (cy >= y0 - e) & (cy <= y1 + e)
            iner |= (cx >= x0 + e) & (cx <= x1 - e) & (cy >= y0 + e) & (cy <= y1 - e)
        sw, sw_by = 0.0, None
        bx0, bx1, by0, by1 = cx.min(), cx.max(), cy.min(), cy.max()
        for vid, segs in rects.items():
            if vid == wid:
                continue
            hit = np.zeros(n, bool)
            for u0, v0, u1, v1 in segs:
                if u1 < bx0 or u0 > bx1 or v1 < by0 or v0 > by1:
                    continue
                hit |= (cx >= u0) & (cx <= u1) & (cy >= v0) & (cy <= v1)
            f = float(hit.sum()) / n
            if f > sw:
                sw, sw_by = f, vid
        cov2, cove = float(inexp.sum()) / n, float(iner.sum()) / n
        ok = cov2 >= COV_MIN and sw <= SWALLOW_MAX
        row.update(agree=ok, n_ink=n, cov2=round(cov2, 4), cov_eroded=round(cove, 4),
                   swallow=round(sw, 4), swallowed_by=sw_by)
        if not ok:
            row["reason"] = "ink outside rectangle" if cov2 < COV_MIN else f"swallowed by word {sw_by}"
        per.append(row)
    # Image consistency: dark pixels explained by word ink or drawn non-text ink.
    img = res["clean"]
    dark = img < 128
    expl = _dilate(idmap > 0, 2)
    nontext = np.zeros_like(dark)
    for o in t["overlays"]:
        a = o["aabb"]
        nontext[int(a[1] * H) - 3:int(a[3] * H) + 3, int(a[0] * W) - 3:int(a[2] * W) + 3] = True
    unexplained = dark & ~expl & ~nontext
    n_ok = sum(1 for r in per if r["agree"])
    return dict(n_words=len(words), n_agree=n_ok, agree_frac=n_ok / max(1, len(words)),
                mean_cov2=float(np.mean([r["cov2"] for r in per if "cov2" in r])),
                mean_cov_eroded=float(np.mean([r["cov_eroded"] for r in per if "cov_eroded" in r])),
                image_dark_px=int(dark.sum()), image_dark_unexplained=int(unexplained.sum()),
                failures=[r for r in per if not r["agree"]][:25])


def check_order(t: dict) -> dict:
    words = {w["id"]: w for w in t["words"] if not w["missing"]}
    viol = []
    ids = sorted(words)
    for a, b in zip(ids, ids[1:]):
        wa, wb = words[a], words[b]
        if wb["line"] < wa["line"]:
            viol.append(dict(a=a, b=b, why="line went backwards"))
            continue
        if wb["line"] != wa["line"] or wa["block"] != wb["block"]:
            continue
        ma = max(wa["segments"], key=lambda g: g["layout_rect_pt"][2] - g["layout_rect_pt"][0])["layout_rect_pt"]
        mb = max(wb["segments"], key=lambda g: g["layout_rect_pt"][2] - g["layout_rect_pt"][0])["layout_rect_pt"]
        xa, xb = (ma[0] + ma[2]) / 2, (mb[0] + mb[2]) / 2
        if wa["ltr"] and wb["ltr"]:
            if not xb > xa:
                viol.append(dict(a=a, b=b, ta=wa["text"], tb=wb["text"], why="LTR run not left-to-right"))
        elif not xb < xa:
            viol.append(dict(a=a, b=b, ta=wa["text"], tb=wb["text"], why="not right-to-left"))
    return dict(n_pairs=len(ids) - 1, n_violations=len(viol), violations=viol[:20])


def check_degraded(res: dict) -> dict:
    """Warped word ink (id map through the same transform, nearest) vs the
    warped truth polygon (expanded by EXPAND_PX via a dilated mask)."""
    t = res["truth"]
    idmap = res["idmap"]
    H, W = idmap.shape
    ink = _ink_index(idmap, len(t["words"]))
    n_ok, n, worst = 0, 0, []
    covs = []
    for w in t["words"]:
        if w["missing"]:
            continue
        n += 1
        if w["id"] not in ink:
            worst.append((0.0, w["id"], w["text"])); continue
        ys, xs = ink[w["id"]]
        polys = [np.array(g["polygon"]) * [W, H] for g in w["segments"]]
        allp = np.vstack(polys)
        x0 = max(0, int(min(allp[:, 0].min(), xs.min())) - 4); x1 = min(W, int(max(allp[:, 0].max(), xs.max())) + 5)
        y0 = max(0, int(min(allp[:, 1].min(), ys.min())) - 4); y1 = min(H, int(max(allp[:, 1].max(), ys.max())) + 5)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        for poly in polys:
            m |= poly_mask(poly, x0, y0, x1, y1)
        m = _dilate(m, EXPAND_PX)
        cov = float(m[ys - y0, xs - x0].sum()) / len(xs)
        covs.append(cov)
        if cov >= COV_MIN:
            n_ok += 1
        else:
            worst.append((round(cov, 3), w["id"], w["text"]))
    worst.sort()
    return dict(n_words=n, n_ok=n_ok, frac=n_ok / max(1, n), mean_cov=float(np.mean(covs)) if covs else 0.0,
                worst=worst[:10])


def contact_sheet(clean_res: dict, deg_res: dict, path: Path, title: str, scale: float = 0.5) -> None:
    ct, dt = clean_res["truth"], deg_res["truth"]
    H, W = clean_res["image"].shape
    gap = 30
    doc = fitz.open()
    pg = doc.new_page(width=2 * W + gap, height=H + 60)
    for k, (res, t) in enumerate(((clean_res, ct), (deg_res, dt))):
        u8 = res["image"]
        pm = fitz.Pixmap(fitz.csGRAY, u8.shape[1], u8.shape[0], np.ascontiguousarray(u8).tobytes(), False)
        ox = k * (W + gap)
        pg.insert_image(fitz.Rect(ox, 60, ox + W, 60 + H), pixmap=pm)
        for w in t["words"]:
            if w["missing"]:
                continue
            for g in w["segments"]:
                pts = [fitz.Point(ox + x * W, 60 + y * H) for x, y in g["polygon"]]
                pg.draw_polyline(pts + [pts[0]], color=(0.1, 0.3, 1.0) if k == 0 else (0.9, 0.1, 0.1), width=1.6)
        for ln in t["lines"]:
            pts = [fitz.Point(ox + x * W, 60 + y * H) for x, y in ln["baseline"]]
            pg.draw_polyline(pts, color=(0.0, 0.6, 0.0), width=1.4)
    pg.insert_text((10, 42), f"{title}   left: clean (blue = word truth)   right: degraded 'photo' "
                             f"(red = warped word truth)   green = baselines", fontsize=30)
    pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(str(path))
    doc.close()


def zoom_crop(res: dict, path: Path, frac=(0.45, 0.06, 1.0, 0.30), boxes: bool = True) -> None:
    """Full-resolution crop with truth drawn, for checking shaping by eye."""
    t = res["truth"]
    u8 = res["image"]
    H, W = u8.shape
    x0, y0, x1, y1 = int(frac[0] * W), int(frac[1] * H), int(frac[2] * W), int(frac[3] * H)
    crop = np.ascontiguousarray(u8[y0:y1, x0:x1])
    doc = fitz.open()
    pg = doc.new_page(width=x1 - x0, height=y1 - y0)
    pg.insert_image(pg.rect, pixmap=fitz.Pixmap(fitz.csGRAY, x1 - x0, y1 - y0, crop.tobytes(), False))
    if boxes:
        for w in t["words"]:
            if w["missing"]:
                continue
            for g in w["segments"]:
                pts = [fitz.Point(x * W - x0, y * H - y0) for x, y in g["polygon"]]
                pg.draw_polyline(pts + [pts[0]], color=(0.9, 0.1, 0.1), width=1.0)
    pg.get_pixmap(alpha=False).save(str(path))
    doc.close()


def main() -> int:
    out = S.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    report = dict(thresholds=dict(cov_min=COV_MIN, swallow_max=SWALLOW_MAX, expand_px=EXPAND_PX,
                                  pass_frac=PASS_FRAC), classes={}, fonts={}, layouts={}, gaps=[])
    thumbs = []
    all_pass = True
    print(f"{'class':13s} {'font':28s} {'words':>5s} {'agree%':>7s} {'order':>5s} "
          f"{'cov2':>6s} {'covE':>6s} {'unexpl_px':>9s} {'degr%':>6s}  verdict")
    for cls in S.FONT_CANDIDATES:
        found = S.resolve_class_font(cls)
        if found is None:
            report["gaps"].append(cls)
            print(f"{cls:13s} GAP: no local font with all Persian letters; skipped")
            continue
        clean = S.generate(S.make_spec(cls, preset="clean", seed=SEED), keep_arrays=True)
        photo = S.generate(S.make_spec(cls, preset="photo", seed=SEED), keep_arrays=True)
        c = check_clean(clean)
        o = check_order(clean["truth"])
        d = check_degraded(photo)
        ok = c["agree_frac"] >= PASS_FRAC and clean["truth"]["notes"]["n_missing"] == 0
        all_pass &= ok
        report["classes"][cls] = dict(font=found[0], path=found[1], clean=c, order=o, degraded=d,
                                      notes=clean["truth"]["notes"], verdict="PASS" if ok else "FAIL")
        print(f"{cls:13s} {found[0]:28s} {c['n_words']:5d} {100 * c['agree_frac']:6.2f}% {o['n_violations']:5d} "
              f"{c['mean_cov2']:6.3f} {c['mean_cov_eroded']:6.3f} {c['image_dark_unexplained']:9d} "
              f"{100 * d['frac']:5.1f}%  {'PASS' if ok else 'FAIL'}")
        sheet = out / f"contact_1a_{cls}.png"
        contact_sheet(clean, photo, sheet, f"{cls}: {found[0]}")
        zoom_crop(clean, out / f"zoom_1a_{cls}_clean.png", boxes=False)
        zoom_crop(clean, out / f"zoom_1a_{cls}_clean_boxes.png")
        zoom_crop(photo, out / f"zoom_1a_{cls}_photo_boxes.png", frac=(0.0, 0.62, 0.6, 0.95))
        thumbs.append((cls, clean, photo))
        # Every other candidate face of the class, clean only.
        for name, path in S.FONT_CANDIDATES[cls]:
            if path == found[1] or not S.font_status(path)["persian_letters"]:
                continue
            spec = S.make_spec(cls, preset="clean", seed=SEED, font=S._rel(path), font_name=name,
                               name=f"{cls}_{name.replace(' ', '')}_clean_s{SEED}")
            if cls != "bold_heading":
                spec["body_font"] = spec["font"]
            r = S.generate(spec, keep_arrays=True)
            cc = check_clean(r); oo = check_order(r["truth"])
            report["fonts"][f"{cls}/{name}"] = dict(path=path, clean=cc, order=oo)
            print(f"{'  (alt)':13s} {name:28s} {cc['n_words']:5d} {100 * cc['agree_frac']:6.2f}% {oo['n_violations']:5d} "
                  f"{cc['mean_cov2']:6.3f} {cc['mean_cov_eroded']:6.3f} {cc['image_dark_unexplained']:9d}")
            zoom_crop(r, out / f"zoom_1a_{cls}_{name.replace(' ', '')}_clean.png", boxes=False)
    print("\npipeline v2 presets (typewriter, memo layout, pool text):")
    for preset in ("tw_photo_medium", "tw_photo_hard"):
        if S.resolve_class_font("typewriter") is None:
            break
        clean = S.generate(S.make_spec("typewriter", preset="clean", seed=SEED, layout="memo", text_source="pool",
                                       leading=1.75, name=f"memo_clean_s{SEED}"), keep_arrays=True)
        hard = S.generate(S.make_spec("typewriter", preset=preset, seed=SEED), keep_arrays=True)
        c = check_clean(clean); o = check_order(clean["truth"]); d = check_degraded(hard)
        report["classes"][f"typewriter/{preset}"] = dict(clean=c, order=o, degraded=d, notes=hard["truth"]["notes"])
        print(f"  {preset:16s} memo clean agree={100 * c['agree_frac']:6.2f}% order_violations={o['n_violations']} "
              f"| {preset} warped ink inside polygon={100 * d['frac']:5.1f}%")
        contact_sheet(clean, hard, out / f"contact_1a_{preset}.png", f"typewriter memo: clean vs {preset}")
    print("\nlayout classes (modern_naskh default face, clean):")
    for lay in S.LAYOUTS:
        r = S.generate(S.make_spec("modern_naskh", layout=lay, preset="clean", seed=23,
                                   name=f"layout_{lay}_clean_s23"), keep_arrays=True)
        cc = check_clean(r); oo = check_order(r["truth"])
        report["layouts"][lay] = dict(clean=cc, order=oo, n_lines=len(r["truth"]["lines"]))
        print(f"  {lay:10s} words={cc['n_words']:4d} lines={len(r['truth']['lines']):3d} "
              f"agree={100 * cc['agree_frac']:6.2f}% order_violations={oo['n_violations']}")
        zoom_crop(r, out / f"zoom_1a_layout_{lay}.png", frac=(0.0, 0.0, 1.0, 1.0), boxes=True)
    # Index sheet: all classes side by side (small).
    if thumbs:
        H, W = thumbs[0][1]["image"].shape
        doc = fitz.open()
        pg = doc.new_page(width=len(thumbs) * (W + 40), height=2 * H + 200)
        for k, (cls, cl, ph) in enumerate(thumbs):
            for j, res in enumerate((cl, ph)):
                u8 = res["image"]
                pg.insert_image(fitz.Rect(k * (W + 40), 120 + j * (H + 40), k * (W + 40) + W, 120 + j * (H + 40) + H),
                                pixmap=fitz.Pixmap(fitz.csGRAY, u8.shape[1], u8.shape[0], u8.tobytes(), False))
            pg.insert_text((k * (W + 40) + 20, 90), cls, fontsize=80)
        pg.get_pixmap(matrix=fitz.Matrix(0.15, 0.15), alpha=False).save(str(out / "contact_1a_index.png"))
        doc.close()
    (out / "smoke_1a.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nreport: {out / 'smoke_1a.json'}")
    print("SMOKE 1a:", "PASS" if all_pass else "FAIL (see per-class verdicts)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
