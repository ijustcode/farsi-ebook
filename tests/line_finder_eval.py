#!/usr/bin/env python3
"""Step 3 evaluation: line finders on synthetic pages and on the frozen real
truth pages (plan §7 Step 3, §13).

Arms (each on every data set):
  ref        current detector ``locate._scan_page_lines`` (reference)
  ref_mask   current detector on the page image with L1's non-text mask whitened
  l1         tests/line_finder.py without masking
  l1_mask    tests/line_finder.py with masking

Lines are graded with the real scorer's own ``real_truth_eval.grade_lines``
(clean one-to-one recall, merges, splits, missed, spurious). On synthetic
pages the truth words are their exact ink boxes (``ink_aabb``; plan Step 1
results), lines are the generator's lines, and there are no ignore zones:
stamps, signatures, underlines and specks that become candidates count as
spurious.

Also measured: text-block edge fit (each matched line's measured and fitted
start/end against the truth line's ends), masking damage (synthetic: exact
text ink masked from the word-id map; real: truth word quads containing masked
ink), and on synthetic pages the share of each non-text overlay kind masked.

No API, no network. Real pages: refuses held-out pages (the scorer does),
writes nothing under books/.

  ./venv/bin/python tests/line_finder_eval.py synth --set dev [--arms ...] [--configs ...]
  ./venv/bin/python tests/line_finder_eval.py synth --set final
  ./venv/bin/python tests/line_finder_eval.py real                 # sparingly; see the run log
  ./venv/bin/python tests/line_finder_eval.py overlay-real --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import line_finder as LF  # noqa: E402
import real_truth_eval as rt  # noqa: E402

OUT = ROOT / "out" / "line_finder"
ARMS = ("ref", "ref_mask", "l1", "l1_mask")

# Seeds: development (tuning) and untouched final check.
SEEDS = {"dev": [101, 102, 103], "final": [301, 302, 303]}

# Synthetic configurations: (label, font class, make_spec kwargs).
# 'photo' is the E18 slope profile (0.1/2.2/3.3 deg) with office overlays;
# 'slope35' pushes the bottom slope to 3.5 deg; tw_photo_* are the v2 presets.
CLASSES = ("modern_naskh", "book_naskh", "typewriter", "nastaliq", "bold_heading", "mixed")


def synth_configs():
    cfg = []
    for cls in CLASSES:
        cfg.append((f"{cls}/clean", cls, dict(preset="clean")))
        cfg.append((f"{cls}/photo", cls, dict(preset="photo")))
        cfg.append((f"{cls}/slope35", cls, dict(preset="photo", degrade=dict(warp_bottom_deg=3.5, warp_mid_deg=2.4))))
        cfg.append((f"{cls}/tw_photo_medium", cls, dict(preset="tw_photo_medium")))
    cfg.append(("typewriter/tw_photo_hard", "typewriter", dict(preset="tw_photo_hard")))
    return cfg


# ---------------------------------------------------------------------------
# Synthetic truth -> real-scorer TruthPage
# ---------------------------------------------------------------------------
def synth_truth_page(truth: dict) -> rt.TruthPage:
    W, H = truth["image"]["width"], truth["image"]["height"]
    words = {w["id"]: w for w in truth["words"]}
    lines = []
    for li, tl in enumerate(truth["lines"]):
        tws = []
        for wi, wid in enumerate(tl["word_ids"]):
            w = words[wid]
            b = w.get("ink_aabb")
            if b is None:
                continue
            q = [(b[0] * W, b[1] * H), (b[2] * W, b[1] * H), (b[2] * W, b[3] * H), (b[0] * W, b[3] * H)]
            if rt._quad_area(q) < 1:
                continue
            tws.append(rt.TWord(li, len(tws), w["text"], q, [0, 0, 0, 0], rt.letters(w["text"]), True))
        if not tws:
            continue
        xs = [p[0] for w in tws for p in w.quad]; ys = [p[1] for w in tws for p in w.quad]
        quad = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
        lines.append(rt.TLine(len(lines), {"cx": 0, "cy": 0, "angle_deg": 0}, [0, 0, 0, 0], quad, tws, False))
        lines[-1].src = tl
    # A token whose ink lies on two printed lines (bidi wrap of an LTR run in the
    # generator's 'mixed' layout) has an ink box spanning both: such truth lines
    # cannot be matched cleanly by any finder. Flagged, reported separately.
    hs = sorted(w.quad[2][1] - w.quad[0][1] for L in lines for w in L.words)
    med = hs[len(hs) // 2] if hs else 0
    for L in lines:
        L.bidi_split = any(w.quad[2][1] - w.quad[0][1] > 2.2 * med for w in L.words)
    return rt.TruthPage("synth", 0, "synthetic", "synthetic", W, H, Path("."), "", Path("."), "", lines, [])


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def current_detector(u8: np.ndarray) -> list[dict]:
    """locate._scan_page_lines on an image wrapped in a one-page PDF of the
    image's aspect (A4 width), like smoke 1b; rects normalised to the page."""
    import fitz
    from farsi2epub import locate
    H, W = u8.shape
    pw = 595.0
    ph = pw * H / W
    doc = fitz.open()
    pg = doc.new_page(width=pw, height=ph)
    pg.insert_image(pg.rect, pixmap=fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(u8).tobytes(), False))
    out = []
    for ln in locate._scan_page_lines(pg):
        r = ln.rect
        out.append({"rect": [r.x0 / pw, r.y0 / ph, r.x1 / pw, r.y1 / ph]})
    doc.close()
    return out


def run_arm(arm: str, gray: np.ndarray, cache: dict) -> tuple[list[dict], dict | None, float]:
    t = time.perf_counter()
    if arm == "ref":
        return current_detector(np.clip(gray, 0, 255).astype(np.uint8)), None, time.perf_counter() - t
    if arm in ("l1", "l1_mask"):
        res = LF.find_lines(gray, use_mask=(arm == "l1_mask"), keep_arrays=True)
        cache[arm] = res
        return LF.regions(res), res, time.perf_counter() - t
    if arm == "ref_mask":
        res = cache.get("l1_mask") or LF.find_lines(gray, use_mask=True, keep_arrays=True)
        cache["l1_mask"] = res
        g = np.array(gray, np.float32)
        g[res["masked_px"]] = 255.0
        t2 = time.perf_counter()
        regs = current_detector(np.clip(g, 0, 255).astype(np.uint8))
        return regs, res, res["seconds"] + time.perf_counter() - t2
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Matching, edge fit, masking damage
# ---------------------------------------------------------------------------
def share_matrix(page: rt.TruthPage, regs: list[dict]):
    polys = [rt.region_poly(r, page.W, page.H) for r in regs]
    units = [L for L in page.lines if not L.excluded]
    S = np.zeros((len(units), len(polys)))
    for i, L in enumerate(units):
        tot = sum(rt._quad_area(w.quad) for w in L.words)
        for j, p in enumerate(polys):
            if rt._overlaps(rt.bounds([L.quad]), rt.bounds([p])):
                S[i, j] = sum(rt.fraction_inside(w.quad, [p]) * rt._quad_area(w.quad) for w in L.words) / tot
    return units, S


def clean_pairs(units, S):
    out = {}
    for i in range(len(units)):
        js = [j for j in range(S.shape[1]) if S[i, j] >= rt.EDGE]
        if len(js) == 1 and [k for k in range(len(units)) if S[k, js[0]] >= rt.TOUCH] == [i]:
            out[i] = js[0]
    return out


def truth_line_ends(page: rt.TruthPage, L) -> tuple[float, float, float]:
    """(right end x, left end x, px per letter) of a truth line in image px."""
    if getattr(L, "src", None) is not None or page.cls == "synthetic":
        xr = max(p[0] for w in L.words for p in w.quad)
        xl = min(p[0] for w in L.words for p in w.quad)
    else:
        q = L.quad        # line box in its frame: TL, TR, BR, BL
        xr = (q[1][0] + q[2][0]) / 2
        xl = (q[0][0] + q[3][0]) / 2
    lw = [(max(p[0] for p in w.quad) - min(p[0] for p in w.quad)) / max(1.0, w.letters) for w in L.words]
    return xr, xl, float(np.median(lw)) if lw else 10.0


def edge_errors(page, units, pairs, res) -> dict:
    """Measured (own ink) and fitted (block edge) start/end against truth."""
    if res is None:
        return {}
    text = [L for L in res["lines"] if L.kind == "text"]
    rows = []
    for i, j in pairs.items():
        if j >= len(text):
            continue
        L = text[j]
        xr, xl, lw = truth_line_ends(page, units[i])
        r = dict(line=units[i].index, letter_px=round(lw, 1),
                 start_meas=L.x1 - xr, end_meas=L.x0 - xl,
                 start_fit=getattr(L, "fit_right", L.x1) - xr, end_fit=getattr(L, "fit_left", L.x0) - xl,
                 start_status=getattr(L, "edge_right_status", None), end_status=getattr(L, "edge_left_status", None))
        rows.append(r)
    def summ(key, letters=False):
        v = [abs(r[key]) / (r["letter_px"] if letters else 1.0) for r in rows]
        if not v:
            return None
        return dict(median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                    max=round(float(np.max(v)), 2), over_1_letter=int(sum(abs(r[key]) > r["letter_px"] for r in rows)))
    return dict(n=len(rows),
                start_measured_px=summ("start_meas"), start_fitted_px=summ("start_fit"),
                end_measured_px=summ("end_meas"), end_fitted_px=summ("end_fit"),
                start_measured_letters=summ("start_meas", True), start_fitted_letters=summ("start_fit", True),
                end_measured_letters=summ("end_meas", True), end_fitted_letters=summ("end_fit", True),
                status_counts=dict(start=_count(r["start_status"] for r in rows), end=_count(r["end_status"] for r in rows)),
                rows=[{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def rectification_check(page: rt.TruthPage, units, pairs, res) -> dict:
    """On synthetic pages the truth baseline is exact (it went through the same
    warp as the ink). For each cleanly matched line: (1) round trip
    from_strip(to_strip(p)) - p over the truth polygon; (2) flatness: spread
    (max - min) in strip rows v of the truth baseline points inside the line's
    extent, i.e. how far the rectified line is from straight; (3) the truth
    line polygon's top and bottom edges in strip rows (their spread)."""
    if res is None:
        return {}
    text = [L for L in res["lines"] if L.kind == "text"]
    W, H = page.W, page.H
    rt_err, flat, flat_rel, edge_sp = [], [], [], []
    S = res["measurements"]["line_spacing_px"]
    for i, j in pairs.items():
        src = getattr(units[i], "src", None)
        if src is None or j >= len(text):
            continue
        L = text[j]
        poly = np.array(src["polygon"], float) * [W, H]
        u, v = L.to_strip(poly[:, 0], poly[:, 1])
        x2, y2 = L.from_strip(u, v)
        rt_err.append(float(np.max(np.hypot(x2 - poly[:, 0], y2 - poly[:, 1]))))
        bl = np.array(src["baseline"], float) * [W, H]
        bl = bl[(bl[:, 0] >= L.x0) & (bl[:, 0] <= L.x1)]
        if len(bl) >= 2:
            _, vb = L.to_strip(bl[:, 0], bl[:, 1])
            flat.append(float(np.ptp(vb)))
            flat_rel.append(float(np.ptp(vb)) / S)
        n = len(poly) // 2
        top = poly[:n]; top = top[(top[:, 0] >= L.x0) & (top[:, 0] <= L.x1)]
        if len(top) >= 2:
            edge_sp.append(float(np.ptp(L.to_strip(top[:, 0], top[:, 1])[1])))
    if not flat:
        return dict(n=0)
    return dict(n=len(flat), roundtrip_max_px=round(max(rt_err), 9),
                baseline_spread_px_median=round(float(np.median(flat)), 2),
                baseline_spread_px_p90=round(float(np.percentile(flat, 90)), 2),
                baseline_spread_px_max=round(float(np.max(flat)), 2),
                baseline_spread_over_S_p90=round(float(np.percentile(flat_rel, 90)), 3),
                top_edge_spread_px_median=round(float(np.median(edge_sp)), 2) if edge_sp else None)


def _count(it):
    c = {}
    for x in it:
        c[str(x)] = c.get(str(x), 0) + 1
    return c


def mask_damage_synth(res, idmap: np.ndarray, truth: dict, ink: np.ndarray) -> dict:
    m = res["masked_px"]
    text = idmap > 0
    lost = m & text
    ids = np.unique(idmap[lost]) - 1
    W, H = truth["image"]["width"], truth["image"]["height"]
    kinds = {}
    for o in truth["overlays"]:
        a = o["aabb"]
        x0, y0, x1, y1 = int(a[0] * W), int(a[1] * H), int(math.ceil(a[2] * W)), int(math.ceil(a[3] * H))
        reg = (slice(max(0, y0), min(H, y1)), slice(max(0, x0), min(W, x1)))
        oink = ink[reg] & ~text[reg]
        n = int(oink.sum())
        if n == 0:
            continue
        k = kinds.setdefault(o["kind"], dict(n=0, masked=0, ink_px=0, masked_px=0))
        mm = int((m[reg] & oink).sum())
        k["n"] += 1
        k["ink_px"] += n
        k["masked_px"] += mm
        k["masked"] += int(mm >= 0.5 * n)
        k["masked_share"] = round(k["masked_px"] / max(1, k["ink_px"]), 3)
    return dict(text_ink_px=int(text.sum()), text_ink_masked_px=int(lost.sum()),
                words_losing_ink=int(len(ids)), words_losing_ink_ids=[int(i) for i in ids[:20]],
                overlays=kinds)


def mask_damage_real(page: rt.TruthPage, res) -> dict:
    m = res["masked_px"]
    ink = res["ink"]
    rows = []
    for w in page.graded_words():
        xs = [p[0] for p in w.quad]; ys = [p[1] for p in w.quad]
        x0, x1 = max(0, int(min(xs))), min(page.W, int(math.ceil(max(xs))))
        y0, y1 = max(0, int(min(ys))), min(page.H, int(math.ceil(max(ys))))
        if x1 <= x0 or y1 <= y0:
            continue
        g = rt.Grid((x0, y0, x1, y1), max_cells=10 ** 7)
        inside = g.mask(w.quad)
        Xi = np.clip(g.X.astype(int), 0, page.W - 1); Yi = np.clip(g.Y.astype(int), 0, page.H - 1)
        wi = ink[Yi, Xi] & inside
        wm = m[Yi, Xi] & inside
        if wm.any():
            rows.append(dict(line=w.li, word=w.wi, masked_px=int(wm.sum()), ink_px=int(wi.sum()),
                             share=round(float(wm.sum()) / max(1, int(wi.sum())), 3)))
    return dict(words_with_masked_ink=len(rows), words_losing_5pct=sum(r["share"] >= 0.05 for r in rows),
                rows=rows)


# ---------------------------------------------------------------------------
# Synthetic runs
# ---------------------------------------------------------------------------
def gen(cls, kw, seed):
    import synth_pages as S
    spec = S.make_spec(cls, seed=seed, **kw)
    return spec, S.generate(spec, keep_arrays=True)


def slope_bucket(deg):
    return "<=3.5" if abs(deg) <= 3.5 else ">3.5"


def eval_synth(seeds, configs, arms, verbose=True) -> dict:
    import synth_pages as S
    pages = []
    for label, cls, kw in configs:
        for sd in seeds:
            try:
                spec, g = gen(cls, kw, sd)
            except S.FontUnavailable:
                continue
            truth = g["truth"]
            tp = synth_truth_page(truth)
            gray = g["image"].astype(np.float32)
            W, H = truth["image"]["width"], truth["image"]["height"]
            slopes = [S._line_slope_deg(tl["baseline"], W, H) for tl in truth["lines"]]
            row = dict(config=label, font_class=cls, seed=sd, name=spec["name"], truth_lines=len(tp.lines),
                       slope_range=[round(min(slopes), 2), round(max(slopes), 2)] if slopes else None, arms={})
            cache = {}
            for arm in arms:
                regs, res, secs = run_arm(arm, gray, cache)
                gl = rt.grade_lines(tp, regs)
                units, Sm = share_matrix(tp, regs)
                pairs = clean_pairs(units, Sm)
                # per-line slope buckets
                by = {}
                for i, L in enumerate(units):
                    b = slope_bucket(slopes[L.src["index"]] if L.src["index"] < len(slopes) else 0.0)
                    d = by.setdefault(b, [0, 0])
                    d[1] += 1
                    d[0] += int(i in pairs)
                a = dict({k: gl[k] for k in ("truth_lines", "candidates", "merged_candidates", "truth_lines_in_merges",
                                              "largest_merge", "split_lines", "missed_lines", "spurious_candidates",
                                              "fragment_candidates")},
                         clean=gl["clean_recall"]["k"], recall=gl["clean_recall"]["rate"], seconds=round(secs, 2),
                         by_slope=by, merges=gl["merges"])
                a["missed_kinds"] = _count(units[i].src["kind"] for i in range(len(units)) if i not in pairs)
                a["bidi_split_lines"] = sum(L.bidi_split for L in units)
                a["clean_excl_bidi"] = sum(1 for i in pairs if not units[i].bidi_split)
                if arm.startswith("l1"):
                    a["rectification"] = rectification_check(tp, units, pairs, res)
                    a["edges"] = {k: v for k, v in edge_errors(tp, units, pairs, res).items() if k != "rows"}
                    a["measurements"] = {k: res["measurements"][k] for k in ("stroke_px", "glyph_height_px", "line_spacing_px",
                                                                             "oversized_ink_share", "background_gradient", "slope_field")}
                if arm in ("l1_mask", "ref_mask") and res is not None:
                    a["mask"] = mask_damage_synth(res, g["idmap"], truth, res["ink"])
                    a["mask_summary"] = res["masked"]
                # spurious candidates lying on overlays
                if arm.startswith("l1") and res is not None:
                    a["spurious_on"] = spurious_kinds(tp, regs, gl, truth)
                row["arms"][arm] = a
            pages.append(row)
            g.get("doc") and g["doc"].close()
            if verbose:
                print(f"  {label:32s} s{sd}: " + "  ".join(
                    f"{arm} {row['arms'][arm]['clean']}/{row['truth_lines']} m{row['arms'][arm]['merged_candidates']}"
                    f" s{row['arms'][arm]['spurious_candidates']}" for arm in arms), flush=True)
    return dict(pages=pages, summary=summarise_synth(pages, arms))


def spurious_kinds(tp, regs, gl, truth):
    """Which overlay kinds the spurious/fragment candidates sit on."""
    units, Sm = share_matrix(tp, regs)
    mx = Sm.max(axis=0) if len(units) else np.zeros(len(regs))
    W, H = tp.W, tp.H
    out = {}
    for j, r in enumerate(regs):
        if mx[j] >= rt.TOUCH:
            continue
        poly = rt.region_poly(r, W, H)
        b = rt.bounds([poly])
        kind = "none"
        for o in truth["overlays"]:
            a = o["aabb"]
            ob = (a[0] * W, a[1] * H, a[2] * W, a[3] * H)
            if rt._overlaps(b, ob):
                kind = o["kind"]
                break
        key = ("spurious:" if mx[j] < 0.05 else "fragment:") + kind
        out[key] = out.get(key, 0) + 1
    return out


def summarise_synth(pages, arms) -> dict:
    out = {}
    groups = {}
    for p in pages:
        for key in (p["config"], p["font_class"] + "/*", "*"):
            groups.setdefault(key, []).append(p)
    for key, ps in sorted(groups.items()):
        out[key] = {}
        for arm in arms:
            rows = [p["arms"][arm] for p in ps if arm in p["arms"]]
            if not rows:
                continue
            tl = sum(r["truth_lines"] for r in rows)
            cl = sum(r["clean"] for r in rows)
            b = {}
            for r in rows:
                for k, (c, n) in r["by_slope"].items():
                    bb = b.setdefault(k, [0, 0]); bb[0] += c; bb[1] += n
            bs = sum(r.get("bidi_split_lines", 0) for r in rows)
            ce = sum(r.get("clean_excl_bidi", 0) for r in rows)
            d = dict(pages=len(rows), clean=cl, truth_lines=tl, recall=round(cl / tl, 4) if tl else None,
                     bidi_split_lines=bs, recall_excl_bidi=round(ce / (tl - bs), 4) if tl - bs else None,
                     merged=sum(r["merged_candidates"] for r in rows),
                     lines_in_merges=sum(r["truth_lines_in_merges"] for r in rows),
                     split=sum(r["split_lines"] for r in rows), missed=sum(r["missed_lines"] for r in rows),
                     spurious=sum(r["spurious_candidates"] for r in rows),
                     max_spurious_page=max(r["spurious_candidates"] for r in rows),
                     fragments=sum(r["fragment_candidates"] for r in rows),
                     recall_slope_le_3_5=(round(b["<=3.5"][0] / b["<=3.5"][1], 4) if b.get("<=3.5") and b["<=3.5"][1] else None),
                     by_slope=b, seconds_mean=round(float(np.mean([r["seconds"] for r in rows])), 2),
                     seconds_max=round(float(np.max([r["seconds"] for r in rows])), 2))
            if any("mask" in r for r in rows):
                mk = [r["mask"] for r in rows if "mask" in r]
                d["text_ink_masked_px"] = sum(m["text_ink_masked_px"] for m in mk)
                d["words_losing_ink"] = sum(m["words_losing_ink"] for m in mk)
                ov = {}
                for m in mk:
                    for k, v in m["overlays"].items():
                        o = ov.setdefault(k, dict(n=0, masked=0, ink_px=0, masked_px=0))
                        for f in o:
                            o[f] += v[f]
                d["overlays_masked"] = ov
            if any("edges" in r and r["edges"].get("n") for r in rows):
                d["edges"] = {}
                for f in ("start_measured_letters", "start_fitted_letters", "end_measured_letters", "end_fitted_letters"):
                    vals = [r["edges"][f] for r in rows if r["edges"].get(f)]
                    d["edges"][f] = dict(median_of_page_medians=round(float(np.median([v["median"] for v in vals])), 2),
                                         over_1_letter=sum(v["over_1_letter"] for v in vals),
                                         n=sum(r["edges"]["n"] for r in rows if r["edges"].get(f)))
            if any("spurious_on" in r for r in rows):
                so = {}
                for r in rows:
                    for k, v in r.get("spurious_on", {}).items():
                        so[k] = so.get(k, 0) + v
                d["spurious_on"] = so
            out[key][arm] = d
    return out


# ---------------------------------------------------------------------------
# Real pages
# ---------------------------------------------------------------------------
def eval_real(arms) -> dict:
    keys = rt.frozen_keys(ROOT)
    out = dict(pages={}, groups={})
    for slug, page in keys:
        t = rt.load_truth(ROOT, slug, page)
        gray = LF.load_gray(t.image_path)
        cache = {}
        prow = dict(key=t.key, group=rt.group_of(t), graded_lines=sum(not L.excluded for L in t.lines), arms={})
        for arm in arms:
            if arm == "ref":
                pi = rt.build_page_input(ROOT, t, None)
                tt = time.perf_counter()
                regs = rt.ScanPageLines().find_lines(pi)
                res, secs = None, time.perf_counter() - tt
            else:
                regs, res, secs = run_arm(arm, gray, cache)
            gl = rt.grade_lines(t, regs)
            units, Sm = share_matrix(t, regs)
            pairs = clean_pairs(units, Sm)
            a = dict({k: v for k, v in gl.items() if k not in ("clean_truth_lines",)}, seconds=round(secs, 2))
            a["unclean_truth_lines"] = [u.index for i, u in enumerate(units) if i not in pairs]
            if arm.startswith("l1") and res is not None:
                a["edges"] = edge_errors(t, units, pairs, res)
                a["measurements"] = res["measurements"]
                a["masked"] = res["masked"]
                a["edges_fit"] = res["edges"]
            if arm in ("l1_mask",) and res is not None:
                a["mask_damage"] = mask_damage_real(t, res)
            prow["arms"][arm] = a
        out["pages"][t.key] = prow
        print(f"  {t.key:24s} " + "  ".join(
            f"{arm} {prow['arms'][arm]['clean_recall']['k']}/{prow['graded_lines']} m{prow['arms'][arm]['merged_candidates']}"
            f" sp{prow['arms'][arm]['spurious_candidates']}" for arm in arms), flush=True)
    for g in sorted({p["group"] for p in out["pages"].values()}):
        ps = [p for p in out["pages"].values() if p["group"] == g]
        out["groups"][g] = {}
        for arm in arms:
            rows = [p["arms"][arm] for p in ps]
            tl = sum(r["truth_lines"] for r in rows); cl = sum(r["clean_recall"]["k"] for r in rows)
            out["groups"][g][arm] = dict(pages=[p["key"] for p in ps], clean=cl, truth_lines=tl,
                                         recall=round(cl / tl, 4) if tl else None,
                                         merged=sum(r["merged_candidates"] for r in rows),
                                         lines_in_merges=sum(r["truth_lines_in_merges"] for r in rows),
                                         split=sum(r["split_lines"] for r in rows),
                                         missed=sum(r["missed_lines"] for r in rows),
                                         spurious=sum(r["spurious_candidates"] for r in rows),
                                         max_spurious_page=max(r["spurious_candidates"] for r in rows))
    return out


# ---------------------------------------------------------------------------
# Overlays (scratch images only)
# ---------------------------------------------------------------------------
def draw_overlay(gray: np.ndarray, res: dict, path: Path, truth_page: rt.TruthPage | None = None, scale: float = 0.5,
                 crop: tuple | None = None):
    """Page with L1 lines (red polygons, blue centre curves), masked ink (green),
    truth line outlines (thin orange) if given."""
    import fitz
    H, W = gray.shape
    rgb = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if "masked_px" in res:
        m = res["masked_px"]
        rgb[m] = (rgb[m] * 0.3 + np.array([0, 200, 0]) * 0.7).astype(np.uint8)
    pm = fitz.Pixmap(fitz.csRGB, W, H, np.ascontiguousarray(rgb).tobytes(), False)
    doc = fitz.open()
    pg = doc.new_page(width=W, height=H)
    pg.insert_image(pg.rect, pixmap=pm)
    if truth_page is not None:
        for L in truth_page.lines:
            q = L.quad
            pg.draw_polyline([fitz.Point(*p) for p in q] + [fitz.Point(*q[0])], color=(1, 0.55, 0), width=1.2)
    for k, L in enumerate(res["lines"]):
        col = (0.9, 0, 0) if L.kind == "text" else (0.6, 0, 0.8)
        poly = L.polygon()
        pg.draw_polyline([fitz.Point(*p) for p in poly] + [fitz.Point(*poly[0])], color=col, width=2)
        bl = L.baseline(24)
        pg.draw_polyline([fitz.Point(*p) for p in bl], color=(0, 0.2, 1), width=1.5)
        pg.insert_text(fitz.Point(max(2, L.x1 + 8), float(L.y_at(L.x1)) + 8), str(k), fontsize=22, color=col)
        fr = getattr(L, "fit_right", None)
        if fr is not None:
            y = float(L.y_at(L.x1))
            pg.draw_line(fitz.Point(fr, y - L.up), fitz.Point(fr, y + L.dn), color=(1, 0, 1), width=3)
    clip = fitz.Rect(*crop) if crop else None
    pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
    pix.save(str(path))
    doc.close()


def overlay_regions(gray, regs, path, scale=0.5, color=(0, 0.6, 0)):
    import fitz
    H, W = gray.shape
    pm = fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(np.clip(gray, 0, 255).astype(np.uint8)).tobytes(), False)
    doc = fitz.open()
    pg = doc.new_page(width=W, height=H)
    pg.insert_image(pg.rect, pixmap=pm)
    for r in regs:
        poly = rt.region_poly(r, W, H)
        pg.draw_polyline([fitz.Point(*p) for p in poly] + [fitz.Point(*poly[0])], color=color, width=2)
    pg.get_pixmap(matrix=fitz.Matrix(scale, scale)).save(str(path))
    doc.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--set", choices=sorted(SEEDS), default="dev")
    s.add_argument("--seeds", type=int, nargs="*")
    s.add_argument("--arms", nargs="*", default=list(ARMS))
    s.add_argument("--configs", nargs="*", help="labels or prefixes (e.g. typewriter/)")
    s.add_argument("--out", type=Path)
    r = sub.add_parser("real")
    r.add_argument("--arms", nargs="*", default=list(ARMS))
    r.add_argument("--out", type=Path, default=OUT / "real.json")
    o = sub.add_parser("overlay-real")
    o.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "synth":
        cfgs = [c for c in synth_configs() if not a.configs or any(c[0] == x or c[0].startswith(x) for x in a.configs)]
        seeds = a.seeds or SEEDS[a.set]
        rep = eval_synth(seeds, cfgs, a.arms)
        rep.update(seeds=seeds, set=a.set, params=LF.PARAMS, version=LF.VERSION)
        out = a.out or OUT / f"synth_{a.set}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
        print_synth_summary(rep["summary"], a.arms)
        print(f"wrote {out}")
    elif a.cmd == "real":
        rep = eval_real(a.arms)
        rep.update(params=LF.PARAMS, version=LF.VERSION, created=rt.now_iso())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
        for g, d in rep["groups"].items():
            for arm, v in d.items():
                print(f"  [{g}] {arm:9s} {v['clean']}/{v['truth_lines']} merged {v['merged']} (lines {v['lines_in_merges']}) "
                      f"split {v['split']} missed {v['missed']} spurious {v['spurious']} (max/page {v['max_spurious_page']})")
        print(f"wrote {a.out}")
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        for slug, page in rt.frozen_keys(ROOT):
            t = rt.load_truth(ROOT, slug, page)
            gray = LF.load_gray(t.image_path)
            res = LF.find_lines(gray, use_mask=True, keep_arrays=True)
            draw_overlay(gray, res, a.out / f"{slug}_{page}_l1mask.png", t, scale=0.45)
            print("wrote", a.out / f"{slug}_{page}_l1mask.png")
    return 0


def _np(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def print_synth_summary(summ, arms):
    print(f"\n{'group':34s} " + " ".join(f"{a:>22s}" for a in arms))
    for key, d in summ.items():
        cells = []
        for a in arms:
            v = d.get(a)
            cells.append(f"{v['clean']:>4d}/{v['truth_lines']:<4d} m{v['merged']:<2d} s{v['spurious']:<3d}" if v else " " * 22)
        print(f"{key:34s} " + " ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    sys.exit(main())
