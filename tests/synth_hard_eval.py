#!/usr/bin/env python3
"""How hard are the synthetic typewriter pages? Old "photo" preset vs the v2
degradations (tests/synth_pages.py), measured with two existing methods:

1. the CURRENT production line detector (``locate._scan_page_lines``, as in
   smoke 1b): detected vs true lines, largest number of true lines merged into
   one detected block, merged blocks;
2. the box-audit draft splitter (``tests/box_audit_draft.draft_known_lines``,
   read-only use, called exactly as ``box_audit.synth_eval_page`` does, which
   is also called for the headline numbers), plus error classes:

   * ``ok`` -- box_audit's own criterion (horizontal IoU >= 0.5 with the
     word's ink box and some vertical overlap);
   * ``edge_exact`` -- both x edges within max(3 px, half a letter) of the ink
     box. A rough stand-in for "the auditor left the box unchanged"; it is NOT
     the owner's audit criterion;
   * O1 cascade -- a line whose box 0 (the line START, right end) covers a
     recorded speck and on which >= 2 boxes match a different word than their
     position says (shifted). ``cascade_any`` counts lines with >= 2 shifted
     boxes whatever started them;
   * O2 boundary error -- a word boundary whose two boxes both match the right
     word but where an edge is off by >= 1 letter on the wrong side (a box
     takes letters of its neighbour or loses its own), reported overall, at
     boundaries recorded as ``o2_trigger`` (gap <= the larger in-word gap of
     the two words) and at tightened boundaries.

No API calls. Writes out/synth/hard_eval.json.

Run: ./venv/bin/python tests/synth_hard_eval.py [--seeds 3 4 5 6] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import synth_pages as S  # noqa: E402
import synth_smoke_1b as B  # noqa: E402
import box_audit as BA  # noqa: E402
import box_audit_draft as BD  # noqa: E402

PHOTO = dict(S.DEGRADE_PRESETS["photo"])
HARD = dict(S.DEGRADE_PRESETS["tw_photo_hard"])
V2_ONLY = {k: v for k, v in HARD.items() if k in S.V2_DEFAULTS}


def _pick(*keys):
    return {k: HARD[k] for k in keys if k in HARD}


# (label, make_spec kwargs). Ablations add one v2 degradation group to the old
# photo preset on the old layout/text, so each effect is seen in isolation.
CONFIGS = [
    ("photo (old)", dict(preset="photo")),
    ("photo+specks", dict(preset="photo", degrade=_pick("specks", "specks_start_frac", "specks_end_frac",
                                                        "specks_size", "specks_size_sigma", "specks_dist_pt"))),
    ("photo+tight", dict(preset="photo", degrade=_pick("tight_frac", "tight_to"))),
    ("photo+big_overlays", dict(preset="photo", degrade=dict(_pick("stamps_big", "signatures_big"),
                                                             stamp=False, signature=False))),
    ("photo+baselines", dict(preset="photo", degrade=_pick("baseline_jitter_px", "baseline_slope_deg",
                                                           "baseline_curve_px"))),
    ("photo+texture", dict(preset="photo", degrade=_pick("tw_ink_var", "tw_halfstrike", "tw_jitter_px",
                                                         "tw_key_bias_px", "tw_fill_counters"))),
    ("photo, memo layout+pool", dict(preset="photo", layout="memo", text_source="pool", leading=1.75)),
    ("tw_photo_medium", dict(preset="tw_photo_medium")),
    ("tw_photo_hard", dict(preset="tw_photo_hard")),
]


def _xiou(a, b):
    i = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    u = max(a[2], b[2]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def classify(image_u8: np.ndarray, truth: dict) -> dict:
    H, W = image_u8.shape
    texts = [ln["text"] for ln in truth["lines"]]
    res = BD.draft_known_lines(image_u8.astype(np.float32), texts, [False] * len(texts))
    words = {w["id"]: w for w in truth["words"]}
    specks = [o for o in truth["overlays"] if o["kind"] == "speck"]
    gaps = {(g["right_id"], g["left_id"]): g for g in truth.get("word_gaps", [])}
    out = dict(n_graded=0, edge_exact=0, shifted=0, o1_lines=0, cascade_any=0, speck_captured_boxes=0,
               o1_start_specks_on_page=sum(1 for s in specks if s.get("o1_trigger")),
               boundaries=0, o2_err=0, o2_trigger_bounds=0, o2_err_at_trigger=0,
               tight_bounds=0, o2_err_at_tight=0)
    for L, tl in zip(res["lines"], truth["lines"]):
        tw = sorted((words[i] for i in tl["word_ids"]), key=lambda w: w["visual_index"])
        t = L["track"]
        boxes = []
        for (x0, x1, dt_, db_) in L["words"]:
            ys = [t.y_at(x0) + dt_, t.y_at(x1) + dt_, t.y_at(x0) + db_, t.y_at(x1) + db_]
            boxes.append([x0 / W, min(ys) / H, x1 / W, max(ys) / H])
        match = []
        for b in boxes:
            best, bi = 0.0, None
            for j, w in enumerate(tw):
                ib = w["ink_aabb"]
                if ib is None or min(b[3], ib[3]) <= max(b[1], ib[1]):
                    continue
                v = _xiou(b, ib)
                if v > best:
                    best, bi = v, j
            match.append(bi if best >= 0.3 else None)
        shifted = sum(1 for k, m in enumerate(match) if m is not None and m != k)
        out["shifted"] += shifted
        cap0 = False
        for k, b in enumerate(boxes):
            for s in specks:
                a = s["aabb"]
                inter = min(b[2], a[2]) - max(b[0], a[0])
                if inter >= 0.5 * (a[2] - a[0]) and min(b[3], a[3]) > max(b[1], a[1]):
                    out["speck_captured_boxes"] += 1
                    cap0 |= k == 0
                    break
        if shifted >= 2:
            out["cascade_any"] += 1
            if cap0:
                out["o1_lines"] += 1
        for k, w in enumerate(tw):
            out["n_graded"] += 1
            ib = w["ink_aabb"]
            if k >= len(boxes) or ib is None:
                continue
            nch = max(1, len(w["text"].replace("‌", "")))
            letter = (ib[2] - ib[0]) / nch
            tol = max(3.0 / W, 0.5 * letter)
            b = boxes[k]
            if abs(b[0] - ib[0]) <= tol and abs(b[2] - ib[2]) <= tol:
                out["edge_exact"] += 1
        # boundaries between visual neighbours k (right) and k+1 (left)
        for k in range(len(tw) - 1):
            r, l = tw[k], tw[k + 1]
            if k + 1 >= len(boxes) or match[k] != k or match[k + 1] != k + 1:
                continue
            ir, il = r["ink_aabb"], l["ink_aabb"]
            if ir is None or il is None:
                continue
            letter = min((ir[2] - ir[0]) / max(1, len(r["text"])), (il[2] - il[0]) / max(1, len(l["text"])))
            br, bl = boxes[k], boxes[k + 1]
            # wrong side by >= 1 letter: right box extends left into l, or
            # stops short inside r; left box extends right into r, or stops short inside l
            err = (br[0] < il[2] - letter or br[0] > ir[0] + letter
                   or bl[2] > ir[0] + letter or bl[2] < il[2] - letter)
            g = gaps.get((r["id"], l["id"]))
            out["boundaries"] += 1
            out["o2_err"] += err
            if g and g.get("o2_trigger"):
                out["o2_trigger_bounds"] += 1
                out["o2_err_at_trigger"] += err
            if g and g.get("tightened"):
                out["tight_bounds"] += 1
                out["o2_err_at_tight"] += err
    return out


def run(seeds, configs, quick=False) -> list[dict]:
    rows = []
    for label, kw in configs:
        for sd in seeds:
            spec = S.make_spec("typewriter", seed=int(sd), **kw)
            g = S.generate(spec)
            img, truth = g["image"], g["truth"]
            det = B.summarise(B.detect(img), truth, None)
            se = BA.synth_eval_page(img, truth)
            cl = classify(img, truth)
            sl = [B_ for B_ in ([S._line_slope_deg(t["baseline"], img.shape[1], img.shape[0]) for t in truth["lines"]])]
            k3 = max(1, len(sl) // 3)
            row = dict(config=label, seed=int(sd), name=spec["name"],
                       det=det["detected"], true=det["true"], max_lines_in_block=det["max_lines_in_block"],
                       merged_blocks=det["merged_blocks"], lines_in_merged=det["lines_in_merged"],
                       tallest=det["tallest"],
                       ok=se["ok"], words=se["words"], ok_iou50=se["ok_iou50"], wrong_unflagged=se["wrong_unflagged"],
                       lines_all_ok=se["lines_all_ok"], uncertain_lines=se["uncertain_lines"],
                       slope_top_mid_bottom=[round(float(np.median(sl[:k3])), 2), round(float(np.median(sl[len(sl) // 2 - k3 // 2: len(sl) // 2 + k3 // 2 + 1])), 2),
                                             round(float(np.median(sl[-k3:])), 2)],
                       notes={k: truth["notes"].get(k) for k in ("n_specks", "n_specks_o1", "n_tightened", "n_boundaries",
                                                                  "n_o2_trigger", "n_words_covered", "letter_gap_px", "stroke_px")},
                       **cl)
            rows.append(row)
            print(f"  {label:24s} s{sd}: det {row['det']:2d}/{row['true']:2d} maxL/blk {row['max_lines_in_block']}  "
                  f"draft ok {row['ok'] / row['words']:.3f} edge-exact {row['edge_exact'] / row['words']:.3f}  "
                  f"O1 {row['o1_lines']} (cascades {row['cascade_any']})  O2 {row['o2_err']}/{row['boundaries']} "
                  f"(at trigger {row['o2_err_at_trigger']}/{row['o2_trigger_bounds']}, at tightened {row['o2_err_at_tight']}/{row['tight_bounds']})",
                  flush=True)
            if quick:
                break
    return rows


def summary(rows) -> list[dict]:
    out = []
    for label in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == label]
        tot = lambda k: sum(r[k] for r in rs)  # noqa: E731
        out.append(dict(config=label, pages=len(rs), det=tot("det"), true=tot("true"),
                        max_lines_in_block=max(r["max_lines_in_block"] for r in rs),
                        merged_blocks=tot("merged_blocks"), lines_in_merged=tot("lines_in_merged"),
                        ok=tot("ok") / tot("words"), edge_exact=tot("edge_exact") / tot("words"),
                        lines_all_ok=f"{tot('lines_all_ok')}/{tot('true')}",
                        o1_lines=tot("o1_lines"), cascade_any=tot("cascade_any"),
                        speck_captured_boxes=tot("speck_captured_boxes"),
                        o2_err=tot("o2_err"), boundaries=tot("boundaries"),
                        o2_err_at_trigger=tot("o2_err_at_trigger"), o2_trigger_bounds=tot("o2_trigger_bounds"),
                        o2_err_at_tight=tot("o2_err_at_tight"), tight_bounds=tot("tight_bounds"),
                        slopes=[r["slope_top_mid_bottom"] for r in rs]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--configs", nargs="*", default=None, help="labels to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="first seed only")
    a = ap.parse_args(argv)
    cfgs = [c for c in CONFIGS if a.configs is None or c[0] in a.configs]
    rows = run(a.seeds, cfgs, a.quick)
    summ = summary(rows)
    print(f"\n{'config':24s} {'det/true':>9s} {'maxL':>4s} {'merged':>6s} {'ok':>6s} {'edgeX':>6s} {'lines ok':>8s} "
          f"{'O1':>3s} {'casc':>4s} {'O2 err/bnd':>11s} {'@trig':>9s} {'@tight':>9s}")
    for s in summ:
        print(f"{s['config']:24s} {s['det']:4d}/{s['true']:<4d} {s['max_lines_in_block']:4d} {s['merged_blocks']:6d} "
              f"{s['ok']:6.3f} {s['edge_exact']:6.3f} {s['lines_all_ok']:>8s} {s['o1_lines']:3d} {s['cascade_any']:4d} "
              f"{s['o2_err']:4d}/{s['boundaries']:<6d} {s['o2_err_at_trigger']:3d}/{s['o2_trigger_bounds']:<5d} "
              f"{s['o2_err_at_tight']:3d}/{s['tight_bounds']:<5d}")
    S.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (S.OUT_DIR / "hard_eval.json").write_text(json.dumps(dict(rows=rows, summary=summ), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                              encoding="utf-8")
    print(f"\nreport: {S.OUT_DIR / 'hard_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
