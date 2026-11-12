#!/usr/bin/env python3
"""Regression test for the Step 3 line finder (tests/line_finder.py, arm L1).

Synthetic pages only (tests/synth_pages.py); no books/, no API, no network.

Covers, on a fixed set (a clean typewriter page, a sloped typewriter photo page
with specks and typed underlines but no large overlays, and a sloped naskh page
with the v1 office overlays):
  * line count and clean one-to-one recall (graded by real_truth_eval.grade_lines);
  * no text ink masked on that set (exact, from the generator's word-id map);
  * specks and underlines do not become lines (no spurious candidate on them);
  * rectification: from_strip(to_strip(p)) == p, and the truth baseline is
    straight in strip coordinates within 0.15 x line spacing (p90);
  * determinism: two runs give identical JSON output.

    ./venv/bin/python tests/line_finder_regression.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import line_finder as LF  # noqa: E402
import line_finder_eval as E  # noqa: E402
import real_truth_eval as rt  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# (label, font class, make_spec kwargs, seed, min clean recall)
SET = [
    ("typewriter clean", "typewriter", dict(preset="clean"), 7, 1.0),
    ("typewriter photo (sloped 1.2-3.5 deg, specks, typed underlines)", "typewriter",
     dict(preset="tw_photo_medium", degrade=dict(stamps_big=0, signatures_big=0)), 7, 1.0),
    ("naskh photo (E18 slope, stamp, signature, underlines, margin marks)", "modern_naskh",
     dict(preset="photo"), 7, 1.0),
]


def main() -> int:
    for label, cls, kw, seed, need in SET:
        spec, g = E.gen(cls, kw, seed)
        truth = g["truth"]
        tp = E.synth_truth_page(truth)
        gray = g["image"].astype(np.float32)
        res = LF.find_lines(gray, use_mask=True, keep_arrays=True)
        regs = LF.regions(res)
        gl = rt.grade_lines(tp, regs)
        units, Sm = E.share_matrix(tp, regs)
        pairs = E.clean_pairs(units, Sm)
        n = len(units)
        print(f"-- {label}: {len(pairs)}/{n} clean, {len(regs)} lines, slope range "
              f"{truth['notes'].get('slope_deg_range')}, {res['seconds']} s")
        so = E.spurious_kinds(tp, regs, gl, truth)
        on_overlay = sum(v for k, v in so.items() if k.split(":")[1] in ("stamp", "signature", "stamp_big", "signature_big"))
        # Known limitation (pinned, not hidden): a stamp ring broken into pieces is
        # not one big component, so its tick marks can form a short extra line.
        check(len(regs) - on_overlay == n, f"{label}: line count {len(regs)} == truth {n} + {on_overlay} "
                                            f"extra candidate(s) on a stamp/signature")
        check(len(pairs) / n >= need, f"{label}: clean recall {len(pairs)}/{n} >= {need:.0%}")
        check(gl["merged_candidates"] == 0, f"{label}: no merged candidates ({gl['merged_candidates']})")
        md = E.mask_damage_synth(res, g["idmap"], truth, res["ink"])
        check(md["text_ink_masked_px"] == 0, f"{label}: no text ink masked ({md['text_ink_masked_px']} px, "
                                             f"{md['words_losing_ink']} words)")
        bad = {k: v for k, v in so.items() if any(x in k for x in ("speck", "underline", "margin"))}
        check(not bad, f"{label}: no line made of specks/underlines/margin marks ({bad or 'none'})")
        if "slope_deg_range" in truth["notes"] and truth["notes"]["slope_deg_range"]:
            check(max(abs(v) for v in truth["notes"]["slope_deg_range"]) >= 3.0,
                  f"{label}: page really is sloped (max line slope {truth['notes']['slope_deg_range']})")
        rc = E.rectification_check(tp, units, pairs, res)
        check(rc.get("n", 0) == len(pairs) and rc["roundtrip_max_px"] < 1e-6,
              f"{label}: rectification round trip exact (max {rc.get('roundtrip_max_px')} px)")
        check(rc.get("baseline_spread_over_S_p90", 9) <= 0.15,
              f"{label}: truth baseline straight in the strip, p90 spread "
              f"{rc.get('baseline_spread_px_p90')} px = {rc.get('baseline_spread_over_S_p90')} S <= 0.15 S")
        # strip image sampling keeps the requested shape and the page frame
        L = [l for l in res["lines"] if l.kind == "text"][len(pairs) // 2]
        strip = L.rectify(gray)
        check(strip.shape == L.strip_shape(), f"{label}: rectify() returns a strip of shape {strip.shape}")
        if label.startswith("typewriter photo"):
            res2 = LF.find_lines(gray, use_mask=True)
            j1, j2 = LF.to_json(res), LF.to_json(res2)
            for j in (j1, j2):
                j.pop("seconds")
            check(json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True),
                  f"{label}: deterministic (identical JSON on a second run)")
            sp = [o for o in truth["overlays"] if o["kind"] == "speck"]
            check(len(sp) > 0, f"{label}: the page carries {len(sp)} specks")
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s)")
        return 1
    print("all line_finder regression checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
