"""$0 probe: does disagreement between Gemini's two readings mark hard pages?

  ./venv/bin/python tests/gate_disagreement_probe.py

Reads only caches (no network). Primary reading = Gemini 4 Carbon whole page
(the first pass `transcribe` keeps; cache of tests/gemini_first_pass_eval.py).
Second reading = Gemini 4 Carbon on fixed bands (arm `gg_carbon`, the hard-page
reader's request). Development pages only: hadpi p1, p2 (hard, two runs of each
reading) and majid p23, p109 (clean, one run each). Never hadpi p3-p5.

Signal per page and run pair: flag rate = share of primary words the second
reading disagrees with, after the lenient folding (joins, ZWNJ, punctuation are
not disagreements), from TE.t2_flags.

CRITERIA, WRITTEN 2026-11-15 BEFORE THE FIRST RUN (owner rule D25):

  SEPARATION  PASS: the lowest flag rate of the 8 hard run pairs is more than
              2x the highest of the 2 clean pairs.
              KILL: any clean pair's flag rate >= any hard pair's flag rate.
              Otherwise: THIN.
  USEFULNESS  PASS: pooled over the 8 hard pairs, at least 60% of the primary
              reading's real errors (against owner truth) lie at flagged words.
              Otherwise FAIL.

  Reading of the outcome:
    both PASS            -> worth proposing a paid test on a whole book
                            (4 pages cannot set a threshold by themselves).
    SEPARATION KILL      -> this signal is not proposed as a page gate.
    anything else        -> report as inconclusive; no paid test proposed on
                            this evidence alone.

Also printed, not criteria: each primary reading's real lenient error and its
own reported confidence, and the flag rate between two runs of the same
reading (the noise floor).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"   # no network
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import transcription_experiments as TE  # noqa: E402

FP = TE.OUT / "gemini_first_pass"
HARD, CLEAN = ("h1", "h2"), ("c23", "c109")


def whole(pid: str, rep: int):
    f = FP / f"{pid}_{rep}.json"
    if not f.is_file():
        return None
    r = json.loads(f.read_text())
    return TE.tokenize_md(r["parsed"]["text_md"]), r["parsed"]["confidence"]


def bands(runner, pid: str, rep: int):
    got = TE._any_tokens(runner, pid, "gg_carbon", rep)
    return None if got is None else got[0]


def main() -> int:
    runner = TE.runner_2b()
    rows = []
    print(f"{'page':5s} {'whole':5s} {'bands':5s} {'flag rate':>16s} {'primary errors':>15s} {'caught':>8s} {'conf':>5s}")
    for pid in HARD + CLEAN:
        truth = TE.truth_tokens(pid)
        for wr in (0, 1):
            w = whole(pid, wr)
            if w is None:
                continue
            for br in (0, 1):
                b = bands(runner, pid, br)
                if b is None:
                    continue
                s = TE.t2_stats(truth, w[0], b)
                flagged = s["flagged_words"] + s["flagged_gaps"]
                rows.append({"page": pid, "whole_run": wr, "band_run": br, "flagged": flagged, **s, "confidence": w[1]})
                print(f"{pid:5s} {wr:<5d} {br:<5d} {flagged:4d}/{s['primary_words']:<4d} {100 * s['flag_rate']:5.1f}%"
                      f" {s['errors']:8d}/{len(truth):<4d} {s['caught']:5d}   {w[1]:5.2f}")
    print("\nnoise floor: two runs of the same reading against each other")
    for pid in HARD:
        truth = TE.truth_tokens(pid)
        a, b = whole(pid, 0), whole(pid, 1)
        if a and b:
            s = TE.t2_stats(truth, a[0], b[0])
            print(f"  {pid} whole run0 vs whole run1: {s['flagged_words'] + s['flagged_gaps']}/{s['primary_words']} = {100 * s['flag_rate']:.1f}%")
        a, b = bands(runner, pid, 0), bands(runner, pid, 1)
        if a and b:
            s = TE.t2_stats(truth, a, b)
            print(f"  {pid} bands run0 vs bands run1: {s['flagged_words'] + s['flagged_gaps']}/{s['primary_words']} = {100 * s['flag_rate']:.1f}%")

    hard = [r for r in rows if r["page"] in HARD]
    clean = [r for r in rows if r["page"] in CLEAN]
    if not hard or not clean:
        print("missing cached readings; nothing judged")
        return 2
    lo_h, hi_c = min(r["flag_rate"] for r in hard), max(r["flag_rate"] for r in clean)
    hi_h = max(r["flag_rate"] for r in hard)
    if lo_h > 2 * hi_c:
        sep = "PASS"
    elif hi_c >= lo_h:
        sep = "KILL"
    else:
        sep = "THIN"
    err, caught = sum(r["errors"] for r in hard), sum(r["caught"] for r in hard)
    use = "PASS" if err and caught / err >= 0.60 else "FAIL"
    print(f"\nhard pairs: {len(hard)}, flag rate {100 * lo_h:.1f}%-{100 * hi_h:.1f}%; clean pairs: {len(clean)}, highest {100 * hi_c:.1f}%")
    print(f"SEPARATION: {sep}")
    print(f"USEFULNESS: {use} ({caught}/{err} primary errors at flagged words, pooled over hard pairs)")
    out = TE.OUT / "gate_disagreement_probe.json"
    out.write_text(json.dumps({"rows": rows, "separation": sep, "usefulness": use}, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
