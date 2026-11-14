#!/usr/bin/env python3
"""One-line-shift risk (Step 4 follow-up, brief item 3a). Synthetic only, $0.

When the reader invents, repeats or drops a line, or the line finder gives a
truth line no detected line of its own, the line-text assignment can pair a
text line with its neighbour's ink. This tool characterises, per text line of
the perturbed reading (word_placer_la.perturb_lines, the seeds and
configurations of tests/word_placer_g2.py), whether its trusted pairing is
shifted (the detected line holds another truth line, or none), whether the
truth line had a detected line of its own, and what the run-time evidence on
that line looks like: the template gain per letter relative to the page's
median over pairings, and the share of the line's tokens that rule TA
locates (T and one of S, G, R agree on both side edges). It reports how many
TA boxes such lines show. Nothing is tuned here.

  ./venv/bin/python tests/word_placer_shift.py --set dev [--la T]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import real_truth_eval as rt  # noqa: E402
import word_placer as WP  # noqa: E402
import word_placer_eval as E  # noqa: E402
import word_placer_g2 as G2  # noqa: E402
import word_placer_la as LA  # noqa: E402


def analyse(page, ev, texts, src, la_mode):
    E.T_OPTS.update(vert="comp", la=la_mode)
    cap = {}
    orig_place, orig_rep = LA.place_with_assignment, E.line_assignment_report

    def place(ev_, r, kinds, texts_, la, **kw):
        cap["la"] = la
        return orig_place(ev_, r, kinds, texts_, la, **kw)

    def rep(*a, **kw):
        out = orig_rep(*a, **kw)
        cap["rep"] = out
        return out
    LA.place_with_assignment, E.line_assignment_report = place, rep
    try:
        arms, info = E.run_T(page, "C", ev=ev, texts=texts, src=src)
    finally:
        LA.place_with_assignment, E.line_assignment_report = orig_place, orig_rep
    la, under = cap["la"], cap["rep"]["under"]
    G = la["gains"]
    nlet = [max(1.0, sum(WP.letters(t) for t in toks)) for toks in texts]
    per = [G[i, j] / nlet[i] for i, j in enumerate(la["assign"]) if j is not None and G[i, j] > 0]
    med = float(np.median(per)) if per else 1.0
    own = {u for u in under if u is not None}
    tp = page["truth_page"]
    graded = {(w.li, w.wi) for w in tp.graded_words()}
    rows = []
    for i, j in enumerate(la["assign"]):
        lis = [k[0] for k in src[i] if isinstance(k[0], int)]
        ti = Counter(lis).most_common(1)[0][0] if lis else None
        keys = [k for k in src[i]]
        ta = [arms["TA"].get(k) for k in keys]
        n_ta = sum(1 for x in ta if x and x["status"] == "located")
        ta_ok = ta_wl = 0
        for k, x in zip(keys, ta):
            if x and x["status"] == "located" and isinstance(k[0], int) and k in graded:
                h = rt.grade_target_h1(tp, [k], x)
                ta_ok += bool(h["correct"])
                ta_wl += h.get("reason") == "wrong line"
        rows.append(dict(
            text_line=i, truth_line=ti, det=j, trusted=bool(la["trusted"][i]),
            under=None if j is None else under[j],
            shifted=(j is not None and ti is not None and under[j] != ti),
            unprinted=ti is None, own_det=(ti in own) if ti is not None else None,
            gain_rel=None if j is None else round(float(G[i, j] / nlet[i] / med), 3),
            ta_share=round(n_ta / max(1, len(keys)), 3), ta_boxes=n_ta, ta_correct=ta_ok, ta_wrong_line=ta_wl,
            tokens=len(keys)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(G2.SEEDS), default="dev")
    ap.add_argument("--la", default="T")
    a = ap.parse_args(argv)
    if os.environ.get("ANTHROPIC_BASE_URL") != "http://127.0.0.1:9":
        raise SystemExit("set ANTHROPIC_BASE_URL=http://127.0.0.1:9 (no network)")
    allrows = []
    for cls, preset in G2.CONFIGS:
        for sd in G2.SEEDS[a.set]:
            page = E.synth_page(cls, dict(preset=preset), sd)
            ev = WP.page_evidence(page["gray"])
            pool_page = E.synth_page(cls, dict(preset="clean"), sd + 1000)
            pool = [w for t in pool_page["texts"] for w in t]
            cases = [("none", [list(t) for t in page["texts"]],
                      [[(li, wi) for wi in range(len(t))] for li, t in enumerate(page["texts"])])]
            for name, kw in G2.PERTURB:
                t2, s2 = LA.perturb_lines(page["texts"], seed=sd * 7 + len(name), pool=pool, **kw)
                cases.append((name, t2, s2))
            for name, t2, s2 in cases:
                rows = analyse(page, ev, t2, s2, a.la)
                for r in rows:
                    r.update(config=f"{cls}/{preset}", seed=sd, perturb=name)
                allrows += rows
                sh = [r for r in rows if r["trusted"] and (r["shifted"] or r["unprinted"])]
                print(f"  {cls}/{preset} s{sd} {name:8s} trusted shifted/unprinted {len(sh)}"
                      f" TA boxes on them {sum(r['ta_boxes'] for r in sh)}"
                      f" (correct {sum(r['ta_correct'] for r in sh)}, wrong line {sum(r['ta_wrong_line'] for r in sh)})",
                      flush=True)
    out = HERE.parent / "out" / "word_placer" / f"shift_{a.set}_{a.la}.json"
    out.write_text(json.dumps(allrows, indent=1), encoding="utf-8")

    def summ(rs, label):
        if not rs:
            print(f"  {label}: none")
            return
        g = sorted(r["gain_rel"] for r in rs if r["gain_rel"] is not None)
        s = sorted(r["ta_share"] for r in rs)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")  # noqa: E731
        print(f"  {label}: n {len(rs)}  gain/letter rel. median p10 {q(g, .1):.2f} p50 {q(g, .5):.2f} p90 {q(g, .9):.2f}"
              f"  TA share p10 {q(s, .1):.2f} p50 {q(s, .5):.2f} p90 {q(s, .9):.2f}  TA boxes {sum(r['ta_boxes'] for r in rs)}"
              f" (correct {sum(r['ta_correct'] for r in rs)}, wrong line {sum(r['ta_wrong_line'] for r in rs)})")
    tr = [r for r in allrows if r["trusted"]]
    print("\nTRUSTED PAIRINGS")
    summ([r for r in tr if not r["shifted"] and not r["unprinted"]], "correct line")
    summ([r for r in tr if r["shifted"] and r["own_det"]], "shifted, truth line has its own detected line")
    summ([r for r in tr if r["shifted"] and not r["own_det"]], "shifted, truth line has no detected line of its own")
    summ([r for r in tr if r["unprinted"]], "unprinted text (invented line, repeated copy)")
    print("  shifted by perturbation:", dict(Counter(r["perturb"] for r in tr if r["shifted"])))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
