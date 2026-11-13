#!/usr/bin/env python3
"""Gate G2 development (Step 4 follow-up): line text perturbed the way a
reader may get it wrong (invented lines, repeated lines, dropped lines;
word_placer_la.perturb_lines, seeded) on synthetic pages with non-text
overlays, through L1 lines -> line-text assignment -> arms. Counts, per
assignment variant (--la length | T | T2): boxes shown on detected lines that
hold no truth text (G2), boxes shown for words that are not printed
(invented lines, second copies), wrong-line boxes (side-edge line test) and
side-edge located-and-correct. Synthetic only; no API.

  ./venv/bin/python tests/word_placer_g2.py --set dev
  ./venv/bin/python tests/word_placer_g2.py --set final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_truth_eval as rt  # noqa: E402
import word_placer as WP  # noqa: E402
import word_placer_eval as E  # noqa: E402
import word_placer_la as LA  # noqa: E402

SEEDS = {"dev": [201, 202], "final": [401, 402]}
CONFIGS = [("typewriter", "tw_photo_medium"), ("typewriter", "tw_photo_hard"), ("modern_naskh", "tw_photo_medium"),
           ("book_naskh", "clean")]
PERTURB = [("invent1", dict(n_invent=1, n_repeat=0, n_drop=0)), ("invent2", dict(n_invent=2, n_repeat=0, n_drop=0)),
           ("repeat1", dict(n_invent=0, n_repeat=1, n_drop=0)), ("drop1", dict(n_invent=0, n_repeat=0, n_drop=1)),
           ("mixed", dict(n_invent=1, n_repeat=1, n_drop=1))]


def measure(page, ev, texts, src, la, arms_=("T", "TA", "ST")):
    E.T_OPTS.update(vert="comp", la=la)
    arms, info = E.run_T(page, "C", ev=ev, texts=texts, src=src)
    tp = page["truth_page"]
    out = {}
    for nm in arms_:
        pw = arms[nm]
        wrows, _ = E.grade_page(page, pw)
        correct = sum(r["correct_h1"] for r in wrows)
        located = sum(1 for r in wrows if r["status"] == "located")
        unprinted = sum(1 for k, x in pw.items() if k and k[0] == "x" and x["status"] in ("located", "line") and x["segments"])
        wl = sum(1 for r in wrows if r["status"] == "located"
                 and rt.grade_target_h1(tp, r["target"], pw[tuple(r["target"][0])]).get("reason") == "wrong line")
        out[nm] = dict(n=len(wrows), correct=correct, located=located, unprinted_boxes=unprinted, wrong_line=wl,
                       g2_boxes=info.get("g2_boxes_on_nontext", {}).get(nm, 0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SEEDS), default="dev")
    ap.add_argument("--la", nargs="*", default=["length", "T", "T2"])
    a = ap.parse_args(argv)
    rows = []
    for cls, preset in CONFIGS:
        for sd in SEEDS[a.set]:
            page = E.synth_page(cls, dict(preset=preset), sd)
            ev = WP.page_evidence(page["gray"])
            pool_page = E.synth_page(cls, dict(preset="clean"), sd + 1000)
            pool = [w for t in pool_page["texts"] for w in t]
            cases = [("none", [list(t) for t in page["texts"]],
                      [[(li, wi) for wi in range(len(t))] for li, t in enumerate(page["texts"])])]
            for name, kw in PERTURB:
                t2, s2 = LA.perturb_lines(page["texts"], seed=sd * 7 + len(name), pool=pool, **kw)
                cases.append((name, t2, s2))
            for name, t2, s2 in cases:
                for la in a.la:
                    m = measure(page, ev, t2, s2, la)
                    rows.append(dict(config=f"{cls}/{preset}", seed=sd, perturb=name, la=la, arms=m))
                    t = m["T"]
                    print(f"  {cls}/{preset} s{sd} {name:8s} {la:6s} T {t['correct']}/{t['n']} p {t['correct']}/{t['located']}"
                          f" g2 {t['g2_boxes']} unprinted {t['unprinted_boxes']} wl {t['wrong_line']}"
                          f" | TA {m['TA']['correct']}/{m['TA']['located']} g2 {m['TA']['g2_boxes']} unprinted {m['TA']['unprinted_boxes']}",
                          flush=True)
    out = HERE.parent / "out" / "word_placer" / f"g2_{a.set}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tot = {}
    for r in rows:
        for nm, c in r["arms"].items():
            d = tot.setdefault((r["la"], r["perturb"], nm), dict(n=0, correct=0, located=0, g2=0, unprinted=0, wl=0))
            d["n"] += c["n"]; d["correct"] += c["correct"]; d["located"] += c["located"]
            d["g2"] += c["g2_boxes"]; d["unprinted"] += c["unprinted_boxes"]; d["wl"] += c["wrong_line"]
    print("\nTOTALS (side-edge measure)")
    for (la, pt, nm), d in sorted(tot.items()):
        print(f"  {la:6s} {pt:8s} {nm:3s} correct {d['correct']}/{d['n']} = {d['correct'] / max(1, d['n']):.1%}"
              f" prec {d['correct'] / max(1, d['located']):.1%}  G2 boxes {d['g2']}  boxes for unprinted words {d['unprinted']}"
              f"  wrong-line {d['wl']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
