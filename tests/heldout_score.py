"""Held-out reading test, stage 2: score the frozen readings of hadpi p3-p5.

Run once, after the owner's text audit says ``status: audited`` for all three
pages:  ./venv/bin/python tests/heldout_score.py

It makes no API call. Scoring uses the byte-exact copy of the experiment
script taken at the freeze (``out/transcription_experiments/
heldout_frozen_script.py``, SHA checked against ``heldout_freeze.json``), so
the scorer is the frozen one even though ``tests/transcription_experiments.py``
has changed since. Readings come from the disk cache only; a missing reading is
reported, never fetched. Results: ``out/transcription_experiments/
heldout_results.json`` and ``heldout_summary.txt`` (page text stays private).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out" / "transcription_experiments"
FROZEN = OUT / "heldout_frozen_script.py"
FREEZE = OUT / "heldout_freeze.json"
TRUTH = ROOT / "books" / "hadpi-articles" / "truth"


def load_frozen() -> types.ModuleType:
    freeze = json.loads(FREEZE.read_text())
    src = FROZEN.read_text(encoding="utf-8")
    if hashlib.sha256(src.encode("utf-8")).hexdigest() != freeze["script"]["sha256"]:
        raise SystemExit("frozen script copy does not match heldout_freeze.json; refusing")
    os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"   # no call can leave the machine
    sys.path.insert(0, str(HERE))
    mod = types.ModuleType("heldout_frozen")
    mod.__file__ = str(HERE / "transcription_experiments.py")   # so ROOT and sibling imports resolve
    sys.modules["heldout_frozen"] = mod
    exec(compile(src, str(FROZEN), "exec"), mod.__dict__)
    mod._HELDOUT_RELEASED = True     # the check above replaces release_held_out()
    return mod


def audited_or_exit() -> dict:
    info = {}
    for page in (3, 4, 5):
        st = json.loads((TRUTH / f"{page:04d}.text_audit.json").read_text())
        txt = (TRUTH / f"{page:04d}.lines.txt").read_bytes()
        sha = hashlib.sha256(txt).hexdigest()
        if st.get("status") != "audited" or not st["export"].get("final") or st["export"]["sha256"] != sha:
            raise SystemExit(f"page {page}: text truth is not a final audited export; refusing to score")
        info[f"h{page}"] = {"truth_sha256": sha, "audit_revision": st["revision"], "audited_at": st["updated_at"]}
    return info


def main() -> None:
    results_path = OUT / "heldout_results.json"
    truth_info = audited_or_exit()
    TE = load_frozen()
    runner = TE.Runner(cap=TE.CAP_HELDOUT_USD, ledger_name=TE.LEDGER_HELDOUT)
    spent_before = runner.ledger.total()
    res = {"truth": truth_info, "pages": {}, "missing": [], "draft": {}, "pairs": {},
           "frozen_script_sha256": json.loads(FREEZE.read_text())["script"]["sha256"]}
    toks = {}
    for pid in TE.HELDOUT_PIDS:
        truth = TE.truth_tokens(pid)
        pr = {"n_truth_words": len(truth),
              "unreadable_in_truth": sum(1 for t in truth if t.text == TE.MARK_UNK)}
        for arm, reps in TE.HELDOUT_PLAN:
            rows = []
            for rep in reps:
                rec = runner.cached(TE.build_request(arm, pid, rep))
                if not rec:
                    res["missing"].append([arm, pid, rep])
                    continue
                out = TE.output_of(rec, arm)
                prefer = "md" if out.get("md") is not None else "lines"
                rows.append({"rep": rep, **TE.score_output(pid, out, prefer)})
                toks[(arm, pid, rep)] = TE.hyp_tokens(out, prefer)
            pr[arm] = rows
        res["pages"][pid] = pr
        page = TE.PAGES[pid][1]
        draft = (TRUTH / f"{page:04d}.lines.draft.txt").read_text(encoding="utf-8").split("\n")
        res["draft"][pid] = TE.draft_vs_truth(truth, draft)
        pairs = {}
        for name, a, b in (("opus run 1 vs opus run 2", ("t1r_opus", 0), ("t1r_opus", 1)),
                           ("opus run 1 vs sonnet run 1", ("t1r_opus", 0), ("t1r", 0))):
            if (a[0], pid, a[1]) in toks and (b[0], pid, b[1]) in toks:
                pairs[name] = TE.t2_stats(truth, toks[(a[0], pid, a[1])], toks[(b[0], pid, b[1])])
        res["pairs"][pid] = pairs
    res["spend_usd"] = runner.ledger.total() - spent_before
    results_path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    L = ["Held-out reading test, hadpi p3-p5, frozen configurations, cached readings, $%.2f spent now"
         % res["spend_usd"], ""]
    L.append(f"{'arm':10s} {'page':4s} {'run':3s} {'words':>5s} {'strict':>12s} {'ortho':>12s} {'lenient':>12s} {'invented':>8s}")
    tot = {}
    for pid, pr in res["pages"].items():
        for arm, _ in TE.HELDOUT_PLAN:
            for row in pr[arm]:
                cells = []
                for f in TE.FOLDINGS:
                    e, n = row[f]["errors"], row[f]["n_truth"]
                    cells.append(f"{e:g}/{n} {100 * e / n:4.1f}%")
                    t = tot.setdefault((arm, f), [0.0, 0])
                    t[0] += e
                    t[1] += n
                L.append(f"{arm:10s} {pid:4s} {row['rep']:<3d} {pr['n_truth_words']:5d} "
                         + " ".join(f"{c:>12s}" for c in cells) + f" {row['invented']:8d}")
    L.append("")
    L.append("Pooled over pages and runs (errors / truth words):")
    for arm, _ in TE.HELDOUT_PLAN:
        L.append(f"  {arm:10s} " + "  ".join(
            f"{f} {tot[(arm, f)][0]:g}/{tot[(arm, f)][1]} = {100 * tot[(arm, f)][0] / tot[(arm, f)][1]:.1f}%"
            for f in TE.FOLDINGS if (arm, f) in tot))
    L.append("")
    L.append("Draft handed to the owner, against the owner's truth (errors outside markers):")
    for pid, d in res["draft"].items():
        L.append(f"  {pid}: strict {d['strict']['unmarked']} unmarked of {d['strict']['errors']} errors; "
                 f"lenient {d['lenient']['unmarked']} unmarked of {d['lenient']['errors']}; "
                 f"markers with a matching alternative {d['markers_with_a_matching_alternative']}")
    L.append("")
    L.append("Two readings as a review flag:")
    for pid, pairs in res["pairs"].items():
        for name, st in pairs.items():
            L.append(f"  {pid} {name}: {json.dumps(st, ensure_ascii=False, default=str)}")
    if res["missing"]:
        L.append("")
        L.append(f"MISSING readings (not fetched): {res['missing']}")
    text = "\n".join(L)
    (OUT / "heldout_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
