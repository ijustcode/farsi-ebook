"""$0 probe: does word disagreement with an Opus band reading mark the first reader's errors?

  ./venv/bin/python tests/review_disagreement_probe.py

Reads only caches (no network; ANTHROPIC_BASE_URL is pointed at a dead port and
Google is never called). Prints counts only, never page text.

Primary reading = Gemini 4 Carbon whole page Markdown (the production first
pass): development cache out/transcription_experiments/gemini_first_pass/
{pid}_{rep}.json (hadpi h1, h2; majid c23, c109), held-out cache
gemini_first_pass_heldout/ (hadpi h3, h4, h5).
Second reading = Opus 5.5 on fixed bands (arm `t1r_opus`), every cached run:
development via tests/transcription_experiments.py (TE), held-out via the
frozen scorer copy (heldout_score.load_frozen), as tests/heldout_score.py does.
Every (primary run x second run) pair per page. Flags come from t2_flags
(lenient folding: joins, ZWNJ and punctuation are not disagreements); the
primary's real errors from lenient_alignment against owner truth; an error is
caught when it lies at a flag (_covered). Held-out pages use the frozen copy's
functions and truth loader; development pages use TE's (the T2 code is
identical in both).
USEFUL FLAGS = flags (flagged words + flagged gaps) that cover at least one real
error event, by the same rule as _covered read from the flag side.
Reference rows, no criterion: the same measures with second = Gemini bands
(`gg_carbon`) where cached.

CRITERIA, FIXED 2026-11-16 BEFORE THE FIRST RUN (owner rule D25). Pooled over all run pairs of a class.
  RECALL    share of the primary's real errors that lie at a flag.
            PASS: held-out hard >= 70% AND dev hard >= 60%. KILL: held-out hard < 50%.
  BURDEN    flags / primary words. PASS: every clean pair <= 2% AND held-out hard pooled <= 10% AND dev hard pooled <= 10%.
  USEFUL    useful flags / flags on hard pages (dev and held-out pooled together).
            PASS: >= 35%. (The Sonnet reviewer's suggestions were right about 20% of the time on the owner's 20 marked pages; a flag where the second reader is the one in error still costs the reviewer one look.)
  Outcome:  all three PASS -> propose the paid test on the owner's 20 marked pages.
            RECALL KILL -> word disagreement with an Opus band reading is not proposed as the reviewer.
            anything else -> inconclusive; report which part failed.
  Held-out pages are scored once and nothing is tuned on them.
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
import heldout_score as HS  # noqa: E402

FP_DEV = TE.OUT / "gemini_first_pass"
FP_HO = TE.OUT / "gemini_first_pass_heldout"
DEV_HARD, HELDOUT, CLEAN = ("h1", "h2"), ("h3", "h4", "h5"), ("c23", "c109")
MAX_REP = 6


def primary_runs(M, folder: Path, pid: str) -> list:
    out = []
    for rep in range(MAX_REP):
        f = folder / f"{pid}_{rep}.json"
        if f.is_file():
            r = json.loads(f.read_text())
            out.append((rep, M.tokenize_md(r["parsed"]["text_md"])))
    return out


def useful_flags(events, ftok: set, fgap: set) -> int:
    """Flags that cover at least one error event (_covered, read from the flag side)."""
    hit_tok, hit_gap = set(), set()
    for e in events:
        if e.kind in ("sub", "ins"):
            if e.hyp in ftok:
                hit_tok.add(e.hyp)
        else:
            if e.hyp in fgap:
                hit_gap.add(e.hyp)
            if e.hyp in ftok:
                hit_tok.add(e.hyp)
            if (e.hyp - 1) in ftok:
                hit_tok.add(e.hyp - 1)
    return len(hit_tok) + len(hit_gap)


def pair_stats(M, truth, primary, second) -> dict:
    ftok, fgap = M.t2_flags(primary, second)
    _, h, _, events = M.lenient_alignment(truth, primary)
    caught = sum(1 for e in events if M._covered(e, ftok, fgap))
    flags = len(ftok) + len(fgap)
    return {"primary_words": len(h), "flags": flags, "flagged_words": len(ftok), "flagged_gaps": len(fgap),
            "errors": len(events), "caught": caught, "useful_flags": useful_flags(events, ftok, fgap),
            "truth_words": len(truth)}


def _pct(a, b) -> str:
    return f"{a}/{b} = {100 * a / b:5.1f}%" if b else f"{a}/0"


def main() -> int:
    truth_info = HS.audited_or_exit()
    FZ = HS.load_frozen()                         # frozen scorer; blocks Anthropic calls
    TE._HELDOUT_RELEASED = True                   # held-out gg_carbon cache keys are built by TE (as heldout_vendor.py)
    dev_runner, runner_2b = TE.Runner(), TE.runner_2b()
    ho_runner = FZ.Runner(cap=FZ.CAP_HELDOUT_USD, ledger_name=FZ.LEDGER_HELDOUT)

    def second_runs(pid: str, arm: str) -> list:
        out = []
        for rep in range(MAX_REP):
            if pid in HELDOUT:
                if arm == "t1r_opus":
                    rec = ho_runner.cached(FZ.build_request(arm, pid, rep))
                    if rec:
                        out.append((rep, FZ.hyp_tokens(FZ.output_of(rec, arm), "lines")))
                else:
                    rec = runner_2b.cached(TE.build_request(arm, pid, rep))
                    if rec:
                        bands = sorted(rec["parsed"]["bands"], key=lambda b: b["band_index"])
                        lines, _ = FZ.stitch_bands([b["lines"] for b in bands])
                        out.append((rep, FZ.tokenize_lines(lines)))
            else:
                got = TE._any_tokens(runner_2b if arm != "t1r_opus" else dev_runner, pid, arm, rep)
                if got is not None:
                    out.append((rep, got[0]))
        return out

    rows, missing = [], []
    hdr = (f"{'second':9s} {'class':8s} {'page':5s} {'p':>2s} {'s':>2s} {'words':>5s} {'flags':>17s}"
           f" {'errors':>6s} {'caught':>17s} {'useful flags':>17s}")
    for arm in ("t1r_opus", "gg_carbon"):
        print(f"\n== second reading: {arm}" + ("" if arm == "t1r_opus" else " (reference, no criterion)"))
        print(hdr)
        for cls, pids in (("dev-hard", DEV_HARD), ("held-out", HELDOUT), ("clean", CLEAN)):
            for pid in pids:
                M = FZ if pid in HELDOUT else TE
                prim = primary_runs(M, FP_HO if pid in HELDOUT else FP_DEV, pid)
                sec = second_runs(pid, arm)
                if not prim:
                    missing.append(f"primary gemini whole page {pid}")
                if not sec:
                    missing.append(f"{arm} {pid}")
                truth = M.truth_tokens(pid)
                for pr, ptoks in prim:
                    for sr, stoks in sec:
                        s = pair_stats(M, truth, ptoks, stoks)
                        rows.append({"second": arm, "class": cls, "page": pid, "primary_run": pr, "second_run": sr, **s})
                        print(f"{arm:9s} {cls:8s} {pid:5s} {pr:2d} {sr:2d} {s['primary_words']:5d} "
                              f"{_pct(s['flags'], s['primary_words']):>17s} {s['errors']:6d} "
                              f"{_pct(s['caught'], s['errors']):>17s} {_pct(s['useful_flags'], s['flags']):>17s}")

    def pool(arm, classes):
        rr = [r for r in rows if r["second"] == arm and r["class"] in classes]
        return {"pairs": len(rr), **{k: sum(r[k] for r in rr) for k in
                                      ("primary_words", "flags", "errors", "caught", "useful_flags")}}

    pooled = {}
    for arm in ("t1r_opus", "gg_carbon"):
        print(f"\nPooled, second = {arm}:")
        pooled[arm] = {}
        for name, classes in (("dev-hard", ("dev-hard",)), ("held-out", ("held-out",)), ("clean", ("clean",)),
                              ("hard (dev + held-out)", ("dev-hard", "held-out"))):
            p = pool(arm, classes)
            pooled[arm][name] = p
            print(f"  {name:22s} pairs {p['pairs']:2d}  recall {_pct(p['caught'], p['errors'])}  "
                  f"burden {_pct(p['flags'], p['primary_words'])}  useful {_pct(p['useful_flags'], p['flags'])}")

    if missing:
        print("\nmissing from caches:", ", ".join(missing))

    P = pooled["t1r_opus"]
    ho, dv, hd = P["held-out"], P["dev-hard"], P["hard (dev + held-out)"]
    clean_rows = [r for r in rows if r["second"] == "t1r_opus" and r["class"] == "clean"]
    if not ho["pairs"] or not dv["pairs"] or not clean_rows or not ho["errors"] or not dv["errors"]:
        print("\nNOT JUDGED: a class has no pairs (or no errors)")
        verdict = {"outcome": "NOT JUDGED"}
    else:
        rec_ho, rec_dv = ho["caught"] / ho["errors"], dv["caught"] / dv["errors"]
        if rec_ho < 0.50:
            recall = "KILL"
        elif rec_ho >= 0.70 and rec_dv >= 0.60:
            recall = "PASS"
        else:
            recall = "FAIL"
        clean_max = max(r["flags"] / r["primary_words"] for r in clean_rows)
        bur_ho, bur_dv = ho["flags"] / ho["primary_words"], dv["flags"] / dv["primary_words"]
        burden = "PASS" if clean_max <= 0.02 and bur_ho <= 0.10 and bur_dv <= 0.10 else "FAIL"
        use = hd["useful_flags"] / hd["flags"] if hd["flags"] else 0.0
        useful = "PASS" if use >= 0.35 else "FAIL"
        if recall == "PASS" and burden == "PASS" and useful == "PASS":
            outcome = "ALL PASS -> propose the paid test on the owner's 20 marked pages"
        elif recall == "KILL":
            outcome = "RECALL KILL -> word disagreement with an Opus band reading is not proposed as the reviewer"
        else:
            failed = [n for n, v in (("RECALL", recall), ("BURDEN", burden), ("USEFUL", useful)) if v != "PASS"]
            outcome = "INCONCLUSIVE (not passed: " + ", ".join(failed) + ")"
        print("\nCriteria (fixed before the first run):")
        print(f"RECALL: {recall} (held-out hard {100 * rec_ho:.1f}% >= 70%?, dev hard {100 * rec_dv:.1f}% >= 60%?; kill < 50% held-out)")
        print(f"BURDEN: {burden} (highest clean pair {100 * clean_max:.2f}% <= 2%?, held-out hard {100 * bur_ho:.1f}% <= 10%?, "
              f"dev hard {100 * bur_dv:.1f}% <= 10%?)")
        print(f"USEFUL: {useful} (hard pages {hd['useful_flags']}/{hd['flags']} = {100 * use:.1f}% >= 35%?)")
        print(f"OUTCOME: {outcome}")
        verdict = {"recall": recall, "burden": burden, "useful": useful, "outcome": outcome,
                   "recall_heldout": rec_ho, "recall_dev": rec_dv, "burden_clean_max": clean_max,
                   "burden_heldout": bur_ho, "burden_dev": bur_dv, "useful_share": use}
    out = TE.OUT / "review_disagreement_probe.json"
    out.write_text(json.dumps({"truth": truth_info, "rows": rows, "pooled": pooled, "missing": missing,
                               "verdict": verdict, "criteria": __doc__[__doc__.index("CRITERIA"):]},
                              ensure_ascii=False, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
