"""Held-out reading test for a reader from another vendor (plan Step 2b).

  ./venv/bin/python tests/heldout_vendor.py read    # paid: 2 runs x hadpi p3-p5, arm gg_carbon
  ./venv/bin/python tests/heldout_vendor.py score   # $0: frozen scorer, owner's audited truth

The arm is `gg_carbon` of tests/transcription_experiments.py exactly as it ran
on the development pages (same bands, instruction and schema as `t1r_opus`).
`read` writes `heldout_freeze_2b.json` before the first call (arm configuration
hash, script hash, date) and refuses to run if the arm's configuration no longer
matches an existing record. Unlike the first held-out test, the truth already
exists, so the freeze only shows that nothing about the arm changed after its
development run. Scoring uses the frozen scorer copy of the first test
(tests/heldout_score.load_frozen), so the numbers are comparable with
`heldout_summary.txt`. Readings and results stay under out/ (page text is private).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out" / "transcription_experiments"
FREEZE_2B = OUT / "heldout_freeze_2b.json"
RESULTS_2B = OUT / "heldout_results_2b.json"
ARM, REPS, PIDS = "gg_carbon", (0, 1), ("h3", "h4", "h5")


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def arm_config(TE) -> dict:
    r = TE.build_request(ARM, "h1", 0)          # built on a development page
    return {"arm": ARM, "model": r.model, "vendor": r.vendor, "extra": r.extra, "max_tokens": r.max_tokens,
            "system_prompt_sha256": _sha(r.system),
            "schema_sha256": _sha(json.dumps(r.schema.model_json_schema(), sort_keys=True)),
            "resolution": TE.GOOGLE_RESOLUTION, "reps": list(REPS)}


def cmd_read() -> None:
    import transcription_experiments as TE
    cfg = arm_config(TE)
    if FREEZE_2B.is_file():
        if json.loads(FREEZE_2B.read_text())["configuration"] != cfg:
            raise SystemExit("arm configuration changed since heldout_freeze_2b.json; refusing")
    else:
        FREEZE_2B.write_text(json.dumps({
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": TE._git("rev-parse", "HEAD"),
            "script_sha256": _sha((HERE / "transcription_experiments.py").read_text(encoding="utf-8")),
            "configuration": cfg,
            "note": "development readings of this arm (h1, h2 x2; c23, c109) were scored before this freeze"}, indent=1))
    TE._HELDOUT_RELEASED = True
    runner = TE.runner_2b()
    for rep in REPS:
        for pid in PIDS:
            try:
                rec = runner.call(TE.build_request(ARM, pid, rep))
            except (TE.BudgetExceeded, TE.AccessError, TE.VendorSkipped) as exc:
                print(f"STOP: {type(exc).__name__}: {TE.redact(str(exc))}")
                return
            except RuntimeError as exc:
                print(f"{pid} rep{rep}: FAILED: {TE.redact(str(exc))}")
                continue
            print(f"{ARM} {pid} rep{rep} hit={rec['cache_hit']} in={rec['usage']['input_tokens']} "
                  f"out={rec['usage']['output_tokens']} ${rec['cost']:.4f} ledger_2b ${runner.ledger.total():.4f}", flush=True)


def cmd_score() -> None:
    import heldout_score as HS
    truth_info = HS.audited_or_exit()
    FZ = HS.load_frozen()                        # frozen scorer; also blocks Anthropic calls
    import transcription_experiments as TE
    TE._HELDOUT_RELEASED = True
    runner = TE.runner_2b()
    first = json.loads((OUT / "heldout_results.json").read_text())
    res = {"truth": truth_info, "arm": arm_config(TE), "pages": {}, "missing": []}
    tot: dict = {}
    print(f"{'arm':10s} {'page':4s} {'run':3s} {'strict':>14s} {'ortho':>14s} {'lenient':>14s} {'invented':>8s}  cost")
    for pid in PIDS:
        rows = []
        for rep in REPS:
            rec = runner.cached(TE.build_request(ARM, pid, rep))
            if not rec:
                res["missing"].append([pid, rep])
                continue
            bands = sorted(rec["parsed"]["bands"], key=lambda b: b["band_index"])
            lines, warns = FZ.stitch_bands([b["lines"] for b in bands])
            row = {"rep": rep, "cost": rec["cost"], "stitch_warnings": len(warns),
                   **FZ.score_output(pid, {"md": None, "lines": lines}, "lines")}
            rows.append(row)
        res["pages"][pid] = rows
        for name, src in ((ARM, rows), ("t1r_opus", first["pages"][pid]["t1r_opus"])):
            for row in src:
                cells = []
                for f in FZ.FOLDINGS:
                    e, n = row[f]["errors"], row[f]["n_truth"]
                    cells.append(f"{e:g}/{n} {100 * e / n:4.1f}%")
                    t = tot.setdefault((name, f), [0.0, 0])
                    t[0] += e
                    t[1] += n
                print(f"{name:10s} {pid:4s} {row['rep']:<3d} " + " ".join(f"{c:>14s}" for c in cells)
                      + f" {row['invented']:8d}  " + (f"${row['cost']:.4f}" if "cost" in row else ""))
    print("\nPooled over pages and runs (errors / truth words):")
    res["pooled"] = {}
    for name in (ARM, "t1r_opus"):
        res["pooled"][name] = {f: tot[(name, f)] for f in FZ.FOLDINGS if (name, f) in tot}
        print(f"  {name:10s} " + "  ".join(f"{f} {v[0]:g}/{v[1]} = {100 * v[0] / v[1]:.1f}%" for f, v in res["pooled"][name].items()))
    if res["missing"]:
        print("missing readings:", res["missing"])
    RESULTS_2B.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    {"read": cmd_read, "score": cmd_score}[sys.argv[1] if len(sys.argv) > 1 else ""]()
