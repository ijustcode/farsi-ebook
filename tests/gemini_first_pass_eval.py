"""Gemini as the first (whole-page, Markdown) reader: development pages only.

  ./venv/bin/python tests/gemini_first_pass_eval.py run     # paid (Google), cached
  ./venv/bin/python tests/gemini_first_pass_eval.py score   # $0

Calls llm.transcribe_page exactly as `transcribe` does (same instruction,
schema and hi-res page image) with MODEL_HARD_GEMINI, on hadpi p1-p2 (two runs)
and majid p23/p109, and scores beside the cached Sonnet first passes (arm
`current`). Cache: out/transcription_experiments/gemini_first_pass/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import transcription_experiments as TE  # noqa: E402
from farsi2epub import config, llm  # noqa: E402

CACHE = TE.OUT / "gemini_first_pass"
PLAN = [("h1", 0), ("h1", 1), ("h2", 0), ("h2", 1), ("c23", 0), ("c109", 0)]


def run() -> None:
    import yaml
    CACHE.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for pid, rep in PLAN:
        path = CACHE / f"{pid}_{rep}.json"
        if path.is_file():
            continue
        slug, page = TE.PAGES[pid]
        meta = yaml.safe_load((ROOT / "books" / slug / "book.yaml").read_text())
        png = (ROOT / "books" / slug / "pages" / "hires" / f"{page:04d}.png").read_bytes()
        result, usage, cost = llm.transcribe_page(None, png, config.MODEL_HARD_GEMINI, page, meta["page_count"], meta.get("title_fa"))
        path.write_text(json.dumps({"pid": pid, "rep": rep, "parsed": result.model_dump(), "usage": usage, "cost": cost},
                                   ensure_ascii=False, indent=1))
        total += cost
        print(f"{pid} rep{rep} in={usage['input_tokens']} out={usage['output_tokens']} ${cost:.4f}", flush=True)
    print(f"spent now ${total:.4f}")


def score() -> None:
    runner = TE.Runner()
    print(f"{'page':5s} {'reader':22s} {'run':3s} {'strict':>14s} {'lenient':>14s} {'conf':>5s}  cost")
    for pid in ("h1", "h2", "c23", "c109"):
        truth = TE.truth_tokens(pid)
        rows = []
        for p, rep in PLAN:
            f = CACHE / f"{p}_{rep}.json"
            if p == pid and f.is_file():
                r = json.loads(f.read_text())
                rows.append(("gemini whole page", rep, r["parsed"]["text_md"], r["parsed"]["confidence"], r["cost"]))
        for rep in (0, 1, 2):
            rec = runner.cached(TE.build_request("current", pid, rep))
            if rec:
                rows.append(("sonnet whole page", rep, rec["parsed"]["text_md"], rec["parsed"]["confidence"], rec["cost"]))
        for name, rep, md, conf, cost in rows:
            hyp = TE.tokenize_md(md)
            s, l = TE.wer(truth, hyp, "strict"), TE.wer(truth, hyp, "lenient")
            print(f"{pid:5s} {name:22s} {rep:<3d} {s['errors']:g}/{s['n_truth']} {100 * s['wer']:4.1f}%".ljust(48)
                  + f"{l['errors']:g}/{l['n_truth']} {100 * l['wer']:4.1f}%".rjust(14) + f" {conf:5.2f}  ${cost:.4f}")


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1] if len(sys.argv) > 1 else ""]()
