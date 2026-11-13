#!/usr/bin/env python3
"""End to end with model-read line text (Step 4 follow-up 2b). $0.

The line text comes from the cached Step 2 responses (``out/transcription_
experiments/cache``), read directly through ``transcription_experiments``'
own loaders (``Runner.cached``, ``output_of``/``stitch_bands`` for band arms,
``perline_output`` for line-crop arms). Nothing is ever sent: a missing
response raises, and callers set ANTHROPIC_BASE_URL to a dead address.

Pipeline per page and reader: model lines (one string per printed line, as
read; empty lines dropped) -> L1 lines -> template-scored line-text
assignment (word_placer_la) -> arms (word_placer_eval.run_T, experiment C
path: L1 lines with given text) -> side-edge measure against the audited boxes.

Counting. Model tokens are aligned to the truth tokens page-wide on folded
text (``real_truth_eval._fold``, difflib, equal runs and equal-length
replacements position by position, as ``word_placer_eval.map_truth_to_tokens``):
* a truth word whose aligned model token is equal (folded) is *read*;
  "located and correct" needs it read, located and side-edge correct;
* a truth word aligned to a different token is *misread*: a box shown for it
  is counted separately (with how many sit on the right ink by the side-edge
  test): the reviewer would see wrong text pointing at the right place;
* a truth word with no aligned model token has *no counterpart*: it cannot be
  located and counts against recall;
* a model token with no truth counterpart (invented, page number, split) is
  *extra*; boxes shown for those are counted.
Precision = located-and-correct / boxes located for read words.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

READERS = ("t1r_opus", "pl_o", "t1r")
PIDS = {("hadpi-articles", 1): "h1", ("hadpi-articles", 2): "h2",
        ("ghessehaye-majid", 23): "c23", ("ghessehaye-majid", 109): "c109"}


def model_lines(reader: str, pid: str, rep: int = 0) -> list[str]:
    import transcription_experiments as TE
    runner = TE.Runner()
    if reader in TE.PERLINE_ARMS:
        o = TE.perline_output(runner, reader, pid, rep, call=False)
    else:
        rec = runner.cached(TE.build_request(reader, pid, rep))
        o = TE.output_of(rec, reader) if rec is not None else None
    if o is None or o.get("lines") is None:
        raise RuntimeError(f"no cached response for {reader} {pid} rep {rep} (refusing to call the API)")
    return [l for l in o["lines"] if l and l.strip()]


def align_to_truth(truth_texts: list[list[str]], lines: list[list[str]]):
    """-> (src per model token: truth key or ('x', l, t)), status per truth key)."""
    import real_truth_eval as rt
    tkeys = [(li, wi) for li, toks in enumerate(truth_texts) for wi in range(len(toks))]
    tf = [rt._fold(truth_texts[li][wi]) for li, wi in tkeys]
    mkeys = [(l, t) for l, toks in enumerate(lines) for t in range(len(toks))]
    mf = [rt._fold(lines[l][t]) for l, t in mkeys]
    src = [[("x", l, t) for t in range(len(toks))] for l, toks in enumerate(lines)]
    status = {k: "missing" for k in tkeys}
    sm = difflib.SequenceMatcher(None, tf, mf, autojunk=False)
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            for k in range(a1 - a0):
                tk, (l, t) = tkeys[a0 + k], mkeys[b0 + k]
                src[l][t] = tk
                status[tk] = "read" if tf[a0 + k] == mf[b0 + k] else "misread"
    return src, status


def score(page, arms, status, names):
    import real_truth_eval as rt
    tp = page["truth_page"]
    graded = [(w.li, w.wi) for w in tp.graded_words()]
    out = {}
    for nm in names:
        pw = arms.get(nm)
        if pw is None:
            continue
        c = dict(n=len(graded), read=0, misread=0, missing=0, correct=0, located_read=0, misread_located=0,
                 misread_right_place=0, extra_located=0, wrong_line=0)
        for k in graded:
            st = status.get(k, "missing")
            c[st] += 1
            x = pw.get(k)
            if not x or x["status"] != "located":
                continue
            h = rt.grade_target_h1(tp, [k], x)
            c["wrong_line"] += h.get("reason") == "wrong line"
            if st == "read":
                c["located_read"] += 1
                c["correct"] += bool(h["correct"])
            elif st == "misread":
                c["misread_located"] += 1
                c["misread_right_place"] += bool(h["correct"])
        c["extra_located"] = sum(1 for k, x in pw.items() if isinstance(k, tuple) and k and k[0] == "x"
                                 and x["status"] == "located")
        out[nm] = c
    return out


def run(keys=None, readers=READERS, names=("T", "ST", "TA", "TQ", "TAQ", "M3", "RG", "S")):
    if os.environ.get("ANTHROPIC_BASE_URL") != "http://127.0.0.1:9":
        raise SystemExit("set ANTHROPIC_BASE_URL=http://127.0.0.1:9 (no network)")
    import word_placer as WP
    import word_placer_eval as E
    E.T_OPTS.update(vert="comp", la="T")
    rows = []
    for (slug, pg), pid in PIDS.items():
        if keys and f"{slug}/{pg}" not in keys:
            continue
        page = E.real_page(slug, pg)
        ev = WP.page_evidence(page["gray"])
        for reader in ("truth",) + tuple(readers):
            t0 = time.perf_counter()
            if reader == "truth":
                lines = [list(t) for t in page["texts"]]
            else:
                lines = [l.split() for l in model_lines(reader, pid)]
            src, status = align_to_truth(page["texts"], lines)
            arms, info = E.run_T(page, "C", ev=ev, texts=lines, src=src)
            sc = score(page, arms, status, names)
            row = dict(page=page["name"], reader=reader, model_lines=len(lines), truth_lines=len(page["texts"]),
                       la=info.get("la_template"), g2_text_lines_on_nontext=info.get("g2_text_lines_on_nontext"),
                       g2_boxes_on_nontext=info.get("g2_boxes_on_nontext", {}), seconds=round(time.perf_counter() - t0, 1),
                       arms=sc)
            rows.append(row)
            t = sc.get("T", {})
            print(f"  {page['name'][:22]:22s} {reader:9s} lines {len(lines)}/{len(page['texts'])}  read {t.get('read')}"
                  f" misread {t.get('misread')} missing {t.get('missing')}  " +
                  "  ".join(f"{nm} {v['correct']}/{v['n']} p {v['correct']}/{v['located_read']}" for nm, v in sc.items()),
                  flush=True)
    return rows


def main(argv=None):
    import argparse
    import datetime as dt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--readers", nargs="*", default=list(READERS))
    ap.add_argument("--note", required=True)
    a = ap.parse_args(argv)
    rows = run(a.keys, a.readers)
    out = ROOT / "out" / "word_placer" / f"e2e_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                   measure="side-edge (word_correct_h1)", note=a.note, rows=rows),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    with (ROOT / "out" / "word_placer" / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] e2e model-read text -> {out.name}"
                f"  readers {a.readers}; note: {a.note}\n")
        for r in rows:
            f.write(f"  {r['page']:24s} {r['reader']:9s} " + "  ".join(
                f"{nm} {v['correct']}/{v['n']} p {v['correct']}/{v['located_read']}" for nm, v in r["arms"].items()) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
