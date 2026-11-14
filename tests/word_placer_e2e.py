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
        ("ghessehaye-majid", 23): "c23", ("ghessehaye-majid", 109): "c109",
        ("bachehaye_ghali", 66): None}          # p66: no Step 2 readings; truth text only


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


def _regroup(tf, mf, mline, a0, a1, b0, b1, tline=None):
    """Pure re-segmentation inside one replace block: walk truth tokens
    tf[a0:a1] and model tokens mf[b0:b1] accumulating folded letters; a group
    closes when both concatenations are equal. A letter mismatch ends the walk
    (the rest of the block stays unmatched). A group's model tokens must lie
    on one model line and its truth words on one printed (truth) line. Returns [(truth idx list, model idx list)]."""
    out = []
    i, j = a0, b0
    while i < a1 and j < b1:
        gi, gj = [i], [j]
        A, B = tf[i], mf[j]
        ok = True
        while A != B:
            if B and A.startswith(B) and j + 1 < b1:
                j += 1; gj.append(j); B += mf[j]
            elif A and B.startswith(A) and i + 1 < a1:
                i += 1; gi.append(i); A += tf[i]
            else:
                ok = False
                break
        if not ok or len({mline[k] for k in gj}) > 1 or (tline is not None and len({tline[k] for k in gi}) > 1):
            break
        if len(gi) > 1 or len(gj) > 1:
            out.append((gi, gj))
        i += 1; j += 1
    return out


def align_to_truth(truth_texts: list[list[str]], lines: list[list[str]], groups_out: list | None = None):
    """-> (src per model token: truth key, ('g', group, n) or ('x', l, t)), status per truth key).

    Rule: page-wide difflib alignment of folded tokens (autojunk off). Equal
    runs: read. Equal-length replace blocks: position by position, read when
    the folded tokens are equal, else misread. Unequal-length replace blocks:
    split into pure re-segmentation groups (identical folded letters, see
    _regroup): a group of N truth words and one model token is a join, one
    truth word and M tokens a split, N:M a regroup; anything else in the block
    stays unmatched (truth words 'missing', model tokens extra). Groups never
    cross a model line or a printed line. If
    groups_out is a list, the groups are appended to it as
    dict(kind, truth=[keys], model=[(l, t)])."""
    import real_truth_eval as rt
    tkeys = [(li, wi) for li, toks in enumerate(truth_texts) for wi in range(len(toks))]
    tf = [rt._fold(truth_texts[li][wi]) for li, wi in tkeys]
    mkeys = [(l, t) for l, toks in enumerate(lines) for t in range(len(toks))]
    mf = [rt._fold(lines[l][t]) for l, t in mkeys]
    mline = [l for l, _ in mkeys]
    tline = [li for li, _ in tkeys]
    src = [[("x", l, t) for t in range(len(toks))] for l, toks in enumerate(lines)]
    status = {k: "missing" for k in tkeys}
    sm = difflib.SequenceMatcher(None, tf, mf, autojunk=False)
    gid = 0
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            for k in range(a1 - a0):
                tk, (l, t) = tkeys[a0 + k], mkeys[b0 + k]
                src[l][t] = tk
                status[tk] = "read" if tf[a0 + k] == mf[b0 + k] else "misread"
        elif tag == "replace":
            for gi, gj in _regroup(tf, mf, mline, a0, a1, b0, b1, tline):
                kind = "join" if len(gj) == 1 else ("split" if len(gi) == 1 else "regroup")
                for n, k in enumerate(gj):
                    l, t = mkeys[k]
                    src[l][t] = ("g", gid, n)
                for k in gi:
                    status[tkeys[k]] = kind
                if groups_out is not None:
                    groups_out.append(dict(kind=kind, gid=gid, truth=[tkeys[k] for k in gi], model=[mkeys[k] for k in gj]))
                gid += 1
    return src, status


def score(page, arms, status, names, groups=()):
    """Side-edge counts per arm (see the module docstring). Groups (joins,
    splits, regroups of identical letters): every model token located and
    the union of their boxes side-edge correct against the union of the
    group's truth words -> all its graded truth words correct; some but not
    all tokens located -> 'partial' (counted as shown and not correct)."""
    import real_truth_eval as rt
    tp = page["truth_page"]
    graded = [(w.li, w.wi) for w in tp.graded_words()]
    gset = set(graded)
    out = {}
    for nm in names:
        pw = arms.get(nm)
        if pw is None:
            continue
        c = dict(n=len(graded), read=0, misread=0, missing=0, join=0, split=0, regroup=0, correct=0, located_read=0,
                 misread_located=0, misread_right_place=0, extra_located=0, wrong_line=0,
                 group_words_correct=0, group_words_located=0, group_words_partial=0,
                 correct_lower_bound=0)
        for k in graded:
            st = status.get(k, "missing")
            c[st] += 1
            x = pw.get(k)
            if st in ("join", "split", "regroup") or not x or x["status"] != "located":
                continue
            h = rt.grade_target_h1(tp, [k], x)
            c["wrong_line"] += h.get("reason") == "wrong line"
            if st == "read":
                c["located_read"] += 1
                c["correct"] += bool(h["correct"])
            elif st == "misread":
                c["misread_located"] += 1
                c["misread_right_place"] += bool(h["correct"])
        c["correct_lower_bound"] = c["correct"]
        for g in groups:
            tg = [k for k in g["truth"] if k in gset]
            if not tg:
                continue
            xs = [pw.get(("g", g["gid"], n)) for n in range(len(g["model"]))]
            loc = [x for x in xs if x and x["status"] == "located" and x.get("segments")]
            if len(loc) == len(xs):
                placed = {"status": "located", "segments": [sg for x in loc for sg in x["segments"]]}
                h = rt.grade_target_h1(tp, tg, placed)
                c["group_words_located"] += len(tg)
                c["located_read"] += len(tg)
                c["wrong_line"] += h.get("reason") == "wrong line"
                if h["correct"]:
                    c["group_words_correct"] += len(tg)
                    c["correct"] += len(tg)
            elif loc:
                c["group_words_partial"] += len(tg)
                c["located_read"] += len(tg)
        c["extra_located"] = sum(1 for k, x in pw.items() if isinstance(k, tuple) and k and k[0] == "x"
                                 and x["status"] == "located")
        out[nm] = c
    return out


def agreement_on_tokens(arms, src):
    """Of model tokens T shows a box for (every-box arm), the share that S&T
    and TA locate: for tokens matched one to one and for tokens in groups."""
    out = {}
    for kind in ("one_to_one", "group"):
        keys = [k for row in src for k in row
                if (kind == "group" and k[0] == "g") or (kind == "one_to_one" and k[0] not in ("g", "x"))]
        shown = [k for k in keys if (arms["Tall"].get(k) or {}).get("status") == "located"]
        out[kind] = dict(tokens=len(keys), t_shown=len(shown),
                         st_located=sum(1 for k in shown if (arms["ST"].get(k) or {}).get("status") == "located"),
                         ta_located=sum(1 for k in shown if (arms["TA"].get(k) or {}).get("status") == "located"))
    return out


def run(keys=None, readers=READERS, names=("T", "ST", "TA", "TQ", "TAQ", "M3", "RG", "S", "Tall"), la="T"):
    if os.environ.get("ANTHROPIC_BASE_URL") != "http://127.0.0.1:9":
        raise SystemExit("set ANTHROPIC_BASE_URL=http://127.0.0.1:9 (no network)")
    import word_placer as WP
    import word_placer_eval as E
    E.T_OPTS.update(vert="comp", la=la)
    rows = []
    for (slug, pg), pid in PIDS.items():
        if keys and f"{slug}/{pg}" not in keys:
            continue
        page = E.real_page(slug, pg)
        ev = WP.page_evidence(page["gray"])
        for reader in ("truth",) + (tuple(readers) if pid else ()):
            t0 = time.perf_counter()
            if reader == "truth":
                lines = [list(t) for t in page["texts"]]
            else:
                lines = [l.split() for l in model_lines(reader, pid)]
            groups = []
            src, status = align_to_truth(page["texts"], lines, groups)
            arms, info = E.run_T(page, "C", ev=ev, texts=lines, src=src)
            sc = score(page, arms, status, names, groups)
            row = dict(page=page["name"], reader=reader, model_lines=len(lines), truth_lines=len(page["texts"]),
                       la=info.get("la_template"), la_mode=la, g2_text_lines_on_nontext=info.get("g2_text_lines_on_nontext"),
                       g2_boxes_on_nontext=info.get("g2_boxes_on_nontext", {}),
                       g2_text_lines_on_ignored=info.get("g2_text_lines_on_ignored"),
                       g2_boxes_on_ignored=info.get("g2_boxes_on_ignored", {}),
                       g2_text_lines_on_marks=info.get("g2_text_lines_on_marks"),
                       g2_boxes_on_marks=info.get("g2_boxes_on_marks", {}),
                       mark_lines=info.get("mark_lines"), mark_pairings_blocked=info.get("mark_pairings_blocked"),
                       mark_pairings_exempt=info.get("mark_pairings_exempt"), seconds=round(time.perf_counter() - t0, 1),
                       arms=sc, groups=dict(join=sum(g["kind"] == "join" for g in groups),
                                             split=sum(g["kind"] == "split" for g in groups),
                                             regroup=sum(g["kind"] == "regroup" for g in groups)),
                       agreement=agreement_on_tokens(arms, src))
            # D14 follow-up: word level with group boxes, and passage level
            import word_placer_passage as PS
            row["words_D14"] = {nm: PS.score_words(page, arms[nm], src, status, groups) for nm in ("TA", "TAG", "ST")}
            row["passages_seeded"] = PS.score_passages(page, arms, src, groups, PS.spans_seeded(page, src, groups), info["unit"])
            row["passages_review"] = PS.score_passages(page, arms, src, groups,
                                                       PS.spans_review(lines, src, seed=E.hash_seed(page["name"], "review")),
                                                       info["unit"])
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
    ap.add_argument("--la", choices=("T", "T2", "TM", "TZ", "TZ2"), default="T")
    ap.add_argument("--note", required=True)
    a = ap.parse_args(argv)
    rows = run(a.keys, a.readers, la=a.la)
    out = ROOT / "out" / "word_placer" / f"e2e_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                   measure="side-edge (word_correct_h1)", note=a.note, rows=rows),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    with (ROOT / "out" / "word_placer" / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] e2e model-read text -> {out.name}"
                f"  readers {a.readers} la {a.la}; note: {a.note}\n")
        for r in rows:
            f.write(f"  {r['page']:24s} {r['reader']:9s} " + "  ".join(
                f"{nm} {v['correct']}/{v['n']} p {v['correct']}/{v['located_read']}" for nm, v in r["arms"].items()) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
