#!/usr/bin/env python3
"""Passage-level and group-level `located` (Step 4 follow-up, owner decision
D14). Scoring helpers used by tests/word_placer_e2e.py. $0, no API.

Rules (fixed before any scoring):

* **Agreement** is the existing test: two arms agree on an extent when both
  outer side edges, measured along the line (reading coordinate of the
  detected line), are within 0.5 letter unit (word_placer_s.AGREE_TOL).
* **Word level, TA** (as before): a word is located when T's box and at least
  one of S, G, R agree on its two edges.
* **Word level with group boxes, TAG** (word_placer_s.group_boxes): a word TA
  does not locate is located-as-a-group when, for a run of 2..3 adjacent
  tokens containing it, T and one of S, G, R agree on the run's two outer
  edges; the box is T's union over the run.
* **Passage level, P-pass:** a passage (consecutive tokens of the reading; one
  segment per printed line it touches) is located when, on every segment, T
  and at least one of S, G, R have a box for every token and agree on the
  segment's two outer edges. Boundaries inside the segment are ignored. The
  box is T's union per segment.
* **Passage from words:** P-wTA (every token TA-located; union of their
  boxes) and P-wTAG (every token TAG-located), for comparison.

Scoring (side-edge measure, real_truth_eval.grade_target_h1): a box is
correct when it passes against the union of exactly the truth words its
tokens map to (read and misread tokens: their own truth word; re-segmented
tokens: their whole group; tokens with no truth counterpart contribute no
truth word, so a box over them is judged against the rest). A re-segmented
word with some but not all of its tokens located counts as shown and not
correct, as in word_placer_e2e.score, so word-level precision matches the
committed end-to-end numbers. Spans:
(a) the eval's 60 seeded truth spans per page, mapped to the model's tokens;
(b) review-like spans: lengths drawn from the corrections stored for these
books (QC suggestions' hunks: 75% one word, 7% two, the rest 3-17), seeded,
over the model's tokens; a span holding one piece of a re-segmented word is
extended to the whole group; spans touching a token with no truth
counterpart are skipped and counted.
"""
from __future__ import annotations

import random

import real_truth_eval as rt

# hunk lengths of stored QC suggestions (hadpi p1-p2, majid, bachehaye; 538 hunks; 0-word
# insertions and three whole-page rewrites left out)
SPAN_WEIGHTS = {1: 404, 2: 36, 3: 16, 4: 9, 5: 10, 6: 8, 7: 5, 8: 5, 9: 5, 10: 3, 11: 1, 13: 1, 15: 1, 16: 1, 17: 2}
AGREE_TOL = 0.5


def expand(key, groups_by_id):
    """Truth words a token stands for."""
    if isinstance(key, tuple) and key and key[0] == "g":
        return list(groups_by_id[key[1]]["truth"])
    if isinstance(key, tuple) and key and key[0] == "x":
        return []
    return [tuple(key)]


def claimed(x, src, groups_by_id, own):
    if x.get("_group"):
        out = []
        for t in x["_group"]:
            out += expand(src[x["_tline"]][t], groups_by_id)
        return out
    return own


def score_words(page, pw, src, status, groups, gset_only=True):
    """Per graded truth word: located / correct, group boxes counted, and the
    extra width of group boxes over the word (letters)."""
    import box_audit as ba
    tp = page["truth_page"]
    gb = {g["gid"]: g for g in groups}
    tok_of = {}
    for l, row in enumerate(src):
        for t, k in enumerate(row):
            for tk in expand(k, gb):
                tok_of.setdefault(tk, []).append(k)
    c = dict(n=0, located=0, correct=0, located_group=0, correct_group=0, misread_located=0, partial=0, widths=[])
    for w in tp.graded_words():
        key = (w.li, w.wi)
        c["n"] += 1
        st = status.get(key, "missing")
        if st == "missing":
            continue
        toks = tok_of.get(key, [])
        xs = [pw.get(k) for k in toks]
        shown = [x is not None and x["status"] == "located" and bool(x["segments"]) for x in xs]
        if not xs or not any(shown):
            continue
        if not all(shown):
            # some but not all tokens of a re-segmented word located: shown and not correct
            # (same as word_placer_e2e.score's 'partial'); a misread word is never partial
            c["located"] += 1
            c["partial"] += 1
            continue
        if st == "misread":
            c["misread_located"] += 1
            continue
        own = sorted({tk for k in toks for tk in expand(k, gb)})
        cl = set(own)
        for x in xs:
            cl |= set(claimed(x, src, gb, own))
        placed = {"status": "located", "segments": [sg for x in xs for sg in x["segments"]]}
        graded = {(v.li, v.wi) for v in tp.graded_words()}
        ok = rt.grade_target_h1(tp, sorted(cl & graded) or own, placed)["correct"] if (cl & graded) else False
        c["located"] += 1
        c["correct"] += bool(ok)
        if any(x.get("_group") for x in xs):
            c["located_group"] += 1
            c["correct_group"] += bool(ok)
            L = tp.lines[w.li]
            fb = ba.quad_to_frame_box(L.frame, [p for sg in placed["segments"] for p in rt.region_poly(sg, tp.W, tp.H)])
            letter = (w.fbox[2] - w.fbox[0]) / max(1.0, w.letters)
            c["widths"].append(round(((fb[2] - fb[0]) - (w.fbox[2] - w.fbox[0])) / letter, 2))
    return c


def _agree_segment(arms, keys, unit):
    """T and at least one of S, G, R have extents for every token and agree
    on the segment's outer edges."""
    def ext(nm):
        e = []
        for k in keys:
            x = arms[nm].get(k) or {}
            s = x.get("_sext")
            if s is None:
                return None
            e.append(s)
        return min(a for a, _ in e), max(b for _, b in e)
    et = ext("Tall")
    if et is None:
        return False
    tol = AGREE_TOL * unit
    for nm in ("S", "Gall", "R"):
        if nm not in arms:
            continue
        eo = ext(nm)
        if eo is not None and abs(et[0] - eo[0]) <= tol and abs(et[1] - eo[1]) <= tol:
            return True
    return False


def spans_review(lines, src, seed, n=200):
    rnd = random.Random(seed)
    toks = [(l, t) for l, row in enumerate(lines) for t in range(len(row))]
    lens, wts = zip(*SPAN_WEIGHTS.items())
    out = []
    for _ in range(n):
        L = rnd.choices(lens, weights=wts)[0]
        if L > len(toks):
            continue
        a = rnd.randrange(0, len(toks) - L + 1)
        out.append(toks[a:a + L])
    return out


def spans_seeded(page, src, groups):
    """The eval's seeded truth spans mapped to model tokens."""
    gb = {g["gid"]: g for g in groups}
    pos = {}
    for l, row in enumerate(src):
        for t, k in enumerate(row):
            for tk in expand(k, gb):
                pos.setdefault(tk, set()).add((l, t))
    _w, spans = rt.build_targets(page["truth_page"], 7, 60)
    out = []
    for tg in spans:
        ps = set()
        ok = True
        for k in tg.words:
            if k not in pos:
                ok = False
                break
            ps |= pos[k]
        out.append(sorted(ps) if ok else None)
    return out


def score_passages(page, arms, src, groups, spans, unit, word_arms=("TA", "TAG")):
    """P-pass plus passage-from-words (P-w<arm>) for each name in word_arms."""
    tp = page["truth_page"]
    gb = {g["gid"]: g for g in groups}
    graded = {(v.li, v.wi) for v in tp.graded_words()}
    res = {nm: dict(n=0, located=0, correct=0, skipped=0) for nm in ("P-pass",) + tuple(f"P-w{w}" for w in word_arms)}
    for sp in spans:
        for r in res.values():
            r["n"] += 1
        if sp is None:
            for r in res.values():
                r["skipped"] += 1
            continue
        # a token that is one piece of a re-segmented word brings the rest of its group
        # (the review asks about printed words; half a printed word has no truth box)
        gids = {src[l][t][1] for l, t in sp if isinstance(src[l][t], tuple) and src[l][t][0] == "g"}
        if gids:
            sp = sorted(set(sp) | {(l, t) for l, row in enumerate(src) for t, k in enumerate(row)
                                   if isinstance(k, tuple) and k and k[0] == "g" and k[1] in gids})
        keys = [src[l][t] for l, t in sp]
        truth = sorted({tk for k in keys for tk in expand(k, gb)} & graded)
        if not truth or any(isinstance(k, tuple) and k and k[0] == "x" for k in keys):
            for r in res.values():
                r["skipped"] += 1
            continue
        by_line = {}
        for (l, t), k in zip(sp, keys):
            by_line.setdefault(l, []).append(k)
        # passage rule
        if all(_agree_segment(arms, ks, unit) for ks in by_line.values()):
            segs = [sg for k in keys for sg in (arms["Tall"].get(k) or {}).get("segments", [])]
            res["P-pass"]["located"] += 1
            res["P-pass"]["correct"] += bool(rt.grade_target_h1(tp, truth, {"status": "located", "segments": segs})["correct"])
        for w in word_arms:
            xs = [arms[w].get(k) for k in keys]
            if all(x and x["status"] == "located" and x["segments"] for x in xs):
                segs = [sg for x in xs for sg in x["segments"]]
                cl = set(truth)
                for x in xs:
                    cl |= set(claimed(x, src, gb, [])) & graded
                r = res[f"P-w{w}"]
                r["located"] += 1
                r["correct"] += bool(rt.grade_target_h1(tp, sorted(cl), {"status": "located", "segments": segs})["correct"])
    return res
