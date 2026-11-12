#!/usr/bin/env python3
"""Step 4 evaluation: word placement (arm G, arm R, draft reference) on
synthetic and real audited pages, experiments A-D (plan §7 Step 4).

Experiments
  A  truth line geometry + truth line text            (isolates the word placer)
  B  L1 lines + truth line text, aligned by length    (end to end with true text)
  C  as B, text corrupted (substitute/delete/insert words; 5% and 15%; seeded)
  D  L1 lines + realistic text, not line-broken: on real pages the stored
     workspace transcription text/NNNN.md; on synthetic pages the truth text
     with line breaks removed and 15% of words corrupted

Arms
  draft   tests/box_audit_draft.py (reference; its own line finder; real pages:
          the stored, untouched audit draft)
  G0      arm G without the "not text" state
  G       arm G with it
  Gc      arm G, width table quantised to classes when BIC prefers it
  R       arm R (render and warp), when built

Grading is the real scorer's own (``real_truth_eval.grade_target`` and
``summarize_targets``). Synthetic truth: every word's exact ink pixels (from
the generator's id map) measured in a straight frame fitted to its line, so
the truth box is ink-tight like the audited boxes.

No API. Real pages: refuses held-out pages (the scorer does), writes nothing
under books/. Every real-page scoring is appended to out/word_placer/run_log.txt.

  ./venv/bin/python tests/word_placer_eval.py synth --set dev [--exps A B] [--arms G0 G]
  ./venv/bin/python tests/word_placer_eval.py synth --set final
  ./venv/bin/python tests/word_placer_eval.py real [--exps A B C D] --note "why this run"
  ./venv/bin/python tests/word_placer_eval.py overlay-real --out DIR
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import box_audit as ba  # noqa: E402
import box_audit_draft as bd  # noqa: E402
import line_finder as LF  # noqa: E402
import real_truth_eval as rt  # noqa: E402
import word_placer as WP  # noqa: E402

OUT = ROOT / "out" / "word_placer"
SEEDS = {"dev": [201, 202], "final": [401, 402]}

# (label, font class, make_spec kwargs)
def synth_configs(full: bool = True):
    cfg = []
    for cls in ("modern_naskh", "book_naskh", "typewriter", "nastaliq", "mixed"):
        cfg.append((f"{cls}/clean", cls, dict(preset="clean")))
    cfg.append(("verse/clean", "modern_naskh", dict(preset="clean", layout="verse")))
    cfg.append(("typewriter/tw_photo_medium", "typewriter", dict(preset="tw_photo_medium")))
    cfg.append(("typewriter/tw_photo_hard", "typewriter", dict(preset="tw_photo_hard")))
    if full:
        cfg.append(("modern_naskh/tw_photo_medium", "modern_naskh", dict(preset="tw_photo_medium")))
        cfg.append(("book_naskh/photo", "book_naskh", dict(preset="photo")))
    return cfg


HARD = {"typewriter/tw_photo_medium", "typewriter/tw_photo_hard", "modern_naskh/tw_photo_medium", "book_naskh/photo"}


# ---------------------------------------------------------------------------
# Synthetic pages -> real-scorer TruthPage with ink-tight frame boxes
# ---------------------------------------------------------------------------
def synth_page(cls, kw, seed):
    import synth_pages as S
    spec = S.make_spec(cls, seed=seed, **kw)
    g = S.generate(spec, keep_arrays=True)
    truth = g["truth"]
    img = g["image"]
    H, W = img.shape
    idmap = g["idmap"]
    ys, xs = np.nonzero(idmap)
    ids = idmap[ys, xs] - 1
    o = np.argsort(ids, kind="stable")
    ids, xs, ys = ids[o], xs[o].astype(np.float64) + 0.5, ys[o].astype(np.float64) + 0.5
    nt = len(truth["words"])
    st = np.searchsorted(ids, np.arange(nt + 1))
    words = {w["id"]: w for w in truth["words"]}
    lines, texts, visual = [], [], []
    for tl in truth["lines"]:
        wids = [w for w in tl["word_ids"] if st[w + 1] > st[w]]
        if not wids:
            continue
        px = np.concatenate([xs[st[w]:st[w + 1]] for w in wids])
        py = np.concatenate([ys[st[w]:st[w + 1]] for w in wids])
        # straight frame: least-squares line through the baseline points
        bl = np.array(tl["baseline"]) * [W, H]
        k = np.polyfit(bl[:, 0], bl[:, 1], 1)[0] if len(bl) >= 2 and np.ptp(bl[:, 0]) > 1 else 0.0
        ang = math.degrees(math.atan(k))
        frame = {"cx": float((px.min() + px.max()) / 2), "cy": float(np.median(py)), "angle_deg": ang}
        tws = []
        for wi, w in enumerate(wids):
            wx, wy = xs[st[w]:st[w + 1]], ys[st[w]:st[w + 1]]
            fr = [ba.page_to_frame(frame, x, y) for x, y in ((wx.min(), wy.min()), (wx.max(), wy.min()),
                                                             (wx.max(), wy.max()), (wx.min(), wy.max()))]
            a = math.radians(ang); c, s = math.cos(a), math.sin(a)
            uu = (wx - frame["cx"]) * c + (wy - frame["cy"]) * s
            vv = -(wx - frame["cx"]) * s + (wy - frame["cy"]) * c
            fbox = [float(uu.min()) - 0.5, float(vv.min()) - 0.5, float(uu.max()) + 0.5, float(vv.max()) + 0.5]
            q = [tuple(p) for p in ba.box_quad_px(frame, fbox)]
            tws.append(rt.TWord(len(lines), wi, words[w]["text"], q, fbox, rt.letters(words[w]["text"]), True))
        lb = [min(t.fbox[0] for t in tws), min(t.fbox[1] for t in tws), max(t.fbox[2] for t in tws), max(t.fbox[3] for t in tws)]
        L = rt.TLine(len(lines), frame, lb, [tuple(p) for p in ba.box_quad_px(frame, lb)], tws, False)
        L.src = tl
        lines.append(L)
        texts.append([words[w]["text"] for w in wids])
        visual.append([words[w]["visual_index"] for w in wids])
    tp = rt.TruthPage(spec["name"], 0, "synthetic", "synthetic", W, H, Path("."), "", Path("."), "", lines, [])
    return dict(name=spec["name"], gray=img.astype(np.float32), truth_page=tp, texts=texts, visual=visual,
                truth=truth, spec=spec, kind="synth",
                paragraphs=synth_paragraphs(truth, lines), md=None)


def synth_paragraphs(truth, lines):
    """Truth text without line breaks: tokens grouped by generator block."""
    out, cur, blk = [], [], None
    words = {w["id"]: w for w in truth["words"]}
    for L in lines:
        b = words[L.src["word_ids"][0]]["block"]
        if blk is not None and b != blk and cur:
            out.append(cur); cur = []
        blk = b
        cur += [w.text for w in L.words]
    if cur:
        out.append(cur)
    return out


def real_page(slug, page):
    t = rt.load_truth(ROOT, slug, page)
    gray = LF.load_gray(t.image_path)
    pi = rt.build_page_input(ROOT, t, None)
    paras = [p["tokens"] for p in bd.markdown_paragraphs(pi.md)] if pi.md else []
    return dict(name=t.key, gray=gray, truth_page=t, texts=pi.line_texts, kind="real", pi=pi,
                paragraphs=paras, md=pi.md)


# ---------------------------------------------------------------------------
# Running arms
# ---------------------------------------------------------------------------
def truth_geoms(tp: rt.TruthPage):
    gs = []
    for L in tp.lines:
        u0, _, u1, _ = L.box
        gs.append(WP.FrameGeom(L.frame["cx"], L.frame["cy"], L.frame["angle_deg"], u0, u1))
    return gs


def corrupt_lines(texts, rate, seed):
    """Page-level text corruption; returns (corrupted texts, src map
    [(line, token) -> truth (line, word) or None], per-truth-word status)."""
    import synth_pages as S
    flat = [(li, wi, t) for li, toks in enumerate(texts) for wi, t in enumerate(toks)]
    inj = S.inject_text_errors([t for _, _, t in flat], rate, seed)
    out = [[] for _ in texts]
    src = [[] for _ in texts]
    k = 0
    # walk the output words; inserted words go to the line of the next source word
    srcs = inj["src_index"]
    nxt_line = None
    for pos, w in enumerate(inj["words"]):
        s = srcs[pos]
        if s is None:
            # find next source word
            nx = next((x for x in srcs[pos + 1:] if x is not None), None)
            li = flat[nx][0] if nx is not None else len(texts) - 1
            out[li].append(w); src[li].append(None)
        else:
            li, wi, _ = flat[s]
            out[li].append(w); src[li].append((li, wi))
    status = {(flat[i][0], flat[i][1]): st for i, st in enumerate(inj["status"])}
    return out, src, status, inj


def run_G(page, exp, *, nottext=True, width_mode="free", texts=None, src=None, ev=None):
    """Returns (placements per truth word {(li, wi): placed}, info)."""
    tp = page["truth_page"]
    ev = ev or WP.page_evidence(page["gray"])
    W, H = tp.W, tp.H
    texts = texts if texts is not None else page["texts"]
    t0 = time.perf_counter()
    info = {}
    per_word = {}
    extra_located = 0
    if exp in ("A", "B", "C", "A15"):
        if exp in ("A", "A15"):
            r = WP.place_given_lines(ev, truth_geoms(tp), texts, nottext=nottext, width_mode=width_mode)
        else:
            dets = WP.l1_lines(ev)
            r = WP.place_assigned_lines(ev, [g for g, _ in dets], [k for _, k in dets], texts, nottext=nottext,
                                        width_mode=width_mode)
            info["line_assignment"] = line_assignment_report(page, r, dets, texts, src)
        for li, res in enumerate(r["results"]):
            for ti, x in enumerate(res):
                key = src[li][ti] if src is not None else (li, ti)
                pl = to_placed(x, W, H)
                if key is None:
                    extra_located += pl["status"] == "located"
                    continue
                per_word[key] = pl
    elif exp == "D":
        dets = WP.l1_lines(ev)
        paras = page["paragraphs"]
        r = WP.place_unbroken(ev, [g for g, _ in dets], [k for _, k in dets], paras, nottext=nottext,
                              width_mode=width_mode)
        toks = r["tokens"]
        m = map_truth_to_tokens(tp, toks) if src is None else src
        inv = {}
        for key, k in m.items():
            inv.setdefault(k, key)
        for k, x in enumerate(r["results_flat"]):
            pl = to_placed(x, W, H)
            key = inv.get(k)
            if key is None:
                extra_located += pl["status"] == "located"
                continue
            per_word[key] = pl
        info["mapped_words"] = len(m)
        info["md_tokens"] = len(toks)
        info["boundary_margins"] = {str(k): round(v, 2) for k, v in r["boundary_margin"].items()}
    info["seconds_place"] = round(time.perf_counter() - t0, 2)
    info["seconds_l1"] = round(ev.seconds, 2)
    info["model"] = r["diag"]["model"]
    info["rounds"] = r["diag"]["rounds"]
    info["located_on_untargeted_tokens"] = extra_located
    return per_word, info


def map_truth_to_tokens(tp, toks):
    """Truth word (li, wi) -> index in a token stream, by folded-token
    alignment (equal runs; equal-length replacements position by position)."""
    aw = [(w.li, w.wi) for L in tp.lines for w in L.words]
    af = [rt._fold(tp.word(*k).text) for k in aw]
    bf = [rt._fold(t) for t in toks]
    sm = difflib.SequenceMatcher(None, af, bf, autojunk=False)
    out = {}
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            for i in range(a1 - a0):
                out[aw[a0 + i]] = b0 + i
    return out


def to_placed(x, W, H):
    st = x["status"]
    if st == "located" and x.get("quad"):
        segs = [{"quad": [[a / W, b / H] for a, b in x["quad"]]}]
    elif st in ("line", "located") and x.get("line_quad"):
        st = "line"
        segs = [{"quad": [[a / W, b / H] for a, b in x["line_quad"]]}]
    else:
        st, segs = "unresolved", []
    line = [{"quad": [[a / W, b / H] for a, b in x["line_quad"]]}] if x.get("line_quad") else []
    return {"status": st, "segments": segs, "confidence": x.get("margin"), "reason": x.get("reason", ""),
            "_quad": x.get("quad"), "_line": line, "_other_ok": x.get("other_ok", True)}


def line_assignment_report(page, r, dets, texts, src):
    """Which detected line got which text line, against the truth line under
    each detected line (by word-area share)."""
    tp = page["truth_page"]
    regs = []
    for (g, kind), li in zip(dets, r["inks"]):
        if li.M:
            q = WP.line_quad(li, float(li.s0[0]), float(li.s1[-1]), 1.0)
            regs.append({"quad": [[a / tp.W, b / tp.H] for a, b in q]})
        else:
            regs.append(None)
    # truth line under each detected line
    under = []
    for reg in regs:
        if reg is None:
            under.append(None)
            continue
        poly = rt.region_poly(reg, tp.W, tp.H)
        best, bl = 0.0, None
        for L in tp.lines:
            if not rt._overlaps(rt.bounds([L.quad]), rt.bounds([poly])):
                continue
            tot = sum(rt._quad_area(w.quad) for w in L.words)
            sh = sum(rt.fraction_inside(w.quad, [poly]) * rt._quad_area(w.quad) for w in L.words) / max(1e-9, tot)
            if sh > best:
                best, bl = sh, L.index
        under.append(bl if best >= 0.5 else None)
    n_text = len(texts)
    wrong = 0
    missing = 0
    nontext_with_text = 0
    nontext = sum(1 for u in under if u is None)
    for i in range(n_text):
        j = r["assign"][i]
        true_li = i      # text line i is truth line i (line-broken truth text)
        if j is None:
            missing += 1
        elif under[j] != true_li:
            wrong += 1
    for j, u in enumerate(under):
        if u is None and j in [a for a in r["assign"] if a is not None]:
            nontext_with_text += 1
    trusted = sum(1 for i in range(n_text) if r["assign"][i] is not None and r["la"]["margin"][i] >= WP.P["LA_MARGIN"])
    wrong_trusted = sum(1 for i in range(n_text) if r["assign"][i] is not None and r["la"]["margin"][i] >= WP.P["LA_MARGIN"]
                        and under[r["assign"][i]] != i)
    return dict(text_lines=n_text, detected=len(dets), nontext_detected=nontext,
                nontext_detected_given_text=nontext_with_text, text_lines_wrong_detected=wrong,
                text_lines_unassigned=missing, trusted=trusted, wrong_and_trusted=wrong_trusted,
                under=under, assign=list(r["assign"]),
                kinds={k: sum(1 for _, kk in dets if kk == k) for k in sorted({kk for _, kk in dets})})


def run_draft(page):
    """Draft splitter reference. Synthetic: run it (own line finder, truth
    line text). Real: the stored untouched audit draft (rt.DraftWords)."""
    tp = page["truth_page"]
    per_word = {}
    if page["kind"] == "real":
        targets = [rt.Target(f"w{w.li}.{w.wi}", [(w.li, w.wi)], w.text) for w in tp.graded_words()]
        res = rt.DraftWords().place(page["pi"], targets)
        for tg, r in zip(targets, res):
            per_word[tg.words[0]] = r
        return per_word, {}
    H, W = page["gray"].shape
    texts = [" ".join(t) for t in page["texts"]]
    res = bd.draft_known_lines(page["gray"], texts, [False] * len(texts))
    for li, (L, vis) in enumerate(zip(res["lines"], page["visual"])):
        t = L["track"]
        boxes = []
        for (x0, x1, dt_, db_) in L["words"]:
            pts = [(x0, t.y_at(x0) + dt_), (x1, t.y_at(x1) + dt_), (x1, t.y_at(x1) + db_), (x0, t.y_at(x0) + db_)]
            boxes.append(pts)
        # draft boxes are in visual (right-to-left) order
        rank = np.argsort(np.argsort(vis))
        for wi, rk in enumerate(rank):
            if rk < len(boxes):
                per_word[(li, wi)] = {"status": "located", "segments": [{"quad": [[a / W, b / H] for a, b in boxes[rk]]}],
                                      "confidence": None, "reason": ""}
    return per_word, {}


def grade_page(page, per_word, seed=7, n_spans=60):
    tp = page["truth_page"]
    words, spans = rt.build_targets(tp, seed, n_spans)
    wrows, srows = [], []
    for tg in words:
        wrows.append(dict(rt.grade_target(tp, tg.words, per_word.get(tg.words[0])), tid=tg.tid))
    for tg in spans:
        srows.append(dict(rt.grade_target(tp, tg.words, span_placed([per_word.get(k) for k in tg.words])), tid=tg.tid))
    return wrows, srows


def span_placed(pls):
    if any(p is None for p in pls):
        return None
    if all(p["status"] == "located" for p in pls):
        return {"status": "located", "segments": [s for p in pls for s in p["segments"]], "reason": ""}
    if all(p["status"] in ("located", "line") for p in pls):
        # line boxes of the lines involved
        segs, seen = [], set()
        for p in pls:
            for s in (p.get("_line") or p["segments"]):
                k = json.dumps(s, sort_keys=True)
                if k not in seen:
                    seen.add(k); segs.append(s)
        return {"status": "line", "segments": segs, "reason": "span has non-located words"}
    return {"status": "unresolved", "segments": [], "reason": "span has unresolved words"}


# ---------------------------------------------------------------------------
# Overlays (scratch images only)
# ---------------------------------------------------------------------------
def draw_overlay(page, per_word, wrows, path, scale=0.6, clip=None, extra=None):
    """Truth word boxes (thin green), located boxes (blue if correct, red if
    not), line-status boxes (thin orange). ``extra``: optional list of
    (quad_px, rgb) to draw (e.g. ink runs)."""
    import fitz
    tp = page["truth_page"]
    gray = np.clip(page["gray"], 0, 255).astype(np.uint8)
    H, W = gray.shape
    doc = fitz.open()
    pg = doc.new_page(width=W, height=H)
    pg.insert_image(pg.rect, pixmap=fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(gray).tobytes(), False))
    ok = {tuple(r["target"][0]): r for r in wrows}
    for L in tp.lines:
        for w in L.words:
            q = w.quad
            pg.draw_polyline([fitz.Point(*p) for p in q] + [fitz.Point(*q[0])], color=(0, 0.7, 0), width=1)
    for key, pl in per_word.items():
        if not pl or not pl.get("segments"):
            continue
        r = ok.get(key)
        for s in pl["segments"]:
            q = [(a * W, b * H) for a, b in s["quad"]] if "quad" in s else None
            if q is None:
                x0, y0, x1, y1 = s["rect"]; q = [(x0 * W, y0 * H), (x1 * W, y0 * H), (x1 * W, y1 * H), (x0 * W, y1 * H)]
            if pl["status"] == "located":
                col = (0, 0.2, 1) if (r and r["correct"]) else (1, 0, 0)
                wd = 2
            else:
                col, wd = (1, 0.6, 0), 0.8
            pg.draw_polyline([fitz.Point(*p) for p in q] + [fitz.Point(*q[0])], color=col, width=wd)
    for q, col in (extra or []):
        pg.draw_polyline([fitz.Point(*p) for p in q] + [fitz.Point(*q[0])], color=col, width=1)
    cl = fitz.Rect(*clip) if clip else None
    pg.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=cl).save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Experiments on one page
# ---------------------------------------------------------------------------
ARMS = ("draft", "G0", "G")
EXPS = ("A", "B", "C5", "C15", "D")


def corrupt_paragraphs(page, rate, seed):
    """Unbroken truth text with page-level corruption; returns (paragraphs,
    map truth word -> token index, status per truth word)."""
    import synth_pages as S
    tp = page["truth_page"]
    keys = [(w.li, w.wi) for L in tp.lines for w in L.words]
    paras = page["paragraphs"]
    flat_para = [pi for pi, p in enumerate(paras) for _ in p]
    words = [t for p in paras for t in p]
    assert len(words) == len(keys)
    inj = S.inject_text_errors(words, rate, seed)
    out = [[] for _ in paras]
    tokmap = {}
    k = 0
    for pos, (w, s) in enumerate(zip(inj["words"], inj["src_index"])):
        if s is None:
            nx = next((x for x in inj["src_index"][pos + 1:] if x is not None), None)
            pi = flat_para[nx] if nx is not None else len(paras) - 1
        else:
            pi = flat_para[s]
        out[pi].append(w)
    # token index in the flattened corrupted stream
    idx = 0
    pos_of_out = []
    for pi, p in enumerate(out):
        pass
    order = []
    for pos, s in enumerate(inj["src_index"]):
        order.append(s)
    # the stream order of `out` equals inj order because paragraphs are contiguous
    for pos, s in enumerate(inj["src_index"]):
        if s is not None:
            tokmap[keys[s]] = pos
    status = {keys[i]: st for i, st in enumerate(inj["status"])}
    return [p for p in out if p], tokmap, status, inj


def corruption_metrics(wrows, status):
    kept = [r for r in wrows if status.get(tuple(r["target"][0])) == "kept"]
    bad = [r for r in wrows if status.get(tuple(r["target"][0])) in ("sub", "del")]
    sub = [r for r in wrows if status.get(tuple(r["target"][0])) == "sub"]
    return dict(kept=len(kept), kept_correct=sum(r["status"] == "located" and r["correct"] for r in kept),
                corrupted=len(bad), corrupted_not_located=sum(r["status"] != "located" for r in bad),
                sub=len(sub), sub_not_located=sum(r["status"] != "located" for r in sub),
                sub_located_correct_box=sum(r["status"] == "located" and r["correct"] for r in sub))


def run_page(page, exps, arms, seed=7, verbose=True, ev=None):
    rows = []
    tp = page["truth_page"]
    ev = ev or WP.page_evidence(page["gray"])
    for exp in exps:
        for arm in arms:
            if arm == "draft" and exp not in ("A", "B"):
                continue
            if arm == "Gc" and exp not in ("A", "B"):
                continue
            if arm == "draft" and exp == "B":
                continue      # the draft has its own lines: reported once, under A and B alike
            t0 = time.perf_counter()
            status = None
            info = {}
            if arm == "draft":
                pw, info = run_draft(page)
            elif exp in ("A", "B"):
                pw, info = run_G(page, exp, nottext=(arm != "G0"), width_mode="classes" if arm == "Gc" else "free", ev=ev)
            elif exp in ("C5", "C15"):
                rate = 0.05 if exp == "C5" else 0.15
                texts, src, status, _ = corrupt_lines(page["texts"], rate, seed=hash_seed(page["name"], exp))
                pw, info = run_G(page, "C", nottext=(arm != "G0"), texts=texts, src=src, ev=ev,
                                 width_mode="classes" if arm == "Gc" else "free")
            elif exp == "D":
                if page["kind"] == "synth":
                    paras, tokmap, status, _ = corrupt_paragraphs(page, 0.15, seed=hash_seed(page["name"], exp))
                    pg2 = dict(page, paragraphs=paras)
                    pw, info = run_G(pg2, "D", nottext=(arm != "G0"), src=tokmap, ev=ev,
                                     width_mode="classes" if arm == "Gc" else "free")
                else:
                    pw, info = run_G(page, "D", nottext=(arm != "G0"), ev=ev,
                                     width_mode="classes" if arm == "Gc" else "free")
            secs = time.perf_counter() - t0
            wrows, srows = grade_page(page, pw, seed=seed)
            sw_ = rt.summarize_targets([(tp, wrows)], True)
            ss_ = rt.summarize_targets([(tp, srows)], False)
            row = dict(page=page["name"], exp=exp, arm=arm, words=_slim(sw_), spans=_slim(ss_),
                       seconds=round(secs + (ev.seconds if arm != "draft" else 0.0), 2),
                       info={k: v for k, v in info.items() if k not in ("rounds",)})
            if status is not None:
                row["corruption"] = corruption_metrics(wrows, status)
            row["_wrows"] = wrows
            if arm != "draft":
                row["cand"] = candidate_table(page, pw)
            rows.append(row)
            if verbose:
                lc = sw_["located_correct"]; pr = sw_["located_precision"]
                extra = ""
                if "corruption" in row:
                    c = row["corruption"]
                    extra = f" kept {c['kept_correct']}/{c['kept']} corrupt-notloc {c['corrupted_not_located']}/{c['corrupted']}"
                if "line_assignment" in info:
                    la = info["line_assignment"]
                    extra += f" LA wrong {la['text_lines_wrong_detected']} unassigned {la['text_lines_unassigned']} nontext->text {la['nontext_detected_given_text']}/{la['nontext_detected']}"
                print(f"    {page['name'][:34]:34s} {exp:3s} {arm:5s} loc&corr {lc['k']:3d}/{lc['n']:3d} = {lc['rate']:.3f}"
                      f"  prec {pr['k']}/{pr['n']}  wl {sw_['wrong_line']['k']}  O1 {sw_['O1_cascades']['count']}"
                      f"  O2 {sw_['O2_boundary']['k']}  {secs:.1f}s{extra}", flush=True)
    return rows


def hash_seed(name, exp):
    import hashlib
    return int(hashlib.sha256(f"{name}:{exp}".encode()).hexdigest()[:8], 16)


def _slim(s):
    keep = ("n", "located_correct", "located_precision", "shown", "wrong_line", "no_box", "status_counts",
            "outcome_counts", "O1_cascades", "O2_boundary", "located_plan_literal_correct", "shown_covering_only")
    return {k: s[k] for k in keep if k in s}


def aggregate(rows, key):
    """Sum counts over rows grouped by key(row) -> {group: {exp/arm: totals}}."""
    out = {}
    for r in rows:
        g = key(r)
        d = out.setdefault(g, {}).setdefault(f"{r['exp']}/{r['arm']}", dict(pages=0, n=0, loc_correct=0, located=0,
                                                                          wrong_line=0, o1=0, o1_words=0, o2=0,
                                                                          line=0, unresolved=0, seconds=[],
                                                                          span_n=0, span_loc_correct=0, span_located=0,
                                                                          kept=0, kept_correct=0, corrupted=0,
                                                                          corrupted_not_located=0, sub=0, sub_not_located=0))
        w = r["words"]
        d["pages"] += 1
        d["n"] += w["n"]
        d["loc_correct"] += w["located_correct"]["k"]
        d["located"] += w["located_precision"]["n"]
        d["wrong_line"] += w["wrong_line"]["k"]
        d["o1"] += w["O1_cascades"]["count"]; d["o1_words"] += w["O1_cascades"]["words"]
        d["o2"] += w["O2_boundary"]["k"]
        d["line"] += w["status_counts"].get("line", 0)
        d["unresolved"] += w["status_counts"].get("unresolved", 0)
        d["seconds"].append(r["seconds"])
        s = r["spans"]
        d["span_n"] += s["n"]; d["span_loc_correct"] += s["located_correct"]["k"]; d["span_located"] += s["located_precision"]["n"]
        c = r.get("corruption")
        if c:
            for k in ("kept", "kept_correct", "corrupted", "corrupted_not_located", "sub", "sub_not_located"):
                d[k] += c[k]
    for g in out.values():
        for d in g.values():
            secs = d.pop("seconds")
            d["seconds_max"] = max(secs) if secs else None
            d["seconds_median"] = statistics.median(secs) if secs else None
    return out


def fmt_agg(agg):
    lines = []
    for g in sorted(agg):
        lines.append(f"  [{g}]")
        for ea in sorted(agg[g]):
            d = agg[g][ea]
            n = max(1, d["n"])
            s = (f"    {ea:10s} loc&corr {d['loc_correct']:5d}/{d['n']:<5d} = {d['loc_correct'] / n:6.1%}"
                 f"  prec {d['loc_correct'] if False else ''}{_prec(d)}  wrong-line {d['wrong_line']}/{d['n']}"
                 f"  O1 {d['o1']} ({d['o1_words']} w)  O2 {d['o2']}  line {d['line']}  unres {d['unresolved']}"
                 f"  spans {d['span_loc_correct']}/{d['span_n']}")
            if d["kept"]:
                s += (f"  | kept correct {d['kept_correct']}/{d['kept']} = {d['kept_correct'] / d['kept']:.1%}"
                      f"  corrupted not-located {d['corrupted_not_located']}/{d['corrupted']}"
                      f" = {d['corrupted_not_located'] / max(1, d['corrupted']):.1%} (subs {d['sub_not_located']}/{d['sub']})")
            s += f"  t<= {d['seconds_max']}s"
            lines.append(s)
    return "\n".join(lines)


def _prec(d):
    # precision of located: loc_correct / located
    return f"{d['loc_correct']}/{d['located']} = {d['loc_correct'] / max(1, d['located']):.1%}"


def eval_synth(seeds, configs, exps, arms, verbose=True):
    import synth_pages as S
    rows = []
    for label, cls, kw in configs:
        for sd in seeds:
            try:
                page = synth_page(cls, kw, sd)
            except S.FontUnavailable:
                continue
            page["config"] = label
            if verbose:
                print(f"  {label} s{sd}: {sum(map(len, page['texts']))} words, {len(page['texts'])} lines", flush=True)
            for r in run_page(page, exps, arms, verbose=verbose):
                r["config"] = label
                r["hard"] = label in HARD
                r.pop("_wrows", None)
                rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--set", choices=sorted(SEEDS), default="dev")
    s.add_argument("--exps", nargs="*", default=list(EXPS))
    s.add_argument("--arms", nargs="*", default=list(ARMS))
    s.add_argument("--configs", nargs="*", default=None)
    s.add_argument("--out", default=None)
    r = sub.add_parser("real")
    r.add_argument("--exps", nargs="*", default=list(EXPS))
    r.add_argument("--arms", nargs="*", default=list(ARMS))
    r.add_argument("--keys", nargs="*", default=None, help="slug/page subset")
    r.add_argument("--note", required=True, help="why this real-page scoring is run (logged)")
    r.add_argument("--out", default=None)
    t2 = sub.add_parser("twofold")
    t2.add_argument("results")
    o = sub.add_parser("overlay-real")
    o.add_argument("--out", required=True)
    o.add_argument("--exp", default="B")
    o.add_argument("--arm", default="G")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "synth":
        cfgs = [c for c in synth_configs() if a.configs is None or c[0] in a.configs]
        rows = eval_synth(SEEDS[a.set], cfgs, a.exps, a.arms)
        agg_cfg = aggregate(rows, lambda r: r["config"])
        agg_cls = aggregate(rows, lambda r: "hard" if r["hard"] else "clean")
        print("\nBY CONFIG\n" + fmt_agg(agg_cfg))
        print("\nCLEAN vs HARD\n" + fmt_agg(agg_cls))
        for grp in ("clean", "hard"):
            for exp in a.exps:
                for arm in a.arms:
                    rs = [r for r in rows if r["exp"] == exp and r["arm"] == arm and r["hard"] == (grp == "hard") and r.get("cand") is not None]
                    if rs:
                        print(f"  sweep {grp} {exp}/{arm}: " + "  ".join(f"t{x['tau']}:{x['rate']:.3f}/{x['precision']:.3f}" for x in sweep(rs)))
        out = Path(a.out) if a.out else OUT / f"synth_{a.set}.json"
        out.write_text(json.dumps(dict(created_at=now(), version=WP.VERSION, params=WP.PARAMS, seeds=SEEDS[a.set],
                                       rows=rows, by_config=agg_cfg, clean_vs_hard=agg_cls),
                                  ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0
    if a.cmd == "real":
        return cmd_real(a)
    if a.cmd == "overlay-real":
        return cmd_overlay_real(a)
    if a.cmd == "twofold":
        res = twofold(a.results)
        for r in res:
            print(f"  {r['exp']}/{r['arm']} tau={r['tau']} chosen on {r['tuned_on']} (train {r['train']['correct']}/{r['train']['n']}, "
                  f"prec {r['train']['precision']}) -> {r['scored']}: {r['test']['correct']}/{r['test']['n']} located&correct, "
                  f"prec {r['test']['correct']}/{r['test']['located']} = {r['test']['precision']}  "
                  f"[default tau {r['test_at_default_tau']['tau']}: {r['test_at_default_tau']['correct']}/{r['test_at_default_tau']['located']}"
                  f" = {r['test_at_default_tau']['precision']}]")
        (OUT / "twofold.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0
    return 1


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def code_fingerprint():
    import hashlib
    h = hashlib.sha256()
    for f in ("word_placer.py", "word_placer_eval.py", "line_finder.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:12]


def real_keys(sel=None):
    keys = rt.frozen_keys(ROOT)
    if sel:
        keys = [k for k in keys if f"{k[0]}/{k[1]}" in sel]
    return keys


def cmd_real(a):
    """Score real audited pages. Every call is logged (plan: keep a log of
    every real-page scoring and what changed afterwards)."""
    keys = real_keys(a.keys)
    rows = []
    pages_meta = []
    for slug, pg in keys:
        page = real_page(slug, pg)
        t = page["truth_page"]
        print(f"  {t.key} ({rt.group_of(t)}): {len(t.graded_words())} graded words", flush=True)
        ev = WP.page_evidence(page["gray"])
        for r in run_page(page, a.exps, a.arms, ev=ev):
            r["group"] = rt.group_of(t)
            r.pop("_wrows", None)
            rows.append(r)
        pages_meta.append(dict(key=t.key, group=rt.group_of(t), graded_words=len(t.graded_words()),
                               audit_sha256=t.audit_sha256))
    agg = aggregate(rows, lambda r: r["group"])
    agg_page = aggregate(rows, lambda r: r["page"])
    print("\nBY GROUP\n" + fmt_agg(agg))
    print("\nBY PAGE\n" + fmt_agg(agg_page))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) if a.out else OUT / f"real_{stamp}.json"
    out.write_text(json.dumps(dict(created_at=now(), version=WP.VERSION, code=code_fingerprint(), params=WP.PARAMS,
                                   note=a.note, pages=pages_meta, rows=rows, by_group=agg, by_page=agg_page),
                              ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    log = OUT / "run_log.txt"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] real scoring -> {out.name}  code {code_fingerprint()}  exps {a.exps} arms {a.arms}\n")
        f.write(f"  pages: {[m['key'] for m in pages_meta]}\n  note: {a.note}\n")
        for g in sorted(agg):
            for ea in sorted(agg[g]):
                d = agg[g][ea]
                f.write(f"  {g:34s} {ea:8s} loc&corr {d['loc_correct']}/{d['n']}  prec {_prec(d)}  wl {d['wrong_line']}"
                        f"  O1 {d['o1']}  O2 {d['o2']}\n")
    print(f"\nwrote {out}; appended {log}")
    return 0


def candidate_table(page, pw):
    """Every word's candidate box graded as if located, with its margin and
    whether the other located conditions (width z, fused valley) hold:
    [[margin, other_ok, correct], ...]. Lets the located threshold be swept."""
    tp = page["truth_page"]
    out = []
    for w in tp.graded_words():
        pl = pw.get((w.li, w.wi))
        if not pl or not pl.get("_quad") or pl.get("confidence") is None:
            continue
        q = pl["_quad"]
        seg = {"quad": [[a / tp.W, b / tp.H] for a, b in q]}
        g = rt.grade_target(tp, [(w.li, w.wi)], {"status": "located", "segments": [seg]})
        out.append([round(float(min(pl["confidence"], 999.0)), 3), bool(pl.get("_other_ok", True)), bool(g["correct"])])
    return out


def sweep(rows, taus=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15)):
    """Located-correct and precision as a function of the margin threshold."""
    n = sum(r["words"]["n"] for r in rows)
    cand = [c for r in rows for c in (r.get("cand") or [])]
    out = []
    for t in taus:
        loc = [c for c in cand if c[0] >= t and c[1]]
        k = sum(c[2] for c in loc)
        out.append(dict(tau=t, located=len(loc), correct=k, n=n, rate=round(k / max(1, n), 4),
                        precision=round(k / max(1, len(loc)), 4)))
    return out


def twofold(results_path, pages=("hadpi-articles/1", "hadpi-articles/2"), target=0.97):
    """Two-fold check of the one parameter tuned against located precision:
    choose the margin threshold on one page (smallest tau whose precision on
    that page reaches `target`), score the other page with it. Reads the
    candidate tables of a stored real run; scores nothing new."""
    d = json.loads(Path(results_path).read_text(encoding="utf-8"))
    taus = [x / 2 for x in range(0, 61)]
    out = []
    for exp in sorted({r["exp"] for r in d["rows"]}):
        for arm in sorted({r["arm"] for r in d["rows"] if r["arm"] != "draft"}):
            rows = {r["page"]: r for r in d["rows"] if r["exp"] == exp and r["arm"] == arm and r.get("cand") is not None}
            if not all(p in rows for p in pages):
                continue
            for tr, te in ((pages[0], pages[1]), (pages[1], pages[0])):
                sw_tr = sweep([rows[tr]], taus)
                ok = [x for x in sw_tr if x["precision"] >= target and x["located"] >= 10]
                tau = ok[0]["tau"] if ok else taus[-1]
                te_ = sweep([rows[te]], [tau])[0]
                tr_ = [x for x in sw_tr if x["tau"] == tau][0]
                dflt = sweep([rows[te]], [WP.P["LOC_MARGIN"]])[0]
                out.append(dict(exp=exp, arm=arm, tuned_on=tr, scored=te, tau=tau,
                                train=dict(correct=tr_["correct"], located=tr_["located"], n=tr_["n"], precision=tr_["precision"]),
                                test=dict(correct=te_["correct"], located=te_["located"], n=te_["n"], precision=te_["precision"]),
                                test_at_default_tau=dict(tau=WP.P["LOC_MARGIN"], correct=dflt["correct"], located=dflt["located"],
                                                         precision=dflt["precision"])))
    return out


def cmd_overlay_real(a):
    """Overlays of placed boxes on each real page (scratch only)."""
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for slug, pg in real_keys():
        page = real_page(slug, pg)
        ev = WP.page_evidence(page["gray"])
        if a.arm == "draft":
            pw, _ = run_draft(page)
        else:
            exp = "B" if a.exp == "B" else a.exp
            if exp in ("C5", "C15"):
                texts, src, status, _ = corrupt_lines(page["texts"], 0.05 if exp == "C5" else 0.15, seed=hash_seed(page["name"], exp))
                pw, _ = run_G(page, "C", nottext=(a.arm != "G0"), texts=texts, src=src, ev=ev)
            else:
                pw, _ = run_G(page, exp, nottext=(a.arm != "G0"), ev=ev)
        wrows, _ = grade_page(page, pw)
        H, W = page["gray"].shape
        name = f"{slug}_{pg}_{a.exp}_{a.arm}"
        draw_overlay(page, pw, wrows, outdir / f"{name}.png", scale=0.5)
        # three horizontal bands at full resolution for detail
        for k, (y0, y1) in enumerate(((0.0, 0.36), (0.32, 0.68), (0.64, 1.0))):
            draw_overlay(page, pw, wrows, outdir / f"{name}_band{k}.png", scale=0.8, clip=(0, y0 * H, W, y1 * H))
        bad = [r for r in wrows if r["status"] == "located" and not r["correct"]]
        print(f"  {name}: located wrong {len(bad)}: " + ", ".join(f"{r['tid']}:{r['outcome']}" for r in bad[:60]))
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] overlays of real pages viewed (exp {a.exp}, arm {a.arm}), code {code_fingerprint()} -> {outdir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
