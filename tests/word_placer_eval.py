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
import os
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
    info["_r"] = r                     # raw placement (not serialised)
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


def within_one_letter(tp, key, placed, row) -> bool:
    """DIAGNOSTIC (not the headline): a located box whose best-covered word
    is the target (not shifted, not on another line) and whose two
    horizontal edges are each within 1.0 letter of the truth word's edges,
    measured in the truth line's frame. The strict rule's tolerance is half a
    letter; this is the looser reading the owner may decide the UI needs."""
    if not placed or placed.get("status") != "located" or not placed.get("segments"):
        return False
    if row.get("outcome") in ("shifted", "wrong_line", "off_text", "no_box"):
        return False
    w = tp.word(*key)
    L = tp.lines[key[0]]
    pts = [p for sgm in placed["segments"] for p in rt.region_poly(sgm, tp.W, tp.H)]
    fb = ba.quad_to_frame_box(L.frame, pts)
    letter = (w.fbox[2] - w.fbox[0]) / max(1.0, w.letters)
    return abs(fb[0] - w.fbox[0]) <= letter and abs(fb[2] - w.fbox[2]) <= letter


def within_half_letter(tp, key, placed, row) -> bool:
    """DIAGNOSTIC (arm T continuation; not the headline): as within_one_letter
    with the strict rule's own horizontal tolerance (half a letter) and no
    vertical test. Separates horizontal placement from the vertical extent
    of the truth box (on hadpi many audited boxes reach well above the
    word's ink, so an ink-tight box with both edges right can miss the 90%
    area cover)."""
    if not placed or placed.get("status") != "located" or not placed.get("segments"):
        return False
    if row.get("outcome") in ("shifted", "wrong_line", "off_text", "no_box"):
        return False
    w = tp.word(*key)
    L = tp.lines[key[0]]
    pts = [p for sgm in placed["segments"] for p in rt.region_poly(sgm, tp.W, tp.H)]
    fb = ba.quad_to_frame_box(L.frame, pts)
    letter = (w.fbox[2] - w.fbox[0]) / max(1.0, w.letters)
    return abs(fb[0] - w.fbox[0]) <= 0.5 * letter and abs(fb[2] - w.fbox[2]) <= 0.5 * letter


def grade_page(page, per_word, seed=7, n_spans=60):
    tp = page["truth_page"]
    words, spans = rt.build_targets(tp, seed, n_spans)
    wrows, srows = [], []
    for tg in words:
        pl = per_word.get(tg.words[0])
        row = dict(rt.grade_target(tp, tg.words, pl), tid=tg.tid)
        row["within1"] = bool(within_one_letter(tp, tg.words[0], pl, row))
        row["withinhalf"] = bool(within_half_letter(tp, tg.words[0], pl, row))
        # instrument change (D12): the horizontal rule word_correct_h1, beside the strict v1
        row["correct_h1"] = bool(row["status"] == "located" and rt.grade_target_h1(tp, tg.words, pl, row)["correct"])
        wrows.append(row)
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
                kept_correct_h1=sum(bool(r.get("correct_h1")) for r in kept),
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
            if arm == "R" and exp not in ("A", "B", "C5"):
                continue
            if arm == "T" and exp not in ("A", "B", "C5", "C15"):
                continue
            if arm == "draft" and exp == "B":
                continue      # the draft has its own lines: reported once, under A and B alike
            t0 = time.perf_counter()
            status = None
            info = {}
            if arm == "T":
                texts = src = None
                if exp in ("C5", "C15"):
                    texts, src, status, _ = corrupt_lines(page["texts"], 0.05 if exp == "C5" else 0.15,
                                                          seed=hash_seed(page["name"], exp))
                arms_t, tinfo = run_T(page, exp, ev=ev, texts=texts, src=src, init=T_OPTS["init"],
                                      with_R=T_OPTS["with_R"])
                secs = time.perf_counter() - t0
                slim = {k: v for k, v in tinfo.items() if not k.startswith("_")}
                g2b = slim.pop("g2_boxes_on_nontext", None)
                for nm, pw_ in arms_t.items():
                    row = _score_arm(page, exp, nm, pw_, slim, status, secs, verbose)
                    if g2b is not None:
                        row["g2"] = dict(text_lines_on_nontext=slim.get("g2_text_lines_on_nontext"),
                                         boxes_on_nontext=g2b.get(nm))
                    row["cand"] = candidate_table(page, pw_) if nm == "T" else None
                    rows.append(row)
                if T_OPTS.get("keep"):
                    T_OPTS["keep"][(page["name"], exp)] = (arms_t, tinfo)
                continue
            if arm == "R":
                texts = src = None
                if exp == "C5":
                    texts, src, status, _ = corrupt_lines(page["texts"], 0.05, seed=hash_seed(page["name"], exp))
                arms_r, rinfo = run_R(page, exp, ev=ev, texts=texts, src=src)
                for nm, pw_ in arms_r.items():
                    rows.append(_score_arm(page, exp, nm, pw_, rinfo, status, time.perf_counter() - t0, verbose))
                continue
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
                       within1=sum(r_["within1"] for r_ in wrows), withinhalf=sum(r_["withinhalf"] for r_ in wrows),
               h1_correct=sum(r_["correct_h1"] for r_ in wrows),
                       seconds=round(secs + (ev.seconds if arm != "draft" else 0.0), 2),
                       info={k: v for k, v in info.items() if k not in ("rounds", "_r")})
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
                    extra = f" kept (side-edge) {c['kept_correct_h1']}/{c['kept']} corrupt-notloc {c['corrupted_not_located']}/{c['corrupted']}"
                if "line_assignment" in info:
                    la = info["line_assignment"]
                    extra += f" LA wrong {la['text_lines_wrong_detected']} unassigned {la['text_lines_unassigned']} nontext->text {la['nontext_detected_given_text']}/{la['nontext_detected']}"
                print(f"    {page['name'][:34]:34s} {exp:3s} {arm:5s} side-edge {row['h1_correct']:3d}/{lc['n']:3d} = {row['h1_correct'] / max(1, lc['n']):.3f}"
                      f"  prec {row['h1_correct']}/{pr['n']}  {secs:.1f}s{extra}", flush=True)
    return rows


def _score_arm(page, exp, name, pw, info, status, secs, verbose):
    tp = page["truth_page"]
    wrows, srows = grade_page(page, pw)
    sw_ = rt.summarize_targets([(tp, wrows)], True)
    ss_ = rt.summarize_targets([(tp, srows)], False)
    row = dict(page=page["name"], exp=exp, arm=name, words=_slim(sw_), spans=_slim(ss_),
               within1=sum(r_["within1"] for r_ in wrows), withinhalf=sum(r_["withinhalf"] for r_ in wrows),
               h1_correct=sum(r_["correct_h1"] for r_ in wrows),
               seconds=round(secs, 2), info=info,
               outcomes_located_wrong=rt._count(x["outcome"] for x in wrows if x["status"] == "located" and not x["correct"]))
    if status is not None:
        row["corruption"] = corruption_metrics(wrows, status)
    if verbose:
        lc = sw_["located_correct"]; pr = sw_["located_precision"]
        extra = ""
        if "corruption" in row:
            c = row["corruption"]
            extra = f" kept (side-edge) {c['kept_correct_h1']}/{c['kept']} corrupt-notloc {c['corrupted_not_located']}/{c['corrupted']}"
        print(f"    {page['name'][:34]:34s} {exp:3s} {name:5s} side-edge {row['h1_correct']:3d}/{lc['n']:3d} = {row['h1_correct'] / max(1, lc['n']):.3f}"
              f"  prec {row['h1_correct']}/{pr['n']}  within1 {row['within1']}  {secs:.1f}s {info.get('font')}{extra}", flush=True)
    return row


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
                                                                          kept=0, kept_correct=0, corrupted=0, within1=0,
                                                                          corrupted_not_located=0, sub=0, sub_not_located=0,
                                                                          withinhalf=0, h1=0, kept_correct_h1=0))
        w = r["words"]
        d["pages"] += 1
        d["within1"] += r.get("within1", 0)
        d["withinhalf"] += r.get("withinhalf", 0)
        d["h1"] += r.get("h1_correct", 0)
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
            for k in ("kept", "kept_correct", "corrupted", "corrupted_not_located", "sub", "sub_not_located", "kept_correct_h1"):
                d[k] += c.get(k, 0)
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
            # D12: the side-edge rule (word_correct_h1) only; the strict area rule stays in the
            # stored rows (reproducibility) but is not printed
            s = (f"    {ea:10s} side-edge correct {d['h1']:5d}/{d['n']:<5d} = {d['h1'] / n:6.1%}"
                 f"  precision {d['h1']}/{d['located']} = {d['h1'] / max(1, d['located']):.1%}"
                 f"  line {d['line']}  unres {d['unresolved']}"
                 f"  [diag: located&within-1-letter {d['within1']}/{d['n']} = {d['within1'] / n:.1%}]")
            if d["kept"]:
                s += (f"  | kept correct (side-edge) {d['kept_correct_h1']}/{d['kept']} = {d['kept_correct_h1'] / d['kept']:.1%}"
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
    s.add_argument("--tinit", choices=("G", "R", "flat"), default="G", help="arm T: initial word positions")
    s.add_argument("--no-tr", action="store_true", help="arm T: skip arm R (no TR, Rc, RcG)")
    s.add_argument("--vert", choices=("old", "line", "comp"), default="old", help="vertical extent rule for the arm T family")
    s.add_argument("--la", choices=("length", "T"), default="length", help="line-text assignment under B/C")
    s.add_argument("--seeds", nargs="*", type=int, default=None, help="override the seed set")
    r = sub.add_parser("real")
    r.add_argument("--exps", nargs="*", default=list(EXPS))
    r.add_argument("--arms", nargs="*", default=list(ARMS))
    r.add_argument("--keys", nargs="*", default=None, help="slug/page subset")
    r.add_argument("--note", required=True, help="why this real-page scoring is run (logged)")
    r.add_argument("--out", default=None)
    r.add_argument("--tinit", choices=("G", "R", "flat"), default="G", help="arm T: initial word positions")
    r.add_argument("--no-tr", action="store_true", help="arm T: skip arm R (no TR, Rc, RcG)")
    r.add_argument("--vert", choices=("old", "line", "comp"), default="old", help="vertical extent rule for the arm T family")
    r.add_argument("--la", choices=("length", "T"), default="length", help="line-text assignment under B/C")
    v = sub.add_parser("vrun")
    v.add_argument("--mode", choices=("mark", "coord"), default="mark")
    v.add_argument("--model", choices=("sonnet", "opus"), default="sonnet")
    v.add_argument("--exps", nargs="*", default=["A"])
    v.add_argument("--keys", nargs="*", default=None)
    v.add_argument("--batch", type=int, default=None)
    v.add_argument("--rep", type=int, default=0)
    v.add_argument("--plan", action="store_true", help="estimate cost only; no API call")
    v.add_argument("--max-lines", type=int, default=None, help="only the first N lines (smoke)")
    v.add_argument("--note", default="")
    t2 = sub.add_parser("twofold")
    t2.add_argument("results")
    tx = sub.add_parser("tx", help="arm T: template transfer between two pages, and pooling")
    tx.add_argument("--keys", nargs="*", default=["hadpi-articles/1", "hadpi-articles/2"])
    tx.add_argument("--synth", default=None, help="class/preset: use two synthetic pages instead")
    tx.add_argument("--seeds", nargs="*", type=int, default=[201, 202])
    tx.add_argument("--sheets", default=None, help="directory for template contact sheets (scratch)")
    tx.add_argument("--note", default="")
    o = sub.add_parser("overlay-real")
    o.add_argument("--out", required=True)
    o.add_argument("--exp", default="B")
    o.add_argument("--arm", default="G")
    o.add_argument("--keys", nargs="*", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if getattr(a, "tinit", None):
        T_OPTS.update(init=a.tinit, with_R=not a.no_tr, vert=a.vert, la=a.la)
    if a.cmd == "synth":
        cfgs = [c for c in synth_configs() if a.configs is None or c[0] in a.configs]
        rows = eval_synth(a.seeds or SEEDS[a.set], cfgs, a.exps, a.arms)
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
        out.write_text(json.dumps(dict(created_at=now(), version=WP.VERSION, params=WP.PARAMS, seeds=a.seeds or SEEDS[a.set],
                                       definitions={k: rt.DEFINITIONS[k] for k in ("word_correct_v1", "word_correct_h1")},
                                       instrument_note=INSTRUMENT_NOTE, t_opts={k: v for k, v in T_OPTS.items() if k != "keep"},
                                       rows=rows, by_config=agg_cfg, clean_vs_hard=agg_cls),
                                  ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0
    if a.cmd == "real":
        return cmd_real(a)
    if a.cmd == "overlay-real":
        return cmd_overlay_real(a)
    if a.cmd == "tx":
        return cmd_tx(a)
    if a.cmd == "vrun":
        return cmd_vrun(a)
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


def _tp_digest():
    import word_placer_t as T
    return ",".join(f"{k}={v}" for k, v in T.TP.items())


def code_fingerprint():
    import hashlib
    h = hashlib.sha256()
    for f in ("word_placer.py", "word_placer_eval.py", "line_finder.py", "word_placer_t.py"):
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
                                   definitions={k: rt.DEFINITIONS[k] for k in ("word_correct_v1", "word_correct_h1")},
                                   instrument_note=INSTRUMENT_NOTE, t_opts={k: v for k, v in T_OPTS.items() if k != "keep"},
                                   note=a.note, pages=pages_meta, rows=rows, by_group=agg, by_page=agg_page),
                              ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    log = OUT / "run_log.txt"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] real scoring -> {out.name}  code {code_fingerprint()}  exps {a.exps} arms {a.arms}"
                + (f"  T init {T_OPTS['init']} vert {T_OPTS['vert']} la {T_OPTS['la']} TP {_tp_digest()}" if "T" in a.arms else "") + "\n")
        f.write(f"  pages: {[m['key'] for m in pages_meta]}\n  note: {a.note}\n")
        for g in sorted(agg):
            for ea in sorted(agg[g]):
                d = agg[g][ea]
                f.write(f"  {g:34s} {ea:8s} side-edge {d['h1']}/{d['n']}  prec {d['h1']}/{d['located']}  unres {d['unresolved']}\n")
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


# ---------------------------------------------------------------------------
# Arm V (line crops; paid) and the hybrid
# ---------------------------------------------------------------------------
def v_lines(page, exp, mode, *, ev, texts=None, src=None):
    """Run G (the source of candidates, the routing signal and the line
    geometry), then build one crop per trusted text line."""
    import word_placer_v as V
    gexp = "A" if exp == "A" else ("C" if exp.startswith("C") else "B")
    pwG, info = run_G(page, gexp, texts=texts, src=src, ev=ev)
    r = info["_r"]
    texts_used = texts if texts is not None else page["texts"]
    lines, meta = [], {}
    for i, toks in enumerate(texts_used):
        if not toks:
            continue
        if gexp == "A":
            j, trusted = i, True
        else:
            j = r["assign"][i]
            trusted = j is not None and r["la"]["margin"][i] >= WP.P["LA_MARGIN"]
        if j is None or not trusted:
            continue
        li = r["inks"][j]
        if not li.M:
            continue
        al = r["diag"]["aligns"][i]
        crop = V.line_strip(page["gray"], li, ev.hc)
        if mode == "mark":
            crop.marks = V.candidate_marks(li, al, crop)
        png = V.render_png(crop, mode)
        lines.append((i + 1, list(toks), crop, png))
        meta[i] = (li, al, crop)
    return pwG, info, r, texts_used, lines, meta


def run_V(page, exp, mode, model, *, ev, run=None, rep=0, batch=None, dry=False, texts=None, src=None,
          max_lines=None):
    """Returns (arms {name: per_word}, cost info). dry=True: plan only."""
    import word_placer_v as V
    pwG, info, r, texts_used, lines, meta = v_lines(page, exp, mode, ev=ev, texts=texts, src=src)
    if max_lines:
        keep = {ln for ln, *_ in lines[:max_lines]}
        lines = [x for x in lines if x[0] in keep]
        meta = {i: m for i, m in meta.items() if i + 1 in keep}
    tag = "s" if model == V.MODEL_SONNET else "o"
    arm = f"V{mode}-{tag}" + (f"-b{batch}" if batch else "") + (f"-r{rep}" if rep else "")
    reqs = V.build_requests(f"{page['name']}:{exp}", lines, mode, model, arm=arm, rep=rep, batch=batch)
    cost = dict(requests=len(reqs), lines=len(lines), expected=sum(q.expected() for _, q in reqs),
                worst=sum(q.worst_case() for _, q in reqs), input_est=sum(q.input_est() for _, q in reqs))
    if dry:
        return None, cost
    answers, line_cost = {}, {}
    actual = 0.0
    for lns, req in reqs:
        rec = run.call(req)
        actual += rec["cost"]
        ans = V.parse_response(rec)
        toks_img = {im["label"]: im["api_tokens_est"] for im in rec.get("images", [])}
        tot = sum(toks_img.values()) or 1
        for ln in lns:
            answers[ln] = ans.get(ln)
            share = toks_img.get(f"{page['name']}:{exp}:l{ln}", tot / len(lns)) / tot
            line_cost[ln] = rec["cost"] * share
    W, H = page["truth_page"].W, page["truth_page"].H
    unit = r["model"].unit
    arms = {f"{arm}": {}, f"VG{mode}-{tag}" + (f"-r{rep}" if rep else ""): {},
            f"H{mode}-{tag}" + (f"-r{rep}" if rep else ""): {}}
    names = list(arms)
    routed_cost = 0.0
    n_routed = 0
    for i, toks in enumerate(texts_used):
        g_line = r["results"][i]
        if i in meta:
            li, al, crop = meta[i]
            lq = next((x.get("line_quad") for x in g_line if x.get("line_quad")), None)
            ans = answers.get(i + 1) or []
            iv = V.answer_intervals(ans, crop, mode, len(toks))
            v = V.v_results(li, toks, iv, ev.sw, unit, ev.hc, lq)
            vg = V.vg_results(li, v, g_line, unit)
            routed = V.route_line(g_line)
            h = V.hybrid_results(g_line, v if routed else None)
            if routed:
                routed_cost += line_cost.get(i + 1, 0.0)
                n_routed += 1
        else:
            v = vg = h = [dict(x) for x in g_line]
        for name, res in zip(names, (v, vg, h)):
            for ti, x in enumerate(res):
                key = src[i][ti] if src is not None else (i, ti)
                if key is None:
                    continue
                arms[name][key] = to_placed(x, W, H)
    cost.update(actual=round(actual, 5), hybrid_routed_lines=n_routed, hybrid_cost=round(routed_cost, 5))
    return arms, cost


def cmd_vrun(a):
    """Arm V on real (or synthetic) pages: plan (no calls) or run."""
    import word_placer_v as V
    model = V.MODEL_OPUS if a.model == "opus" else V.MODEL_SONNET
    keys = real_keys(a.keys)
    run = None if a.plan else V.runner(cap=V.CAP_USD)
    rows, costs = [], {}
    for slug, pg in keys:
        page = real_page(slug, pg)
        ev = WP.page_evidence(page["gray"])
        for exp in a.exps:
            texts = src = status = None
            if exp == "C5":
                texts, src, status, _ = corrupt_lines(page["texts"], 0.05, seed=hash_seed(page["name"], exp))
            arms, cost = run_V(page, exp, a.mode, model, ev=ev, run=run, rep=a.rep, batch=a.batch,
                               dry=a.plan, texts=texts, src=src, max_lines=a.max_lines)
            costs[f"{page['name']}:{exp}"] = cost
            print(f"  {page['name']} {exp} {a.mode} {model}: {cost}", flush=True)
            if a.plan:
                continue
            for name, pw in arms.items():
                wrows, srows = grade_page(page, pw)
                sw_ = rt.summarize_targets([(page["truth_page"], wrows)], True)
                ss_ = rt.summarize_targets([(page["truth_page"], srows)], False)
                row = dict(page=page["name"], exp=exp, arm=name, words=_slim(sw_), spans=_slim(ss_),
                           within1=sum(x["within1"] for x in wrows), seconds=0.0, info={},
                           group=rt.group_of(page["truth_page"]), cost=cost,
                           outcomes_located_wrong=rt._count(x["outcome"] for x in wrows if x["status"] == "located" and not x["correct"]),
                           reasons_not_located=rt._count((pw.get(tuple(x["target"][0])) or {}).get("reason", "").split(":")[0]
                                                      for x in wrows if x["status"] != "located"))
                if status is not None:
                    row["corruption"] = corruption_metrics(wrows, status)
                rows.append(row)
                lc, pr = sw_["located_correct"], sw_["located_precision"]
                print(f"    {name:14s} loc&corr {lc['k']}/{lc['n']} = {lc['rate']:.3f}  prec {pr['k']}/{pr['n']}"
                      f"  within1 {row['within1']}  wl {sw_['wrong_line']['k']}  O1 {sw_['O1_cascades']['count']}"
                      f"  O2 {sw_['O2_boundary']['k']}" + (f"  kept {row['corruption']['kept_correct']}/{row['corruption']['kept']}"
                                                           f" corrupt-notloc {row['corruption']['corrupted_not_located']}/{row['corruption']['corrupted']}"
                                                           if "corruption" in row else ""), flush=True)
    tot = sum(c.get("expected", 0) for c in costs.values())
    print(f"\n{'PLAN' if a.plan else 'RUN'} {a.mode} {model}: expected ${tot:.4f}, worst ${sum(c['worst'] for c in costs.values()):.4f}"
          + ("" if a.plan else f", actual ${sum(c.get('actual', 0) for c in costs.values()):.4f}"))
    if not a.plan:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = OUT / f"v_{a.mode}_{a.model}_{'-'.join(a.exps)}_b{a.batch or 0}_r{a.rep}_{stamp}.json"
        out.write_text(json.dumps(dict(created_at=now(), code=code_fingerprint(), mode=a.mode, model=model,
                                       batch=a.batch, rep=a.rep, rows=rows, costs=costs,
                                       ledger_total=V.runner().ledger.total()),
                                  ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
        with (OUT / "run_log.txt").open("a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] arm V real scoring -> {out.name} code {code_fingerprint()} mode {a.mode} model {model}"
                    f" exps {a.exps} batch {a.batch} rep {a.rep}; note: {a.note}\n")
        print(f"wrote {out}")
    return 0


def run_R(page, exp, *, ev, texts=None, src=None):
    """Arm R (render + DTW) and R-and-G agreement. $0."""
    import word_placer_r as R
    import word_placer_v as V
    gexp = "A" if exp == "A" else ("C" if exp.startswith("C") else "B")
    pwG, info = run_G(page, gexp, texts=texts, src=src, ev=ev)
    r = info["_r"]
    texts_used = texts if texts is not None else page["texts"]
    todo = []
    for i, toks in enumerate(texts_used):
        if not toks:
            continue
        if gexp == "A":
            j, trusted = i, True
        else:
            j = r["assign"][i]
            trusted = j is not None and r["la"]["margin"][i] >= WP.P["LA_MARGIN"]
        if j is None or not trusted or not r["inks"][j].M:
            continue
        todo.append((i, r["inks"][j], list(toks)))
    own = (page.get("spec") or {}).get("font")
    font, fcosts = R.choose_font([(li, t) for _, li, t in todo], ev.sw, exclude=own)
    W, H = page["truth_page"].W, page["truth_page"].H
    unit = r["model"].unit
    arms = {"R": {}, "RG": {}}
    done = {}
    for i, li, toks in todo:
        iv, _c = R.align_line_r(li, toks, font, ev.sw) if font else (None, None)
        g_line = r["results"][i]
        lq = next((x.get("line_quad") for x in g_line if x.get("line_quad")), None)
        ans = [dict(interval=x, present=True, why="" if x else "no rendered span") for x in (iv or [None] * len(toks))]
        rv = V.v_results(li, toks, ans, ev.sw, unit, ev.hc, lq)
        rv = [dict(x, reason=x["reason"].replace("V", "R", 1)) for x in rv]
        done[i] = (rv, V.vg_results(li, rv, g_line, unit))
    for i, toks in enumerate(texts_used):
        g_line = r["results"][i]
        rv, rg = done.get(i, ([dict(x) for x in g_line], [dict(x) for x in g_line]))
        for name, res in (("R", rv), ("RG", rg)):
            for ti, x in enumerate(res):
                key = src[i][ti] if src is not None else (i, ti)
                if key is not None:
                    arms[name][key] = to_placed(x, W, H)
    return arms, dict(font=os.path.basename(font) if font else None, font_costs=fcosts)


INSTRUMENT_NOTE = ("h1 (word_correct_h1, owner decision D12) is the headline on every page; the strict area rule "
                   "word_correct_v1 is printed beside it. The difference between the two columns is an instrument "
                   "change, not a placement improvement.")
T_ARMS = ("T", "Tall", "TG", "TR", "Rc", "RcG", "R", "RG", "S", "SR", "SG", "ST", "M2", "M3", "TA", "TQ", "TAQ")
R_DEP = ("TR", "Rc", "RcG", "R", "RG", "SR", "M2", "M3")
T_OPTS = dict(init="G", with_R=True, keep=None, vert="old", la="length")
# set from the command line (--tinit, --no-tr, --vert, --la). vert "old": each
# arm's own vertical extent (the runs of record); "line": one shared rule
# (word_placer_s.vertical_box). la "length": §4.2 length assignment; "T":
# template-scored assignment (word_placer_la).


def _todo_lines(r, texts_used, gexp):
    todo = []
    for i, toks in enumerate(texts_used):
        if not toks:
            continue
        if gexp == "A":
            j, trusted = i, True
        else:
            j = r["assign"][i]
            trusted = j is not None and r["la"]["margin"][i] >= WP.P["LA_MARGIN"]
        if j is None or not trusted or not r["inks"][j].M:
            continue
        todo.append((i, r["inks"][j], list(toks)))
    return todo


def run_T(page, exp, *, ev, texts=None, src=None, init="G", with_R=True, model=None, learn=True, iters=None):
    """Arm T (page-trained template aligner) and its agreement variants, plus
    Rc (arm R rendered from glyphs cut out of the page). $0.

    init: initial word positions for the templates: "G" (arm G's best
    alignment), "R" (arm R's warp) or "flat" (line extent split by G widths).
    model: a trained TModel to decode with (template transfer); learn=False
    decodes with it unchanged."""
    import copy
    import word_placer_r as Rr
    import word_placer_t as T
    import word_placer_v as V
    gexp = "A" if exp == "A" else ("C" if exp.startswith("C") else "B")
    pwG, ginfo = run_G(page, gexp, texts=texts, src=src, ev=ev)
    r = ginfo["_r"]
    texts_used = texts if texts is not None else page["texts"]
    info = dict(init=init)
    if gexp != "A" and T_OPTS.get("la") == "T":
        r = template_reassign(page, ev, r, texts_used, src, ginfo, info)
    todo = _todo_lines(r, texts_used, gexp)
    W, H = page["truth_page"].W, page["truth_page"].H
    gm = r["model"]
    unit = gm.unit
    # arm R (font render) for TR, Rc and the R init
    rv_by, rg_by, riv_by = {}, {}, {}
    if with_R or init == "R":
        t0 = time.perf_counter()
        own = (page.get("spec") or {}).get("font")
        font, _fc = Rr.choose_font([(li, t) for _, li, t in todo], ev.sw, exclude=own)
        for i, li, toks in todo:
            iv, _c = Rr.align_line_r(li, toks, font, ev.sw) if font else (None, None)
            g_line = r["results"][i]
            lq = next((x.get("line_quad") for x in g_line if x.get("line_quad")), None)
            ans = [dict(interval=x, present=True, why="" if x else "no rendered span") for x in (iv or [None] * len(toks))]
            rv = V.v_results(li, toks, ans, ev.sw, unit, ev.hc, lq)
            rv_by[i] = [dict(x, reason=x["reason"].replace("V", "R", 1)) for x in rv]
            rg_by[i] = V.vg_results(li, rv_by[i], g_line, unit)
            riv_by[i] = iv
        info["font"] = os.path.basename(font) if font else None
        info["seconds_R"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    cs = None
    if model is not None:                    # transfer: same columns per letter as the model's page
        cs = model.cs * unit / model.n_hint_unit
    lines = [T.make_line(li, toks, ev.hc, ev.sw, cs=cs) for _, li, toks in todo]
    ivs = []
    for (i, li, toks), L in zip(todo, lines):
        if init == "R" and riv_by.get(i):
            ivs.append([riv_by[i][ti] for ti in L.order])
        elif init == "flat":
            ivs.append(None)
        else:
            al = r["diag"]["aligns"][i]
            ivs.append([None if d["deleted"] else (d["s0"], d["s1"]) for d in al.words])
    if init == "flat":
        ivs = flat_intervals(lines, gm, r)
    gw = lambda k: T.g_width(gm, k)  # noqa: E731
    if model is None:
        m = T.initial_model(lines, ivs, gw, unit)
    else:
        m = copy.deepcopy(model)
        for L in lines:                       # unseen units: G width hints
            for k in L.ukeys:
                m.n_hint.setdefault(k, int(np.clip(round(gw(k) / m.cs), T.TP["N_MIN"], 99)))
    aligns, hist = T.train(lines, m, unit, iters=iters, learn=learn)
    info["seconds_T"] = round(time.perf_counter() - t0, 2)
    info["history"] = hist
    info["units"] = len(m.tmpl)
    info["unit_occ"] = sorted(m.occ.values(), reverse=True)
    info["_model"] = m
    info["_lines"] = lines
    info["_aligns"] = aligns
    info["_todo_idx"] = [i for i, _, _ in todo]
    # Rc: templates from words where R and G agree, one shot, warped like R
    rc_by = {}
    if with_R:
        rc_iv = []
        for (i, li, toks), L in zip(todo, lines):
            v = [None] * L.nwords
            for w, ti in enumerate(L.order):
                if rg_by[i][ti]["status"] == "located" and riv_by[i] and riv_by[i][ti] is not None:
                    v[w] = riv_by[i][ti]
            rc_iv.append(v)
        mrc = T.initial_model(lines, rc_iv, gw, unit)
        gsnap = gm.snapshot()["gap"]
        gap_cols = {T.JOIN: 0.0, T.PIECE: max(0.0, gsnap[0][0]) / lines[0].cs if lines else 0.0,
                    T.WORD: max(0.0, gsnap[1][0]) / lines[0].cs if lines else 0.0, T.EDGE: 0.0}
        for (i, li, toks), L in zip(todo, lines):
            iv = T.rc_align(L, mrc, gap_cols, ev.sw)
            g_line = r["results"][i]
            lq = next((x.get("line_quad") for x in g_line if x.get("line_quad")), None)
            ans = [dict(interval=x, present=True, why="" if x else "no span") for x in (iv or [None] * len(toks))]
            rc = V.v_results(li, toks, ans, ev.sw, unit, ev.hc, lq)
            rc = [dict(x, reason=x["reason"].replace("V", "Rc", 1)) for x in rc]
            rc_by[i] = (rc, V.vg_results(li, rc, g_line, unit))
    # arm S: the audit draft's word splitter inside the same lines, same text
    import word_placer_s as Ss
    t0 = time.perf_counter()
    squads = Ss.s_page(page["gray"], [(i, li.geom, toks) for i, li, toks in todo], ev.S)
    info["seconds_S"] = round(time.perf_counter() - t0, 2)
    arms = {k: {} for k in T_ARMS + (("Gv",) if T_OPTS.get("vert") in ("line", "comp") else ())}
    done = {}
    for (i, li, toks), L, al in zip(todo, lines, aligns):
        tres, _ = T.word_results(L, al, ev.sw, unit, ev.hc)
        g_line = r["results"][i]
        lq = next((x.get("line_quad") for x in g_line if x.get("line_quad")), None)
        tall = T.every_box(tres)
        sres = Ss.s_results(squads[i], lq)
        d = {"T": tres, "Tall": tall, "TG": T.agree(li, tall, g_line, unit), "S": sres}
        per = {"T": tall, "G": g_line, "S": sres}
        if with_R:
            d["TR"] = T.agree(li, tall, rv_by[i], unit)
            d["Rc"], d["RcG"] = rc_by[i]
            d["R"], d["RG"] = rv_by[i], rg_by[i]
            per["R"] = rv_by[i]
        for name, rule in (("SR", ("S", "R")), ("SG", ("S", "G")), ("ST", ("S", "T")), ("M2", ("M", 2)), ("M3", ("M", 3))):
            if (rule[0] == "M" and with_R) or (rule[0] != "M" and all(a in per for a in rule)):
                d[name] = Ss.agreement_line(li, {k: per[k] for k in (rule if rule[0] != "M" else per)}, unit, rule)
        # follow-up 2a: rules between T and S&T, defined before scoring (word_placer_s)
        guard = Ss.t_guard_line(L, al, lambda k: T.g_width(gm, k), unit)
        others = {k: per[k] for k in ("S", "G", "R") if k in per}
        d["TA"] = Ss.t_any_agree(li, tall, others, unit)
        d["TQ"] = Ss.apply_guard(tres, guard)
        d["TAQ"] = Ss.apply_guard(d["TA"], guard)
        done[i] = d
    if T_OPTS.get("vert") in ("line", "comp"):
        for (i, li, toks) in todo:
            band = Ss.line_band(li)
            if band is None:
                continue
            done[i]["Gv"] = [dict(x) for x in r["results"][i]]
            for name in list(done[i]):
                if T_OPTS["vert"] == "line":
                    done[i][name] = [dict(x, quad=Ss.vertical_box(li, x.get("quad"), band)) if x.get("quad") else x
                                     for x in done[i][name]]
                else:
                    done[i][name] = Ss.vertical_line_comp(li, done[i][name], ev.sw, unit, ev.hc)
    for i, toks in enumerate(texts_used):
        g_line = r["results"][i]
        for name in T_ARMS + (("Gv",) if T_OPTS.get("vert") in ("line", "comp") else ()):
            res = done.get(i, {}).get(name)
            if res is None:
                if name in R_DEP and not with_R:
                    continue
                res = [dict(x) if x["status"] == "unresolved" else dict(x, status="line") for x in g_line]
            for ti, x in enumerate(res):
                key = src[i][ti] if src is not None else (i, ti)
                if key is not None:
                    arms[name][key] = to_placed(x, W, H)
    if not with_R:
        for k in R_DEP:
            arms.pop(k, None)
    # G2 (D6): text lines on non-text detected lines, boxes shown there
    la = ginfo.get("line_assignment")
    if la is not None:
        under = la["under"]
        bad_lines = {i for i, j in enumerate(r["assign"]) if j is not None and under[j] is None}
        info["g2_text_lines_on_nontext"] = len(bad_lines)
        g2 = {}
        for name, pw in arms.items():
            n = 0
            for i in bad_lines:
                for ti in range(len(texts_used[i])):
                    key = src[i][ti] if src is not None else (i, ti)
                    x = pw.get(key) if key is not None else None
                    n += bool(x and x["status"] in ("located", "line") and x["segments"])
            g2[name] = n
        info["g2_boxes_on_nontext"] = g2
    return arms, info


def template_reassign(page, ev, r, texts_used, src, ginfo, info):
    """Follow-up 2: replace the length assignment by the template-scored one
    (word_placer_la); arm G is re-run with it, and the G2 report redone."""
    import word_placer_la as LA
    import word_placer_t as T
    t0 = time.perf_counter()
    gm0 = r["model"]
    todo0 = _todo_lines(r, texts_used, "B")
    lines0 = [T.make_line(li, toks, ev.hc, ev.sw) for _, li, toks in todo0]
    ivs0 = [[None if d["deleted"] else (d["s0"], d["s1"]) for d in r["diag"]["aligns"][i].words] for i, _, _ in todo0]
    m0 = T.initial_model(lines0, ivs0, lambda k: T.g_width(gm0, k), gm0.unit)
    T.train(lines0, m0, gm0.unit, iters=LA.LP["TRAIN_ITERS"])
    dets = WP.l1_lines(ev)
    kinds = [k for _, k in dets]
    old_trusted = sum(1 for i in range(len(texts_used)) if r["assign"][i] is not None
                      and r["la"]["margin"][i] >= WP.P["LA_MARGIN"])
    la = LA.template_assign(ev, r, kinds, texts_used, m0)
    r2 = LA.place_with_assignment(ev, r, kinds, texts_used, la)
    ginfo["line_assignment"] = line_assignment_report(page, r2, dets, texts_used, src)
    info["la_template"] = dict(trusted_before=old_trusted, trusted_after=sum(la["trusted"]),
                               changed=sum(1 for a, b in zip(r["assign"], la["assign"]) if a != b),
                               seconds=round(time.perf_counter() - t0, 2))
    return r2


def run_T_pooled(pages, exp="A"):
    """Arm T with templates learned jointly over several pages of one book
    (pooled), each page's columns scaled to a common letter pitch.
    pages: [(page, ev)] -> {page name: {arm: per_word}}, info."""
    import word_placer_t as T
    ctx = []
    allL, allI = [], []
    ref = None
    for page, ev in pages:
        pwG, ginfo = run_G(page, exp, ev=ev)
        r = ginfo["_r"]
        todo = _todo_lines(r, page["texts"], exp)
        gm = r["model"]
        if ref is None:
            ref = (max(1.0, T.TP["COL_SW"] * ev.sw), gm.unit)
        cs = ref[0] * gm.unit / ref[1]
        lines = [T.make_line(li, toks, ev.hc, ev.sw, cs=cs) for _, li, toks in todo]
        ivs = [[None if d["deleted"] else (d["s0"], d["s1"]) for d in r["diag"]["aligns"][i].words] for i, _, _ in todo]
        ctx.append((page, ev, r, todo, lines, gm))
        allL += lines
        allI += ivs
    gm0 = ctx[0][5]
    m = T.initial_model(allL, allI, lambda k: T.g_width(gm0, k) * 1.0, ref[1])
    t0 = time.perf_counter()
    aligns, hist = T.train(allL, m, ref[1])
    out = {}
    k = 0
    for page, ev, r, todo, lines, gm in ctx:
        W, H = page["truth_page"].W, page["truth_page"].H
        arms = {"T": {}, "Tall": {}}
        for (i, li, toks), L in zip(todo, lines):
            tres, _ = T.word_results(L, aligns[k], ev.sw, gm.unit, ev.hc)
            k += 1
            for name, res in (("T", tres), ("Tall", T.every_box(tres))):
                for ti, x in enumerate(res):
                    arms[name][(i, ti)] = to_placed(x, W, H)
        out[page["name"]] = arms
    return out, dict(history=hist, seconds=round(time.perf_counter() - t0, 2), units=len(m.tmpl), _model=m)


def cmd_tx(a):
    """Arm T template transfer and pooling on two pages of one book (real
    keys, or synthetic seeds of one config): (1) per page; (2) templates
    learned on one page, applied to the other with no re-estimation, both
    ways; (3) templates pooled over both pages. Contact sheets of the learned
    templates go to --sheets (scratch; templates are page images)."""
    import word_placer_t as T
    if a.synth:
        cls, preset = a.synth.split("/")
        pages = [synth_page(cls, dict(preset=preset), sd) for sd in a.seeds]
    else:
        pages = [real_page(*k) for k in real_keys(a.keys)]
    evs = [WP.page_evidence(p["gray"]) for p in pages]
    rows = []

    def score(page, pw, mode, arm, extra=None):
        wrows, _ = grade_page(page, pw)
        sm = rt.summarize_targets([(page["truth_page"], wrows)], True)
        row = dict(page=page["name"], mode=mode, arm=arm, n=sm["n"], correct=sm["located_correct"]["k"],
                   h1=sum(r_["correct_h1"] for r_ in wrows),
                   located=sm["located_precision"]["n"], within1=sum(r_["within1"] for r_ in wrows), **(extra or {}))
        rows.append(row)
        print(f"  {page['name'][:34]:34s} {mode:14s} {arm:5s} side-edge {row['h1']}/{row['n']}  prec {row['h1']}/{row['located']}"
              f"  within1 {row['within1']}", flush=True)

    models = []
    for p, ev in zip(pages, evs):
        arms, info = run_T(p, "A", ev=ev, with_R=False)
        models.append(info["_model"])
        for nm in ("T", "Tall"):
            score(p, arms[nm], "own", nm, dict(seconds=info["seconds_T"]))
        if a.sheets:
            Path(a.sheets).mkdir(parents=True, exist_ok=True)
            keys = T.contact_sheet(info["_model"], str(Path(a.sheets) / f"templates_{p['name'].replace('/', '_')}.png"))
            print(f"    templates: {len(keys)} units; occurrences {sorted(info['_model'].occ.values(), reverse=True)[:12]} ...")
    for i, j in ((0, 1), (1, 0)):
        arms, info = run_T(pages[j], "A", ev=evs[j], with_R=False, model=models[i], learn=False)
        seen = sum(1 for L in info["_lines"] for k in L.ukeys if k in models[i].tmpl)
        tot = sum(len(L.ukeys) for L in info["_lines"])
        for nm in ("T", "Tall"):
            score(pages[j], arms[nm], f"from {i + 1}", nm, dict(units_seen=seen, units=tot))
    out, info = run_T_pooled(list(zip(pages, evs)), "A")
    for p in pages:
        for nm in ("T", "Tall"):
            score(p, out[p["name"]][nm], "pooled", nm)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUT / f"tx_{'synth' if a.synth else 'real'}_{stamp}.json"
    path.write_text(json.dumps(dict(created_at=now(), code=code_fingerprint(), note=a.note, tp=_tp_digest(), rows=rows),
                               ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    if not a.synth:
        with (OUT / "run_log.txt").open("a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] arm T transfer/pooling real scoring -> {path.name}  code {code_fingerprint()}  TP {_tp_digest()}\n"
                    f"  pages: {[p['name'] for p in pages]}\n  note: {a.note}\n")
            for r_ in rows:
                f.write(f"  {r_['page']:24s} {r_['mode']:8s} {r_['arm']:5s} side-edge {r_['h1']}/{r_['n']}  prec {r_['h1']}/{r_['located']}\n")
    print(f"wrote {path}")
    return 0


def flat_intervals(lines, gm, r):
    """Flat start: each line's core ink extent split over its words by G
    width (plus one in-word gap per piece and a word gap per word)."""
    import word_placer_t as T
    gs = gm.snapshot()["gap"]
    out = []
    for L in lines:
        li = L.li
        if not li.M:
            out.append(None); continue
        a, b = float(li.s0[0]), float(li.s1[-1])
        wsum = []
        for w in range(L.nwords):
            ks = [k for k in range(len(L.ukeys)) if L.uword[k] == w]
            wsum.append(sum(T.g_width(gm, L.ukeys[k]) for k in ks) +
                        sum(max(0.0, gs[0][0]) for k in ks[:-1] if L.btype[k] == T.PIECE))
        gapw = max(0.0, gs[1][0])
        tot = sum(wsum) + gapw * (len(wsum) - 1)
        sc = (b - a) / max(1.0, tot)
        x = a
        iv = []
        for w, ww in enumerate(wsum):
            iv.append((x, x + ww * sc))
            x += (ww + gapw) * sc
        out.append(iv)
    return out


def cmd_overlay_real(a):
    """Overlays of placed boxes on each real page (scratch only)."""
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for slug, pg in real_keys(a.keys):
        page = real_page(slug, pg)
        ev = WP.page_evidence(page["gray"])
        if a.arm in T_ARMS:
            texts = src = None
            if a.exp in ("C5", "C15"):
                texts, src, _st, _ = corrupt_lines(page["texts"], 0.05 if a.exp == "C5" else 0.15,
                                                   seed=hash_seed(page["name"], a.exp))
            arms_t, _ = run_T(page, a.exp, ev=ev, texts=texts, src=src, init=T_OPTS["init"],
                              with_R=a.arm in ("TR", "Rc", "RcG"))
            pw = arms_t[a.arm]
        elif a.arm == "draft":
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
