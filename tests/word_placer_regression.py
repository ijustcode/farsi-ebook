#!/usr/bin/env python3
"""Regression test for tests/word_placer.py (Step 4 arm G and line-text
assignment). Synthetic pages only; no books/, no API, no network.

  ./venv/bin/python tests/word_placer_regression.py

Covers: accuracy floors on a fixed synthetic set (clean typewriter, clean
naskh, a tight-spacing typewriter page, a speck page); a constructed O1 case
(a mark just before a line start) that the "not text" state fixes and its
absence does not; O2 (word gaps no wider than a letter gap) placed by width;
fused word pairs (zero gap) never located on the wrong ink; error injection
leaving uncorrupted words in place; non-text detected lines receiving no text;
statuses never including an approximate guess; determinism.
Arm T (tests/word_placer_t.py, section 9): templates learned from a synthetic
typewriter photo page with no face rendered, a tight-gap (O2) page resolved by
appearance where G fails, the O1 marks, substituted words flagged with their
neighbours kept, a monotone objective after round 1 without collapse or drift,
determinism. Arm S and the agreement rules (section 10). The shared vertical
rule on a tight-leading page (section 11) and the template-scored line-text
assignment with an inserted non-text line and a dropped text line (section 12).
The rules between T and S&T and the end-to-end token alignment (section 13).
Re-segmentation groups and gate G2 with perturbed line text (section 14).

Floors are set below the values measured when the test was written (listed
beside each), so they catch regressions, not noise.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_truth_eval as rt  # noqa: E402
import synth_pages as S  # noqa: E402
import word_placer as WP  # noqa: E402
import word_placer_eval as E  # noqa: E402

FAILS = []
ALLOWED = {"located", "line", "unresolved"}


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def page_of(cls, seed, **kw):
    return E.synth_page(cls, kw, seed)


def score(page, pw):
    wrows, _ = E.grade_page(page, pw)
    sm = rt.summarize_targets([(page["truth_page"], wrows)], True)
    return wrows, sm


def statuses(pw):
    return {p["status"] for p in pw.values()}


def main() -> int:
    # ---- 1. accuracy floors on a fixed synthetic set (experiment A and B) ----
    print("1. accuracy floors")
    fixed = [
        ("typewriter clean", page_of("typewriter", 211, preset="clean"), 0.85, 0.98),      # measured A 0.884
        ("modern naskh clean", page_of("modern_naskh", 211, preset="clean"), 0.85, 0.98),
        ("typewriter tight spacing", page_of("typewriter", 211, preset="clean",
                                             degrade=dict(tight_frac=0.6, tight_to=[0.0, 0.8])), 0.50, 0.97),  # measured A 0.575, B 0.621,
        ("typewriter specks", page_of("typewriter", 213, preset="clean",
                                      degrade=dict(specks=40, specks_start_frac=0.7, specks_end_frac=0.2,
                                                   specks_size=2.0, specks_dist_pt=[1.0, 6.0])), 0.85, 0.97),
    ]
    for name, page, floor, pfloor in fixed:
        for exp in ("A", "B"):
            pw, info = E.run_G(page, exp)
            _, sm = score(page, pw)
            lc, pr = sm["located_correct"], sm["located_precision"]
            check(lc["rate"] >= floor, f"{name} {exp}: located&correct {lc['k']}/{lc['n']} = {lc['rate']:.3f} >= {floor}")
            check((pr["rate"] or 0) >= pfloor, f"{name} {exp}: precision {pr['k']}/{pr['n']} >= {pfloor}")
            check(statuses(pw) <= ALLOWED, f"{name} {exp}: statuses {sorted(statuses(pw))} only located/line/unresolved")
            check(sm["wrong_line"]["k"] == 0, f"{name} {exp}: wrong line {sm['wrong_line']['k']} == 0")

    # ---- 2. constructed O1: a mark just before a line start ----
    print("2. O1: stray mark before the line start")
    page = page_of("typewriter", 214, preset="clean")
    gray = page["gray"].copy()
    tp = page["truth_page"]
    ev0 = WP.page_evidence(gray)
    fixed_lines, broken_lines, tried = 0, 0, 0
    # lines whose first (rightmost) word has >= 3 letters: a one-letter word
    # the size of the mark is genuinely ambiguous and must not be located
    chosen = [L.index for L in tp.lines if len(L.words) >= 4 and L.words[0].letters >= 3][:5]
    for n_, li in enumerate(chosen):
        L = tp.lines[li]
        w0 = L.words[0]                          # first word in reading order = rightmost
        xs = [p[0] for p in w0.quad]; ys = [p[1] for p in w0.quad]
        letter = (max(xs) - min(xs)) / max(1.0, w0.letters)
        # beyond a word gap right of the line start, staggered (marks at one
        # offset on every line would form a text-block edge of their own)
        cx = max(xs) + (2.5 + 0.9 * (n_ % 3)) * letter
        cy = (min(ys) + max(ys)) / 2
        r = max(2.0, 1.3 * ev0.sw)               # a narrow-letter sized mark, at line height
        Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
        gray[((X - cx) / (0.8 * r)) ** 2 + ((Y - cy) / (1.6 * r)) ** 2 <= 1.0] = 20.0
        tried += 1
    page2 = dict(page, gray=gray)
    ev = WP.page_evidence(gray)
    pwG, _ = E.run_G(page2, "A", nottext=True, ev=ev)
    pw0, _ = E.run_G(page2, "A", nottext=False, ev=ev)
    wG, _ = score(page2, pwG)
    w0_, _ = score(page2, pw0)
    okG = {tuple(r["target"][0]): r["status"] == "located" and r["correct"] for r in wG}
    ok0 = {tuple(r["target"][0]): r["status"] == "located" and r["correct"] for r in w0_}
    pwU, _ = E.run_G(page, "A", nottext=True)          # the same page without marks
    wU, _ = score(page, pwU)
    okU = {tuple(r["target"][0]): r["status"] == "located" and r["correct"] for r in wU}
    lost = 0
    for li in chosen:
        fixed_lines += bool(okG.get((li, 0)))
        broken_lines += not ok0.get((li, 0))
        lost += sum(1 for k, v in okU.items() if k[0] == li and v and not okG.get(k))
    check(fixed_lines == tried, f"with 'not text': first word of {fixed_lines}/{tried} marked lines located and correct")
    # side effect, not the O1 mechanism: the marks also change page-level
    # measurements (L1's integer glyph height 15 -> 14 px on this small-font
    # page), which moves the core band and the learned fusion rate
    check(lost <= 2 * tried, f"with 'not text': other words of marked lines lost vs the unmarked page: {lost} <= {2 * tried}")
    check(broken_lines >= tried - 2, f"without it: first word of {broken_lines}/{tried} marked lines NOT located-and-correct")
    # the without-state arm must not claim the mark: never a confident wrong box there
    bad0 = sum(1 for r in w0_ if r["status"] == "located" and not r["correct"])
    check(bad0 <= 2, f"without 'not text': wrong located boxes on the page {bad0} <= 2 (it falls back, it does not guess)")

    # ---- 3. O2 and fused pairs, placed by width ----
    print("3. O2 / fused boundaries")
    page = page_of("typewriter", 215, preset="clean", degrade=dict(tight_frac=0.5, tight_to=[0.0, 1.0]))
    pw, _ = E.run_G(page, "A")
    wrows, sm = score(page, pw)
    ok = {tuple(r["target"][0]): r for r in wrows}
    idpos = {}
    for L in page["truth_page"].lines:
        for wi, wid in enumerate([w for w in L.src["word_ids"]]):
            idpos[wid] = (L.index, wi)
    o2_words, o2_ok, o2_bad, fused_bad, fused_n = set(), 0, 0, 0, 0
    for g in page["truth"]["word_gaps"]:
        if not g.get("o2_trigger"):
            continue
        for wid in (g["right_id"], g["left_id"]):
            if wid in idpos:
                o2_words.add(idpos[wid])
        if g.get("gap_px") is not None and g["gap_px"] <= 0:
            for wid in (g["right_id"], g["left_id"]):
                r = ok.get(idpos.get(wid))
                if r is not None:
                    fused_n += 1
                    fused_bad += r["status"] == "located" and not r["correct"]
    for k in o2_words:
        r = ok.get(k)
        if r is None:
            continue
        o2_ok += r["status"] == "located" and r["correct"]
        o2_bad += r["status"] == "located" and not r["correct"]
    n = len(o2_words)
    # Floors, not targets: measured 2026-11-12 at 77/182 correct (101/182 at LOC_MARGIN 4), 8 located-wrong
    # (7% of located), fused 4/20 located-wrong. O2 is NOT solved by width on
    # this page (plan §13 O2 target unmet); these floors only catch regressions.
    check(n >= 100, f"tight page has {n} words at O2 boundaries (gap <= in-word gap)")
    check(o2_ok >= 0.38 * n, f"O2 words located and correct {o2_ok}/{n} >= 38% (floor; measured 42% at LOC_MARGIN 6)")
    check(o2_bad <= 0.10 * max(1, o2_ok + o2_bad), f"O2 words located but wrong {o2_bad} <= 10% of located (floor; measured 7%)")
    check(fused_bad <= max(1, 0.25 * fused_n), f"fused pairs (gap 0): located-but-wrong {fused_bad}/{fused_n} <= 25% (floor; measured 20%)")

    # ---- 4. error injection: uncorrupted words stay in place ----
    print("4. error injection (15%)")
    page = page_of("typewriter", 211, preset="clean")
    texts, src, status, _ = E.corrupt_lines(page["texts"], 0.15, seed=99)
    pw, _ = E.run_G(page, "C", texts=texts, src=src)
    wrows, _ = score(page, pw)
    c = E.corruption_metrics(wrows, status)
    check(c["kept_correct"] >= 0.75 * c["kept"], f"kept words correct {c['kept_correct']}/{c['kept']} >= 75% (floor; measured 80%; plan target 90% unmet)")
    check(c["corrupted_not_located"] >= 0.5 * c["corrupted"],
          f"corrupted positions not located {c['corrupted_not_located']}/{c['corrupted']} >= 50%")
    check(statuses(pw) <= ALLOWED, "corrupted text: statuses only located/line/unresolved")

    # ---- 5. non-text detected lines receive no text ----
    print("5. line-text assignment: non-text lines")
    page = page_of("typewriter", 216, preset="tw_photo_medium")
    pw, info = E.run_G(page, "B")
    la = info["line_assignment"]
    check(la["nontext_detected"] >= 1, f"page has {la['nontext_detected']} detected non-text lines")
    check(la["nontext_detected_given_text"] == 0, f"non-text lines given text: {la['nontext_detected_given_text']} == 0")
    check(la["wrong_and_trusted"] <= 1, f"trusted text lines on the wrong detected line: {la['wrong_and_trusted']} <= 1 (measured 1: a known failure, see report)")

    # ---- 6. determinism ----
    print("6. determinism")
    page = page_of("typewriter", 211, preset="clean", degrade=dict(tight_frac=0.6, tight_to=[0.0, 0.8]))
    a1, _ = E.run_G(page, "B")
    a2, _ = E.run_G(page, "B")
    same = all(a1[k]["status"] == a2[k]["status"] and a1[k]["segments"] == a2[k]["segments"] for k in a1) and a1.keys() == a2.keys()
    check(same, "two runs give identical statuses and boxes")


    # ---- 7. arm V machinery (no API: canned answers and a fake client) ----
    print("7. arm V: marks, parsing, budget guard, cache, hybrid routing")
    import tempfile
    from types import SimpleNamespace
    import word_placer_v as V
    page = page_of("typewriter", 216, preset="tw_photo_medium")
    ev = WP.page_evidence(page["gray"])
    _, _, r, texts, lines, meta = E.v_lines(page, "A", "mark", ev=ev)
    ln, toks, crop, png = lines[2]
    check(len(crop.marks) >= len(toks) + 1, f"line has {len(crop.marks)} candidate marks for {len(toks)} words")
    check(V.text_area_untouched(png, crop), "mark overlay never touches the text area (margin only)")
    sep = [b[1] - a[1] for a, b in zip(crop.marks, crop.marks[1:])]
    check(min(sep) * crop.scale >= V.VP["MARK_MIN_SEP"] - 1e-6, f"marks at least {V.VP['MARK_MIN_SEP']} crop px apart")
    _, _, _, _, lines_c, _ = E.v_lines(page, "A", "coord", ev=ev)
    check(V.text_area_untouched(lines_c[2][3], lines_c[2][2]), "ruler overlay never touches the text area")
    import transcription_experiments as TE
    im = V.make_image(png, crop.img.shape[1], crop.img.shape[0], "t")
    check(TE.api_size(im, V.MODEL_SONNET) == (crop.img.shape[1], crop.img.shape[0]),
          f"crop {crop.img.shape[1]}x{crop.img.shape[0]} is not downsized by the API (<= 2576 px, <= 4784 tokens)")
    # parsing canned answers
    check(V.parse_word("3-7") == (3, 7, True), "parse 'R-L'")
    check(V.parse_word("3-?") == (3, None, True), "parse 'R-?' (no visible boundary)")
    check(V.parse_word("absent") == (None, None, False), "parse 'absent' (word not on the line)")
    check(V.parse_word("garbage")[2] is None, "unparseable answer is flagged, not guessed")
    n = len(crop.marks)
    canned = ["1-3", "3-?", "absent"] + ["x"] * (len(toks) - 3)
    iv = V.answer_intervals(canned, crop, "mark", len(toks))
    check(iv[0]["interval"] is not None and iv[1]["interval"] is None and iv[2]["present"] is False,
          "canned answers: located span, touching edge -> no span, absent -> not on line")
    # stray ink outside every word: a span between marks 2 and 3 leaves mark 1..2 ink unassigned
    iv2 = V.answer_intervals(["2-3"], crop, "mark", 1)
    q = WP.snap_words(crop and meta[ln - 1][0], [iv2[0]["interval"]], ev.sw, r["model"].unit, ev.hc)[0]
    s_ext = V.quad_s_extent(meta[ln - 1][0], q) if q else None
    m2 = dict((k, s_) for k, s_, _ in crop.marks)
    check(s_ext is not None and s_ext[0] >= m2[2] - 1.0, "ink right of the word's right mark is left out (not text)")
    vres = V.v_results(meta[ln - 1][0], toks, iv, ev.sw, r["model"].unit, ev.hc, None)
    check({x["status"] for x in vres} <= ALLOWED, "V statuses only located/line/unresolved")
    # hybrid routing
    g_ok = [dict(status="located"), dict(status="located")]
    g_mixed = [dict(status="located"), dict(status="line")]
    check(not V.route_line(g_ok) and V.route_line(g_mixed), "hybrid asks V only for lines G did not fully locate")
    hy = V.hybrid_results(g_mixed, [dict(status="line", tag="v0"), dict(status="located", tag="v1")])
    check(hy[0]["status"] == "located" and "tag" not in hy[0] and hy[1].get("tag") == "v1",
          "hybrid keeps G's located words and takes V for the rest")
    # budget guard and cache with a fake client
    calls = []

    class FakeMessages:
        def parse(self, **kw):
            calls.append(kw["model"])
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
                                   parsed_output=V.VAnswer(lines=[V.VLine(line=ln, words=["1-2"] * len(toks))]),
                                   stop_reason="end_turn")

    fake = lambda: SimpleNamespace(messages=FakeMessages())  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        run = V.runner(cap=1.0, client_factory=fake, out=Path(tmp))
        (_, req), = V.build_requests("synth:A", [lines[2]], "mark", V.MODEL_SONNET, arm="t", batch=1)
        rec1 = run.call(req)
        rec2 = run.call(req)
        check(len(calls) == 1 and rec2.get("cache_hit") is True, "second identical request is a cache hit (no second call)")
        tiny = V.runner(cap=0.001, client_factory=fake, out=Path(tmp) / "tiny")
        (_, req_o), = V.build_requests("synth:A", [lines[3]], "mark", V.MODEL_OPUS, arm="t", batch=1)
        refused = False
        try:
            tiny.call(req_o)
        except TE.BudgetExceeded:
            refused = True
        check(refused and len(calls) == 1, "budget guard refuses a call whose worst case would pass the cap (no call made)")
        check(req_o.worst_case() > req.worst_case(), "Opus worst case is padded above Sonnet's")

    # ---- 8. arm R (render + warp) on a clean page, own face excluded ----
    print("8. arm R")
    page = page_of("modern_naskh", 211, preset="clean")
    ev = WP.page_evidence(page["gray"])
    arms_r, rinfo = E.run_R(page, "A", ev=ev)
    _, smR = score(page, arms_r["R"])
    check(rinfo["font"] is not None and "Vazirmatn" not in str(rinfo["font"]), f"R chose {rinfo['font']} (page's own face excluded)")
    check(statuses(arms_r["R"]) <= ALLOWED and statuses(arms_r["RG"]) <= ALLOWED, "R statuses only located/line/unresolved")
    check(smR["located_correct"]["rate"] >= 0.0, f"R located&correct {smR['located_correct']['k']}/{smR['located_correct']['n']} (reported, no floor)")

    arm_t_checks()
    arm_s_checks()
    vertical_and_assignment_checks()
    rules_2a_and_e2e_checks()
    groups_and_g2_checks()

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s)")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


def _ok_map(page, pw):
    wrows, sm = score(page, pw)
    return {tuple(r["target"][0]): r["status"] == "located" and r["correct"] for r in wrows}, wrows, sm


def _add_marks(page, chosen, ev0):
    """The section-2 construction: a narrow mark at line height beyond a word
    gap right of each chosen line's start."""
    gray = page["gray"].copy()
    tp = page["truth_page"]
    for n_, li in enumerate(chosen):
        w0 = tp.lines[li].words[0]
        xs = [p[0] for p in w0.quad]; ys = [p[1] for p in w0.quad]
        letter = (max(xs) - min(xs)) / max(1.0, w0.letters)
        cx = max(xs) + (2.5 + 0.9 * (n_ % 3)) * letter
        cy = (min(ys) + max(ys)) / 2
        r = max(2.0, 1.3 * ev0.sw)
        Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
        gray[((X - cx) / (0.8 * r)) ** 2 + ((Y - cy) / (1.6 * r)) ** 2 <= 1.0] = 20.0
    return dict(page, gray=gray)


def arm_t_checks():
    """Arm T (tests/word_placer_t.py): page-trained templates. Run without
    arm R (with_R=False), so nothing is rendered from any font: the shapes
    come from the page alone."""
    import word_placer_t as T
    print("9. arm T: templates learned from the page")
    # ---- 9a. floors on a typewriter photo page; no font rendered ----
    page = page_of("typewriter", 211, preset="tw_photo_medium")
    ev = WP.page_evidence(page["gray"])
    arms, info = E.run_T(page, "A", ev=ev, with_R=False)
    check("font" not in info and not (set(arms) & set(E.R_DEP)), "no face rendered (arm R off): no R-dependent arms")
    okT, wT, smT = _ok_map(page, arms["T"])
    lc, pr = smT["located_correct"], smT["located_precision"]
    check(lc["rate"] >= T_FLOORS["photo_rate"], f"tw_photo_medium A: T located&correct {lc['k']}/{lc['n']} = {lc['rate']:.3f} >= {T_FLOORS['photo_rate']}")
    check((pr["rate"] or 0) >= T_FLOORS["photo_prec"], f"tw_photo_medium A: T precision {pr['k']}/{pr['n']} >= {T_FLOORS['photo_prec']}")
    pwG, _ = E.run_G(page, "A", ev=ev)
    _, _, smG = _ok_map(page, pwG)
    check(lc["k"] >= smG["located_correct"]["k"] + 0.05 * lc["n"],
          f"T beats G on this page by >= 5 points ({lc['k']} vs {smG['located_correct']['k']} of {lc['n']})")
    allst = set().union(*[statuses(p) for p in arms.values()])
    check(allst <= ALLOWED, f"T statuses {sorted(allst)} only located/line/unresolved")
    bypass_box = sum(1 for p in arms["T"].values() if "bypassed" in p.get("reason", "") and p["status"] == "located")
    check(bypass_box == 0, "a bypassed word is never located")
    # ---- 9b. convergence: monotone objective, no collapse, no drift ----
    hist = info["history"]
    objs = [h["objective"] for h in hist[1:]]          # round 0 precedes the one-time unit-length re-estimate
    worst_up = max([b - a for a, b in zip(objs, objs[1:])] or [0.0])
    check(worst_up <= 1e-4 * abs(objs[0]), f"objective non-increasing after round 1 (largest rise {worst_up:.2f}; rounds {len(hist)})")
    m = info["_model"]
    gm = E.run_G(page, "A", ev=ev)[1]["_r"]["model"]
    ratios = [len(m.tmpl[k]) * m.cs / max(1.0, T.g_width(gm, k)) for k in m.tmpl if m.occ.get(k, 0) >= 5]
    check(ratios and 0.5 <= min(ratios) and max(ratios) <= 2.0,
          f"frequent units keep a plausible width: template / G width in [{min(ratios):.2f}, {max(ratios):.2f}] within [0.5, 2]")
    arms0, _ = E.run_T(page, "A", ev=ev, with_R=False, iters=0)
    ok0, w0, sm0 = _ok_map(page, arms0["Tall"])
    _, wA, smA = _ok_map(page, arms["Tall"])
    check(smA["located_correct"]["k"] >= sm0["located_correct"]["k"] - 0.01 * len(wA),
          f"no collapse: every-box correct {smA['located_correct']['k']} after training vs {sm0['located_correct']['k']} with the initial templates (>= minus 1%)")
    shifted = sum(1 for r in wA if r.get("outcome") == "shifted")
    check(shifted <= 0.02 * len(wA), f"no one-word drift: shifted boxes {shifted} <= 2% of words")
    # ---- 9c. determinism ----
    arms2, _ = E.run_T(page, "A", ev=ev, with_R=False)
    same = all(arms["T"][k]["status"] == arms2["T"][k]["status"] and arms["T"][k]["segments"] == arms2["T"][k]["segments"]
               for k in arms["T"]) and arms["T"].keys() == arms2["T"].keys()
    check(same, "T: two runs give identical statuses and boxes")
    # ---- 9d. O2: tight word gaps resolved by appearance where G fails ----
    page = page_of("typewriter", 215, preset="clean", degrade=dict(tight_frac=0.5, tight_to=[0.0, 1.0]))
    ev = WP.page_evidence(page["gray"])
    idpos = {wid: (L.index, wi) for L in page["truth_page"].lines for wi, wid in enumerate(L.src["word_ids"])}
    o2 = set()
    for g in page["truth"]["word_gaps"]:
        if g.get("o2_trigger"):
            o2 |= {idpos[w] for w in (g["right_id"], g["left_id"]) if w in idpos}
    okG, _, _ = _ok_map(page, E.run_G(page, "A", ev=ev)[0])
    arms, _ = E.run_T(page, "A", ev=ev, with_R=False)
    okT, wT, _ = _ok_map(page, arms["T"])
    nT, nG = sum(okT.get(k, False) for k in o2), sum(okG.get(k, False) for k in o2)
    badT = sum(1 for r in wT if tuple(r["target"][0]) in o2 and r["status"] == "located" and not r["correct"])
    check(nT >= T_FLOORS["o2_rate"] * len(o2) and nT >= nG + 0.2 * len(o2),
          f"O2 words located&correct: T {nT}/{len(o2)} (G {nG}); T >= {T_FLOORS['o2_rate']:.0%} and >= G + 20 points")
    check(badT <= 0.05 * max(1, nT + badT), f"O2 words located but wrong (T) {badT} <= 5% of located")
    # ---- 9e. O1: stray mark before the line start ----
    page = page_of("typewriter", 214, preset="clean")
    tp = page["truth_page"]
    ev0 = WP.page_evidence(page["gray"])
    chosen = [L.index for L in tp.lines if len(L.words) >= 4 and L.words[0].letters >= 3][:5]
    page2 = _add_marks(page, chosen, ev0)
    arms, _ = E.run_T(page2, "A", ev=WP.page_evidence(page2["gray"]), with_R=False)
    okT, _, _ = _ok_map(page2, arms["T"])
    first = sum(bool(okT.get((li, 0))) for li in chosen)
    check(first >= len(chosen) - 1, f"O1: first word of {first}/{len(chosen)} marked lines located and correct (mark left as not text)")
    # ---- 9f. a substituted word is flagged; its neighbours keep their boxes ----
    page = page_of("typewriter", 211, preset="tw_photo_medium")
    ev = WP.page_evidence(page["gray"])
    texts = [list(t) for t in page["texts"]]
    src = [[(li, wi) for wi in range(len(t))] for li, t in enumerate(texts)]
    lines_ = [li for li, t in enumerate(texts) if len(t) >= 6][:8]
    pool = [w for t in texts for w in t if len(w) >= 4]
    subs = []
    for n_, li in enumerate(lines_):
        old = texts[li][3]
        cand = [w for w in pool if abs(len(w) - len(old)) <= 1 and not (set(w) & set(old) - set("اوی"))]
        if cand:
            texts[li][3] = cand[n_ % len(cand)]
            subs.append(li)
    armsC, _ = E.run_T(page, "A", ev=ev, texts=texts, src=src, with_R=False)
    armsU, _ = E.run_T(page, "A", ev=ev, with_R=False)
    okC, _, _ = _ok_map(page, armsC["T"])
    okU, _, _ = _ok_map(page, armsU["T"])
    flagged = sum(armsC["T"][(li, 3)]["status"] != "located" for li in subs)
    nb = [(li, w) for li in subs for w in (2, 4) if okU.get((li, w))]
    kept = sum(bool(okC.get(k)) for k in nb)
    check(len(subs) >= 6, f"{len(subs)} lines with a substituted word (letters disjoint from the original)")
    check(flagged >= len(subs) - 1, f"substituted words not located: {flagged}/{len(subs)}")
    check(kept >= 0.8 * len(nb), f"their neighbours keep correct located boxes: {kept}/{len(nb)} (>= 80%)")


def arm_s_checks():
    """Arm S (the audit draft's splitter in external lines) and the
    agreement rules, on a clean synthetic page (own face excluded from R)."""
    print("10. arm S and agreement rules")
    page = page_of("book_naskh", 211, preset="clean")
    ev = WP.page_evidence(page["gray"])
    arms, info = E.run_T(page, "A", ev=ev)
    _, _, smS = _ok_map(page, arms["S"])
    check(smS["located_correct"]["rate"] >= 0.6, f"S located&correct {smS['located_correct']['k']}/{smS['n']} >= 60% (measured 78% at writing)")
    for nm in ("SR", "SG", "ST", "M2", "M3"):
        _, _, sm = _ok_map(page, arms[nm])
        check(statuses(arms[nm]) <= ALLOWED, f"{nm} statuses only located/line/unresolved")
        check((sm["located_precision"]["rate"] or 0) >= 0.97, f"{nm} precision {sm['located_precision']['k']}/{sm['located_precision']['n']} >= 97%")
    n2 = sum(p["status"] == "located" for p in arms["M2"].values())
    n3 = sum(p["status"] == "located" for p in arms["M3"].values())
    check(n2 >= n3, f"M2 locates at least as many words as M3 ({n2} >= {n3})")


def vertical_and_assignment_checks():
    """Follow-ups: the shared vertical rule (word_placer_s.vertical_line_comp)
    on a tight-leading page with dots and marks; the template-scored line-text
    assignment (word_placer_la) with an extra non-text line and a dropped
    text line."""
    print("11. shared vertical rule (tight leading)")
    page = E.synth_page("modern_naskh", dict(preset="clean", leading=1.35), 218)
    ev = WP.page_evidence(page["gray"])
    res = {}
    for v in ("old", "comp"):
        E.T_OPTS["vert"] = v
        res[v] = E.run_T(page, "A", ev=ev, with_R=False)[0]
    E.T_OPTS["vert"] = "old"
    _, wO, smO = _ok_map(page, res["old"]["T"])
    _, wC, smC = _ok_map(page, res["comp"]["T"])
    check(smC["located_correct"]["k"] >= smO["located_correct"]["k"] - 0.01 * smO["n"],
          f"comp rule keeps strict correct: {smC['located_correct']['k']} vs old {smO['located_correct']['k']} of {smO['n']}")
    check(smC["wrong_line"]["k"] == 0, f"comp rule: wrong line {smC['wrong_line']['k']} == 0 at leading 1.35")
    intr = sum(1 for r in wC if r["status"] == "located" and r.get("intrusion_other_line_letters", 0) >= 0.5)
    check(intr == 0, f"comp rule: no located box takes half a letter of another line ({intr})")
    tp = page["truth_page"]
    import box_audit as ba
    cov = tot = 0
    for r in wC:
        if r["status"] != "located" or not r["correct"]:
            continue
        li, wi = r["target"][0]; w = tp.word(li, wi); L = tp.lines[li]
        pts = [p for sg in res["comp"]["T"][(li, wi)]["segments"] for p in rt.region_poly(sg, tp.W, tp.H)]
        fb = ba.quad_to_frame_box(L.frame, pts)
        tot += 1
        cov += fb[1] <= w.fbox[1] + 1.0 and fb[3] >= w.fbox[3] - 1.0
    check(tot and cov >= 0.95 * tot, f"comp rule: box spans the word's full ink height (dots, marks) for {cov}/{tot} correct boxes")

    print("12. template-scored line-text assignment")
    E.T_OPTS["la"] = "T"
    try:
        page = page_of("typewriter", 219, preset="clean")
        tp = page["truth_page"]
        gray = page["gray"].copy()
        # an extra non-text line: a handwritten-looking stroke between lines 4 and 5
        a, b = tp.lines[4], tp.lines[5]
        ya = max(p[1] for w in a.words for p in w.quad); yb = min(p[1] for w in b.words for p in w.quad)
        xs = [p[0] for w in a.words for p in w.quad]
        x0, x1 = min(xs), max(xs)
        yc = (ya + yb) / 2
        X = np.arange(int(x0 + 0.2 * (x1 - x0)), int(x1 - 0.1 * (x1 - x0)))
        Y = yc + 0.25 * (yb - ya) * np.sin(X / 9.0)
        for dx in range(-1, 2):
            for dy in range(-2, 3):
                gray[np.clip((Y + dy).astype(int), 0, gray.shape[0] - 1), np.clip(X + dx, 0, gray.shape[1] - 1)] = 25.0
        page2 = dict(page, gray=gray)
        ev = WP.page_evidence(gray)
        arms, info = E.run_T(page2, "B", ev=ev, with_R=False)
        _, w, sm = _ok_map(page2, arms["T"])
        la = info.get("la_template", {})
        check(sm["wrong_line"]["k"] == 0, f"extra non-text line: wrong-line boxes {sm['wrong_line']['k']} == 0")
        check(la.get("trusted_after", 0) >= la.get("trusted_before", 99) - 1,
              f"extra non-text line: trusted text lines {la.get('trusted_after')} (length assignment {la.get('trusted_before')})")
        base, _ = E.run_T(page, "B", ev=WP.page_evidence(page["gray"]), with_R=False)
        _, _, sm0 = _ok_map(page, base["T"])
        check(sm["located_correct"]["k"] >= sm0["located_correct"]["k"] - 0.02 * sm["n"],
              f"extra non-text line: T located&correct {sm['located_correct']['k']} vs {sm0['located_correct']['k']} without it (of {sm['n']})")
        # a dropped text line: the reading misses printed line 6
        texts = [list(t) for li_, t in enumerate(page["texts"]) if li_ != 6]
        src = [[(li_, wi) for wi in range(len(t))] for li_, t in enumerate(page["texts"]) if li_ != 6]
        ev = WP.page_evidence(page["gray"])
        arms, info = E.run_T(page, "B", ev=ev, texts=texts, src=src, with_R=False)
        _, w, sm = _ok_map(page, arms["T"])
        check(sm["wrong_line"]["k"] == 0, f"dropped text line: wrong-line boxes {sm['wrong_line']['k']} == 0")
        lost = [r for r in w if r["target"][0][0] == 6 and r["status"] == "located"]
        check(not lost, "dropped text line: no word of the missing line is located (it has no text)")
        check((sm["located_precision"]["rate"] or 0) >= 0.97, f"dropped text line: precision {sm['located_precision']['k']}/{sm['located_precision']['n']} >= 97%")
        g2 = info.get("g2_boxes_on_nontext", {}).get("T", 0)
        check(g2 == 0, f"G2: boxes shown on non-text detected lines {g2} == 0")
    finally:
        E.T_OPTS["la"] = "length"


def rules_2a_and_e2e_checks():
    """Follow-up 2a rules (TA, TQ, TAQ) and the end-to-end token alignment
    (word_placer_e2e), synthetic input only; no cache, no network."""
    print("13. rules between T and S&T; end-to-end alignment")
    import word_placer_e2e as X
    page = page_of("typewriter", 211, preset="tw_photo_medium")
    ev = WP.page_evidence(page["gray"])
    texts = [list(t) for t in page["texts"]]
    src = [[(li, wi) for wi in range(len(t))] for li, t in enumerate(texts)]
    subs = []
    for li, t in enumerate(texts):
        if len(t) >= 6 and len(subs) < 6:
            t[3] = "ققققق"            # a word whose letters are not printed anywhere on the line
            subs.append(li)
    arms, info = E.run_T(page, "A", ev=ev, texts=texts, src=src)
    for nm in ("TA", "TQ", "TAQ"):
        check(statuses(arms[nm]) <= ALLOWED, f"{nm} statuses only located/line/unresolved")
    loc = lambda nm: {k for k, v in arms[nm].items() if v["status"] == "located"}  # noqa: E731
    check(loc("TA") <= loc("Tall"), "TA locates only words that T shows a box for")
    check(loc("TAQ") <= loc("TA") and loc("TQ") <= loc("T"), "the guard only removes boxes")
    byp = [li for li in subs if "bypassed" in arms["T"][(li, 3)]["reason"]]
    nb_loc = sum(1 for li in byp for w in (2, 4) if arms["TQ"][(li, w)]["status"] == "located")
    check(byp and nb_loc == 0, f"TQ: neighbours of {len(byp)} bypassed words are never located ({nb_loc})")
    _, _, smQ = _ok_map(page, arms["TQ"])
    check((smQ["located_precision"]["rate"] or 0) >= 0.93, f"TQ precision {smQ['located_precision']['k']}/{smQ['located_precision']['n']} >= 93%")
    truth = [["aa", "bb", "cc"], ["dd", "ee"]]
    model = [["aa", "bx", "cc"], ["ee", "zz"]]
    src2, st2 = X.align_to_truth(truth, model)
    check(st2[(0, 0)] == "read" and st2[(0, 1)] == "misread" and st2[(1, 0)] == "missing" and st2[(1, 1)] == "read",
          f"e2e alignment: read / misread / missing classified ({st2})")
    check(src2[1][1][0] == "x" and src2[1][0] == (1, 1), "e2e alignment: extra token is unmatched, shifted token found")
    old_env = os.environ.pop("ANTHROPIC_BASE_URL", None)
    try:
        refused = False
        try:
            X.run(keys=["none/0"], readers=())
        except SystemExit:
            refused = True
        check(refused, "e2e refuses to run unless the API base URL is a dead local address")
    finally:
        if old_env is not None:
            os.environ["ANTHROPIC_BASE_URL"] = old_env


def groups_and_g2_checks():
    """Follow-ups: re-segmented words in the end-to-end alignment (joins,
    splits, letter changes, line ends) and gate G2 with perturbed reader text
    (invented, repeated and dropped lines; --la T2), synthetic only."""
    print("14. re-segmentation groups; G2 with perturbed line text")
    import word_placer_e2e as X
    import word_placer_la as LA
    truth = [["ab", "cd", "ef"], ["gh", "ij", "kl"], ["mn", "op"]]
    g = []
    src, st = X.align_to_truth(truth, [["abcd", "ef"], ["g", "h", "ij", "kl"], ["mn", "op"]], g)
    check([x["kind"] for x in g] == ["join", "split"] and st[(0, 0)] == st[(0, 1)] == "join" and st[(1, 0)] == "split",
          f"2-into-1 join and 1-into-2 split found as groups ({[x['kind'] for x in g]})")
    g = []
    _, st = X.align_to_truth(truth, [["abce", "ef"], ["gh", "ij", "kl"], ["mn", "op"]], g)
    check(not g and st[(0, 0)] in ("missing", "misread") and st[(0, 1)] in ("missing", "misread"),
          "a join that also changes a letter is not a group")
    g = []
    _, st = X.align_to_truth(truth, [["ab", "cd", "ef"], ["gh", "ij", "klmn"], ["op"]], g)
    check(not g, "a join across a printed line end is not a group (model tokens on one line only)")
    g = []
    src, st = X.align_to_truth(truth, [["ab", "cd", "ef"], ["gh", "ijkl"], ["mn", "op"]], g)
    check([x["kind"] for x in g] == ["join"] and g[0]["truth"] == [(1, 1), (1, 2)] and src[1][1] == ("g", 0, 0),
          "a join at a line end is a group of that line")
    # G2 with perturbed text, confirmed assignment
    page = page_of("typewriter", 216, preset="tw_photo_medium")
    ev = WP.page_evidence(page["gray"])
    E.T_OPTS.update(la="T2")
    try:
        base, _ = E.run_T(page, "C", ev=ev, texts=[list(t) for t in page["texts"]],
                          src=[[(li, wi) for wi in range(len(t))] for li, t in enumerate(page["texts"])], with_R=False)
        _, _, sm0 = _ok_map(page, base["T"])
        pool = [w for t in page_of("typewriter", 1216, preset="clean")["texts"] for w in t]
        n = len(page["texts"])
        for name, kw in (("invented line", dict(n_invent=1, n_repeat=0, n_drop=0)),
                         ("repeated line", dict(n_invent=0, n_repeat=1, n_drop=0)),
                         ("dropped line", dict(n_invent=0, n_repeat=0, n_drop=1))):
            t2, s2 = LA.perturb_lines(page["texts"], seed=17, pool=pool, **kw)
            if name == "invented line":     # at the end, next to the signature and stamp zone
                k = next(i for i, row in enumerate(s2) if row and row[0][0] == "x")
                t2.append(t2.pop(k)); s2.append(s2.pop(k))
            arms, info = E.run_T(page, "C", ev=ev, texts=t2, src=s2, with_R=False)
            _, w, sm = _ok_map(page, arms["T"])
            unprinted = sum(1 for kk, x in arms["T"].items() if kk[0] == "x" and x["status"] == "located"
                            and name == "invented line")
            check(info.get("g2_boxes_on_nontext", {}).get("T", 0) == 0, f"{name}: no box on a non-text detected line")
            check(sm["wrong_line"]["k"] == 0, f"{name}: wrong-line boxes {sm['wrong_line']['k']} == 0")
            if name == "invented line":
                check(unprinted == 0, f"{name}: no located box for the invented words ({unprinted})")
            lost = sm0["located_correct"]["k"] - sm["located_correct"]["k"]
            # a dropped line loses its own words; the free-loop test may also leave a neighbour
            # unresolved (conservative), but nothing may shift (wrong-line 0 above)
            allow = (2 * max(len(t) for t in page["texts"]) + 2) if name == "dropped line" else 3
            check(lost <= allow, f"{name}: other words keep their boxes (lost {lost} <= {allow} of {sm0['located_correct']['k']})")
    finally:
        E.T_OPTS.update(la="length")


T_FLOORS = dict(photo_rate=0.70, photo_prec=0.93, o2_rate=0.70)


if __name__ == "__main__":
    sys.exit(main())
