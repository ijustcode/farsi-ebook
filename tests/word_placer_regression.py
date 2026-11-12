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

Floors are set below the values measured when the test was written (listed
beside each), so they catch regressions, not noise.
"""
from __future__ import annotations

import math
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

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s)")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
