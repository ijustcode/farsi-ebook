#!/usr/bin/env python3
"""Real-page truth scorer (tests/real_truth_eval.py): metric definitions on
hand-built cases with known answers, freeze/refusal contracts, exclusions.

Synthetic and offline: writes a one-page audit document with Latin
placeholder tokens into a temp dir, never reads books/, never calls the API.

    ./venv/bin/python tests/real_truth_eval_regression.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import fitz  # noqa: E402

import box_audit as ba  # noqa: E402
import real_truth_eval as rt  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


W, H = 1000, 700
WORD_W, GAP, HALF_H = 100.0, 20.0, 20.0   # 5 letters per word -> 20 px per letter
LETTER = WORD_W / 5
CY = [100, 200, 300, 400, 550]            # line 0 is text-flagged; line 4 slopes 3 degrees
ANGLE = [0, 0, 0, 0, 3.0]


def word_box(wi):
    u1 = 300 - wi * (WORD_W + GAP)       # RTL: word 0 is rightmost
    return [u1 - WORD_W, -HALF_H, u1, HALF_H]


def make_root(tmp: Path, slug="synthbook", page=1) -> Path:
    tdir = tmp / "books" / slug / "truth"
    tdir.mkdir(parents=True)
    img = tdir / f"{page:04d}.page.png"
    fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, W, H), False).save(str(img))
    txt = tdir / f"{page:04d}.lines.txt"
    tokens = [[f"{chr(97 + li)}{chr(97 + wi)}xyz" for wi in range(5)] for li in range(5)]
    txt.write_text("\n".join(" ".join(t) for t in tokens) + "\n")
    lines = []
    for li in range(5):
        frame = dict(cx=500.0, cy=float(CY[li]), angle_deg=ANGLE[li])
        words = [dict(box=word_box(wi), state="accepted", uncertain=False, reasons=[]) for wi in range(5)]
        lines.append(dict(tokens=tokens[li], frame=frame, box=ba.union_box([w["box"] for w in words], 2),
                          words=words, state="accepted", line_fixed=False, uncertain=False, reasons=[],
                          notes=[], text_flag=None))
    lines[0]["text_flag"] = {"note": "synthetic flag", "at": "2026-01-01T00:00:00+00:00"}
    doc = dict(schema_version=ba.SCHEMA_VERSION, tool_version=ba.TOOL_VERSION, kind="audit",
               page_id="p_synthetic0000000000", slug=slug, page=page, source_pdf_sha256="0" * 64,
               image=dict(path=ba.rel(img, tmp), sha256=ba.file_sha(img), width_px=W, height_px=H),
               text_source=dict(path=ba.rel(txt, tmp), sha256=ba.file_sha(txt), kind="owner_text_truth_lines"),
               status="draft", revision=1, lines=lines, regions=[])
    ba.derive(doc)
    errs = ba.validate_page(doc)
    assert not errs, errs
    assert doc["status"] == "audited"
    ba.write_json_atomic(ba.audit_path(tmp, slug, page), doc)
    return tmp


def rect_of_frame(t, li, box):
    """Normalised axis-aligned bounds of a frame box of line li."""
    q = ba.box_quad_px(t.lines[li].frame, box)
    xs, ys = [p[0] for p in q], [p[1] for p in q]
    return {"rect": [min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H]}


def placed(*regions, status="located"):
    return {"status": status, "segments": list(regions)}


def main():
    with tempfile.TemporaryDirectory() as d:
        root = make_root(Path(d))
        slug = "synthbook"

        # --- freeze ---------------------------------------------------------
        rec = rt.freeze_page(root, slug, 1, cls="typewritten_photo")
        check(rec["excluded_lines"] == [{"line": 0, "reason": "text_flag", "auto": True,
                                         "note_sha256": rec["excluded_lines"][0]["note_sha256"]}],
              "text-flagged line 0 excluded automatically at freeze")
        check(rec["graded_lines"] == 4 and rec["graded_words"] == 20 and rec["words"] == 25,
              "freeze counts: 4/5 lines, 20/25 words graded")
        check(rec["provenance"] == rt.OWNER, "owner-audited provenance recorded")
        check("synthetic flag" not in json.dumps(rec), "freeze manifest stores the flag note's hash, not its text")
        t = rt.load_truth(root, slug, 1)
        check(len(t.graded_words()) == 20, "graded words exclude the flagged line")

        # --- word metrics on hand-built cases --------------------------------
        def g(target, *regions, status="located"):
            return rt.grade_target(t, target, placed(*regions, status=status))

        exact = g([(1, 2)], rect_of_frame(t, 1, word_box(2)))
        check(exact["correct"] and exact["outcome"] == "correct", "exact box is correct")
        check(exact["iou"] == 1.0, f"IoU of the exact box is 1.0 ({exact['iou']})")

        b = word_box(2)
        swallow = g([(1, 2)], rect_of_frame(t, 1, [b[0] - GAP - WORD_W, b[1], b[2], b[3]]))
        check(not swallow["correct"] and swallow["plan_correct"] and swallow["outcome"] == "too_wide_other_word",
              "over-wide box swallowing the neighbour: v1 incorrect, plan-literal correct, too_wide_other_word")
        tall = g([(1, 2)], rect_of_frame(t, 1, [b[0], b[1] - 40, b[2], b[3]]))
        check(not tall["correct"] and tall["plan_correct"] and tall["outcome"] == "too_wide_blank",
              f"over-tall box into blank space fails on overreach ({tall['overreach']})")
        slack = g([(1, 2)], rect_of_frame(t, 1, [b[0] - 4, b[1] - 3, b[2] + 4, b[3] + 3]))
        check(slack["correct"], "a few pixels of slack stays correct")

        wl = g([(1, 2)], rect_of_frame(t, 2, word_box(2)))
        check(wl["wrong_line"] and wl["outcome"] == "wrong_line" and not wl["correct"], "box on the next line is wrong_line")

        # O2: shifted 1.5 letters toward the next word (left): loses 1.5 letters, takes 0.5 letter of w3.
        o2 = g([(3, 2)], rect_of_frame(t, 3, [b[0] - 1.5 * LETTER, b[1], b[2] - 1.5 * LETTER, b[3]]))
        check(o2["outcome"] == "o2_boundary" and abs(o2["o2"]["missing_letters"] - 1.5) < 0.05,
              f"edge off by 1.5 letters is an O2 boundary error ({o2.get('o2')})")
        o2b = g([(3, 1)], rect_of_frame(t, 3, [word_box(1)[0], b[1], word_box(1)[2] - 1.0 * LETTER, b[3]]))
        check(o2b["outcome"] == "o2_boundary", "box losing its last letter is O2")
        sub = g([(3, 1)], rect_of_frame(t, 3, [word_box(1)[0] + 0.3 * LETTER, b[1], word_box(1)[2], b[3]]))
        check(sub["correct"], "0.3-letter shortfall (94% covered) is still correct")

        # Sloped line (3 degrees): the axis-aligned bounds of the word quad is correct.
        slope = g([(4, 0)], rect_of_frame(t, 4, word_box(0)))
        check(slope["correct"], f"axis-aligned box of a 3-degree word is correct (overreach {slope['overreach']})")
        steep = g([(4, 0)], {"rect": [t.word(4, 0).quad[0][0] / W, (550 - 20) / H, t.word(4, 0).quad[1][0] / W, (550 + 20) / H]})
        check(not steep["correct"], "a horizontal box ignoring the slope misses part of the word")

        # Multi-word span across a line wrap.
        span = g([(1, 4), (2, 0)], rect_of_frame(t, 1, word_box(4)), rect_of_frame(t, 2, word_box(0)))
        check(span["correct"], "two-segment span across a line wrap is correct")
        envelope = rt.grade_target(t, [(1, 4), (2, 0)], placed(
            {"rect": [min(rect_of_frame(t, 1, word_box(4))["rect"][0], rect_of_frame(t, 2, word_box(0))["rect"][0]),
                      100 / H - 20 / H, 0.9, 300 / H]}))
        check(not envelope["correct"], "the envelope of a wrapped span is not correct (segments are graded)")

        # --- word-map summary: O1 cascade, denominators ----------------------
        words, spans = rt.build_targets(t, 7, 30)
        check(len(words) == 20 and all(k[0] != 0 for tg in words + spans for k in tg.words),
              "targets never include excluded-line words")
        res = []
        for tg in words:
            li, wi = tg.words[0]
            if li == 2 and wi < 4:       # O1: words 0-3 each placed on the next word's ink
                res.append(placed(rect_of_frame(t, 2, word_box(wi + 1))))
            elif li == 2:
                res.append({"status": "unresolved", "segments": []})
            elif li == 1 and wi == 0:
                res.append(placed(rect_of_frame(t, 1, word_box(0)), status="estimated"))
            else:
                res.append(placed(rect_of_frame(t, li, word_box(wi))))
        rows = [rt.grade_target(t, tg.words, r) for tg, r in zip(words, res)]
        s = rt.summarize_targets([(t, rows)], True)
        check(s["n"] == 20, "word-map denominator is the 20 graded words")
        check(s["O1_cascades"] == {"count": 1, "words": 4, "longest": 4}, f"one O1 cascade of 4 words ({s['O1_cascades']})")
        check(s["located_correct"]["k"] == 14 and s["located_precision"]["n"] == 18,
              f"located&correct 14/20, 18 located ({s['located_correct']}, {s['located_precision']})")
        check(s["shown_correct"]["k"] == 15 and s["no_box"]["k"] == 1, "estimated box counts as shown, not located")
        lo, hi = s["located_correct"]["wilson95"]
        check(0.45 < lo < 0.7 and 0.8 < hi < 0.95, f"Wilson interval reported ({lo}, {hi})")

        # Excluded line: a box on it is never counted against the target's line or as text.
        on_ex = g([(1, 1)], rect_of_frame(t, 0, word_box(1)))
        check(on_ex["outcome"] == "off_text" and not on_ex["wrong_line"],
              "a box on the excluded line touches no graded text (off_text, not wrong_line)")

        # --- lines: merged, split, spurious, ignored -------------------------
        def line_rect(li, u0=None, u1=None):
            bx = list(t.lines[li].box)
            if u0 is not None:
                bx[0] = u0
            if u1 is not None:
                bx[2] = u1
            return rect_of_frame(t, li, bx)
        r1 = line_rect(1)["rect"]
        merged = {"rect": [line_rect(2)["rect"][0], line_rect(2)["rect"][1], line_rect(3)["rect"][2], line_rect(3)["rect"][3]]}
        lb = t.lines[4].box
        mid = (lb[0] + lb[2]) / 2
        cands = [{"rect": r1}, merged, line_rect(4, u1=mid), line_rect(4, u0=mid), line_rect(0),
                 {"rect": [0.05, 0.93, 0.15, 0.97]}]
        L = rt.grade_lines(t, cands)
        check(L["truth_lines"] == 4 and L["excluded_lines"] == 1, "line denominators exclude the flagged line")
        check(L["clean_recall"]["k"] == 1 and L["clean_truth_lines"] == [1], "only line 1 is a clean one-to-one match")
        check(L["merged_candidates"] == 1 and L["truth_lines_in_merges"] == 2, "lines 2+3 merged in one candidate")
        check(L["split_lines"] == 1 and L["missed_lines"] == 0, "line 4 split in two halves, not missed")
        check(L["spurious_candidates"] == 1 and L["ignored_candidates"] == 1,
              f"blank-area box spurious, excluded-line box ignored ({L['spurious_candidates']}, {L['ignored_candidates']})")
        check(L["clean_precision"]["n"] == 5, "ignored candidate is out of the precision denominator")
        perfect = rt.grade_lines(t, rt.TruthLines().find_lines(rt.build_page_input(root, t, None)))
        check(perfect["clean_recall"]["rate"] == 1.0 and perfect["clean_precision"]["rate"] == 1.0,
              "truth lines fed back score 100% recall and precision")

        # --- text-free word segments ------------------------------------------
        segs = [rect_of_frame(t, 1, word_box(wi)) for wi in range(5)]
        segs.append(rect_of_frame(t, 2, [word_box(1)[0], -HALF_H, word_box(0)[2], HALF_H]))
        b0 = word_box(0)
        segs += [rect_of_frame(t, 3, [b0[0], -HALF_H, b0[0] + 50, HALF_H]),
                 rect_of_frame(t, 3, [b0[0] + 50, -HALF_H, b0[2], HALF_H])]
        S = rt.grade_word_segments(t, segs)
        check(S["clean_recall"]["k"] == 5 and S["merged_boxes"] == 1 and S["truth_words_in_merges"] == 2
              and S["split_words"] == 1, f"word segments: 5 clean, 1 merge of 2, 1 split ({S})")

        # --- sanity: truth fed back ---------------------------------------------
        pi = rt.build_page_input(root, t, None)
        tw = rt.TruthWords().place(pi, words + spans)
        trows = [rt.grade_target(t, tg.words, r) for tg, r in zip(words + spans, tw)]
        check(all(r["correct"] for r in trows), "truth words fed back are 100% correct (words and spans)")

        # --- truth-to-markdown mapping (query construction) -----------------------
        md = "# " + " ".join(pi.line_texts[1]) + "\n\n" + " ".join(pi.line_texts[2][:2]) + " qqqqq " + " ".join(pi.line_texts[2][3:])
        m = rt.map_truth_to_md(pi, md)
        check(all((1, wi) in m and m[(1, wi)][2] for wi in range(5)), "md mapping: heading markup ignored, exact words mapped")
        check((2, 2) in m and not m[(2, 2)][2], "md mapping: a misread word maps position-wise and is marked inexact")
        check((3, 0) not in m, "md mapping: words absent from the transcription are not queryable")

        # --- refusals -------------------------------------------------------------
        ap = ba.audit_path(root, slug, 1)
        doc = json.loads(ap.read_text())
        doc["revision"] += 7
        doc["cursor"] = {"line": 3}
        ba.write_json_atomic(ap, doc)
        check(rt.load_truth(root, slug, 1).graded_words() is not None,
              "navigation-only change (revision/cursor) is not a truth change")
        doc["lines"][1]["words"][0]["box"][0] -= 5
        ba.derive(doc)
        ba.write_json_atomic(ap, doc)
        try:
            rt.load_truth(root, slug, 1)
            check(False, "changed audit file is refused")
        except rt.TruthError as e:
            check("no longer matches" in str(e) and "freeze --key synthbook/1 --force" in str(e),
                  "changed audit file is refused with the re-freeze command")
        try:
            rt.freeze_page(root, slug, 1, cls="typewritten_photo")
            check(False, "re-freeze without --force is refused")
        except rt.TruthError:
            check(True, "re-freeze without --force is refused")
        rt.freeze_page(root, slug, 1, cls="typewritten_photo", force=True)
        check(rt.load_truth(root, slug, 1).audit_sha256 == rt.content_sha(json.loads(ap.read_text())),
              "--force re-freeze accepts the new audit")
        for fn in (lambda: rt.freeze_page(root, "hadpi-articles", 3), lambda: rt.load_truth(root, "hadpi-articles", 4),
                   lambda: rt.evaluate(root, [("hadpi-articles", 5)], line_finders=[], word_placers=[], segmenters=())):
            try:
                fn()
                check(False, "held-out page refused")
            except rt.TruthError as e:
                check("held out" in str(e), "held-out hadpi page refused (freeze/load/evaluate)")
        # Agent-audited pages are their own group.
        doc2 = json.loads(ap.read_text())
        doc2["status"], doc2["previous_audit"] = "previously_audited", {"auditor": "agent"}
        ap2 = ba.audit_path(root, slug, 2)
        doc2["page"] = 2
        ba.write_json_atomic(ap2, doc2)
        r2 = rt.freeze_page(root, slug, 2, cls="printed_scan")
        t2 = rt.load_truth(root, slug, 2)
        check(r2["provenance"] == rt.AGENT and rt.group_of(t2) != rt.group_of(t) and "not owner-audited" in r2["note"],
              "agent-audited page is frozen under its own group")
        rep = rt.evaluate(root, [(slug, 1), (slug, 2)], line_finders=["truth"], word_placers=["truth"], segmenters=())
        check(set(rep["words"]["truth"]["groups"]) == {"owner_audited/typewritten_photo", "agent_audited/printed_scan"},
              "results are grouped by provenance/class, never pooled across provenance")
        check(math.isclose(rep["words"]["truth"]["groups"]["owner_audited/typewritten_photo"]["located_correct"]["rate"], 1.0),
              "evaluate: truth placer scores 100% on the synthetic page")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED")
        sys.exit(1)
    print("all real_truth_eval checks passed")


if __name__ == "__main__":
    main()
