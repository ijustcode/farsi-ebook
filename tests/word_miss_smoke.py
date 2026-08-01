#!/usr/bin/env python3
"""Smoke test for the word-miss-distance metric (see METRIC.md).

Four sections, all offline and free:

  (a) unit    — the metric core, including METRIC.md's own worked example, and
                the no-covered-words degenerate path through `_grade`.
  (b) identity — prove exact-span context, not the candidate box, identifies
                 repeated and mistranscribed target phrases.
  (c) frozen   — re-score the FROZEN sample and prove the v5 instrument and
                 exact saved boxes reproduce their reference bit-identically.
                 Every verdict count, acc@1 and acc@0 in every report section
                 must match the reference report.
  (d) sanity  — the metric agrees with verdicts where it must.

Run:  ./venv/bin/python tests/word_miss_smoke.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bbox_score  # noqa: E402
from farsi2epub.workspace import Workspace  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# REBASED 2026-08-01 for score instrument v5's target-identity fix. V4 let the
# candidate box choose among fuzzy/repeated truth windows, so the thing under
# test could move its own answer key. V5 resolves the exact Markdown span via
# stable surrounding page context, with concat-exact split/merge handling and
# a one-to-one fuzzy fallback. This is a SANCTIONED INSTRUMENT CORRECTION, not
# locator progress: v5 regrades v4's exact serialized boxes and v4 stays intact.
REFERENCE = PROJECT_ROOT / "out" / "score_word_miss_baseline_v5.json"
V4_REFERENCE = PROJECT_ROOT / "out" / "score_word_miss_baseline_v4.json"
FROZEN = PROJECT_ROOT / "tests" / "data" / "bbox_sample240.json"

REF_ACC1 = 0.8255
REF_ACC0 = 0.3362
REF_TRUTH_SHA1 = "e38e0ed322edb770c33a40a0680db60c8b146b14"
REF_BOX_SHA1 = "6d739ee16febdf344664dea7c37d7698a103615b"

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


# ---------------------------------------------------------------------------
# (a) unit
# ---------------------------------------------------------------------------


def section_a() -> None:
    print("\n== (a) unit ==")
    wmd = bbox_score._word_miss_distances

    # METRIC.md: "my father bought some milk"; target = milk (index 4), the
    # box covers {my, father} = {0, 1}; nearest covered word is father.
    check("METRIC.md worked example", wmd({4}, {0, 1}) == [3], str(wmd({4}, {0, 1})))
    check("covered target scores 0", wmd({2}, {1, 2, 3}) == [0])
    check("multi-word target", wmd({5, 6}, {2, 3}) == [2, 3], str(wmd({5, 6}, {2, 3})))
    check("nearest on either side", wmd({5}, {0, 7}) == [2], str(wmd({5}, {0, 7})))
    check(
        "over-wide box costs nothing",
        wmd({3, 4}, set(range(0, 20))) == [0, 0],
    )

    # A hand-built reading whose rectangle covered no words at all.
    reading = {
        "line_text_full": ["alpha beta gamma", "delta epsilon zeta"],
        "boxed_text": "",
        "boxed_line_index": -1,
        "legible": True,
    }
    variants = [bbox_score.locate._norm_words("delta epsilon")]
    g = bbox_score._grade(reading, variants)
    check(
        "empty box -> degenerate no_covered_words",
        g["degenerate"] == "no_covered_words",
        str(g["degenerate"]),
    )
    check("empty box -> word_miss_sum is None", g["word_miss_sum"] is None)
    check(
        "empty box -> target_word_count survives",
        g["target_word_count"] == 2,
        str(g["target_word_count"]),
    )
    check("page_word_count recorded", g["page_word_count"] == 6, str(g["page_word_count"]))

    # A real, measurable miss through _grade: the box covers "alpha beta"
    # (indices 0,1) and the truth sits at the far end of the reading. The
    # located truth window is the locator's business (it may run a word wide),
    # so assert the METRIC's property against whatever window it reported
    # rather than hardcoding one.
    reading2 = dict(reading, boxed_text="alpha beta", boxed_line_index=0)
    g2 = bbox_score._grade(reading2, [bbox_score.locate._norm_words("epsilon zeta")])
    words2, _ = bbox_score._flatten(reading2)
    truth2 = bbox_score.locate._norm_words(g2["truth_text"])
    t0 = bbox_score._find_contiguous(words2, truth2, [0] * len(words2), -1)
    expected = bbox_score._word_miss_distances(
        set(range(t0, t0 + len(truth2))), {0, 1}
    )
    check(
        "measurable miss through _grade",
        g2["word_miss_distances"] == expected
        and g2["word_miss_sum"] == sum(expected)
        and all(d > 0 for d in expected),
        f"{g2['word_miss_distances']} vs {expected} sum={g2['word_miss_sum']}",
    )
    check("measurable miss is not degenerate", g2["degenerate"] is None)

    # An exact box scores zero.
    reading3 = dict(reading, boxed_text="delta epsilon", boxed_line_index=1)
    g3 = bbox_score._grade(reading3, [bbox_score.locate._norm_words("delta epsilon")])
    check(
        "exact box scores 0",
        g3["verdict"] == "exact" and g3["word_miss_sum"] == 0,
        f"{g3['verdict']} sum={g3['word_miss_sum']}",
    )

    # Wilcoxon helper: identical inputs must be all-zero deltas -> underpowered.
    p, n = bbox_score._wilcoxon_signed_rank_p([0.0] * 40)
    check("wilcoxon all-zero -> None", p is None and n == 0, f"p={p} n={n}")
    p, n = bbox_score._wilcoxon_signed_rank_p([-3.0] * 20)
    check(
        "wilcoxon one-sided improvement is significant",
        p is not None and p < 0.05 and n == 20,
        f"p={p} n={n}",
    )
    check(
        "average-rank ties",
        bbox_score._ranked([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0],
        str(bbox_score._ranked([1.0, 1.0, 3.0])),
    )


# ---------------------------------------------------------------------------
# (b) target identity is box-independent
# ---------------------------------------------------------------------------


def _frozen_page_case(case_id: str) -> tuple[dict, str, dict, list[list[str]]]:
    blob = json.loads(FROZEN.read_text(encoding="utf-8"))
    case = next(c for c in blob["cases"] if c["id"] == case_id)
    ws = Workspace.load(case["slug"])
    page_md = bbox_score._page_md_matching_sha(
        ws, case["page"], case["md_sha1"]
    )
    if page_md is None:
        raise AssertionError(f"missing replay Markdown for {case_id}")
    hit = bbox_score._PageReadCache(ws).get(
        bbox_score._page_read_key(
            case["slug"], case["page"], bbox_score.MODEL_STRONG
        )
    )
    if hit is None or hit.get("reading") is None:
        raise AssertionError(f"missing frozen page truth for {case_id}")
    variants, insertion = bbox_score._variants_for(case)
    if insertion:
        raise AssertionError(f"unexpected insertion case: {case_id}")
    return case, page_md, hit["reading"], variants


def section_b() -> None:
    print("\n== (b) exact-span target identity is box-independent ==")

    # p4 is the concrete repeated/fuzzy-lure failure: the hunk says کرمانی,
    # the page actually prints کرمی at that span, and a different line contains
    # the lexically stronger مرادی کرمانی. Instrument v4 let a misplaced box
    # nominate that other line as its own answer key.
    #
    # p13 is the wrong-token failure: جماته/جماعته is really چمباتمه between
    # stable neighbours, while نماز جماعت elsewhere is the tempting false hit.
    examples = (
        ("bachehaye_ghali:p4:h0", "کرمی", "مرادی کرمانی"),
        ("bachehaye_ghali:p13:h1", "چمباتمه", "نماز جماعت"),
    )
    for case_id, printed_target, false_lure in examples:
        case, page_md, reading, variants = _frozen_page_case(case_id)
        grades = []
        for covered in (printed_target, false_lure):
            probe = dict(
                reading,
                boxed_text=covered,
                boxed_line_index=-1,
            )
            grades.append(
                bbox_score._grade(
                    probe,
                    variants,
                    page_md=page_md,
                    query_span=tuple(case["query"]["span"]),
                )
            )
        on_target, on_lure = grades
        check(
            f"{case_id}: exact span resolves printed target",
            on_target["truth_text"] == printed_target
            and on_target["truth_identity_source"] == "span_context"
            and on_target["word_miss_sum"] == 0,
            str(
                {
                    "truth": on_target["truth_text"],
                    "source": on_target["truth_identity_source"],
                    "miss": on_target["word_miss_sum"],
                }
            ),
        )
        check(
            f"{case_id}: moving box onto lexical lure cannot move truth",
            on_lure["truth_text"] == printed_target
            and on_lure["word_miss_sum"] is not None
            and on_lure["word_miss_sum"] > 0,
            str(
                {
                    "truth": on_lure["truth_text"],
                    "miss": on_lure["word_miss_sum"],
                }
            ),
        )

    # Tokenization is part of target identity. The old ±1 matcher could match
    # both query words to one printed word and silently omit a required
    # neighbour. These cases cover whitespace split/merge, ASCII-vs-Persian
    # footnote digits, fused boundary words, a transposed target word, and the
    # rule that an alt containing appended footnote text cannot expand a
    # literal primary target.
    target_set_examples = (
        ("bachehaye_ghali:p9:h100002", ["۴", "خود", "با"]),
        ("bachehaye_ghali:p12:h1", ["خریده", "۳"]),
        ("bachehaye_ghali:p15:h100002", ["۲", "علو", "علی"]),
        ("bachehaye_ghali:p23:h100001", ["یی۳", "طرف", "جو"]),
        (
            "bachehaye_ghali:p69:h2",
            ["میانپاهاشبود", "و", "لبهاش", "را", "بادکردهبود"],
        ),
        ("bachehaye_ghali:p70:h4", ["می"]),
        ("bachehaye_ghali:p74:h0", ["تا", "شد"]),
        ("bachehaye_ghali:p74:h1", ["رفتیم"]),
        (
            "boof-e-koor:p93:h100001",
            ["درمیان", "این", "فشار", "گوارا", "عرق"],
        ),
    )
    for case_id, expected_words in target_set_examples:
        case, page_md, reading, variants = _frozen_page_case(case_id)
        words, _line_of = bbox_score._flatten(reading)
        span = case["query"].get("span")
        info = bbox_score._resolve_truth_window(
            words,
            variants,
            page_md,
            tuple(span) if span else None,
        )
        actual = words[info["t_start"] : info["t_start"] + info["t_len"]]
        check(
            f"{case_id}: target set keeps every required printed word",
            actual == expected_words and info["truth_confident"] is True,
            str(actual),
        )

    # An exact repeated token is the minimal falsification test: the span says
    # the SECOND نشان. A box on the first occurrence must lose, even though its
    # text is byte-for-byte identical to the real target.
    page_md = "پیش اول نشان پس سپس پیش دوم نشان پایان"
    start = page_md.rindex("نشان")
    reading = {
        "line_text_full": [page_md],
        "boxed_text": "نشان",
        "boxed_line_index": 0,
        "legible": True,
    }
    variants = [bbox_score.locate._norm_words("نشان")]
    wrong = bbox_score._grade(
        reading,
        variants,
        boxed_start_hint=2,
        page_md=page_md,
        query_span=(start, start + len("نشان")),
    )
    right = bbox_score._grade(
        reading,
        variants,
        boxed_start_hint=7,
        page_md=page_md,
        query_span=(start, start + len("نشان")),
    )
    check(
        "repeated token: span selects second occurrence without box anchor",
        wrong["word_miss_sum"] == 5
        and right["word_miss_sum"] == 0
        and wrong["truth_identity_source"] == "span_context",
        f"first={wrong['word_miss_sum']} second={right['word_miss_sum']}",
    )

    # Without an exact span/context contract, repeated exact text has no safe
    # identity. It is a judge limitation, not a locator miss and not a licence
    # to let the box nominate its preferred occurrence.
    ambiguous = bbox_score._grade(
        {
            "line_text_full": ["آغاز گل میان گل پایان"],
            "boxed_text": "گل",
            "boxed_line_index": 0,
            "legible": True,
        },
        [bbox_score.locate._norm_words("گل")],
    )
    check(
        "no-span repeated target is a neutral judge failure",
        ambiguous["verdict"] == "judge_unusable"
        and ambiguous["degenerate"] == "judge_unusable"
        and ambiguous["truth_identity_source"] == "query_ambiguous"
        and ambiguous["word_miss_sum"] is None,
        str(
            {
                "verdict": ambiguous["verdict"],
                "degenerate": ambiguous["degenerate"],
                "source": ambiguous["truth_identity_source"],
            }
        ),
    )


# ---------------------------------------------------------------------------
# (c) frozen v5 reproducibility
# ---------------------------------------------------------------------------


def section_c() -> dict:
    print("\n== (c) frozen v5 exact-box reproducibility ==")
    if not REFERENCE.is_file() or not V4_REFERENCE.is_file():
        check(
            "v4 and v5 reference reports present",
            False,
            f"v4={V4_REFERENCE.is_file()} v5={REFERENCE.is_file()}",
        )
        return {}
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    v4 = json.loads(V4_REFERENCE.read_text(encoding="utf-8"))

    tmp = Path(tempfile.mkdtemp(prefix="word_miss_smoke_")) / "rescore.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "bbox_score.py"),
        "score",
        "--regrade-report",
        str(V4_REFERENCE),
        "--offline",
        "--no-refine",
        "--out",
        str(tmp),
    ]
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.is_file():
        check(
            "re-score of frozen sample ran",
            False,
            f"rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}",
        )
        return {}
    check("re-score of frozen sample ran", True)
    new = json.loads(tmp.read_text(encoding="utf-8"))

    check(
        "score instrument is v5",
        new["run"].get("score_cache_version") == 5,
        str(new["run"].get("score_cache_version")),
    )
    check(
        "cost was $0",
        float(new["run"].get("cost_usd") or 0.0) == 0.0,
        str(new["run"].get("cost_usd")),
    )
    check(
        f"truth_sha1 == {REF_TRUTH_SHA1[:12]}",
        new["run"].get("truth_sha1") == REF_TRUTH_SHA1,
        str(new["run"].get("truth_sha1")),
    )
    check(
        f"box_sha1 == {REF_BOX_SHA1[:12]}",
        new["run"].get("box_sha1") == REF_BOX_SHA1
        and ref["run"].get("box_sha1") == REF_BOX_SHA1,
        f"new={new['run'].get('box_sha1')} ref={ref['run'].get('box_sha1')}",
    )
    v4_source_sha1 = bbox_score._sha1(
        V4_REFERENCE.read_text(encoding="utf-8")
    )
    check(
        "v5 provenance names the exact v4 source bytes",
        new["run"].get("regrade_source_sha1") == v4_source_sha1
        and ref["run"].get("regrade_source_sha1") == v4_source_sha1,
        str(new["run"].get("regrade_source_sha1")),
    )
    check(
        f"acc@1 == {REF_ACC1}",
        new["overall"].get("acc@1") == REF_ACC1,
        str(new["overall"].get("acc@1")),
    )
    check(
        f"acc@0 == {REF_ACC0}",
        new["overall"].get("acc@0") == REF_ACC0,
        str(new["overall"].get("acc@0")),
    )
    check("overall metrics bit-identical", new["overall"] == ref["overall"])

    for section in ("by_tier", "by_book", "by_kind", "by_issue_type", "by_feature"):
        check(
            f"{section}: metrics bit-identical",
            new.get(section) == ref.get(section),
        )

    # Strongest form: every v5 per-case field reproduces, and every serialized
    # treatment box is byte-for-byte-equivalent JSON to both v4 and v5.
    check(
        "all v5 per-case fields bit-identical",
        new["cases"] == ref["cases"],
    )
    v4_boxes = [(c["id"], c.get("box")) for c in v4["cases"]]
    ref_boxes = [(c["id"], c.get("box")) for c in ref["cases"]]
    new_boxes = [(c["id"], c.get("box")) for c in new["cases"]]
    check(
        "v4 -> v5 -> replay serialized boxes are exactly identical",
        v4_boxes == ref_boxes == new_boxes,
    )

    compare_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "bbox_score.py"),
        "compare",
        "--baseline", str(V4_REFERENCE),
        "--candidate", str(REFERENCE),
    ]
    refused = subprocess.run(compare_cmd, capture_output=True, text=True)
    check(
        "compare refuses instrument drift by default",
        refused.returncode == 2
        and "DIFFERENT scoring instruments" in refused.stderr,
        f"rc={refused.returncode} stderr={refused.stderr[:160]!r}",
    )
    gated = subprocess.run(
        compare_cmd + ["--allow-instrument-drift", "--gate"],
        capture_output=True,
        text=True,
    )
    check(
        "instrument-drift override is diagnostic-only (never gateable)",
        gated.returncode == 2 and "diagnostic only" in gated.stderr,
        f"rc={gated.returncode} stderr={gated.stderr[:160]!r}",
    )
    diagnostic = subprocess.run(
        compare_cmd + ["--allow-instrument-drift"],
        capture_output=True,
        text=True,
    )
    check(
        "explicit instrument-drift diagnostic is allowed",
        diagnostic.returncode == 0
        and "do not interpret as locator progress" in diagnostic.stderr,
        f"rc={diagnostic.returncode} stderr={diagnostic.stderr[:160]!r}",
    )
    return new


# ---------------------------------------------------------------------------
# (d) correlation sanity
# ---------------------------------------------------------------------------


def section_d(report: dict) -> None:
    print("\n== (d) correlation sanity ==")
    if not report:
        check("report available for sanity checks", False, "section (c) did not run")
        return
    cases = report["cases"]

    exact = [c for c in cases if c["verdict"] == "exact"]
    bad = [c["id"] for c in exact if c.get("word_miss_sum") != 0]
    check(
        f"every exact case scores 0 (n={len(exact)})",
        not bad,
        f"{len(bad)} offenders: {bad[:5]}",
    )

    wrong_line = [
        c for c in cases
        if c["verdict"] == "wrong_line" and c.get("word_miss_effective") is not None
    ]
    mean_wl = (
        sum(float(c["word_miss_effective"]) for c in wrong_line) / len(wrong_line)
        if wrong_line else None
    )
    check(
        f"wrong_line mean word_miss > 10 (n={len(wrong_line)})",
        mean_wl is not None and mean_wl > 10,
        f"mean={mean_wl}",
    )

    o = report["overall"]
    graded = [
        c for c in cases if c["verdict"] not in bbox_score._EXCLUDED_VERDICTS
    ]
    perfect = [
        c
        for c in graded
        if c.get("degenerate") is None and c.get("word_miss_sum") == 0
    ]
    expected_perfect_rate = round(len(perfect) / len(graded), 4)
    check(
        "perfect_rate is measured-zero / FULL graded denominator",
        o["perfect_rate"] == expected_perfect_rate
        and o["n_denominator"] == len(graded),
        f"reported={o['perfect_rate']} expected={expected_perfect_rate} "
        f"denominator={o['n_denominator']}/{len(graded)}",
    )
    # Do not compare this to acc@1: the golden metric deliberately charges
    # nothing for over-wide boxes, while acc@1's start-shift guard can reject
    # a box that still covers every target word.
    # Degenerates must be counted, never dropped from perfect_rate.
    n_den = o["n_denominator"]
    measured = [
        c
        for c in graded
        if c.get("degenerate") is None
        and c.get("word_miss_effective") is not None
    ]
    check(
        "magnitude excludes degenerates but headline denominator keeps them",
        o["word_miss_n"] == len(measured) <= n_den,
        f"measured={o['word_miss_n']} denominator={n_den}",
    )
    check(
        "penalized cases are excluded from the magnitude",
        all(
            c.get("word_miss_effective") is None
            for c in cases
            if c.get("degenerate") in bbox_score._PENALIZED_DEGENERATE
        ),
    )
    check("mean_iou still reported", o.get("mean_iou") is not None, str(o.get("mean_iou")))


def main() -> int:
    section_a()
    section_b()
    report = section_c()
    section_d(report)
    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} check(s): " + ", ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
