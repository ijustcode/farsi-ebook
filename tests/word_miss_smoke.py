#!/usr/bin/env python3
"""Smoke test for the word-miss-distance metric (see METRIC.md).

Three sections, all offline and free:

  (a) unit    — the metric core, including METRIC.md's own worked example, and
                the no-covered-words degenerate path through `_grade`.
  (b) additive — re-score the FROZEN sample and prove nothing existing moved:
                every verdict count, acc@1 and acc@0 in every report section
                must be bit-identical to the reference report. If this fails
                you changed behaviour; fix the code, never the assertion.
  (c) sanity  — the new metric agrees with the old verdicts where it must.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# REBASED 2026-07-31 for the judge alignment fix (JUDGE_FIX.md). The previous
# reference (out/score_010_precedence_refine.json, acc@1 0.7787 / acc@0 0.3915)
# was produced by a judge that mapped reading lines onto the wrong printed ink
# whenever two lines fused into one detected row run. Rebasing the constants is
# the SANCTIONED exception to "fix the code, never the assertion": the task was
# to change grading behaviour. The check itself stays — it still proves the
# scorer reproduces a frozen reference bit-identically.
REFERENCE = PROJECT_ROOT / "out" / "score_word_miss_baseline_v4.json"
FROZEN = PROJECT_ROOT / "tests" / "data" / "bbox_sample240.json"

REF_ACC1 = 0.8638
REF_ACC0 = 0.4809
REF_TRUTH_SHA1 = "e38e0ed322edb770c33a40a0680db60c8b146b14"

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
# (b) purely-additive proof
# ---------------------------------------------------------------------------


def section_b() -> dict:
    print("\n== (b) purely additive: re-score of the frozen sample ==")
    if not REFERENCE.is_file():
        check("reference report present", False, f"missing {REFERENCE}")
        return {}
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))

    tmp = Path(tempfile.mkdtemp(prefix="word_miss_smoke_")) / "rescore.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "bbox_score.py"),
        "score",
        "--cases",
        str(FROZEN),
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
        f"acc@1 == {REF_ACC1}",
        new["overall"].get("acc@1") == REF_ACC1,
        str(new["overall"].get("acc@1")),
    )
    check(
        f"acc@0 == {REF_ACC0}",
        new["overall"].get("acc@0") == REF_ACC0,
        str(new["overall"].get("acc@0")),
    )
    check(
        "overall verdicts identical",
        new["overall"].get("verdicts") == ref["overall"].get("verdicts"),
        f"{new['overall'].get('verdicts')} vs {ref['overall'].get('verdicts')}",
    )

    for section in ("by_tier", "by_book", "by_kind", "by_issue_type", "by_feature"):
        keys = sorted(set(ref.get(section, {})) | set(new.get(section, {})))
        check(f"{section}: same keys", set(ref.get(section, {})) == set(new.get(section, {})))
        bad: list[str] = []
        for k in keys:
            a, b = ref.get(section, {}).get(k, {}), new.get(section, {}).get(k, {})
            for field in ("acc@1", "acc@0", "verdicts", "n", "n_denominator"):
                if a.get(field) != b.get(field):
                    bad.append(f"{k}.{field}: {a.get(field)} != {b.get(field)}")
        check(f"{section}: acc/verdicts bit-identical", not bad, "; ".join(bad[:5]))

    # Strongest form: every pre-existing per-case field must be untouched.
    ref_cases = {c["id"]: c for c in ref["cases"]}
    new_cases = {c["id"]: c for c in new["cases"]}
    check("case id set identical", set(ref_cases) == set(new_cases))
    diffs: list[str] = []
    for cid in sorted(set(ref_cases) & set(new_cases)):
        a, b = ref_cases[cid], new_cases[cid]
        for field in ("verdict", "acc1", "shift_words", "line_delta", "align_score"):
            if a.get(field) != b.get(field):
                diffs.append(f"{cid}.{field}: {a.get(field)} != {b.get(field)}")
    check(
        "per-case verdict/acc1/shift/line_delta identical",
        not diffs,
        "; ".join(diffs[:5]),
    )
    return new


# ---------------------------------------------------------------------------
# (c) correlation sanity
# ---------------------------------------------------------------------------


def section_c(report: dict) -> None:
    print("\n== (c) correlation sanity ==")
    if not report:
        check("report available for sanity checks", False, "section (b) did not run")
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

    # perfect_rate can never exceed acc@0's population share plus nothing:
    # a perfect case is exactly a measured zero, so it must not exceed acc@1.
    o = report["overall"]
    check(
        "perfect_rate <= acc@1",
        o["perfect_rate"] <= o["acc@1"],
        f"{o['perfect_rate']} vs {o['acc@1']}",
    )
    # Degenerates must be counted, never dropped from perfect_rate.
    n_den = o["n_denominator"]
    n_perfect = round(o["perfect_rate"] * n_den)
    check(
        "perfect_rate uses the FULL graded denominator",
        n_perfect + o["word_miss_n"] - n_perfect <= n_den
        and o["word_miss_n"] <= n_den,
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
    report = section_b()
    section_c(report)
    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} check(s): " + ", ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
