#!/usr/bin/env python3
"""Synthetic-only regressions for v6. No book files, images, or APIs are read.

These tests verify the instrument and guardrails; they are NOT a validation of
real locator accuracy or evidence for promoting a production algorithm.
Run after activating venv: python tests/bbox_metrics_regression.py
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import bbox_benchmark as bb
import bbox_metrics as m


def page(pid="p"):
    lines = []
    for li in range(2):
        lid = f"{pid}:l{li}"
        words = [{"word_id": f"{lid}:w{wi}", "order": wi, "text": f"واژه{wi}",
                  "rect": [.86-.1*wi, .1+.3*li, .94-.1*wi, .2+.3*li]} for wi in range(8)]
        lines.append({"line_id": lid, "order": li, "rect": [.15, .1+.3*li, .95, .2+.3*li], "words": words})
    return {"image_sha256": m.sha(pid), "status": "audited", "auditor": "synthetic fixture",
            "audited_at": "2026-09-05T00:00:00Z", "ambiguities_resolved": True,
            "unreadable_reason": "", "lines": lines}


def target(pid="p", indices=(3,)):
    words = [w for line in page(pid)["lines"] for w in line["words"]]
    return {"status": "audited", "ambiguity_resolution": "resolved",
            "word_ids": [words[i]["word_id"] for i in indices], "insertion_anchor": None, "note": ""}


def box_at(indices=(3,), pid="p"):
    words = [w for line in page(pid)["lines"] for w in line["words"]]
    return {"segments": [words[i]["rect"] for i in indices]}


def refreeze(data):
    """Seal wholly fabricated test data, never any real benchmark annotation."""
    data["manifest_sha256"] = hashlib.sha256(bb._json_file_bytes(data["manifest"])).hexdigest()
    data["annotations"]["manifest_sha256"] = data["manifest_sha256"]
    data["annotation_sha256"] = hashlib.sha256(bb._json_file_bytes(data["annotations"])).hexdigest()
    data["integrity"] = bb._integrity(data["manifest"], data["annotations"])
    data["benchmark_sha256"] = m.sha({k: v for k, v in data.items() if k != "benchmark_sha256"})
    return data


def benchmark(npages=2, cases_per_page=1):
    manifest = {"schema_version": bb.SCHEMA_VERSION,
                "split_policy": {"id": bb.SPLIT_ID, "unit": "page", "heldout_min_cases": 1, "heldout_min_books": 1},
                "render_policy": {"id": bb.RENDER_ID, "long_edge_px": bb.LONG_EDGE_STD},
                "books": [], "pages": [], "cases": []}
    annotations = {"schema_version": bb.SCHEMA_VERSION, "pages": {}, "targets": {}}
    boxes = {}
    for pi in range(npages+1):
        pid, slug = f"p{pi:03d}", f"synthetic-book-{pi%2}"
        split = "heldout" if pi < npages else "dev"
        cids = [f"{pid}:c{ci:03d}" for ci in range(cases_per_page)]
        image = {"path": f"pages/{pid}.png", "sha256": m.sha(pid), "width_px": 1200, "height_px": 1568,
                 "render": {"render_id": bb.RENDER_ID, "long_edge_px": bb.LONG_EDGE_STD, "alpha": False},
                 "source": {"pdf_sha256": m.sha(slug), "page_width_pt": 600, "page_height_pt": 784}}
        manifest["pages"].append({"page_id": pid, "slug": slug, "page": pi+1, "split": split,
                                  "markdown_sha256": m.sha([pid, "md"]), "image": image, "case_ids": cids})
        annotations["pages"][pid] = page(pid)
        for cid in cids:
            manifest["cases"].append({"case_id": cid, "page_id": pid, "kind": "replacement", "split": split,
                                      "target_annotation_id": "target:"+cid,
                                      "hunk": {"old": "واژه", "new": "واژه۲", "markdown_span": [0, 4],
                                               "candidate_spans": [[0, 4]], "ctx_before": "", "ctx_after": "", "occurrence": 1}})
            annotations["targets"][cid] = target(pid)
            boxes[cid] = box_at(pid=pid)
    for slug in sorted({p["slug"] for p in manifest["pages"]}):
        ps = [p for p in manifest["pages"] if p["slug"] == slug]
        manifest["books"].append({"slug": slug, "pdf_sha256": m.sha(slug), "page_count": len(ps),
                                  "case_count": sum(len(p["case_ids"]) for p in ps)})
    data = refreeze({"schema_version": bb.SCHEMA_VERSION, "manifest": manifest, "annotations": annotations})
    errors = bb.validate_frozen(data)
    if errors:
        raise AssertionError(errors)
    return data, {"boxes": boxes}


class GeometryRegression(unittest.TestCase):
    def test_area_union_does_not_doublecount_overlap(self):
        self.assertAlmostEqual(m.coverage([0, 0, 1, 1], [[0, 0, .6, 1], [.4, 0, .8, 1]]), .8)
        self.assertAlmostEqual(m.coverage([0, 0, 1, 1], [[0, 0, 1, .6], [0, .4, 1, .8]]), .8)

    def test_disjoint_segments_use_union_not_envelope(self):
        expected = target(indices=(7, 8))
        segments = box_at((7, 8))
        segments.update(x0=0., y0=0., x1=1., y1=1.)
        g = m.grade(page(), expected, segments)
        self.assertTrue(g["full"])
        self.assertTrue(g["tight"])
        self.assertEqual(g["excess"], 0.)
        self.assertEqual(g["word_miss"], 0)

    def test_count_equality_is_not_coverage(self):
        g = m.grade(page(), target(indices=(2, 3)), box_at((4, 5)))
        self.assertFalse(g["full"])
        self.assertFalse(g["buffered"])
        self.assertEqual(g["word_miss_distances"], [2, 1])
        self.assertEqual(g["excess"], 2.)

    def test_ninety_percent_every_word_not_average(self):
        wr = page()["lines"][0]["words"][3]["rect"]
        def crop(f):
            return {"segments": [[wr[0], wr[1], wr[0]+f*(wr[2]-wr[0]), wr[3]]]}
        self.assertTrue(m.grade(page(), target(), crop(.9))["full"])
        self.assertFalse(m.grade(page(), target(), crop(.8999))["full"])
        almost = crop(.8)
        almost["segments"].append(page()["lines"][0]["words"][2]["rect"])
        g = m.grade(page(), target(indices=(2, 3)), almost)
        self.assertFalse(g["full"])  # Average target area is .90; one word is missing.
        self.assertEqual(g["word_miss_distances"], [0, 1])

    def test_fractional_excess_and_two_neighbors(self):
        words = page()["lines"][0]["words"]
        r = words[1]["rect"]
        treatment = box_at()
        treatment["segments"].append([r[0], r[1], (r[0]+r[2])/2, r[3]])
        g = m.grade(page(), target(), treatment)
        self.assertTrue(g["full"])
        self.assertFalse(g["tight"])
        self.assertTrue(g["buffered"])
        self.assertAlmostEqual(g["excess"], .5)
        self.assertFalse(m.grade(page(), target(), box_at((0, 3)))["buffered"])
        self.assertTrue(m.grade(page(), target(), box_at((1, 2, 3, 4, 5)))["buffered"])
        self.assertFalse(m.grade(page(), target(), box_at((1, 2, 3, 4, 5, 6)))["buffered"])

    def test_word_miss_crosses_line_in_reading_order(self):
        g = m.grade(page(), target(indices=(8,)), box_at((7,)))
        self.assertEqual(g["word_miss_distances"], [1])
        self.assertTrue(g["wrong_line"])

    def test_oversized_box_full_but_excess_not_tight(self):
        g = m.grade(page(), target(), {"x0": 0, "y0": 0, "x1": 1, "y1": 1})
        self.assertTrue(g["full"])
        self.assertFalse(g["tight"])
        self.assertFalse(g["buffered"])
        self.assertEqual(g["excess"], 15.)
        self.assertFalse(g["wrong_line"])  # Additional lines are excess, not a shifted placement.
        self.assertEqual(g["off_target_line_word_coverage"], 8.)

    def test_no_box_and_no_covered_word_fail_without_synthetic_distance(self):
        for box in (None, {"x0": 0, "y0": .8, "x1": 1, "y1": .9}):
            g = m.grade(page(), target(), box)
            self.assertTrue(g["unresolved"])
            self.assertFalse(g["full"])
            self.assertIsNone(g["word_miss"])
        self.assertEqual(m.grade(page(), target(), None)["unresolved_reason"], "no_box")

    def test_invalid_boxes_rejected_even_for_unknown_truth(self):
        unknown = {**target(), "status": "unreadable"}
        for bad in ({}, {"segments": []}, {"segments": "bad"}, {"x0": 0, "y0": 0, "x1": float("nan"), "y1": 1},
                    {"x0": False, "y0": 0, "x1": 1, "y1": 1}, {"segments": [[0, 0, 1, 0]]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.grade(page(), unknown, bad)


class InsertionRegression(unittest.TestCase):
    def insertion(self, before, after):
        t = target(indices=())
        t["insertion_anchor"] = {"before_word_id": before, "after_word_id": after}
        return t

    def test_gap_midpoint_separate_from_word_success(self):
        t = self.insertion("p:l0:w2", "p:l0:w3")
        a, b = page()["lines"][0]["words"][2:4]
        x = (a["rect"][0]+b["rect"][2])/2
        g = m.grade(page(), t, {"x0": x-.002, "y0": .14, "x1": x+.002, "y1": .16}, "insertion")
        self.assertTrue(g["boundary_hit"])
        self.assertFalse(g["full"])
        summary = m.metrics([g, m.grade(page(), target(), box_at())])
        self.assertEqual(summary["denominator"], 1)
        self.assertEqual(summary["insertion_boundary_hit_rate"], 1.)
        self.assertEqual(summary["full_target_coverage_rate"], 1.)

    def test_wrap_requires_both_endpoints(self):
        t = self.insertion("p:l0:w7", "p:l1:w0")
        self.assertFalse(m.grade(page(), t, box_at((7,)), "insertion")["boundary_hit"])
        self.assertTrue(m.grade(page(), t, box_at((7, 8)), "insertion")["boundary_hit"])

    def test_bad_missing_reversed_nonadjacent_anchors_rejected(self):
        for anchors in ((None, None), ("p:l0:w3", "p:l0:w2"), ("p:l0:w2", "p:l0:w4"), (None, "p:l0:w3")):
            with self.subTest(anchors=anchors), self.assertRaises(ValueError):
                m.grade(page(), self.insertion(*anchors), None, "insertion")
        t = self.insertion("p:l0:w2", "p:l0:w3")
        t["word_ids"] = ["p:l0:w2"]
        with self.assertRaises(ValueError):
            m.grade(page(), t, None, "insertion")

    def test_omitted_printed_words_use_word_coverage(self):
        t = self.insertion("p:l0:w2", "p:l0:w4")
        t["word_ids"] = ["p:l0:w3"]
        g = m.grade(page(), t, box_at((3,)), "insertion")
        self.assertTrue(g["full"])
        self.assertFalse(g["insertion"])
        self.assertEqual(m.metrics([g])["denominator"], 1)

    def test_unknown_insertion_stays_in_its_fixed_denominator(self):
        t = {**self.insertion(None, None), "status": "unreadable"}
        g = m.grade(page(), t, None, "insertion")
        summary = m.metrics([g, m.grade(page(), {**target(), "status": "unreadable"}, None, "replacement")])
        self.assertEqual(summary["denominator"], 1)
        self.assertEqual(summary["judge_uncertain"], 2)
        self.assertEqual(summary["insertion_cases"], 1)
        self.assertEqual(summary["insertion_boundary_hit_rate"], 0.)
        self.assertEqual(summary["full_target_coverage_rate"], 0.)


class ReportRegression(unittest.TestCase):
    def test_real_score_and_identical_box_regrade(self):
        frozen, boxes = benchmark()
        original = m.score(frozen, boxes)
        repeated = m.score(frozen, original, regrade=True)
        self.assertEqual(original["run"]["box_sha256"], repeated["run"]["box_sha256"])
        self.assertEqual(original["cases"], repeated["cases"])
        self.assertEqual(original["overall"], repeated["overall"])
        self.assertTrue(repeated["run"]["identical_serialized_boxes"])
        self.assertTrue(repeated["run"]["source"]["regrade"])

    def test_historical_regrade_mapping_and_sha_proof(self):
        frozen, boxes = benchmark()
        selected = [c["case_id"] for c in frozen["manifest"]["cases"] if c["split"] == "heldout"]
        rows = [{"id": f"old:{i}", "box": boxes["boxes"][cid]} for i, cid in enumerate(selected)]
        historical = {"run": {"box_sha1": m._historical_box_sha1(rows)}, "cases": rows}
        mapping = {r["id"]: cid for r, cid in zip(rows, selected)}
        report = m.score(frozen, historical, regrade=True, id_map=mapping)
        self.assertEqual(report["run"]["source"]["source_box_sha1"], historical["run"]["box_sha1"])
        self.assertEqual([r["box"] for r in report["cases"]], [r["box"] for r in rows])
        historical["cases"][0]["box"] = None
        with self.assertRaisesRegex(ValueError, "box_sha1 mismatch"):
            m.score(frozen, historical, regrade=True, id_map=mapping)

    def test_missing_duplicate_and_extra_cases_rejected(self):
        frozen, boxes = benchmark()
        bad = copy.deepcopy(boxes)
        del bad["boxes"][frozen["manifest"]["cases"][0]["case_id"]]
        with self.assertRaisesRegex(ValueError, "omits"):
            m.score(frozen, bad)
        duplicate = {"cases": [{"id": "duplicate", "box": None}]*2}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            m.score(frozen, duplicate)
        bad = copy.deepcopy(boxes)
        bad["boxes"]["foreign"] = None
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            m.score(frozen, bad)
        with self.assertRaisesRegex(ValueError, "SHA|proof"):
            m.score(frozen, {"cases": [{"id": k, "box": v} for k, v in boxes["boxes"].items()]}, regrade=True)

    def test_duplicate_json_fields_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"bad.json"
            for value in ('{"boxes":{"a":null,"a":null}}', '{"n":NaN}'):
                p.write_text(value)
                with self.assertRaises(ValueError):
                    m._read_json(p)

    def test_pending_and_tampered_frozen_truth_refused(self):
        frozen, boxes = benchmark()
        frozen["annotations"]["pages"]["p000"]["status"] = "pending"
        refreeze(frozen)  # Even correctly rehashed labels cannot bypass audit validation.
        with self.assertRaisesRegex(ValueError, "pending"):
            m.score(frozen, boxes)
        frozen, boxes = benchmark()
        frozen["annotations"]["pages"]["p000"]["lines"][0]["words"][3]["rect"][0] += .001
        with self.assertRaisesRegex(ValueError, "mismatch"):
            m.score(frozen, boxes)

    def test_unknown_truth_stays_in_coverage_denominator(self):
        frozen, boxes = benchmark()
        cid = frozen["manifest"]["cases"][0]["case_id"]
        frozen["annotations"]["targets"][cid].update(status="unreadable", word_ids=[], note="Synthetic occlusion")
        refreeze(frozen)
        report = m.score(frozen, boxes)
        self.assertEqual(report["overall"]["denominator"], 2)
        self.assertEqual(report["overall"]["full_target_coverage_rate"], .5)
        self.assertEqual(report["overall"]["ordinary_judge_uncertain"], 1)

    def test_compare_refuses_truth_targets_images_instrument_and_denominator_drift(self):
        frozen, boxes = benchmark()
        baseline = m.score(frozen, boxes)
        for mutate in (lambda f: f["annotations"]["targets"]["p000:c000"].update(word_ids=["p000:l0:w2"]),
                       lambda f: f["annotations"]["pages"]["p000"]["lines"][0]["words"][3]["rect"].__setitem__(0, .561)):
            newer = copy.deepcopy(frozen)
            mutate(newer)
            refreeze(newer)
            candidate = m.score(newer, boxes)
            with self.assertRaisesRegex(ValueError, "incomparable"):
                m.compare(baseline, candidate)
        candidate = copy.deepcopy(baseline)
        candidate["run"]["instrument_config"]["version"] = 999
        candidate["run"]["instrument_sha256"] = m.sha(candidate["run"]["instrument_config"])
        m._seal(candidate)
        with self.assertRaisesRegex(ValueError, "incomparable"):
            m.compare(baseline, candidate)
        candidate = copy.deepcopy(baseline)
        del candidate["run"]["truth_sha256"]
        m._seal(candidate)
        with self.assertRaisesRegex(ValueError, "provenance"):
            m.compare(baseline, candidate)

    def test_compare_refuses_edited_aggregates_boxes_and_report_hash(self):
        frozen, boxes = benchmark()
        baseline = m.score(frozen, boxes)
        candidate = copy.deepcopy(baseline)
        candidate["overall"]["full_target_coverage_rate"] = 0
        m._seal(candidate)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            m.compare(baseline, candidate)
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["box"] = None
        m._seal(candidate)
        with self.assertRaisesRegex(ValueError, "box hash"):
            m.compare(baseline, candidate)
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["full"] = False
        with self.assertRaisesRegex(ValueError, "integrity"):
            m.compare(baseline, candidate)

    def test_cli_score_compare_and_refusal_exit_codes(self):
        frozen, boxes = benchmark()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(td)
            (root/"frozen.json").write_text(json.dumps(frozen))
            (root/"boxes.json").write_text(json.dumps(boxes))
            args = ["score", "--benchmark", str(root/"frozen.json"), "--boxes", str(root/"boxes.json"), "--out", str(root/"score.json")]
            self.assertEqual(m.main(args), 0)
            self.assertEqual(m.main(["compare", "--baseline", str(root/"score.json"), "--candidate", str(root/"score.json"), "--gate"]), 1)
            (root/"boxes.json").write_text('{"boxes": {}}')
            self.assertEqual(m.main(args), 2)


class StatisticalRegression(unittest.TestCase):
    def test_page_clustering_does_not_treat_same_page_as_independent(self):
        stats = m._cluster_statistics({"one-page": [1]*200}, seed=7, resamples=999)
        self.assertEqual(stats["page_cluster_permutation_p"], .5)
        self.assertEqual(stats["page_clusters"], 1)
        self.assertEqual(stats["paired_cases"], 200)

    def test_page_statistics_repeat_and_are_order_invariant(self):
        deltas = {f"p{i}": [1]*i+[0]*(20-i) for i in range(20)}
        a = m._cluster_statistics(deltas, seed=11, resamples=999)
        b = m._cluster_statistics(dict(reversed(list(deltas.items()))), seed=11, resamples=999)
        self.assertEqual(a, b)
        self.assertLess(a["page_cluster_permutation_p"], .05)
        self.assertGreater(a["coverage_delta_ci95"][0], 0)

    def treatments(self, npages=20, ncases=10):
        frozen, good = benchmark(npages, ncases)
        bad = copy.deepcopy(good)
        for i, c in enumerate(frozen["manifest"]["cases"]):
            if i % 2 == 0:
                bad["boxes"][c["case_id"]] = box_at((0,), c["page_id"])
        return frozen, bad, good

    def test_gate_positive_control_is_synthetic_only(self):
        frozen, bad, good = self.treatments()
        result, code = m.compare(m.score(frozen, bad), m.score(frozen, good), True, resamples=999)
        self.assertEqual(code, 0, result)
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["fixed"], 100)
        self.assertEqual(result["jointly_covered"], 100)

    def test_gate_rejects_joint_excess_and_unresolved_regression(self):
        frozen, bad, good = self.treatments()
        baseline = m.score(frozen, bad)
        wider = copy.deepcopy(good)
        for cid in wider["boxes"]:
            wider["boxes"][cid] = {"x0": 0, "y0": 0, "x1": 1, "y1": 1}
        result, code = m.compare(baseline, m.score(frozen, wider), True, resamples=999)
        self.assertEqual(code, 1)
        self.assertTrue(any("excess" in f for f in result["failures"]))
        unresolved = copy.deepcopy(good)
        unresolved["boxes"]["p000:c000"] = None
        result, code = m.compare(baseline, m.score(frozen, unresolved), True, resamples=999)
        self.assertEqual(code, 1)
        self.assertTrue(any("unresolved" in f for f in result["failures"]))

    def test_wrong_to_none_cannot_become_success_or_pass_gate(self):
        frozen, _bad, good = self.treatments()
        wrong = copy.deepcopy(good)
        for cid in wrong["boxes"]:
            wrong["boxes"][cid] = box_at((8,))
        absent = {"boxes": {cid: None for cid in good["boxes"]}}
        a, b = m.score(frozen, wrong), m.score(frozen, absent)
        self.assertEqual(a["overall"]["full_target_coverage_rate"], b["overall"]["full_target_coverage_rate"])
        result, code = m.compare(a, b, True, resamples=999)
        self.assertEqual(code, 1)
        self.assertTrue(any("unresolved" in f for f in result["failures"]))
        self.assertEqual(b["overall"]["word_miss_measured_cases"], 0)

    def test_gate_rejects_small_and_development_population(self):
        frozen, bad, good = self.treatments(19, 10)
        result, code = m.compare(m.score(frozen, bad), m.score(frozen, good), True, resamples=999)
        self.assertEqual(code, 1)
        self.assertTrue(any("200" in f for f in result["failures"]))
        result, code = m.compare(m.score(frozen, bad, "dev"), m.score(frozen, good, "dev"), True, resamples=999)
        self.assertEqual(code, 1)
        self.assertTrue(any("heldout split" in f for f in result["failures"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
