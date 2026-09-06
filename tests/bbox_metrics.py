#!/usr/bin/env python3
"""Score serialized boxes against frozen, independently audited page geometry.

This is instrument v6, separate from the unchanged v5 bbox_score.py. No OCR,
locator, production word rectangles, network calls, or answer-key inference run.
No accuracy claim is possible before independent human audit and freeze.

Examples (activate venv first):
  python tests/bbox_metrics.py score --benchmark frozen.json --boxes boxes.json --out score.json
  python tests/bbox_metrics.py score --benchmark frozen.json --regrade-report old.json --out regrade.json
  python tests/bbox_metrics.py compare --baseline A.json --candidate B.json --gate

Treatment format: {"boxes": {case_id: box_or_null}}. Boxes contain normalized
x0/y0/x1/y1 and/or a nonempty segments list of normalized rectangles. The union
of segments is scored, never its envelope. Null explicitly records unresolved.
Every selected frozen case must be present; extra IDs must belong to the same
frozen benchmark. Historical reports use {"cases": [{"id": ..., "box": ...}]}.
An explicit --id-map maps historical IDs to frozen IDs when identities differ.
Historical regrading requires the source's serialized-box SHA proof, preserves
its exact boxes, and records source, mapping, and output SHA provenance.

Full target coverage means >=90% area of EVERY audited target word. Tight also
requires zero non-target word area; buffered allows at most the two immediate
reading-order neighbors on either side. Fractional excess sums non-target word
area fractions. Distance uses nearest >=90%-covered words in reading order;
when no such word exists it remains null, never a synthetic penalty. Unresolved
includes null boxes and boxes covering no word. Judge-uncertain cases stay in
fixed denominators as failures, separately counted. Insertion cases always use
a separate denominator; their audited RTL gap midpoint (both endpoints at a
line wrap) must be contained in the box union.

Compare uses seeded paired page-cluster sign permutation and page-cluster
bootstrap intervals. Promotion requires >=200 evaluable ordinary held-out
cases, >=2 books, >=10 pages, significant coverage improvement, non-increasing
excess among jointly fully covered cases, unresolved rate and wrong-line rate.
A gate result is benchmark evidence, not automatic deployment authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys

VERSION = 6
THRESHOLD = .90
EPS = 1e-12
SCHEMA = "farsi2epub.bbox-metrics/v6"
INSTRUMENT = {
    "printed_insertions": "word-coverage-between-fixed-anchors",
    "version": VERSION,
    "target_word_area_threshold": THRESHOLD,
    "fraction_epsilon": EPS,
    "rectangles": "normalized-page-coordinates-exact-union-v1",
    "distance": "nearest-90pct-covered-word-in-reading-order-v1",
    "buffer": "two-immediate-neighbors-before-and-after-target-v1",
    "wrong_line": "covered-words-exist-but-none-on-target-lines-v1",
    "unresolved": "null-box-or-no-90pct-covered-word-v1",
    "insertion": "rtl-gap-midpoint-both-wrap-endpoints-v1",
    "gate": "page-sign-permutation-one-sided-bootstrap-200-2books-10pages-v1",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {x}")))


def rectangle(value):
    if isinstance(value, dict):
        if any(k not in value for k in ("x0", "y0", "x1", "y1")):
            raise ValueError("rectangle lacks coordinates")
        value = [value[k] for k in ("x0", "y0", "x1", "y1")]
    if (not isinstance(value, (list, tuple)) or len(value) != 4
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)
        or not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1)):
        raise ValueError("invalid normalized rectangle")
    return list(value)


def _segments(box):
    if box is None:
        return []
    if not isinstance(box, dict):
        raise ValueError("box must be a rectangle object or null")
    # Catch corrupt envelope coordinates even when only segments are scored.
    if any(k in box for k in ("x0", "y0", "x1", "y1")):
        rectangle(box)
    if box.get("segments") is not None:
        if not isinstance(box["segments"], list) or not box["segments"]:
            raise ValueError("segments must be a nonempty list; use null for unresolved")
        return [rectangle(r) for r in box["segments"]]
    return [rectangle(box)]


def area(rect):
    return (rect[2]-rect[0])*(rect[3]-rect[1])


def coverage(word, boxes):
    """Exact union of clipped rectangles, including overlapping segments."""
    clips = []
    for b in boxes:
        r = [max(word[0], b[0]), max(word[1], b[1]), min(word[2], b[2]), min(word[3], b[3])]
        if r[0] < r[2] and r[1] < r[3]:
            clips.append(r)
    xs = sorted({x for r in clips for x in (r[0], r[2])})
    total = 0.
    for left, right in zip(xs, xs[1:]):
        runs = sorted((r[1], r[3]) for r in clips if r[0] < right and r[2] > left)
        length, end = 0., -math.inf
        for a, b in runs:
            length += max(0., b-max(a, end))
            end = max(end, b)
        total += (right-left)*length
    return min(1., max(0., total/area(word)))


def _insertion_points(words, ids, anchor):
    if not isinstance(anchor, dict):
        raise ValueError("insertion requires independently audited anchors")
    before, after = anchor.get("before_word_id"), anchor.get("after_word_id")
    if before is None and after is None:
        raise ValueError("insertion requires an anchor")
    if any(w is not None and w not in ids for w in (before, after)):
        raise ValueError("unknown insertion anchor")
    a, b = ids.index(before) if before is not None else None, ids.index(after) if after is not None else None
    if a is not None and b is not None:
        if b != a+1:
            raise ValueError("insertion anchors must be adjacent in reading order")
        ra, rb = words[a][0]["rect"], words[b][0]["rect"]
        if words[a][1] == words[b][1]:
            return [((ra[0]+rb[2])/2, (ra[1]+ra[3]+rb[1]+rb[3])/4)]
        return [(ra[0], (ra[1]+ra[3])/2), (rb[2], (rb[1]+rb[3])/2)]
    if a is not None and a != len(ids)-1 or b is not None and b != 0:
        raise ValueError("single insertion anchor is allowed only at a page boundary")
    r = words[a if a is not None else b][0]["rect"]
    return [(r[0] if a is not None else r[2], (r[1]+r[3])/2)]


def grade(page, target, box, kind=None):
    """Grade already-audited truth; score() first validates the frozen wrapper."""
    rects = _segments(box)  # Invalid geometry never disappears behind uncertainty.
    insertion_case = kind == "insertion" if kind is not None else target.get("insertion_anchor") is not None
    insertion = insertion_case and not target.get("word_ids")
    result = {"judge_uncertain": False, "insertion": insertion, "full": False,
              "tight": False, "buffered": False, "unresolved": box is None,
              "unresolved_reason": "no_box" if box is None else None,
              "wrong_line": False, "excess": None, "word_miss": None,
              "word_miss_distances": None, "target_coverage": [], "covered_word_ids": []}
    if page["status"] != "audited" or target["status"] != "audited":
        result.update(judge_uncertain=True, judge_uncertain_reason="independent_truth_unreadable")
        if insertion:
            result["boundary_hit"] = False
        return result
    words = [(w, line["line_id"]) for line in page["lines"] for w in line["words"]]
    ids = [w["word_id"] for w, _ in words]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("audited page needs unique words")
    fractions = [coverage(rectangle(w["rect"]), rects) for w, _ in words]
    if insertion:
        if target.get("word_ids"):
            raise ValueError("insertion word_ids must be empty; anchors have a separate denominator")
        points = _insertion_points(words, ids, target.get("insertion_anchor"))
        hit = all(any(r[0]-EPS <= x <= r[2]+EPS and r[1]-EPS <= y <= r[3]+EPS for r in rects)
                  for x, y in points)
        result.update(boundary_hit=hit, boundary_points=points,
                      anchor_word_coverage={wid: fractions[ids.index(wid)]
                                            for wid in target["insertion_anchor"].values() if wid is not None})
        return result
    word_ids = target.get("word_ids")
    if not isinstance(word_ids, list) or not word_ids or len(word_ids) != len(set(word_ids)) or any(w not in ids for w in word_ids):
        raise ValueError("ordinary target requires unique audited word IDs")
    if insertion_case:
        anchor = target.get("insertion_anchor") or {}
        before, after = anchor.get("before_word_id"), anchor.get("after_word_id")
        if (not before and not after) or any(w is not None and w not in ids for w in (before, after)):
            raise ValueError("printed insertion requires fixed anchors")
        start = ids.index(before)+1 if before else 0
        stop = ids.index(after) if after else len(ids)
        if word_ids != ids[start:stop] or start >= stop:
            raise ValueError("printed insertion words must lie exactly between anchors")
    required = {ids.index(wid) for wid in word_ids}
    covered = {i for i, f in enumerate(fractions) if f >= THRESHOLD-EPS}
    touched = {i for i, f in enumerate(fractions) if f > EPS}
    extras = touched-required
    lo, hi = min(required), max(required)
    allowed = set(range(max(0, lo-2), lo)) | set(range(hi+1, min(len(words), hi+3)))
    full = required <= covered
    distances = [0 if i in covered else min(abs(i-j) for j in covered) for i in sorted(required)] if covered else None
    target_lines = {words[i][1] for i in required}
    result.update(full=full, tight=full and not extras, buffered=full and extras <= allowed,
                  excess=sum(f for i, f in enumerate(fractions) if i not in required),
                  off_target_line_word_coverage=sum(f for i, f in enumerate(fractions) if words[i][1] not in target_lines),
                  target_coverage=[fractions[i] for i in sorted(required)],
                  word_miss=sum(distances) if distances is not None else None,
                  word_miss_distances=distances, unresolved=not bool(covered),
                  unresolved_reason=("no_box" if box is None else "no_covered_words") if not covered else None,
                  wrong_line=bool(covered) and not any(words[i][1] in target_lines for i in covered),
                  covered_word_ids=[ids[i] for i in sorted(covered)])
    return result


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def metrics(rows):
    regular = [r for r in rows if not r["insertion"]]
    known = [r for r in regular if not r["judge_uncertain"]]
    insertions = [r for r in rows if r["insertion"]]
    d = len(regular)
    def rate(key):
        return sum(bool(r.get(key)) for r in regular)/d if d else None
    misses = [r["word_miss"] for r in known if r["word_miss"] is not None]
    excess = [r["excess"] for r in known if r["excess"] is not None]
    return {"cases": len(rows), "denominator": d, "evaluable_cases": len(known),
            "judge_uncertain": sum(r["judge_uncertain"] for r in rows),
            "ordinary_judge_uncertain": d-len(known),
            "full_target_coverage_rate": rate("full"), "tight_success_rate": rate("tight"),
            "buffered_success_rate": rate("buffered"), "unresolved_rate": rate("unresolved"),
            "unresolved_reasons": dict(sorted(Counter(r["unresolved_reason"] for r in regular if r["unresolved"]).items())),
            "wrong_line_rate": rate("wrong_line"), "excess_mean": statistics.mean(excess) if excess else None,
            "excess_measured_cases": len(excess),
            "word_miss_mean": statistics.mean(misses) if misses else None,
            "word_miss_median": statistics.median(misses) if misses else None,
            "word_miss_p90": _percentile(misses, .90), "word_miss_measured_cases": len(misses),
            "word_miss_unmeasured_cases": d-len(misses),
            "word_miss_histogram": dict(sorted(Counter(str(v) for v in misses).items())),
            "insertion_cases": len(insertions),
            "insertion_evaluable_cases": sum(not r["judge_uncertain"] for r in insertions),
            "insertion_boundary_hits": sum(bool(r.get("boundary_hit")) for r in insertions),
            "insertion_boundary_hit_rate": sum(bool(r.get("boundary_hit")) for r in insertions)/len(insertions) if insertions else None,
            "insertion_unresolved": sum(r["unresolved"] for r in insertions)}


def _validate_benchmark(data):
    import bbox_benchmark as bb
    errors = bb.validate_frozen(data)
    if errors:
        raise ValueError("invalid audited benchmark: " + "; ".join(errors[:5]))


def load_benchmark(path):
    data = _read_json(path)
    _validate_benchmark(data)
    return data


def _historical_box_sha1(rows):
    payload = [[r.get("case_id", r.get("id")), r["box"]]
               for r in sorted(rows, key=lambda r: r.get("case_id", r.get("id")))]
    return hashlib.sha1(canonical(payload)).hexdigest()


def _load_boxes(boxes_input, regrade=False, id_map=None):
    if not isinstance(boxes_input, dict):
        raise ValueError("treatment must be a JSON object")
    provenance = {"input_sha256": sha(boxes_input)}
    if "boxes" in boxes_input:
        if regrade:
            raise ValueError("--regrade-report requires serialized report cases and a box SHA proof")
        boxes = boxes_input["boxes"]
        if not isinstance(boxes, dict):
            raise ValueError("boxes must map case IDs to explicit rectangles or null")
    else:
        rows = boxes_input.get("cases")
        if not isinstance(rows, list) or not rows:
            raise ValueError("box report requires a nonempty cases list")
        boxes = {}
        for row in rows:
            cid = row.get("case_id", row.get("id"))
            if "box" not in row or not isinstance(cid, str) or not cid or cid in boxes:
                raise ValueError("box report has missing/duplicate identity or an absent serialized box")
            if "case_id" in row and "id" in row and row["case_id"] != row["id"]:
                raise ValueError("box report has conflicting case_id and id")
            boxes[cid] = row["box"]
        run = boxes_input.get("run", {})
        sha1_proof, sha256_proof = run.get("box_sha1"), run.get("box_sha256")
        if sha1_proof is not None and sha1_proof != _historical_box_sha1(rows):
            raise ValueError("source serialized box_sha1 mismatch")
        if sha256_proof is not None and sha256_proof != sha(boxes):
            raise ValueError("source serialized box_sha256 mismatch")
        if regrade and not (sha1_proof or sha256_proof):
            raise ValueError("regrading requires source box_sha1 or box_sha256 proof")
        if boxes_input.get("schema_version") == SCHEMA:
            _validate_report(boxes_input)
        provenance.update(source_box_sha1=sha1_proof, source_box_sha256=sha256_proof)
    if any(not isinstance(cid, str) or not cid for cid in boxes):
        raise ValueError("box IDs must be nonempty strings")
    for box in boxes.values():
        _segments(box)
    original = boxes
    if id_map is not None:
        if not isinstance(id_map, dict) or not id_map or any(not isinstance(k, str) or not isinstance(v, str) for k, v in id_map.items()):
            raise ValueError("id map must map historical ID strings to frozen ID strings")
        if any(cid not in boxes for cid in id_map) or len(set(id_map.values())) != len(id_map):
            raise ValueError("id map is not one-to-one or refers to absent source IDs")
        boxes = {new: original[old] for old, new in id_map.items()}
        provenance.update(id_map_sha256=sha(id_map), id_map=dict(sorted(id_map.items())),
                          unmapped_source_cases=len(original)-len(id_map))
    provenance.update(serialized_input_box_sha256=sha(original),
                      mapped_input_box_sha256=sha(boxes), regrade=bool(regrade))
    return boxes, provenance


def _seal(report):
    report["report_sha256"] = sha({k: v for k, v in report.items() if k != "report_sha256"})
    return report


def score(benchmark, boxes_input, split="heldout", *, regrade=False, id_map=None):
    _validate_benchmark(benchmark)
    if split not in {"dev", "heldout", "all"}:
        raise ValueError("unknown split")
    manifest, annotations = benchmark["manifest"], benchmark["annotations"]
    boxes, source = _load_boxes(boxes_input, regrade, id_map)
    pages = {p["page_id"]: p for p in manifest["pages"]}
    population = sorted([c for c in manifest["cases"] if split == "all" or c["split"] == split], key=lambda c: c["case_id"])
    if not population:
        raise ValueError("selected split has no frozen cases")
    all_ids = {c["case_id"] for c in manifest["cases"]}
    unknown = set(boxes)-all_ids
    if unknown:
        raise ValueError("box report contains IDs outside frozen population; use an explicit --id-map for historical conversion")
    missing = [c["case_id"] for c in population if c["case_id"] not in boxes]
    if missing:
        raise ValueError(f"box report omits {len(missing)} cases; represent unresolved explicitly as null")
    if boxes_input.get("schema_version") == SCHEMA:
        previous = boxes_input["run"]
        if previous["benchmark_sha256"] != benchmark["benchmark_sha256"] or previous["split"] != split:
            raise ValueError("regrade cannot change the benchmark, truth, targets or split")
    rows = []
    for c in population:
        p = pages[c["page_id"]]
        result = grade(annotations["pages"][c["page_id"]], annotations["targets"][c["case_id"]], boxes[c["case_id"]], c["kind"])
        rows.append({"case_id": c["case_id"], "page_id": c["page_id"], "slug": p["slug"],
                     "kind": c["kind"], "split": c["split"], "box": boxes[c["case_id"]], **result})
    selected_boxes = {r["case_id"]: r["box"] for r in rows}
    # Serialized values, including union-envelope metadata, remain byte-canonical
    # equivalents of their input. No production locator is constructed here.
    if sha(selected_boxes) != sha({c["case_id"]: boxes[c["case_id"]] for c in population}):
        raise ValueError("serialized geometry changed during regrade")
    provenance = {"instrument": VERSION, "instrument_config": INSTRUMENT,
                  "instrument_sha256": sha(INSTRUMENT), "threshold": THRESHOLD,
                  "benchmark_sha256": benchmark["benchmark_sha256"], "split": split,
                  "frozen_integrity": benchmark["integrity"],
                  "population_sha256": sha([(r["case_id"], r["page_id"], r["slug"], r["kind"], r["split"]) for r in rows]),
                  "images_sha256": benchmark["integrity"]["image_set_sha256"],
                  "truth_sha256": benchmark["integrity"]["truth_sha256"],
                  "targets_sha256": benchmark["integrity"]["targets_sha256"],
                  "box_sha256": sha(selected_boxes), "identical_serialized_boxes": True,
                  "source": source}
    return _seal({"schema_version": SCHEMA, "run": provenance, "overall": metrics(rows), "cases": rows,
                  "by_book": {slug: metrics([r for r in rows if r["slug"] == slug]) for slug in sorted({r["slug"] for r in rows})}})


def _validate_report(report):
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA:
        raise ValueError("not a v6 report; regrade serialized boxes against audited truth first")
    if report.get("report_sha256") != sha({k: v for k, v in report.items() if k != "report_sha256"}):
        raise ValueError("report integrity mismatch")
    run, rows = report["run"], report["cases"]
    required = ("instrument", "instrument_config", "instrument_sha256", "threshold", "benchmark_sha256", "split",
                "population_sha256", "images_sha256", "truth_sha256", "targets_sha256", "box_sha256", "frozen_integrity")
    if any(key not in run or run[key] is None for key in required):
        raise ValueError("report lacks comparable provenance")
    if sha(run["instrument_config"]) != run["instrument_sha256"]:
        raise ValueError("instrument configuration hash mismatch")
    integrity = run["frozen_integrity"]
    if not isinstance(integrity, dict) or integrity.get("validated") is not True or integrity.get("complete") is not True:
        raise ValueError("report lacks complete independently audited truth")
    if any(run[k] != integrity[j] for k, j in (("images_sha256", "image_set_sha256"), ("truth_sha256", "truth_sha256"), ("targets_sha256", "targets_sha256"))):
        raise ValueError("report truth/image/target integrity mismatch")
    if not isinstance(rows, list) or not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("report cases must be nonempty and unique")
    if rows != sorted(rows, key=lambda r: r["case_id"]):
        raise ValueError("report population is not canonical")
    if sha([(r["case_id"], r["page_id"], r["slug"], r["kind"], r["split"]) for r in rows]) != run["population_sha256"]:
        raise ValueError("report population hash mismatch")
    page_meta = {}
    for row in rows:
        if row["split"] not in {"dev", "heldout"} or run["split"] != "all" and row["split"] != run["split"]:
            raise ValueError("report case split mismatch")
        meta = (row["slug"], row["split"])
        if row["page_id"] in page_meta and page_meta[row["page_id"]] != meta:
            raise ValueError("report splits a page across books or folds")
        page_meta[row["page_id"]] = meta
        for key in ("insertion", "full", "tight", "buffered", "unresolved", "wrong_line", "judge_uncertain"):
            if not isinstance(row.get(key), bool):
                raise ValueError(f"report case lacks boolean {key}")
        if row["insertion"] != (row["kind"] == "insertion"):
            raise ValueError("insertion denominator differs from frozen kind")
        if row["full"] and (row["insertion"] or row["judge_uncertain"] or row["unresolved"] or row["wrong_line"]):
            raise ValueError("unscorable, unresolved or wrong-line case cannot be full coverage")
        _segments(row["box"])
    if sha({r["case_id"]: r["box"] for r in rows}) != run["box_sha256"]:
        raise ValueError("report serialized box hash mismatch")
    if report["overall"] != metrics(rows):
        raise ValueError("aggregate metrics disagree with fixed cases")
    expected_books = {slug: metrics([r for r in rows if r["slug"] == slug]) for slug in sorted({r["slug"] for r in rows})}
    if report["by_book"] != expected_books:
        raise ValueError("per-book metrics disagree with fixed cases")


def _cluster_statistics(page_deltas, *, seed, resamples):
    """Paired sign flips and bootstrap resample entire pages, never corrections."""
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 999:
        raise ValueError("use an integer seed and at least 999 resamples")
    clusters = [(sum(page_deltas[k]), len(page_deltas[k])) for k in sorted(page_deltas)]
    if not clusters:
        return {"page_clusters": 0, "paired_cases": 0, "coverage_delta": None,
                "coverage_fixed_minus_broken": 0, "page_cluster_permutation_p": None,
                "coverage_delta_ci95": [None, None], "permutation_method": "unavailable"}
    observed, count = sum(s for s, n in clusters), sum(n for s, n in clusters)
    nonzero = [s for s, n in clusters if s]
    if len(nonzero) <= 16:
        sums = (sum(v*sign for v, sign in zip(nonzero, signs)) for signs in itertools.product((-1, 1), repeat=len(nonzero)))
        p = sum(v >= observed for v in sums)/(2**len(nonzero))
        method = "exact-paired-page-sign-permutation-one-sided"
    else:
        rng = random.Random(seed)
        p = (1+sum(sum(v*rng.choice((-1, 1)) for v in nonzero) >= observed for _ in range(resamples)))/(resamples+1)
        method = "monte-carlo-paired-page-sign-permutation-one-sided"
    rng = random.Random(seed+1)
    draws = []
    for _ in range(resamples):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(sum(s for s, n in chosen)/sum(n for s, n in chosen))
    return {"page_clusters": len(clusters), "nonzero_page_clusters": len(nonzero), "paired_cases": count,
            "coverage_fixed_minus_broken": observed, "coverage_delta": observed/count,
            "page_cluster_permutation_p": p, "coverage_delta_ci95": [_percentile(draws, .025), _percentile(draws, .975)],
            "permutation_method": method, "seed": seed, "resamples": resamples}


def compare(base, candidate, gate=False, *, seed=7, resamples=9999):
    _validate_report(base)
    _validate_report(candidate)
    for key in ("instrument", "instrument_sha256", "threshold", "benchmark_sha256", "split", "population_sha256", "images_sha256", "truth_sha256", "targets_sha256", "frozen_integrity"):
        if base["run"][key] != candidate["run"][key]:
            raise ValueError(f"incomparable {key}")
    a = {r["case_id"]: r for r in base["cases"]}
    b = {r["case_id"]: r for r in candidate["cases"]}
    if set(a) != set(b):
        raise ValueError("case population differs")
    page_deltas = {}
    fixed = broken = 0
    for cid in a:
        if any(a[cid][k] != b[cid][k] for k in ("page_id", "slug", "kind", "split", "insertion", "judge_uncertain")):
            raise ValueError("case identity or independent audit denominator drift")
        if a[cid]["insertion"]:
            continue
        delta = int(b[cid]["full"])-int(a[cid]["full"])
        fixed += delta == 1
        broken += delta == -1
        page_deltas.setdefault(a[cid]["page_id"], []).append(delta)
    stats = _cluster_statistics(page_deltas, seed=seed, resamples=resamples)
    joint = [cid for cid in a if a[cid]["full"] and b[cid]["full"]]
    excess_delta = statistics.mean(b[cid]["excess"]-a[cid]["excess"] for cid in joint) if joint else None
    failures = []
    if stats["coverage_fixed_minus_broken"] <= 0 or stats["page_cluster_permutation_p"] is None or stats["page_cluster_permutation_p"] >= .05:
        failures.append("full coverage improvement is absent or not significant")
    if stats["page_clusters"] < 10:
        failures.append("fewer than 10 independent page clusters")
    if excess_delta is None or excess_delta > EPS:
        failures.append("jointly-covered excess did not remain non-increasing")
    for name in ("unresolved_rate", "wrong_line_rate"):
        av, bv = base["overall"][name], candidate["overall"][name]
        if av is None or bv is None or bv > av+EPS:
            failures.append(f"{name} regressed or unavailable")
    measurable = [r for r in b.values() if not r["judge_uncertain"] and not r["insertion"]]
    if len(measurable) < 200 or len({r["slug"] for r in measurable}) < 2:
        failures.append("need 200 evaluable held-out ordinary corrections across at least two books")
    if candidate["run"]["split"] != "heldout":
        failures.append("promotion requires heldout split")
    if candidate["run"]["instrument_sha256"] != sha(INSTRUMENT):
        failures.append("promotion requires the current registered instrument")
    result = {**stats, "fixed": fixed, "broken": broken, "jointly_covered": len(joint),
              "joint_excess_delta": excess_delta, "gate_pass": not failures, "failures": failures,
              "insertion_cases": candidate["overall"]["insertion_cases"],
              "judge_uncertain_cases": candidate["overall"]["judge_uncertain"],
              "interpretation": "Paired benchmark evidence only; audit quality and treatment independence require review."}
    return result, (1 if gate and failures else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="grade an explicit fixed box population against audited frozen truth")
    s.add_argument("--benchmark", required=True)
    inputs = s.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--boxes", help="serialized boxes; every selected case is required")
    inputs.add_argument("--regrade-report", help="instrument-only regrade; requires source serialized-box SHA proof")
    s.add_argument("--id-map", help="JSON object mapping historical case IDs to frozen case IDs")
    s.add_argument("--split", choices=["dev", "heldout", "all"], default="heldout")
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare identical frozen truth/population using paired page clusters")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--gate", action="store_true", help="exit 1 when promotion evidence fails")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--resamples", type=int, default=9999)
    c.add_argument("--out", help="optional comparison JSON output")
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            path = Path(args.regrade_report or args.boxes)
            report = score(load_benchmark(args.benchmark), _read_json(path), args.split,
                           regrade=bool(args.regrade_report), id_map=_read_json(args.id_map) if args.id_map else None)
            report["run"]["source"]["source_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            _seal(report)
            code, display = 0, report["overall"]
        else:
            report, code = compare(_read_json(args.baseline), _read_json(args.candidate), args.gate,
                                   seed=args.seed, resamples=args.resamples)
            display = report
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(display, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
