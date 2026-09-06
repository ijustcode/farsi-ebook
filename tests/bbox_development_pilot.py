#!/usr/bin/env python3
"""Replay the independently audited scan-page 66 development diagnostic.

This intentionally scores one development page only.  The fixture contains
the frozen word rectangles, audited target identities, and exact serialized
boxes from both saved offline captures.  It must never be imported by
production code or described as held-out/promotion evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import bbox_metrics


SCHEMA = "farsi2epub.bbox-development-diagnostic/v1"
FIXTURE_SCHEMA = "farsi2epub.bbox-development-pilot/v1"
DEFAULT_FIXTURE = Path(__file__).parent / "data" / "bbox_scan66_development_pilot.json"


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fixture(fixture: dict) -> None:
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("not the scan66 development-pilot fixture")
    source, audit = fixture.get("source", {}), fixture.get("audit", {})
    if source.get("split") != "dev" or source.get("page_id") != "p_b0f70bf518eea8b9a7c1":
        raise ValueError("pilot must remain the single declared development page")
    if audit.get("decision") != "approved" or not audit.get("auditor") or not audit.get("audited_at"):
        raise ValueError("fixture lacks independent target-audit provenance")
    if not isinstance(fixture.get("page", {}).get("lines"), list):
        raise ValueError("fixture lacks audited page words")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("pilot must contain exactly ten audited targets")
    ids = [c.get("case_id") for c in cases]
    if len(set(ids)) != len(ids) or any(c.get("target", {}).get("status") != "audited" for c in cases):
        raise ValueError("pilot targets must be unique and audited")
    integrity = fixture.get("integrity", {})
    expected = {
        "page_truth_sha256": sha(fixture["page"]),
        "targets_sha256": sha([{k: c[k] for k in ("case_id", "kind", "issue_type", "target")}
                                for c in cases]),
        "treatments_sha256": sha(fixture.get("treatments")),
        "case_population_sha256": sha([(cid, source["page_id"], "dev") for cid in ids]),
    }
    if integrity.get("complete") is not True or any(integrity.get(k) != v for k, v in expected.items()):
        raise ValueError("pilot fixture integrity mismatch")
    treatments = fixture.get("treatments", {})
    if set(treatments) != {"historical_full_query", "consolidated_offline"}:
        raise ValueError("pilot requires the two declared saved treatments")
    for name, treatment in treatments.items():
        if set(treatment.get("boxes", {})) != set(ids) or set(treatment.get("states", {})) != set(ids):
            raise ValueError(f"{name} does not preserve the identical target population")
        for box in treatment["boxes"].values():
            bbox_metrics._segments(box)


def verify_sources(fixture: dict, root: Path) -> None:
    checks = {
        fixture["source"]["image_path"]: fixture["source"]["image_sha256"],
        "out/consolidated_bbox/benchmark/manifest.json": fixture["source"]["manifest_file_sha256"],
        "out/consolidated_bbox/annotation_workflow/reviewed/scan66.json": fixture["source"]["reviewed_geometry_file_sha256"],
        "out/consolidated_bbox/annotation_workflow/scan66.targets.draft.json": fixture["source"]["target_draft_file_sha256"],
        "out/consolidated_bbox/annotation_workflow/reviewed/scan66.targets.json": fixture["source"]["target_audit_file_sha256"],
    }
    checks.update({t["source_file"]: t["source_file_sha256"]
                   for t in fixture["treatments"].values()})
    for rel, expected in checks.items():
        path = root / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch or absent: {rel}")


def failure_category(row: dict) -> str | None:
    if row["full"]:
        return None
    if row["box"] is None:
        return "unresolved_no_saved_box"
    if row["wrong_line"]:
        return "wrong_line"
    coverage = row["target_coverage"]
    if coverage and max(coverage) == 0 and row["covered_word_ids"]:
        return "adjacent_word_only"
    if any(0 < value < bbox_metrics.THRESHOLD for value in coverage):
        return "partial_target_boundary"
    if row["covered_word_ids"]:
        return "target_words_missed"
    return "no_words_covered"


def score_fixture(fixture: dict) -> dict:
    validate_fixture(fixture)
    page = fixture["page"]
    reports = {}
    for name, treatment in fixture["treatments"].items():
        rows = []
        for case in fixture["cases"]:
            cid = case["case_id"]
            box = treatment["boxes"][cid]
            result = bbox_metrics.grade(page, case["target"], box, case["kind"])
            row = {"case_id": cid, "captured_state": treatment["states"][cid],
                   "box": box, **result}
            row["failure_category"] = failure_category(row)
            rows.append(row)
        reports[name] = {
            "label": treatment["label"],
            "source_file_sha256": treatment["source_file_sha256"],
            "metrics": bbox_metrics.metrics(rows),
            "failure_categories": dict(sorted(Counter(
                r["failure_category"] for r in rows if r["failure_category"]).items())),
            "cases": rows,
        }
    old = {r["case_id"]: r for r in reports["historical_full_query"]["cases"]}
    new = {r["case_id"]: r for r in reports["consolidated_offline"]["cases"]}
    fixed = sum(not old[c]["full"] and new[c]["full"] for c in old)
    broken = sum(old[c]["full"] and not new[c]["full"] for c in old)
    report = {
        "schema_version": SCHEMA,
        "interpretation": "One-page development diagnostic only; no statistical inference or promotion claim.",
        "instrument": bbox_metrics.VERSION,
        "instrument_sha256": sha(bbox_metrics.INSTRUMENT),
        "fixture_integrity": fixture["integrity"],
        "source": fixture["source"],
        "audit": fixture["audit"],
        "treatments": reports,
        "paired_diagnostic": {
            "cases": len(old), "page_clusters": 1, "fixed": fixed, "broken": broken,
            "full_coverage_delta": (fixed-broken)/len(old),
            "statistical_inference": "unavailable_single_development_page",
        },
        "limitations": [
            "The consolidated capture was offline and had no saved scan evidence for this page; its null boxes diagnose availability, not placement geometry.",
            "The historical control changes the full-query contract and is not an exact old-production snapshot.",
            "One development page cannot select a detector, establish general accuracy, or satisfy the held-out promotion gate.",
        ],
        "cost_effect": "Replay is fully offline and costs $0. No paid evidence is acquired or added to runtime caches.",
    }
    report["report_sha256"] = sha(report)
    return report


def summary(report: dict) -> dict:
    return {
        "interpretation": report["interpretation"],
        "historical_full_query": report["treatments"]["historical_full_query"]["metrics"],
        "historical_failures": report["treatments"]["historical_full_query"]["failure_categories"],
        "consolidated_offline": report["treatments"]["consolidated_offline"]["metrics"],
        "consolidated_failures": report["treatments"]["consolidated_offline"]["failure_categories"],
        "paired_diagnostic": report["paired_diagnostic"],
        "cost_effect": report["cost_effect"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-sources", action="store_true",
                        help="also require the ignored source artifacts and check their byte hashes")
    args = parser.parse_args(argv)
    try:
        fixture = _read(args.fixture)
        validate_fixture(fixture)
        if args.verify_sources:
            verify_sources(fixture, Path(__file__).resolve().parents[1])
        report = score_fixture(fixture)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        print(json.dumps(summary(report), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
