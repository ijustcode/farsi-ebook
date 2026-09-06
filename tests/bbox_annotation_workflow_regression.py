#!/usr/bin/env python3
"""Offline regressions for the blinded bbox annotation workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bbox_annotation_workflow as workflow
import bbox_benchmark as benchmark


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(ch: str) -> str:
    return ch * 64


def page(page_id: str, number: int, split: str, case_ids: list[str]) -> dict:
    return {
        "page_id": page_id, "slug": "synthetic", "page": number, "split": split,
        "markdown_sha256": sha("a"), "case_ids": case_ids,
        "image": {
            "path": f"pages/{page_id}.png", "sha256": sha("b" if number == 1 else "c"),
            "width_px": 800, "height_px": 1200,
            "render": {"render_id": benchmark.RENDER_ID, "engine": "synthetic",
                       "engine_version": "1", "long_edge_px": benchmark.LONG_EDGE_STD,
                       "alpha": False, "zoom": 2.0},
            "source": {"pdf_sha256": sha("d"), "page_width_pt": 400.0,
                       "page_height_pt": 600.0},
        },
    }


def case(case_id: str, page_id: str, split: str) -> dict:
    return {
        "case_id": case_id, "page_id": page_id, "split": split,
        "kind": "replacement", "issue_type": "synthetic",
        "target_annotation_id": "target:" + case_id,
        "hunk": {"old": "ساختگی", "new": "آزمایش", "markdown_span": [0, 6],
                 "candidate_spans": [[0, 6]], "ctx_before": "", "ctx_after": "",
                 "occurrence": 1, "anchor_unique": True},
    }


def fixture() -> tuple[dict, dict, str]:
    manifest = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "split_policy": {"id": benchmark.SPLIT_ID, "unit": "page",
                         "heldout_min_cases": 1, "heldout_min_books": 1},
        "render_policy": {"id": benchmark.RENDER_ID,
                          "long_edge_px": benchmark.LONG_EDGE_STD},
        "books": [{"slug": "synthetic", "pdf_sha256": sha("d"),
                   "page_count": 2, "case_count": 2}],
        "pages": [page("p-held", 1, "heldout", ["c-held"]),
                  page("p-dev", 2, "dev", ["c-dev"])],
        "cases": [case("c-held", "p-held", "heldout"),
                  case("c-dev", "p-dev", "dev")],
    }
    raw = benchmark._json_file_bytes(manifest)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    annotations = benchmark._annotation_template(manifest, manifest_sha)
    return manifest, annotations, manifest_sha


def lines(page_id: str) -> list[dict]:
    return [{
        "line_id": f"{page_id}:l0", "order": 0, "rect": [0.1, 0.1, 0.9, 0.2],
        "words": [
            {"word_id": f"{page_id}:l0:w0", "order": 0, "text": "راست",
             "rect": [0.7, 0.11, 0.88, 0.19]},
            {"word_id": f"{page_id}:l0:w1", "order": 1, "text": "چپ",
             "rect": [0.5, 0.11, 0.68, 0.19]},
        ],
    }]


def reviewed_for(page_def: dict, *, ambiguity: bool = False) -> dict:
    packet = workflow.packet_for_page(page_def)
    draft = workflow.initial_draft(packet, "Annotator One")
    draft["lines"] = lines(packet["page_id"])
    if ambiguity:
        draft["ambiguities"] = [{"ambiguity_id": "a1", "description": "touching words"}]
    draft = workflow.seal_draft(packet, draft)
    review = workflow.initial_review(packet, draft, "Auditor Two")
    review.update({"decision": "approved", "geometry_checked": True,
                   "ambiguities_checked": True, "discrepancies_checked": True,
                   "audited_at": "2026-09-05T12:00:00Z"})
    if ambiguity:
        review["ambiguity_resolutions"]["a1"] = {
            "outcome": "resolved", "note": "confirmed as two separate printed words"
        }
    return workflow.seal_review(packet, draft, review)


def test_queue_is_complete_deterministic_and_blinded() -> None:
    manifest, _annotations, manifest_sha = fixture()
    first = workflow.build_queue(manifest, manifest_sha)
    second = workflow.build_queue(copy.deepcopy(manifest), manifest_sha)
    check(first == second, "queue is not deterministic")
    check([t["page_id"] for t in first["tasks"]] == ["p-dev", "p-held"],
          "queue was sampled or is not canonical")
    check(all(set(task) == workflow.PACKET_KEYS for task in first["tasks"]),
          "blinded task leaked manifest/correction/locator fields")
    serialized = json.dumps(first["tasks"])
    for leaked in ("heldout", "synthetic", "c-held", "hunk", "locator", "bbox", "split"):
        check(leaked not in serialized, f"queue leaked {leaked}")
    check(not workflow.validate_queue(first, manifest, manifest_sha), "valid queue rejected")


def test_hash_identity_and_explicit_resolution_guards() -> None:
    manifest, _annotations, _manifest_sha = fixture()
    payload = reviewed_for(manifest["pages"][0], ambiguity=True)
    packet, draft, review = payload["packet"], payload["draft"], payload["review"]
    check(not workflow.validate_reviewed_page(packet, draft, review), "valid proof rejected")
    bad = copy.deepcopy(review)
    bad["auditor"] = " annotator   one "
    bad["review_sha256"] = workflow._sha(workflow._unsigned(bad, "review_sha256"))
    check(any("distinct" in e for e in workflow.validate_reviewed_page(packet, draft, bad)),
          "same-person audit accepted")
    bad = copy.deepcopy(review)
    bad["draft_sha256"] = sha("f")
    bad["review_sha256"] = workflow._sha(workflow._unsigned(bad, "review_sha256"))
    check(any("exact draft" in e for e in workflow.validate_reviewed_page(packet, draft, bad)),
          "audit detached from exact draft hash")
    bad = copy.deepcopy(review)
    bad["image_sha256"] = sha("f")
    bad["review_sha256"] = workflow._sha(workflow._unsigned(bad, "review_sha256"))
    check(any("image_sha256" in e for e in workflow.validate_reviewed_page(packet, draft, bad)),
          "audit detached from image hash")
    bad = copy.deepcopy(review)
    bad["ambiguity_resolutions"]["a1"] = {"outcome": "pending", "note": ""}
    bad["review_sha256"] = workflow._sha(workflow._unsigned(bad, "review_sha256"))
    check(any("explicit resolved" in e for e in workflow.validate_reviewed_page(packet, draft, bad)),
          "unresolved ambiguity accepted")
    bad = copy.deepcopy(review)
    bad["discrepancies"] = [{"discrepancy_id": "d1", "description": "edge",
                             "outcome": "open", "resolution": ""}]
    bad["review_sha256"] = workflow._sha(workflow._unsigned(bad, "review_sha256"))
    check(any("explicit resolution" in e for e in workflow.validate_reviewed_page(packet, draft, bad)),
          "open discrepancy accepted")


def test_merge_audits_only_reviewed_page_and_leaves_targets_pending() -> None:
    manifest, annotations, manifest_sha = fixture()
    payload = reviewed_for(manifest["pages"][0], ambiguity=True)
    merged = workflow.merge_reviewed(
        manifest, annotations, [payload], manifest_sha256=manifest_sha
    )
    page_ann = merged["pages"]["p-held"]
    check(page_ann["status"] == "audited" and page_ann["lines"] == payload["draft"]["lines"],
          "independently approved page was not audited exactly")
    check(page_ann["auditor"] == "Auditor Two" and page_ann["ambiguities_resolved"] is True,
          "page audit provenance was not imported")
    check(merged["pages"]["p-dev"] == annotations["pages"]["p-dev"],
          "unreviewed page changed")
    check(merged["targets"] == annotations["targets"],
          "reviewed page geometry assigned or audited correction targets")
    check(merged["targets"]["c-held"]["status"] == "pending"
          and not merged["targets"]["c-held"]["word_ids"],
          "target assignment did not remain pending")
    check(not benchmark.validate_annotations(manifest, merged,
              manifest_sha256=manifest_sha, require_audit=False),
          "merged annotations violate benchmark import contract")
    check(any("target audit is pending" in e for e in benchmark.validate_annotations(
              manifest, merged, manifest_sha256=manifest_sha, require_audit=True)),
          "geometry review accidentally made benchmark freeze-ready")


def test_tampering_and_automatic_geometry_cannot_certify() -> None:
    manifest, annotations, manifest_sha = fixture()
    payload = reviewed_for(manifest["pages"][0])
    tampered = copy.deepcopy(payload)
    tampered["draft"]["lines"][0]["words"][0]["rect"][0] += 0.01
    check(any("draft_sha256" in e for e in workflow.validate_reviewed_payload(tampered)),
          "geometry changed after audit was accepted")
    try:
        workflow.merge_reviewed(manifest, annotations, [tampered], manifest_sha256=manifest_sha)
    except workflow.WorkflowError:
        pass
    else:
        raise AssertionError("tampered reviewed payload was imported")
    raw = copy.deepcopy(annotations)
    raw["pages"]["p-held"]["lines"] = lines("p-held")
    check(raw["pages"]["p-held"]["status"] == "pending",
          "test fixture incorrectly auto-certified geometry")
    try:
        workflow.merge_reviewed(manifest, raw, [], manifest_sha256=manifest_sha)
    except workflow.WorkflowError:
        raise AssertionError("pending generated geometry should remain valid pending input")
    check(raw["pages"]["p-held"]["status"] == "pending",
          "geometry generation changed audit status")


def test_status_population_and_evaluable_counts() -> None:
    manifest, annotations, manifest_sha = fixture()
    queue = workflow.build_queue(manifest, manifest_sha)
    payload = reviewed_for(manifest["pages"][0])
    merged = workflow.merge_reviewed(manifest, annotations, [payload],
                                     manifest_sha256=manifest_sha)
    report = workflow.status_report(
        manifest, queue=queue, drafts=[payload["draft"]], reviewed=[payload],
        annotations=merged, manifest_sha256=manifest_sha,
    )
    heldout = report["population"]["heldout"]
    check(heldout == {"population_pages": 1, "population_cases": 1,
                      "audited_pages": 1, "audited_cases": 0, "evaluable_cases": 0},
          "status conflated reviewed geometry with audited/evaluable target truth")
    check(report["workflow"]["queued_pages"] == 2
          and report["workflow"]["independently_reviewed_pages"] == 1,
          "workflow progress counts are wrong")


def test_cli_round_trip_uses_only_synthetic_data() -> None:
    manifest, annotations, _manifest_sha = fixture()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        manifest_path = root / "manifest.json"
        annotations_path = root / "annotations.json"
        manifest_path.write_bytes(benchmark._json_file_bytes(manifest))
        annotations["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        annotations_path.write_bytes(benchmark._json_file_bytes(annotations))
        script = HERE / "bbox_annotation_workflow.py"
        subprocess.run([sys.executable, str(script), "create-queue", "--manifest",
                        str(manifest_path), "--out-dir", str(root / "work")], check=True,
                       capture_output=True, text=True)
        packet_path = root / "work" / "packets" / "p-held.json"
        packet = json.loads(packet_path.read_text())
        check(set(packet) == workflow.PACKET_KEYS, "CLI packet is not blinded")
        result = subprocess.run([sys.executable, str(script), "status", "--manifest",
                                 str(manifest_path), "--queue", str(root / "work" / "queue.json"),
                                 "--annotations", str(annotations_path)], check=True,
                                capture_output=True, text=True)
        check("heldout population: all 1 pages / 1 cases; audited: 0 pages / 0 cases; evaluable: 0 cases"
              in result.stdout, "CLI status omits population versus audited/evaluable counts")


def test_real_manifest_metadata_counts_without_reading_annotations_or_images() -> None:
    manifest_path = PROJECT_ROOT / "out" / "consolidated_bbox" / "benchmark" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = workflow.status_report(
        manifest, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    heldout = report["population"]["heldout"]
    check(heldout["population_pages"] == 97 and heldout["population_cases"] == 299,
          "status does not report the fixed 97-page/299-case heldout population")
    check(heldout["audited_pages"] == heldout["audited_cases"] == heldout["evaluable_cases"] == 0,
          "absent annotation input was reported as audited truth")


def main() -> int:
    tests = [
        test_queue_is_complete_deterministic_and_blinded,
        test_hash_identity_and_explicit_resolution_guards,
        test_merge_audits_only_reviewed_page_and_leaves_targets_pending,
        test_tampering_and_automatic_geometry_cannot_certify,
        test_status_population_and_evaluable_counts,
        test_cli_round_trip_uses_only_synthetic_data,
        test_real_manifest_metadata_counts_without_reading_annotations_or_images,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} bbox annotation workflow regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
