#!/usr/bin/env python3
"""Blinded, independently reviewed page-geometry annotation workflow.

This tool deliberately separates page/word rectangle annotation from correction
target assignment.  Queue packets contain no book, split, correction, or locator
information.  A reviewed page may populate and audit page geometry in the bbox
benchmark template, but it never changes a target annotation from ``pending``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bbox_benchmark as benchmark


SCHEMA_VERSION = "farsi2epub.bbox-annotation-workflow/v1"
PACKET_KEYS = {
    "page_id",
    "image_sha256",
    "image_path",
    "width_px",
    "height_px",
}


class WorkflowError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _identity(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _aware_time(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def packet_for_page(page: dict) -> dict:
    """Return the complete blinded contract for one manifest page."""
    image = page["image"]
    return {
        "page_id": page["page_id"],
        "image_sha256": image["sha256"],
        "image_path": image["path"],
        "width_px": image["width_px"],
        "height_px": image["height_px"],
    }


def validate_packet(packet: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, dict):
        return ["packet must be an object"]
    if set(packet) != PACKET_KEYS:
        errors.append(
            "packet keys must be exactly page_id, image_sha256, image_path, "
            "width_px, height_px"
        )
    if not isinstance(packet.get("page_id"), str) or not packet["page_id"].strip():
        errors.append("packet page_id is required")
    if not benchmark._valid_sha256(packet.get("image_sha256")):
        errors.append("packet image_sha256 is invalid")
    image_path = packet.get("image_path")
    if (
        not isinstance(image_path, str)
        or not image_path
        or Path(image_path).is_absolute()
        or ".." in Path(image_path or ".").parts
    ):
        errors.append("packet image_path must be relative and contained")
    for key in ("width_px", "height_px"):
        value = packet.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"packet {key} must be a positive integer")
    return errors


def build_queue(manifest: dict, manifest_sha256: str) -> dict:
    errors = benchmark._validate_manifest(manifest)
    if errors:
        raise WorkflowError("invalid manifest:\n- " + "\n- ".join(errors))
    tasks = [packet_for_page(page) for page in manifest["pages"]]
    tasks.sort(key=lambda item: item["page_id"])
    if len(tasks) != len(manifest["pages"]) or len({p["page_id"] for p in tasks}) != len(tasks):
        raise WorkflowError("queue does not span each manifest page exactly once")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "ordering": "page_id-v1",
        "tasks": tasks,
        "queue_sha256": _sha(tasks),
    }


def validate_queue(queue: Any, manifest: dict, manifest_sha256: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(queue, dict):
        return ["queue must be an object"]
    if queue.get("schema_version") != SCHEMA_VERSION:
        errors.append("queue schema_version mismatch")
    if queue.get("manifest_sha256") != manifest_sha256:
        errors.append("queue manifest_sha256 mismatch")
    tasks = queue.get("tasks")
    if not isinstance(tasks, list):
        return errors + ["queue tasks must be a list"]
    for index, task in enumerate(tasks):
        errors.extend(f"task {index}: {e}" for e in validate_packet(task))
    expected = sorted(
        (packet_for_page(page) for page in manifest.get("pages", [])),
        key=lambda item: item["page_id"],
    )
    if tasks != expected:
        errors.append("queue must contain every manifest page exactly once in page_id order")
    if queue.get("ordering") != "page_id-v1":
        errors.append("queue ordering mismatch")
    if queue.get("queue_sha256") != _sha(tasks):
        errors.append("queue_sha256 mismatch")
    return errors


def export_queue(manifest_path: Path, out_dir: Path) -> dict:
    manifest = _load(manifest_path)
    queue = build_queue(manifest, _file_sha(manifest_path))
    packet_dir = out_dir / "packets"
    for packet in queue["tasks"]:
        _write(packet_dir / f"{packet['page_id']}.json", packet)
    _write(out_dir / "queue.json", queue)
    return {"pages": len(queue["tasks"]), "queue_sha256": queue["queue_sha256"]}


def initial_draft(packet: dict, annotator: str) -> dict:
    errors = validate_packet(packet)
    if errors:
        raise WorkflowError("invalid packet:\n- " + "\n- ".join(errors))
    if not _identity(annotator):
        raise WorkflowError("annotator identity is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "page_id": packet["page_id"],
        "image_sha256": packet["image_sha256"],
        "annotator": " ".join(annotator.split()),
        "status": "draft",
        "page_disposition": "annotated",
        "unreadable_reason": "",
        "lines": [],
        "ambiguities": [],
        "draft_sha256": None,
    }


def _validate_rect(value: Any, where: str, errors: list[str]) -> bool:
    before = len(errors)
    benchmark._rect(value, where, errors)
    return len(errors) == before


def validate_lines(page_id: str, lines: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(lines, list):
        return ["draft lines must be a list"]
    seen_words: set[str] = set()
    for li, line in enumerate(lines):
        where = f"lines[{li}]"
        if not isinstance(line, dict):
            errors.append(f"{where}: line must be an object")
            continue
        expected_line_id = f"{page_id}:l{li}"
        if line.get("line_id") != expected_line_id:
            errors.append(f"{where}: line_id must be {expected_line_id}")
        if line.get("order") != li:
            errors.append(f"{where}: line order must be {li}")
        line_ok = _validate_rect(line.get("rect"), where, errors)
        words = line.get("words")
        if not isinstance(words, list):
            errors.append(f"{where}: words must be a list")
            continue
        for wi, word in enumerate(words):
            word_where = f"{where}.words[{wi}]"
            if not isinstance(word, dict):
                errors.append(f"{word_where}: word must be an object")
                continue
            expected_word_id = f"{expected_line_id}:w{wi}"
            if word.get("word_id") != expected_word_id:
                errors.append(f"{word_where}: word_id must be {expected_word_id}")
            elif expected_word_id in seen_words:
                errors.append(f"{word_where}: duplicate word_id")
            seen_words.add(expected_word_id)
            if word.get("order") != wi:
                errors.append(f"{word_where}: word order must be {wi} (RTL reading order)")
            if not isinstance(word.get("text"), str) or not word["text"].strip():
                errors.append(f"{word_where}: non-empty text is required")
            word_ok = _validate_rect(word.get("rect"), word_where, errors)
            if line_ok and word_ok:
                x0, y0, x1, y1 = word["rect"]
                lx0, ly0, lx1, ly1 = line["rect"]
                if x0 < lx0 or y0 < ly0 or x1 > lx1 or y1 > ly1:
                    errors.append(f"{word_where}: word rect must be inside line rect")
    return errors


def _unsigned(value: dict, hash_key: str) -> dict:
    return {key: item for key, item in value.items() if key != hash_key}


def seal_draft(packet: dict, draft: dict) -> dict:
    sealed = copy.deepcopy(draft)
    sealed["status"] = "submitted"
    sealed["draft_sha256"] = _sha(_unsigned(sealed, "draft_sha256"))
    errors = validate_draft(packet, sealed)
    if errors:
        raise WorkflowError("draft submission failed:\n- " + "\n- ".join(errors))
    return sealed


def validate_draft(packet: Any, draft: Any) -> list[str]:
    errors = validate_packet(packet)
    if not isinstance(draft, dict):
        return errors + ["draft must be an object"]
    allowed = {
        "schema_version", "page_id", "image_sha256", "annotator", "status",
        "page_disposition", "unreadable_reason", "lines", "ambiguities",
        "draft_sha256",
    }
    optional = {"method", "regions", "source_draft_sha256"}
    if not allowed.issubset(draft) or set(draft)-allowed-optional:
        errors.append("draft fields do not match the workflow schema")
    if "source_draft_sha256" in draft and not benchmark._valid_sha256(draft["source_draft_sha256"]):
        errors.append("source_draft_sha256 must identify the original annotation artifact")
    if draft.get("schema_version") != SCHEMA_VERSION:
        errors.append("draft schema_version mismatch")
    if draft.get("page_id") != packet.get("page_id"):
        errors.append("draft page_id does not match packet")
    if draft.get("image_sha256") != packet.get("image_sha256"):
        errors.append("draft image_sha256 does not match packet")
    if not _identity(draft.get("annotator")):
        errors.append("draft annotator identity is required")
    if draft.get("status") != "submitted":
        errors.append("draft must be submitted before review")
    disposition = draft.get("page_disposition")
    if disposition not in {"annotated", "unreadable"}:
        errors.append("draft page_disposition must be annotated or unreadable")
    lines = draft.get("lines")
    errors.extend(validate_lines(str(packet.get("page_id") or ""), lines))
    if disposition == "annotated" and (not isinstance(lines, list) or not lines or not any(line.get("words") for line in lines if isinstance(line, dict))):
        errors.append("annotated draft requires at least one word rectangle")
    if disposition == "unreadable":
        if lines:
            errors.append("unreadable draft must not contain geometry")
        if not str(draft.get("unreadable_reason") or "").strip():
            errors.append("unreadable draft requires a reason")
    elif draft.get("unreadable_reason"):
        errors.append("annotated draft must not have an unreadable reason")
    ambiguities = draft.get("ambiguities")
    if not isinstance(ambiguities, list):
        errors.append("draft ambiguities must be a list")
        ambiguities = []
    ids: list[str] = []
    for i, ambiguity in enumerate(ambiguities):
        if not isinstance(ambiguity, dict) or set(ambiguity) != {"ambiguity_id", "description"}:
            errors.append(f"ambiguities[{i}] must contain only ambiguity_id and description")
            continue
        ambiguity_id = ambiguity.get("ambiguity_id")
        if not isinstance(ambiguity_id, str) or not ambiguity_id.strip():
            errors.append(f"ambiguities[{i}]: ambiguity_id is required")
        else:
            ids.append(ambiguity_id)
        if not str(ambiguity.get("description") or "").strip():
            errors.append(f"ambiguities[{i}]: description is required")
    if len(ids) != len(set(ids)):
        errors.append("draft ambiguity IDs must be unique")
    digest = draft.get("draft_sha256")
    if not benchmark._valid_sha256(digest) or digest != _sha(_unsigned(draft, "draft_sha256")):
        errors.append("draft_sha256 does not bind the exact submitted draft")
    return errors


def initial_review(packet: dict, draft: dict, auditor: str) -> dict:
    errors = validate_draft(packet, draft)
    if errors:
        raise WorkflowError("cannot review invalid draft:\n- " + "\n- ".join(errors))
    if not _identity(auditor):
        raise WorkflowError("auditor identity is required")
    if _identity(auditor) == _identity(draft["annotator"]):
        raise WorkflowError("annotator and auditor must be distinct identities")
    return {
        "schema_version": SCHEMA_VERSION,
        "page_id": packet["page_id"],
        "image_sha256": packet["image_sha256"],
        "draft_sha256": draft["draft_sha256"],
        "auditor": " ".join(auditor.split()),
        "audited_at": "",
        "decision": "pending",
        "geometry_checked": False,
        "ambiguities_checked": False,
        "discrepancies_checked": False,
        "ambiguity_resolutions": {
            item["ambiguity_id"]: {"outcome": "pending", "note": ""}
            for item in draft["ambiguities"]
        },
        "discrepancies": [],
        "review_sha256": None,
    }


def validate_reviewed_page(packet: Any, draft: Any, review: Any) -> list[str]:
    """Pure proof check for one independently reviewed page draft."""
    errors = validate_draft(packet, draft)
    if not isinstance(review, dict):
        return errors + ["review must be an object"]
    allowed = {
        "schema_version", "page_id", "image_sha256", "draft_sha256", "auditor",
        "audited_at", "decision", "geometry_checked", "ambiguities_checked",
        "discrepancies_checked", "ambiguity_resolutions", "discrepancies",
        "review_sha256",
    }
    if set(review) != allowed:
        errors.append("review fields do not match the workflow schema")
    if review.get("schema_version") != SCHEMA_VERSION:
        errors.append("review schema_version mismatch")
    if review.get("page_id") != packet.get("page_id") or review.get("page_id") != draft.get("page_id"):
        errors.append("review page_id binding mismatch")
    if review.get("image_sha256") != packet.get("image_sha256") or review.get("image_sha256") != draft.get("image_sha256"):
        errors.append("review image_sha256 binding mismatch")
    if review.get("draft_sha256") != draft.get("draft_sha256"):
        errors.append("review does not bind the exact draft_sha256")
    auditor = _identity(review.get("auditor"))
    if not auditor:
        errors.append("review auditor identity is required")
    if auditor and auditor == _identity(draft.get("annotator")):
        errors.append("annotator and auditor must be distinct identities")
    if review.get("decision") != "approved":
        errors.append("review decision must be approved")
    for key in ("geometry_checked", "ambiguities_checked", "discrepancies_checked"):
        if review.get(key) is not True:
            errors.append(f"review {key} must be true")
    if not _aware_time(review.get("audited_at")):
        errors.append("review audited_at must be a timezone-aware ISO-8601 timestamp")
    expected_ambiguities = {
        item["ambiguity_id"]
        for item in draft.get("ambiguities", [])
        if isinstance(item, dict) and isinstance(item.get("ambiguity_id"), str)
    }
    resolutions = review.get("ambiguity_resolutions")
    if not isinstance(resolutions, dict) or set(resolutions) != expected_ambiguities:
        errors.append("review must resolve exactly every declared draft ambiguity")
        resolutions = resolutions if isinstance(resolutions, dict) else {}
    for ambiguity_id, resolution in resolutions.items():
        if (
            not isinstance(resolution, dict)
            or set(resolution) != {"outcome", "note"}
            or resolution.get("outcome") != "resolved"
            or not str(resolution.get("note") or "").strip()
        ):
            errors.append(f"ambiguity {ambiguity_id}: explicit resolved outcome and note required")
    discrepancies = review.get("discrepancies")
    if not isinstance(discrepancies, list):
        errors.append("review discrepancies must be a list")
        discrepancies = []
    discrepancy_ids: list[str] = []
    for i, discrepancy in enumerate(discrepancies):
        required = {"discrepancy_id", "description", "outcome", "resolution"}
        if not isinstance(discrepancy, dict) or set(discrepancy) != required:
            errors.append(f"discrepancies[{i}] must contain exactly {sorted(required)}")
            continue
        discrepancy_ids.append(str(discrepancy.get("discrepancy_id") or ""))
        if not discrepancy_ids[-1] or not str(discrepancy.get("description") or "").strip():
            errors.append(f"discrepancies[{i}]: identity and description are required")
        if discrepancy.get("outcome") != "resolved" or not str(discrepancy.get("resolution") or "").strip():
            errors.append(f"discrepancies[{i}]: explicit resolution is required")
    if len(discrepancy_ids) != len(set(discrepancy_ids)):
        errors.append("review discrepancy IDs must be unique")
    digest = review.get("review_sha256")
    if not benchmark._valid_sha256(digest) or digest != _sha(_unsigned(review, "review_sha256")):
        errors.append("review_sha256 does not bind the exact review")
    return errors


def seal_review(packet: dict, draft: dict, review: dict) -> dict:
    sealed = copy.deepcopy(review)
    sealed["review_sha256"] = _sha(_unsigned(sealed, "review_sha256"))
    errors = validate_reviewed_page(packet, draft, sealed)
    if errors:
        raise WorkflowError("review submission failed:\n- " + "\n- ".join(errors))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "packet": copy.deepcopy(packet),
        "draft": copy.deepcopy(draft),
        "review": sealed,
    }
    payload["reviewed_sha256"] = _sha(payload)
    return payload


def validate_reviewed_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["reviewed payload must be an object"]
    allowed = {"schema_version", "packet", "draft", "review", "reviewed_sha256"}
    errors: list[str] = []
    if set(payload) != allowed:
        errors.append("reviewed payload fields do not match the workflow schema")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("reviewed payload schema_version mismatch")
    errors.extend(
        validate_reviewed_page(
            payload.get("packet"), payload.get("draft"), payload.get("review")
        )
    )
    digest = payload.get("reviewed_sha256")
    unsigned = _unsigned(payload, "reviewed_sha256")
    if not benchmark._valid_sha256(digest) or digest != _sha(unsigned):
        errors.append("reviewed_sha256 does not bind the exact reviewed payload")
    return errors


def merge_reviewed(
    manifest: dict,
    annotations: dict,
    payloads: Iterable[dict],
    *,
    manifest_sha256: str,
) -> dict:
    errors = benchmark.validate_annotations(
        manifest,
        annotations,
        manifest_sha256=manifest_sha256,
        require_audit=False,
    )
    if errors:
        raise WorkflowError("base annotations are invalid:\n- " + "\n- ".join(errors))
    pages = {page["page_id"]: page for page in manifest["pages"]}
    merged = copy.deepcopy(annotations)
    seen: set[str] = set()
    for payload in payloads:
        payload_errors = validate_reviewed_payload(payload)
        if payload_errors:
            raise WorkflowError("invalid reviewed payload:\n- " + "\n- ".join(payload_errors))
        packet, draft, review = payload["packet"], payload["draft"], payload["review"]
        page_id = packet["page_id"]
        if page_id not in pages or packet != packet_for_page(pages[page_id]):
            raise WorkflowError(f"{page_id}: reviewed packet does not match manifest")
        if page_id in seen:
            raise WorkflowError(f"{page_id}: duplicate reviewed page import")
        seen.add(page_id)
        for case_id in pages[page_id]["case_ids"]:
            target = merged["targets"][case_id]
            if (
                target.get("status") != "pending"
                or target.get("word_ids")
                or target.get("ambiguity_resolution") != "pending"
            ):
                raise WorkflowError(
                    f"{case_id}: target assignment is not pending; reviewed geometry import cannot change it"
                )
        current = merged["pages"][page_id]
        if current.get("lines") and current["lines"] != draft["lines"]:
            raise WorkflowError(f"{page_id}: import conflicts with existing page geometry")
        unreadable = draft["page_disposition"] == "unreadable"
        current.update(
            {
                "image_sha256": packet["image_sha256"],
                "status": "unreadable" if unreadable else "audited",
                "auditor": review["auditor"],
                "audited_at": review["audited_at"],
                "ambiguities_resolved": True,
                "unreadable_reason": draft["unreadable_reason"] if unreadable else "",
                "lines": copy.deepcopy(draft["lines"]),
            }
        )
    errors = benchmark.validate_annotations(
        manifest,
        merged,
        manifest_sha256=manifest_sha256,
        require_audit=False,
    )
    if errors:
        raise WorkflowError("merged annotations are invalid:\n- " + "\n- ".join(errors))
    return merged


def status_report(
    manifest: dict,
    *,
    queue: dict | None = None,
    drafts: Iterable[dict] = (),
    reviewed: Iterable[dict] = (),
    annotations: dict | None = None,
    manifest_sha256: str,
) -> dict:
    page_defs = {page["page_id"]: page for page in manifest["pages"]}
    case_defs = {case["case_id"]: case for case in manifest["cases"]}
    invalid: list[str] = []
    if queue is not None:
        invalid.extend("queue: " + error for error in validate_queue(queue, manifest, manifest_sha256))
    valid_drafts: dict[str, dict] = {}
    for draft in drafts:
        page_id = str(draft.get("page_id") or "") if isinstance(draft, dict) else ""
        page = page_defs.get(page_id)
        packet = packet_for_page(page) if page else {}
        errors = validate_draft(packet, draft)
        if errors:
            invalid.extend(f"draft {page_id or '?'}: {error}" for error in errors)
        else:
            valid_drafts[page_id] = draft
    valid_reviews: dict[str, dict] = {}
    for payload in reviewed:
        errors = validate_reviewed_payload(payload)
        page_id = str(payload.get("packet", {}).get("page_id") or "?") if isinstance(payload, dict) else "?"
        if not errors and page_id in page_defs and payload["packet"] == packet_for_page(page_defs[page_id]):
            valid_reviews[page_id] = payload
        else:
            invalid.extend(f"review {page_id}: {error}" for error in (errors or ["packet does not match manifest"]))
    draft_ambiguities = sum(len(draft["ambiguities"]) for draft in valid_drafts.values())
    reviewed_ambiguities = sum(
        len(payload["review"]["ambiguity_resolutions"])
        for payload in valid_reviews.values()
    )
    truth: dict[str, dict[str, int]] = {}
    if annotations is not None:
        errors = benchmark.validate_annotations(
            manifest, annotations, manifest_sha256=manifest_sha256, require_audit=False
        )
        invalid.extend("annotations: " + error for error in errors)
    for split in ("all", "dev", "heldout"):
        page_ids = {
            page_id for page_id, page in page_defs.items()
            if split == "all" or page["split"] == split
        }
        case_ids = {
            case_id for case_id, case in case_defs.items() if case["page_id"] in page_ids
        }
        pages_ann = annotations.get("pages", {}) if isinstance(annotations, dict) else {}
        targets_ann = annotations.get("targets", {}) if isinstance(annotations, dict) else {}
        audited_page_ids = {
            page_id for page_id in page_ids
            if pages_ann.get(page_id, {}).get("status") in {"audited", "unreadable"}
        }
        audited_cases = {
            case_id for case_id in case_ids
            if case_defs[case_id]["page_id"] in audited_page_ids
            and targets_ann.get(case_id, {}).get("status") in {"audited", "unreadable"}
        }
        evaluable_cases = {
            case_id for case_id in case_ids
            if pages_ann.get(case_defs[case_id]["page_id"], {}).get("status") == "audited"
            and targets_ann.get(case_id, {}).get("status") == "audited"
        }
        truth[split] = {
            "population_pages": len(page_ids),
            "population_cases": len(case_ids),
            "audited_pages": len(audited_page_ids),
            "audited_cases": len(audited_cases),
            "evaluable_cases": len(evaluable_cases),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "population": truth,
        "workflow": {
            "queued_pages": len(queue.get("tasks", [])) if queue and not any(e.startswith("queue:") for e in invalid) else 0,
            "submitted_draft_pages": len(valid_drafts),
            "independently_reviewed_pages": len(valid_reviews),
            "draft_ambiguities": draft_ambiguities,
            "reviewed_ambiguity_resolutions": reviewed_ambiguities,
        },
        "invalid_inputs": invalid,
    }


def _json_files(directory: Path | None) -> list[dict]:
    if directory is None:
        return []
    return [_load(path) for path in sorted(directory.glob("*.json"))]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create-queue", help="write blinded packets for every manifest page")
    make.add_argument("--manifest", type=Path, required=True)
    make.add_argument("--out-dir", type=Path, required=True)
    init_d = sub.add_parser("init-draft", help="start an editable page geometry draft")
    init_d.add_argument("--packet", type=Path, required=True)
    init_d.add_argument("--annotator", required=True)
    init_d.add_argument("--out", type=Path, required=True)
    submit_d = sub.add_parser("submit-draft", help="validate and hash an edited draft")
    submit_d.add_argument("--packet", type=Path, required=True)
    submit_d.add_argument("--draft", type=Path, required=True)
    submit_d.add_argument("--out", type=Path, required=True)
    init_r = sub.add_parser("init-review", help="start independent review of a submitted draft")
    init_r.add_argument("--packet", type=Path, required=True)
    init_r.add_argument("--draft", type=Path, required=True)
    init_r.add_argument("--auditor", required=True)
    init_r.add_argument("--out", type=Path, required=True)
    submit_r = sub.add_parser("submit-review", help="seal an approved independent review")
    submit_r.add_argument("--packet", type=Path, required=True)
    submit_r.add_argument("--draft", type=Path, required=True)
    submit_r.add_argument("--review", type=Path, required=True)
    submit_r.add_argument("--out", type=Path, required=True)
    merge = sub.add_parser("import-reviewed", help="merge approved page geometry; targets stay pending")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--annotations", type=Path, required=True)
    merge.add_argument("--reviewed", type=Path, action="append", required=True)
    merge.add_argument("--out", type=Path, required=True)
    status = sub.add_parser("status", help="report queue, review, audit, and evaluable progress")
    status.add_argument("--manifest", type=Path, required=True)
    status.add_argument("--queue", type=Path)
    status.add_argument("--draft-dir", type=Path)
    status.add_argument("--reviewed-dir", type=Path)
    status.add_argument("--annotations", type=Path)
    status.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create-queue":
        result = export_queue(args.manifest, args.out_dir)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "init-draft":
        _write(args.out, initial_draft(_load(args.packet), args.annotator))
        return 0
    if args.command == "submit-draft":
        _write(args.out, seal_draft(_load(args.packet), _load(args.draft)))
        return 0
    if args.command == "init-review":
        _write(args.out, initial_review(_load(args.packet), _load(args.draft), args.auditor))
        return 0
    if args.command == "submit-review":
        _write(args.out, seal_review(_load(args.packet), _load(args.draft), _load(args.review)))
        return 0
    manifest = _load(args.manifest)
    manifest_sha256 = _file_sha(args.manifest)
    if args.command == "import-reviewed":
        merged = merge_reviewed(
            manifest,
            _load(args.annotations),
            (_load(path) for path in args.reviewed),
            manifest_sha256=manifest_sha256,
        )
        _write(args.out, merged)
        return 0
    report = status_report(
        manifest,
        queue=_load(args.queue) if args.queue else None,
        drafts=_json_files(args.draft_dir),
        reviewed=_json_files(args.reviewed_dir),
        annotations=_load(args.annotations) if args.annotations else None,
        manifest_sha256=manifest_sha256,
    )
    if args.out:
        _write(args.out, report)
    heldout = report["population"]["heldout"]
    print(
        "heldout population: all "
        f"{heldout['population_pages']} pages / {heldout['population_cases']} cases; "
        f"audited: {heldout['audited_pages']} pages / {heldout['audited_cases']} cases; "
        f"evaluable: {heldout['evaluable_cases']} cases"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if report["invalid_inputs"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
