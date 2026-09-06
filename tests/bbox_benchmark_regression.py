#!/usr/bin/env python3
"""Standalone regressions for the human bbox benchmark contract.

Run with::

    ./venv/bin/python tests/bbox_benchmark_regression.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bbox_benchmark as bb


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha(ch: str) -> str:
    return ch * 64


def _page(page_id: str, slug: str, page: int, split: str, case_ids: list[str]) -> dict:
    return {
        "page_id": page_id,
        "slug": slug,
        "page": page,
        "split": split,
        "markdown_sha256": _sha("a" if page == 1 else "b"),
        "case_ids": case_ids,
        "image": {
            "path": f"pages/{page_id}.png",
            "sha256": _sha("c" if page == 1 else "d"),
            "width_px": 800,
            "height_px": 1200,
            "render": {
                "render_id": bb.RENDER_ID,
                "engine": "PyMuPDF",
                "engine_version": "test",
                "long_edge_px": bb.LONG_EDGE_STD,
                "alpha": False,
                "zoom": 2.0,
            },
            "source": {
                "pdf_sha256": _sha("e"),
                "page_width_pt": 400.0,
                "page_height_pt": 600.0,
            },
        },
    }


def _case(case_id: str, page_id: str, split: str, kind: str) -> dict:
    old, new, span = ("کهنه", "نو", [0, 4])
    if kind == "insertion":
        old, new, span = "", "افزوده", [5, 5]
    return {
        "case_id": case_id,
        "page_id": page_id,
        "split": split,
        "kind": kind,
        "issue_type": "test",
        "target_annotation_id": "target:" + case_id,
        "hunk": {
            "old": old,
            "new": new,
            "markdown_span": span,
            "candidate_spans": [span],
            "ctx_before": "پیش ",
            "ctx_after": " پس",
            "occurrence": 1,
            "anchor_unique": True,
        },
    }


def _line(page_id: str, words: list[str]) -> dict:
    line_id = f"{page_id}:l0"
    out = []
    for i, text in enumerate(words):
        right = 0.9 - i * 0.2
        out.append(
            {
                "word_id": f"{line_id}:w{i}",
                "order": i,
                "text": text,
                "rect": [right - 0.12, 0.1, right, 0.16],
            }
        )
    return {"line_id": line_id, "order": 0, "rect": [0.05, 0.08, 0.95, 0.18], "words": out}


def _valid_objects() -> tuple[dict, dict, dict]:
    heldout_case = _case("c_hold", "p_hold", "heldout", "replacement")
    dev_case = _case("c_dev", "p_dev", "dev", "insertion")
    manifest = {
        "schema_version": bb.SCHEMA_VERSION,
        "split_policy": {
            "id": bb.SPLIT_ID,
            "unit": "page",
            "heldout_min_cases": 1,
            "heldout_min_books": 1,
        },
        "render_policy": {"id": bb.RENDER_ID, "long_edge_px": bb.LONG_EDGE_STD},
        "books": [
            {"slug": "book", "pdf_sha256": _sha("e"), "page_count": 2, "case_count": 2}
        ],
        "pages": [
            _page("p_hold", "book", 1, "heldout", ["c_hold"]),
            _page("p_dev", "book", 2, "dev", ["c_dev"]),
        ],
        "cases": [heldout_case, dev_case],
    }
    manifest_sha = bb._sha256_bytes(bb._json_file_bytes(manifest))
    hold_line = _line("p_hold", ["واژه", "هدف", "آخر"])
    dev_line = _line("p_dev", ["آغاز", "میانه", "پایان"])
    annotations = {
        "schema_version": bb.SCHEMA_VERSION,
        "manifest_sha256": manifest_sha,
        "pages": {
            "p_hold": {
                "image_sha256": _sha("c"),
                "status": "audited",
                "auditor": "human",
                "audited_at": "2026-09-05T12:00:00Z",
                "ambiguities_resolved": True,
                "unreadable_reason": "",
                "lines": [hold_line],
            },
            "p_dev": {
                "image_sha256": _sha("d"),
                "status": "audited",
                "auditor": "human",
                "audited_at": "2026-09-05T12:01:00+00:00",
                "ambiguities_resolved": True,
                "unreadable_reason": "",
                "lines": [dev_line],
            },
        },
        "targets": {
            "c_hold": {
                "status": "audited",
                "ambiguity_resolution": "resolved",
                "word_ids": ["p_hold:l0:w0", "p_hold:l0:w1"],
                "insertion_anchor": None,
                "note": "",
            },
            "c_dev": {
                "status": "audited",
                "ambiguity_resolution": "resolved",
                "word_ids": [],
                "insertion_anchor": {
                    "before_word_id": "p_dev:l0:w0",
                    "after_word_id": "p_dev:l0:w1",
                },
                "note": "",
            },
        },
    }
    frozen = {
        "schema_version": bb.SCHEMA_VERSION,
        "manifest_sha256": manifest_sha,
        "annotation_sha256": bb._sha256_bytes(bb._json_file_bytes(annotations)),
        "manifest": manifest,
        "annotations": annotations,
        "integrity": bb._integrity(manifest, annotations),
    }
    frozen["benchmark_sha256"] = bb._sha256_bytes(bb._canonical_bytes(frozen))
    return manifest, annotations, frozen


def _rehash(frozen: dict) -> None:
    frozen["manifest_sha256"] = bb._sha256_bytes(bb._json_file_bytes(frozen["manifest"]))
    frozen["annotations"]["manifest_sha256"] = frozen["manifest_sha256"]
    frozen["annotation_sha256"] = bb._sha256_bytes(
        bb._json_file_bytes(frozen["annotations"])
    )
    frozen["integrity"] = bb._integrity(frozen["manifest"], frozen["annotations"])
    unsigned = {k: v for k, v in frozen.items() if k != "benchmark_sha256"}
    frozen["benchmark_sha256"] = bb._sha256_bytes(bb._canonical_bytes(unsigned))


def test_pure_import_and_valid_freeze() -> None:
    check("farsi2epub.locate" not in sys.modules, "validator import loaded production locator")
    manifest, annotations, frozen = _valid_objects()
    check(bb._validate_manifest(manifest) == [], "synthetic manifest should validate")
    check(
        bb.validate_annotations(
            manifest,
            annotations,
            manifest_sha256=frozen["manifest_sha256"],
            require_audit=True,
        )
        == [],
        "synthetic audited annotations should validate",
    )
    check(bb.validate_frozen(frozen) == [], "valid frozen wrapper should validate")


def test_freeze_rejects_pending_and_tampering() -> None:
    manifest, annotations, frozen = _valid_objects()
    pending = copy.deepcopy(annotations)
    pending["pages"]["p_hold"]["status"] = "pending"
    errors = bb.validate_annotations(
        manifest,
        pending,
        manifest_sha256=frozen["manifest_sha256"],
        require_audit=True,
    )
    check(any("page audit is pending" in e for e in errors), "pending page was accepted")

    tampered = copy.deepcopy(frozen)
    tampered["annotations"]["targets"]["c_hold"]["word_ids"] = ["p_hold:l0:w2"]
    check(bb.validate_frozen(tampered), "tampered target/hash was accepted")
    tampered = copy.deepcopy(frozen)
    tampered["manifest"]["cases"][0]["tier"] = "scan_vlm"
    _rehash(tampered)
    check(
        any("forbidden locator-derived field" in e for e in bb.validate_frozen(tampered)),
        "locator-derived manifest field was accepted",
    )


def test_target_identity_and_insertion_guards() -> None:
    _manifest, _annotations, frozen = _valid_objects()
    bad = copy.deepcopy(frozen)
    bad["annotations"]["targets"]["c_hold"]["word_ids"] = [
        "p_hold:l0:w0",
        "p_hold:l0:w2",
    ]
    _rehash(bad)
    check(
        any("must be contiguous" in e for e in bb.validate_frozen(bad)),
        "disjoint target IDs were accepted",
    )
    bad = copy.deepcopy(frozen)
    bad["annotations"]["targets"]["c_dev"]["word_ids"] = ["p_dev:l0:w1"]
    _rehash(bad)
    check(
        any("insertion target must be exactly" in e for e in bb.validate_frozen(bad)),
        "insertion target words were accepted",
    )
    bad = copy.deepcopy(frozen)
    bad["annotations"]["targets"]["c_dev"]["insertion_anchor"] = {
        "before_word_id": "p_dev:l0:w2",
        "after_word_id": "p_dev:l0:w0",
    }
    _rehash(bad)
    check(
        any("insertion target must be exactly" in e for e in bb.validate_frozen(bad)),
        "reversed insertion anchors were accepted",
    )
    bad = copy.deepcopy(frozen)
    bad["annotations"]["targets"]["c_dev"]["insertion_anchor"] = {
        "before_word_id": "p_dev:l0:w1",
        "after_word_id": None,
    }
    _rehash(bad)
    check(
        any("insertion target must be exactly" in e for e in bb.validate_frozen(bad)),
        "single middle insertion anchor was accepted",
    )


def test_printed_insertion_target():
    _, _, frozen = _valid_objects()
    target = frozen["annotations"]["targets"]["c_dev"]
    target["insertion_anchor"] = {"before_word_id":"p_dev:l0:w0", "after_word_id":"p_dev:l0:w2"}
    target["word_ids"] = ["p_dev:l0:w1"]
    _rehash(frozen)
    check(not bb.validate_frozen(frozen), "printed omitted word between anchors rejected")


def test_unreadable_is_explicit_and_complete() -> None:
    _manifest, _annotations, frozen = _valid_objects()
    unreadable = copy.deepcopy(frozen)
    page = unreadable["annotations"]["pages"]["p_dev"]
    page.update(
        {
            "status": "unreadable",
            "unreadable_reason": "page is physically torn",
            "lines": [],
        }
    )
    target = unreadable["annotations"]["targets"]["c_dev"]
    target.update(
        {
            "status": "unreadable",
            "word_ids": [],
            "insertion_anchor": {"before_word_id": None, "after_word_id": None},
            "note": "insertion point cannot be read",
        }
    )
    _rehash(unreadable)
    check(bb.validate_frozen(unreadable) == [], "explicit unreadable case should validate")
    unreadable["annotations"]["targets"]["c_dev"]["note"] = ""
    _rehash(unreadable)
    check(
        any("requires a note" in e for e in bb.validate_frozen(unreadable)),
        "unexplained unreadable target was accepted",
    )


def test_deterministic_page_split() -> None:
    pages = [
        {"page_id": f"p{i}", "slug": f"book{i % 3}", "page": i + 1, "case_ids": [f"c{i}_{j}" for j in range(3)]}
        for i in range(100)
    ]
    other = copy.deepcopy(pages)
    bb._assign_splits(pages, heldout_min=90, heldout_books_min=2)
    bb._assign_splits(other, heldout_min=90, heldout_books_min=2)
    check(
        [(p["page_id"], p["split"]) for p in pages]
        == [(p["page_id"], p["split"]) for p in other],
        "split assignment is not deterministic",
    )
    heldout = [p for p in pages if p["split"] == "heldout"]
    check(sum(len(p["case_ids"]) for p in heldout) >= 90, "heldout minimum not met")
    check(len({p["slug"] for p in heldout}) >= 2, "heldout book minimum not met")


def test_exported_artifact_contract() -> None:
    root = PROJECT_ROOT / "out" / "consolidated_bbox" / "benchmark"
    manifest_path = root / "manifest.json"
    annotations_path = root / "annotations.template.json"
    check(manifest_path.is_file() and annotations_path.is_file(), "export artifact is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    check(bb._validate_manifest(manifest) == [], "exported manifest does not validate")
    check(len(manifest["cases"]) == 698, "unexpected exported correction count")
    check(len(manifest["pages"]) == 227, "unexpected exported page count")
    check(len(manifest["books"]) == 6, "unexpected exported book count")
    check(
        sum(c["split"] == "heldout" for c in manifest["cases"]) == 299,
        "unexpected heldout correction count",
    )
    check(
        bb.validate_annotations(
            manifest,
            annotations,
            manifest_path=manifest_path,
            require_audit=False,
        )
        == [],
        "pending annotation template does not validate for import",
    )
    check(
        all(p["status"] == "pending" and not p["lines"] for p in annotations["pages"].values()),
        "template contains fabricated or audited page geometry",
    )
    check(
        all(t["status"] == "pending" and not t["word_ids"] for t in annotations["targets"].values()),
        "template contains fabricated or audited targets",
    )
    for page in manifest["pages"]:
        image = root / page["image"]["path"]
        check(image.is_file(), f"missing clean page image {image.name}")
        check(bb._sha256_file(image) == page["image"]["sha256"], f"image hash drift {image.name}")
    check(
        any(len(c["hunk"]["old"].split()) > 5 for c in manifest["cases"]),
        "manifest appears to cap every hunk to locator query length",
    )


def main() -> int:
    tests = [
        test_pure_import_and_valid_freeze,
        test_freeze_rejects_pending_and_tampering,
        test_target_identity_and_insertion_guards,
        test_printed_insertion_target,
        test_unreadable_is_explicit_and_complete,
        test_deterministic_page_split,
        test_exported_artifact_contract,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} bbox benchmark regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
