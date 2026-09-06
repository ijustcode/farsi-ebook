#!/usr/bin/env python3
"""Build and freeze a human-audited, locator-independent bbox benchmark.

The exporter reads the same pending QC hunks as the review UI, but deliberately
never calls ``locate`` or ``review._box_specs``.  Its page PNGs contain no
overlays.  Human annotations live under ``out/`` and are never imported by the
production package.

Typical use::

    ./venv/bin/python tests/bbox_benchmark.py export \
        --out-dir out/consolidated_bbox/benchmark
    # Open annotate.html, annotate pages, and download annotations.json.
    ./venv/bin/python tests/bbox_benchmark.py import-annotations \
        --manifest out/consolidated_bbox/benchmark/manifest.json \
        --annotations ~/Downloads/annotations.json \
        --out out/consolidated_bbox/benchmark/annotations.draft.json
    ./venv/bin/python tests/bbox_benchmark.py freeze \
        --manifest out/consolidated_bbox/benchmark/manifest.json \
        --annotations out/consolidated_bbox/benchmark/annotations.draft.json \
        --out out/consolidated_bbox/benchmark/frozen.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from farsi2epub.config import LONG_EDGE_STD  # noqa: E402
from farsi2epub.workspace import DEFAULT_BOOKS_ROOT, Workspace  # noqa: E402


SCHEMA_VERSION = "farsi2epub.bbox-benchmark/v1"
RENDER_ID = "pymupdf-clean-long-edge-v1"
SPLIT_ID = "page-sha256-40pct-heldout-v1"
DEFAULT_HELDOUT_MIN = 200
DEFAULT_HELDOUT_BOOKS_MIN = 2


class BenchmarkError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_file_bytes(value))


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path} must contain a JSON object")
    return value


def _id(prefix: str, *parts: Any) -> str:
    return prefix + "_" + _sha256_bytes(_canonical_bytes(list(parts)))[:20]


def _nth_index(text: str, needle: str, occurrence: int) -> int:
    if occurrence < 1:
        return -1
    if not needle:
        return 0 if occurrence == 1 else -1
    start = 0
    for _ in range(occurrence):
        found = text.find(needle, start)
        if found < 0:
            return -1
        start = found + 1
    return found


def _all_spans(text: str, needle: str) -> list[list[int]]:
    if not needle:
        return []
    spans: list[list[int]] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return spans
        spans.append([found, found + len(needle)])
        start = found + 1


def _hunk_span(text: str, hunk: dict) -> tuple[list[int] | None, list[list[int]]]:
    """Resolve the uncapped old fragment using its production hunk anchors."""
    old = hunk["old"]
    before = hunk["ctx_before"]
    after = hunk["ctx_after"]
    needle = before + old + after
    occurrence = 1 if hunk.get("unique") else int(hunk.get("occurrence") or 1)
    anchor_start = _nth_index(text, needle, occurrence)
    anchored = None
    if anchor_start >= 0:
        start = anchor_start + len(before)
        anchored = [start, start + len(old)]
    candidates = _all_spans(text, old)
    if not old and anchored is not None:
        candidates = [anchored]
    return anchored, candidates


def _kind(hunk: dict) -> str:
    if hunk.get("edit_only"):
        return "finding"
    if not hunk.get("old") and hunk.get("new"):
        return "insertion"
    if hunk.get("old") and not hunk.get("new"):
        return "deletion"
    return "replacement"


def _iter_workspaces(books_root: Path, requested: list[str] | None) -> Iterable[Workspace]:
    slugs = requested or sorted(
        p.name for p in books_root.iterdir() if p.is_dir() and (p / "book.yaml").is_file()
    )
    for slug in slugs:
        yield Workspace.load(slug, books_root=books_root)


def _collect(books_root: Path, books: list[str] | None) -> tuple[list[dict], list[dict]]:
    # Lazy by design: importing this module to validate a frozen benchmark
    # must not even import the production locator. Export alone needs review's
    # pure hunk derivation and pending-QC reader.
    from farsi2epub import review

    page_rows: list[dict] = []
    case_rows: list[dict] = []
    for ws in _iter_workspaces(books_root, books):
        pdf_sha256 = _sha256_file(ws.pdf_path)
        for page_no in ws.pages_done():
            _sidecar, issues, hunks, markdown, panel_kind = review._page_box_inputs(ws, page_no)
            if panel_kind is None or not hunks:
                continue
            markdown_sha256 = _sha256_bytes(markdown.encode("utf-8"))
            page_id = _id("p", ws.slug, page_no, pdf_sha256)
            page_cases: list[dict] = []
            for hunk in hunks:
                span, candidate_spans = _hunk_span(markdown, hunk)
                issue = None
                issue_idx = hunk.get("issue_idx")
                if isinstance(issue_idx, int) and 0 <= issue_idx < len(issues):
                    issue = issues[issue_idx]
                case_id = _id(
                    "c", page_id, hunk["id"], markdown_sha256, hunk["old"], hunk["new"], span
                )
                kind = _kind(hunk)
                page_cases.append(
                    {
                        "case_id": case_id,
                        "page_id": page_id,
                        "kind": kind,
                        "issue_type": (issue or {}).get("type") or "unlinked",
                        "hunk": {
                            "old": hunk["old"],
                            "new": hunk["new"],
                            "markdown_span": span,
                            "candidate_spans": candidate_spans,
                            "ctx_before": hunk["ctx_before"],
                            "ctx_after": hunk["ctx_after"],
                            "occurrence": int(hunk.get("occurrence") or 1),
                            "anchor_unique": bool(hunk.get("unique")),
                        },
                        "target_annotation_id": "target:" + case_id,
                    }
                )
            if not page_cases:
                continue
            page_rows.append(
                {
                    "page_id": page_id,
                    "slug": ws.slug,
                    "page": page_no,
                    "pdf_path": ws.pdf_path,
                    "pdf_sha256": pdf_sha256,
                    "markdown_sha256": markdown_sha256,
                    "case_ids": [c["case_id"] for c in page_cases],
                }
            )
            case_rows.extend(page_cases)
    return page_rows, case_rows


def _assign_splits(
    pages: list[dict], *, heldout_min: int, heldout_books_min: int
) -> None:
    if heldout_min < 1:
        raise BenchmarkError("--heldout-min must be positive")
    total_cases = sum(len(p["case_ids"]) for p in pages)
    if total_cases < heldout_min:
        raise BenchmarkError(
            f"only {total_cases} corrections exist; cannot reserve {heldout_min} heldout"
        )
    ranked = sorted(
        pages,
        key=lambda p: (
            _sha256_bytes(f"{SPLIT_ID}:{p['slug']}:{p['page']}".encode()),
            p["page_id"],
        ),
    )
    heldout: set[str] = {
        p["page_id"]
        for p in ranked
        if int(_sha256_bytes(f"{SPLIT_ID}:{p['slug']}:{p['page']}".encode())[:8], 16)
        % 100
        < 40
    }

    def counts() -> tuple[int, set[str]]:
        selected = [p for p in pages if p["page_id"] in heldout]
        return sum(len(p["case_ids"]) for p in selected), {p["slug"] for p in selected}

    for page in ranked:
        count, slugs = counts()
        if count >= heldout_min and len(slugs) >= heldout_books_min:
            break
        heldout.add(page["page_id"])
    count, slugs = counts()
    if count < heldout_min or len(slugs) < heldout_books_min:
        raise BenchmarkError(
            f"cannot make heldout split: {count} cases across {len(slugs)} books"
        )
    if len(heldout) == len(pages):
        # Keep at least one development page; move the lowest-ranked page only
        # when the heldout requirements remain satisfied.
        for page in ranked:
            heldout.remove(page["page_id"])
            count, slugs = counts()
            if count >= heldout_min and len(slugs) >= heldout_books_min:
                break
            heldout.add(page["page_id"])
    for page in pages:
        page["split"] = "heldout" if page["page_id"] in heldout else "dev"


def _render_page(page_row: dict, images_dir: Path) -> dict:
    path = Path(page_row.pop("pdf_path"))
    doc = fitz.open(str(path))
    try:
        page = doc[page_row["page"] - 1]
        rect = page.rect
        zoom = LONG_EDGE_STD / max(rect.width, rect.height) if max(rect.width, rect.height) else 1.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png = pix.tobytes("png")
        width_px, height_px = pix.width, pix.height
        width_pt, height_pt = rect.width, rect.height
    finally:
        doc.close()
    image_name = page_row["page_id"] + ".png"
    image_path = images_dir / image_name
    image_path.write_bytes(png)
    return {
        "path": "pages/" + image_name,
        "sha256": _sha256_bytes(png),
        "width_px": width_px,
        "height_px": height_px,
        "render": {
            "render_id": RENDER_ID,
            "engine": "PyMuPDF",
            "engine_version": fitz.VersionBind,
            "long_edge_px": LONG_EDGE_STD,
            "alpha": False,
            "zoom": zoom,
        },
        "source": {
            "pdf_sha256": page_row.pop("pdf_sha256"),
            "page_width_pt": width_pt,
            "page_height_pt": height_pt,
        },
    }


def _annotation_template(manifest: dict, manifest_sha256: str) -> dict:
    pages = {
        p["page_id"]: {
            "image_sha256": p["image"]["sha256"],
            "status": "pending",
            "auditor": "",
            "audited_at": "",
            "ambiguities_resolved": False,
            "unreadable_reason": "",
            "lines": [],
        }
        for p in manifest["pages"]
    }
    targets = {}
    for case in manifest["cases"]:
        insertion = case["kind"] == "insertion"
        targets[case["case_id"]] = {
            "status": "pending",
            "ambiguity_resolution": "pending",
            "word_ids": [],
            "insertion_anchor": (
                {"before_word_id": None, "after_word_id": None} if insertion else None
            ),
            "note": "",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "pages": pages,
        "targets": targets,
    }


def export_benchmark(
    out_dir: Path,
    *,
    books_root: Path = DEFAULT_BOOKS_ROOT,
    books: list[str] | None = None,
    heldout_min: int = DEFAULT_HELDOUT_MIN,
    heldout_books_min: int = DEFAULT_HELDOUT_BOOKS_MIN,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "pages"
    images_dir.mkdir(parents=True, exist_ok=True)
    pages, cases = _collect(books_root, books)
    if not pages:
        raise BenchmarkError("no pending review hunks found")
    _assign_splits(pages, heldout_min=heldout_min, heldout_books_min=heldout_books_min)
    for page in pages:
        page["image"] = _render_page(page, images_dir)
    pages.sort(key=lambda p: (p["slug"], p["page"]))
    split_by_page = {p["page_id"]: p["split"] for p in pages}
    for case in cases:
        case["split"] = split_by_page[case["page_id"]]
    cases.sort(key=lambda c: (c["page_id"], c["case_id"]))
    books_out = []
    for slug in sorted({p["slug"] for p in pages}):
        subset = [p for p in pages if p["slug"] == slug]
        books_out.append(
            {
                "slug": slug,
                "pdf_sha256": subset[0]["image"]["source"]["pdf_sha256"],
                "page_count": len(subset),
                "case_count": sum(len(p["case_ids"]) for p in subset),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_policy": {
            "id": SPLIT_ID,
            "unit": "page",
            "heldout_min_cases": heldout_min,
            "heldout_min_books": heldout_books_min,
        },
        "render_policy": {"id": RENDER_ID, "long_edge_px": LONG_EDGE_STD},
        "books": books_out,
        "pages": pages,
        "cases": cases,
    }
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)
    annotations = _annotation_template(manifest, manifest_sha256)
    _write_json(out_dir / "annotations.template.json", annotations)
    (out_dir / "annotate.html").write_text(
        _annotation_html(manifest, annotations), encoding="utf-8"
    )
    summary = summarize(manifest)
    _write_json(out_dir / "summary.json", summary)
    return summary


def summarize(manifest: dict) -> dict:
    out: dict[str, Any] = {
        "pages": len(manifest["pages"]),
        "cases": len(manifest["cases"]),
        "books": len(manifest["books"]),
        "splits": {},
        "by_book": {},
    }
    for split in ("dev", "heldout"):
        pages = [p for p in manifest["pages"] if p["split"] == split]
        page_ids = {p["page_id"] for p in pages}
        cases = [c for c in manifest["cases"] if c["page_id"] in page_ids]
        out["splits"][split] = {
            "pages": len(pages),
            "cases": len(cases),
            "books": sorted({p["slug"] for p in pages}),
        }
    for book in manifest["books"]:
        slug = book["slug"]
        out["by_book"][slug] = {
            "pages": sum(p["slug"] == slug for p in manifest["pages"]),
            "cases": sum(
                any(p["page_id"] == c["page_id"] and p["slug"] == slug for p in manifest["pages"])
                for c in manifest["cases"]
            ),
        }
    return out


def _rect(value: Any, where: str, errors: list[str]) -> bool:
    if not isinstance(value, list) or len(value) != 4 or any(
        not isinstance(x, (int, float)) or isinstance(x, bool) for x in value
    ):
        errors.append(f"{where}: rect must be four numbers")
        return False
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        errors.append(f"{where}: rect must satisfy 0 <= x0 < x1 <= 1 and y likewise")
        return False
    return True


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _validate_manifest(manifest: dict) -> list[str]:
    """Validate the immutable population without touching workspace files."""
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    pages = manifest.get("pages")
    cases = manifest.get("cases")
    books = manifest.get("books")
    if not isinstance(pages, list) or not isinstance(cases, list) or not isinstance(books, list):
        return errors + ["manifest books, pages, and cases must be lists"]
    page_defs = {p.get("page_id"): p for p in pages if isinstance(p, dict) and p.get("page_id")}
    case_defs = {c.get("case_id"): c for c in cases if isinstance(c, dict) and c.get("case_id")}
    book_defs = {b.get("slug"): b for b in books if isinstance(b, dict) and b.get("slug")}
    if len(page_defs) != len(pages):
        errors.append("duplicate or missing manifest page identity")
    if len(case_defs) != len(cases):
        errors.append("duplicate or missing manifest case identity")
    if len(book_defs) != len(books):
        errors.append("duplicate or missing manifest book identity")

    # These keys betray accidental reuse of locator output. Human gold uses
    # only `rect`, nested below annotation lines/words, outside the manifest.
    forbidden = {"box", "bbox", "tier", "locator", "segments", "query"}

    def walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in forbidden:
                    errors.append(f"{where}: forbidden locator-derived field {key}")
                walk(item, where + "." + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{where}[{i}]")

    walk(manifest, "manifest")
    expected_page_cases: dict[str, list[str]] = {page_id: [] for page_id in page_defs}
    for case_id, case in case_defs.items():
        page_id = case.get("page_id")
        if page_id not in page_defs:
            errors.append(f"{case_id}: unknown page_id")
            continue
        expected_page_cases[page_id].append(case_id)
        if case.get("split") != page_defs[page_id].get("split"):
            errors.append(f"{case_id}: case split differs from its page split")
        if case.get("kind") not in {"replacement", "deletion", "insertion", "finding"}:
            errors.append(f"{case_id}: invalid kind")
        if case.get("target_annotation_id") != "target:" + case_id:
            errors.append(f"{case_id}: target_annotation_id mismatch")
        hunk = case.get("hunk")
        if not isinstance(hunk, dict):
            errors.append(f"{case_id}: hunk must be an object")
            continue
        if not isinstance(hunk.get("old"), str) or not isinstance(hunk.get("new"), str):
            errors.append(f"{case_id}: hunk old/new must be strings")
        span = hunk.get("markdown_span")
        if span is not None and (
            not isinstance(span, list)
            or len(span) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) for x in span)
            or not 0 <= span[0] <= span[1]
        ):
            errors.append(f"{case_id}: invalid markdown_span")
        candidates = hunk.get("candidate_spans")
        if not isinstance(candidates, list) or any(
            not isinstance(s, list)
            or len(s) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) for x in s)
            or not 0 <= s[0] <= s[1]
            for s in (candidates if isinstance(candidates, list) else [])
        ):
            errors.append(f"{case_id}: invalid candidate_spans")
    for page_id, page in page_defs.items():
        if page.get("slug") not in book_defs:
            errors.append(f"{page_id}: unknown book slug")
        if not isinstance(page.get("page"), int) or page["page"] < 1:
            errors.append(f"{page_id}: page must be a positive integer")
        if page.get("split") not in {"dev", "heldout"}:
            errors.append(f"{page_id}: invalid split")
        if not _valid_sha256(page.get("markdown_sha256")):
            errors.append(f"{page_id}: invalid markdown_sha256")
        declared = page.get("case_ids")
        if (
            not isinstance(declared, list)
            or len(declared) != len(set(declared))
            or set(declared) != set(expected_page_cases.get(page_id, []))
        ):
            errors.append(f"{page_id}: case_ids do not exactly match cases")
        image = page.get("image")
        if not isinstance(image, dict):
            errors.append(f"{page_id}: image must be an object")
            continue
        if not _valid_sha256(image.get("sha256")):
            errors.append(f"{page_id}: invalid image sha256")
        image_path = image.get("path")
        if not isinstance(image_path, str) or Path(image_path).is_absolute() or ".." in Path(image_path or ".").parts:
            errors.append(f"{page_id}: image path must be relative and contained")
        if not isinstance(image.get("width_px"), int) or image["width_px"] < 1:
            errors.append(f"{page_id}: invalid image width")
        if not isinstance(image.get("height_px"), int) or image["height_px"] < 1:
            errors.append(f"{page_id}: invalid image height")
        render = image.get("render")
        if not isinstance(render, dict) or render.get("render_id") != RENDER_ID:
            errors.append(f"{page_id}: render provenance mismatch")
        elif render.get("long_edge_px") != LONG_EDGE_STD or render.get("alpha") is not False:
            errors.append(f"{page_id}: render settings mismatch")
        source = image.get("source")
        if not isinstance(source, dict) or not _valid_sha256(source.get("pdf_sha256")):
            errors.append(f"{page_id}: invalid source PDF hash")
        elif page.get("slug") in book_defs and source["pdf_sha256"] != book_defs[page["slug"]].get("pdf_sha256"):
            errors.append(f"{page_id}: page/book PDF hash mismatch")
    for slug, book in book_defs.items():
        if not _valid_sha256(book.get("pdf_sha256")):
            errors.append(f"{slug}: invalid PDF hash")
        actual_pages = [p for p in pages if p.get("slug") == slug]
        actual_cases = sum(len(p.get("case_ids", [])) for p in actual_pages)
        if book.get("page_count") != len(actual_pages) or book.get("case_count") != actual_cases:
            errors.append(f"{slug}: declared page/case counts mismatch")
    policy = manifest.get("split_policy")
    if not isinstance(policy, dict) or policy.get("id") != SPLIT_ID or policy.get("unit") != "page":
        errors.append("split policy mismatch")
    else:
        heldout = [c for c in cases if c.get("split") == "heldout"]
        heldout_books = {p["slug"] for p in pages if p.get("split") == "heldout"}
        if len(heldout) < int(policy.get("heldout_min_cases", DEFAULT_HELDOUT_MIN)):
            errors.append("heldout case minimum is not met")
        if len(heldout_books) < int(policy.get("heldout_min_books", DEFAULT_HELDOUT_BOOKS_MIN)):
            errors.append("heldout book minimum is not met")
        if not any(p.get("split") == "dev" for p in pages):
            errors.append("development split is empty")
    return errors


def _parse_audit_time(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_annotations(
    manifest: dict,
    annotations: dict,
    *,
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
    require_audit: bool,
) -> list[str]:
    errors = _validate_manifest(manifest)
    if annotations.get("schema_version") != SCHEMA_VERSION:
        errors.append("annotation schema_version mismatch")
    expected_manifest_sha = manifest_sha256
    if expected_manifest_sha is None and manifest_path is not None:
        expected_manifest_sha = _sha256_file(manifest_path)
    if expected_manifest_sha is None:
        expected_manifest_sha = _sha256_bytes(_json_file_bytes(manifest))
    if annotations.get("manifest_sha256") != expected_manifest_sha:
        errors.append("annotation manifest_sha256 does not match manifest bytes")
    page_defs = {p["page_id"]: p for p in manifest.get("pages", [])}
    case_defs = {c["case_id"]: c for c in manifest.get("cases", [])}
    page_anns = annotations.get("pages")
    target_anns = annotations.get("targets")
    if not isinstance(page_anns, dict) or set(page_anns) != set(page_defs):
        errors.append("annotations.pages keys must exactly match manifest page_ids")
        page_anns = page_anns if isinstance(page_anns, dict) else {}
    if not isinstance(target_anns, dict) or set(target_anns) != set(case_defs):
        errors.append("annotations.targets keys must exactly match manifest case_ids")
        target_anns = target_anns if isinstance(target_anns, dict) else {}

    words_by_page: dict[str, dict[str, tuple[int, int]]] = {}
    for page_id, page_def in page_defs.items():
        ann = page_anns.get(page_id, {})
        if not isinstance(ann, dict):
            errors.append(f"{page_id}: page annotation must be an object")
            ann = {}
        if ann.get("image_sha256") != page_def["image"]["sha256"]:
            errors.append(f"{page_id}: image_sha256 mismatch")
        status = ann.get("status")
        if status not in {"pending", "audited", "unreadable"}:
            errors.append(f"{page_id}: invalid status")
        if require_audit and status == "pending":
            errors.append(f"{page_id}: page audit is pending")
        if status in {"audited", "unreadable"}:
            if not str(ann.get("auditor") or "").strip():
                errors.append(f"{page_id}: auditor is required")
            if not _parse_audit_time(ann.get("audited_at")):
                errors.append(f"{page_id}: audited_at must be an ISO-8601 timestamp")
            if ann.get("ambiguities_resolved") is not True:
                errors.append(f"{page_id}: ambiguities_resolved must be true")
        if status == "unreadable" and not str(ann.get("unreadable_reason") or "").strip():
            errors.append(f"{page_id}: unreadable_reason is required")
        lines = ann.get("lines")
        if not isinstance(lines, list):
            errors.append(f"{page_id}: lines must be a list")
            continue
        seen_lines: set[str] = set()
        seen_words: dict[str, tuple[int, int]] = {}
        line_orders = []
        for line_pos, line in enumerate(lines):
            where = f"{page_id}.lines[{line_pos}]"
            if not isinstance(line, dict):
                errors.append(f"{where}: line must be an object")
                continue
            line_id = line.get("line_id")
            expected_line_id = f"{page_id}:l{line_pos}"
            if line_id != expected_line_id:
                errors.append(f"{where}: line_id must be {expected_line_id}")
            elif line_id in seen_lines:
                errors.append(f"{where}: duplicate line_id {line_id}")
            else:
                seen_lines.add(line_id)
            line_orders.append(line.get("order"))
            _rect(line.get("rect"), where, errors)
            words = line.get("words")
            if not isinstance(words, list):
                errors.append(f"{where}: words must be a list")
                continue
            orders = []
            for word_pos, word in enumerate(words):
                wwhere = f"{where}.words[{word_pos}]"
                if not isinstance(word, dict):
                    errors.append(f"{wwhere}: word must be an object")
                    continue
                word_id = word.get("word_id")
                expected_word_id = f"{expected_line_id}:w{word_pos}"
                if word_id != expected_word_id:
                    errors.append(f"{wwhere}: word_id must be {expected_word_id}")
                elif word_id in seen_words:
                    errors.append(f"{wwhere}: duplicate word_id {word_id}")
                else:
                    seen_words[word_id] = (line_pos, word_pos)
                if not isinstance(word.get("text"), str) or not word["text"].strip():
                    errors.append(f"{wwhere}: non-empty text is required")
                orders.append(word.get("order"))
                word_rect_ok = _rect(word.get("rect"), wwhere, errors)
                if word_rect_ok and isinstance(line.get("rect"), list) and len(line["rect"]) == 4:
                    x0, y0, x1, y1 = word["rect"]
                    lx0, ly0, lx1, ly1 = line["rect"]
                    if x0 < lx0 or y0 < ly0 or x1 > lx1 or y1 > ly1:
                        errors.append(f"{wwhere}: word rect must be inside its line rect")
            if orders != list(range(len(words))):
                errors.append(f"{where}: word order must be contiguous 0..n-1 (RTL reading order)")
        if line_orders != list(range(len(lines))):
            errors.append(f"{page_id}: line order must be contiguous 0..n-1")
        if require_audit and status == "audited" and not seen_words:
            errors.append(f"{page_id}: audited page has no words")
        if status == "unreadable" and lines:
            errors.append(f"{page_id}: unreadable page must not contain geometry")
        words_by_page[page_id] = seen_words

    for case_id, case_def in case_defs.items():
        ann = target_anns.get(case_id, {})
        if not isinstance(ann, dict):
            errors.append(f"{case_id}: target annotation must be an object")
            ann = {}
        status = ann.get("status")
        if status not in {"pending", "audited", "unreadable"}:
            errors.append(f"{case_id}: invalid target status")
        if status == "audited" and page_anns.get(case_def["page_id"], {}).get("status") != "audited":
            errors.append(f"{case_id}: audited target requires an audited page")
        if require_audit and status == "pending":
            errors.append(f"{case_id}: target audit is pending")
        if status == "unreadable":
            if not str(ann.get("note") or "").strip():
                errors.append(f"{case_id}: unreadable target requires a note")
            if ann.get("word_ids"):
                errors.append(f"{case_id}: unreadable target must not have word_ids")
            continue
        if require_audit and ann.get("ambiguity_resolution") != "resolved":
            errors.append(f"{case_id}: ambiguity_resolution must be resolved")
        words = ann.get("word_ids")
        if (
            not isinstance(words, list)
            or any(not isinstance(word_id, str) for word_id in words)
            or len(words) != len(set(words))
        ):
            errors.append(f"{case_id}: word_ids must be a duplicate-free list")
            words = []
        word_index = words_by_page.get(case_def["page_id"], {})
        missing = [word_id for word_id in words if word_id not in word_index]
        if missing:
            errors.append(f"{case_id}: unknown word_ids {missing[:3]}")
        ordered_ids = [
            word_id
            for word_id, _pos in sorted(word_index.items(), key=lambda item: item[1])
        ]
        flat_index = {word_id: i for i, word_id in enumerate(ordered_ids)}
        positions = [flat_index[w] for w in words if w in flat_index]
        if positions != sorted(positions):
            errors.append(f"{case_id}: word_ids must follow page reading order")
        if positions and positions != list(range(positions[0], positions[0] + len(positions))):
            errors.append(f"{case_id}: target word_ids must be contiguous")
        if case_def["kind"] == "insertion":
            anchor = ann.get("insertion_anchor")
            if not isinstance(anchor, dict):
                errors.append(f"{case_id}: insertion_anchor is required")
            else:
                before = anchor.get("before_word_id")
                after = anchor.get("after_word_id")
                anchors = [before, after]
                if any(word_id is not None and not isinstance(word_id, str) for word_id in anchors):
                    errors.append(f"{case_id}: insertion anchor IDs must be strings or null")
                    before = after = None
                    anchors = []
                if require_audit and not any(anchors):
                    errors.append(f"{case_id}: insertion needs at least one fixed anchor word")
                for word_id in anchors:
                    if word_id is not None and word_id not in word_index:
                        errors.append(f"{case_id}: unknown insertion anchor {word_id}")
                start = flat_index[before] + 1 if before in flat_index else 0
                stop = flat_index[after] if after in flat_index else len(ordered_ids)
                expected = ordered_ids[start:stop] if start <= stop else None
                if status == "audited" and words != expected:
                    errors.append(f"{case_id}: insertion target must be exactly between ordered anchors; boundary anchors must be adjacent")
        elif require_audit and status == "audited" and not words:
            errors.append(f"{case_id}: audited non-insertion target needs word_ids")
        elif ann.get("insertion_anchor") is not None:
            errors.append(f"{case_id}: non-insertion target must not have insertion_anchor")
    for page_id, page_ann in page_anns.items():
        if isinstance(page_ann, dict) and page_ann.get("status") == "unreadable":
            for case_id in page_defs[page_id].get("case_ids", []):
                if target_anns.get(case_id, {}).get("status") != "unreadable":
                    errors.append(f"{case_id}: target on unreadable page must be unreadable")
    return errors


def _integrity(manifest: dict, annotations: dict) -> dict:
    pages_population = [
        {
            key: page[key]
            for key in ("page_id", "slug", "page", "split", "markdown_sha256", "case_ids")
        }
        for page in manifest["pages"]
    ]
    population = {
        "schema_version": manifest["schema_version"],
        "split_policy": manifest["split_policy"],
        "books": manifest["books"],
        "pages": pages_population,
        "cases": manifest["cases"],
    }
    image_set = [
        {"page_id": page["page_id"], "image": page["image"]}
        for page in manifest["pages"]
    ]
    result = {
        "validator": "tests/bbox_benchmark.py:v1",
        "validated": True,
        "complete": True,
        "population_sha256": _sha256_bytes(_canonical_bytes(population)),
        "image_set_sha256": _sha256_bytes(_canonical_bytes(image_set)),
        "geometry_sha256": _sha256_bytes(_canonical_bytes(annotations["pages"])),
        "targets_sha256": _sha256_bytes(_canonical_bytes(annotations["targets"])),
    }
    result["truth_sha256"] = _sha256_bytes(
        _canonical_bytes(
            {
                key: result[key]
                for key in (
                    "population_sha256",
                    "image_set_sha256",
                    "geometry_sha256",
                    "targets_sha256",
                )
            }
        )
    )
    return result


def validate_frozen(data: dict) -> list[str]:
    """Pure validation entry point for metrics; importing it loads no locator."""
    if not isinstance(data, dict):
        return ["frozen benchmark must be an object"]
    errors: list[str] = []
    manifest = data.get("manifest")
    annotations = data.get("annotations")
    if not isinstance(manifest, dict) or not isinstance(annotations, dict):
        return ["frozen benchmark requires manifest and annotations objects"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("frozen schema_version mismatch")
    manifest_sha = _sha256_bytes(_json_file_bytes(manifest))
    annotation_sha = _sha256_bytes(_json_file_bytes(annotations))
    if data.get("manifest_sha256") != manifest_sha:
        errors.append("frozen manifest_sha256 mismatch")
    if data.get("annotation_sha256") != annotation_sha:
        errors.append("frozen annotation_sha256 mismatch")
    errors.extend(
        validate_annotations(
            manifest,
            annotations,
            manifest_sha256=manifest_sha,
            require_audit=True,
        )
    )
    try:
        expected_integrity = _integrity(manifest, annotations)
    except (KeyError, TypeError):
        errors.append("cannot derive frozen integrity from malformed data")
    else:
        if data.get("integrity") != expected_integrity:
            errors.append("frozen integrity hashes or audit marker mismatch")
    unsigned = {key: value for key, value in data.items() if key != "benchmark_sha256"}
    if data.get("benchmark_sha256") != _sha256_bytes(_canonical_bytes(unsigned)):
        errors.append("frozen benchmark_sha256 mismatch")
    return errors


def import_annotations(manifest_path: Path, annotations_path: Path, out_path: Path) -> None:
    manifest = _load_json(manifest_path)
    annotations = _load_json(annotations_path)
    errors = validate_annotations(
        manifest, annotations, manifest_path=manifest_path, require_audit=False
    )
    if errors:
        raise BenchmarkError("annotation import failed:\n- " + "\n- ".join(errors))
    _write_json(out_path, annotations)


def freeze(manifest_path: Path, annotations_path: Path, out_path: Path, *, books_root: Path = DEFAULT_BOOKS_ROOT) -> dict:
    manifest = _load_json(manifest_path)
    annotations = _load_json(annotations_path)
    errors = validate_annotations(
        manifest, annotations, manifest_path=manifest_path, require_audit=True
    )
    base = manifest_path.parent
    for page in manifest.get("pages", []):
        image_path = base / page["image"]["path"]
        if not image_path.is_file():
            errors.append(f"{page['page_id']}: image file is missing")
        elif _sha256_file(image_path) != page["image"]["sha256"]:
            errors.append(f"{page['page_id']}: image file hash mismatch")
        else:
            pix = fitz.Pixmap(str(image_path))
            if (pix.width, pix.height) != (
                page["image"]["width_px"], page["image"]["height_px"]
            ):
                errors.append(f"{page['page_id']}: image dimensions mismatch")
    for book in manifest.get("books", []):
        pdf = books_root / book["slug"] / "source.pdf"
        if not pdf.is_file() or _sha256_file(pdf) != book["pdf_sha256"]:
            errors.append(f"{book['slug']}: source PDF is missing or changed")
    case_defs = {c["case_id"]: c for c in manifest.get("cases", [])}
    for page in manifest.get("pages", []):
        workspace_root = books_root / page["slug"]
        md_path = workspace_root / "text" / f"{page['page']:04d}.md"
        if not md_path.is_file():
            errors.append(f"{page['page_id']}: source Markdown is missing")
            continue
        markdown = md_path.read_text(encoding="utf-8")
        if _sha256_bytes(markdown.encode("utf-8")) != page["markdown_sha256"]:
            errors.append(f"{page['page_id']}: source Markdown changed after export")
            continue
        pdf_path = workspace_root / "source.pdf"
        if pdf_path.is_file():
            doc = fitz.open(str(pdf_path))
            try:
                pdf_page = doc[page["page"] - 1]
                rect = pdf_page.rect
                source = page["image"]["source"]
                if (
                    abs(rect.width - source["page_width_pt"]) > 1e-6
                    or abs(rect.height - source["page_height_pt"]) > 1e-6
                ):
                    errors.append(f"{page['page_id']}: source PDF page dimensions changed")
                render = page["image"]["render"]
                zoom = LONG_EDGE_STD / max(rect.width, rect.height) if max(rect.width, rect.height) else 1.0
                if abs(float(render.get("zoom", -1)) - zoom) > 1e-12:
                    errors.append(f"{page['page_id']}: render zoom provenance mismatch")
                else:
                    rendered = pdf_page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom), alpha=False
                    ).tobytes("png")
                    if _sha256_bytes(rendered) != page["image"]["sha256"]:
                        errors.append(
                            f"{page['page_id']}: clean PNG is not the declared source-page render"
                        )
            finally:
                doc.close()
        for case_id in page["case_ids"]:
            hunk = case_defs[case_id]["hunk"]
            span = hunk["markdown_span"]
            if span is not None and markdown[span[0] : span[1]] != hunk["old"]:
                errors.append(f"{case_id}: full hunk span no longer matches source Markdown")
            expected_candidates = _all_spans(markdown, hunk["old"])
            if not hunk["old"] and span is not None:
                expected_candidates = [span]
            if hunk["candidate_spans"] != expected_candidates:
                errors.append(f"{case_id}: hunk candidate spans changed")
    heldout = [c for c in manifest.get("cases", []) if c.get("split") == "heldout"
               and annotations.get("targets", {}).get(c["case_id"], {}).get("status") == "audited"
               and annotations.get("pages", {}).get(c["page_id"], {}).get("status") == "audited"]
    heldout_books = {
        p["slug"] for p in manifest.get("pages", []) if p["page_id"] in {c["page_id"] for c in heldout}
    }
    policy = manifest.get("split_policy", {})
    if len(heldout) < int(policy.get("heldout_min_cases", DEFAULT_HELDOUT_MIN)):
        errors.append("heldout case minimum is not met")
    if len(heldout_books) < int(policy.get("heldout_min_books", DEFAULT_HELDOUT_BOOKS_MIN)):
        errors.append("heldout book minimum is not met")
    page_splits = {p["page_id"]: p["split"] for p in manifest.get("pages", [])}
    if any(c.get("split") != page_splits.get(c.get("page_id")) for c in manifest.get("cases", [])):
        errors.append("case split differs from its page split")
    if errors:
        raise BenchmarkError("freeze refused:\n- " + "\n- ".join(errors))
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _sha256_file(manifest_path),
        "annotation_sha256": _sha256_bytes(_json_file_bytes(annotations)),
        "manifest": manifest,
        "annotations": annotations,
        "integrity": _integrity(manifest, annotations),
    }
    frozen["benchmark_sha256"] = _sha256_bytes(_canonical_bytes(frozen))
    frozen_errors = validate_frozen(frozen)
    if frozen_errors:
        raise BenchmarkError("internal frozen validation failed:\n- " + "\n- ".join(frozen_errors))
    _write_json(out_path, frozen)
    return {
        **summarize(manifest),
        "benchmark_sha256": frozen["benchmark_sha256"],
        "unreadable_pages": sum(
            p.get("status") == "unreadable" for p in annotations["pages"].values()
        ),
        "unreadable_cases": sum(
            t.get("status") == "unreadable" for t in annotations["targets"].values()
        ),
    }


def _annotation_html(manifest: dict, annotations: dict) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    annotations_json = json.dumps(annotations, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Bbox benchmark annotation</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;font:14px system-ui;background:#181818;color:#eee}}
header{{position:sticky;top:0;z-index:5;background:#222;padding:10px;display:flex;gap:8px;align-items:center}}
button,input,select{{font:inherit;padding:6px}}main{{display:grid;grid-template-columns:minmax(420px,2fr) minmax(340px,1fr);height:calc(100vh - 55px)}}
#stage{{position:relative;overflow:auto;background:#333;text-align:center}}#page{{max-width:100%;display:block;margin:auto;user-select:none}}
#overlay{{position:absolute;left:0;top:0;pointer-events:auto}}.rect{{position:absolute;border:2px solid #31c6ff;background:#31c6ff22;pointer-events:auto}}.word{{border-color:#ffd166;background:#ffd16622}}
.selected{{background:#ff5d8f66}}aside{{overflow:auto;padding:14px}}textarea{{width:100%;height:80px}}.case{{border-top:1px solid #555;padding:10px 0}}.muted{{color:#aaa}}label{{display:block;margin:6px 0}}
</style></head><body><header>
<button id="prev">←</button><select id="pages"></select><button id="next">→</button>
<button id="lineMode">Draw line</button><button id="wordMode">Draw word</button>
<button id="undoWord">Undo word</button><button id="undoLine">Undo line</button>
<button id="download">Download annotations.json</button><input id="load" type="file" accept="application/json">
<span id="status"></span></header><main><div id="stage"><img id="page"><div id="overlay"></div></div><aside>
<h3 id="title"></h3><label>Status <select id="pageStatus"><option>pending</option><option>audited</option><option>unreadable</option></select></label>
<label>Auditor <input id="auditor"></label><label>Audited at <input id="auditedAt" placeholder="2026-09-04T12:00:00Z"></label>
<label><input id="resolved" type="checkbox"> Ambiguities resolved</label><label>Unreadable reason <input id="unreadable"></label>
<p class="muted">Draw lines in top-to-bottom reading order. Select a line, then draw its words in RTL reading order. Click words to assign them to the active case.</p>
<div id="cases"></div></aside></main>
<script>const manifest={manifest_json};let ann={annotations_json};
const byPage=Object.fromEntries(manifest.pages.map(p=>[p.page_id,p]));let initialized=false,idx=0,mode='line',activeLine=null,activeCase=null,lastWordId=null,drag=null;
const el=id=>document.getElementById(id), page=el('page'),overlay=el('overlay');
manifest.pages.forEach((p,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{p.split}} · ${{p.slug}} p${{p.page}} · ${{p.case_ids.length}} cases`;el('pages').appendChild(o)}});
function saveFields(){{let p=manifest.pages[idx],a=ann.pages[p.page_id];a.status=el('pageStatus').value;a.auditor=el('auditor').value;a.audited_at=el('auditedAt').value;a.ambiguities_resolved=el('resolved').checked;a.unreadable_reason=el('unreadable').value}}
function loadPage(n){{if(initialized)saveFields();initialized=true;idx=Math.max(0,Math.min(manifest.pages.length-1,n));let p=manifest.pages[idx],a=ann.pages[p.page_id];el('pages').value=idx;page.src=p.image.path;el('title').textContent=`${{p.slug}} page ${{p.page}} (${{p.split}})`;el('pageStatus').value=a.status;el('auditor').value=a.auditor;el('auditedAt').value=a.audited_at;el('resolved').checked=a.ambiguities_resolved;el('unreadable').value=a.unreadable_reason;activeLine=null;activeCase=p.case_ids[0]||null;lastWordId=null;renderCases();renderRects()}}
function rectStyle(r){{return `left:${{r[0]*100}}%;top:${{r[1]*100}}%;width:${{(r[2]-r[0])*100}}%;height:${{(r[3]-r[1])*100}}%`}}
function renderRects(){{overlay.innerHTML='';overlay.style.left=page.offsetLeft+'px';overlay.style.top=page.offsetTop+'px';overlay.style.width=page.clientWidth+'px';overlay.style.height=page.clientHeight+'px';let p=manifest.pages[idx],a=ann.pages[p.page_id];a.lines.forEach((l,li)=>{{let d=document.createElement('div');d.className='rect';d.dataset.line=li;d.style.cssText=rectStyle(l.rect);d.title=l.line_id;d.onclick=e=>{{e.stopPropagation();activeLine=li;mode='word';renderRects()}};overlay.appendChild(d);l.words.forEach((w,wi)=>{{let t=activeCase&&ann.targets[activeCase],chosen=t&&(t.word_ids.includes(w.word_id)||(t.insertion_anchor&&(t.insertion_anchor.before_word_id===w.word_id||t.insertion_anchor.after_word_id===w.word_id))||lastWordId===w.word_id),x=document.createElement('div');x.className='rect word'+(chosen?' selected':'');x.style.cssText=rectStyle(w.rect);x.title=w.word_id+' '+w.text;x.onclick=e=>{{e.stopPropagation();activeLine=li;if(activeCase){{let c=manifest.cases.find(x=>x.case_id===activeCase),t=ann.targets[activeCase];if(c.kind==='insertion')lastWordId=w.word_id;else{{let ids=t.word_ids,k=ids.indexOf(w.word_id);k<0?ids.push(w.word_id):ids.splice(k,1);const order=a.lines.flatMap(l=>l.words.map(w=>w.word_id));ids.sort((a,b)=>order.indexOf(a)-order.indexOf(b))}}renderRects();renderCases()}}}};overlay.appendChild(x)}})}})}}
function renderCases(){{let p=manifest.pages[idx],box=el('cases');box.innerHTML='';p.case_ids.forEach(cid=>{{let c=manifest.cases.find(x=>x.case_id===cid),t=ann.targets[cid],d=document.createElement('div');d.className='case';d.dir='rtl';d.innerHTML=`<button data-c="${{cid}}">Select target</button> <b>${{c.kind}}</b><div>${{escapeHtml(c.hunk.old)}} → ${{escapeHtml(c.hunk.new)}}</div><div class="muted">Before: ${{escapeHtml(c.hunk.ctx_before)}} · After: ${{escapeHtml(c.hunk.ctx_after)}}</div><label>Status <select data-status="${{cid}}"><option>pending</option><option>audited</option><option>unreadable</option></select></label><label>Ambiguity <select data-amb="${{cid}}"><option>pending</option><option>resolved</option></select></label><label>Note <input data-note="${{cid}}"></label><div class="muted">Target IDs: ${{t.word_ids.join(', ')||'none'}}${{t.insertion_anchor?' · anchors '+JSON.stringify(t.insertion_anchor):''}}</div>${{c.kind==='insertion'?'<button data-before="'+cid+'">Set preceding anchor from highlighted word</button> <button data-after="'+cid+'">Set following anchor from highlighted word</button> <button data-targetword="'+cid+'">Toggle highlighted printed target word</button>':''}}`;box.appendChild(d);d.querySelector('[data-c]').onclick=()=>{{activeCase=cid;lastWordId=null;renderRects();renderCases()}};let s=d.querySelector('[data-status]');s.value=t.status;s.onchange=()=>t.status=s.value;let a=d.querySelector('[data-amb]');a.value=t.ambiguity_resolution;a.onchange=()=>t.ambiguity_resolution=a.value;let note=d.querySelector('[data-note]');note.value=t.note;note.oninput=()=>t.note=note.value;let tw=d.querySelector('[data-targetword]');if(tw)tw.onclick=()=>{{if(!lastWordId)return alert('Click one word first.');let k=t.word_ids.indexOf(lastWordId);k<0?t.word_ids.push(lastWordId):t.word_ids.splice(k,1);let order=ann.pages[p.page_id].lines.flatMap(l=>l.words.map(w=>w.word_id));t.word_ids.sort((a,b)=>order.indexOf(a)-order.indexOf(b));renderRects();renderCases()}};d.querySelectorAll('[data-before],[data-after]').forEach(b=>b.onclick=()=>{{if(!lastWordId)return alert('Click one word first.');if(b.dataset.before)t.insertion_anchor.before_word_id=lastWordId;else t.insertion_anchor.after_word_id=lastWordId;lastWordId=null;renderRects();renderCases()}})}})}}
function escapeHtml(s){{let d=document.createElement('div');d.textContent=s;return d.innerHTML}}
overlay.onpointerdown=e=>{{if(e.target!==overlay && !(mode==='word' && e.target.classList.contains('rect')&&!e.target.classList.contains('word')))return;if(e.target.dataset.line!==undefined)activeLine=+e.target.dataset.line;let r=overlay.getBoundingClientRect();drag={{x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height}}}};
overlay.onpointerup=e=>{{if(!drag)return;let r=overlay.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height,box=[Math.max(0,Math.min(drag.x,x)),Math.max(0,Math.min(drag.y,y)),Math.min(1,Math.max(drag.x,x)),Math.min(1,Math.max(drag.y,y))],p=manifest.pages[idx],a=ann.pages[p.page_id];drag=null;if(box[2]-box[0]<.002||box[3]-box[1]<.002)return;if(mode==='line'){{let n=a.lines.length;a.lines.push({{line_id:`${{p.page_id}}:l${{n}}`,order:n,rect:box,words:[]}});activeLine=n}}else if(activeLine!==null){{let l=a.lines[activeLine],n=l.words.length,txt=prompt('Printed word text (RTL reading order):','');if(txt)l.words.push({{word_id:`${{l.line_id}}:w${{n}}`,order:n,text:txt,rect:box}})}}renderRects()}};
page.onload=()=>renderRects();window.onresize=()=>renderRects();el('prev').onclick=()=>loadPage(idx-1);el('next').onclick=()=>loadPage(idx+1);el('pages').onchange=e=>loadPage(+e.target.value);el('lineMode').onclick=()=>mode='line';el('wordMode').onclick=()=>mode='word';el('undoWord').onclick=()=>{{let a=ann.pages[manifest.pages[idx].page_id];if(activeLine!==null&&a.lines[activeLine])a.lines[activeLine].words.pop();renderRects()}};el('undoLine').onclick=()=>{{let a=ann.pages[manifest.pages[idx].page_id];a.lines.pop();activeLine=a.lines.length? a.lines.length-1:null;renderRects()}};
el('download').onclick=()=>{{saveFields();let b=new Blob([JSON.stringify(ann,null,2)+'\\n'],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='annotations.json';a.click();URL.revokeObjectURL(a.href)}};
el('load').onchange=async e=>{{ann=JSON.parse(await e.target.files[0].text());initialized=false;loadPage(idx)}};loadPage(0);</script></body></html>"""


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="export clean pages and annotation templates")
    ex.add_argument("--out-dir", type=Path, required=True)
    ex.add_argument("--books-root", type=Path, default=DEFAULT_BOOKS_ROOT)
    ex.add_argument("--book", action="append", dest="books")
    ex.add_argument("--heldout-min", type=int, default=DEFAULT_HELDOUT_MIN)
    ex.add_argument("--heldout-books-min", type=int, default=DEFAULT_HELDOUT_BOOKS_MIN)
    imp = sub.add_parser("import-annotations", help="validate and normalize an editor download")
    imp.add_argument("--manifest", type=Path, required=True)
    imp.add_argument("--annotations", type=Path, required=True)
    imp.add_argument("--out", type=Path, required=True)
    fr = sub.add_parser("freeze", help="freeze only fully audited annotations")
    fr.add_argument("--manifest", type=Path, required=True)
    fr.add_argument("--annotations", type=Path, required=True)
    fr.add_argument("--out", type=Path, required=True)
    fr.add_argument("--books-root", type=Path, default=DEFAULT_BOOKS_ROOT)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            result = export_benchmark(
                args.out_dir,
                books_root=args.books_root,
                books=args.books,
                heldout_min=args.heldout_min,
                heldout_books_min=args.heldout_books_min,
            )
        elif args.command == "import-annotations":
            import_annotations(args.manifest, args.annotations, args.out)
            result = {"imported": str(args.out)}
        else:
            result = freeze(args.manifest, args.annotations, args.out, books_root=args.books_root)
    except BenchmarkError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
