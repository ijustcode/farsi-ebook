#!/usr/bin/env python3
"""Box-audit packet: drafted word rectangles for real pages, and a local tool
in which the owner checks and fixes every rectangle (plan §6.2, real truth).

Evaluation tooling only; production code never imports ``tests/``. No API
calls, no new dependencies (numpy + PyMuPDF), no network beyond 127.0.0.1.

Commands (run from the repo root with ./venv/bin/python):

    tests/box_audit.py serve [--port 8770] [--open]     # the audit tool
    tests/box_audit.py status                            # progress per page
    tests/box_audit.py draft --slug S --page N [--force] # write NNNN.geometry.draft.json
    tests/box_audit.py import-pilot [--force]            # bachehaye_ghali p66 fixture
    tests/box_audit.py overlay --key S/N [--which draft|audit] --out PNG
    tests/box_audit.py export --key S/N                  # audited page -> workflow lines
    tests/box_audit.py synth-eval [--seeds 3 4 5 6]      # draft accuracy on synthetic pages

Files, all under ``books/<slug>/truth/`` (private, gitignored):

    NNNN.geometry.draft.json      untouched machine draft (never rewritten by the tool)
    NNNN.geometry.audit.json      the owner's audited version (created from the draft on first open)
    NNNN.geometry.audit.undo.jsonl  undo stack (one record per action)
    NNNN.page.png                 audit image, when rendered from the PDF at native resolution

Data contract (``SCHEMA_VERSION``): the line/word record shape of
``bbox_annotation_workflow`` (``line_id = <page_id>:l<i>``, ``word_id =
<line_id>:w<j>``, ``order``, ``text``, normalised ``rect``) extended with what
sloped photographed lines need: each line has a *frame* (centre and angle in
image pixels); line and word ``box`` values are axis-aligned rectangles in that
frame, so every word rectangle follows its line's slope. ``quad`` (TL, TR, BR,
BL; normalised x right, y down) and ``rect`` (its normalised bounds) are derived
from frame + box on every save. Word ``state`` is ``draft``, ``accepted`` or
``fixed``; line ``state`` is ``draft``, ``editing``, ``accepted`` or ``fixed``.
A page is ``audited`` only when every line was explicitly accepted (after any
fixes) with as many boxes as truth words (or a flagged text problem).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import box_audit_draft as bd  # noqa: E402

SCHEMA_VERSION = "farsi2epub.box-audit/v1"
TOOL_VERSION = "box_audit/1"
PILOT_FIXTURE = ROOT / "tests" / "data" / "bbox_scan66_development_pilot.json"
FONT = ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"
DEFAULT_PORT = 8770

# Held out for the final check (plan §6.2): the tool refuses to draft them.
HELD_OUT = {("hadpi-articles", 3), ("hadpi-articles", 4), ("hadpi-articles", 5)}

WORD_STATES = {"draft", "accepted", "fixed"}
LINE_STATES = {"draft", "editing", "accepted", "fixed"}
PAGE_STATES = {"draft", "in_progress", "audited", "previously_audited"}
DONE = {"accepted", "fixed"}


class AuditError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def page_id_for(slug: str, page: int, pdf_sha: str) -> str:
    """Same identity as tests/bbox_benchmark.py ``_id("p", slug, page, pdf_sha)``."""
    return "p_" + hashlib.sha256(canonical([slug, page, pdf_sha])).hexdigest()[:20]


def rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Geometry: line frames
# ---------------------------------------------------------------------------
def frame_to_page(frame: dict, u: float, v: float) -> tuple[float, float]:
    a = math.radians(frame["angle_deg"])
    c, s = math.cos(a), math.sin(a)
    return frame["cx"] + u * c - v * s, frame["cy"] + u * s + v * c


def page_to_frame(frame: dict, x: float, y: float) -> tuple[float, float]:
    a = math.radians(frame["angle_deg"])
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - frame["cx"], y - frame["cy"]
    return dx * c + dy * s, -dx * s + dy * c


def box_quad_px(frame: dict, box) -> list[tuple[float, float]]:
    u0, v0, u1, v1 = box
    return [frame_to_page(frame, u0, v0), frame_to_page(frame, u1, v0),
            frame_to_page(frame, u1, v1), frame_to_page(frame, u0, v1)]


def quad_to_frame_box(frame: dict, pts) -> list[float]:
    fr = [page_to_frame(frame, x, y) for x, y in pts]
    return [min(p[0] for p in fr), min(p[1] for p in fr), max(p[0] for p in fr), max(p[1] for p in fr)]


def _norm_quad(pts, W, H):
    return [[round(min(1.0, max(0.0, x / W)), 6), round(min(1.0, max(0.0, y / H)), 6)] for x, y in pts]


def _rect_of(q):
    xs = [p[0] for p in q]
    ys = [p[1] for p in q]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_box(boxes, pad: float = 0.0):
    return [min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
            max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad]


def fix_box(box, min_side: float = 1.0) -> list[float]:
    u0, v0, u1, v1 = [float(x) for x in box]
    if u1 < u0:
        u0, u1 = u1, u0
    if v1 < v0:
        v0, v1 = v1, v0
    if u1 - u0 < min_side:
        u1 = u0 + min_side
    if v1 - v0 < min_side:
        v1 = v0 + min_side
    return [round(u0, 2), round(v0, 2), round(u1, 2), round(v1, 2)]


# ---------------------------------------------------------------------------
# Page documents
# ---------------------------------------------------------------------------
def truth_dir(root: Path, slug: str) -> Path:
    return root / "books" / slug / "truth"


def draft_path(root: Path, slug: str, page: int) -> Path:
    return truth_dir(root, slug) / f"{page:04d}.geometry.draft.json"


def audit_path(root: Path, slug: str, page: int) -> Path:
    return truth_dir(root, slug) / f"{page:04d}.geometry.audit.json"


def undo_path(root: Path, slug: str, page: int) -> Path:
    return truth_dir(root, slug) / f"{page:04d}.geometry.audit.undo.jsonl"


def derive(doc: dict) -> dict:
    """Recompute ids, orders, text, quads and rects from frames + boxes, keep
    every word box inside its line box, and recompute the page status."""
    W, H = doc["image"]["width_px"], doc["image"]["height_px"]
    pid = doc["page_id"]
    for li, line in enumerate(doc["lines"]):
        line["order"] = li
        line["line_id"] = f"{pid}:l{li}"
        line["text"] = " ".join(line["tokens"])
        line["box"] = fix_box(line["box"])
        for wi, w in enumerate(line["words"]):
            w["order"] = wi
            w["word_id"] = f"{line['line_id']}:w{wi}"
            w["text"] = line["tokens"][wi] if wi < len(line["tokens"]) else None
            w["box"] = fix_box(w["box"])
            w["quad"] = _norm_quad(box_quad_px(line["frame"], w["box"]), W, H)
            w["rect"] = _rect_of(w["quad"])
        if line["words"]:
            u = union_box([w["box"] for w in line["words"]])
            b = line["box"]
            line["box"] = fix_box([min(b[0], u[0]), min(b[1], u[1]), max(b[2], u[2]), max(b[3], u[3])])
        line["quad"] = _norm_quad(box_quad_px(line["frame"], line["box"]), W, H)
        line["rect"] = _rect_of(line["quad"])
        line["count_ok"] = len(line["words"]) == len(line["tokens"])
    if doc.get("kind") == "audit":
        complete = bool(doc["lines"]) and all(
            l["state"] in DONE and (l["count_ok"] or l.get("text_flag")) for l in doc["lines"])
        if complete:
            if doc["status"] not in ("previously_audited", "audited"):
                doc["status"] = "audited"
                doc["audited_at"] = now_iso()
        else:
            touched = doc.get("revision", 0) > 0 or any(l["state"] != "draft" for l in doc["lines"])
            doc["status"] = "in_progress" if touched else "draft"
            doc["audited_at"] = None
    return doc


def page_summary(doc: dict) -> dict:
    lines = doc["lines"]
    return dict(
        key=f"{doc['slug']}/{doc['page']}",
        slug=doc["slug"], page=doc["page"], status=doc["status"],
        lines=len(lines), done=sum(1 for l in lines if l["state"] in DONE),
        uncertain=sum(1 for l in lines if l.get("uncertain")),
        uncertain_open=sum(1 for l in lines if l.get("uncertain") and l["state"] not in DONE),
        words=sum(len(l["words"]) for l in lines),
        text_flags=sum(1 for l in lines if l.get("text_flag")),
        label=doc.get("label") or f"{doc['slug']} p{doc['page']}",
    )


def validate_page(doc: dict) -> list[str]:
    """Data-contract check for draft and audit documents."""
    e: list[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        e.append("schema_version mismatch")
    if doc.get("kind") not in ("draft", "audit"):
        e.append("kind must be draft or audit")
    for k in ("page_id", "slug", "page", "image", "text_source", "tool_version", "lines", "status"):
        if k not in doc:
            e.append(f"missing {k}")
    if e:
        return e
    if doc["status"] not in PAGE_STATES:
        e.append("bad page status")
    img = doc["image"]
    for k in ("path", "sha256", "width_px", "height_px"):
        if k not in img:
            e.append(f"image.{k} missing")
    ts = doc["text_source"]
    for k in ("path", "sha256", "kind"):
        if k not in ts:
            e.append(f"text_source.{k} missing")
    W, H = img.get("width_px", 1), img.get("height_px", 1)
    pid = doc["page_id"]
    for li, line in enumerate(doc["lines"]):
        where = f"lines[{li}]"
        if line.get("line_id") != f"{pid}:l{li}" or line.get("order") != li:
            e.append(f"{where}: line_id/order must be {pid}:l{li}/{li}")
        if line.get("state") not in LINE_STATES:
            e.append(f"{where}: bad state")
        if line.get("tokens") != line.get("text", "").split():
            e.append(f"{where}: tokens must be the whitespace split of text")
        fr = line.get("frame") or {}
        if not all(isinstance(fr.get(k), (int, float)) for k in ("cx", "cy", "angle_deg")):
            e.append(f"{where}: frame needs cx, cy, angle_deg")
            continue
        lb = line.get("box")
        q = _norm_quad(box_quad_px(fr, lb), W, H)
        if any(abs(a - b) > 1e-5 for p, r in zip(q, line.get("quad", [])) for a, b in zip(p, r)):
            e.append(f"{where}: quad does not match frame+box")
        r = line.get("rect", [])
        if not (len(r) == 4 and 0 <= r[0] <= r[2] <= 1 and 0 <= r[1] <= r[3] <= 1):
            e.append(f"{where}: rect must be normalised")
        for wi, w in enumerate(line.get("words", [])):
            ww = f"{where}.words[{wi}]"
            if w.get("word_id") != f"{line.get('line_id')}:w{wi}" or w.get("order") != wi:
                e.append(f"{ww}: word_id/order")
            if w.get("state") not in WORD_STATES:
                e.append(f"{ww}: bad state")
            exp_text = line["tokens"][wi] if wi < len(line["tokens"]) else None
            if w.get("text") != exp_text:
                e.append(f"{ww}: text must be token {wi} of the line")
            b = w.get("box")
            if not (isinstance(b, list) and len(b) == 4 and b[0] < b[2] and b[1] < b[3]):
                e.append(f"{ww}: bad box")
                continue
            if b[0] < lb[0] - 0.01 or b[1] < lb[1] - 0.01 or b[2] > lb[2] + 0.01 or b[3] > lb[3] + 0.01:
                e.append(f"{ww}: word box outside line box")
            q = _norm_quad(box_quad_px(fr, b), W, H)
            if any(abs(a - c) > 1e-5 for p, rr in zip(q, w.get("quad", [])) for a, c in zip(p, rr)):
                e.append(f"{ww}: quad does not match frame+box")
            if "uncertain" not in w:
                e.append(f"{ww}: uncertain flag missing")
    if doc["status"] == "audited":
        for li, line in enumerate(doc["lines"]):
            if line["state"] not in DONE:
                e.append(f"audited page has an unreviewed line {li}")
            if not line["count_ok"] and not line.get("text_flag"):
                e.append(f"audited page line {li}: box count differs from word count")
    return e


def to_workflow_lines(doc: dict) -> list[dict]:
    """The page in ``bbox_annotation_workflow``'s line/word shape (axis-aligned
    normalised rects; the quads are the precise geometry)."""
    out = []
    for line in doc["lines"]:
        words = []
        for w in line["words"]:
            words.append(dict(word_id=w["word_id"], order=w["order"], text=w["text"] or "", rect=w["rect"]))
        out.append(dict(line_id=line["line_id"], order=line["order"], rect=line["rect"], words=words))
    return out


# ---------------------------------------------------------------------------
# Building drafts
# ---------------------------------------------------------------------------
def _base_doc(kind, slug, page, pdf_sha, image_path: Path, image_meta: dict, text_path: Path,
              text_kind: str, root: Path) -> dict:
    pm = fitz.Pixmap(str(image_path))
    return dict(
        schema_version=SCHEMA_VERSION, tool_version=TOOL_VERSION, kind=kind,
        page_id=page_id_for(slug, page, pdf_sha), slug=slug, page=page,
        source_pdf_sha256=pdf_sha,
        image=dict(path=rel(image_path, root), sha256=file_sha(image_path), width_px=pm.width,
                   height_px=pm.height, **image_meta),
        text_source=dict(path=rel(text_path, root), sha256=file_sha(text_path), kind=text_kind),
        coords=("Line frame: centre (cx, cy) and angle in image pixels, angle clockwise (y down). "
                "box = [u0, v0, u1, v1] in the line frame (u along the line, v down). quad = TL, TR, BR, BL "
                "normalised to the image (x right, y down); rect = bounds of quad."),
        status="draft", created_at=now_iso(), lines=[], regions=[],
    )


def audit_image_for(root: Path, slug: str, page: int) -> tuple[Path, dict]:
    """Highest-resolution page-frame image: the workspace hi-res render, or a
    render of the full PDF page at the embedded scan's native resolution when
    that is finer. Coordinates are normalised to the full page either way."""
    ws = root / "books" / slug
    hires = ws / "pages" / "hires" / f"{page:04d}.png"
    doc = fitz.open(str(ws / "source.pdf"))
    pg = doc[page - 1]
    native = 0.0
    for im in pg.get_images(full=True):
        for r in pg.get_image_rects(im[0]):
            if r.width > 0.3 * pg.rect.width:
                native = max(native, im[2] / r.width, im[3] / r.height)
    hz = 0.0
    if hires.exists():
        pm = fitz.Pixmap(str(hires))
        hz = pm.width / pg.rect.width
    meta = dict(frame="full PDF page", native_px_per_pt=round(native, 4))
    if hires.exists():
        meta["workspace_image"] = dict(path=rel(hires, root), sha256=file_sha(hires))
    if native > hz * 1.05:
        out = truth_dir(root, slug) / f"{page:04d}.page.png"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            pg.get_pixmap(matrix=fitz.Matrix(native, native)).save(str(out))
        meta["zoom"] = round(native, 4)
        meta["rendered_from"] = "source.pdf at the embedded image's native resolution"
        return out, meta
    meta["zoom"] = round(hz, 4)
    meta["rendered_from"] = "workspace hi-res render (finer than the embedded image)"
    return hires, meta


def _line_record(i, text, track, words_px, word_reasons, reasons, notes, H, pad=3.0) -> dict:
    """Convert one drafted line (track + per-word [x0, x1, dt, db]) to a frame record."""
    xs, ys = np.asarray(track.xs, float), np.asarray(track.ys, float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0
    x0 = min(w[0] for w in words_px)
    x1 = max(w[1] for w in words_px)
    cx = (x0 + x1) / 2
    frame = dict(cx=round(cx, 2), cy=round(float(track.y_at(cx)), 2), angle_deg=round(math.degrees(math.atan(slope)), 4))
    words = []
    for k, (a, b, dt_, db_) in enumerate(words_px):
        pts = [(a, track.y_at(a) + dt_), (b, track.y_at(b) + dt_), (b, track.y_at(b) + db_), (a, track.y_at(a) + db_)]
        wr = list(word_reasons[k])
        words.append(dict(box=fix_box(quad_to_frame_box(frame, pts)), state="draft",
                          uncertain=bool(wr), reasons=wr))
    lbox = union_box([w["box"] for w in words], pad)
    return dict(tokens=text.split(), frame=frame, box=lbox, words=words, state="draft",
                line_fixed=False, uncertain=bool(reasons), reasons=list(reasons), notes=list(notes),
                text_flag=None)


def known_line_texts(path: Path) -> tuple[list[str], list[bool]]:
    lines, gap = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            lines.append(" ".join(raw.split()))
            gap.append(False)
        elif gap:
            gap[-1] = True
    return lines, gap


def draft_page(root: Path, slug: str, page: int, force: bool = False) -> Path:
    if (slug, page) in HELD_OUT:
        raise AuditError(f"{slug} p{page} is held out (plan §6.2); it must not be drafted or tuned on")
    out = draft_path(root, slug, page)
    if out.exists() and not force:
        raise AuditError(f"{out} exists; pass --force to replace the untouched draft")
    ws = root / "books" / slug
    pdf_sha = file_sha(ws / "source.pdf")
    image_path, meta = audit_image_for(root, slug, page)
    gray = bd.load_gray(image_path)
    H, W = gray.shape
    lines_txt = truth_dir(root, slug) / f"{page:04d}.lines.txt"
    region = (int(0.05 * W), int(0.03 * H), int(0.97 * W), int(0.92 * H))
    if lines_txt.exists():
        texts, gap = known_line_texts(lines_txt)
        res = bd.draft_known_lines(gray, texts, gap, region=region)
        doc = _base_doc("draft", slug, page, pdf_sha, image_path, meta, lines_txt, "owner_text_truth_lines", root)
        method = "known printed lines (owner text truth): track/strip line finder, line-to-text DP, word-gap DP"
    else:
        md = ws / "text" / f"{page:04d}.md"
        if not md.exists():
            raise AuditError(f"no text source for {slug} p{page}")
        region = (int(0.08 * W), int(0.03 * H), int(0.97 * W), int(0.95 * H))
        res = bd.draft_unbroken(gray, md.read_text(encoding="utf-8"), region=region)
        doc = _base_doc("draft", slug, page, pdf_sha, image_path, meta, md, "transcription_markdown", root)
        method = ("transcription Markdown (not line-broken): every detected line in the text block, "
                  "words distributed by ink width + word-gap fit (paragraph ends forced), word-gap DP")
        for r in res.get("regions", []):
            x0, y0, x1, y1 = r["rect_px"]
            doc["regions"].append(dict(region_id=f"{doc['page_id']}:r{len(doc['regions'])}", kind=r["kind"],
                                       rect=[x0 / W, y0 / H, x1 / W, y1 / H],
                                       note="not in the transcription; not a graded text line"))
    for i, L in enumerate(res["lines"]):
        doc["lines"].append(_line_record(i, L["text"], L["track"], L["words"], L["word_reasons"],
                                         L["reasons"], L["notes"], H))
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res["info"].items()
            if k in ("spacing", "pitch", "alpha", "gap_threshold", "zwnj_gap", "candidates", "paragraphs")}
    doc["draft"] = dict(method=method, draft_version=bd.DRAFT_VERSION, params=info, region_px=list(region))
    derive(doc)
    errs = validate_page(doc)
    if errs:
        raise AuditError("draft failed validation:\n- " + "\n- ".join(errs[:20]))
    write_json_atomic(out, doc)
    return out


def import_pilot(root: Path, force: bool = False, fixture_path: Path = PILOT_FIXTURE) -> Path:
    """Expose the audited bachehaye_ghali p66 fixture in this format."""
    fx = json.loads(fixture_path.read_text(encoding="utf-8"))
    src = fx["source"]
    slug, page = src["slug"], int(src["page"])
    out = draft_path(root, slug, page)
    if out.exists() and not force:
        raise AuditError(f"{out} exists; pass --force")
    ws = root / "books" / slug
    pdf_sha = file_sha(ws / "source.pdf")
    image_path, meta = audit_image_for(root, slug, page)
    doc = _base_doc("draft", slug, page, pdf_sha, image_path, meta, fixture_path, "audited_fixture", root)
    if doc["page_id"] != src["page_id"]:
        raise AuditError(f"page id mismatch: {doc['page_id']} vs fixture {src['page_id']}")
    W, H = doc["image"]["width_px"], doc["image"]["height_px"]
    for line in fx["page"]["lines"]:
        x0, y0, x1, y1 = line["rect"]
        frame = dict(cx=round((x0 + x1) / 2 * W, 2), cy=round((y0 + y1) / 2 * H, 2), angle_deg=0.0)
        words = []
        for w in line["words"]:
            a, b, c, d = w["rect"]
            words.append(dict(box=fix_box([a * W - frame["cx"], b * H - frame["cy"], c * W - frame["cx"], d * H - frame["cy"]]),
                              state="accepted", uncertain=False, reasons=[]))
        doc["lines"].append(dict(tokens=[w["text"] for w in line["words"]], frame=frame,
                                 box=fix_box([x0 * W - frame["cx"], y0 * H - frame["cy"], x1 * W - frame["cx"], y1 * H - frame["cy"]]),
                                 words=words, state="accepted", line_fixed=False, uncertain=False,
                                 reasons=[], notes=[], text_flag=None))
    for r in fx["page"].get("regions", []):
        doc["regions"].append(dict(region_id=r["region_id"], kind=r["kind"], rect=r["rect"], text=r.get("text"),
                                   note=r.get("evaluation_scope", "")))
    doc["status"] = "previously_audited"
    doc["label"] = f"{slug} p{page} (previously audited)"
    doc["previous_audit"] = dict(fixture=rel(fixture_path, root), fixture_sha256=file_sha(fixture_path),
                                 auditor=fx["audit"].get("auditor"), audited_at=fx["audit"].get("audited_at"),
                                 decision=fx["audit"].get("decision"),
                                 note=("Geometry audited earlier by an independent agent audit against the full "
                                       "page image; shown here for spot checks. Coordinates were normalised to "
                                       f"the benchmark render {src['image_path']} (same full-page frame)."))
    derive(doc)
    errs = validate_page(doc)
    if errs:
        raise AuditError("pilot import failed validation:\n- " + "\n- ".join(errs[:20]))
    write_json_atomic(out, doc)
    return out


# ---------------------------------------------------------------------------
# Audit state: load / save / operations / undo
# ---------------------------------------------------------------------------
def discover(root: Path) -> list[tuple[str, int]]:
    out = []
    books = root / "books"
    if not books.exists():
        return out
    for p in sorted(books.glob("*/truth/*.geometry.draft.json")):
        try:
            out.append((p.parts[-3], int(p.name[:4])))
        except ValueError:
            continue
    return out


def load_audit(root: Path, slug: str, page: int) -> dict:
    """The audit document; created from the untouched draft on first open."""
    ap = audit_path(root, slug, page)
    if ap.exists():
        return json.loads(ap.read_text(encoding="utf-8"))
    dp = draft_path(root, slug, page)
    if not dp.exists():
        raise AuditError(f"no draft for {slug} p{page}")
    doc = json.loads(dp.read_text(encoding="utf-8"))
    doc["kind"] = "audit"
    doc["draft_file_sha256"] = file_sha(dp)
    doc["audit_started_at"] = now_iso()
    doc["audited_at"] = None
    doc["revision"] = 0
    doc["cursor"] = dict(line=0)
    if doc["status"] != "previously_audited":
        doc["status"] = "draft"
    doc["tool_version"] = TOOL_VERSION
    derive(doc)
    write_json_atomic(ap, doc)
    return doc


def save_audit(root: Path, doc: dict) -> None:
    derive(doc)
    errs = validate_page(doc)
    if errs:
        raise AuditError("refusing to save an invalid page:\n- " + "\n- ".join(errs[:10]))
    write_json_atomic(audit_path(root, doc["slug"], doc["page"]), doc)


def _mark_edit(line: dict) -> None:
    line["state"] = "editing"


def _word(line, i):
    if not (0 <= i < len(line["words"])):
        raise AuditError(f"no word {i} on this line")
    return line["words"][i]


def _ink_cache(root: Path, doc: dict, cache: dict):
    key = doc["image"]["path"]
    if key not in cache:
        cache[key] = bd.binarize(bd.load_gray(root / key))
    return cache[key]


def robust_band(line: dict) -> list[float] | None:
    """Vertical extent of the line's own boxes, ignoring boxes whose centre
    lies far from the others (a box that sits on another printed line)."""
    ws = [w["box"] for w in line["words"]]
    if len(ws) < 3:
        return None
    vc = np.array([(b[1] + b[3]) / 2 for b in ws])
    h = float(np.median([b[3] - b[1] for b in ws]))
    med = float(np.median(vc))
    keep = [b for b, c in zip(ws, vc) if abs(c - med) <= 0.6 * max(h, 1.0)]
    if len(keep) < max(2, len(ws) // 2):
        return None
    return [min(b[1] for b in keep) - 3.0, max(b[3] for b in keep) + 3.0]


def _neighbour_gap(doc: dict, li: int, d: int) -> float | None:
    j = li + d
    if not (0 <= j < len(doc["lines"])):
        return None
    o = doc["lines"][j]["frame"]
    _, v = page_to_frame(doc["lines"][li]["frame"], o["cx"], o["cy"])
    return abs(v) if abs(v) > 1 else None


def _resegment(root: Path, doc: dict, li: int, ink_cache: dict | None, repair: bool = True,
               why: str | None = None) -> str | None:
    """Re-split line ``li``'s words inside its outline (the R action).

    With ``repair``, an outline that was stretched by a box lying on another
    printed line is first shrunk back to the band of the line's own boxes.
    Returns a note when it repaired the outline."""
    line = doc["lines"][li]
    ink = _ink_cache(root, doc, ink_cache if ink_cache is not None else {})
    fr = line["frame"]
    u0, v0, u1, v1 = line["box"]
    note = None
    if repair:
        band = robust_band(line)
        if band and (v1 - v0) > 1.4 * (band[1] - band[0]):
            v0, v1 = max(v0, band[0]), min(v1, band[1])
            note = "outline shrunk back to the line's own ink before re-splitting"
    h = max(8.0, v1 - v0)
    gaps = [g for g in (_neighbour_gap(doc, li, -1), _neighbour_gap(doc, li, 1)) if g]
    sp = float(np.median(gaps)) if gaps else 1.15 * h
    up = min(max(-v0, 0.3 * sp), 0.5 * (_neighbour_gap(doc, li, -1) or 1.2 * sp))
    dn = min(max(v1, 0.3 * sp), 0.5 * (_neighbour_gap(doc, li, 1) or 1.2 * sp))
    xa, ya = frame_to_page(fr, u0, 0)
    xb, yb = frame_to_page(fr, u1, 0)
    tr = bd.Track(xs=np.array([xa, xb]), ys=np.array([ya, yb]))
    boxes, wr, lr = bd.segment_words(ink, tr, xa, xb, line["tokens"], sp, up, dn)
    words = []
    for k, (x0, x1, dt_, db_) in enumerate(boxes):
        pts = [(x0, tr.y_at(x0) + dt_), (x1, tr.y_at(x1) + dt_), (x1, tr.y_at(x1) + db_), (x0, tr.y_at(x0) + db_)]
        b = quad_to_frame_box(fr, pts)
        b = [max(b[0], u0), max(b[1], v0), min(b[2], u1), min(b[3], v1)]   # never outside the outline
        reasons = list(wr[k]) + ([why] if why else [])
        words.append(dict(box=fix_box(b), state="draft", uncertain=bool(reasons), reasons=reasons))
    line["words"] = words
    line["box"] = fix_box([u0, v0, u1, v1])
    line["line_fixed"] = True
    _mark_edit(line)
    return note


def _check_on_line(line: dict, box) -> None:
    """A word box must sit on its own printed line: its centre inside the
    line outline (vertically), so the outline cannot be stretched over a
    neighbouring line by accident."""
    v0, v1 = line["box"][1], line["box"][3]
    c = (box[1] + box[3]) / 2
    if not (v0 <= c <= v1):
        raise AuditError("that box is centred outside this line's outline (on another printed line?). "
                         "If the outline is wrong, fix it first in line outline mode (L).")


def apply_op(root: Path, doc: dict, li: int, op: str, args: dict, ink_cache: dict | None = None) -> list[int]:
    """Apply one action to the document in memory. Returns the indices of the
    lines it changed (for the undo record). Raises AuditError on a bad action."""
    if op == "set_cursor":
        doc["cursor"] = dict(line=int(args.get("line", li)))
        return []
    if not (0 <= li < len(doc["lines"])):
        raise AuditError(f"no line {li}")
    line = doc["lines"][li]
    if op == "accept_line":
        if not line["count_ok"] and not line.get("text_flag"):
            raise AuditError(f"line has {len(line['words'])} boxes for {len(line['tokens'])} words: split or merge "
                             "boxes until they match, or flag the text as wrong")
        changed = line["line_fixed"] or any(w["state"] == "fixed" for w in line["words"])
        for w in line["words"]:
            if w["state"] == "draft":
                w["state"] = "accepted"
        line["state"] = "fixed" if changed else "accepted"
        line["reviewed_at"] = now_iso()
        return [li]
    if op == "set_word_box":
        w = _word(line, int(args["word"]))
        _check_on_line(line, fix_box(args["box"]))
        w["box"] = fix_box(args["box"])
        w["state"] = "fixed"
        _mark_edit(line)
        return [li]
    if op == "set_boundary":
        i = int(args["word"])
        a, b = _word(line, i), _word(line, i + 1)    # a: right word, b: the next (left) word
        u = float(args["u"])
        gap = max(0.0, a["box"][0] - b["box"][2])
        half = min(gap / 2, 6.0)
        if not (b["box"][0] + 1 < u < a["box"][2] - 1):
            raise AuditError("boundary must stay inside the two words")
        a["box"][0] = u + half
        b["box"][2] = u - half
        a["state"] = b["state"] = "fixed"
        _mark_edit(line)
        return [li]
    if op == "split":
        i = int(args["word"])
        w = _word(line, i)
        u = float(args["u"])
        u0, v0, u1, v1 = w["box"]
        if not (u0 + 1 < u < u1 - 1):
            raise AuditError("split point must be inside the box")
        right = dict(w, box=[u + 1, v0, u1, v1], state="fixed", reasons=list(w.get("reasons", [])))
        left = dict(w, box=[u0, v0, u - 1, v1], state="fixed", reasons=list(w.get("reasons", [])))
        line["words"][i:i + 1] = [right, left]
        _mark_edit(line)
        return [li]
    if op == "merge":
        i = int(args["word"])
        a, b = _word(line, i), _word(line, i + 1)
        m = dict(a, box=union_box([a["box"], b["box"]]), state="fixed",
                 uncertain=a.get("uncertain", False) or b.get("uncertain", False),
                 reasons=list(dict.fromkeys(a.get("reasons", []) + b.get("reasons", []))))
        line["words"][i:i + 2] = [m]
        _mark_edit(line)
        return [li]
    if op == "delete_box":
        i = int(args["word"])
        _word(line, i)
        del line["words"][i]
        line["line_fixed"] = True
        _mark_edit(line)
        return [li]
    if op == "add_box":
        i = int(args.get("index", len(line["words"])))
        _check_on_line(line, fix_box(args["box"]))
        line["words"].insert(max(0, min(i, len(line["words"]))),
                             dict(box=fix_box(args["box"]), state="fixed", uncertain=False, reasons=[]))
        _mark_edit(line)
        return [li]
    if op == "set_line_box":
        line["box"] = fix_box(args["box"])
        line["line_fixed"] = True
        _mark_edit(line)
        return [li]
    if op == "move_line":
        du, dv = float(args.get("du", 0)), float(args.get("dv", 0))
        x, y = frame_to_page(line["frame"], du, dv)
        line["frame"]["cx"], line["frame"]["cy"] = round(x, 2), round(y, 2)
        line["line_fixed"] = True
        _mark_edit(line)
        return [li]
    if op == "rotate_line":
        line["frame"]["angle_deg"] = round(line["frame"]["angle_deg"] + float(args["delta_deg"]), 4)
        line["line_fixed"] = True
        _mark_edit(line)
        return [li]
    if op == "resegment":
        repaired = _resegment(root, doc, li, ink_cache)
        if repaired:
            line["notes"] = list(dict.fromkeys(line.get("notes", []) + [repaired]))
        return [li]
    if op == "flag_text":
        note = str(args.get("note", "")).strip()
        if not note:
            raise AuditError("say what is wrong with the text (a short note)")
        line["text_flag"] = dict(note=note, at=now_iso())
        return [li]
    if op == "unflag_text":
        line["text_flag"] = None
        if line["state"] in DONE and not line["count_ok"]:
            line["state"] = "editing"
        return [li]
    if op in ("push_word", "pull_word"):
        # Rule: the moved word's box is never carried across (it lies on the
        # other printed line). The giving line keeps its other boxes, hand
        # fixes included, and only drops the box at the end the word left.
        # The receiving line's boxes were made for a different word list, so
        # it is re-split inside its unchanged outline (as R does) and its
        # boxes come back as draft and uncertain.
        if li + 1 >= len(doc["lines"]):
            raise AuditError("there is no next line")
        nxt = doc["lines"][li + 1]
        if op == "push_word":       # last word of this line -> start of the next line
            give, take, gi, ti = line, nxt, li, li + 1
            if len(give["tokens"]) < 2:
                raise AuditError("a line keeps at least one word")
            tok = give["tokens"].pop()
            take["tokens"].insert(0, tok)
            if len(give["words"]) > len(give["tokens"]):
                give["words"].pop()          # leftmost box: the word that moved
        else:                       # first word of the next line -> end of this line
            give, take, gi, ti = nxt, line, li + 1, li
            if len(give["tokens"]) < 2:
                raise AuditError("a line keeps at least one word")
            tok = give["tokens"].pop(0)
            take["tokens"].append(tok)
            if len(give["words"]) > len(give["tokens"]):
                give["words"].pop(0)         # rightmost box: the word that moved
        _resegment(root, doc, ti, ink_cache, repair=False,
                   why="re-split after a word moved across the line break")
        for L in (give, take):
            L["line_fixed"] = True
            _mark_edit(L)
        return [li, li + 1]
    raise AuditError(f"unknown action {op!r}")


class AuditStore:
    """All audit state on disk; every action is saved before it is reported."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.ink: dict = {}

    def pages(self) -> list[dict]:
        out = []
        for slug, page in discover(self.root):
            try:
                out.append(page_summary(load_audit(self.root, slug, page)))
            except (AuditError, OSError, json.JSONDecodeError) as exc:
                out.append(dict(key=f"{slug}/{page}", slug=slug, page=page, error=str(exc)))
        return out

    def get(self, slug: str, page: int) -> dict:
        with self.lock:
            return load_audit(self.root, slug, page)

    def act(self, slug: str, page: int, li: int, op: str, args: dict, rev: int | None = None) -> dict:
        with self.lock:
            doc = load_audit(self.root, slug, page)
            if rev is not None and op != "set_cursor" and int(rev) != int(doc.get("revision", 0)):
                raise Conflict(doc)
            before_meta = dict(status=doc["status"], audited_at=doc.get("audited_at"), cursor=copy.deepcopy(doc.get("cursor")))
            snapshot = copy.deepcopy(doc["lines"])
            changed = apply_op(self.root, doc, li, op, args or {}, self.ink)
            doc["revision"] = int(doc.get("revision", 0)) + 1
            if changed:
                rec = dict(at=now_iso(), op=op, line=li, args=args or {}, revision_before=doc["revision"] - 1,
                           meta_before=before_meta, lines_before={str(i): snapshot[i] for i in changed})
                append_jsonl(undo_path(self.root, slug, page), rec)
            save_audit(self.root, doc)
            return doc

    def undo(self, slug: str, page: int) -> tuple[dict, dict | None]:
        with self.lock:
            doc = load_audit(self.root, slug, page)
            up = undo_path(self.root, slug, page)
            if not up.exists():
                return doc, None
            recs = [l for l in up.read_bytes().splitlines() if l.strip()]
            if not recs:
                return doc, None
            rec = json.loads(recs[-1])
            for k, v in rec["lines_before"].items():
                doc["lines"][int(k)] = v
            doc["status"] = rec["meta_before"]["status"]
            doc["audited_at"] = rec["meta_before"]["audited_at"]
            doc["cursor"] = dict(line=int(rec["line"]))
            doc["revision"] = int(doc.get("revision", 0)) + 1
            save_audit(self.root, doc)
            tmp = up.with_name(up.name + ".tmp")
            tmp.write_bytes(b"".join(r + b"\n" for r in recs[:-1]))
            os.replace(tmp, up)
            return doc, rec


class Conflict(Exception):
    def __init__(self, doc):
        super().__init__("the page changed in another tab; reloaded")
        self.doc = doc


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def _parse_key(key: str) -> tuple[str, int]:
    try:
        slug, page = key.rsplit("/", 1)
        return slug, int(page)
    except (ValueError, AttributeError):
        raise AuditError(f"bad page key {key!r}")


def make_handler(store: AuditStore, port: int):
    from box_audit_ui import PAGE_HTML

    allowed = None

    class Handler(BaseHTTPRequestHandler):
        server_version = "box-audit/1"

        def log_message(self, fmt, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if "json" in ctype or "html" in ctype else "max-age=3600")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, value):
            self._send(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    html = PAGE_HTML.replace("__PORT__", str(port))
                    return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                if u.path == "/api/ping":
                    return self._json(200, dict(ok=True, tool=TOOL_VERSION))
                if u.path == "/api/pages":
                    return self._json(200, dict(ok=True, pages=store.pages()))
                if u.path == "/api/page":
                    slug, page = _parse_key(q.get("key", [""])[0])
                    return self._json(200, dict(ok=True, doc=store.get(slug, page)))
                if u.path == "/img":
                    slug, page = _parse_key(q.get("key", [""])[0])
                    doc = store.get(slug, page)
                    p = (store.root / doc["image"]["path"]).resolve()
                    if store.root.resolve() not in p.parents:
                        raise AuditError("image outside the repository")
                    return self._send(200, p.read_bytes(), "image/png")
                if u.path == "/font/vazirmatn.ttf" and FONT.exists():
                    return self._send(200, FONT.read_bytes(), "font/ttf")
                return self._json(404, dict(ok=False, error="not found"))
            except AuditError as exc:
                return self._json(400, dict(ok=False, error=str(exc)))
            except Exception as exc:  # report, never crash the server
                return self._json(500, dict(ok=False, error=f"{type(exc).__name__}: {exc}"))

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                slug, page = _parse_key(body.get("key", ""))
                if u.path == "/api/op":
                    doc = store.act(slug, page, int(body.get("line", 0)), str(body.get("op")), body.get("args") or {},
                                    body.get("rev"))
                    return self._json(200, dict(ok=True, doc=doc, summary=page_summary(doc)))
                if u.path == "/api/undo":
                    doc, rec = store.undo(slug, page)
                    return self._json(200, dict(ok=True, doc=doc, undone=(rec or {}).get("op"),
                                                summary=page_summary(doc)))
                return self._json(404, dict(ok=False, error="not found"))
            except Conflict as c:
                return self._json(409, dict(ok=False, error=str(c), doc=c.doc))
            except AuditError as exc:
                return self._json(400, dict(ok=False, error=str(exc)))
            except Exception as exc:
                return self._json(500, dict(ok=False, error=f"{type(exc).__name__}: {exc}"))

    return Handler


def serve(root: Path, port: int, open_browser: bool = False) -> None:
    store = AuditStore(root)
    pages = store.pages()
    if not pages:
        raise SystemExit("no drafts found under books/*/truth/; run the draft commands first")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"Box audit: {url}")
    for p in pages:
        if "error" in p:
            print(f"  {p['key']}: ERROR {p['error']}")
        else:
            print(f"  {p['label']}: {p['done']}/{p['lines']} lines reviewed, status {p['status']}")
    print("Every action is saved to books/<slug>/truth/NNNN.geometry.audit.json at once. Ctrl-C stops the server.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Overlay rendering (for looking at a draft or an audit without the tool)
# ---------------------------------------------------------------------------
def render_overlay(root: Path, doc: dict, out: Path, scale: float = 0.6, clip=None, labels: bool = True) -> None:
    import html as _html
    img = root / doc["image"]["path"]
    W, H = doc["image"]["width_px"], doc["image"]["height_px"]
    pdf = fitz.open()
    pg = pdf.new_page(width=W, height=H)
    pg.insert_image(pg.rect, filename=str(img))
    arch = fitz.Archive(str(FONT.parent))
    css = "@font-face{font-family:vz;src:url(%s);}" % FONT.name
    for line in doc["lines"]:
        unc = line.get("uncertain")
        lq = [fitz.Point(x * W, y * H) for x, y in line["quad"]]
        pg.draw_polyline(lq + [lq[0]], color=(1, 0, 0) if unc else (0.6, 0.6, 0.6), width=1.5, dashes="[6 4] 0")
        for k, w in enumerate(line["words"]):
            q = [fitz.Point(x * W, y * H) for x, y in w["quad"]]
            col = (0.85, 0.45, 0) if w.get("uncertain") else ((0, 0.55, 0) if k % 2 == 0 else (0, 0.2, 0.9))
            pg.draw_polyline(q + [q[0]], color=col, width=2)
            if labels and w.get("text"):
                x0 = min(p.x for p in q); x1 = max(p.x for p in q); y0 = min(p.y for p in q)
                r = fitz.Rect(x0 - 30, y0 - 24, x1 + 30, y0 - 1)
                pg.insert_htmlbox(r, '<div dir="rtl" style="text-align:center;font-size:15px;color:#c00;font-family:vz">%s</div>'
                                  % _html.escape(w["text"]), css=css, archive=arch)
    for r in doc.get("regions", []):
        x0, y0, x1, y1 = r["rect"]
        pg.draw_rect(fitz.Rect(x0 * W, y0 * H, x1 * W, y1 * H), color=(0.6, 0, 0.6), width=1.5, dashes="[3 3] 0")
    cl = fitz.Rect(*[c * d for c, d in zip(clip, (W, H, W, H))]) if clip else None
    pg.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=cl).save(str(out))


# ---------------------------------------------------------------------------
# Synthetic evaluation of the drafting method
# ---------------------------------------------------------------------------
def _xiou(a, b):
    i = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    u = max(a[2], b[2]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def synth_eval_page(image_u8: np.ndarray, truth: dict) -> dict:
    """Draft a synthetic page from its exact line texts and grade every word
    against its ink box. ``ok`` = horizontal IoU >= 0.5 with any vertical
    overlap (thin marks such as '-' make full IoU meaningless); ``ok_iou50`` is
    the stricter full IoU >= 0.5."""
    H, W = image_u8.shape
    texts = [l["text"] for l in truth["lines"]]
    res = bd.draft_known_lines(image_u8.astype(np.float32), texts, [False] * len(texts))
    words = {w["id"]: w for w in truth["words"]}
    n = ok = ok50 = flagged = wrong_flagged = 0
    lines_ok = 0
    uncertain_lines = 0
    for L, tl in zip(res["lines"], truth["lines"]):
        tw = sorted((words[i] for i in tl["word_ids"]), key=lambda w: w["visual_index"])
        t = L["track"]
        all_ok = True
        uncertain_lines += bool(L["reasons"])
        for k, (x0, x1, dt_, db_) in enumerate(L["words"]):
            ys = [t.y_at(x0) + dt_, t.y_at(x1) + dt_, t.y_at(x0) + db_, t.y_at(x1) + db_]
            box = [x0 / W, min(ys) / H, x1 / W, max(ys) / H]
            ib = tw[k]["ink_aabb"] if k < len(tw) else None
            good = ib is not None and _xiou(box, ib) >= 0.5 and min(box[3], ib[3]) > max(box[1], ib[1])
            good50 = ib is not None and _iou(box, ib) >= 0.5
            f = bool(L["word_reasons"][k]) or bool(L["reasons"])
            n += 1; ok += good; ok50 += good50; flagged += f; wrong_flagged += f and not good
            all_ok &= good
        lines_ok += all_ok
    return dict(words=n, ok=ok, ok_iou50=ok50, flagged=flagged, wrong=n - ok, wrong_unflagged=(n - ok) - wrong_flagged,
                lines=len(res["lines"]), lines_all_ok=lines_ok, uncertain_lines=uncertain_lines,
                candidates=res["info"]["candidates"])


def synth_eval(seeds, presets=("photo", "clean")) -> list[dict]:
    import synth_pages as sp
    rows = []
    for preset in presets:
        for seed in seeds:
            spec = sp.make_spec("typewriter", preset=preset, seed=int(seed))
            g = sp.generate(spec)
            r = synth_eval_page(g["image"], g["truth"])
            r.update(name=spec["name"])
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the audit tool on 127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--open", action="store_true", help="open a browser tab")
    s.add_argument("--root", type=Path, default=None,
                   help="repository-like folder holding books/<slug>/truth (default: this repository); for testing on a copy")
    sub.add_parser("status", help="progress per page")
    d = sub.add_parser("draft", help="write the untouched draft for one page")
    d.add_argument("--slug", required=True)
    d.add_argument("--page", type=int, required=True)
    d.add_argument("--force", action="store_true")
    ip = sub.add_parser("import-pilot", help="expose the audited bachehaye_ghali p66 fixture")
    ip.add_argument("--force", action="store_true")
    o = sub.add_parser("overlay", help="render a PNG of a draft or audit")
    o.add_argument("--key", required=True, help="slug/page, e.g. hadpi-articles/1")
    o.add_argument("--which", choices=("draft", "audit"), default="draft")
    o.add_argument("--out", required=True)
    o.add_argument("--scale", type=float, default=0.6)
    o.add_argument("--clip", type=float, nargs=4, default=None, help="x0 y0 x1 y1 (fractions)")
    e = sub.add_parser("export", help="print an audited page in the annotation-workflow line shape")
    e.add_argument("--key", required=True)
    se = sub.add_parser("synth-eval", help="draft accuracy on synthetic typewriter pages")
    se.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6])
    a = ap.parse_args(argv)
    root = ROOT
    try:
        if a.cmd == "serve":
            serve(a.root.resolve() if a.root else root, a.port, a.open)
        elif a.cmd == "status":
            for p in AuditStore(root).pages():
                print(json.dumps(p, ensure_ascii=False))
        elif a.cmd == "draft":
            print(draft_page(root, a.slug, a.page, a.force))
        elif a.cmd == "import-pilot":
            print(import_pilot(root, a.force))
        elif a.cmd == "overlay":
            slug, page = _parse_key(a.key)
            path = draft_path(root, slug, page) if a.which == "draft" else audit_path(root, slug, page)
            render_overlay(root, json.loads(path.read_text(encoding="utf-8")), Path(a.out), a.scale, a.clip)
            print(a.out)
        elif a.cmd == "export":
            import bbox_annotation_workflow as wf
            slug, page = _parse_key(a.key)
            doc = json.loads(audit_path(root, slug, page).read_text(encoding="utf-8"))
            if doc["status"] not in ("audited", "previously_audited"):
                raise AuditError("page is not audited yet")
            lines = to_workflow_lines(doc)
            errs = wf.validate_lines(doc["page_id"], lines)
            if errs:
                raise AuditError("workflow validation failed:\n- " + "\n- ".join(errs[:10]))
            print(json.dumps(dict(page_id=doc["page_id"], image_sha256=doc["image"]["sha256"], lines=lines),
                             ensure_ascii=False, indent=1))
        elif a.cmd == "synth-eval":
            rows = synth_eval(a.seeds)
            for r in rows:
                print(f"{r['name']}: words {r['ok']}/{r['words']} = {r['ok'] / r['words']:.3f} "
                      f"(full IoU>=0.5: {r['ok_iou50'] / r['words']:.3f}); lines all-correct "
                      f"{r['lines_all_ok']}/{r['lines']}; flagged words {r['flagged']}; wrong {r['wrong']}, "
                      f"wrong and unflagged {r['wrong_unflagged']}; uncertain lines {r['uncertain_lines']}")
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
