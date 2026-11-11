"""Review-time box feedback capture (docs/bbox-old-fonts-plan.md section 6.3).

Every look a reviewer gives a box becomes one append-only record in a single
project-level store, ``box_truth.jsonl`` (next to ``qc_history.json``,
gitignored). Feedback OBSERVES the placement engine; it never changes a box,
a status, or any engine cache.

Record identity
  * ``image.sha256``  -- SHA-256 of the exact PNG bytes the review UI serves
    for the page (``review._image_path_for``: the hi-res render when present).
    This is the same identity the benchmark/annotation tools bind truth to.
  * ``correction_key`` -- the benchmark's case id
    (``tests/bbox_benchmark._collect``): ``_id("c", page_id, hunk_id,
    markdown_sha256, old, new, span)`` with ``page_id = _id("p", slug, page,
    pdf_sha256)``. Unlinked QC findings (no hunk) use ``_id("c", page_id,
    "issue", markdown_sha256, type, description, snippet)``; free-form text
    edits use ``_id("t", page_id, markdown_sha256, start, end, old, new)``.

Split rule (``SPLIT_RULE``): ``heldout`` when
``int(sha256("box-truth-split-v1:" + image_sha256), 16) % 2 == 1``, else
``dev``. Deterministic, about 50/50, and a page image is always in one split.
Tune on ``dev`` only; gate on ``heldout`` only.

Latest wins per (image sha, correction key, action family). Families:
``box_verdict`` (drag_fix | confirm | wrong_line), ``spot_check``,
``text_edit``. Undo appends a ``retract`` record naming the retracted
``record_id``; the previous surviving record of that family becomes current.
Nothing is ever deleted or rewritten.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
import secrets
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .workspace import PROJECT_ROOT

SCHEMA = "farsi2epub.box-truth/v1"
SPLIT_RULE = "box-truth-image-sha256-parity-v1"
_SPLIT_SALT = "box-truth-split-v1:"
DEFAULT_STORE_PATH = PROJECT_ROOT / "box_truth.jsonl"
STORE_ENV = "FARSI2EPUB_BOX_TRUTH_PATH"
SEED_ENV = "FARSI2EPUB_SPOT_SEED"
SPOT_SAMPLE_SIZE = 5
MAX_BODY_BYTES = 64 * 1024
MAX_RECTS = 16
MIN_RECT_SIDE = 1e-4
MAX_TEXT_EDITS_PER_ACCEPT = 200

ACTIONS = ("drag_fix", "confirm", "wrong_line", "spot_check", "text_edit", "retract")
FAMILY = {"drag_fix": "box_verdict", "confirm": "box_verdict", "wrong_line": "box_verdict",
          "spot_check": "spot_check", "text_edit": "text_edit"}
SPOT_VERDICTS = ("right", "wrong", "skipped")
REQUIRED_FIELDS = ("schema", "record_id", "ts", "action", "family", "session_id", "book",
                   "page", "image", "pdf_sha256", "markdown_sha256", "split", "split_rule",
                   "correction_key", "view_key", "target", "shown", "engine", "human",
                   "page_conditions")


# ---------------------------------------------------------------------------
# identities (pure)
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    # Byte-identical to tests/bbox_benchmark._canonical_bytes.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _id(prefix: str, *parts: Any) -> str:
    # Byte-identical to tests/bbox_benchmark._id.
    return prefix + "_" + hashlib.sha256(_canonical_bytes(list(parts))).hexdigest()[:20]


def split_for(image_sha256: str) -> str:
    """Deterministic dev/heldout assignment for one page image."""
    h = hashlib.sha256((_SPLIT_SALT + image_sha256).encode("ascii")).hexdigest()
    return "heldout" if int(h, 16) % 2 == 1 else "dev"


def page_id(slug: str, page: int, pdf_sha256: str) -> str:
    return _id("p", slug, page, pdf_sha256)


def _bench_nth_index(text: str, needle: str, occurrence: int) -> int:
    # Mirrors tests/bbox_benchmark._nth_index.
    if occurrence < 1:
        return -1
    if not needle:
        return 0 if occurrence == 1 else -1
    start = 0
    found = -1
    for _ in range(occurrence):
        found = text.find(needle, start)
        if found < 0:
            return -1
        start = found + 1
    return found


def hunk_span(text: str, hunk: dict) -> Optional[list[int]]:
    """The anchored span of tests/bbox_benchmark._hunk_span."""
    before, old, after = hunk["ctx_before"], hunk["old"], hunk["ctx_after"]
    occurrence = 1 if hunk.get("unique") else int(hunk.get("occurrence") or 1)
    start = _bench_nth_index(text, before + old + after, occurrence)
    if start < 0:
        return None
    start += len(before)
    return [start, start + len(old)]


def hunk_correction_key(pid: str, md_sha256: str, text: str, hunk: dict) -> str:
    return _id("c", pid, hunk["id"], md_sha256, hunk["old"], hunk["new"], hunk_span(text, hunk))


def issue_correction_key(pid: str, md_sha256: str, issue: dict) -> str:
    return _id("c", pid, "issue", md_sha256, issue.get("type"), issue.get("description"),
               issue.get("snippet"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_FILE_SHA: dict[str, tuple[tuple, str]] = {}
_FILE_SHA_LOCK = threading.Lock()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, memoized on its stat stamp."""
    st = path.stat()
    stamp = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)
    key = str(path.resolve())
    with _FILE_SHA_LOCK:
        hit = _FILE_SHA.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _FILE_SHA_LOCK:
        _FILE_SHA[key] = (stamp, value)
    return value


def _png_size(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with path.open("rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    except OSError:
        pass
    return None, None


# ---------------------------------------------------------------------------
# validation (pure)
# ---------------------------------------------------------------------------


class FeedbackError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_rect(value: Any) -> dict:
    """A normalized page rectangle {x0,y0,x1,y1} with 0 <= x0 < x1 <= 1, same for y."""
    if not isinstance(value, dict) or set(value) - {"x0", "y0", "x1", "y1"} \
            or any(k not in value for k in ("x0", "y0", "x1", "y1")):
        raise FeedbackError(400, "rectangle must be an object with exactly x0, y0, x1, y1")
    r = {k: value[k] for k in ("x0", "y0", "x1", "y1")}
    if not all(_finite_number(v) for v in r.values()):
        raise FeedbackError(400, "rectangle coordinates must be finite numbers")
    if not (0 <= r["x0"] < r["x1"] <= 1 and 0 <= r["y0"] < r["y1"] <= 1):
        raise FeedbackError(400, "rectangle must lie inside the page (0..1) with x0<x1, y0<y1")
    if r["x1"] - r["x0"] < MIN_RECT_SIDE or r["y1"] - r["y0"] < MIN_RECT_SIDE:
        raise FeedbackError(400, "rectangle is too small")
    return {k: float(v) for k, v in r.items()}


def validate_rects(value: Any, *, required: bool) -> list[dict]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise FeedbackError(400, "rects must be a list")
    if required and not value:
        raise FeedbackError(400, "at least one rectangle is required")
    if len(value) > MAX_RECTS:
        raise FeedbackError(400, f"at most {MAX_RECTS} rectangles")
    return [validate_rect(r) for r in value]


def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON number {name}")


def parse_body(headers, rfile) -> dict:
    """Read a bounded JSON object body; FeedbackError on anything else."""
    raw_len = headers.get("Content-Length")
    try:
        length = int(raw_len)
    except (TypeError, ValueError):
        raise FeedbackError(411, "Content-Length required")
    if length < 0:
        raise FeedbackError(400, "invalid Content-Length")
    if length > MAX_BODY_BYTES:
        raise FeedbackError(413, f"body larger than {MAX_BODY_BYTES} bytes")
    raw = rfile.read(length) if length else b""
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise FeedbackError(400, f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        raise FeedbackError(400, "body must be a JSON object")
    return data


def _int_field(data: dict, name: str) -> int:
    v = data.get(name)
    if not isinstance(v, int) or isinstance(v, bool):
        raise FeedbackError(400, f"{name} must be an integer")
    return v


def _str_field(data: dict, name: str, *, optional: bool = False, max_len: int = 200) -> Optional[str]:
    v = data.get(name)
    if v is None and optional:
        return None
    if not isinstance(v, str) or not v or len(v) > max_len:
        raise FeedbackError(400, f"{name} must be a non-empty string")
    return v


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class BoxTruthStore:
    """Append-only JSON-lines store. Thread- and process-safe appends.

    Each append takes a thread lock plus an exclusive ``flock`` on the file,
    repairs a torn final line left by a crash (by terminating it, never by
    rewriting it), writes the whole record as one line and fsyncs. Readers
    skip any line that does not parse to a record object.
    """

    _LOCKS: dict[str, threading.Lock] = {}
    _GUARD = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)
        with self._GUARD:
            self._lock = self._LOCKS.setdefault(str(self.path.resolve()), threading.Lock())

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))
        if "\n" in line:
            raise ValueError("record serialized across lines")
        data = (line + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    size = os.fstat(fd).st_size
                    if size and os.pread(fd, 1, size - 1) != b"\n":
                        data = b"\n" + data  # terminate a torn line; never rewrite it
                    view = memoryview(data)
                    while view:
                        n = os.write(fd, view)
                        view = view[n:]
                    os.fsync(fd)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def read(self) -> tuple[list[dict], int]:
        """(records in file order, number of malformed lines skipped)."""
        return read_records(self.path)


def read_records(path: Path) -> tuple[list[dict], int]:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return [], 0
    records, bad = [], 0
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError):
            bad += 1
            continue
        if (not isinstance(rec, dict) or rec.get("schema") != SCHEMA
                or rec.get("action") not in ACTIONS or not isinstance(rec.get("record_id"), str)):
            bad += 1
            continue
        records.append(rec)
    return records, bad


def identity(rec: dict) -> tuple:
    return ((rec.get("image") or {}).get("sha256"), rec.get("correction_key"), rec.get("family"))


def effective(records: Iterable[dict]) -> dict[tuple, dict]:
    """Latest surviving record per (image sha, correction key, family)."""
    stacks: dict[tuple, list[dict]] = {}
    retracted: set[str] = set()
    for rec in records:
        if rec.get("action") == "retract":
            if isinstance(rec.get("retracts"), str):
                retracted.add(rec["retracts"])
            continue
        stacks.setdefault(identity(rec), []).append(rec)
    out = {}
    for key, recs in stacks.items():
        alive = [r for r in recs if r["record_id"] not in retracted]
        if alive:
            out[key] = alive[-1]
    return out


def store_path() -> Path:
    env = os.environ.get(STORE_ENV)
    return Path(env) if env else DEFAULT_STORE_PATH


# ---------------------------------------------------------------------------
# page snapshot: exactly what the UI shows, computed server-side
# ---------------------------------------------------------------------------


def _engine_info(refiner, n: int, md: str, query) -> dict:
    from . import llm
    from .geometry_attempt import ATTEMPT_VERSION
    from .page_map import DERIVATION_VERSION
    from .scan import DETECTOR, RENDER_VERSION
    import fitz
    info = {"derivation_version": DERIVATION_VERSION, "attempt_version": ATTEMPT_VERSION,
            "render_version": RENDER_VERSION, "detector": DETECTOR,
            "region_reader_version": llm.REGION_READER_VERSION, "pymupdf": fitz.VersionBind,
            "bbox_mode": None, "reader_model": None, "reader_prompt_sha256": None,
            "page_key": None, "box_key": None}
    if refiner is not None:
        try:
            info.update(bbox_mode=refiner.mode, reader_model=refiner.model,
                        reader_prompt_sha256=refiner.prompt_hash,
                        page_key=refiner._page_key(n), box_key=refiner._key(n, md, query))
        except Exception:
            pass
    return info


def page_snapshot(ws, n: int) -> Optional[dict]:
    """Per-correction context for page ``n`` as the review UI currently shows it.

    Reuses review's own derivation (``_page_box_inputs``/``_box_specs``) and its
    shown-box logic (``_attach_boxes``) on fresh copies, so the recorded box is
    the rendered box, chip status and reason included. Read-only: no engine
    state is written. None when the page has no image to tie records to.
    """
    from . import review  # lazy: review imports this module
    img_path = review._image_path_for(ws, n)
    if img_path is None:
        return None
    sidecar, issues, hunks, text, panel_kind = review._page_box_inputs(ws, n)
    refiner = review._REFINER
    try:
        pdf_sha = refiner.source_hash() if refiner is not None else sha256_file(ws.pdf_path)
    except Exception:
        pdf_sha = sha256_file(ws.pdf_path)
    md_sha = sha256_bytes(text.encode("utf-8"))
    image_sha = sha256_file(img_path)
    width, height = _png_size(img_path)
    pid = page_id(ws.slug, n, pdf_sha)
    specs = review._box_specs(issues, hunks, text) if (issues or hunks) else []
    queries = [s[1] for s in specs]
    located = list(review._locate_queries_cached(str(ws.pdf_path), n, text, tuple(queries))) if specs else []
    results = None
    if refiner is not None and specs:
        try:
            located = refiner.apply_cached(n, text, queries, located)
            results = refiner.results(n, text, queries, located)
        except Exception:
            results = None
    if specs:
        review._attach_boxes(specs, located, results)
    summary = review._page_map_summary(n)
    try:
        meta_source = ws.meta.get("source_type")
    except Exception:
        meta_source = None
    conditions = {
        "page_map": copy.deepcopy(summary) if isinstance(summary, dict) else {},
        "source_type": meta_source,
        "transcription": {k: sidecar.get(k) for k in ("model_used", "resolution", "quality_score",
                                                      "confidence", "flags")},
    }
    try:
        rel = str(img_path.relative_to(ws.root))
    except ValueError:
        rel = img_path.name
    page = {
        "book": ws.slug, "page": n, "text": text, "markdown_sha256": md_sha,
        "pdf_sha256": pdf_sha, "page_id": pid,
        "image": {"sha256": image_sha, "path": rel,
                  "kind": "hires" if img_path == ws.page_hires_path(n) else "std",
                  "width_px": width, "height_px": height},
        "split": split_for(image_sha), "page_conditions": conditions,
        "pending": any(r.status == "pending" for r in results) if results else False,
        "corrections": {},
    }
    for view_key, query, _unused, hunk, issue in specs:
        obj = hunk if hunk is not None else issue
        chip = obj.get("placement") or {}
        if hunk is not None:
            ck = hunk_correction_key(pid, md_sha, text, hunk)
            iss = issues[hunk["issue_idx"]] if hunk.get("issue_idx") is not None \
                and hunk["issue_idx"] < len(issues) else {}
            kind = "finding" if hunk.get("edit_only") else (
                "insertion" if not hunk["old"] and hunk["new"] else
                "deletion" if hunk["old"] and not hunk["new"] else "replacement")
            target = {"kind": kind, "old": hunk["old"],
                      "suggested": None if hunk.get("edit_only") else hunk["new"],
                      "issue_type": iss.get("type"), "issue_description": iss.get("description"),
                      "span": hunk_span(text, hunk), "hunk_id": hunk["id"]}
        else:
            ck = issue_correction_key(pid, md_sha, issue)
            target = {"kind": "finding", "old": issue.get("snippet") or "", "suggested": None,
                      "issue_type": issue.get("type"), "issue_description": issue.get("description"),
                      "span": None, "hunk_id": None}
        target["query"] = {"text": query.text, "span": list(query.span) if query.span else None,
                           "alternatives": list(query.alts), "kind": query.kind}
        box = copy.deepcopy(obj.get("box"))
        if isinstance(box, dict):
            box.pop("debug", None)
        page["corrections"][view_key] = {
            "view_key": view_key, "correction_key": ck, "target": target,
            "shown": {"box": box, "status": chip.get("status"), "source": chip.get("source"),
                      "label": chip.get("label"), "reason": chip.get("reason"),
                      "evidence": chip.get("evidence") or [], "kind": chip.get("kind"),
                      "buffer_before": chip.get("buffer_before", 0),
                      "buffer_after": chip.get("buffer_after", 0)},
            "engine": _engine_info(refiner, n, text, query),
        }
    return page


def _boxes_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        def segs(box):
            raw = box.get("segments") or [box]
            return [tuple(round(float(s[k]), 6) for k in ("x0", "y0", "x1", "y1")) for s in raw]
        return segs(a) == segs(b)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_record(page: dict, corr: Optional[dict], action: str, *, session_id: str,
                 human: dict, extra: Optional[dict] = None) -> dict:
    rec = {
        "schema": SCHEMA, "record_id": uuid.uuid4().hex, "ts": _now(), "action": action,
        "family": FAMILY[action], "session_id": session_id,
        "book": page["book"], "page": page["page"], "image": dict(page["image"]),
        "pdf_sha256": page["pdf_sha256"], "markdown_sha256": page["markdown_sha256"],
        "split": page["split"], "split_rule": SPLIT_RULE,
        "correction_key": corr["correction_key"] if corr else None,
        "view_key": corr["view_key"] if corr else None,
        "target": copy.deepcopy(corr["target"]) if corr else None,
        "shown": copy.deepcopy(corr["shown"]) if corr else None,
        "engine": copy.deepcopy(corr["engine"]) if corr else None,
        "human": human,
        "page_conditions": copy.deepcopy(page["page_conditions"]),
    }
    if extra:
        rec.update(extra)
    return rec


def validate_record(rec: Any) -> list[str]:
    """Contract check used by tests and the replay tool. Empty list = valid."""
    errs = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    for f in REQUIRED_FIELDS:
        if f not in rec:
            errs.append(f"missing {f}")
    if errs:
        return errs
    if rec["schema"] != SCHEMA:
        errs.append("bad schema")
    if rec["action"] not in ACTIONS:
        errs.append("bad action")
    if rec["action"] != "retract" and rec["family"] != FAMILY.get(rec["action"]):
        errs.append("family does not match action")
    if not re.fullmatch(r"[0-9a-f]{32}", str(rec["record_id"])):
        errs.append("bad record_id")
    if not re.fullmatch(r"[0-9a-f]{64}", str((rec["image"] or {}).get("sha256"))):
        errs.append("bad image sha256")
    elif rec["split"] != split_for(rec["image"]["sha256"]):
        errs.append("split does not follow the split rule")
    if rec["split_rule"] != SPLIT_RULE:
        errs.append("bad split_rule")
    if not isinstance(rec["page"], int) or isinstance(rec["page"], bool):
        errs.append("page must be int")
    human = rec.get("human")
    if not isinstance(human, dict) or not isinstance(human.get("rects"), list):
        errs.append("human.rects must be a list")
    else:
        for r in human["rects"]:
            try:
                validate_rect(r)
            except FeedbackError as exc:
                errs.append("human rect: " + exc.message)
    if rec["action"] in ("drag_fix", "confirm", "wrong_line", "spot_check"):
        if not isinstance(rec["correction_key"], str):
            errs.append("box action needs a correction_key")
        if not isinstance(rec.get("shown"), dict):
            errs.append("box action needs the shown box")
    if rec["action"] == "drag_fix" and not (human or {}).get("rects"):
        errs.append("drag_fix needs rects")
    if rec["action"] == "spot_check":
        sc = rec.get("spot_check")
        if not isinstance(sc, dict) or sc.get("verdict") not in SPOT_VERDICTS \
                or not isinstance(sc.get("seed"), int):
            errs.append("spot_check needs verdict and seed")
    if rec["action"] == "text_edit" and not isinstance(rec.get("text_edit"), dict):
        errs.append("text_edit needs text_edit")
    if rec["action"] == "retract" and not isinstance(rec.get("retracts"), str):
        errs.append("retract needs retracts")
    return errs


def summarize_correction(records_eff: dict[tuple, dict], image_sha: str, ck: str) -> dict:
    """Effective feedback for one correction, for the UI."""
    out = {}
    with_rects = []
    for fam in ("box_verdict", "spot_check", "text_edit"):
        rec = records_eff.get((image_sha, ck, fam))
        if rec is None:
            continue
        out[fam] = {"record_id": rec["record_id"], "action": rec["action"], "ts": rec["ts"],
                    "verdict": (rec.get("human") or {}).get("verdict"),
                    "rects": (rec.get("human") or {}).get("rects") or []}
        if out[fam]["rects"] and fam != "text_edit":
            with_rects.append(rec)
    with_rects.sort(key=lambda r: r["ts"])
    return {"effective": out,
            "human_rects": (with_rects[-1]["human"]["rects"] if with_rects else []),
            "human_rects_record": (with_rects[-1]["record_id"] if with_rects else None)}


def draw_sample(entries: list, seed: int, k: int) -> tuple[list, list]:
    """(population in canonical order, seeded uniform sample of size <= k).

    ``entries`` are tuples whose first two items are (image sha, correction
    key); only those two decide the order, so status, confidence, page and
    input order cannot influence which corrections are chosen.
    """
    population = sorted(entries, key=lambda t: (t[0], t[1]))
    k = min(k, len(population))
    return population, (random.Random(seed).sample(population, k) if k else [])


# ---------------------------------------------------------------------------
# session (one per review server)
# ---------------------------------------------------------------------------


class FeedbackSession:
    def __init__(self, ws, surfaced: list[int], *, path: Optional[Path] = None,
                 seed: Optional[int] = None, sample_size: int = SPOT_SAMPLE_SIZE):
        self.ws = ws
        self.surfaced = list(surfaced)
        self.store = BoxTruthStore(path if path is not None else store_path())
        if seed is None:
            env = os.environ.get(SEED_ENV)
            seed = int(env) if env and env.lstrip("-").isdigit() else secrets.randbits(32)
        self.seed = int(seed)
        self.sample_size = sample_size
        self.session_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.spot: Optional[dict] = None  # drawn once per session

    # -- spot check -------------------------------------------------------

    def spot_sample(self) -> dict:
        """Draw (once) a seeded uniform sample of boxed corrections.

        The population is every open correction on the surfaced pages whose
        shown box is non-null, ordered by (image sha, correction key) -- an
        order unrelated to status, confidence or page -- and sampled with
        ``random.Random(seed)``. Refused while any page still has pending
        placements, so in-flight refinement cannot shape the population.
        """
        with self.lock:
            if self.spot is not None:
                return self.spot
        pages = []
        for n in self.surfaced:
            snap = page_snapshot(self.ws, n)
            if snap is None:
                continue
            if snap["pending"]:
                return {"pending": True}
            pages.append(snap)
        population = []
        for snap in pages:
            for corr in snap["corrections"].values():
                if corr["shown"]["box"] is not None:
                    population.append((snap["image"]["sha256"], corr["correction_key"], snap, corr))
        population, chosen = draw_sample(population, self.seed, self.sample_size)
        pop_sha = sha256_bytes(_canonical_bytes([[s["book"], s["page"], ck] for _, ck, s, _ in population]))
        k = len(chosen)
        items = []
        for i, (_img, ck, snap, corr) in enumerate(chosen):
            items.append({"sample_index": i, "page": snap["page"], "key": corr["view_key"],
                          "correction_key": ck, "old": corr["target"]["old"],
                          "suggested": corr["target"]["suggested"],
                          "_page": snap, "_corr": corr})
        spot = {"pending": False, "seed": self.seed, "sample_size": k,
                "population_size": len(population), "population_sha256": pop_sha, "items": items}
        with self.lock:
            if self.spot is None:
                self.spot = spot
                print(f"[feedback] spot-check sample drawn: seed={self.seed} "
                      f"population={len(population)} sample={k} population_sha256={pop_sha[:12]}",
                      flush=True)
            return self.spot

    @staticmethod
    def public_spot(spot: dict) -> dict:
        if spot.get("pending"):
            return {"ok": True, "pending": True}
        return {"ok": True, "pending": False, "seed": spot["seed"], "sample_size": spot["sample_size"],
                "population_size": spot["population_size"],
                "population_sha256": spot["population_sha256"],
                "items": [{k: v for k, v in it.items() if not k.startswith("_")} for it in spot["items"]]}

    # -- page feedback ----------------------------------------------------

    def _snapshot(self, n: int) -> dict:
        if n not in self.surfaced:
            raise FeedbackError(404, f"page {n} is not in this review")
        snap = page_snapshot(self.ws, n)
        if snap is None:
            raise FeedbackError(409, f"page {n} has no page image to attach feedback to")
        return snap

    def page_feedback(self, n: int) -> dict:
        snap = self._snapshot(n)
        eff = effective(self.store.read()[0])
        img = snap["image"]["sha256"]
        corrections = {}
        for vk, corr in snap["corrections"].items():
            corrections[vk] = {"correction_key": corr["correction_key"],
                               "has_box": corr["shown"]["box"] is not None,
                               **summarize_correction(eff, img, corr["correction_key"])}
        return {"ok": True, "page": n, "image_sha256": img, "split": snap["split"],
                "corrections": corrections}

    def _correction(self, snap: dict, data: dict) -> dict:
        key = _str_field(data, "key", max_len=32)
        corr = snap["corrections"].get(key)
        if corr is None:
            raise FeedbackError(404, f"unknown correction {key!r} on page {snap['page']}")
        ck = _str_field(data, "correction_key", optional=True, max_len=64)
        if ck is not None and ck != corr["correction_key"]:
            raise FeedbackError(409, "correction changed since the page was loaded; reload")
        return corr

    def _client_agrees(self, data: dict, corr: dict) -> Optional[bool]:
        if "shown_box" not in data:
            return None
        sb = data["shown_box"]
        if sb is not None and not isinstance(sb, dict):
            raise FeedbackError(400, "shown_box must be an object or null")
        return _boxes_equal(sb, corr["shown"]["box"])

    def record_box_action(self, data: dict) -> dict:
        action = data.get("action")
        if action not in ("drag_fix", "confirm", "wrong_line"):
            raise FeedbackError(400, "action must be drag_fix, confirm or wrong_line")
        n = _int_field(data, "page")
        snap = self._snapshot(n)
        corr = self._correction(snap, data)
        rects = validate_rects(data.get("rects"), required=action == "drag_fix")
        if action != "drag_fix" and rects:
            raise FeedbackError(400, f"{action} takes no rectangles")
        if action in ("confirm", "wrong_line") and corr["shown"]["box"] is None:
            raise FeedbackError(409, f"no box is shown for this correction; {action} needs one")
        agrees = self._client_agrees(data, corr)
        verdict = {"drag_fix": "fixed", "confirm": "on_target", "wrong_line": "wrong_line"}[action]
        rec = build_record(snap, corr, action, session_id=self.session_id,
                           human={"verdict": verdict, "rects": rects})
        rec["shown"]["client_agrees"] = agrees
        self.store.append(rec)
        return self._after_write(snap, corr, rec)

    def _after_write(self, snap: dict, corr: dict, rec: dict) -> dict:
        eff = effective(self.store.read()[0])
        return {"ok": True, "record_id": rec["record_id"], "action": rec["action"],
                "split": rec["split"], "page": snap["page"], "key": corr["view_key"],
                "correction_key": corr["correction_key"],
                **summarize_correction(eff, snap["image"]["sha256"], corr["correction_key"])}

    def record_spot(self, data: dict) -> dict:
        with self.lock:
            spot = self.spot
        if spot is None or spot.get("pending"):
            raise FeedbackError(409, "no spot-check sample has been drawn in this session")
        idx = _int_field(data, "sample_index")
        if not 0 <= idx < len(spot["items"]):
            raise FeedbackError(404, "unknown spot-check item")
        verdict = data.get("verdict")
        if verdict not in SPOT_VERDICTS:
            raise FeedbackError(400, "verdict must be right, wrong or skipped")
        rects = validate_rects(data.get("rects"), required=False)
        if rects and verdict != "wrong":
            raise FeedbackError(400, "rectangles are only accepted with a 'wrong' verdict")
        item = spot["items"][idx]
        snap, corr = item["_page"], item["_corr"]
        rec = build_record(snap, corr, "spot_check", session_id=self.session_id,
                           human={"verdict": verdict, "rects": rects},
                           extra={"spot_check": {"seed": spot["seed"], "sample_index": idx,
                                                 "sample_size": spot["sample_size"],
                                                 "population_size": spot["population_size"],
                                                 "population_sha256": spot["population_sha256"],
                                                 "verdict": verdict}})
        rec["shown"]["client_agrees"] = None
        self.store.append(rec)
        out = self._after_write(snap, corr, rec)
        out["sample_index"] = idx
        return out

    def retract(self, data: dict) -> dict:
        rid = _str_field(data, "record_id", max_len=32)
        if not re.fullmatch(r"[0-9a-f]{32}", rid):
            raise FeedbackError(400, "record_id must be 32 hex characters")
        records = self.store.read()[0]
        target = next((r for r in records if r["record_id"] == rid and r["action"] != "retract"), None)
        if target is None:
            raise FeedbackError(404, "unknown record")
        if any(r.get("action") == "retract" and r.get("retracts") == rid for r in records):
            raise FeedbackError(409, "record already retracted")
        if target.get("book") != self.ws.slug:
            raise FeedbackError(404, "record belongs to another book")
        rec = {k: copy.deepcopy(target.get(k)) for k in REQUIRED_FIELDS}
        rec.update(record_id=uuid.uuid4().hex, ts=_now(), action="retract",
                   session_id=self.session_id, retracts=rid,
                   human={"verdict": None, "rects": []})
        self.store.append(rec)
        eff = effective(self.store.read()[0])
        return {"ok": True, "record_id": rec["record_id"], "retracts": rid,
                "page": target["page"], "key": target.get("view_key"),
                "correction_key": target["correction_key"],
                **summarize_correction(eff, target["image"]["sha256"], target["correction_key"])}

    # -- text edits (Accept) ----------------------------------------------

    def accept_snapshot(self, n: int) -> Optional[dict]:
        """Taken BEFORE Accept rewrites the page, so keys bind the pre-edit text."""
        if n not in self.surfaced:
            return None
        return page_snapshot(self.ws, n)

    def record_accept(self, snap: Optional[dict], on_disk_text: str, new_text: str,
                      hunk_decisions: list) -> int:
        """One text_edit record per changed correction span (approved/edited
        hunks) plus one per free-form edit chunk outside them. Returns count."""
        if snap is None or new_text == on_disk_text or snap["text"] != on_disk_text:
            return 0
        by_id = {}
        for d in hunk_decisions or []:
            if isinstance(d, dict):
                try:
                    by_id[int(d.get("id"))] = d
                except (TypeError, ValueError):
                    continue
        covered: list[tuple[int, int]] = []
        written = 0
        for corr in snap["corrections"].values():
            t = corr["target"]
            if t["hunk_id"] is None:
                continue
            d = by_id.get(t["hunk_id"]) or {}
            decision = d.get("decision")
            if decision == "approved" and t["suggested"] is not None:
                final = t["suggested"]
            elif decision == "edited" and isinstance(d.get("text"), str):
                final = d["text"]
            else:
                continue
            if final == t["old"]:
                continue
            if t["span"]:
                covered.append(tuple(t["span"]))
            rec = build_record(snap, corr, "text_edit", session_id=self.session_id,
                               human={"verdict": decision, "rects": []},
                               extra={"text_edit": {"decision": decision, "old": t["old"],
                                                    "new": final, "span": t["span"],
                                                    "scope": "correction"}})
            self.store.append(rec)
            written += 1
            if written >= MAX_TEXT_EDITS_PER_ACCEPT:
                return written
        for start, end, old, new in _edit_chunks(on_disk_text, new_text):
            if any(start < ce and cs < end or (start == end and cs <= start <= ce)
                   for cs, ce in covered):
                continue
            ck = _id("t", snap["page_id"], snap["markdown_sha256"], start, end, old, new)
            pseudo = {"view_key": None, "correction_key": ck,
                      "target": {"kind": "free_edit", "old": old, "suggested": None,
                                 "issue_type": None, "issue_description": None,
                                 "span": [start, end], "hunk_id": None, "query": None},
                      "shown": None, "engine": None}
            rec = build_record(snap, pseudo, "text_edit", session_id=self.session_id,
                               human={"verdict": "free_edit", "rects": []},
                               extra={"text_edit": {"decision": "free_edit", "old": old, "new": new,
                                                    "span": [start, end], "scope": "free"}})
            self.store.append(rec)
            written += 1
            if written >= MAX_TEXT_EDITS_PER_ACCEPT:
                break
        return written


_TOKEN_RE = re.compile(r"\S+\s*|\s+")


def _edit_chunks(a: str, b: str) -> list[tuple[int, int, str, str]]:
    """Token-level changed chunks (start, end in ``a``, old text, new text)."""
    import difflib
    ta = [m.span() for m in _TOKEN_RE.finditer(a)]
    tb = [m.span() for m in _TOKEN_RE.finditer(b)]
    wa = [a[s:e] for s, e in ta]
    wb = [b[s:e] for s, e in tb]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        start = ta[i1][0] if i1 < len(ta) else len(a)
        end = ta[i2 - 1][1] if i2 > i1 else start
        out.append((start, end, a[start:end], "".join(wb[j1:j2])))
    return out


# ---------------------------------------------------------------------------
# HTTP glue (called from review.py's handler; returns (status, json body))
# ---------------------------------------------------------------------------


def handle_get(session: Optional[FeedbackSession], path: str) -> tuple[int, dict]:
    if session is None:
        return 404, {"ok": False, "error": "feedback capture is not enabled"}
    try:
        if path == "/feedback/spot":
            return 200, session.public_spot(session.spot_sample())
        m = re.fullmatch(r"/feedback/(\d{1,6})", path)
        if not m:
            raise FeedbackError(404, "not found")
        return 200, session.page_feedback(int(m.group(1)))
    except FeedbackError as exc:
        return exc.status, {"ok": False, "error": exc.message}


def handle_post(session: Optional[FeedbackSession], path: str, headers, rfile) -> tuple[int, dict]:
    if session is None:
        return 404, {"ok": False, "error": "feedback capture is not enabled"}
    try:
        routes = {"/feedback": session.record_box_action, "/feedback/spot": session.record_spot,
                  "/feedback/undo": session.retract}
        handler = routes.get(path)
        if handler is None:
            raise FeedbackError(404, "not found")
        return 200, handler(parse_body(headers, rfile))
    except FeedbackError as exc:
        return exc.status, {"ok": False, "error": exc.message}
