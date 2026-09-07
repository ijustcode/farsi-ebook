"""Unified image placement, evidence acquisition, and bounded replay.

Only successful unmarked image readings are durable evidence. No PDF-position
guess, old derived rectangle, QC coordinate, or benchmark annotation is used.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import fitz

from . import llm, locate
from .config import MODEL_STRONG, PRICES, LONG_EDGE_HI
from .page_map import (DERIVATION_VERSION, PlacementResult, PrintedWord, place,
                       supported_groups, supported_windows, target_span, insertion_slot)

RENDER_VERSION = 1
DETECTOR = "projection_v2_wordgaps"
DEFAULT_MAX_COST = 5.0
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_LOCK_GUARD = threading.Lock()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _claim(path: Path):
    """Same-key claims coordinate both threads and concurrent review processes."""
    with _LOCK_GUARD:
        lock = _THREAD_LOCKS.setdefault(str(path.resolve()), threading.RLock())
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _valid_rect(rect):
    if isinstance(rect, dict):
        rect = [rect.get(k) for k in ("x0", "y0", "x1", "y1")]
    return (isinstance(rect, (list, tuple)) and len(rect) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in rect)
            and 0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1)


def _cached_result(data):
    if not isinstance(data, dict) or data.get("status") not in {"located", "buffered"}:
        return None
    box = data.get("box")
    if not isinstance(box, dict) or box.get("source") != "scan" or not _valid_rect(box):
        return None
    segments = box.get("segments")
    if not isinstance(segments, list) or not segments or not all(isinstance(r, dict) and _valid_rect(r) for r in segments):
        return None
    for k, fn in (("x0", min), ("y0", min), ("x1", max), ("y1", max)):
        if abs(box[k] - fn(r[k] for r in segments)) > 1e-7:
            return None
    for k in ("buffer_before", "buffer_after"):
        if type(data.get(k, 0)) is not int or not 0 <= data.get(k, 0) <= 2:
            return None
        if type(box.get(k, 0)) is not int or box.get(k, 0) != data.get(k, 0):
            return None
    try:
        result = PlacementResult(**data)
    except (TypeError, ValueError):
        return None
    return result


def _cached_words(cached, key):
    if cached.get("key") != key or not isinstance(cached.get("words"), list):
        return []
    words = []
    for w in cached["words"]:
        if (not isinstance(w, dict) or not isinstance(w.get("text"), str) or not w["text"]
                or not (_valid_rect(w.get("rect")) or (w.get("supported") is False and w.get("rect") == [0,0,0,0])) or type(w.get("line")) is not int or w["line"] < 0
                or type(w.get("supported")) is not bool or not isinstance(w.get("evidence"), list)
                or not all(isinstance(e, str) for e in w["evidence"])):
            return []
        try:
            words.append(PrintedWord(**w))
        except TypeError:
            return []
    return words


class Budget:
    def __init__(self, limit=DEFAULT_MAX_COST):
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("bbox max cost must be a finite nonnegative amount")
        self.limit, self.spent, self.reserved, self.uncertain = limit, 0.0, 0.0, 0.0
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + self.uncertain + amount > self.limit + 1e-12:
                return False
            self.reserved += amount
            return True

    def settle(self, amount, actual):
        with self.lock:
            self.reserved = max(0., self.reserved-amount)
            if actual is None:
                self.uncertain += amount  # don't recycle possibly billed requests
            else:
                self.spent += actual

    def snapshot(self):
        with self.lock:
            return {"limit": self.limit, "spent": self.spent,
                    "reserved": self.reserved, "uncertain": self.uncertain}


class PlacementService:
    def __init__(self, ws, model=MODEL_STRONG, *, mode="auto", max_cost=DEFAULT_MAX_COST,
                 reader=None):
        if mode not in {"auto", "offline"}:
            raise ValueError("bbox mode must be auto or offline")
        if model not in PRICES:
            raise ValueError(f"No price configured for bbox model {model!r}; cannot enforce budget")
        self.ws, self.model, self.mode = ws, model, mode
        self.root = ws.root / "locate_evidence"
        self.budget = Budget(max_cost)
        self.reader = reader
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self._done = {}
        self._failures = {}
        self._client = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bbox")
        self._scheduled = set()
        self.run_id = uuid.uuid4().hex
        self._acquisition_error = None
        self._source_stat = None
        self._source_hash = None
        self.prompt_hash = hashlib.sha256(llm.READ_REGIONS_SYSTEM.encode()).hexdigest()

    def source_hash(self):
        stat = self.ws.pdf_path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)
        with self.lock:
            if stamp != self._source_stat:
                self._source_hash = hashlib.sha256(self.ws.pdf_path.read_bytes()).hexdigest()
                self._source_stat = stamp
            return self._source_hash

    def _page_key(self, n):
        return digest([self.source_hash(), n, self.model, RENDER_VERSION, LONG_EDGE_HI,
                       fitz.VersionBind, self.prompt_hash, llm.REGION_READER_VERSION,
                       DETECTOR, DERIVATION_VERSION])

    def _key(self, n, md, q):
        page_key = self._page_key(n)
        revision = _read_json(self.root / "pages" / f"{page_key}.json").get("revision")
        return digest([page_key, revision, md, asdict(q)])

    def _observation_key(self, png):
        return digest([hashlib.sha256(png).hexdigest(), self.model, self.prompt_hash,
                       llm.REGION_READER_VERSION, RENDER_VERSION, fitz.VersionBind])

    @staticmethod
    def _valid(ob):
        return (isinstance(ob, dict) and ob.get("legible") is True
                and ob.get("complete") is True and isinstance(ob.get("lines"), list)
                and bool(ob["lines"]) and all(isinstance(s, str) and s.strip() for s in ob["lines"]))

    def _observation(self, key):
        ob = _read_json(self.root / "readings" / f"{key}.json")
        return ob if self._valid(ob) and ob.get("key") == key else None

    def _read(self, pngs, page_no, allow_api):
        keys = [self._observation_key(png) for png in pngs]
        observations = [self._observation(k) for k in keys]
        if not allow_api or self.cancelled.is_set():
            return observations, keys
        missing = sorted({k for k, ob in zip(keys, observations) if ob is None})
        if not missing:
            return observations, keys
        if self._acquisition_error:
            raise RuntimeError(self._acquisition_error)
        with ExitStack() as stack:
            for key in missing:
                stack.enter_context(_claim(self.root / "readings" / f"{key}.json"))
            missing = [k for k in missing if self._observation(k) is None]
            for offset in range(0, len(missing), 4):
                if self.cancelled.is_set():
                    break
                batch_keys = missing[offset:offset+4]
                batch = [pngs[keys.index(k)] for k in batch_keys]
                prices = PRICES[self.model]
                # Conservative upper bound from encoded image bytes + schema/
                # prompt overhead; cap every request, including retries. No SDK
                # retries are allowed behind the budget owner's back.
                reserved = ((sum(len(b)*4/3 for b in batch) + 16000) * prices["in"]
                            + llm.REGION_OUTPUT_TOKENS * prices["out"]) / 1_000_000
                if not self.budget.reserve(reserved):
                    raise RuntimeError("budget_exhausted")
                actual = None
                try:
                    if self.reader is not None:
                        parsed, _usage, actual = self.reader(batch, self.model, page_no)
                    else:
                        with self.lock:
                            if self._client is None:
                                self._client = llm.get_client().with_options(max_retries=0, timeout=60.)
                        parsed, _usage, actual = llm.read_regions(self._client, batch, self.model, page_no)
                    rows = parsed.regions if parsed is not None and hasattr(parsed, "regions") else (parsed or [])
                    by_id = {}
                    duplicated = set()
                    for row in rows:
                        ob = row.model_dump() if hasattr(row, "model_dump") else row
                        idx = ob.get("region_index")
                        if idx in by_id:
                            duplicated.add(idx)
                        by_id[idx] = ob
                    for i, key in enumerate(batch_keys):
                        ob = by_id.get(i)
                        if i in duplicated or not self._valid(ob):
                            # Failed observations are diagnostics, never reusable
                            # readings. Preserve why a bounded retry was needed.
                            with self.lock:
                                self._failures.setdefault("regions", []).append({
                                    "page": page_no, "key": key,
                                    "reason": "duplicate_region" if i in duplicated else "invalid_reading",
                                    "legible": ob.get("legible") if isinstance(ob, dict) else None,
                                    "complete": ob.get("complete") if isinstance(ob, dict) else None,
                                    "line_count": len(ob["lines"]) if isinstance(ob, dict) and isinstance(ob.get("lines"), list) else None,
                                })
                            continue
                        ob = {**ob, "key": key, "image_sha256": hashlib.sha256(batch[i]).hexdigest(),
                              "model": self.model, "prompt_sha256": self.prompt_hash,
                              "schema_version": llm.REGION_READER_VERSION,
                              "render_version": RENDER_VERSION, "render_engine": fitz.VersionBind}
                        _write_json(self.root / "readings" / f"{key}.json", ob)
                except llm.anthropic.APIStatusError as exc:
                    if exc.status_code in {400, 401, 403, 404, 413, 422, 429}:
                        actual = 0.0  # request rejected before inference
                    error = exc.body.get("error", {}) if isinstance(exc.body, dict) else {}
                    with self.lock:
                        if exc.status_code in {401, 403}:
                            self._acquisition_error = "reader_access_denied"
                        elif exc.status_code == 400 and "credit balance" in str(error.get("message", "")).lower():
                            self._acquisition_error = "reader_credit_unavailable"
                        self._failures["reader"] = {"error_type": type(exc).__name__,
                            "status_code": exc.status_code,
                            "message": str(error.get("message", "request rejected"))[:500]}
                    raise
                finally:
                    self.budget.settle(reserved, actual)
                    receipt = {"run_id": self.run_id, "model": self.model,
                               "updated_at": time.time(), **self.budget.snapshot()}
                    _write_json(self.root / "runs" / f"{self.run_id}.json", receipt)
                    _write_json(self.root / "last_run_cost.json", receipt)
        return [self._observation(k) for k in keys], keys

    @staticmethod
    def _render(page, rect, factor=1.):
        scale = LONG_EDGE_HI / max(page.rect.width, page.rect.height) * factor
        clip = fitz.Rect(rect) & page.rect
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                               alpha=False).tobytes("png")

    @staticmethod
    def _retry_line_clip(page_rect, lines, index):
        """Acquire detached ink without turning crop padding into word evidence.

        Adjacent detected bodies bound the expansion. A fused or overlapping
        pair is still rejected by the reader; padding cannot certify a split.
        """
        r = fitz.Rect(lines[index].rect)
        pad = r.height * .4
        top = max(page_rect.y0, r.y0-pad)
        bottom = min(page_rect.y1, r.y1+pad)
        if index and lines[index-1].rect.y1 <= r.y0:
            top = max(top, (lines[index-1].rect.y1+r.y0)/2)
        if index+1 < len(lines) and lines[index+1].rect.y0 >= r.y1:
            bottom = min(bottom, (r.y1+lines[index+1].rect.y0)/2)
        return fitz.Rect(max(page_rect.x0, r.x0-2), top,
                         min(page_rect.x1, r.x1+2), bottom)

    def page_map_summary(self, n):
        return _read_json(self.root / "pages" / f"{self._page_key(n)}.json").get("summary", {})

    def ensure_page_map(self, n, *, allow_api=None):
        """Acquire reusable evidence independently of Markdown and corrections."""
        allow_api = self.mode == "auto" if allow_api is None else allow_api
        key = self._page_key(n)
        with _claim(self.root / "pages" / f"{key}.json"):
            cached = _read_json(self.root / "pages" / f"{key}.json")
            words = _cached_words(cached, key)
            if cached.get("complete") and cached.get("summary") and (words or
                    cached["summary"].get("status") == "blank_nontext" and cached.get("words") == []):
                return words, cached["summary"]
            with fitz.open(self.ws.pdf_path) as doc:
                page = doc[n-1]
                embedded = locate._page_words(page)
                if locate._tier_a_usable(embedded):
                    words = []
                    groups = locate._cluster_rect_lines([r for r, _ in embedded])
                    for line_id, group in enumerate(groups):
                        for rect in sorted(group, key=lambda r: -r.x1):
                            text = next(t for r, t in embedded if r == rect)
                            nr = locate._rect_to_fracs(rect * page.rotation_matrix, page.rect)
                            words.append(PrintedWord(text, [nr[k] for k in ("x0", "y0", "x1", "y1")], line_id, ["pdf_words"]))
                    rows = [asdict(w) for w in words]
                    summary = {"status": "complete", "detected_lines": len(groups), "read_lines": len(groups),
                               "words": len(words), "verified_words": len(words), "failures": [], "source": "match"}
                    if self._page_key(n) != key:
                        raise RuntimeError("source_changed")
                    _write_json(self.root / "pages" / f"{key}.json", {"key": key, "words": rows,
                                "revision": digest(rows), "complete": True, "summary": summary,
                                "normalized_words": [locate._fold_word(w.text) for w in words],
                                "word_ids": [digest([key, w.line, j]) for j, w in enumerate(words)]})
                    return words, summary
            self._build_map(n, "", [], allow_api)
            cached = _read_json(self.root / "pages" / f"{key}.json")
            return _cached_words(cached, key), cached.get("summary", {})

    def _build_map(self, n, md, queries, allow_api):
        page_key = self._page_key(n)
        with fitz.open(self.ws.pdf_path) as doc:
            page = doc[n-1]
            lines = locate._scan_page_lines(page)
            map_path = self.root / "pages" / f"{page_key}.json"
            previous = _cached_words(_read_json(map_path), page_key)
            if not previous:
                legacy_key = digest([self.source_hash(), n, self.model, RENDER_VERSION, LONG_EDGE_HI,
                                     fitz.VersionBind, self.prompt_hash, llm.REGION_READER_VERSION, DETECTOR, 4])
                previous = _cached_words(_read_json(self.root / "pages" / f"{legacy_key}.json"), legacy_key)
            blocked = []
            def read(pngs, number, api):
                try:
                    return self._read(pngs, number, api and not blocked)
                except Exception as exc:
                    _write_json(self.root / "attempts" / f"{self.run_id}-{n}.json",
                                {"page": n, "error_type": type(exc).__name__, "time": time.time()})
                    blocked.append(self._acquisition_error or ("budget_exhausted" if str(exc) == "budget_exhausted" else "api_failure"))
                    return self._read(pngs, number, False)
            if not lines:
                summary = {"status": "blank_nontext", "detected_lines": 0,
                           "read_lines": 0, "words": 0, "verified_words": 0, "failures": []}
                _write_json(map_path, {"key": page_key, "words": [], "complete": True,
                                     "summary": summary, "revision": digest([])})
                return [], False
            # Full-width crops retain neighboring detached letters and dots;
            # explicit line identity is fixed before recognition.
            clips = []
            for line in lines:
                r = fitz.Rect(line.rect)
                r.x0 = max(page.rect.x0, r.x0-2)
                r.x1 = min(page.rect.x1, r.x1+2)
                r.y0 = max(page.rect.y0, r.y0-1)
                r.y1 = min(page.rect.y1, r.y1+1)
                clips.append(r)
            observations, line_keys = read([self._render(page, r) for r in clips], n, allow_api)
            # Older successful high-resolution observations remain compatible.
            # Replay them before acquiring a differently padded retry crop.
            old_retry = [i for i, ob in enumerate(observations) if ob is None or len(ob["lines"]) != 1]
            if old_retry:
                old_obs, old_keys = read([self._render(page, clips[i], 1.5)
                                               for i in old_retry], n, False)
                for i, ob, key in zip(old_retry, old_obs, old_keys):
                    if ob is not None and len(ob["lines"]) == 1:
                        observations[i], line_keys[i] = ob, key
            # Retry once with room for detached marks and higher resolution.
            retry = [i for i, ob in enumerate(observations) if ob is None or len(ob["lines"]) != 1]
            if retry:
                retry_clips = [self._retry_line_clip(page.rect, lines, i) for i in retry]
                obs2, keys2 = read([self._render(page, r, 1.5) for r in retry_clips], n, allow_api)
                for i, ob, key in zip(retry, obs2, keys2):
                    if ob is not None and len(ob["lines"]) == 1:
                        observations[i], line_keys[i] = ob, key
            # A second, overlapping contextual read can recover a missing line.
            # Independently read neighbors must agree exactly; line count alone
            # never establishes identity, and ambiguous/split regions stay local.
            contextual = [i for i, ob in enumerate(observations)
                          if ob is None or len(ob["lines"]) != 1]
            for i in contextual:
                lo, hi = max(0, i-1), min(len(lines)-1, i+1)
                neighbors = [j for j in range(lo, hi+1) if j != i]
                if not neighbors or any(observations[j] is None or len(observations[j]["lines"]) != 1 for j in neighbors):
                    continue
                region = locate._union_rects([self._retry_line_clip(page.rect, lines, j) for j in range(lo, hi+1)])
                context_obs, context_keys = read([self._render(page, region, 1.5)], n, allow_api)
                ob = context_obs[0]
                if ob is None or len(ob["lines"]) != hi-lo+1:
                    continue
                if all(locate._norm_words(ob["lines"][j-lo]) == locate._norm_words(observations[j]["lines"][0]) for j in neighbors):
                    observations[i] = {**ob, "lines": [ob["lines"][i-lo]]}
                    line_keys[i] = context_keys[0]
            complete = all(ob is not None and len(ob["lines"]) == 1 for ob in observations)
            if not complete:
                with self.lock:
                    self._failures[page_key] = {"reason": "page_reading_incomplete",
                        "line_indices": [i for i, ob in enumerate(observations)
                                         if ob is None or len(ob["lines"]) != 1]}
                # Keep unreadable lines as explicit barriers. Unrelated
                # unreadable text must not discard independently read lines.
            words = []
            for i, (line, ob) in enumerate(zip(lines, observations)):
                rect = locate._rect_to_fracs(line.rect, page.rect)
                tokens = (locate._norm_words(ob["lines"][0])
                          if ob is not None and len(ob["lines"]) == 1 else ["�"])
                for token in tokens:
                    words.append(PrintedWord(token, [rect[k] for k in ("x0","y0","x1","y1")],
                                             i, [line_keys[i]], False))
            needed = set(range(len(lines)))
            verified = {}
            def persist():
                result = []
                for line_id in range(len(lines)):
                    current = verified.get(line_id) or [w for w in words if w.line == line_id]
                    old = [w for w in previous if w.line == line_id]
                    if [w.text for w in old] == [w.text for w in current]:
                        current = [a if a.supported else b for a, b in zip(current, old)]
                    elif old and (any(w.supported for w in old) or all(w.text == "�" for w in current)):
                        # Conflicting new text cannot erase previously supported
                        # identities. Keep that line until evidence reconciles it.
                        current = old
                    result.extend(current)
                failures = [{"line": i, "reason": "missing_reading" if any(w.text == "�" for w in result if w.line == i)
                             else "missing_word_geometry"} for i in range(len(lines))
                            if any(not w.supported for w in result if w.line == i)]
                done = not failures
                summary = {"status": "complete" if done else "acquisition_blocked" if blocked or self.cancelled.is_set() else "partial",
                           "detected_lines": len(lines),
                           "read_lines": sum(not any(w.text == "�" for w in result if w.line == i) for i in range(len(lines))),
                           "words": sum(w.text != "�" for w in result),
                           "verified_words": sum(w.supported for w in result),
                           "failures": failures, "acquisition_reasons": sorted(set(blocked + (["cancelled"] if self.cancelled.is_set() else [])))}
                if self._page_key(n) != page_key:
                    raise RuntimeError("source_changed")
                rows = [asdict(w) for w in result]
                originals = {}
                for line_id, observation in enumerate(observations):
                    if observation and len(observation["lines"]) == 1:
                        tokens = [t for t in observation["lines"][0].split() if locate._fold_word(t)]
                        if [locate._fold_word(t) for t in tokens] == [w.text for w in result if w.line == line_id]:
                            originals[line_id] = tokens
                word_text = [{"original": originals.get(i, [None] * len([w for w in result if w.line == i]))[j],
                              "normalized": w.text}
                             for i in range(len(lines)) for j, w in enumerate(w for w in result if w.line == i)]
                _write_json(map_path, {"key": page_key, "words": rows, "revision": digest(rows),
                            "word_text": word_text, "summary": summary, "complete": done, "reading_complete": summary["read_lines"] == len(lines),
                            "word_ids": [digest([page_key, i, j, w.text]) for i in range(len(lines)) for j, w in enumerate(w for w in result if w.line == i)],
                            "lines": [{"id": i, "text": observations[i]["lines"][0] if observations[i] else None} for i in range(len(lines))],
                            "detector": DETECTOR, "coordinate_space": "normalized_original_page"})
                return result
            persist()
            for i in sorted(needed):
                if observations[i] is None or len(observations[i]["lines"]) != 1:
                    continue
                old = [w for w in previous if w.line == i]
                if old and all(w.supported for w in old) and [w.text for w in old] == locate._norm_words(observations[i]["lines"][0]):
                    verified[i] = old
                    continue
                line = lines[i]
                # Each gap-derived physical group is independently read. Equal
                # token counts alone never make these rectangles trustworthy.
                # Coalesce nearby fragments into physical groups before paying
                # for recognition. Identity still requires independent text.
                group_clips = []
                expanded = self._retry_line_clip(page.rect, lines, i)
                for r in line.words:
                    c = fitz.Rect(r)
                    c.y0, c.y1 = expanded.y0, expanded.y1
                    if group_clips and group_clips[-1].x0 - c.x1 < line.rect.height * .25:
                        group_clips[-1] |= c
                    else:
                        group_clips.append(c)
                physical_groups = [fitz.Rect(r) for r in group_clips]
                # Short groups retain linguistic context and amortize the
                # reader overhead without implying word-count geometry.
                group_clips = [locate._union_rects(physical_groups[j:j+2])
                               for j in range(0, len(physical_groups), 2)]
                group_clips = [fitz.Rect(max(page.rect.x0, r.x0-1), r.y0,
                                         min(page.rect.x1, r.x1+1), r.y1) for r in group_clips]
                group_obs, group_keys = read([self._render(page, r) for r in group_clips], n, allow_api)
                missing = [j for j, ob in enumerate(group_obs) if ob is None or len(ob["lines"]) != 1]
                if missing:
                    obs2, keys2 = read([self._render(page, group_clips[j], 1.5) for j in missing], n, allow_api)
                    for j, ob, key in zip(missing, obs2, keys2):
                        if ob is not None and len(ob["lines"]) == 1:
                            group_obs[j], group_keys[j] = ob, key
                groups = []
                for ob, r, key in zip(group_obs, group_clips, group_keys):
                    if ob is None or len(ob["lines"]) != 1:
                        continue
                    nr = locate._rect_to_fracs(r, page.rect)
                    groups.append((ob["lines"][0], [nr[k] for k in ("x0","y0","x1","y1")], key))
                verified[i] = supported_groups(observations[i]["lines"][0], groups, i, line_keys[i])
                if not verified[i]:
                    windows = []
                    for length in (1,3):
                        for start in range(len(physical_groups)-length+1):
                            r = locate._union_rects(physical_groups[start:start+length])
                            r.y0, r.y1 = expanded.y0, expanded.y1
                            windows.append(r)
                    obs3, keys3 = read([self._render(page,r) for r in windows],n,allow_api)
                    for ob,r,key in zip(obs3,windows,keys3):
                        if ob is not None and len(ob['lines']) == 1:
                            nr=locate._rect_to_fracs(r,page.rect)
                            groups.append((ob['lines'][0],[nr[k] for k in ('x0','y0','x1','y1')],key))
                    verified[i] = supported_windows(observations[i]['lines'][0],groups,i,line_keys[i])
                persist()
            result = persist()
            return result, bool(result)

    def results(self, n, md, queries, boxes=None):
        if boxes is None:
            boxes = locate.locate_queries(self.ws.pdf_path, n, md, queries)
        out = []
        for q, box in zip(queries, boxes):
            if box is not None and box.get("source") == "match":
                out.append(PlacementResult(box.get("status", "located"), box,
                                           evidence=["pdf_words"],
                                           buffer_before=box.get("buffer_before", 0),
                                           buffer_after=box.get("buffer_after", 0),
                                           kind=box.get("kind", "words")))
                continue
            key = self._key(n, md, q)
            with self.lock:
                result = self._done.get(key)
            cached = _read_json(self.root / "boxes" / f"{key}.json")
            if result is None and cached.get("key") == key:
                result = _cached_result(cached.get("result"))
            out.append(result or PlacementResult("pending", reason="evidence_pending"))
        return out

    def apply_cached(self, n, md, queries, boxes):
        return [r.box for r in self.results(n, md, queries, boxes)]

    def pending(self, n, md, queries, boxes):
        return any(r.status == "pending" for r in self.results(n, md, queries, boxes))

    def refine(self, n, md, queries, boxes, *, allow_api=None):
        allow_api = self.mode == "auto" if allow_api is None else allow_api
        results = self.results(n, md, queries, boxes)
        missing = [i for i, r in enumerate(results) if r.status == "pending"]
        if not missing:
            return [r.box for r in results]
        page_key = self._page_key(n)
        with _LOCK_GUARD:
            lock = _THREAD_LOCKS.setdefault(str(self.root / page_key), threading.RLock())
        with lock:
            try:
                cached = _read_json(self.root / "pages" / f"{page_key}.json")
                words = _cached_words(cached, page_key)
                # Cached page geometry can answer a new Markdown/query for free.
                for i in missing:
                    if words:
                        result = place(md, queries[i], words, source=cached.get("summary", {}).get("source", "scan"))
                        if result.box is not None:
                            results[i] = result
                still = [i for i in missing if results[i].box is None]
                if still:
                    words, summary = self.ensure_page_map(n, allow_api=allow_api)
                    complete = bool(words)
                    for i in still:
                        results[i] = (place(md, queries[i], words, source=summary.get("source", "scan")) if complete
                                      else PlacementResult("unresolved", reason="page_reading_incomplete"))
                        if results[i].box is None and summary.get("acquisition_reasons"):
                            results[i].reason += ": " + ", ".join(summary["acquisition_reasons"])
            except Exception as exc:
                with self.lock:
                    self._failures[page_key] = {"error_type": type(exc).__name__, "time": time.time()}
                for i in missing:
                    if results[i].box is None:
                        reason = self._acquisition_error or ("budget_exhausted" if str(exc) == "budget_exhausted" else "evidence_unavailable")
                        results[i] = PlacementResult("unresolved", reason=reason)
            if self._page_key(n) != page_key or self.cancelled.is_set():
                return [None] * len(queries)
            for i in missing:
                key = self._key(n, md, queries[i])
                with self.lock:
                    self._done[key] = results[i]
                if results[i].box is not None and allow_api:
                    _write_json(self.root / "boxes" / f"{key}.json",
                                {"key": key, "result": results[i].to_dict()})
        return [r.box for r in results]

    def replay(self, n, md, queries, boxes):
        return self.refine(n, md, queries, boxes, allow_api=False)

    def schedule(self, n, md, queries, boxes):
        """Deduplicate pending requests, including queries changed after an edit."""
        key = digest([self._key(n, md, q) for q in queries])
        with self.lock:
            if self.cancelled.is_set() or key in self._scheduled:
                return
            self._scheduled.add(key)
        def work():
            try:
                self.refine(n, md, queries, boxes)
            finally:
                with self.lock:
                    self._scheduled.discard(key)
        try:
            self._executor.submit(work)
        except RuntimeError:
            with self.lock:
                self._scheduled.discard(key)

    def close(self):
        self.cancelled.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
