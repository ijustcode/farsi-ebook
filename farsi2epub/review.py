"""Interactive human-review workflow for correcting LLM transcriptions.

Serves a small local web app (stdlib http.server + jinja2, no external
network resources) that lets a human read each flagged page's source image
next to its transcribed Markdown, decide on each QC-suggested correction
hunk individually, edit the text freely, and accept the page before EPUB
assembly.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import json
import math
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from jinja2 import Environment
from markupsafe import Markup

from . import llm, qc
from .config import MODEL_STRONG
from .locate import Query, locate_queries, refine_scan_boxes
from .workspace import PROJECT_ROOT, Workspace

DEFAULT_PORT = 8765
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"

# ---------------------------------------------------------------------------
# sidecar helpers
# ---------------------------------------------------------------------------


def _read_sidecar(ws: Workspace, n: int) -> dict:
    path = ws.page_meta_path(n)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_sidecar(ws: Workspace, n: int, data: dict) -> None:
    """Write the sidecar JSON atomically (write to a temp file, then replace)."""
    path = ws.page_meta_path(n)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    tmp_path.replace(path)


def _ensure_orig_backup(ws: Workspace, n: int, text: str) -> bool:
    """Back up the pre-review transcription to text/NNNN.orig.md, atomically,
    only if no backup exists yet (the first human save wins; later saves keep
    the original original). Returns True if a backup was written now.
    """
    path = ws.page_orig_path(n)
    if path.exists():
        return False
    _write_text_atomic(path, text)
    return True


def _image_path_for(ws: Workspace, n: int) -> Optional[Path]:
    hi = ws.page_hires_path(n)
    if hi.is_file():
        return hi
    std = ws.page_image_path(n)
    if std.is_file():
        return std
    return None


# ---------------------------------------------------------------------------
# QC suggestion helpers
# ---------------------------------------------------------------------------


def _pending_suggestion(sidecar: dict) -> Optional[dict]:
    """Return the sidecar's `qc` dict if it has a pending, actionable
    suggestion (verdict fail, suggested text present, not yet resolved by a
    human), else None.
    """
    qc_data = sidecar.get("qc")
    if not qc_data:
        return None
    if (
        qc_data.get("verdict") == "fail"
        and qc_data.get("suggested_text_md") is not None
        and qc_data.get("suggestion_status") == "pending"
    ):
        return qc_data
    return None


def _first_qc_issue_type(sidecar: dict) -> str:
    issues = (sidecar.get("qc") or {}).get("issues") or []
    return issues[0]["type"] if issues else "human_edit"


def _json_for_script(obj) -> Markup:
    """Serialize `obj` as JSON safe to inline inside a
    <script type="application/json"> element: pure ASCII, and with every "<"
    escaped as \\u003c (a valid JSON escape) so neither "</script" nor "<!--"
    can appear in the page source.
    """
    encoded = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return Markup(encoded.replace("<", "\\u003c"))


# ---------------------------------------------------------------------------
# QC box overlays (tiered locate.py match/layout/scan, VLM-refined scan_vlm,
# model bbox fallback)
# ---------------------------------------------------------------------------

_LOCATE_VLM_CACHE_VERSION = 2  # v2 preserves ordered per-line segments


@functools.lru_cache(maxsize=512)
def _locate_queries_cached(
    pdf_path: str, page_no: int, page_md: str, queries: tuple[Query, ...]
) -> tuple:
    """Memoized locate.locate_queries so browser refreshes don't rescan the
    PDF. Keyed on (pdf path, page, page markdown, queries) — page_md in the key
    means an Accept that rewrites the page re-derives fresh boxes on the next
    GET. Any failure degrades to no boxes.
    """
    try:
        return tuple(locate_queries(pdf_path, page_no, page_md, list(queries)))
    except Exception:
        return (None,) * len(queries)


def _nth_index(hay: str, needle: str, nth: int) -> int:
    """0-based start of the `nth` (1-based) occurrence of `needle`, or -1.
    Mirrors the JS nthIndexOf used by the hunk-apply logic."""
    idx = -1
    for _ in range(nth):
        idx = hay.find(needle, idx + 1)
        if idx == -1:
            return -1
    return idx


def _cap_words(s: str, n: int = 5) -> str:
    """Prefix of `s` covering at most `n` whitespace-delimited words, preserving
    the original characters (so char offsets/ZWNJ still line up with the source).
    A long flagged phrase would otherwise locate to a whole PDF line, which the
    click-to-zoom cannot zoom into.
    """
    matches = list(re.finditer(r"\S+", s))
    if len(matches) <= n:
        return s
    return s[: matches[n - 1].end()]


def _bbox_to_box(bbox) -> Optional[dict]:
    """Convert a sidecar model bbox ([x0,y0,x1,y1] in 0-1000 ints) to a
    0-1-fraction box dict with source "model"; None when absent/invalid.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) / 1000.0 for v in bbox)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "source": "model"}


class _ScanBoxRefiner:
    """Upgrades "scan"-sourced boxes to VLM-verified "scan_vlm" boxes via
    locate.refine_scan_boxes, billing each (page text, query) at most once per
    book: every outcome — including a None failure — is cached in
    books/<slug>/locate_vlm.json, keyed on a sha1 of (cache version, page_no,
    sha1(page_md), query text, alts, span). page_md in the key gives the same invalidation
    semantics as _locate_queries_cached: an Accept that rewrites the page
    re-refines fresh boxes on the next GET. Two paths share the cache:
    apply_cached (pure disk-cache lookup, never touches the API — the render
    path and /boxes route use it so GET / is instant) and refine (the
    API-calling warmer, driven by the background pipeline and bbox_eval). The
    instance lock guards only the cache dict, its atomic file save, and the
    in-flight key set — the API call itself runs outside it so several pages
    refine concurrently; keys registered in-flight are skipped by other
    callers instead of blocking (a later poll picks the results up). The
    anthropic client is created lazily on the first strip actually sent.
    """

    def __init__(self, ws: Workspace, model: str):
        self.ws = ws
        self.model = model
        self.cache_path = ws.root / "locate_vlm.json"
        self.lock = threading.Lock()
        self._cache: Optional[dict] = None
        self._client = None
        self._inflight: set[str] = set()

    def _load_cache(self) -> dict:
        if self._cache is None:
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        """Atomic cache write (temp file + os.replace), same as the sidecars."""
        tmp_path = self.cache_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        os.replace(tmp_path, self.cache_path)

    def _get_client(self):
        with self.lock:
            if self._client is None:
                self._client = llm.get_client()
            return self._client

    @staticmethod
    def _key(page_no: int, page_md: str, q: Query) -> str:
        md_sha = hashlib.sha1(page_md.encode("utf-8")).hexdigest()
        raw = json.dumps(
            [
                _LOCATE_VLM_CACHE_VERSION,
                page_no,
                md_sha,
                q.text,
                list(q.alts),
                list(q.span) if q.span else None,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _scan_indices(boxes: list[Optional[dict]]) -> list[int]:
        return [
            i for i, b in enumerate(boxes) if b is not None and b.get("source") == "scan"
        ]

    def apply_cached(
        self,
        page_no: int,
        page_md: str,
        queries: list[Query],
        boxes: list[Optional[dict]],
    ) -> list[Optional[dict]]:
        """Return `boxes` with scan-sourced entries upgraded to their cached
        scan_vlm box where one exists. Pure disk-cache lookup — never creates
        a client or calls the API, so it is safe (and fast) on the render
        path; uncached entries pass through as plain scan boxes.
        """
        scan_idx = self._scan_indices(boxes)
        if not scan_idx:
            return boxes
        out = list(boxes)
        with self.lock:
            cache = self._load_cache()
            for i in scan_idx:
                entry = cache.get(self._key(page_no, page_md, queries[i]))
                if entry and entry.get("box"):
                    out[i] = dict(entry["box"])
        return out

    def pending(
        self,
        page_no: int,
        page_md: str,
        queries: list[Query],
        boxes: list[Optional[dict]],
    ) -> bool:
        """True when any scan-sourced entry in `boxes` (the raw located boxes,
        before apply_cached upgrades) has no cache entry yet — i.e. the
        background pipeline still owes this page a refinement pass. Negative
        cache entries count as resolved.
        """
        scan_idx = self._scan_indices(boxes)
        if not scan_idx:
            return False
        with self.lock:
            cache = self._load_cache()
            return any(
                self._key(page_no, page_md, queries[i]) not in cache for i in scan_idx
            )

    def refine(
        self,
        page_no: int,
        page_md: str,
        queries: list[Query],
        boxes: list[Optional[dict]],
    ) -> list[Optional[dict]]:
        """Return `boxes` with scan-sourced entries upgraded to their refined
        scan_vlm box where refinement succeeded (cached or fresh); every other
        entry — match/layout hits, Nones — passes through untouched. Called by
        the background pipeline and tests/bbox_eval.py, never by the render
        path. Logs one stderr line per page that actually called the API;
        cache-only pages log nothing. Keys another thread is already refining
        are skipped, not waited on — the caller's next cache read sees them.
        """
        scan_idx = self._scan_indices(boxes)
        if not scan_idx:
            return boxes
        out = list(boxes)
        keys = {i: self._key(page_no, page_md, queries[i]) for i in scan_idx}
        with self.lock:
            cache = self._load_cache()
            for i in scan_idx:
                entry = cache.get(keys[i])
                if entry and entry.get("box"):
                    out[i] = dict(entry["box"])
            # Claim only keys nobody has cached or is currently refining; the
            # claim (not the whole API round-trip) is what the lock protects.
            uncached = {
                i
                for i in scan_idx
                if keys[i] not in cache and keys[i] not in self._inflight
            }
            if not uncached:
                return out
            self._inflight.update(keys[i] for i in uncached)

        try:
            total_cost = 0.0
            total_strips = 0

            def _read(strips: list[bytes]) -> list[list[str]]:
                nonlocal total_cost, total_strips
                lines, _usage, cost = llm.read_strips(
                    self._get_client(), strips, self.model, page_no
                )
                total_cost += cost
                total_strips += len(strips)
                return lines

            # Mask cached/in-flight positions so refine_scan_boxes only works
            # (and the VLM only reads strips for) positions we claimed. The
            # API call runs outside the lock so pages refine concurrently.
            masked: list[Optional[dict]] = [
                boxes[i] if i in uncached else None for i in range(len(boxes))
            ]
            refined = refine_scan_boxes(
                str(self.ws.pdf_path), page_no, page_md, queries, masked, _read
            )
            per_cost = total_cost / len(uncached)
            with self.lock:
                cache = self._load_cache()
                for i in uncached:
                    box = refined[i] if i < len(refined) else None
                    cache[keys[i]] = {"box": box, "cost": per_cost}
                    if box is not None:
                        out[i] = dict(box)
                self._save_cache()
            if total_strips:
                print(
                    f"bbox refine p{page_no}: {total_strips} strips, ${total_cost:.4f}",
                    file=sys.stderr,
                )
        finally:
            with self.lock:
                self._inflight.difference_update(keys[i] for i in uncached)
        return out


# Configured by run_review (None = refinement off: disabled by flag, no API
# key, or review.py used outside the server e.g. in tests).
_REFINER: Optional[_ScanBoxRefiner] = None


def _box_specs(
    issues: list[dict],
    hunks: list[dict],
    text: str,
) -> list[tuple[str, Query, Optional[list], Optional[dict], Optional[dict]]]:
    """Locate-query specs for a page's hunks and unlinked issues:
    (key, Query, model-bbox fallback, hunk-or-None, issue-or-None). Shared by
    _build_boxes, the /boxes route, and the background refine pipeline so all
    three derive byte-identical queries — and therefore identical VLM cache
    keys.

    Per hunk the query is its old text at its exact markdown span (computed via
    nth-occurrence of ctx_before+old+ctx_after, mirroring the JS apply logic);
    whitespace-only hunks query the whole context window (old alone may be pure
    whitespace). Issues not linked to any hunk are queried by their snippet.
    Non-whitespace hunks also carry the capped corrected text as a Query alt:
    for wrong-word findings the old text is exactly what the image does NOT
    say, so refinement also tries the fix.
    """
    linked = {h["issue_idx"] for h in hunks if h["issue_idx"] is not None}
    specs: list[tuple[str, Query, Optional[list], Optional[dict], Optional[dict]]] = []
    for h in hunks:
        needle = h["ctx_before"] + h["old"] + h["ctx_after"]
        occurrence = 1 if h["unique"] else h["occurrence"]
        pos = _nth_index(text, needle, occurrence)
        if h["ws_only"]:
            qtext = needle
            span = (pos, pos + len(needle)) if pos != -1 else None
        else:
            qtext = h["old"]
            if pos != -1:
                start = pos + len(h["ctx_before"])
                span = (start, start + len(h["old"]))
            else:
                span = None
        fallback = None
        if h["issue_idx"] is not None and h["issue_idx"] < len(issues):
            fallback = issues[h["issue_idx"]].get("bbox")
        # Cap the locating query to a few words so the box is a zoomable
        # fraction of the line (not the whole line). The hunk's full old/new
        # text is untouched — only where we draw/zoom the box changes.
        qtext_c = _cap_words(qtext)
        span_c = (span[0], span[0] + len(qtext_c)) if span is not None else None
        alts: tuple[str, ...] = ()
        if not h["ws_only"]:
            new_c = _cap_words(h["new"])
            if new_c:
                alts = (new_c,)
        specs.append((f"h{h['id']}", Query(qtext_c, span_c, alts), fallback, h, None))
    for i, iss in enumerate(issues):
        if i in linked:
            continue
        specs.append(
            (f"i{i}", Query(_cap_words(iss.get("snippet") or ""), None), iss.get("bbox"), None, iss)
        )
    return specs


def _tear_profile(key: str, pair_index: int) -> list[tuple[float, float]]:
    """Stable normalized (vertical-position, inward-offset) tear profile.

    Adjacent RTL line segments request the same pair index on their facing
    left/right edges, producing complementary mirrored tears. This is
    presentation geometry only; locator/cache data remains rectangular.
    """
    seed = 2166136261
    for byte in f"{key}:{pair_index}".encode("utf-8"):
        seed ^= byte
        seed = (seed * 16777619) & 0xFFFFFFFF
    seed = seed or 1

    def random_unit() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 4294967296

    points = [(0.0, 0.0)]
    steps = 15  # seven irregular peaks plus clean endpoints
    for i in range(1, steps):
        jitter = (random_unit() - 0.5) * 0.22
        t = (i + jitter) / steps
        peak = i % 2 == 1
        offset = (
            0.62 + random_unit() * 0.38
            if peak
            else 0.08 + random_unit() * 0.24
        )
        points.append((t, offset))
    points.append((1.0, 0.0))
    return points


def _ripped_segment_points(
    segment: dict, index: int, count: int, key: str
) -> list[tuple[float, float]]:
    """Clockwise SVG polygon points for one percent-coordinate segment."""
    left_torn = index < count - 1  # RTL line exit
    right_torn = index > 0  # RTL continuation entry
    depth = min(0.8, max(0.06, segment["w"] * 0.16), segment["w"] * 0.24)

    def edge(side: str, profile: list[tuple[float, float]]) -> list[tuple[float, float]]:
        x = segment["x0"] if side == "left" else segment["x1"]
        sign = 1.0 if side == "left" else -1.0
        return [
            (
                x + sign * depth * offset,
                segment["y0"] + segment["h"] * t,
            )
            for t, offset in profile
        ]

    left = (
        edge("left", _tear_profile(key, index))
        if left_torn
        else [(segment["x0"], segment["y0"]), (segment["x0"], segment["y1"])]
    )
    right = (
        edge("right", _tear_profile(key, index - 1))
        if right_torn
        else [(segment["x1"], segment["y0"]), (segment["x1"], segment["y1"])]
    )
    return [left[0], *right, left[-1], *reversed(left[1:-1])]


def _ripped_segment_path(segment: dict, index: int, count: int, key: str) -> str:
    points = _ripped_segment_points(segment, index, count, key)
    return "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in points) + " Z"


def _attach_boxes(
    specs: list[tuple[str, Query, Optional[list], Optional[dict], Optional[dict]]],
    located: list[Optional[dict]],
) -> list[dict]:
    """Set "box" on each spec's hunk/issue from its located box (falling back
    to the model-estimated sidecar bbox) and return the template-facing box
    list (percent values). Shared by _build_boxes and the /boxes route.
    """
    boxes: list[dict] = []
    for (key, _query, fallback, hunk, issue), loc in zip(specs, located):
        if loc:
            box = loc
        else:
            box = _bbox_to_box(fallback)
        if hunk is not None:
            hunk["box"] = box
        if issue is not None:
            issue["box"] = box
        if box is None:
            continue
        view_box = {
            "key": key,
            "x0": box["x0"] * 100.0,
            "y0": box["y0"] * 100.0,
            "x1": box["x1"] * 100.0,
            "y1": box["y1"] * 100.0,
            "w": (box["x1"] - box["x0"]) * 100.0,
            "h": (box["y1"] - box["y0"]) * 100.0,
            "source": box["source"],
        }
        segments = box.get("segments")
        if box.get("source") == "scan_vlm" and isinstance(segments, list):
            view_box["segments"] = [
                {
                    "x0": segment["x0"] * 100.0,
                    "y0": segment["y0"] * 100.0,
                    "x1": segment["x1"] * 100.0,
                    "y1": segment["y1"] * 100.0,
                    "w": (segment["x1"] - segment["x0"]) * 100.0,
                    "h": (segment["y1"] - segment["y0"]) * 100.0,
                }
                for segment in segments
                if isinstance(segment, dict)
                and all(k in segment for k in ("x0", "y0", "x1", "y1"))
            ]
            if len(view_box["segments"]) > 1:
                view_box["paths"] = [
                    _ripped_segment_path(
                        segment, index, len(view_box["segments"]), key
                    )
                    for index, segment in enumerate(view_box["segments"])
                ]
        boxes.append(view_box)
    return boxes


def _build_boxes(
    ws: Workspace,
    n: int,
    issues: list[dict],
    hunks: list[dict],
    text: str,
) -> list[dict]:
    """Attach a "box" (0-1 fractions + source, or None) to every hunk and
    every issue, and return the template-facing box list (percent values).

    A tiered locate.py hit ("match"/"layout"/"scan") wins; otherwise the
    (linked) issue's model-estimated sidecar bbox ("model"); otherwise no box.
    When a refiner is configured, "scan" boxes (proportional pixel estimates,
    often a few words off) are upgraded to their disk-cached VLM-verified
    "scan_vlm" boxes — cache lookups only, never the API, so rendering stays
    instant: the background pipeline warms the cache and the browser polls
    /boxes/<n> to swap refined boxes in. Any failure degrades to the
    un-refined boxes, same philosophy as _locate_queries_cached.
    """
    for iss in issues:
        iss.setdefault("box", None)
    if not issues and not hunks:
        return []

    specs = _box_specs(issues, hunks, text)
    located = list(
        _locate_queries_cached(str(ws.pdf_path), n, text, tuple(s[1] for s in specs))
    )

    refiner = _REFINER
    if refiner is not None:
        try:
            located = refiner.apply_cached(n, text, [s[1] for s in specs], located)
        except Exception:
            pass

    return _attach_boxes(specs, located)


def _boxes_payload(ws: Workspace, n: int) -> dict:
    """JSON body for GET /boxes/<n>: {"pending": true} while the background
    pipeline still owes the page uncached scan refinements, else the full
    refreshed geometry — "boxes" in the template's percent form plus
    "hunk_boxes"/"issue_boxes" maps in the payload's 0-1-fraction form so the
    client can patch click-to-zoom data in place. Runs only the cheap tiers
    and cache lookups; never the API. With no refiner configured the answer
    is always non-pending.
    """
    _sidecar, issues, hunks, text, _panel_kind = _page_box_inputs(ws, n)
    for iss in issues:
        iss.setdefault("box", None)
    if not issues and not hunks:
        return {"pending": False, "boxes": [], "hunk_boxes": {}, "issue_boxes": {}}

    specs = _box_specs(issues, hunks, text)
    queries = [s[1] for s in specs]
    located = list(_locate_queries_cached(str(ws.pdf_path), n, text, tuple(queries)))

    refiner = _REFINER
    if refiner is not None:
        try:
            if refiner.pending(n, text, queries, located):
                return {"pending": True}
            located = refiner.apply_cached(n, text, queries, located)
        except Exception:
            pass

    boxes = _attach_boxes(specs, located)
    return {
        "pending": False,
        "boxes": boxes,
        "hunk_boxes": {str(h["id"]): h.get("box") for h in hunks},
        "issue_boxes": {str(i): iss.get("box") for i, iss in enumerate(issues)},
    }


# ---------------------------------------------------------------------------
# hunk derivation (pure helpers, unit-tested)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+\s*|\s+")


def _tokenize_ws(text: str) -> list[tuple[int, int]]:
    """Character spans of whitespace-preserving tokens. Each token is a run
    of non-space characters plus its trailing whitespace (or a leading pure
    whitespace run). The spans tile `text` exactly, so any token range maps
    back to an exact substring.
    """
    return [m.span() for m in _TOKEN_RE.finditer(text)]


def _find_all(hay: str, needle: str) -> list[int]:
    """Start offsets of every (possibly overlapping) occurrence of `needle`."""
    if not needle:
        return [0]
    positions: list[int] = []
    start = 0
    while True:
        i = hay.find(needle, start)
        if i == -1:
            return positions
        positions.append(i)
        start = i + 1


def _derive_hunks(
    old_text: str,
    new_text: str,
    ctx_tokens: int = 3,
    max_ctx_tokens: int = 12,
    display_ctx_tokens: int = 15,
) -> list[dict]:
    """Token-level change hunks between `old_text` and `new_text`.

    Diffs whitespace-preserving tokens with difflib, merges non-equal opcode
    spans separated by <= 2*ctx_tokens+1 equal tokens, and attaches enough
    surrounding context to each hunk to locate it uniquely in `old_text`.
    Context growth is clamped so it never reaches into a neighboring hunk's
    changed tokens — distinct hunks therefore apply independently in any
    order. `old`/`new`/`ctx_*` are exact substrings, so splicing `new` in
    place of `old` between the contexts reproduces `new_text` once every
    hunk is applied. Returns [] when the texts are equal.

    Each hunk also carries `show_before`/`show_after`: a wider (~15 token),
    display-only readability window, independently clamped to the same
    neighboring-hunk boundaries as the anchor context so it never overlaps
    another hunk's changed region. These are presentation-only — never used
    for locating/anchoring — and get an "… " / " …" marker when the
    surrounding page text extends further than what's shown.
    """
    if old_text == new_text:
        return []

    old_spans = _tokenize_ws(old_text)
    new_spans = _tokenize_ws(new_text)
    old_tokens = [old_text[a:b] for a, b in old_spans]
    new_tokens = [new_text[a:b] for a, b in new_spans]

    sm = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    groups: list[tuple[int, int, int, int]] = []
    cur: Optional[tuple[int, int, int, int]] = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if cur is not None and (i1 - cur[1]) <= 2 * ctx_tokens + 1:
            cur = (cur[0], i2, cur[2], j2)
        else:
            if cur is not None:
                groups.append(cur)
            cur = (i1, i2, j1, j2)
    if cur is not None:
        groups.append(cur)

    hunks: list[dict] = []
    for idx, (o1, o2, n1, n2) in enumerate(groups):
        o_start = old_spans[o1][0] if o1 < len(old_spans) else len(old_text)
        o_end = old_spans[o2 - 1][1] if o2 > o1 else o_start
        n_start = new_spans[n1][0] if n1 < len(new_spans) else len(new_text)
        n_end = new_spans[n2 - 1][1] if n2 > n1 else n_start
        old_frag = old_text[o_start:o_end]
        new_frag = new_text[n_start:n_end]

        # Per-side context caps: never reach into a neighboring hunk's
        # changed tokens (that would break order-independent application).
        def _side_caps(cap: int) -> tuple[int, int]:
            b, a = cap, cap
            if idx > 0:
                b = min(b, o1 - groups[idx - 1][1])
            if idx + 1 < len(groups):
                a = min(a, groups[idx + 1][0] - o2)
            return b, a

        before_cap, after_cap = _side_caps(max_ctx_tokens)

        def _ctx(k: int) -> tuple[str, str]:
            kb = min(k, before_cap)
            ka = min(k, after_cap)
            b_idx = max(0, o1 - kb)
            cb_start = old_spans[b_idx][0] if b_idx < len(old_spans) and o1 > 0 else o_start
            a_idx = min(len(old_spans), o2 + ka)
            ca_end = old_spans[a_idx - 1][1] if a_idx > o2 else o_end
            return old_text[cb_start:o_start], old_text[o_end:ca_end]

        k = ctx_tokens
        while True:
            ctx_before, ctx_after = _ctx(k)
            needle = ctx_before + old_frag + ctx_after
            positions = _find_all(old_text, needle)
            if len(positions) == 1 or k >= max_ctx_tokens:
                break
            k = min(max_ctx_tokens, k + 3)

        unique = len(positions) == 1
        if unique:
            occurrence = 1
        else:
            own_start = o_start - len(ctx_before)
            occurrence = positions.index(own_start) + 1 if own_start in positions else 1

        # Display-only readability context: wider than the anchor context,
        # still clamped to the neighboring hunks' boundaries. Never used for
        # locating — only for showing the reviewer more of the sentence.
        disp_before_cap, disp_after_cap = _side_caps(display_ctx_tokens)
        db_idx = max(0, o1 - disp_before_cap)
        show_cb_start = old_spans[db_idx][0] if db_idx < len(old_spans) and o1 > 0 else o_start
        da_idx = min(len(old_spans), o2 + disp_after_cap)
        show_ca_end = old_spans[da_idx - 1][1] if da_idx > o2 else o_end
        show_before = old_text[show_cb_start:o_start]
        show_after = old_text[o_end:show_ca_end]
        if db_idx > 0:
            show_before = "… " + show_before
        if da_idx < len(old_spans):
            show_after = show_after + " …"

        hunks.append(
            {
                "id": idx,
                "old": old_frag,
                "new": new_frag,
                "ctx_before": ctx_before,
                "ctx_after": ctx_after,
                "unique": unique,
                "occurrence": occurrence,
                "ws_only": old_frag.split() == new_frag.split(),
                "issue_idx": None,
                "show_before": show_before,
                "show_after": show_after,
            }
        )
    return hunks


def _link_issues_to_hunks(issues: list[dict], hunks: list[dict]) -> None:
    """Attach each hunk to the QC issue whose snippet best overlaps it.

    For each hunk, picks the issue maximizing
    |words(snippet) ∩ words(old + new + ctx)| / |words(snippet)|
    and sets hunk["issue_idx"] when the score is >= 0.4. Several hunks may
    share one issue; unmatched hunks keep issue_idx = None.
    """
    for h in hunks:
        hay = set(
            (h["old"] + " " + h["new"] + " " + h["ctx_before"] + " " + h["ctx_after"]).split()
        )
        best_idx: Optional[int] = None
        best_score = 0.0
        for i, issue in enumerate(issues):
            snippet_words = set((issue.get("snippet") or "").split())
            if not snippet_words:
                continue
            score = len(snippet_words & hay) / len(snippet_words)
            if score > best_score:
                best_score = score
                best_idx = i
        h["issue_idx"] = best_idx if best_score >= 0.4 else None


# QC issues carry a flagged snippet but no fix (verdict-pass issues, an
# "identical" or absent suggestion). We synthesize edit-only hunks for them so
# a reviewer can still correct the phrase in place. Ids sit far above real hunk
# ids (0..N) so the two id spaces never collide; the offset is reproduced
# server-side in _decide_accept_outcome so client/server hunk ids agree.
_FINDING_ID_BASE = 100000


def _derive_finding_hunks(
    text: str, issues: list[dict], real_hunks: list[dict]
) -> list[dict]:
    """Edit-only hunk objects for QC issues that have a snippet but no linked
    fix hunk. Shaped like a real hunk (so the whole hunk edit/apply/accept path
    works unchanged) with old == new == snippet, empty context, and
    edit_only=True. A snippet that isn't an exact substring is still emitted;
    the client's replaceHunk then reports "apply manually".
    """
    linked = {h["issue_idx"] for h in real_hunks if h.get("issue_idx") is not None}
    out: list[dict] = []
    for i, iss in enumerate(issues):
        if i in linked:
            continue
        snip = (iss.get("snippet") or "").strip()
        if not snip:
            continue
        positions = _find_all(text, snip)
        out.append(
            {
                "id": _FINDING_ID_BASE + i,
                "old": snip,
                "new": snip,
                "ctx_before": "",
                "ctx_after": "",
                "unique": len(positions) == 1,
                "occurrence": 1,
                "ws_only": False,
                "issue_idx": i,
                "show_before": "",
                "show_after": "",
                "edit_only": True,
            }
        )
    return out


# ---------------------------------------------------------------------------
# accept outcome (pure, unit-tested)
# ---------------------------------------------------------------------------


@dataclass
class _AcceptOutcome:
    reviewed: str                       # "edited" | "accepted"
    suggestion_status: Optional[str]    # new qc.suggestion_status or None = leave as-is
    events: list[tuple[str, str, str, str]]  # (detected_by, issue_type, old_frag, new_frag)


def _finding_edit_events(
    finding_hunks: list[dict],
    hunk_decisions: Optional[list[dict]],
    issues: list[dict],
    default_issue_type: str,
) -> tuple[list[tuple[str, str, str, str]], int]:
    """Human-edit history events for edit-only finding hunks the reviewer
    changed. Returns (events, n_edited). Ids are matched against the same
    _derive_finding_hunks output produced client-side, so decisions line up.
    """
    by_id: dict[int, dict] = {}
    for d in hunk_decisions or []:
        try:
            by_id[int(d.get("id"))] = d
        except (TypeError, ValueError):
            continue
    events: list[tuple[str, str, str, str]] = []
    n_edited = 0
    for h in finding_hunks:
        d = by_id.get(h["id"]) or {}
        if d.get("decision") != "edited":
            continue
        if h["issue_idx"] is not None and h["issue_idx"] < len(issues):
            issue_type = issues[h["issue_idx"]].get("type") or default_issue_type
        else:
            issue_type = default_issue_type
        events.append(("human_edit", issue_type, h["old"], d.get("text", h["old"])))
        n_edited += 1
    return events, n_edited


def _decide_accept_outcome(
    sidecar: dict,
    on_disk_text: str,
    new_text: str,
    hunk_decisions: Optional[list[dict]],
) -> _AcceptOutcome:
    """Decide, for one /accept POST, the sidecar `reviewed` value, the new QC
    suggestion_status (None = leave untouched), and the QC history events to
    record. Pure function (no I/O) so it can be unit-tested directly.
    """
    text_changed = new_text != on_disk_text
    reviewed = "edited" if text_changed else "accepted"
    default_issue_type = _first_qc_issue_type(sidecar)
    pending = _pending_suggestion(sidecar)

    if pending is None:
        qc_data = sidecar.get("qc") or {}
        dangling = (
            qc_data.get("suggestion_status") == "pending"
            and qc_data.get("suggested_text_md") is None
        )
        if not dangling:
            events: list[tuple[str, str, str, str]] = []
            if text_changed:
                events.append(("human_edit", default_issue_type, on_disk_text, new_text))
            return _AcceptOutcome(reviewed, None, events)
        # Verdict-fail page with flagged phrases but no suggestion: each flagged
        # phrase is an edit-only finding hunk. Record per-finding human edits.
        issues = qc_data.get("issues") or []
        finding_hunks = _derive_finding_hunks(on_disk_text, issues, [])
        events, n_finding_edited = _finding_edit_events(
            finding_hunks, hunk_decisions, issues, default_issue_type
        )
        if text_changed and n_finding_edited == 0:
            events.append(("human_edit", default_issue_type, on_disk_text, new_text))
        status = "edited" if (n_finding_edited or text_changed) else "rejected"
        return _AcceptOutcome(reviewed, status, events)

    suggested = pending.get("suggested_text_md") or ""
    issues = pending.get("issues") or []

    if new_text == suggested:
        return _AcceptOutcome(
            reviewed,
            "accepted",
            [("suggestion_accepted", default_issue_type, on_disk_text, new_text)],
        )

    if hunk_decisions:
        # Re-derive hunks server-side (same helper -> same ids); never trust
        # client-sent hunk text except the explicit "edited" replacement.
        hunks = _derive_hunks(on_disk_text, suggested)
        _link_issues_to_hunks(issues, hunks)
        finding_hunks = _derive_finding_hunks(on_disk_text, issues, hunks)
        by_id: dict[int, dict] = {}
        for d in hunk_decisions:
            try:
                by_id[int(d.get("id"))] = d
            except (TypeError, ValueError):
                continue
        events = []
        n_approved = 0
        n_edited = 0
        for h in hunks:
            d = by_id.get(h["id"]) or {}
            decision = d.get("decision", "pending")
            if h["issue_idx"] is not None and h["issue_idx"] < len(issues):
                issue_type = issues[h["issue_idx"]].get("type") or default_issue_type
            else:
                issue_type = default_issue_type
            if decision == "approved":
                detected_by, decided = "suggestion_accepted", h["new"]
                n_approved += 1
            elif decision == "edited":
                detected_by, decided = "suggestion_edited", d.get("text", h["new"])
                n_edited += 1
            else:  # rejected or pending -> the original text stayed
                detected_by, decided = "suggestion_rejected", h["old"]
            events.append((detected_by, issue_type, h["old"], decided))
        # Flagged phrases with no suggestion (unlinked issues) ride alongside.
        finding_events, n_finding_edited = _finding_edit_events(
            finding_hunks, hunk_decisions, issues, default_issue_type
        )
        events.extend(finding_events)
        if hunks and n_approved == len(hunks) and n_finding_edited == 0:
            status = "accepted"
        elif n_approved == 0 and n_edited == 0 and n_finding_edited == 0:
            status = "rejected"
        else:
            status = "edited"
        return _AcceptOutcome(reviewed, status, events)

    # Pending suggestion but no hunk decisions were sent.
    if not text_changed:
        return _AcceptOutcome(
            reviewed,
            "rejected",
            [("suggestion_rejected", default_issue_type, on_disk_text, suggested)],
        )
    return _AcceptOutcome(
        reviewed,
        "edited",
        [("suggestion_edited", default_issue_type, on_disk_text, new_text)],
    )


# ---------------------------------------------------------------------------
# review reset
# ---------------------------------------------------------------------------


def reset_reviews(ws: Workspace) -> dict:
    """Undo human review state for every done page: clear reviewed /
    review_skipped, re-flag needs_review, return resolved QC suggestions to
    "pending", and drop this book's human events from the QC history. Text
    edits and .orig.md backups are kept.
    """
    pages_reset = 0
    for n in ws.pages_done():
        sidecar = _read_sidecar(ws, n)
        if not (sidecar.get("reviewed") or sidecar.get("review_skipped")):
            continue
        sidecar["needs_review"] = True
        sidecar.pop("reviewed", None)
        sidecar.pop("review_skipped", None)
        qc_data = sidecar.get("qc")
        if qc_data and qc_data.get("suggestion_status") in ("accepted", "edited", "rejected"):
            qc_data["suggestion_status"] = "pending"
        _write_sidecar(ws, n, sidecar)
        pages_reset += 1
    events_removed = qc.remove_human_events(ws.slug)
    return {"pages_reset": pages_reset, "events_removed": events_removed}


# ---------------------------------------------------------------------------
# page selection (review budget)
# ---------------------------------------------------------------------------


def _select_pages_for_review(ws: Workspace, budget_all: bool = False) -> tuple[list[int], list[int]]:
    """Return (surfaced, skipped) page numbers.

    `surfaced` are the pages to actually show in the review UI: those with
    needs_review == true in their sidecar, plus pages whose deterministic
    validators recorded issues and that no human has reviewed yet (so
    validator findings surface even when they stayed below the needs_review
    threshold). The list is capped at ceil(total_transcribed_pages / 5),
    keeping the lowest quality_score first.
    `skipped` are flagged pages that were cut off by the budget; their
    sidecar gets a "review_skipped": true note but needs_review stays true.

    When `budget_all` is True the cap is disabled: every flagged page is
    surfaced and none are skipped.
    """
    done = ws.pages_done()
    total = len(done)
    budget = math.ceil(total / 5) if total else 0
    if budget_all:
        budget = total

    flagged: list[tuple[float, int]] = []
    for n in done:
        sidecar = _read_sidecar(ws, n)
        issues = (sidecar.get("validators") or {}).get("issues") or []
        if sidecar.get("needs_review") or (issues and not sidecar.get("reviewed")):
            flagged.append((sidecar.get("quality_score", 0.0), n))

    # Lowest quality_score first.
    flagged.sort(key=lambda t: (t[0], t[1]))

    surfaced = [n for _, n in flagged[:budget]]
    skipped = [n for _, n in flagged[budget:]]

    for n in skipped:
        sidecar = _read_sidecar(ws, n)
        if not sidecar.get("review_skipped"):
            sidecar["review_skipped"] = True
            _write_sidecar(ws, n, sidecar)

    return sorted(surfaced), sorted(skipped)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<title>farsi2epub review &mdash; {{ slug }}</title>
<style>
@font-face {
  font-family: "Vazirmatn";
  src: url("/font/vazirmatn.ttf") format("truetype");
  font-weight: normal;
  font-style: normal;
}
* { box-sizing: border-box; }
body {
  font-family: "Vazirmatn", Tahoma, sans-serif;
  background: #1b1c20;
  color: #e8e8ea;
  margin: 0;
  padding: 0;
}
header {
  position: sticky;
  top: 0;
  background: #26272c;
  border-bottom: 1px solid #3a3b42;
  padding: 0.9rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  z-index: 10;
  direction: ltr;
}
header .title { font-size: 1.1rem; font-weight: 600; }
header .progress { font-size: 0.95rem; color: #b7b9c2; }
header a.done {
  background: #3a6ff0;
  color: white;
  padding: 0.45rem 1rem;
  border-radius: 6px;
  text-decoration: none;
  font-size: 0.9rem;
  cursor: pointer;
  border: none;
}
main { padding: 1.5rem; max-width: 1400px; margin: 0 auto; }
.skipped-note {
  background: #3a2f1c;
  border: 1px solid #6b5522;
  color: #f0d9a0;
  border-radius: 8px;
  padding: 0.8rem 1.2rem;
  margin-bottom: 1.5rem;
  direction: rtl;
  font-size: 0.95rem;
}
.page-block {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 2rem;
  padding: 1.2rem;
  border: 1px solid #3a3b42;
  border-radius: 10px;
  background: #212227;
  transition: opacity 0.2s;
}
.page-block.done { opacity: 0.55; }
.page-block .col-img { flex: 0 0 48%; max-width: 48%; }
.page-block .col-img img { max-width: 100%; border-radius: 6px; border: 1px solid #3a3b42; display: block; }
.img-viewport {
  position: sticky;
  top: 4.5rem;
  overflow: hidden;
  border-radius: 6px;
  cursor: zoom-in;
}
.img-wrap {
  position: relative;
  display: inline-block;
  max-width: 100%;
  transform-origin: 0 0;
  transition: transform 0.55s cubic-bezier(0.22, 1, 0.36, 1);
  will-change: transform;
}
.img-wrap.panning { transition: none; }
.img-viewport.zoomed { cursor: grab; }
.img-viewport.panning { cursor: grabbing; }
.qc-box { position: absolute; border-radius: 2px; }
.qc-box-single.qc-box-match { border: 2px solid #3a9ff0; background: rgba(58,159,240,.12); }
.qc-box-single.qc-box-layout { border: 2px solid #2fb6a8; background: rgba(47,182,168,.12); }
.qc-box-single.qc-box-scan { border: 2px solid #7ac65c; background: rgba(122,198,92,.12); }
.qc-box-single.qc-box-scan_vlm { border: 2px solid #2e8b57; background: rgba(46,139,87,.12); }
.qc-box-single.qc-box-model { border: 2px dashed #f0a03a; background: rgba(240,160,58,.10); }
.qc-box-single.hot { outline: 2px solid #ffffff; }
.qc-box-single.zoom-target { animation: qc-glow 0.6s ease-in-out 2; z-index: 5; }
.qc-box-multipart {
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
  pointer-events: none;
  z-index: 2;
}
.qc-box-multipart .qc-box-shape {
  fill: rgba(46,139,87,.12);
  stroke: #2e8b57;
  stroke-width: 2;
  stroke-linejoin: bevel;
  vector-effect: non-scaling-stroke;
  pointer-events: all;
}
.qc-box-multipart.hot .qc-box-shape {
  filter: drop-shadow(0 0 2px #ffffff);
}
.qc-box-multipart.zoom-target { z-index: 5; }
.qc-box-multipart.zoom-target .qc-box-shape {
  animation: qc-svg-glow 0.6s ease-in-out 2;
}
@keyframes qc-glow {
  0%   { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
  50%  { box-shadow: 0 0 0 5px rgba(255,255,255,.85); }
  100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}
@keyframes qc-svg-glow {
  0%   { filter: drop-shadow(0 0 0 rgba(255,255,255,0)); }
  50%  { filter: drop-shadow(0 0 5px rgba(255,255,255,.95)); }
  100% { filter: drop-shadow(0 0 0 rgba(255,255,255,0)); }
}
.hunk-item.hot { border-color: #3a9ff0; }
.hunk-item.active {
  box-shadow: inset 0 0 0 2px #f0c24a, 0 0 8px rgba(240,194,74,.45);
  background: #2a2b31;
}
.qc-issue { cursor: zoom-in; }
.qc-legend {
  font-size: 0.8rem;
  color: #b7b9c2;
  margin-top: 0.5rem;
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
}
.legend-swatch {
  display: inline-block;
  width: 16px;
  height: 10px;
  border-radius: 2px;
  margin-inline-end: 0.35rem;
  vertical-align: middle;
}
.legend-swatch.legend-match { border: 2px solid #3a9ff0; background: rgba(58,159,240,.12); }
.legend-swatch.legend-layout { border: 2px solid #2fb6a8; background: rgba(47,182,168,.12); }
.legend-swatch.legend-scan { border: 2px solid #7ac65c; background: rgba(122,198,92,.12); }
.legend-swatch.legend-scan_vlm { border: 2px solid #2e8b57; background: rgba(46,139,87,.12); }
.legend-swatch.legend-model { border: 2px dashed #f0a03a; background: rgba(240,160,58,.10); }
.page-block .col-text { flex: 0 0 48%; max-width: 48%; display: flex; flex-direction: column; }
.meta-row {
  direction: rtl;
  font-size: 0.85rem;
  color: #b7b9c2;
  margin-bottom: 0.6rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1rem;
}
.meta-row span.pill {
  background: #2f3037;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  border: 1px solid #3a3b42;
}
.flags { color: #f0a0a0; }
.meta-row span.pill.issues { border-color: #7a5a20; color: #f0c78a; }
.meta-row span.pill.qc-pill { border-color: #6a2f6f; color: #e2a8ea; }
.qc-panel {
  direction: rtl;
  border: 1px solid #6a2f6f;
  background: #241a26;
  border-radius: 8px;
  padding: 0.9rem 1.1rem;
  margin-bottom: 0.8rem;
}
.qc-panel-title { font-weight: 600; margin-bottom: 0.5rem; color: #e2a8ea; }
.qc-group { margin-bottom: 0.9rem; }
.qc-group:last-child { margin-bottom: 0; }
.qc-issue-list { margin: 0 0 0.5rem; padding-inline-start: 1.2rem; }
.qc-issue-list li { margin-bottom: 0.5rem; }
.qc-issue-head {
  font-size: 0.9rem;
  direction: rtl;
  unicode-bidi: isolate;
}
.qc-issue-type {
  direction: ltr;
  unicode-bidi: isolate;
  display: inline-block;
  font-family: "Courier New", monospace;
}
.qc-issue-description {
  direction: rtl;
  unicode-bidi: isolate;
}
.qc-snippet {
  font-family: "Courier New", monospace;
  font-size: 0.8rem;
  color: #b7b9c2;
  background: #17181b;
  border-radius: 4px;
  padding: 0.3rem 0.5rem;
  margin-top: 0.2rem;
  direction: rtl;
  unicode-bidi: plaintext;
}
.qc-note {
  font-size: 0.9rem;
  color: #f0c78a;
  background: #17181b;
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
}
.hunk-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.hunk-item {
  border: 1px solid #3a3b42;
  border-left: 4px solid #55565e;
  border-radius: 6px;
  background: #17181b;
  padding: 0.6rem 0.8rem;
}
.hunk-item.approved { border-left-color: #2fa35a; }
.hunk-item.rejected { border-left-color: #d9534f; }
.hunk-item.edited { border-left-color: #3a6ff0; }
.hunk-issue { font-size: 0.85rem; color: #e2a8ea; margin-bottom: 0.35rem; }
.hunk-diff {
  direction: rtl;
  unicode-bidi: plaintext;
  white-space: pre-wrap;
  font-size: 0.95rem;
  line-height: 1.9;
  margin-bottom: 0.4rem;
}
.hunk-diff del { background: #4a1f22; color: #f5b5b8; text-decoration: line-through; }
.hunk-diff ins { background: #1f3a24; color: #b6e6bf; text-decoration: none; }
.hunk-diff.hunk-noswitch { color: #9a9ba3; font-style: italic; font-size: 0.85rem; }
.hunk-ctx { color: #83848c; font-size: 0.85em; }
.hunk-actions { direction: ltr; display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap; }
.hunk-actions button { font-size: 0.8rem; padding: 0.3rem 0.7rem; }
.hunk-edit { margin-top: 0.5rem; }
.hunk-edit textarea {
  flex: none;
  min-height: 4em;
  width: 100%;
  font-size: 0.95rem;
  margin-bottom: 0.4rem;
}
textarea {
  flex: 1;
  min-height: 420px;
  font-family: "Vazirmatn", Tahoma, sans-serif;
  font-size: 1.05rem;
  line-height: 1.8;
  direction: rtl;
  padding: 0.8rem;
  border-radius: 6px;
  border: 1px solid #3a3b42;
  background: #17181b;
  color: #e8e8ea;
  resize: vertical;
}
.actions { margin-top: 0.7rem; display: flex; gap: 0.6rem; direction: ltr; }
button {
  font-family: inherit;
  font-size: 0.9rem;
  padding: 0.5rem 1.1rem;
  border-radius: 6px;
  border: 1px solid #3a3b42;
  background: #2f3037;
  color: #e8e8ea;
  cursor: pointer;
}
button.accept { background: #2fa35a; color: white; border-color: #2fa35a; }
button:disabled { opacity: 0.5; cursor: default; }
.status-note { font-size: 0.85rem; color: #8fce9f; direction: ltr; align-self: center; }
.status-note.err { color: #f5b5b8; }
.meta-row span.pill.refine-chip { color: #9a9ba3; border-color: #3a3b42; font-size: 0.75rem; }
</style>
</head>
<body>
<header>
  <div class="title">Review &mdash; {{ slug }}</div>
  <div class="progress" id="progress">{{ reviewed_count }} of {{ total_count }} reviewed</div>
  <a class="done" href="#" id="done-link">Done</a>
</header>
<main>
  {% if skipped %}
  <div class="skipped-note">
    {{ skipped|length }} additional flagged page(s) were auto-accepted despite flags due to the review budget: {{ skipped|join(', ') }}
  </div>
  {% endif %}

  {% for p in pages %}
  <div class="page-block{% if p.reviewed %} done{% endif %}" id="block-{{ p.page }}" data-page="{{ p.page }}">
    <div class="col-img">
      {% if p.image_url %}
      <div class="img-viewport" id="viewport-{{ p.page }}" data-page="{{ p.page }}">
        <div class="img-wrap" id="imgwrap-{{ p.page }}">
          <img src="{{ p.image_url }}" alt="page {{ p.page }}">
        </div>
      </div>
      {% else %}
      <div>(no image available for page {{ p.page }})</div>
      {% endif %}
    </div>
    <div class="col-text">
      <div class="meta-row">
        <span class="pill">page {{ p.page }}</span>
        <span class="pill">model: {{ p.model_used }}</span>
        <span class="pill">confidence: {{ "%.2f"|format(p.confidence) }}</span>
        <span class="pill">quality: {{ "%.2f"|format(p.quality_score) }}</span>
        {% if p.flags %}
        <span class="pill flags">flags: {{ p.flags|join(', ') }}</span>
        {% endif %}
        {% if p.validator_issues %}
        <span class="pill issues">validators: {{ p.validator_issues|join(', ') }}</span>
        {% endif %}
        {% if p.qc_issue_types %}
        <span class="pill qc-pill">QC: {{ p.qc_issue_types|join(', ') }}</span>
        {% endif %}
      </div>
      {% if p.qc_panel %}
      <div class="qc-panel" id="qc-panel-{{ p.page }}">
        <div class="qc-panel-title">QC findings</div>
        {% for g in p.qc_panel.groups %}
        <div class="qc-group">
          <ul class="qc-issue-list">
            <li>
              {% if g.issue %}
              <div class="qc-issue-head{% if g.idx is not none %} qc-issue{% endif %}"{% if g.idx is not none %} data-page="{{ p.page }}" data-issue="{{ g.idx }}"{% endif %}><bdi class="qc-issue-type" dir="ltr">{{ g.issue.type }}</bdi><span aria-hidden="true"> &mdash; </span><span class="qc-issue-description" dir="rtl" lang="fa">{{ g.issue.description }}</span></div>
              {% if g.issue.snippet %}
              <div class="qc-snippet" dir="rtl">{{ g.issue.snippet }}</div>
              {% endif %}
              {% else %}
              <div class="qc-issue-head">{{ g.other_label }}</div>
              {% endif %}
            </li>
          </ul>
          {% if g.hunks %}
          <ul class="hunk-list">
            {% for h in g.hunks %}
            <li class="hunk-item{% if h.edit_only %} edit-only{% endif %}" id="hunk-{{ p.page }}-{{ h.id }}" data-page="{{ p.page }}" data-hunk="{{ h.id }}">
              {% if h.edit_only %}
              <div class="hunk-diff hunk-noswitch" dir="rtl">no suggested fix &mdash; edit the flagged text</div>
              {% elif h.ws_only %}
              <div class="hunk-diff">&para; line/paragraph-break change</div>
              {% else %}
              <div class="hunk-diff" dir="rtl"><span class="hunk-ctx">{{ h.show_before }}</span>{% if h.old %}<del><bdi>{{ h.old }}</bdi></del>{% endif %} {% if h.new %}<ins><bdi>{{ h.new }}</bdi></ins>{% endif %}<span class="hunk-ctx">{{ h.show_after }}</span></div>
              {% endif %}
              <div class="hunk-actions">
                <button onclick="approveHunk({{ p.page }}, {{ h.id }})"{% if h.edit_only %} disabled title="no suggested fix to approve"{% endif %}>Approve</button>
                <button onclick="toggleEditHunk({{ p.page }}, {{ h.id }})">Edit</button>
                <button onclick="rejectHunk({{ p.page }}, {{ h.id }})"{% if h.edit_only %} disabled title="no suggested fix to reject"{% endif %}>Reject</button>
                <button id="undo-{{ p.page }}-{{ h.id }}" style="display:none" onclick="undoHunk({{ p.page }}, {{ h.id }})">Undo</button>
                <span class="status-note" id="hunk-status-{{ p.page }}-{{ h.id }}"></span>
              </div>
              <div class="hunk-edit" id="hunk-edit-{{ p.page }}-{{ h.id }}" style="display:none">
                <textarea dir="rtl" lang="fa" id="hunk-edit-text-{{ p.page }}-{{ h.id }}">{% if h.edit_only %}{{ h.old }}{% else %}{{ h.new }}{% endif %}</textarea>
                <button onclick="applyEditedHunk({{ p.page }}, {{ h.id }})">Apply my text</button>
              </div>
            </li>
            {% endfor %}
          </ul>
          {% endif %}
        </div>
        {% endfor %}
        {% if p.qc_panel.kind == 'identical' %}
        <div class="qc-note">QC flagged these but its suggested text matches the current text &mdash; edit any inline if a fix is needed.</div>
        {% elif p.qc_panel.kind == 'no_suggestion' %}
        <div class="qc-note">QC flagged these but produced no suggested correction &mdash; edit any inline if a fix is needed.</div>
        {% endif %}
        {% if p.boxes %}
        <div class="qc-legend">
          <span><span class="legend-swatch legend-match"></span>located (word match)</span>
          <span><span class="legend-swatch legend-layout"></span>located (layout)</span>
          <span><span class="legend-swatch legend-scan"></span>located (image layout)</span>
          <span><span class="legend-swatch legend-scan_vlm"></span>located (image + VLM)</span>
          <span><span class="legend-swatch legend-model"></span>model estimate</span>
        </div>
        {% endif %}
      </div>
      {% endif %}
      <script type="application/json" id="payload-{{ p.page }}">{{ p.payload_json }}</script>
      <textarea dir="rtl" lang="fa" id="text-{{ p.page }}">{{ p.text }}</textarea>
      <div class="actions">
        <button class="accept" onclick="acceptPage({{ p.page }})">Accept</button>
        <span class="status-note" id="status-{{ p.page }}"></span>
      </div>
    </div>
  </div>
  {% endfor %}
</main>
<script>
var BBOX_REFINE = {{ 'true' if bbox_refine else 'false' }};
var payloadCache = {};
var hunkState = {};

function showStatus(page, msg, isError) {
  var el = document.getElementById('status-' + page);
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle('err', !!isError);
}

function showHunkStatus(page, id, msg, isError) {
  var el = document.getElementById('hunk-status-' + page + '-' + id);
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle('err', !!isError);
}

function pagePayload(page) {
  if (payloadCache[page]) return payloadCache[page];
  var el = document.getElementById('payload-' + page);
  if (!el) {
    showStatus(page, 'internal error: payload element missing for page ' + page, true);
    return null;
  }
  try {
    var data = JSON.parse(el.textContent);
    payloadCache[page] = data;
    return data;
  } catch (e) {
    showStatus(page, 'internal error: could not parse page payload: ' + e, true);
    return null;
  }
}

// -- finding-box rendering -------------------------------------------------
// Refined wrapped scan findings carry ordered line-local rectangles. Their
// stable torn SVG paths are derived server-side from the finding key and
// rectangle geometry, never persisted in the locator cache.

function setFocusGeometry(el, segment) {
  el.setAttribute('data-focus-x0', segment.x0);
  el.setAttribute('data-focus-y0', segment.y0);
  el.setAttribute('data-focus-w', segment.w);
  el.setAttribute('data-focus-h', segment.h);
}

function createBoxElement(page, b) {
  var id = 'box-' + page + '-' + b.key;
  var segments = b.source === 'scan_vlm' && Array.isArray(b.segments)
    ? b.segments : [];
  if (segments.length > 1) {
    var ns = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('class', 'qc-box qc-box-multipart qc-box-' + b.source);
    svg.setAttribute('id', id);
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'multi-line located finding');
    setFocusGeometry(svg, segments[0]);
    for (var i = 0; i < segments.length; i++) {
      var path = document.createElementNS(ns, 'path');
      path.setAttribute('class', 'qc-box-shape');
      path.setAttribute('d', b.paths[i]);
      path.setAttribute('data-segment', i);
      svg.appendChild(path);
    }
    return svg;
  }
  var div = document.createElement('div');
  div.className = 'qc-box qc-box-single qc-box-' + b.source;
  div.id = id;
  div.style.left = b.x0.toFixed(2) + '%';
  div.style.top = b.y0.toFixed(2) + '%';
  div.style.width = b.w.toFixed(2) + '%';
  div.style.height = b.h.toFixed(2) + '%';
  setFocusGeometry(div, segments.length === 1 ? segments[0] : b);
  return div;
}

function renderBoxes(page, boxes) {
  var wrap = document.getElementById('imgwrap-' + page);
  if (!wrap) return;
  var old = wrap.querySelectorAll('.qc-box');
  var i;
  for (i = 0; i < old.length; i++) old[i].parentNode.removeChild(old[i]);
  for (i = 0; i < boxes.length; i++) wrap.appendChild(createBoxElement(page, boxes[i]));
}

function renderInitialBoxes() {
  var blocks = document.querySelectorAll('.page-block');
  for (var i = 0; i < blocks.length; i++) {
    var page = blocks[i].getAttribute('data-page');
    var payload = pagePayload(page);
    renderBoxes(page, payload && payload.boxes ? payload.boxes : []);
  }
}

function getHunk(page, id) {
  var p = pagePayload(page);
  if (!p || !p.hunks) return null;
  for (var i = 0; i < p.hunks.length; i++) {
    if (p.hunks[i].id === id) return p.hunks[i];
  }
  showHunkStatus(page, id, 'internal error: unknown hunk ' + id, true);
  return null;
}

function hunkStateFor(page, id) {
  if (!hunkState[page]) hunkState[page] = {};
  if (!hunkState[page][id]) {
    var h = getHunk(page, id);
    hunkState[page][id] = {decision: 'pending', applied: h ? h.old : ''};
  }
  return hunkState[page][id];
}

function nthIndexOf(hay, needle, nth) {
  var idx = -1;
  for (var i = 0; i < nth; i++) {
    idx = hay.indexOf(needle, idx + 1);
    if (idx === -1) return -1;
  }
  return idx;
}

function styleHunk(page, id) {
  var li = document.getElementById('hunk-' + page + '-' + id);
  var st = hunkStateFor(page, id);
  if (li) {
    li.classList.remove('approved', 'rejected', 'edited');
    if (st.decision !== 'pending') li.classList.add(st.decision);
  }
  var undo = document.getElementById('undo-' + page + '-' + id);
  if (undo) undo.style.display = st.decision === 'pending' ? 'none' : '';
}

function replaceHunk(page, id, replacement) {
  var h = getHunk(page, id);
  if (!h) return false;
  var st = hunkStateFor(page, id);
  var ta = document.getElementById('text-' + page);
  if (!ta) {
    showHunkStatus(page, id, 'internal error: textarea missing', true);
    return false;
  }
  var needle = h.ctx_before + st.applied + h.ctx_after;
  var nth = h.unique ? 1 : h.occurrence;
  var pos = nthIndexOf(ta.value, needle, nth);
  if (pos === -1) {
    showHunkStatus(page, id, 'passage not found \\u2014 the text was edited by hand; apply manually', true);
    return false;
  }
  var start = pos + h.ctx_before.length;
  var end = start + st.applied.length;
  ta.value = ta.value.slice(0, start) + replacement + ta.value.slice(end);
  st.applied = replacement;
  return true;
}

function approveHunk(page, id) {
  var h = getHunk(page, id);
  if (!h) return;
  if (!replaceHunk(page, id, h.new)) return;
  var st = hunkStateFor(page, id);
  st.decision = 'approved';
  styleHunk(page, id);
  showHunkStatus(page, id, 'approved', false);
}

function rejectHunk(page, id) {
  var h = getHunk(page, id);
  if (!h) return;
  var st = hunkStateFor(page, id);
  if (st.applied !== h.old && !replaceHunk(page, id, h.old)) return;
  st.decision = 'rejected';
  styleHunk(page, id);
  showHunkStatus(page, id, 'rejected (original kept)', false);
}

function toggleEditHunk(page, id) {
  var box = document.getElementById('hunk-edit-' + page + '-' + id);
  var ta = document.getElementById('hunk-edit-text-' + page + '-' + id);
  if (!box || !ta) {
    showHunkStatus(page, id, 'internal error: edit box missing', true);
    return;
  }
  if (box.style.display === 'none') {
    var h = getHunk(page, id);
    var st = hunkStateFor(page, id);
    ta.value = st.decision === 'edited' ? st.applied : (h ? h.old : '');
    box.style.display = '';
    showHunkStatus(page, id, 'editing \\u2014 change the text, then Apply my text', false);
  } else {
    box.style.display = 'none';
    showHunkStatus(page, id, '', false);
  }
}

function applyEditedHunk(page, id) {
  var ta = document.getElementById('hunk-edit-text-' + page + '-' + id);
  if (!ta) {
    showHunkStatus(page, id, 'internal error: edit box missing', true);
    return;
  }
  if (!replaceHunk(page, id, ta.value)) return;
  var st = hunkStateFor(page, id);
  st.decision = 'edited';
  styleHunk(page, id);
  showHunkStatus(page, id, 'edited text applied', false);
}

function undoHunk(page, id) {
  var h = getHunk(page, id);
  if (!h) return;
  var st = hunkStateFor(page, id);
  if (st.applied !== h.old && !replaceHunk(page, id, h.old)) return;
  st.decision = 'pending';
  styleHunk(page, id);
  showHunkStatus(page, id, 'undone (original restored)', false);
}

function updateProgress(delta) {
  var el = document.getElementById('progress');
  var parts = el.textContent.match(/(\\d+) of (\\d+)/);
  if (!parts) return;
  var current = parseInt(parts[1], 10) + delta;
  el.textContent = current + ' of ' + parts[2] + ' reviewed';
}

function markDone(page) {
  var block = document.getElementById('block-' + page);
  if (block && !block.classList.contains('done')) {
    block.classList.add('done');
    updateProgress(1);
  }
}

function disablePage(page) {
  var block = document.getElementById('block-' + page);
  if (!block) return;
  var buttons = block.querySelectorAll('button');
  for (var i = 0; i < buttons.length; i++) buttons[i].disabled = true;
}

async function acceptPage(page) {
  var p = pagePayload(page);
  if (!p) return;
  var ta = document.getElementById('text-' + page);
  if (!ta) {
    showStatus(page, 'internal error: textarea missing', true);
    return;
  }
  var body = {page: page, text: ta.value};
  var pendingCount = 0;
  var approvedCount = 0;
  var editedCount = 0;
  var rejectedCount = 0;
  if (p.hunks && p.hunks.length) {
    var hunks = [];
    for (var i = 0; i < p.hunks.length; i++) {
      var h = p.hunks[i];
      var st = hunkStateFor(page, h.id);
      var item = {id: h.id, decision: st.decision};
      if (st.decision === 'edited') item.text = st.applied;
      if (st.decision === 'pending') pendingCount++;
      else if (st.decision === 'approved') approvedCount++;
      else if (st.decision === 'edited') editedCount++;
      else if (st.decision === 'rejected') rejectedCount++;
      hunks.push(item);
    }
    body.hunks = hunks;
  }
  showStatus(page, 'saving...', false);
  try {
    var resp = await fetch('/accept', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    var data = await resp.json();
    if (data.ok) {
      var baseMsg = data.reviewed === 'edited' ? 'saved (edited)' : 'accepted';
      var msg = baseMsg;
      if (p.hunks && p.hunks.length) {
        msg = approvedCount + ' approved, ' + editedCount + ' edited, ' +
              rejectedCount + ' rejected, ' + pendingCount +
              ' undecided (kept original) \\u2014 ' + baseMsg;
      }
      showStatus(page, msg, false);
      markDone(page);
      disablePage(page);
      stopBoxPolling(page);
    } else {
      showStatus(page, 'error: ' + (data.error || 'unknown'), true);
    }
  } catch (e) {
    showStatus(page, 'error: ' + e, true);
  }
}

// Hover linking between hunk items and their page-image boxes (delegated;
// mouseenter/leave do not bubble, so mouseover/out + relatedTarget checks).
function handleBoxHover(ev, on) {
  var t = ev.target;
  if (!t || !t.closest) return;
  var rel = ev.relatedTarget;
  var item = t.closest('.hunk-item');
  if (item && !(rel && item.contains(rel))) {
    var box = document.getElementById(
      'box-' + item.getAttribute('data-page') + '-h' + item.getAttribute('data-hunk'));
    if (box) box.classList.toggle('hot', on);
  }
  var boxEl = t.closest('.qc-box');
  if (boxEl && !(rel && boxEl.contains(rel))) {
    boxEl.classList.toggle('hot', on);
    var m = boxEl.id.match(/^box-(\\d+)-h(\\d+)$/);
    if (m) {
      var hunkEl = document.getElementById('hunk-' + m[1] + '-' + m[2]);
      if (hunkEl) {
        hunkEl.classList.toggle('hot', on);
        if (on) hunkEl.scrollIntoView({block: 'nearest'});
      }
    }
  }
}
document.addEventListener('mouseover', function (ev) { handleBoxHover(ev, true); });
document.addEventListener('mouseout', function (ev) { handleBoxHover(ev, false); });

// -- click-to-zoom on the page image --------------------------------------
// Per-page current zoom target (box id, or null when zoomed out).
var zoomState = {};
// Per-page live transform (s/tx/ty) so pan can adjust it while zoomed.
var zoomXf = {};
var pan = null;
var suppressClick = false;

function clampNum(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function clearGlow(page) {
  var vp = document.getElementById('viewport-' + page);
  if (!vp) return;
  var els = vp.querySelectorAll('.zoom-target');
  for (var i = 0; i < els.length; i++) els[i].classList.remove('zoom-target');
}

// Mark the correction box tied to the zoomed finding as active (one per page).
function setActiveHunk(page, boxId) {
  var block = document.getElementById('block-' + page);
  if (!block) return;
  var items = block.querySelectorAll('.hunk-item.active');
  for (var i = 0; i < items.length; i++) items[i].classList.remove('active');
  var m = boxId.match(/^box-(\\d+)-h(\\d+)$/);
  if (!m) return;
  var hunkEl = document.getElementById('hunk-' + m[1] + '-' + m[2]);
  if (hunkEl) hunkEl.classList.add('active');
}

function clearActiveHunk(page) {
  var block = document.getElementById('block-' + page);
  if (!block) return;
  var items = block.querySelectorAll('.hunk-item.active');
  for (var i = 0; i < items.length; i++) items[i].classList.remove('active');
}

function zoomOut(page) {
  var wrap = document.getElementById('imgwrap-' + page);
  if (!wrap) return;
  wrap.style.transform = '';
  zoomState[page] = null;
  delete zoomXf[page];
  var vp = document.getElementById('viewport-' + page);
  if (vp) vp.classList.remove('zoomed');
  clearGlow(page);
  clearActiveHunk(page);
}

function zoomTo(page, boxId, forceRetarget) {
  var box = document.getElementById(boxId);
  var wrap = document.getElementById('imgwrap-' + page);
  var vp = document.getElementById('viewport-' + page);
  if (!box || !wrap || !vp) return;
  // Re-click the current target zooms back out.
  if (zoomState[page] === boxId && !forceRetarget) { zoomOut(page); return; }
  clearGlow(page);

  // Every overlay carries explicit focus geometry. For a wrapped scan_vlm
  // finding this is segments[0] (the upper-line beginning in RTL reading
  // order), not the page-spanning compatibility envelope.
  var fx = parseFloat(box.getAttribute('data-focus-x0')) / 100;
  var fy = parseFloat(box.getAttribute('data-focus-y0')) / 100;
  var fw = parseFloat(box.getAttribute('data-focus-w')) / 100;
  var fh = parseFloat(box.getAttribute('data-focus-h')) / 100;
  if (!(fw > 0) || !(fh > 0)) return;

  var vw = vp.clientWidth, vh = vp.clientHeight;
  var iw = wrap.offsetWidth, ih = wrap.offsetHeight;
  if (!iw || !ih) return;

  var sW = 0.5 * vw / (fw * iw);
  var sH = 0.8 * vh / (fh * ih);
  var s = clampNum(Math.min(sW, sH), 1, 4);

  var cx = (fx + fw / 2) * iw;
  var cy = (fy + fh / 2) * ih;
  var tx = vw / 2 - s * cx;
  var ty = vh / 2 - s * cy;
  // Keep the scaled image covering the viewport (no empty gaps at the edges).
  tx = clampNum(tx, Math.min(0, vw - s * iw), Math.max(0, vw - s * iw));
  ty = clampNum(ty, Math.min(0, vh - s * ih), Math.max(0, vh - s * ih));

  wrap.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
  zoomState[page] = boxId;
  zoomXf[page] = {s: s, tx: tx, ty: ty};
  document.getElementById('viewport-' + page).classList.add('zoomed');
  setActiveHunk(page, boxId);
  // Pulse the target box once the glide has landed.
  setTimeout(function () {
    if (zoomState[page] !== boxId) return;
    box.classList.add('zoom-target');
  }, 580);
}

// Delegated click. Never preventDefault, so the hunk buttons' onclick handlers
// (Approve/Edit/Reject/Undo) keep firing normally.
document.addEventListener('click', function (ev) {
  if (suppressClick) { suppressClick = false; return; }
  var t = ev.target;
  if (!t || !t.closest) return;
  var onControl = !!t.closest('button, textarea, input, a');
  var item = t.closest('.hunk-item');
  // Inside a hunk item, its action buttons also drive the zoom (in/retarget
  // only — never toggle out mid-decision). Controls elsewhere (Accept, edit
  // textareas, links) are left alone.
  if (onControl && !(item && t.closest('button'))) return;

  if (item) {
    var hpage = item.getAttribute('data-page');
    var hid = item.getAttribute('data-hunk');
    var hboxId = 'box-' + hpage + '-h' + hid;
    if (onControl && zoomState[hpage] === hboxId) return;
    zoomTo(hpage, hboxId);
    return;
  }
  var box = t.closest('.qc-box');
  if (box) {
    var mb = box.id.match(/^box-(\\d+)-/);
    if (mb) zoomTo(mb[1], box.id);
    return;
  }
  var issueHead = t.closest('.qc-issue');
  if (issueHead) {
    var ipage = issueHead.getAttribute('data-page');
    var idx = issueHead.getAttribute('data-issue');
    var boxId = 'box-' + ipage + '-i' + idx;
    if (!document.getElementById(boxId)) {
      // Linked issue has no i-box of its own; use its group's first hunk box.
      var grp = issueHead.closest('.qc-group');
      var hi = grp ? grp.querySelector('.hunk-item') : null;
      if (hi) boxId = 'box-' + ipage + '-h' + hi.getAttribute('data-hunk');
    }
    zoomTo(ipage, boxId);
    return;
  }
  var vp = t.closest('.img-viewport');
  if (vp) { zoomOut(vp.getAttribute('data-page')); return; }
});

// -- click-drag to pan while zoomed ---------------------------------------
document.addEventListener('mousedown', function (ev) {
  var vp = ev.target.closest && ev.target.closest('.img-viewport');
  if (!vp) return;
  var page = vp.getAttribute('data-page');
  if (!zoomState[page] || !zoomXf[page]) return;
  pan = {page: page, startX: ev.clientX, startY: ev.clientY,
         baseTx: zoomXf[page].tx, baseTy: zoomXf[page].ty, moved: false};
  var wrap = document.getElementById('imgwrap-' + page);
  if (wrap) wrap.classList.add('panning');
  vp.classList.add('panning');
  ev.preventDefault();
});

document.addEventListener('mousemove', function (ev) {
  if (!pan) return;
  var page = pan.page;
  var dx = ev.clientX - pan.startX;
  var dy = ev.clientY - pan.startY;
  if (Math.abs(dx) + Math.abs(dy) > 4) pan.moved = true;
  var wrap = document.getElementById('imgwrap-' + page);
  var vp = document.getElementById('viewport-' + page);
  if (!wrap || !vp || !zoomXf[page]) return;
  var s = zoomXf[page].s;
  var iw = wrap.offsetWidth, ih = wrap.offsetHeight;
  var vw = vp.clientWidth, vh = vp.clientHeight;
  var newTx = clampNum(pan.baseTx + dx, Math.min(0, vw - s * iw), Math.max(0, vw - s * iw));
  var newTy = clampNum(pan.baseTy + dy, Math.min(0, vh - s * ih), Math.max(0, vh - s * ih));
  wrap.style.transform = 'translate(' + newTx + 'px,' + newTy + 'px) scale(' + s + ')';
  zoomXf[page].tx = newTx;
  zoomXf[page].ty = newTy;
});

document.addEventListener('mouseup', function (ev) {
  if (!pan) return;
  var wrap = document.getElementById('imgwrap-' + pan.page);
  if (wrap) wrap.classList.remove('panning');
  var vp = document.getElementById('viewport-' + pan.page);
  if (vp) vp.classList.remove('panning');
  if (pan.moved) suppressClick = true;
  pan = null;
});

document.addEventListener('keydown', function (ev) {
  if (ev.key === 'Escape') {
    for (var page in zoomState) { if (zoomState[page]) zoomOut(page); }
  }
});

// -- progressive scan_vlm box refinement -----------------------------------
// While the background pipeline warms the VLM cache, pages whose payload
// still carries plain "scan" boxes poll /boxes/<n> and swap the refined
// geometry in place. Gated on BBOX_REFINE so a --no-bbox-refine server
// (scan boxes present but never refined) never polls.
var pollTimers = {};

function hasScanBox(p) {
  var lists = [p.hunks || [], p.issues || []];
  for (var li = 0; li < lists.length; li++) {
    for (var i = 0; i < lists[li].length; i++) {
      var b = lists[li][i].box;
      if (b && b.source === 'scan') return true;
    }
  }
  return false;
}

function setRefineChip(page, show) {
  var el = document.getElementById('refine-chip-' + page);
  if (show && !el) {
    var row = document.querySelector('#block-' + page + ' .meta-row');
    if (!row) return;
    el = document.createElement('span');
    el.className = 'pill refine-chip';
    el.id = 'refine-chip-' + page;
    el.setAttribute('dir', 'rtl');
    el.setAttribute('lang', 'fa');
    el.textContent = 'دقت جعبه‌ها در حال بهبود…';
    row.appendChild(el);
  } else if (!show && el) {
    el.parentNode.removeChild(el);
  }
}

function stopBoxPolling(page) {
  if (pollTimers[page]) {
    clearTimeout(pollTimers[page]);
    delete pollTimers[page];
  }
  setRefineChip(page, false);
}

function applyRefinedBoxes(page, data) {
  // Patch the in-memory payload (pagePayload always serves from payloadCache
  // once parsed) so click-to-zoom and hunk apply use the new geometry.
  var p = pagePayload(page);
  var i;
  if (p) {
    if (p.hunks) {
      for (i = 0; i < p.hunks.length; i++) {
        p.hunks[i].box = data.hunk_boxes[String(p.hunks[i].id)] || null;
      }
    }
    if (p.issues) {
      for (i = 0; i < p.issues.length; i++) {
        p.issues[i].box = data.issue_boxes[String(i)] || null;
      }
    }
  }
  // Initial and progressively-refined overlays share the same renderer. If
  // this page was already zoomed, keep it active and recenter on the refined
  // beginning segment rather than toggling out.
  var activeId = zoomState[page] || null;
  renderBoxes(page, data.boxes || []);
  if (activeId) {
    if (document.getElementById(activeId)) zoomTo(page, activeId, true);
    else zoomOut(page);
  }
}

function pollBoxes(page) {
  fetch('/boxes/' + page).then(function (r) {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(function (data) {
    if (!pollTimers[page]) return;  // stopped (e.g. page accepted) mid-flight
    if (data.pending) return;       // pipeline still working; keep polling
    applyRefinedBoxes(page, data);
    stopBoxPolling(page);
  }).catch(function () { /* transient error; the next tick retries */ });
}

function startBoxPolling() {
  if (!BBOX_REFINE) return;
  var blocks = document.querySelectorAll('.page-block');
  var offset = 0;
  for (var i = 0; i < blocks.length; i++) {
    var block = blocks[i];
    if (block.classList.contains('done')) continue;
    var page = block.getAttribute('data-page');
    var p = pagePayload(page);
    if (!p || !hasScanBox(p)) continue;
    setRefineChip(page, true);
    // Self-rescheduling timeout (not setInterval) with staggered first
    // ticks, so a big book's polls don't all burst at once.
    (function (pg, delay) {
      pollTimers[pg] = setTimeout(function tick() {
        pollBoxes(pg);
        if (pollTimers[pg]) pollTimers[pg] = setTimeout(tick, 2500);
      }, delay);
    })(page, offset);
    offset += 350;
  }
}
renderInitialBoxes();
startBoxPolling();

document.getElementById('done-link').addEventListener('click', async function (ev) {
  ev.preventDefault();
  await fetch('/quit', {method: 'POST'});
  document.body.innerHTML = '<main><h2 style="font-family:sans-serif;color:#eee;padding:2rem;">Review server stopped. You may close this tab.</h2></main>';
});
</script>
</body>
</html>
"""

_env = Environment(autoescape=True)
_TEMPLATE = _env.from_string(_PAGE_TEMPLATE)


def _relpath_for_image(ws: Workspace, path: Path) -> str:
    rel = path.relative_to(ws.root)
    return "/media/" + str(rel).replace("\\", "/")


def _page_box_inputs(
    ws: Workspace, n: int
) -> tuple[dict, list[dict], list[dict], str, Optional[str]]:
    """(sidecar, issues, hunks, text, panel_kind) for page `n` — everything
    box-building needs, derived once here so _page_view, the /boxes route,
    the background refine pipeline, and tests/bbox_eval.py all agree.
    panel_kind is None when the page has no pending QC suggestion (issues and
    hunks are then empty and no boxes get built), else "no_suggestion" /
    "identical" / "hunks".
    """
    sidecar = _read_sidecar(ws, n)
    md_path = ws.page_md_path(n)
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""

    issues: list[dict] = []
    hunks: list[dict] = []
    panel_kind: Optional[str] = None
    qc_data = sidecar.get("qc")
    if (
        qc_data
        and qc_data.get("verdict") == "fail"
        and qc_data.get("suggestion_status") == "pending"
    ):
        issues = [
            {
                "type": i.get("type"),
                "description": i.get("description"),
                "snippet": i.get("snippet"),
                "bbox": i.get("bbox"),
            }
            for i in qc_data.get("issues") or []
        ]
        suggested = qc_data.get("suggested_text_md")
        if suggested is None:
            panel_kind = "no_suggestion"
        elif suggested == text:
            panel_kind = "identical"
        else:
            panel_kind = "hunks"
            hunks = _derive_hunks(text, suggested)
            _link_issues_to_hunks(issues, hunks)

        # Every flagged phrase without a linked fix hunk (all issues on a
        # no_suggestion/identical page, or unlinked issues on a hunks page)
        # gets an edit-only hunk so it can still be corrected in place.
        hunks += _derive_finding_hunks(text, issues, hunks)

    return sidecar, issues, hunks, text, panel_kind


def _page_view(ws: Workspace, n: int) -> dict:
    """Build the template + payload model for one surfaced page."""
    sidecar, issues, hunks, text, panel_kind = _page_box_inputs(ws, n)
    img_path = _image_path_for(ws, n)
    reviewed = bool(sidecar.get("reviewed")) and not sidecar.get("needs_review")

    validator_issues = (sidecar.get("validators") or {}).get("issues") or []
    qc_data = sidecar.get("qc")
    qc_issue_types = [i.get("type") for i in (qc_data or {}).get("issues", []) or []]

    qc_panel: Optional[dict] = None
    has_suggestion = panel_kind in ("identical", "hunks")
    matches_current = panel_kind == "identical"

    # Overlay boxes: sets "box" on each hunk / unlinked issue, returns the
    # template list (percent geometry). Must run before view_hunks copies.
    boxes = _build_boxes(ws, n, issues, hunks, text)

    if panel_kind is not None:
        view_hunks = []
        for h in hunks:
            label = None
            if h["issue_idx"] is not None and h["issue_idx"] < len(issues):
                iss = issues[h["issue_idx"]]
                label = f"{iss.get('type')} — {iss.get('description')}"
            view_hunks.append({**h, "issue_label": label})

        # Group hunks under the issue they were linked to (via issue_idx) so
        # the template can render each finding immediately followed by its
        # own fix box(es). Hunks with no linked issue land in a trailing
        # "Other correction" group; issues with no linked hunk still render
        # (their edit-only finding hunk lands under them via issue_idx).
        issue_to_hunks: dict[int, list[dict]] = {}
        other_hunks: list[dict] = []
        for h in view_hunks:
            idx = h["issue_idx"]
            if idx is not None and idx < len(issues):
                issue_to_hunks.setdefault(idx, []).append(h)
            else:
                other_hunks.append(h)
        groups = [
            {"idx": i, "issue": issue, "other_label": None, "hunks": issue_to_hunks.get(i, [])}
            for i, issue in enumerate(issues)
        ]
        if other_hunks:
            groups.append(
                {"idx": None, "issue": None, "other_label": "Other correction", "hunks": other_hunks}
            )

        qc_panel = {"kind": panel_kind, "issues": issues, "hunks": view_hunks, "groups": groups}

    payload = {
        "page": n,
        "issues": issues,
        "hunks": hunks,
        "boxes": boxes,
        "has_suggestion": has_suggestion,
        "suggestion_matches_current": matches_current,
    }

    return {
        "page": n,
        "model_used": sidecar.get("model_used", "?"),
        "confidence": sidecar.get("confidence", 0.0) or 0.0,
        "quality_score": sidecar.get("quality_score", 0.0) or 0.0,
        "flags": sidecar.get("flags", []) or [],
        "validator_issues": validator_issues,
        "qc_issue_types": qc_issue_types,
        "qc_panel": qc_panel,
        "boxes": boxes,
        "payload_json": _json_for_script(payload),
        "text": text,
        "image_url": _relpath_for_image(ws, img_path) if img_path else None,
        "reviewed": reviewed,
    }


def _render_index(
    ws: Workspace,
    surfaced: list[int],
    skipped: list[int],
    bbox_refine: bool = False,
) -> str:
    pages = [_page_view(ws, n) for n in surfaced]
    reviewed_count = sum(1 for p in pages if p["reviewed"])
    return _TEMPLATE.render(
        slug=ws.slug,
        pages=pages,
        skipped=skipped,
        reviewed_count=reviewed_count,
        total_count=len(pages),
        bbox_refine=bbox_refine,
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _ReviewState:
    def __init__(
        self,
        ws: Workspace,
        surfaced: list[int],
        skipped: list[int],
    ):
        self.ws = ws
        self.surfaced = surfaced
        self.skipped = skipped
        self.lock = threading.Lock()
        self.edited: set[int] = set()
        self.accepted: set[int] = set()
        # Per-correction (hunk) decisions the user clicked, keyed by page so a
        # re-accepted page overwrites rather than double-counts; the page-level
        # sets above stay coarse.
        self.hunk_decisions_by_page: dict[int, dict[str, int]] = {}
        self.httpd: Optional[ThreadingHTTPServer] = None


# Background scan-box refinement fan-out. The render path never calls the
# API, so these workers are what actually warms locate_vlm.json; the browser
# polls /boxes/<n> and swaps refined boxes in as pages complete.
_REFINE_WORKERS = 4


def _start_refine_pipeline(state: _ReviewState) -> None:
    """Warm the VLM box cache for every surfaced page, in review order, on
    _REFINE_WORKERS daemon threads. Each worker rebuilds a page's queries and
    located boxes exactly the way _build_boxes does, then calls
    refiner.refine — the refiner's in-flight set keeps two workers (or a
    worker and bbox_eval) from double-billing the same keys. Per-page errors
    are logged and skipped; the page just keeps its plain scan boxes.
    """
    refiner = _REFINER
    if refiner is None:
        return
    todo: queue.Queue[int] = queue.Queue()
    for n in state.surfaced:
        todo.put(n)
    print(
        f"bbox refine: background pipeline over {len(state.surfaced)} pages "
        f"({_REFINE_WORKERS} workers)",
        file=sys.stderr,
    )

    def _worker() -> None:
        while True:
            try:
                n = todo.get_nowait()
            except queue.Empty:
                return
            try:
                _sidecar, issues, hunks, text, _kind = _page_box_inputs(state.ws, n)
                if not issues and not hunks:
                    continue
                specs = _box_specs(issues, hunks, text)
                queries = [s[1] for s in specs]
                located = list(
                    _locate_queries_cached(
                        str(state.ws.pdf_path), n, text, tuple(queries)
                    )
                )
                refiner.refine(n, text, queries, located)
            except Exception as exc:
                print(f"bbox refine p{n}: pipeline error: {exc}", file=sys.stderr)

    for _ in range(_REFINE_WORKERS):
        threading.Thread(target=_worker, daemon=True).start()


def _make_handler(state: _ReviewState):
    ws = state.ws

    class Handler(BaseHTTPRequestHandler):
        server_version = "farsi2epub-review/2.0"

        def log_message(self, fmt, *args):  # silence default stderr logging
            pass

        # -- helpers ---------------------------------------------------

        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        # -- routes ------------------------------------------------------

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                index_html = _render_index(
                    ws, state.surfaced, state.skipped, bbox_refine=_REFINER is not None
                )
                self._send_bytes(index_html.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path.startswith("/boxes/"):
                try:
                    n = int(path[len("/boxes/"):])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                if n not in state.surfaced:
                    self.send_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                try:
                    self._send_json(_boxes_payload(ws, n))
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            if path == "/font/vazirmatn.ttf":
                if FONT_PATH.is_file():
                    self._send_bytes(FONT_PATH.read_bytes(), "font/ttf")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "font not found")
                return

            if path.startswith("/media/"):
                rel = path[len("/media/"):]
                candidate = (ws.root / rel).resolve()
                try:
                    candidate.relative_to(ws.root.resolve())
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN, "forbidden")
                    return
                if candidate.is_file():
                    self._send_bytes(candidate.read_bytes(), "image/png")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return

            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/accept":
                try:
                    payload = self._read_json_body()
                    n = int(payload["page"])
                    text = payload["text"]
                    if not isinstance(text, str):
                        raise ValueError("text must be a string")
                    hunk_decisions = payload.get("hunks") or []
                    if not isinstance(hunk_decisions, list):
                        raise ValueError("hunks must be a list")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return

                try:
                    md_path = ws.page_md_path(n)
                    on_disk_text = (
                        md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
                    )
                    sidecar = _read_sidecar(ws, n)
                    outcome = _decide_accept_outcome(sidecar, on_disk_text, text, hunk_decisions)

                    backed_up = False
                    if text != on_disk_text:
                        backed_up = _ensure_orig_backup(ws, n, on_disk_text)
                        _write_text_atomic(md_path, text)

                    sidecar["needs_review"] = False
                    sidecar["reviewed"] = outcome.reviewed
                    if outcome.suggestion_status is not None and sidecar.get("qc"):
                        sidecar["qc"]["suggestion_status"] = outcome.suggestion_status
                    _write_sidecar(ws, n, sidecar)
                except (OSError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return

                with state.lock:
                    if outcome.reviewed == "edited":
                        state.edited.add(n)
                        state.accepted.discard(n)
                    else:
                        state.accepted.add(n)
                        state.edited.discard(n)
                    counts: dict[str, int] = {}
                    for d in hunk_decisions:
                        dec = d.get("decision") if isinstance(d, dict) else None
                        if dec in ("approved", "edited", "rejected", "pending"):
                            counts[dec] = counts.get(dec, 0) + 1
                    state.hunk_decisions_by_page[n] = counts

                for detected_by, issue_type, old_frag, new_frag in outcome.events:
                    try:
                        qc.record_event(
                            book=ws.slug,
                            page=n,
                            detected_by=detected_by,
                            issue_type=issue_type,
                            source_type=ws.meta.get("source_type"),
                            model_used=sidecar.get("model_used"),
                            flags=sidecar.get("flags"),
                            confidence=sidecar.get("confidence"),
                            char_signals=qc.char_signals_from_diff(old_frag, new_frag),
                        )
                    except Exception:
                        pass

                self._send_json(
                    {
                        "ok": True,
                        "reviewed": outcome.reviewed,
                        "suggestion_status": outcome.suggestion_status,
                        "backed_up": backed_up,
                    }
                )
                return

            if path == "/quit":
                self._send_json({"ok": True})

                def _shutdown():
                    if state.httpd is not None:
                        state.httpd.shutdown()

                threading.Thread(target=_shutdown, daemon=True).start()
                return

            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    return Handler


def _find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    port = preferred
    for _ in range(200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                port += 1
                continue
            return port
    raise RuntimeError("could not find a free port")


# ---------------------------------------------------------------------------
# backgroundable server state (books/<slug>/review/server.json)
# ---------------------------------------------------------------------------


def _server_state_path(ws: Workspace) -> Path:
    return ws.review_dir / "server.json"


def _write_server_state(ws: Workspace, pid: int, port: int, url: str) -> None:
    """Atomically write the review server's state file (temp file + replace)."""
    path = _server_state_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pid": pid,
        "port": port,
        "url": url,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def _clear_server_state(ws: Workspace) -> None:
    """Best-effort removal of the server state file."""
    try:
        _server_state_path(ws).unlink()
    except FileNotFoundError:
        pass


def read_server_state(ws: Workspace) -> Optional[dict]:
    """Return the persisted server state if a server is actually live, else
    None (deleting a stale file if the pid is dead or nothing is listening).
    """
    path = _server_state_path(ws)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data["pid"])
        port = int(data["port"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _clear_server_state(ws)
        return None
    except PermissionError:
        pass  # alive, just owned by another user

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            pass
    except OSError:
        _clear_server_state(ws)
        return None

    return data


def launch_review_background(
    ws: Workspace,
    budget_all: bool = False,
    open_browser: bool = True,
    bbox_refine: bool = True,
    bbox_refine_model: str = MODEL_STRONG,
    lan: bool = False,
) -> str:
    """Ensure a review server is running for `ws`, starting one detached if
    needed. Returns its URL. Raises RuntimeError if a newly-spawned server
    doesn't come up within a short timeout.
    """
    existing = read_server_state(ws)
    if existing is not None:
        return existing["url"]

    ws.review_dir.mkdir(parents=True, exist_ok=True)
    log_path = ws.review_dir / "server.log"
    log_file = open(log_path, "w", encoding="utf-8")

    args = [sys.argv[0], "review", ws.slug, "--_child"]
    if budget_all:
        args.append("--all")
    args.append("--bbox-refine" if bbox_refine else "--no-bbox-refine")
    args += ["--bbox-refine-model", bbox_refine_model]
    if lan:
        args.append("--lan")

    subprocess.Popen(
        args,
        start_new_session=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = read_server_state(ws)
        if state is not None:
            return state["url"]
        time.sleep(0.25)

    raise RuntimeError(f"review server did not start within 5s; check {log_path}")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_review(
    ws: Workspace,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    budget_all: bool = False,
    bbox_refine: bool = True,
    bbox_refine_model: str = MODEL_STRONG,
) -> None:
    surfaced, skipped = _select_pages_for_review(ws, budget_all=budget_all)

    if not surfaced:
        if skipped:
            # Shouldn't really happen (skipped is a subset cut from surfaced
            # selection), but guard anyway.
            print(f"Nothing surfaced for review; {len(skipped)} page(s) auto-accepted despite flags.")
        else:
            print("Nothing needs review. All transcribed pages look good.")
        return

    if skipped:
        print(
            f"Review budget reached: {len(skipped)} flagged page(s) auto-accepted "
            f"despite flags (not shown): {skipped}"
        )

    # Scan-box VLM refinement: on by default, but the server must stay fully
    # usable offline — no key (or --no-bbox-refine) just means plain scan boxes.
    global _REFINER
    _REFINER = None
    if bbox_refine:
        llm.load_env()
        if os.environ.get("ANTHROPIC_API_KEY"):
            _REFINER = _ScanBoxRefiner(ws, bbox_refine_model)
        else:
            print("bbox refine: no ANTHROPIC_API_KEY, scan boxes will not be refined")

    state = _ReviewState(ws, surfaced, skipped)

    # Warm the VLM box cache in the background so GET / serves instantly with
    # whatever is cached; the browser polls /boxes/<n> for the rest.
    if _REFINER is not None:
        _start_refine_pipeline(state)

    handler_cls = _make_handler(state)

    free_port = _find_free_port(port)
    httpd = ThreadingHTTPServer(("0.0.0.0", free_port), handler_cls)
    state.httpd = httpd

    url = f"http://127.0.0.1:{free_port}/"
    _write_server_state(ws, os.getpid(), free_port, url)
    print(f"Review server running at {url}")
    print(f"Surfaced {len(surfaced)} page(s) for review: {surfaced}")
    print("Press Ctrl+C when finished (or click Done in the page).")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        server_thread.join()
    except KeyboardInterrupt:
        httpd.shutdown()
        server_thread.join()
    finally:
        httpd.server_close()
        _clear_server_state(ws)

    edited = len(state.edited)
    accepted = len(state.accepted)
    remaining = 0
    for n in surfaced:
        sidecar = _read_sidecar(ws, n)
        if sidecar.get("needs_review"):
            remaining += 1
    remaining += len(skipped)

    totals = {"approved": 0, "edited": 0, "rejected": 0, "pending": 0}
    for counts in state.hunk_decisions_by_page.values():
        for k, v in counts.items():
            totals[k] += v

    print("")
    print(f"Summary: pages       — {edited} edited, {accepted} accepted, {remaining} still flagged")
    if any(totals.values()):
        print(
            f"         corrections — {totals['approved']} approved, {totals['edited']} edited, "
            f"{totals['rejected']} rejected, {totals['pending']} undecided"
        )
