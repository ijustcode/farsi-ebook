"""Optional review box engine: Gemini ``box_2d`` boxes (``review --bbox-engine gemini``).

One Gemini 4 Carbon call per page: the page image plus a numbered list of
passages (target words in ⟦ ⟧ with up to four context words each side). The
request is arm A2 of ``tests/box_options_eval.py`` exactly (wording W1, schema
``Answer1``, thinking low, image resolution ultra_high); the wording, schema and
request constants live here and the eval imports them, so the two cannot drift
(``tests/gemini_boxes_regression.py`` holds the pair together).

Results are ``PlacementResult``: ``located`` when Gemini returned at least one
valid box for the passage, ``unresolved`` with a reason otherwise. Never an
estimate. Successful answers are cached on disk under
``books/<slug>/locate_evidence/gemini_boxes/<key>.json``; a cached page never
calls the API again. Page images are sent to Google.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import threading
import time
import uuid
from typing import Optional

from pydantic import BaseModel

from . import config, hard_reader
from .page_map import PlacementResult
from .scan import Budget, DEFAULT_MAX_COST, _claim, _read_json, _write_json

# ---------------------------------------------------------------------------
# Frozen request (arm A2 of tests/box_options_eval.py; do not change one side only)
# ---------------------------------------------------------------------------
# Convention from Google's documentation (read 2026-11-15:
# https://aistudio.google.com/docs/image-understanding, object detection;
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection):
# box_2d = [ymin, xmin, ymax, xmax], integers normalised to 0-1000 relative to
# the image's height (y) and width (x), origin top left. Image resolution
# (https://aistudio.google.com/docs/media-resolution): per-item "resolution"
# with a token budget (ultra_high 2240 tokens on Gemini 3/4); whether pixels are
# kept at any level is not documented, so the hi-res page image is sent as is.
A2_MODEL = "gemini-4-carbon"
A2_THINKING = "low"
A2_RESOLUTION = "ultra_high"
CONTEXT_WORDS = 4

A2_WORDING_W1 = (
    "You are given one scanned page image and a numbered list of passages from the text of that page. "
    "In each passage the target words are enclosed in ⟦ and ⟧; the words around them are context, given only so "
    "that you can find the right place on the page (the same words may occur elsewhere on the page).\n"
    "For each passage, find the target words on the page image and return the bounding box of the target words "
    "only, without the context words. If the target words run over more than one printed line, return one box per "
    "printed line, in reading order.\n"
    "Each box is box_2d = [ymin, xmin, ymax, xmax]: integers normalized to 0-1000 relative to the image height (y) "
    "and width (x), with the origin at the top left of the image.\n"
    "If you cannot find a passage on the page, return an empty list of boxes for it. Return every passage id "
    "exactly once.")


class Box1(BaseModel):
    box_2d: list[int]


class Passage1(BaseModel):
    id: int
    boxes: list[Box1]


class Answer1(BaseModel):
    passages: list[Passage1]


# Identity of everything above that shapes the request (part of the cache key).
WORDING_VERSION = (f"a2-W1/{hashlib.sha256(A2_WORDING_W1.encode()).hexdigest()[:16]}"
                   f"/think={A2_THINKING}/res={A2_RESOLUTION}/ctx={CONTEXT_WORDS}")
ATTEMPTS = 2
MAX_OUT_BASE, MAX_OUT_PER_PASSAGE, MAX_OUT_CAP = 400, 150, 12000
IMAGE_TOKENS = 2240
WORKERS = 4


def passage_list(items: list[dict]) -> str:
    rows = [f"{k}. {it['before']} ⟦{it['text']}⟧ {it['after']}".replace("  ", " ") for k, it in enumerate(items, 1)]
    return "Passages:\n" + "\n".join(rows)


def request_body(png: bytes, items: list[dict], max_out: int) -> dict:
    body = hard_reader.gemini_items_body(A2_MODEL, [A2_WORDING_W1, (png, "image/png"), passage_list(items)],
                                         Answer1, max_out, thinking_level=A2_THINKING)
    assert body["input"][1]["resolution"] == A2_RESOLUTION
    return body


def max_output_tokens(n_passages: int) -> int:
    return min(MAX_OUT_CAP, MAX_OUT_BASE + MAX_OUT_PER_PASSAGE * n_passages)


def worst_case(body: dict, image_tokens: int = IMAGE_TOKENS) -> float:
    """Upper bound in USD for one request (as tests/box_options_eval.worst_case)."""
    p = config.gemini_price()
    text = sum(len(it.get("text", "").encode("utf-8")) for it in body["input"] if it["type"] == "text")
    n_img = sum(1 for it in body["input"] if it["type"] == "image")
    inp = text // 2 + 200 + n_img * image_tokens
    return 1.10 * inp / 1e6 * p["in"] + body["generation_config"]["max_output_tokens"] / 1e6 * p["out"]


def box_to_segment(v) -> Optional[dict]:
    """box_2d [ymin, xmin, ymax, xmax] in 0-1000 -> {x0, y0, x1, y1} in 0-1; None when degenerate."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    if not all(type(c) in (int, float) and 0 <= c <= 1000 for c in v):
        return None
    y0, x0, y1, x1 = v
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x0": x0 / 1000, "y0": y0 / 1000, "x1": x1 / 1000, "y1": y1 / 1000}


def answers(parsed: dict, n: int) -> dict:
    """passage number (1-based) -> raw boxes; duplicates: first kept (as the eval's a2_answers)."""
    out = {}
    for p in (parsed or {}).get("passages") or []:
        if isinstance(p, dict) and type(p.get("id")) is int and 1 <= p["id"] <= n and p["id"] not in out:
            out[p["id"]] = p.get("boxes") or []
    return out


def placement_from_boxes(boxes: list) -> PlacementResult:
    segs = [s for s in (box_to_segment((b or {}).get("box_2d")) for b in boxes) if s is not None]
    if not segs:
        return PlacementResult("unresolved", reason="gemini: not found", evidence=["gemini"])
    segs.sort(key=lambda s: (s["y0"] + s["y1"], -s["x1"]))       # printed lines top to bottom
    box = {"x0": min(s["x0"] for s in segs), "y0": min(s["y0"] for s in segs),
           "x1": max(s["x1"] for s in segs), "y1": max(s["y1"] for s in segs),
           "segments": segs, "source": "gemini", "engine": "gemini", "status": "located",
           "kind": "words", "buffer_before": 0, "buffer_after": 0}
    return PlacementResult("located", box, evidence=["gemini"])


# ---------------------------------------------------------------------------
# Passages from the page Markdown
# ---------------------------------------------------------------------------
_FN_REF = re.compile(r"\[\^\d+\]:?")
_LINE_MARK = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|[-*+][ \t]+(?=\S))", re.M)
_INLINE = re.compile(r"\*\*|__|`|(?<!\S)\*|\*(?!\S)")
_HEMISTICH = re.compile(r"(?<=\s)---(?=\s)")


def plain(text: str) -> str:
    """Markdown syntax stripped (headings, emphasis, footnote marks, hemistich
    separators, unreadable markers), whitespace collapsed."""
    t = _FN_REF.sub(" ", text)
    t = _LINE_MARK.sub("", t)
    t = _INLINE.sub("", t)
    t = _HEMISTICH.sub(" ", t)
    t = t.replace(hard_reader.UNREADABLE_MD, " ")
    return " ".join(t.split())


def query_span(md: str, q) -> Optional[tuple[int, int]]:
    span = getattr(q, "span", None)
    if span is not None:
        a, b = span
        if 0 <= a <= b <= len(md):
            return a, b
        return None
    text = (getattr(q, "text", "") or "").strip()
    if text and md.count(text) == 1:          # an unlinked finding whose snippet occurs once
        a = md.index(text)
        return a, a + len(text)
    return None


def passage_for(md: str, q) -> tuple[Optional[dict], str]:
    """({before, text, after}, "") or (None, reason)."""
    span = query_span(md, q)
    if span is None:
        return None, "gemini: correction has no span in the page text"
    a, b = span
    before = plain(md[:a]).split()
    after = plain(md[b:]).split()
    target = plain(md[a:b])
    if not target:
        # An insertion point (or markup only): box the words on either side of it.
        left, right = before[-1:], after[:1]
        if not left and not right:
            return None, "gemini: no printed words at this correction"
        before, after = before[:-1] if left else before, after[1:] if right else after
        target = " ".join(left + right)
    return {"before": " ".join(before[-CONTEXT_WORDS:]), "text": target,
            "after": " ".join(after[:CONTEXT_WORDS])}, ""


# ---------------------------------------------------------------------------
# Service (drop-in for scan.PlacementService as review uses it)
# ---------------------------------------------------------------------------
class GeminiBoxService:
    engine = "gemini"

    def __init__(self, ws, *, mode="auto", max_cost=DEFAULT_MAX_COST, api_key=None, post=None):
        if mode not in {"auto", "offline"}:
            raise ValueError("bbox mode must be auto or offline")
        self.ws, self.mode, self.model = ws, mode, A2_MODEL
        self.root = ws.root / "locate_evidence" / "gemini_boxes"
        self.budget = Budget(max_cost)
        self.prompt_hash = hashlib.sha256(A2_WORDING_W1.encode()).hexdigest()
        self.api_key = api_key if api_key is not None else (hard_reader.gemini_key() if mode == "auto" else None)
        self._post = post or hard_reader._gemini_post
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self._failed: dict[str, str] = {}       # cache key -> reason (session only)
        self._scheduled: set[str] = set()
        self._summary: dict[int, dict] = {}
        self._png_shas: dict[int, tuple] = {}
        self._stopped: Optional[str] = None      # access/billing refusal: no more calls
        self._executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="gemini-boxes")
        self.run_id = uuid.uuid4().hex
        self.calls = 0

    # -- identities ------------------------------------------------------
    def source_hash(self):
        return hashlib.sha256(self.ws.pdf_path.read_bytes()).hexdigest()

    def _png_path(self, n):
        hi = self.ws.page_hires_path(n)
        if hi.is_file():
            return hi
        std = self.ws.page_image_path(n)
        return std if std.is_file() else None

    def _page_key(self, n):
        p = self._png_path(n)
        if p is None:
            return None
        st = p.stat()
        stamp = (str(p), st.st_size, st.st_mtime_ns, st.st_ino)
        with self.lock:
            hit = self._png_shas.get(n)
            if hit and hit[0] == stamp:
                return hit[1]
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.lock:
            self._png_shas[n] = (stamp, sha)
        return sha

    def _plan(self, n, md, queries):
        """(passages, index of each query's passage or None, reasons, png sha, cache key)."""
        items, index, reasons = [], [], []
        for q in queries:
            item, why = passage_for(md, q)
            if item is None:
                index.append(None)
                reasons.append(why)
            else:
                index.append(len(items))
                reasons.append("")
                items.append(item)
        png_sha = self._page_key(n)
        key = None
        if items and png_sha:
            key = hashlib.sha256(json.dumps([png_sha, self.model, WORDING_VERSION,
                                             [[it["before"], it["text"], it["after"]] for it in items]],
                                            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return items, index, reasons, png_sha, key

    def _key(self, n, md, q):
        return self._plan(n, md, [q])[4]

    def _cached(self, key):
        data = _read_json(self.root / f"{key}.json")
        return data if data.get("key") == key and isinstance(data.get("parsed"), dict) else None

    def _blocked_reason(self):
        if self.mode == "offline":
            return "gemini: offline, no cached boxes"
        if not self.api_key:
            return "no Gemini key"
        if self._stopped:
            return self._stopped
        return None

    # -- review interface --------------------------------------------------
    def results(self, n, md, queries, boxes=None):
        items, index, reasons, png_sha, key = self._plan(n, md, queries)
        out: list[PlacementResult] = []
        cached = self._cached(key) if key else None
        with self.lock:
            failed = self._failed.get(key) if key else None
        page_reason = None
        if items and not png_sha:
            page_reason = "gemini: no page image"
        elif items and cached is None:
            page_reason = failed or self._blocked_reason()
        ans = answers(cached["parsed"], len(items)) if cached else {}
        for i, q in enumerate(queries):
            if index[i] is None:
                out.append(PlacementResult("unresolved", reason=reasons[i], evidence=["gemini"]))
            elif cached is not None:
                out.append(placement_from_boxes(ans.get(index[i] + 1, [])))
            elif page_reason:
                out.append(PlacementResult("unresolved", reason=page_reason, evidence=["gemini"]))
            else:
                out.append(PlacementResult("pending", reason="evidence_pending"))
        with self.lock:
            self._summary[n] = {"located": sum(r.status == "located" for r in out), "total": len(out),
                                "pending": any(r.status == "pending" for r in out)}
        return out

    def apply_cached(self, n, md, queries, boxes=None):
        return [r.box for r in self.results(n, md, queries, boxes)]

    def pending(self, n, md, queries, boxes=None):
        return any(r.status == "pending" for r in self.results(n, md, queries, boxes))

    def page_map_summary(self, n):
        """No page map: keeps the Step 0 banner off (no detected_lines) and
        gives review a summary line."""
        with self.lock:
            s = dict(self._summary.get(n) or {"located": 0, "total": 0, "pending": False})
        return {"source": "gemini", "engine": "gemini", "status": "pending" if s["pending"] else "gemini",
                "gemini_located": s["located"], "gemini_total": s["total"]}

    def refine(self, n, md, queries, boxes=None, *, allow_api=None):
        """Synchronous: acquire the page if needed, then return the boxes."""
        allow_api = self.mode == "auto" if allow_api is None else allow_api
        items, _index, _reasons, png_sha, key = self._plan(n, md, queries)
        if allow_api and key and self._cached(key) is None and self._blocked_reason() is None:
            self._acquire(n, items, png_sha, key)
        return [r.box for r in self.results(n, md, queries, boxes)]

    def replay(self, n, md, queries, boxes=None):
        return self.refine(n, md, queries, boxes, allow_api=False)

    def schedule(self, n, md, queries, boxes=None):
        items, _index, _reasons, png_sha, key = self._plan(n, md, queries)
        if not key:
            return
        with self.lock:
            if self.cancelled.is_set() or key in self._scheduled or key in self._failed:
                return
            if self._cached(key) is not None or self._blocked_reason() is not None:
                return
            self._scheduled.add(key)

        def work():
            try:
                self._acquire(n, items, png_sha, key)
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

    # -- acquisition -------------------------------------------------------
    def _fail(self, key, reason):
        with self.lock:
            self._failed[key] = reason

    def _receipt(self):
        snap = self.budget.snapshot()
        _write_json(self.root / "last_run_cost.json",
                    {"run_id": self.run_id, "model": self.model, "spent": round(snap["spent"], 6),
                     "reserved": round(snap["reserved"], 6), "uncertain": round(snap["uncertain"], 6),
                     "limit": snap["limit"], "calls": self.calls, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def _acquire(self, n, items, png_sha, key):
        path = self.root / f"{key}.json"
        with _claim(path):
            if self._cached(key) is not None or self.cancelled.is_set():
                return
            png_path = self._png_path(n)
            png = png_path.read_bytes()
            if hashlib.sha256(png).hexdigest() != png_sha:
                self._fail(key, "gemini: page image changed")
                return
            body = request_body(png, items, max_output_tokens(len(items)))
            price = config.gemini_price()
            last = "gemini: no attempt"
            for attempt in range(ATTEMPTS):
                blocked = self._blocked_reason()
                if blocked or self.cancelled.is_set():
                    last = blocked or "gemini: cancelled"
                    break
                wc = worst_case(body)
                if not self.budget.reserve(wc):
                    last = "budget exhausted"
                    break
                try:
                    status, data = self._post(body, self.api_key)
                except OSError as exc:
                    self.budget.settle(wc, None)          # possibly billed: never recycled
                    self.calls += 1
                    last = f"gemini: transport {type(exc).__name__}"
                    self._receipt()
                    continue
                self.calls += 1
                if status >= 400:
                    pre = status in (400, 401, 403, 404, 413, 422, 429)
                    self.budget.settle(wc, 0.0 if pre else None)
                    msg = str(((data or {}).get("error") or {}).get("message", ""))[:200]
                    if self.api_key:
                        msg = msg.replace(self.api_key, "[key]")
                    low = msg.lower()
                    last = f"gemini: http {status}"
                    self._receipt()
                    if status in (401, 403) or "billing" in low or "quota" in low:
                        with self.lock:
                            self._stopped = f"gemini: access refused (http {status})"
                        last = self._stopped
                        break
                    if status in (400, 404, 413, 422):
                        break
                    continue
                parsed, usage = hard_reader.gemini_parse(data or {})
                cost = usage["input_tokens"] / 1e6 * price["in"] + usage["output_tokens"] / 1e6 * price["out"]
                self.budget.settle(wc, cost)
                self._receipt()
                if parsed is None:
                    last = "gemini: unparsed answer"
                    continue
                try:
                    parsed = Answer1.model_validate(parsed).model_dump()
                except Exception:                    # noqa: BLE001
                    last = "gemini: answer did not fit the schema"
                    continue
                _write_json(path, {"key": key, "model": self.model, "version": WORDING_VERSION,
                                   "png_sha256": png_sha, "page": n, "passages": items, "parsed": parsed,
                                   "usage": usage, "cost": cost, "price": price})
                return
            self._fail(key, last)
