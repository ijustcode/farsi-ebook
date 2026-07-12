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
import json
import math
import re
import socket
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from jinja2 import Environment
from markupsafe import Markup

from . import qc, render
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
# QC box overlays (PDF text-layer location + model bbox fallback)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def _locate_snippets_cached(pdf_path: str, page_no: int, queries: tuple[str, ...]) -> tuple:
    """Memoized render.locate_snippets so browser refreshes don't rescan the
    PDF. Keyed on (pdf path, page, queries); any failure degrades to no boxes.
    """
    try:
        return tuple(render.locate_snippets(pdf_path, page_no, list(queries)))
    except Exception:
        return (None,) * len(queries)


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


def _build_boxes(
    ws: Workspace,
    n: int,
    issues: list[dict],
    hunks: list[dict],
    pdf_boxes_enabled: bool,
) -> list[dict]:
    """Attach a "box" (0-1 fractions + source, or None) to every hunk and
    every issue, and return the template-facing box list (percent values).

    Per hunk the query is its old text (context-prefixed for whitespace-only
    hunks); issues not linked to any hunk are queried by their snippet.
    A confident PDF text-layer hit wins; otherwise the (linked) issue's
    model-estimated sidecar bbox; otherwise no box.
    """
    for iss in issues:
        iss.setdefault("box", None)
    if not issues and not hunks:
        return []

    linked = {h["issue_idx"] for h in hunks if h["issue_idx"] is not None}
    specs: list[tuple[str, str, Optional[list], Optional[dict], Optional[dict]]] = []
    for h in hunks:
        query = (h["ctx_before"] + h["old"]) if h["ws_only"] else h["old"]
        fallback = None
        if h["issue_idx"] is not None and h["issue_idx"] < len(issues):
            fallback = issues[h["issue_idx"]].get("bbox")
        specs.append((f"h{h['id']}", query, fallback, h, None))
    for i, iss in enumerate(issues):
        if i in linked:
            continue
        specs.append((f"i{i}", iss.get("snippet") or "", iss.get("bbox"), None, iss))

    pdf_results: tuple = (None,) * len(specs)
    if pdf_boxes_enabled and specs:
        pdf_results = _locate_snippets_cached(
            str(ws.pdf_path), n, tuple(s[1] for s in specs)
        )

    boxes: list[dict] = []
    for (key, _query, fallback, hunk, issue), pdf_box in zip(specs, pdf_results):
        if pdf_box:
            box = {**pdf_box, "source": "pdf"}
        else:
            box = _bbox_to_box(fallback)
        if hunk is not None:
            hunk["box"] = box
        if issue is not None:
            issue["box"] = box
        if box is None:
            continue
        boxes.append(
            {
                "key": key,
                "x0": box["x0"] * 100.0,
                "y0": box["y0"] * 100.0,
                "x1": box["x1"] * 100.0,
                "y1": box["y1"] * 100.0,
                "w": (box["x1"] - box["x0"]) * 100.0,
                "h": (box["y1"] - box["y0"]) * 100.0,
                "source": box["source"],
            }
        )
    return boxes


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


# ---------------------------------------------------------------------------
# accept outcome (pure, unit-tested)
# ---------------------------------------------------------------------------


@dataclass
class _AcceptOutcome:
    reviewed: str                       # "edited" | "accepted"
    suggestion_status: Optional[str]    # new qc.suggestion_status or None = leave as-is
    events: list[tuple[str, str, str, str]]  # (detected_by, issue_type, old_frag, new_frag)


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
        suggestion_status = "rejected" if dangling else None
        events: list[tuple[str, str, str, str]] = []
        if text_changed:
            events.append(("human_edit", default_issue_type, on_disk_text, new_text))
        return _AcceptOutcome(reviewed, suggestion_status, events)

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
        if hunks and n_approved == len(hunks):
            status = "accepted"
        elif n_approved == 0 and n_edited == 0:
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
.img-wrap { position: relative; display: inline-block; max-width: 100%; }
.qc-box { position: absolute; border-radius: 2px; }
.qc-box-pdf { border: 2px solid #3a9ff0; background: rgba(58,159,240,.12); }
.qc-box-model { border: 2px dashed #f0a03a; background: rgba(240,160,58,.10); }
.qc-box.hot { outline: 2px solid #ffffff; }
.hunk-item.hot { border-color: #3a9ff0; }
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
.legend-swatch.legend-pdf { border: 2px solid #3a9ff0; background: rgba(58,159,240,.12); }
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
.qc-issue-head { font-size: 0.9rem; }
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
      <div class="img-wrap">
        <img src="{{ p.image_url }}" alt="page {{ p.page }}">
        {% for b in p.boxes %}
        <div class="qc-box qc-box-{{ b.source }}" id="box-{{ p.page }}-{{ b.key }}" style="left:{{ '%.2f'|format(b.x0) }}%;top:{{ '%.2f'|format(b.y0) }}%;width:{{ '%.2f'|format(b.w) }}%;height:{{ '%.2f'|format(b.h) }}%"></div>
        {% endfor %}
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
        {% if p.qc_panel.kind == 'hunks' %}
        {% for g in p.qc_panel.groups %}
        <div class="qc-group">
          <ul class="qc-issue-list">
            <li>
              {% if g.issue %}
              <div class="qc-issue-head">{{ g.issue.type }} &mdash; {{ g.issue.description }}</div>
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
            <li class="hunk-item" id="hunk-{{ p.page }}-{{ h.id }}" data-page="{{ p.page }}" data-hunk="{{ h.id }}">
              {% if h.ws_only %}
              <div class="hunk-diff">&para; line/paragraph-break change</div>
              {% else %}
              <div class="hunk-diff" dir="rtl"><span class="hunk-ctx">{{ h.show_before }}</span>{% if h.old %}<del><bdi>{{ h.old }}</bdi></del>{% endif %} {% if h.new %}<ins><bdi>{{ h.new }}</bdi></ins>{% endif %}<span class="hunk-ctx">{{ h.show_after }}</span></div>
              {% endif %}
              <div class="hunk-actions">
                <button onclick="approveHunk({{ p.page }}, {{ h.id }})">Approve</button>
                <button onclick="toggleEditHunk({{ p.page }}, {{ h.id }})">Edit</button>
                <button onclick="rejectHunk({{ p.page }}, {{ h.id }})">Reject</button>
                <button id="undo-{{ p.page }}-{{ h.id }}" style="display:none" onclick="undoHunk({{ p.page }}, {{ h.id }})">Undo</button>
                <span class="status-note" id="hunk-status-{{ p.page }}-{{ h.id }}"></span>
              </div>
              <div class="hunk-edit" id="hunk-edit-{{ p.page }}-{{ h.id }}" style="display:none">
                <textarea dir="rtl" lang="fa" id="hunk-edit-text-{{ p.page }}-{{ h.id }}">{{ h.new }}</textarea>
                <button onclick="applyEditedHunk({{ p.page }}, {{ h.id }})">Apply my text</button>
              </div>
            </li>
            {% endfor %}
          </ul>
          {% endif %}
        </div>
        {% endfor %}
        {% else %}
        <ul class="qc-issue-list">
          {% for issue in p.qc_panel.issues %}
          <li>
            <div class="qc-issue-head">{{ issue.type }} &mdash; {{ issue.description }}</div>
            {% if issue.snippet %}
            <div class="qc-snippet" dir="rtl">{{ issue.snippet }}</div>
            {% endif %}
          </li>
          {% endfor %}
        </ul>
        {% if p.qc_panel.kind == 'identical' %}
        <div class="qc-note">QC reported issues but its suggested text is identical to the current text &mdash; nothing to apply.</div>
        {% elif p.qc_panel.kind == 'no_suggestion' %}
        <div class="qc-note">QC reported issues but produced no suggested correction &mdash; fix manually if needed.</div>
        {% endif %}
        {% endif %}
        {% if p.boxes %}
        <div class="qc-legend">
          <span><span class="legend-swatch legend-pdf"></span>located in PDF text</span>
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
  if (p.has_suggestion && p.hunks && p.hunks.length) {
    var hunks = [];
    for (var i = 0; i < p.hunks.length; i++) {
      var h = p.hunks[i];
      var st = hunkStateFor(page, h.id);
      var item = {id: h.id, decision: st.decision};
      if (st.decision === 'edited') item.text = st.applied;
      if (st.decision === 'pending') pendingCount++;
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
      var msg = data.reviewed === 'edited' ? 'saved (edited)' : 'accepted';
      if (pendingCount > 0) {
        msg = pendingCount + ' correction(s) undecided (kept original) \\u2014 ' + msg;
      }
      showStatus(page, msg, false);
      markDone(page);
      disablePage(page);
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


def _page_view(ws: Workspace, n: int, pdf_boxes_enabled: bool = False) -> dict:
    """Build the template + payload model for one surfaced page."""
    sidecar = _read_sidecar(ws, n)
    md_path = ws.page_md_path(n)
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    img_path = _image_path_for(ws, n)
    reviewed = bool(sidecar.get("reviewed")) and not sidecar.get("needs_review")

    validator_issues = (sidecar.get("validators") or {}).get("issues") or []
    qc_data = sidecar.get("qc")
    qc_issue_types = [i.get("type") for i in (qc_data or {}).get("issues", []) or []]

    issues: list[dict] = []
    hunks: list[dict] = []
    qc_panel: Optional[dict] = None
    panel_kind: Optional[str] = None
    has_suggestion = False
    matches_current = False

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
            has_suggestion = True
            matches_current = True
            panel_kind = "identical"
        else:
            has_suggestion = True
            panel_kind = "hunks"
            hunks = _derive_hunks(text, suggested)
            _link_issues_to_hunks(issues, hunks)

    # Overlay boxes: sets "box" on each hunk / unlinked issue, returns the
    # template list (percent geometry). Must run before view_hunks copies.
    boxes = _build_boxes(ws, n, issues, hunks, pdf_boxes_enabled)

    if panel_kind == "hunks":
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
        # (description-only, no hunk-list under them).
        issue_to_hunks: dict[int, list[dict]] = {}
        other_hunks: list[dict] = []
        for h in view_hunks:
            idx = h["issue_idx"]
            if idx is not None and idx < len(issues):
                issue_to_hunks.setdefault(idx, []).append(h)
            else:
                other_hunks.append(h)
        groups = [
            {"issue": issue, "other_label": None, "hunks": issue_to_hunks.get(i, [])}
            for i, issue in enumerate(issues)
        ]
        if other_hunks:
            groups.append({"issue": None, "other_label": "Other correction", "hunks": other_hunks})

        qc_panel = {"kind": "hunks", "issues": issues, "hunks": view_hunks, "groups": groups}
    elif panel_kind is not None:
        qc_panel = {"kind": panel_kind, "issues": issues, "hunks": [], "groups": []}

    payload = {
        "page": n,
        "issues": issues,
        "hunks": hunks,
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
    pdf_boxes_enabled: bool = False,
) -> str:
    pages = [_page_view(ws, n, pdf_boxes_enabled=pdf_boxes_enabled) for n in surfaced]
    reviewed_count = sum(1 for p in pages if p["reviewed"])
    return _TEMPLATE.render(
        slug=ws.slug,
        pages=pages,
        skipped=skipped,
        reviewed_count=reviewed_count,
        total_count=len(pages),
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
        pdf_boxes_enabled: bool = False,
    ):
        self.ws = ws
        self.surfaced = surfaced
        self.skipped = skipped
        self.pdf_boxes_enabled = pdf_boxes_enabled
        self.lock = threading.Lock()
        self.edited: set[int] = set()
        self.accepted: set[int] = set()
        self.suggestion_accepted = 0
        self.suggestion_edited = 0
        self.suggestion_rejected = 0
        self.httpd: Optional[ThreadingHTTPServer] = None


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
                    ws,
                    state.surfaced,
                    state.skipped,
                    pdf_boxes_enabled=state.pdf_boxes_enabled,
                )
                self._send_bytes(index_html.encode("utf-8"), "text/html; charset=utf-8")
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
                    if outcome.suggestion_status == "accepted":
                        state.suggestion_accepted += 1
                    elif outcome.suggestion_status == "edited":
                        state.suggestion_edited += 1
                    elif outcome.suggestion_status == "rejected":
                        state.suggestion_rejected += 1

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


def _find_free_port(preferred: int) -> int:
    port = preferred
    for _ in range(200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue
            return port
    raise RuntimeError("could not find a free port")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_review(
    ws: Workspace,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    budget_all: bool = False,
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

    # One-time gate: draw PDF-located boxes only when the embedded text layer
    # is real Unicode text (glyph-soup/scanned books get model boxes only).
    try:
        pdf_boxes_enabled = render.text_layer_usable(ws.pdf_path)
    except Exception:
        pdf_boxes_enabled = False

    state = _ReviewState(ws, surfaced, skipped, pdf_boxes_enabled=pdf_boxes_enabled)
    handler_cls = _make_handler(state)

    free_port = _find_free_port(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", free_port), handler_cls)
    state.httpd = httpd

    url = f"http://127.0.0.1:{free_port}/"
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

    edited = len(state.edited)
    accepted = len(state.accepted)
    remaining = 0
    for n in surfaced:
        sidecar = _read_sidecar(ws, n)
        if sidecar.get("needs_review"):
            remaining += 1
    remaining += len(skipped)

    print("")
    print(f"Summary: edited {edited}, accepted {accepted}, remaining flagged {remaining}.")
    if state.suggestion_accepted or state.suggestion_edited or state.suggestion_rejected:
        print(
            "Suggestion outcomes: "
            f"accepted {state.suggestion_accepted}, "
            f"edited {state.suggestion_edited}, "
            f"rejected {state.suggestion_rejected}."
        )
