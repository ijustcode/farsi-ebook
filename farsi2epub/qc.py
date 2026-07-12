"""Auto/manual quality-control orchestration with a learning history.

Auto QC verifies transcribed pages against their source image (the strong
model acting as a second reader), writes a suggested correction into each
page sidecar's ``qc`` key, and records every finding to ``qc_history.json``.
The history feeds a lightweight "common-denominator" model: features that
co-occur with real issues (a book's source type, a model, a flag, a
confidence bucket, a character signal) accumulate a *lift* weight, and the
next run risk-scores pages so verification budget lands where issues cluster.

Manual QC just launches the human review UI.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from . import config, llm
from .workspace import PROJECT_ROOT, Workspace

# Append-only learning store at the project root (gitignored).
QC_HISTORY_PATH = PROJECT_ROOT / "qc_history.json"

_HISTORY_LOCK = threading.Lock()

# Page-selection tuning.
RISK_THRESHOLD = 0.15      # risk_score at/above this selects an otherwise-clean page
RANDOM_SAMPLE_FRAC = 0.05  # ~5% of clean pages sampled anyway, so history keeps learning

# Cost-estimate assumptions for one verifier pass on MODEL_STRONG.
QC_TOKENS_IN = 4500
QC_TOKENS_OUT = 1500

_ZWNJ = "‌"
# QCIssue.type -> char_signal vocabulary term (unmapped types contribute no signal).
_ISSUE_SIGNAL = {
    "missing_heading": "heading",
    "wrong_heading_level": "heading",
    "zwnj_error": "zwnj",
    "heh_boundary": "heh_boundary",
    "footnote_marker": "footnote",
    "digit_error": "digits",
    "punctuation": "punctuation",
    "verse_structure": "verse",
    "table_structure": "table",
}


# ---------------------------------------------------------------------------
# history store
# ---------------------------------------------------------------------------


def _confidence_bucket(confidence: float) -> str:
    """Map a 0-1 confidence to one of the four history buckets."""
    if confidence < 0.7:
        return "<0.7"
    if confidence < 0.8:
        return "0.7-0.8"
    if confidence < 0.9:
        return "0.8-0.9"
    return "0.9-1.0"


def _load_history() -> dict:
    """Load the history store, tolerating a missing or corrupt file."""
    path = QC_HISTORY_PATH
    if not path.is_file():
        return {"events": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            return data
    except Exception:
        pass
    return {"events": []}


def _save_history(data: dict) -> None:
    """Write the history store atomically (temp file + replace)."""
    path = QC_HISTORY_PATH
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_events() -> list[dict]:
    """Return all recorded history events (empty list if none)."""
    return _load_history().get("events", [])


# detected_by values recorded by the human review UI (vs "auto_qc" etc.).
HUMAN_DETECTED_BY = {"human_edit", "suggestion_accepted", "suggestion_edited", "suggestion_rejected"}


def remove_human_events(book: str) -> int:
    """Drop every human-review event for `book` from the history store.

    Used by `review --reset` so re-reviewing a book does not double-count
    outcomes. Returns the number of events removed.
    """
    with _HISTORY_LOCK:
        data = _load_history()
        events = data.get("events", [])
        kept = [
            e
            for e in events
            if not (e.get("book") == book and e.get("detected_by") in HUMAN_DETECTED_BY)
        ]
        removed = len(events) - len(kept)
        if removed:
            data["events"] = kept
            _save_history(data)
        return removed


def record_event(
    book: str,
    page: int,
    detected_by: str,
    issue_type: str,
    *,
    source_type: Optional[str] = None,
    model_used: Optional[str] = None,
    flags: Optional[list[str]] = None,
    confidence: Optional[float] = None,
    char_signals: Optional[list[str]] = None,
) -> None:
    """Append one event to the history store (thread-safe, load-tolerant)."""
    event = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "book": book,
        "page": page,
        "source_type": source_type,
        "model_used": model_used,
        "detected_by": detected_by,
        "issue_type": issue_type,
        "features": {
            "flags": list(flags) if flags else [],
            "confidence_bucket": _confidence_bucket(confidence) if confidence is not None else None,
            "char_signals": list(char_signals) if char_signals else [],
        },
    }
    with _HISTORY_LOCK:
        data = _load_history()
        data["events"].append(event)
        _save_history(data)


# ---------------------------------------------------------------------------
# character-level change signals (difflib)
# ---------------------------------------------------------------------------


def _verse_mask(lines: list[str]) -> list[bool]:
    """True for each line that sits inside a ```verse fenced block."""
    mask = [False] * len(lines)
    fence: Optional[str] = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("```"):
            fence = s[3:].strip() if fence is None else None
            continue  # the fence line itself is not counted as content
        if fence == "verse":
            mask[i] = True
    return mask


def char_signals_from_diff(old_text: str, new_text: str) -> list[str]:
    """Classify what changed between two page texts into char-signal terms.

    Vocabulary: heading, zwnj, heh_boundary, digits, punctuation, footnote,
    verse, table. Line-structural signals (heading/table/verse) come from a
    line diff; content signals from a word diff. Returns a sorted unique list.
    """
    signals: set[str] = set()

    # -- line-structural signals -----------------------------------------
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    old_verse = _verse_mask(old_lines)
    new_verse = _verse_mask(new_lines)
    line_sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in line_sm.get_opcodes():
        if tag == "equal":
            continue
        for idx in range(i1, i2):
            s = old_lines[idx].strip()
            if s.startswith("#"):
                signals.add("heading")
            if s.startswith("|"):
                signals.add("table")
            if old_verse[idx]:
                signals.add("verse")
        for idx in range(j1, j2):
            s = new_lines[idx].strip()
            if s.startswith("#"):
                signals.add("heading")
            if s.startswith("|"):
                signals.add("table")
            if new_verse[idx]:
                signals.add("verse")

    # -- content signals over the changed word fragments -----------------
    old_words = old_text.split()
    new_words = new_text.split()
    word_sm = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    changed: list[str] = []
    for tag, i1, i2, j1, j2 in word_sm.get_opcodes():
        if tag == "equal":
            continue
        changed.extend(old_words[i1:i2])
        changed.extend(new_words[j1:j2])
    frag = " ".join(changed)

    if _ZWNJ in frag:
        signals.add("zwnj")
    # ه at a join boundary: followed by ZWNJ or by a common attached suffix.
    if re.search(rf"ه(?:{_ZWNJ}|ها|ای|اش|ام|اند|یی|تر|تری)", frag):
        signals.add("heh_boundary")
    if re.search(r"[0-9۰-۹]", frag):
        signals.add("digits")
    if re.search(r"[،؛؟«».:]", frag):
        signals.add("punctuation")
    if "[^" in frag:
        signals.add("footnote")

    return sorted(signals)


# ---------------------------------------------------------------------------
# feature weights ("common denominators") + risk score
# ---------------------------------------------------------------------------


def _event_features(event: dict) -> list[str]:
    """Namespaced feature keys describing one event."""
    feats: list[str] = []
    st = event.get("source_type")
    if st:
        feats.append(f"source:{st}")
    mu = event.get("model_used")
    if mu:
        feats.append(f"model:{mu}")
    features = event.get("features") or {}
    for f in features.get("flags") or []:
        feats.append(f"flag:{f}")
    cb = features.get("confidence_bucket")
    if cb:
        feats.append(f"conf:{cb}")
    for s in features.get("char_signals") or []:
        feats.append(f"signal:{s}")
    return feats


def compute_feature_weights(events: list[dict]) -> dict[str, float]:
    """Per-feature issue *lift*, clamped to [0.5, 3.0].

    An event is an issue event unless its ``issue_type`` is "pass". For a
    feature observed in >= 3 events, lift is the feature's issue rate divided
    by the overall issue rate; features seen fewer times (or when no issues
    exist at all) get the neutral weight 1.0.
    """
    total = len(events)
    if total == 0:
        return {}
    issue_total = sum(1 for e in events if e.get("issue_type") != "pass")
    base_rate = issue_total / total  # 0.0 when there are no issue events yet

    feat_total: Counter = Counter()
    feat_issue: Counter = Counter()
    for e in events:
        is_issue = e.get("issue_type") != "pass"
        for f in set(_event_features(e)):
            feat_total[f] += 1
            if is_issue:
                feat_issue[f] += 1

    weights: dict[str, float] = {}
    for f, n in feat_total.items():
        if n < 3 or base_rate == 0:
            weights[f] = 1.0
        else:
            lift = (feat_issue[f] / n) / base_rate
            weights[f] = max(0.5, min(3.0, lift))
    return weights


def risk_score(sidecar: dict, source_type: str, weights: dict[str, float]) -> float:
    """(1 - quality_score) x product of the page's matching feature lifts."""
    q = sidecar.get("quality_score")
    base = 1.0 - (q if q is not None else 0.0)

    feats: list[str] = []
    if source_type:
        feats.append(f"source:{source_type}")
    mu = sidecar.get("model_used")
    if mu:
        feats.append(f"model:{mu}")
    for f in sidecar.get("flags") or []:
        feats.append(f"flag:{f}")
    conf = sidecar.get("confidence")
    if conf is not None:
        feats.append(f"conf:{_confidence_bucket(conf)}")

    product = 1.0
    for f in feats:
        product *= weights.get(f, 1.0)
    return base * product


# ---------------------------------------------------------------------------
# sidecar helpers
# ---------------------------------------------------------------------------


def _read_sidecar(ws: Workspace, n: int) -> dict:
    with open(ws.page_meta_path(n), "r", encoding="utf-8") as f:
        return json.load(f)


def _write_sidecar(ws: Workspace, n: int, data: dict) -> None:
    """Write the sidecar JSON atomically (temp file + replace)."""
    path = ws.page_meta_path(n)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _image_path_for(ws: Workspace, n: int) -> Optional[Path]:
    """Prefer the high-res render, fall back to standard (mirrors review.py)."""
    hi = ws.page_hires_path(n)
    if hi.is_file():
        return hi
    std = ws.page_image_path(n)
    if std.is_file():
        return std
    return None


# ---------------------------------------------------------------------------
# auto QC orchestration
# ---------------------------------------------------------------------------


@dataclass
class _QCState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    total_cost: float = 0.0
    verified: int = 0
    passed: int = 0
    failed: int = 0


def _qc_per_page_cost() -> float:
    prices = config.PRICES[config.MODEL_STRONG]
    return (QC_TOKENS_IN / 1_000_000) * prices["in"] + (QC_TOKENS_OUT / 1_000_000) * prices["out"]


def _clean_bbox(b) -> Optional[list[int]]:
    """Sanitize a model-reported bbox: must be a 4-item list/tuple of numbers.
    Values are rounded to ints and clamped to [0, 1000]; the box must still
    have positive width and height afterwards. Anything else returns None.
    """
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    vals: list[int] = []
    for v in b:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        vals.append(max(0, min(1000, int(round(v)))))
    x0, y0, x1, y1 = vals
    if x0 >= x1 or y0 >= y1:
        return None
    return vals


def _select_pages(ws: Workspace, all_pages: bool, force: bool = False) -> tuple[list[int], list[int]]:
    """Choose pages to verify. Returns (selected, skipped_pending).

    Pages whose previous QC suggestion is still pending are normally diverted
    to `skipped_pending`; with `force` they go through normal selection (and
    a re-verification will replace their old suggestion).
    """
    weights = compute_feature_weights(load_events())
    source_type = ws.meta.get("source_type", "")

    forced: list[int] = []          # always: needs_review or validator issues
    risky: list[int] = []           # risk_score >= threshold
    clean: list[int] = []           # everything else (random-sample pool)
    skipped_pending: list[int] = []

    for n in ws.pages_done():
        try:
            sc = _read_sidecar(ws, n)
        except Exception:
            continue
        if sc.get("is_blank"):
            continue
        qc = sc.get("qc")
        if not force and qc and qc.get("suggestion_status") == "pending":
            skipped_pending.append(n)
            continue
        if all_pages:
            forced.append(n)
            continue
        issues = (sc.get("validators") or {}).get("issues") or []
        if sc.get("needs_review") or issues:
            forced.append(n)
        elif risk_score(sc, source_type, weights) >= RISK_THRESHOLD:
            risky.append(n)
        else:
            clean.append(n)

    selected = set(forced) | set(risky)
    if not all_pages and clean:
        k = max(1, round(len(clean) * RANDOM_SAMPLE_FRAC))
        k = min(k, len(clean))
        selected.update(random.sample(clean, k))
    return sorted(selected), sorted(skipped_pending)


def _verify_one(ws: Workspace, client, n: int, source_type: str, state: _QCState) -> str:
    """Verify one page, write its qc sidecar key, record events. Returns status."""
    img_path = _image_path_for(ws, n)
    if img_path is None:
        raise FileNotFoundError(f"no rendered image for page {n}")
    png = img_path.read_bytes()
    text_md = ws.page_md_path(n).read_text(encoding="utf-8")

    report, _usage, cost = llm.qc_verify_page(client, png, text_md, config.MODEL_STRONG, n)
    is_fail = report.verdict == "fail"

    # Update the sidecar (re-read fresh so we don't clobber concurrent writes).
    sc = _read_sidecar(ws, n)
    sc["qc"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verifier_model": config.MODEL_STRONG,
        "verdict": report.verdict,
        "issues": [
            {
                "type": i.type,
                "description": i.description,
                "snippet": i.snippet,
                "bbox": _clean_bbox(i.bbox),
            }
            for i in report.issues
        ],
        "suggested_text_md": report.suggested_text_md if is_fail else None,
        "suggestion_status": "pending" if is_fail else None,
        "cost_usd": round(cost, 6),
    }
    if is_fail:
        sc["needs_review"] = True
    _write_sidecar(ws, n, sc)

    # Record history events for the learning model.
    model_used = sc.get("model_used")
    flags = sc.get("flags")
    confidence = sc.get("confidence")
    if is_fail and report.issues:
        for issue in report.issues:
            sig = _ISSUE_SIGNAL.get(issue.type)
            record_event(
                ws.slug, n, "auto_qc", issue.type,
                source_type=source_type, model_used=model_used, flags=flags,
                confidence=confidence, char_signals=[sig] if sig else [],
            )
    else:
        record_event(
            ws.slug, n, "auto_qc", "pass",
            source_type=source_type, model_used=model_used, flags=flags,
            confidence=confidence, char_signals=[],
        )

    with state.lock:
        state.total_cost += cost
        state.verified += 1
        if is_fail:
            state.failed += 1
        else:
            state.passed += 1

    n_issues = len(report.issues)
    tag = "FAIL" if is_fail else "pass"
    return f"page {n:>4}  {tag}  {n_issues} issue(s)  ${cost:.4f}"


def _run_auto(ws: Workspace, all_pages: bool, assume_yes: bool, force: bool = False) -> None:
    source_type = ws.meta.get("source_type", "")

    selected, skipped_pending = _select_pages(ws, all_pages, force=force)
    if skipped_pending:
        click.echo(f"Skipping {len(skipped_pending)} page(s) with a pending qc suggestion: {skipped_pending}")
    if force:
        repending = []
        for n in selected:
            try:
                sc = _read_sidecar(ws, n)
            except Exception:
                continue
            if (sc.get("qc") or {}).get("suggestion_status") == "pending":
                repending.append(n)
        if repending:
            click.echo(
                f"--force: re-verifying {len(repending)} page(s) with pending suggestions "
                "(their old suggestions will be replaced)"
            )
    if not selected:
        click.echo("No pages to verify.")
        return

    per_page = _qc_per_page_cost()
    estimate = per_page * len(selected)
    click.echo(f"Auto QC will verify {len(selected)} page(s) with {config.MODEL_STRONG}:")
    click.echo(f"  {selected}")
    click.echo(f"Estimated cost: ${estimate:.4f} (~{QC_TOKENS_IN} in / {QC_TOKENS_OUT} out tokens per page)")
    if not assume_yes and not click.confirm("Proceed?"):
        click.echo("Aborted.")
        return

    client = llm.get_client()
    state = _QCState()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_verify_one, ws, client, n, source_type, state): n for n in selected}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                click.echo(f"  {fut.result()}")
            except Exception as exc:
                click.echo(f"  page {n:>4}  ERROR: {exc}")

    click.echo("")
    click.echo(
        f"Verified {state.verified} page(s): {state.passed} passed, {state.failed} failed. "
        f"Total cost: ${state.total_cost:.4f}"
    )

    if state.failed:
        from . import review  # lazy: review imports qc (circular at module level)

        click.echo("Launching review on flagged pages ...")
        review.run_review(ws, budget_all=True)


def run_qc(
    ws: Workspace,
    mode: str,
    all_pages: bool = False,
    assume_yes: bool = False,
    force: bool = False,
) -> None:
    """Run quality control over a book workspace.

    mode "auto": risk-select pages (or all), confirm cost, verify each with
    ``llm.qc_verify_page``, write the sidecar ``qc`` key + history events, then
    open the review UI on any flagged pages. `force` also re-verifies pages
    whose previous suggestion is still pending (replacing it).
    mode "manual": launch the review UI directly.
    """
    if mode == "manual":
        from . import review  # lazy: avoid circular import (review imports qc)

        review.run_review(ws, budget_all=all_pages)
    elif mode == "auto":
        _run_auto(ws, all_pages, assume_yes, force=force)
    else:
        raise ValueError(f"unknown qc mode: {mode!r} (expected 'auto' or 'manual')")
