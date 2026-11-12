#!/usr/bin/env python3
"""Real-page geometry truth: freeze audited pages, then score line finders and
word placers against them (plan §6.2 real truth, §1 targets, §13 O1/O2).

Evaluation tooling only; production code never imports ``tests/``. No API
calls: the live-engine adapters run ``PlacementService`` offline on a scratch
copy of the workspace, with a reader that raises and ``ANTHROPIC_BASE_URL``
forced to a dead local port.

Commands (repo root, ``./venv/bin/python``):

    tests/real_truth_eval.py freeze [--key slug/page ...] [--force] [--class C]
    tests/real_truth_eval.py check                       # truth sanity report
    tests/real_truth_eval.py run [--out out/real_truth] [--seed 7] [--spans 60]

Files (private, gitignored): ``books/<slug>/truth/truth_freeze.json`` (one
record per frozen page) and the results under ``out/real_truth/``.

Adapter interfaces (Step 3 / Step 4 candidates plug in here; see ``LINE_FINDERS``
and ``WORD_PLACERS``):

* line finder: ``find_lines(page: PageInput) -> list[Region]``
* word placer: ``place(page: PageInput, targets: list[Target]) -> list[Placed]``

``Region`` is ``{"rect": [x0, y0, x1, y1]}`` or ``{"quad": [[x, y] * 4]}``,
normalised to the full page (x right, y down). ``Placed`` is ``{"status":
"located"|"buffered"|"line"|"estimated"|"unresolved", "segments": [Region...],
"confidence": float|None, "reason": str}``; the union of segments is graded,
never its envelope. A target is one truth word (the word map) or a reading-order
span of 2-6 words (closer to a review correction).
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import box_audit as ba  # noqa: E402
import box_audit_draft as bd  # noqa: E402

SCHEMA = "farsi2epub.real-truth-eval/v1"
FREEZE_SCHEMA = "farsi2epub.real-truth-freeze/v1"
FREEZE_NAME = "truth_freeze.json"
HELD_OUT = set(ba.HELD_OUT)          # hadpi p3-p5: refused everywhere
SLUG_CLASS = {"hadpi-articles": "typewritten_photo",
              "ghessehaye-majid": "printed_scan",
              "bachehaye_ghali": "printed_scan"}
OWNER, AGENT = "owner_audited", "agent_audited"

COVER = 0.90          # plan §1: >= 90% of the truth word's area inside the box
INTRUDE_LETTERS = 0.5  # intrusion into any other word must stay under half a letter
OVERREACH = 0.25      # blank area outside the allowance <= 25% of the target area
EDGE, TOUCH = 0.50, 0.25  # line edges: word-area share of a truth line
MIN_TOUCH = 0.10      # a word is "touched" by a box at >= 10% of its area

DEFINITIONS = {
    "plan_section_1_quote": ('"correct" = at least 90% of the truth word\'s area inside the box and '
                             'at most one adjacent word of excess'),
    "word_correct_v1": (
        "A box (union of its segments, polygons in truth-image pixels) is correct for target words T "
        "when (1) every target word quad is >= 90% inside the box; (2) every other truth word W on the "
        "page loses less than 0.5 letter to the box, where letters lost = letters(W) * area(W & box) / "
        "area(W) and letters() counts non-combining characters; (3) box area outside the allowance "
        "(each target's line-frame box widened by half a letter on each side along the line and by 25% "
        "of its height above and below) is <= 25% of the targets' area. (2) and (3) make an over-wide "
        "box fail, both into neighbouring ink and into blank space. CHOICE: this is stricter than the "
        "plan's literal 'at most one adjacent word of excess', which would accept a box that swallows "
        "a whole neighbour; the plan-literal rate is reported beside it."),
    "word_correct_plan_literal": (
        "Every target >= 90% covered, and at most one non-target word touched (>= 10% of its area), "
        "which must be the reading-order neighbour of the span."),
    "located_correct_rate": "words with status 'located' and word_correct_v1, over all graded words",
    "shown_covering_only": ("diagnostic only, NOT a correctness rate: any shown box covering >= 90% of every "
                            "target, however wide (the v5-style blind spot); shows how much over-width hides"),
    "located_precision": "of words with status 'located', the share that are word_correct_v1",
    "wrong_line": ("a box is shown (any status), it touches text, and the truth line holding most of "
                   "its covered word area is not a target line; over all graded words"),
    "no_box": "no segments returned (unresolved, pending, or not queryable), over all graded words",
    "shifted": ("single word, box shown, right line, not correct, and another word on the same line is "
                "covered more (fraction >= 0.5) than the target: the box sits on a neighbour's ink"),
    "O1_cascade": ("a maximal run of >= 2 consecutive words of one line that are all 'shifted' with "
                   "the same direction of shift (plan §13 O1: one stray ink run displaces every later "
                   "box on the line)"),
    "O2_boundary": ("single word, box shown, right line, the target is the best-covered word "
                    "(>= 0.5), but not correct because an edge is off by about a letter or more: the "
                    "target misses >= 0.5 letter, or the box takes >= 0.5 letter (but under half) of "
                    "the next/previous word on the same line (plan §13 O2). A box that swallows half "
                    "a neighbour or more is 'too_wide_other_word', not O2"),
    "lines": ("Units are truth lines, measured by word area: share(L, C) = area(words of L inside C) / "
              "area(words of L). Edge: share >= 0.5. Touch: share >= 0.25. Clean: L has exactly one "
              "edge C and C touches no other truth line. Merged candidate: edges to >= 2 truth lines. "
              "Split truth line: no edge, but >= 2 candidates each hold >= 15% and together >= 50%, or "
              ">= 2 edges. Missed: < 50% of L's word area inside the union of all candidates. Spurious: "
              "a candidate touching no graded line at >= 5% that is not mostly (>= 50% of its area) "
              "inside an ignore zone; fragment: touches a line at 5-25% only. Clean recall = clean / "
              "graded truth lines; clean precision = clean / non-ignored candidates. Truth geometry is "
              "the sloped line-frame quads, never their axis-aligned bounds."),
    "word_segments": ("Text-free word boxes (the current detector's word-gap runs): a truth word is "
                      "cleanly segmented when some single box is word_correct_v1 for it. Merged box: "
                      ">= 50% of >= 2 truth words. Split word: no box >= 90% and >= 2 boxes >= 15%. "
                      "Missed: < 50% inside the union. Spurious: touches no graded word at >= 10% and "
                      "is not mostly inside an ignore zone."),
    "excluded": ("Lines with an owner text flag are excluded automatically at freeze. Their words are "
                 "in no denominator, they are never targets, and because their geometry is known to be "
                 "wrong they are not used for intrusion or wrong-line tests; candidate regions mostly "
                 "inside them (expanded by one line spacing up and down) or inside annotated non-text "
                 "regions (running heads, dialogue markers) are 'ignored', not spurious."),
    "intervals": ("Wilson 95% intervals treat words as independent draws. Words on one page are "
                  "correlated (same image, same lines), so these intervals are optimistic; with two "
                  "pages per class no page-level interval is meaningful and none is given."),
    "page_frame": ("All coordinates are fractions of the full PDF page; truth pixels are the audit "
                   "image's pixels (the same full-page frame at its own resolution)."),
}


class TruthError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def letters(token: str) -> float:
    return bd.char_weight(token or "")


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def _fold(tok: str) -> str:
    from farsi2epub import locate
    return locate._fold_word(tok)


# ---------------------------------------------------------------------------
# Truth pages and the freeze manifest
# ---------------------------------------------------------------------------
@dataclass
class TWord:
    li: int
    wi: int
    text: str
    quad: list            # 4 (x, y) image pixels
    fbox: list            # [u0, v0, u1, v1] in its line frame
    letters: float
    graded: bool


@dataclass
class TLine:
    index: int
    frame: dict
    box: list
    quad: list
    words: list
    excluded: bool
    reason: str | None = None


@dataclass
class TruthPage:
    slug: str
    page: int
    cls: str
    provenance: str
    W: int
    H: int
    image_path: Path
    image_sha256: str
    audit_path: Path
    audit_sha256: str
    lines: list
    ignore_zones: list          # list of quads (px)
    freeze: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.slug}/{self.page}"

    def graded_words(self):
        return [w for l in self.lines if not l.excluded for w in l.words]

    def word(self, li, wi):
        return self.lines[li].words[wi]


UI_STATE = ("cursor", "revision")   # bumped by navigation in the audit tool, not truth


def content_sha(doc: dict) -> str:
    """SHA-256 of the audit document without the audit tool's UI state.

    The box-audit tool increments ``revision`` and moves ``cursor`` on every
    action, including navigation that changes nothing, so the file bytes change
    while the truth does not. Every line, word, flag, status and image field is
    hashed."""
    return hashlib.sha256(ba.canonical({k: v for k, v in doc.items() if k not in UI_STATE})).hexdigest()


def freeze_path(root: Path, slug: str) -> Path:
    return ba.truth_dir(root, slug) / FREEZE_NAME


def refuse_held_out(slug: str, page: int):
    if (slug, int(page)) in HELD_OUT:
        raise TruthError(f"{slug} p{page} is held out (plan §6.2: hadpi p3-p5 are the final check); "
                         "it must not be frozen, scored or tuned on")


def _excluded_lines(doc: dict, extra: dict | None = None) -> list[dict]:
    out = []
    for li, line in enumerate(doc["lines"]):
        if line.get("text_flag"):
            note = line["text_flag"].get("note", "") if isinstance(line["text_flag"], dict) else str(line["text_flag"])
            out.append({"line": li, "reason": "text_flag", "auto": True,
                        "note_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest()})
    for li, why in sorted((extra or {}).items()):
        if not any(e["line"] == li for e in out):
            out.append({"line": li, "reason": why, "auto": False})
    return sorted(out, key=lambda e: e["line"])


def freeze_page(root: Path, slug: str, page: int, *, force: bool = False, cls: str | None = None,
                extra_excluded: dict | None = None) -> dict:
    refuse_held_out(slug, page)
    ap = ba.audit_path(root, slug, page)
    if not ap.exists():
        raise TruthError(f"no audit file {ap}")
    doc = json.loads(ap.read_text(encoding="utf-8"))
    if doc.get("status") not in ("audited", "previously_audited"):
        raise TruthError(f"{slug} p{page} is not audited (status {doc.get('status')!r}); finish the audit first")
    errs = ba.validate_page(doc)
    if errs:
        raise TruthError(f"{slug} p{page} audit fails the box-audit contract: " + "; ".join(errs[:5]))
    image = root / doc["image"]["path"]
    image_sha = ba.file_sha(image)
    if image_sha != doc["image"]["sha256"]:
        raise TruthError(f"{slug} p{page}: audit image {image} changed since the audit (sha mismatch)")
    provenance = AGENT if doc["status"] == "previously_audited" or doc.get("previous_audit") else OWNER
    cls = cls or SLUG_CLASS.get(slug)
    if not cls:
        raise TruthError(f"no page class known for {slug}; pass --class")
    excluded = _excluded_lines(doc, extra_excluded)
    ex = {e["line"] for e in excluded}
    audit_date = doc.get("audited_at") or (doc.get("previous_audit") or {}).get("audited_at")
    record = {
        "slug": slug, "page": page, "class": cls, "provenance": provenance,
        "audit_file": ba.rel(ap, root), "audit_file_sha256": ba.file_sha(ap),
        "audit_sha256": content_sha(doc), "audit_sha256_scope": "audit JSON without " + "/".join(UI_STATE),
        "audit_status": doc["status"], "audit_date": audit_date, "audit_revision": doc.get("revision"),
        "image": {"path": doc["image"]["path"], "sha256": image_sha,
                  "width_px": doc["image"]["width_px"], "height_px": doc["image"]["height_px"]},
        "text_source_kind": doc["text_source"]["kind"],
        "lines": len(doc["lines"]), "words": sum(len(l["words"]) for l in doc["lines"]),
        "graded_lines": sum(1 for i in range(len(doc["lines"])) if i not in ex),
        "graded_words": sum(len(l["words"]) for i, l in enumerate(doc["lines"]) if i not in ex),
        "excluded_lines": excluded,
        "frozen_at": now_iso(), "freeze_tool": SCHEMA,
    }
    if provenance == AGENT:
        record["note"] = ("agent-audited, not owner-audited: geometry from an earlier independent agent "
                          "audit; never pooled with owner-audited pages")
    fp = freeze_path(root, slug)
    man = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {"schema": FREEZE_SCHEMA, "pages": {}}
    k = f"{page:04d}"
    old = man["pages"].get(k)
    if old and not force:
        same = all(old.get(f) == record.get(f) for f in ("audit_sha256", "excluded_lines", "class"))
        if same and old["image"]["sha256"] == image_sha:
            return old
        raise TruthError(f"{slug} p{page} is already frozen with different content; pass --force to re-freeze")
    man["pages"][k] = record
    ba.write_json_atomic(fp, man)
    return record


def refreeze_hint(slug, page):
    return (f"re-freeze after checking the change: ./venv/bin/python tests/real_truth_eval.py freeze "
            f"--key {slug}/{page} --force")


def load_truth(root: Path, slug: str, page: int) -> TruthPage:
    """A frozen truth page; refuses held-out pages and any drift from the freeze."""
    refuse_held_out(slug, page)
    fp = freeze_path(root, slug)
    man = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {"pages": {}}
    rec = man["pages"].get(f"{page:04d}")
    if not rec:
        raise TruthError(f"{slug} p{page} has no freeze record in {fp}; freeze it first: "
                         f"./venv/bin/python tests/real_truth_eval.py freeze --key {slug}/{page}")
    ap = root / rec["audit_file"]
    if not ap.exists():
        raise TruthError(f"{slug} p{page}: audit file {ap} is missing; refusing to grade")
    doc = json.loads(ap.read_text(encoding="utf-8"))
    if content_sha(doc) != rec["audit_sha256"]:
        raise TruthError(f"{slug} p{page}: the audit file no longer matches its freeze record "
                         f"(frozen {rec['frozen_at']}); refusing to grade. " + refreeze_hint(slug, page))
    image = root / rec["image"]["path"]
    if not image.exists() or ba.file_sha(image) != rec["image"]["sha256"]:
        raise TruthError(f"{slug} p{page}: the audit image no longer matches its freeze record; "
                         "refusing to grade. " + refreeze_hint(slug, page))
    W, H = int(doc["image"]["width_px"]), int(doc["image"]["height_px"])
    ex = {e["line"]: e["reason"] for e in rec["excluded_lines"]}
    lines = []
    for li, L in enumerate(doc["lines"]):
        words = []
        for wi, w in enumerate(L["words"]):
            text = L["tokens"][wi] if wi < len(L["tokens"]) else ""
            words.append(TWord(li, wi, text, [tuple(p) for p in ba.box_quad_px(L["frame"], w["box"])],
                               list(w["box"]), letters(text), li not in ex))
        lines.append(TLine(li, L["frame"], list(L["box"]), [tuple(p) for p in ba.box_quad_px(L["frame"], L["box"])],
                           words, li in ex, ex.get(li)))
    spacing = statistics.median([b.frame["cy"] - a.frame["cy"] for a, b in zip(lines, lines[1:])] or [0.0])
    zones = []
    for L in lines:
        if L.excluded:
            b = L.box
            zones.append([tuple(p) for p in ba.box_quad_px(L.frame, [b[0], b[1] - spacing, b[2], b[3] + spacing])])
    for r in doc.get("regions") or []:
        x0, y0, x1, y1 = r["rect"]
        zones.append([(x0 * W, y0 * H), (x1 * W, y0 * H), (x1 * W, y1 * H), (x0 * W, y1 * H)])
    return TruthPage(doc["slug"], int(doc["page"]), rec["class"], rec["provenance"], W, H, image,
                     rec["image"]["sha256"], ap, rec["audit_sha256"], lines, zones, rec)


def frozen_keys(root: Path) -> list[tuple[str, int]]:
    out = []
    for fp in sorted((root / "books").glob(f"*/truth/{FREEZE_NAME}")):
        man = json.loads(fp.read_text(encoding="utf-8"))
        for k, rec in sorted(man["pages"].items()):
            if (rec["slug"], rec["page"]) not in HELD_OUT:
                out.append((rec["slug"], int(rec["page"])))
    return out


# ---------------------------------------------------------------------------
# Geometry: convex polygons on a pixel grid
# ---------------------------------------------------------------------------
def region_poly(region, W, H) -> list:
    if "quad" in region:
        return [(float(x) * W, float(y) * H) for x, y in region["quad"]]
    x0, y0, x1, y1 = region["rect"]
    return [(x0 * W, y0 * H), (x1 * W, y0 * H), (x1 * W, y1 * H), (x0 * W, y1 * H)]


def bounds(polys):
    xs = [p[0] for q in polys for p in q]
    ys = [p[1] for q in polys for p in q]
    return min(xs), min(ys), max(xs), max(ys)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class Grid:
    """Pixel-centre sampling of a local window; areas are cell counts x cell area."""

    def __init__(self, box, max_cells=400_000):
        x0, y0, x1, y1 = box
        x0, y0 = math.floor(x0), math.floor(y0)
        x1, y1 = math.ceil(x1), math.ceil(y1)
        area = max(1, (x1 - x0) * (y1 - y0))
        self.step = max(1, math.ceil(math.sqrt(area / max_cells)))
        xs = np.arange(x0 + self.step / 2, max(x1, x0 + 1), self.step)
        ys = np.arange(y0 + self.step / 2, max(y1, y0 + 1), self.step)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.cell = float(self.step * self.step)

    def mask(self, poly):
        m = None
        pos = neg = None
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            c = (bx - ax) * (self.Y - ay) - (by - ay) * (self.X - ax)
            pos = (c >= 0) if pos is None else pos & (c >= 0)
            neg = (c <= 0) if neg is None else neg & (c <= 0)
        m = pos | neg
        return m

    def union(self, polys):
        m = np.zeros(self.X.shape, bool)
        for p in polys:
            m |= self.mask(p)
        return m

    def area(self, m):
        return float(m.sum()) * self.cell


def _quad_area(q):
    s = 0.0
    for i in range(len(q)):
        x0, y0 = q[i]
        x1, y1 = q[(i + 1) % len(q)]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2


def fraction_inside(word_quad, polys) -> float:
    """Share of a truth quad's area inside the union of candidate polygons."""
    wb = bounds([word_quad])
    near = [p for p in polys if _overlaps(wb, bounds([p]))]
    if not near:
        return 0.0
    g = Grid(wb)
    wm = g.mask(word_quad)
    tot = wm.sum()
    if not tot:
        return 0.0
    return float((wm & g.union(near)).sum()) / float(tot)


# ---------------------------------------------------------------------------
# Word grading
# ---------------------------------------------------------------------------
def allowance_quad(page: TruthPage, w: TWord):
    L = page.lines[w.li]
    u0, v0, u1, v1 = w.fbox
    lw = (u1 - u0) / max(1.0, w.letters)
    h = v1 - v0
    return [tuple(p) for p in ba.box_quad_px(L.frame, [u0 - lw / 2, v0 - OVERREACH * h, u1 + lw / 2, v1 + OVERREACH * h])]


def grade_target(page: TruthPage, target: list, placed: dict | None) -> dict:
    """Grade one target (list of (li, wi)) against one placement."""
    tw = [page.word(li, wi) for li, wi in target]
    tset = set(target)
    t_lines = sorted({w.li for w in tw})
    row = {"target": [list(t) for t in target], "status": (placed or {}).get("status", "unresolved"),
           "reason": (placed or {}).get("reason", ""), "confidence": (placed or {}).get("confidence")}
    segs = (placed or {}).get("segments") or []
    if not segs:
        row.update(outcome="no_box", shown=False, correct=False, plan_correct=False, wrong_line=False)
        return row
    polys = [region_poly(s, page.W, page.H) for s in segs]
    bb = bounds(polys)
    cov = [fraction_inside(w.quad, polys) for w in tw]
    fracs = {}
    by_line = {}
    for w in page.graded_words():
        if not _overlaps(bb, bounds([w.quad])):
            continue
        f = fraction_inside(w.quad, polys)
        if f > 0:
            fracs[(w.li, w.wi)] = f
            by_line[w.li] = by_line.get(w.li, 0.0) + f * _quad_area(w.quad)
    intr = {k: f * page.word(*k).letters for k, f in fracs.items() if k not in tset}
    # Blank overreach: box area outside the per-target allowance.
    g = Grid(bb)
    bm = g.union(polys)
    am = g.union([allowance_quad(page, w) for w in tw])
    over = g.area(bm & ~am) / max(1e-9, sum(_quad_area(w.quad) for w in tw))
    max_intr = max(intr.values(), default=0.0)
    correct = min(cov) >= COVER and max_intr < INTRUDE_LETTERS and over <= OVERREACH
    # Plan-literal: at most one touched non-target word, adjacent in reading order.
    order = [(w.li, w.wi) for w in page.graded_words()]
    pos = {k: i for i, k in enumerate(order)}
    lo, hi = min(pos[t] for t in target), max(pos[t] for t in target)
    touched = [k for k, f in fracs.items() if k not in tset and f >= MIN_TOUCH]
    plan_correct = min(cov) >= COVER and (not touched or (len(touched) == 1 and pos[touched[0]] in (lo - 1, hi + 1)))
    best_line = max(by_line, key=by_line.get) if by_line else None
    touches_text = any(f >= MIN_TOUCH for f in fracs.values())
    wrong_line = touches_text and best_line not in t_lines
    same_l = max((v for k, v in intr.items() if k[0] in t_lines), default=0.0)
    other_l = max((v for k, v in intr.items() if k[0] not in t_lines), default=0.0)
    row.update(shown=True, correct=bool(correct), plan_correct=bool(plan_correct), wrong_line=bool(wrong_line),
               coverage=[round(c, 4) for c in cov], max_intrusion_letters=round(max_intr, 3),
               intrusion_same_line_letters=round(same_l, 3), intrusion_other_line_letters=round(other_l, 3),
               overreach=round(over, 4), best_line=best_line)
    if len(tw) == 1:
        g2 = Grid(bounds(polys + [tw[0].quad]))
        b2, t2 = g2.union(polys), g2.mask(tw[0].quad)
        row["iou"] = round(float((b2 & t2).sum()) / max(1.0, float((b2 | t2).sum())), 4)
    if correct:
        row["outcome"] = "correct"
    elif not touches_text:
        row["outcome"] = "off_text"
    elif wrong_line:
        row["outcome"] = "wrong_line"
    elif len(tw) == 1:
        t = tw[0]
        same = {k: f for k, f in fracs.items() if k[0] == t.li}
        bk = max(same, key=same.get) if same else None
        if bk is not None and bk != (t.li, t.wi) and same[bk] >= 0.5 and same[bk] > cov[0]:
            row["outcome"] = "shifted"
            row["shift"] = bk[1] - t.wi
        else:
            miss_letters = (1 - cov[0]) * t.letters
            # Boundary error: an edge off by a letter or more, but neither half
            # the target lost nor half a neighbour swallowed (that is too wide).
            nbs = [(intr.get((t.li, t.wi + d), 0.0), fracs.get((t.li, t.wi + d), 0.0)) for d in (-1, 1)]
            nb = max((x for x, f in nbs if f < 0.5), default=0.0)
            if cov[0] >= 0.5 and (miss_letters >= 0.5 or nb >= INTRUDE_LETTERS):
                row["outcome"] = "o2_boundary"
                row["o2"] = {"missing_letters": round(miss_letters, 2), "neighbour_letters": round(nb, 2)}
            elif cov[0] < COVER:
                row["outcome"] = "sub_letter_shortfall" if cov[0] >= 0.5 else "partial"
            elif max_intr >= INTRUDE_LETTERS:
                row["outcome"] = "too_wide_other_word"
            else:
                row["outcome"] = "too_wide_blank"
    else:
        if min(cov) < COVER:
            row["outcome"] = "partial"
        elif max_intr >= INTRUDE_LETTERS:
            row["outcome"] = "too_wide_other_word"
        else:
            row["outcome"] = "too_wide_blank"
    return row


def cascades(page: TruthPage, rows: list[dict]) -> list[dict]:
    """O1: maximal runs of >= 2 consecutive same-line words shifted the same way."""
    by = {tuple(r["target"][0]): r for r in rows if len(r["target"]) == 1}
    out = []
    for L in page.lines:
        if L.excluded:
            continue
        run = []
        def flush():
            if len(run) >= 2:
                out.append({"line": L.index, "first_word": run[0][0], "length": len(run),
                            "direction": run[0][1]})
        for w in L.words:
            r = by.get((w.li, w.wi))
            s = r.get("shift") if r and r.get("outcome") == "shifted" else None
            d = (1 if s > 0 else -1) if s else None
            if d is not None and run and run[-1][1] == d and run[-1][0] == w.wi - 1:
                run.append((w.wi, d))
            else:
                flush()
                run = [(w.wi, d)] if d is not None else []
        flush()
    return out


def summarize_targets(page_rows: list[tuple[TruthPage, list[dict]]], single: bool) -> dict:
    n = sum(len(r) for _, r in page_rows)
    rows = [x for _, r in page_rows for x in r]
    loc = [x for x in rows if x["status"] == "located"]
    out = {
        "n": n,
        "located_correct": rate(sum(x["correct"] for x in loc), n),
        "located_precision": rate(sum(x["correct"] for x in loc), len(loc)),
        "located_plan_literal_correct": rate(sum(x["plan_correct"] for x in loc), n),
        "shown": rate(sum(x["shown"] for x in rows), n),
        "shown_correct": rate(sum(x["correct"] for x in rows), n),
        "shown_precision": rate(sum(x["correct"] for x in rows), sum(x["shown"] for x in rows)),
        "shown_plan_literal_correct": rate(sum(x["plan_correct"] for x in rows), n),
        "shown_covering_only": rate(sum(x["shown"] and min(x["coverage"]) >= COVER for x in rows), n),
        "wrong_line": rate(sum(x["wrong_line"] for x in rows), n),
        "no_box": rate(sum(not x["shown"] for x in rows), n),
        "status_counts": dict(sorted(_count(x["status"] for x in rows).items())),
        "outcome_counts": dict(sorted(_count(x["outcome"] for x in rows).items())),
    }
    if single:
        cs = [c for p, r in page_rows for c in cascades(p, r)]
        out["O1_cascades"] = {"count": len(cs), "words": sum(c["length"] for c in cs),
                              "longest": max((c["length"] for c in cs), default=0)}
        out["O2_boundary"] = rate(sum(x["outcome"] == "o2_boundary" for x in rows), n)
        wide = [x for x in rows if x["outcome"] == "too_wide_other_word"]
        out["too_wide_other_word_by_direction"] = {
            "same_line_only": sum(x["intrusion_same_line_letters"] >= INTRUDE_LETTERS > x["intrusion_other_line_letters"] for x in wide),
            "other_line_only": sum(x["intrusion_other_line_letters"] >= INTRUDE_LETTERS > x["intrusion_same_line_letters"] for x in wide),
            "both": sum(min(x["intrusion_same_line_letters"], x["intrusion_other_line_letters"]) >= INTRUDE_LETTERS for x in wide)}
        ious = [x["iou"] for x in rows if "iou" in x]
        if ious:
            out["iou_median_shown"] = round(statistics.median(ious), 4)
            out["iou_ge_0_8"] = rate(sum(i >= 0.8 for i in ious), n)
        reasons = _count(x["reason"].split(":")[0] if x["reason"] else "" for x in rows if not x["shown"])
        out["no_box_reasons"] = dict(sorted(reasons.items()))
    return out


def _count(it):
    c = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


# ---------------------------------------------------------------------------
# Segmentation grading (lines; text-free word boxes)
# ---------------------------------------------------------------------------
def _ignored(page: TruthPage, poly) -> bool:
    if not page.ignore_zones:
        return False
    g = Grid(bounds([poly]))
    m = g.mask(poly)
    tot = m.sum()
    return bool(tot) and float((m & g.union(page.ignore_zones)).sum()) / float(tot) >= 0.5


def grade_lines(page: TruthPage, regions: list[dict]) -> dict:
    polys = [region_poly(r, page.W, page.H) for r in regions]
    units = [L for L in page.lines if not L.excluded]
    share = np.zeros((len(units), len(polys)))
    union_share = []
    for i, L in enumerate(units):
        tot = sum(_quad_area(w.quad) for w in L.words)
        for j, p in enumerate(polys):
            if _overlaps(bounds([L.quad]), bounds([p])):
                share[i, j] = sum(fraction_inside(w.quad, [p]) * _quad_area(w.quad) for w in L.words) / tot
        near = [p for p in polys if _overlaps(bounds([L.quad]), bounds([p]))]
        union_share.append(sum(fraction_inside(w.quad, near) * _quad_area(w.quad) for w in L.words) / tot if near else 0.0)
    ignored = [_ignored(page, p) for p in polys]
    edges = [[j for j in range(len(polys)) if share[i, j] >= EDGE] for i in range(len(units))]
    touches = [[i for i in range(len(units)) if share[i, j] >= TOUCH] for j in range(len(polys))]
    clean = [i for i, js in enumerate(edges) if len(js) == 1 and touches[js[0]] == [i]]
    merged = [j for j in range(len(polys)) if sum(share[:, j] >= EDGE) >= 2]
    split = [i for i, js in enumerate(edges)
             if len(js) >= 2 or (not js and sum(share[i] >= 0.15) >= 2 and union_share[i] >= 0.5)]
    missed = [i for i in range(len(units)) if union_share[i] < 0.5]
    mx = share.max(axis=0) if len(units) else np.zeros(len(polys))
    spurious = [j for j in range(len(polys)) if mx[j] < 0.05 and not ignored[j]]
    fragment = [j for j in range(len(polys)) if 0.05 <= mx[j] < TOUCH and not ignored[j]]
    counted = sum(1 for j in range(len(polys)) if not ignored[j] or mx[j] >= TOUCH)
    return {
        "truth_lines": len(units), "excluded_lines": sum(L.excluded for L in page.lines),
        "candidates": len(polys), "ignored_candidates": sum(1 for j in range(len(polys)) if ignored[j] and mx[j] < TOUCH),
        "clean_recall": rate(len(clean), len(units)),
        "clean_precision": rate(len(clean), counted),
        "merged_candidates": len(merged),
        "truth_lines_in_merges": int(sum(int(sum(share[:, j] >= EDGE)) for j in merged)),
        "largest_merge": int(max((sum(share[:, j] >= EDGE) for j in merged), default=0)),
        "split_lines": len(split), "missed_lines": len(missed),
        "spurious_candidates": len(spurious), "fragment_candidates": len(fragment),
        "clean_truth_lines": [units[i].index for i in clean],
        "merges": [[units[i].index for i in range(len(units)) if share[i, j] >= EDGE] for j in merged],
    }


def grade_word_segments(page: TruthPage, regions: list[dict]) -> dict:
    polys = [region_poly(r, page.W, page.H) for r in regions]
    units = page.graded_words()
    pb = [bounds([p]) for p in polys]
    clean, split, missed = [], [], []
    hits = {}  # (unit index, poly index) -> fraction
    for i, w in enumerate(units):
        wb = bounds([w.quad])
        near = [j for j in range(len(polys)) if _overlaps(wb, pb[j])]
        fr = {j: fraction_inside(w.quad, [polys[j]]) for j in near}
        fr = {j: f for j, f in fr.items() if f > 0}
        for j, f in fr.items():
            hits[(i, j)] = f
        ok = any(grade_target(page, [(w.li, w.wi)], {"status": "located", "segments": [regions[j]]})["correct"]
                 for j, f in fr.items() if f >= COVER)
        if ok:
            clean.append(i)
        if (fraction_inside(w.quad, [polys[j] for j in near]) if near else 0.0) < 0.5:
            missed.append(i)
        elif max(fr.values(), default=0) < COVER and sum(f >= 0.15 for f in fr.values()) >= 2:
            split.append(i)
    per_poly = {}
    for (i, j), f in hits.items():
        per_poly.setdefault(j, []).append(f)
    merged = [j for j, fs in per_poly.items() if sum(f >= 0.5 for f in fs) >= 2]
    mx = {j: max(fs) for j, fs in per_poly.items()}
    ignored = [_ignored(page, p) for p in polys]
    spurious = [j for j in range(len(polys)) if mx.get(j, 0) < MIN_TOUCH and not ignored[j]]
    counted = sum(1 for j in range(len(polys)) if not ignored[j] or mx.get(j, 0) >= MIN_TOUCH)
    good_polys = set()
    for i in clean:
        w = units[i]
        for j in range(len(polys)):
            if hits.get((i, j), 0) >= COVER and grade_target(page, [(w.li, w.wi)], {"status": "located", "segments": [regions[j]]})["correct"]:
                good_polys.add(j)
    return {"truth_words": len(units), "boxes": len(polys),
            "clean_recall": rate(len(clean), len(units)),
            "clean_precision": rate(len(good_polys), counted),
            "merged_boxes": len(merged),
            "truth_words_in_merges": sum(sum(f >= 0.5 for f in per_poly[j]) for j in merged),
            "split_words": len(split), "missed_words": len(missed), "spurious_boxes": len(spurious)}


# ---------------------------------------------------------------------------
# Adapter inputs: pages and targets
# ---------------------------------------------------------------------------
@dataclass
class Target:
    tid: str
    words: list               # [(li, wi), ...] in reading order
    text: str                 # truth text of the span (space-joined tokens)


@dataclass
class PageInput:
    truth: TruthPage
    root: Path
    pdf_path: Path
    md: str                   # workspace transcription (text/NNNN.md), may be ""
    line_texts: list          # truth tokens per line, all lines (excluded ones too)
    excluded_lines: list
    scratch: Path | None = None

    @property
    def image_path(self):
        return self.truth.image_path


def build_page_input(root: Path, truth: TruthPage, scratch: Path | None) -> PageInput:
    ws = root / "books" / truth.slug
    md_path = ws / "text" / f"{truth.page:04d}.md"
    return PageInput(truth, root, ws / "source.pdf", md_path.read_text(encoding="utf-8") if md_path.exists() else "",
                     [[w.text for w in L.words] for L in truth.lines],
                     [L.index for L in truth.lines if L.excluded], scratch)


def build_targets(truth: TruthPage, seed: int, n_spans: int) -> tuple[list[Target], list[Target]]:
    order = [(w.li, w.wi) for w in truth.graded_words()]
    words = [Target(f"w{li}.{wi}", [(li, wi)], truth.word(li, wi).text) for li, wi in order]
    rng = random.Random(f"{seed}:{truth.slug}:{truth.page}")
    spans, seen = [], set()
    # Spans run in reading order and may wrap a line break, like a review
    # correction; they never include an excluded line's words.
    attempts = 0
    while len(spans) < n_spans and attempts < 50 * max(1, n_spans):
        attempts += 1
        k = rng.randint(2, 6)
        s = rng.randrange(0, max(1, len(order) - k + 1))
        sel = order[s:s + k]
        if len(sel) < 2 or any(b[0] - a[0] > 1 for a, b in zip(sel, sel[1:])):
            continue
        if tuple(sel) in seen:
            continue
        seen.add(tuple(sel))
        spans.append(Target(f"s{len(spans)}", sel, " ".join(truth.word(*t).text for t in sel)))
    return words, spans


# ---------------------------------------------------------------------------
# Line finders
# ---------------------------------------------------------------------------
class TruthLines:
    name = "truth_lines (sanity: audited truth fed back)"

    def find_lines(self, page: PageInput):
        t = page.truth
        return [{"quad": [[x / t.W, y / t.H] for x, y in L.quad]} for L in t.lines if not L.excluded]


class DraftLines:
    name = "draft_lines (box_audit draft, untouched)"

    def find_lines(self, page: PageInput):
        t = page.truth
        d = json.loads(ba.draft_path(page.root, t.slug, t.page).read_text(encoding="utf-8"))
        return [{"quad": L["quad"]} for L in d["lines"]]


class ScanPageLines:
    name = "current detector locate._scan_page_lines"

    def find_lines(self, page: PageInput):
        import fitz
        from farsi2epub import locate
        with fitz.open(str(page.pdf_path)) as doc:
            pg = doc[page.truth.page - 1]
            out = []
            for line in locate._scan_page_lines(pg):
                r = locate._rect_to_fracs(line.rect, pg.rect)
                out.append({"rect": [r["x0"], r["y0"], r["x1"], r["y1"]]})
            return out

    def word_regions(self, page: PageInput):
        import fitz
        from farsi2epub import locate
        with fitz.open(str(page.pdf_path)) as doc:
            pg = doc[page.truth.page - 1]
            out = []
            for line in locate._scan_page_lines(pg):
                for w in line.words:
                    r = locate._rect_to_fracs(w, pg.rect)
                    out.append({"rect": [r["x0"], r["y0"], r["x1"], r["y1"]]})
            return out


LINE_FINDERS = {"truth": TruthLines, "draft": DraftLines, "scan_page_lines": ScanPageLines}


# ---------------------------------------------------------------------------
# Word placers
# ---------------------------------------------------------------------------
def _seg_from_quad(quad, W, H):
    return {"quad": [[x / W, y / H] for x, y in quad]}


class TruthWords:
    name = "truth_words (sanity: audited truth fed back)"

    def place(self, page: PageInput, targets):
        t = page.truth
        return [{"status": "located", "segments": [_seg_from_quad(t.word(*k).quad, t.W, t.H) for k in tg.words],
                 "confidence": None, "reason": ""} for tg in targets]


class DraftWords:
    """The audit tool's untouched draft, mapped to audit words by global token alignment."""
    name = "draft_words (box_audit draft, untouched)"

    def place(self, page: PageInput, targets):
        t = page.truth
        d = json.loads(ba.draft_path(page.root, t.slug, t.page).read_text(encoding="utf-8"))
        dw = [(L, w) for L in d["lines"] for w in L["words"]]
        dtok = [w["text"] or "" for _, w in dw]
        aw = [(w.li, w.wi) for L in t.lines for w in L.words]
        atok = [t.word(*k).text for k in aw]
        m = {}
        sm = difflib.SequenceMatcher(None, atok, dtok, autojunk=False)
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
                for i in range(a1 - a0):
                    m[aw[a0 + i]] = dw[b0 + i][1]["quad"]
        out = []
        for tg in targets:
            if all(k in m for k in tg.words):
                out.append({"status": "located", "segments": [{"quad": m[k]} for k in tg.words],
                            "confidence": None, "reason": ""})
            else:
                out.append({"status": "unresolved", "segments": [], "reason": "no_draft_word"})
        return out


def _md_tokens(md: str):
    return [(m.start(), m.end(), _fold(m.group())) for m in re.finditer(r"\S+", md) if _fold(m.group())]


def map_truth_to_md(page: PageInput, md: str) -> dict:
    """Truth word -> (start, end) span of its aligned token in ``md``.

    Folded-token alignment (difflib). Equal runs map 1:1; replaced runs of equal
    length map position by position (a misread word is still the word the UI
    would ask for). Everything else is not queryable."""
    t = page.truth
    aw = [(w.li, w.wi) for L in t.lines for w in L.words]
    afold = [_fold(t.word(*k).text) for k in aw]
    keep = [i for i, f in enumerate(afold) if f]
    toks = _md_tokens(md)
    sm = difflib.SequenceMatcher(None, [afold[i] for i in keep], [x[2] for x in toks], autojunk=False)
    out = {}
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            for i in range(a1 - a0):
                out[aw[keep[a0 + i]]] = (toks[b0 + i][0], toks[b0 + i][1], tag == "equal")
    return out


def _scratch_workspace(page: PageInput) -> Path:
    """Copy what PlacementService reads into a scratch books root (never books/)."""
    base = page.scratch or Path(tempfile.mkdtemp(prefix="real_truth_ws_"))
    dst = base / page.truth.slug
    if not dst.exists():
        src = page.root / "books" / page.truth.slug
        dst.mkdir(parents=True)
        for name in ("source.pdf", "book.yaml"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        for name in ("text", "locate_evidence"):
            if (src / name).exists():
                shutil.copytree(src / name, dst / name)
    return base


def _no_reader(*_a, **_k):
    raise RuntimeError("real_truth_eval: the reader is disabled ($0 evaluation)")


class _EngineBase:
    md_source = "workspace"     # "workspace" (text/NNNN.md) or "truth" (truth lines)

    def _md(self, page):
        if self.md_source == "truth":
            return "\n".join(" ".join(toks) for toks in page.line_texts)
        return page.md

    def _queries(self, page, targets, md):
        from farsi2epub.locate import Query
        m = map_truth_to_md(page, md)
        qs = []
        for tg in targets:
            a, b = m.get(tg.words[0]), m.get(tg.words[-1])
            if a is None or b is None or b[1] <= a[0]:
                qs.append(None)
            else:
                qs.append(Query(md[a[0]:b[1]], (a[0], b[1]), (), "replacement"))
        return qs

    def _service(self, page):
        os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
        from farsi2epub.scan import PlacementService
        from farsi2epub.workspace import Workspace
        books = _scratch_workspace(page)
        return PlacementService(Workspace(page.truth.slug, books_root=books), mode="offline", reader=_no_reader)

    @staticmethod
    def _to_placed(r):
        if r is None:
            return {"status": "unresolved", "segments": [], "reason": "not_in_transcription"}
        st = r.status if r.status in ("located", "buffered", "estimated", "unresolved") else "unresolved"
        box = r.box or {}
        segs = box.get("segments") or ([box] if box.get("x0") is not None else [])
        return {"status": st, "reason": r.reason or (r.status if st != r.status else ""),
                "segments": [{"rect": [s["x0"], s["y0"], s["x1"], s["y1"]]} for s in segs] if r.box else [],
                "confidence": None}


class EngineOffline(_EngineBase):
    """The live PlacementService, offline, on evidence already on disk."""

    def __init__(self, md_source="workspace"):
        self.md_source = md_source
        self.name = f"live engine offline (PlacementService.replay, md={md_source})"

    def place(self, page, targets):
        md = self._md(page)
        qs = self._queries(page, targets, md)
        svc = self._service(page)
        try:
            live = [q for q in qs if q is not None]
            n = page.truth.page
            svc.replay(n, md, live, None)
            res = iter(svc.results(n, md, live))
            return [self._to_placed(next(res) if q is not None else None) for q in qs]
        finally:
            svc.close()


class EstimatedHistorical(_EngineBase):
    """What the pre-Step-0 refine() drew: best_attempt with no no-evidence gate.

    Calls geometry_attempt.best_attempt directly on today's offline page map;
    production code is unchanged. Historical, not a candidate."""

    def __init__(self, md_source="workspace"):
        self.md_source = md_source
        self.name = f"historical pre-Step-0 estimated path (best_attempt, md={md_source})"

    def place(self, page, targets):
        from farsi2epub.geometry_attempt import PageProjection, best_attempt
        from farsi2epub.page_map import place
        from farsi2epub.scan import _read_json
        md = self._md(page)
        qs = self._queries(page, targets, md)
        svc = self._service(page)
        try:
            n = page.truth.page
            words, summary = svc.ensure_page_map(n, allow_api=False)
            cached = _read_json(svc.root / "pages" / f"{svc._page_key(n)}.json")
            proj = PageProjection(md, words, cached.get("regions", []))
            out = []
            for q in qs:
                if q is None:
                    out.append(self._to_placed(None))
                    continue
                strict = place(md, q, words, source=summary.get("source", "scan"))
                out.append(self._to_placed(best_attempt(md, q, words, proj, strict)))
            return out
        finally:
            svc.close()


WORD_PLACERS = {"truth": TruthWords, "draft": DraftWords,
                "engine": lambda: EngineOffline("workspace"),
                "engine_truth_text": lambda: EngineOffline("truth"),
                "historical_estimated": lambda: EstimatedHistorical("workspace"),
                "historical_estimated_truth_text": lambda: EstimatedHistorical("truth")}


# ---------------------------------------------------------------------------
# Truth sanity checks
# ---------------------------------------------------------------------------
def check_truth(truth: TruthPage) -> dict:
    issues = []
    cys = [L.frame["cy"] for L in truth.lines]
    for i, (a, b) in enumerate(zip(cys, cys[1:])):
        if b <= a:
            issues.append({"kind": "line_order", "line": i + 1})
    lw_all = []
    for L in truth.lines:
        for a, b in zip(L.words, L.words[1:]):
            # RTL reading order: each next word lies left of (lower u than) the previous.
            ca, cb = (a.fbox[0] + a.fbox[2]) / 2, (b.fbox[0] + b.fbox[2]) / 2
            if cb >= ca:
                issues.append({"kind": "word_order", "line": L.index, "word": b.wi})
            ov = min(a.fbox[2], b.fbox[2]) - max(a.fbox[0], b.fbox[0])
            if ov > 0:
                frac = ov / max(1e-9, min(a.fbox[2] - a.fbox[0], b.fbox[2] - b.fbox[0]))
                if frac > 0.10:
                    issues.append({"kind": "word_overlap", "line": L.index, "word": b.wi,
                                   "overlap_of_narrower": round(frac, 3)})
        for w in L.words:
            lw_all.append((w.fbox[2] - w.fbox[0]) / max(1, w.letters))
    med = statistics.median(lw_all) if lw_all else 0
    for L in truth.lines:
        for w in L.words:
            lw = (w.fbox[2] - w.fbox[0]) / max(1, w.letters)
            if med and (lw > 3 * med or lw < med / 3):
                issues.append({"kind": "letter_width_outlier", "line": L.index, "word": w.wi,
                               "px_per_letter": round(lw, 1), "page_median": round(med, 1)})
    # Words overlapping words of other lines.
    for La, Lb in zip(truth.lines, truth.lines[1:]):
        for a in La.words:
            for b in Lb.words:
                if _overlaps(bounds([a.quad]), bounds([b.quad])):
                    f = fraction_inside(a.quad, [b.quad])
                    if f > 0.05:
                        issues.append({"kind": "cross_line_overlap", "line": La.index, "word": a.wi,
                                       "other_line": Lb.index, "other_word": b.wi, "fraction": round(f, 3)})
    angles = [L.frame["angle_deg"] for L in truth.lines]
    for i, (a, b) in enumerate(zip(angles, angles[1:])):
        if abs(b - a) > 1.0:
            issues.append({"kind": "angle_jump", "line": i + 1, "from_deg": round(a, 2), "to_deg": round(b, 2)})
    return {"key": truth.key, "lines": len(truth.lines), "words": sum(len(L.words) for L in truth.lines),
            "angle_deg_range": [round(min(angles), 2), round(max(angles), 2)] if angles else None,
            "px_per_letter_median": round(med, 1), "issues": issues,
            "issue_counts": dict(sorted(_count(i["kind"] for i in issues).items()))}


# ---------------------------------------------------------------------------
# Running everything
# ---------------------------------------------------------------------------
def group_of(t: TruthPage) -> str:
    return f"{t.provenance}/{t.cls}"


def evaluate(root: Path, keys, *, line_finders=None, word_placers=None, seed=7, n_spans=60,
             scratch: Path | None = None, segmenters=("scan_page_lines",)) -> dict:
    line_finders = line_finders if line_finders is not None else list(LINE_FINDERS)
    word_placers = word_placers if word_placers is not None else list(WORD_PLACERS)
    pages = []
    for slug, page in keys:
        truth = load_truth(root, slug, page)
        pages.append((truth, build_page_input(root, truth, scratch)))
    out = {"schema": SCHEMA, "created_at": now_iso(), "seed": seed, "spans_per_page": n_spans,
           "definitions": DEFINITIONS,
           "pages": [{"key": t.key, "class": t.cls, "provenance": t.provenance, "group": group_of(t),
                      "audit_sha256": t.audit_sha256, "image_sha256": t.image_sha256,
                      "frozen_at": t.freeze.get("frozen_at"), "audit_date": t.freeze.get("audit_date"),
                      "graded_lines": sum(not L.excluded for L in t.lines),
                      "graded_words": len(t.graded_words()),
                      "excluded_lines": t.freeze.get("excluded_lines")} for t, _ in pages],
           "lines": {}, "word_segments": {}, "words": {}, "spans": {}, "rows": {}, "query_mapping": {}}
    for name in line_finders:
        f = LINE_FINDERS[name]()
        per = {}
        for t, pi in pages:
            try:
                per[t.key] = grade_lines(t, f.find_lines(pi))
            except FileNotFoundError as e:
                per[t.key] = {"error": str(e)}
        out["lines"][name] = {"adapter": f.name, "pages": per, "groups": _line_groups(pages, per)}
    for name in segmenters:
        f = LINE_FINDERS[name]()
        per = {t.key: grade_word_segments(t, f.word_regions(pi)) for t, pi in pages}
        out["word_segments"][name] = {"adapter": f.name + " word-gap runs", "pages": per,
                                      "groups": _seg_groups(pages, per)}
    for name in word_placers:
        p = WORD_PLACERS[name]()
        wrows, srows = {}, {}
        for t, pi in pages:
            words, spans = build_targets(t, seed, n_spans)
            res = p.place(pi, words + spans)
            wrows[t.key] = [dict(grade_target(t, tg.words, r), tid=tg.tid) for tg, r in zip(words, res[:len(words)])]
            srows[t.key] = [dict(grade_target(t, tg.words, r), tid=tg.tid) for tg, r in zip(spans, res[len(words):])]
            if isinstance(p, _EngineBase):
                md = p._md(pi)
                m = map_truth_to_md(pi, md)
                gw = [(w.li, w.wi) for w in t.graded_words()]
                out["query_mapping"].setdefault(name, {})[t.key] = {
                    "graded_words": len(gw), "queryable": sum(k in m for k in gw),
                    "exact_text": sum(1 for k in gw if k in m and m[k][2])}
        tbl = {}
        for kind, rows in (("words", wrows), ("spans", srows)):
            per = {t.key: summarize_targets([(t, rows[t.key])], kind == "words") for t, _ in pages}
            groups = {}
            for g in sorted({group_of(t) for t, _ in pages}):
                groups[g] = summarize_targets([(t, rows[t.key]) for t, _ in pages if group_of(t) == g], kind == "words")
                groups[g]["pages"] = [t.key for t, _ in pages if group_of(t) == g]
                groups[g]["page_rates_located_correct"] = [per[t.key]["located_correct"]["rate"] for t, _ in pages if group_of(t) == g]
            out[kind][name] = {"adapter": p.name, "pages": per, "groups": groups}
            if kind == "words":
                for t, _ in pages:
                    per[t.key]["O1_cascade_list"] = cascades(t, rows[t.key])
        out["rows"][name] = {"words": wrows, "spans": srows}
    return out


def _line_groups(pages, per):
    groups = {}
    for g in sorted({group_of(t) for t, _ in pages}):
        ks = [t.key for t, _ in pages if group_of(t) == g and "error" not in per[t.key]]
        tl = sum(per[k]["truth_lines"] for k in ks)
        clean = sum(per[k]["clean_recall"]["k"] for k in ks)
        counted = sum(per[k]["clean_precision"]["n"] for k in ks)
        groups[g] = {"pages": ks, "clean_recall": rate(clean, tl), "clean_precision": rate(clean, counted),
                     **{f: sum(per[k][f] for k in ks) for f in ("merged_candidates", "truth_lines_in_merges",
                                                               "split_lines", "missed_lines", "spurious_candidates",
                                                               "fragment_candidates", "candidates")}}
    return groups


def _seg_groups(pages, per):
    groups = {}
    for g in sorted({group_of(t) for t, _ in pages}):
        ks = [t.key for t, _ in pages if group_of(t) == g]
        groups[g] = {"pages": ks,
                     "clean_recall": rate(sum(per[k]["clean_recall"]["k"] for k in ks), sum(per[k]["truth_words"] for k in ks)),
                     "clean_precision": rate(sum(per[k]["clean_precision"]["k"] for k in ks), sum(per[k]["clean_precision"]["n"] for k in ks)),
                     **{f: sum(per[k][f] for k in ks) for f in ("merged_boxes", "truth_words_in_merges", "split_words",
                                                               "missed_words", "spurious_boxes", "boxes")}}
    return groups


def _fmt(r):
    if not r or r.get("n") in (None, 0):
        return "  -  (0/0)"
    ci = r["wilson95"]
    return f"{r['k']}/{r['n']} = {100 * r['rate']:.1f}% [{100 * ci[0]:.0f}-{100 * ci[1]:.0f}]"


def table(report: dict) -> str:
    out = ["Real-page truth evaluation  (k/n = rate [Wilson 95%; words treated as independent, optimistic])",
           f"schema {report['schema']}  seed {report['seed']}  created {report['created_at']}", ""]
    out.append("Pages:")
    for p in report["pages"]:
        ex = ", ".join(f"line {e['line']} ({e['reason']})" for e in p["excluded_lines"]) or "none"
        out.append(f"  {p['key']:<22} {p['group']:<32} lines {p['graded_lines']:>3}  words {p['graded_words']:>4}  excluded: {ex}")
    out.append("")
    for name, blk in report["lines"].items():
        out.append(f"LINES  {blk['adapter']}")
        for k, r in list(blk["pages"].items()) + [(f"[{g}]", r) for g, r in blk["groups"].items()]:
            if "error" in r:
                out.append(f"  {k:<36} error: {r['error']}")
                continue
            out.append(f"  {k:<36} recall {_fmt(r['clean_recall'])}  prec {_fmt(r['clean_precision'])}  "
                       f"cand {r['candidates']}  merged {r['merged_candidates']} (lines in merges {r['truth_lines_in_merges']})  "
                       f"split {r['split_lines']}  missed {r['missed_lines']}  spurious {r['spurious_candidates']}  frag {r['fragment_candidates']}")
        out.append("")
    for name, blk in report["word_segments"].items():
        out.append(f"WORD SEGMENTS (text-free)  {blk['adapter']}")
        for k, r in list(blk["pages"].items()) + [(f"[{g}]", r) for g, r in blk["groups"].items()]:
            out.append(f"  {k:<36} clean {_fmt(r['clean_recall'])}  prec {_fmt(r['clean_precision'])}  boxes {r['boxes']}  "
                       f"merged {r['merged_boxes']} (words {r['truth_words_in_merges']})  split {r['split_words']}  "
                       f"missed {r['missed_words']}  spurious {r['spurious_boxes']}")
        out.append("")
    for kind in ("words", "spans"):
        for name, blk in report[kind].items():
            out.append(f"{kind.upper()}  {blk['adapter']}")
            for k, r in list(blk["pages"].items()) + [(f"[{g}]", r) for g, r in blk["groups"].items()]:
                extra = ""
                if kind == "words":
                    tw = r["too_wide_other_word_by_direction"]
                    extra = (f"  O1 {r['O1_cascades']['count']} ({r['O1_cascades']['words']} w)  O2 {r['O2_boundary']['k']}"
                             f"  too-wide same/other/both {tw['same_line_only']}/{tw['other_line_only']}/{tw['both']}")
                out.append(f"  {k:<36} located&correct {_fmt(r['located_correct'])}  prec(located) {_fmt(r['located_precision'])}  "
                           f"shown&correct {r['shown_correct']['k']} (plan-literal {r['shown_plan_literal_correct']['k']}, covering-only {r['shown_covering_only']['k']})  wrong-line {r['wrong_line']['k']}  no-box {r['no_box']['k']}{extra}")
                out.append(f"  {'':<36} status {r['status_counts']}")
            out.append("")
    if report.get("query_mapping"):
        out.append("Query mapping (truth word -> md token; unmapped words cannot be asked for):")
        for name, per in report["query_mapping"].items():
            for k, m in per.items():
                out.append(f"  {name:<34} {k:<22} queryable {m['queryable']}/{m['graded_words']}  exact text {m['exact_text']}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="record audited pages in books/<slug>/truth/truth_freeze.json")
    f.add_argument("--key", action="append", help="slug/page; default: every audited page found")
    f.add_argument("--force", action="store_true", help="re-freeze a page whose audit changed")
    f.add_argument("--class", dest="cls", help="page class for slugs without a default")
    sub.add_parser("check", help="truth sanity report for every frozen page")
    r = sub.add_parser("run", help="score every adapter on every frozen page")
    r.add_argument("--out", type=Path, default=ROOT / "out" / "real_truth")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--spans", type=int, default=60)
    r.add_argument("--scratch", type=Path, help="scratch books root for the offline engine copy")
    r.add_argument("--placers", nargs="*", help=f"subset of {sorted(WORD_PLACERS)}")
    r.add_argument("--lines", nargs="*", help=f"subset of {sorted(LINE_FINDERS)}")
    a = ap.parse_args(argv)
    root = a.root
    try:
        if a.cmd == "freeze":
            keys = [ba._parse_key(k) for k in a.key] if a.key else [
                k for k in ba.discover(root) if k not in HELD_OUT and ba.audit_path(root, *k).exists()]
            for slug, page in keys:
                rec = freeze_page(root, slug, page, force=a.force, cls=a.cls)
                ex = "; ".join(f"line {e['line']} excluded ({e['reason']}{', automatic' if e['auto'] else ''})"
                               for e in rec["excluded_lines"]) or "no lines excluded"
                print(f"frozen {slug} p{page}: {rec['provenance']} / {rec['class']}, {rec['graded_lines']}/{rec['lines']} "
                      f"lines and {rec['graded_words']}/{rec['words']} words graded; {ex}; audit sha {rec['audit_sha256'][:12]}")
            print("Lines with an owner text flag are excluded automatically.")
        elif a.cmd == "check":
            for slug, page in frozen_keys(root):
                rep = check_truth(load_truth(root, slug, page))
                print(json.dumps({k: rep[k] for k in ("key", "lines", "words", "angle_deg_range", "px_per_letter_median", "issue_counts")}))
                for i in rep["issues"]:
                    print("   ", json.dumps(i))
        else:
            keys = frozen_keys(root)
            if not keys:
                raise TruthError("no frozen pages; run freeze first")
            # A fresh copy per run: derived caches from one run never feed the next.
            base = a.scratch or Path(tempfile.gettempdir())
            base.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix="real_truth_ws_", dir=base))
            rep = evaluate(root, keys, line_finders=a.lines, word_placers=a.placers, seed=a.seed,
                           n_spans=a.spans, scratch=scratch)
            rep["scratch_books_root"] = str(scratch)
            rep["truth_checks"] = [check_truth(load_truth(root, s, p)) for s, p in keys]
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "results.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            txt = table(rep)
            (a.out / "summary.txt").write_text(txt, encoding="utf-8")
            print(txt)
            print(f"wrote {a.out / 'results.json'} and {a.out / 'summary.txt'}")
    except TruthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
