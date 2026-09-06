"""Correction-independent page evidence and content-anchored placement.

No benchmark/truth files are read here. Word geometry is supported by a second
reading of physical crop groups, not assigned by character widths or counts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Optional
import re

import fitz

from . import locate

DERIVATION_VERSION = 4


@dataclass
class PrintedWord:
    text: str
    rect: list[float]
    line: int
    evidence: list[str] = field(default_factory=list)
    supported: bool = True
    group: str = ""


@dataclass
class PlacementResult:
    status: str
    box: Optional[dict] = None
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    buffer_before: int = 0
    buffer_after: int = 0
    kind: str = "words"

    def to_dict(self) -> dict:
        return asdict(self)


def _similar(a: list[str], b: list[str], fragments: bool = False) -> float:
    aa, bb = "".join(a), "".join(b)
    if not aa or not bb:
        return 0.0
    score = SequenceMatcher(None, aa, bb, autojunk=False).ratio()
    # A Unicode PDF can emit reversed glyph fragments. Never apply this to
    # the image reader, whose logical word order is part of its contract.
    return max(score, locate._wsim(aa, bb)) if fragments else score


def _context(md: str, q: locate.Query) -> tuple[list[str], list[str]]:
    if q.span is None or not (0 <= q.span[0] <= q.span[1] <= len(md)):
        return [], []
    before = locate._norm_words(md[:q.span[0]])[-8:]
    after = locate._norm_words(md[q.span[1]:])[:8]
    return before, after


def _anchor_score(anchor: list[str], reader: list[str], at: int,
                  before: bool, fragments: bool) -> float:
    if not anchor:
        return 1.0
    scores = []
    for length in range(max(1, len(anchor) - 3), len(anchor) + 4):
        part = reader[max(0, at-length):at] if before else reader[at:at+length]
        scores.append(_similar(anchor, part, fragments))
    return max(scores, default=0.0)


def target_span(md: str, q: locate.Query, reader: list[str], *,
                fragments: bool = False) -> Optional[tuple[int, int]]:
    """Resolve occurrence by text and exact Markdown anchors, never geometry."""
    if q.kind == "insertion" and insertion_slot(md, q, reader) is not None:
        return None
    before, after = _context(md, q)
    variants = [locate._norm_words(q.text)]
    variants += [locate._norm_words(t) for t in q.alts]
    variants = [v for v in variants if v]
    if not variants:
        return None
    candidates: dict[tuple[int, int], float] = {}
    has_context = len(before) + len(after) >= 2
    for variant in variants:
        n = len(variant)
        max_len = min(len(reader), n * 3 + 6 if fragments else n + max(3, n // 3))
        for start in range(len(reader)):
            left = _anchor_score(before, reader, start, True, fragments)
            if has_context and before and left < 0.72:
                continue
            for length in range(max(1, n - max(2, n // 3)), max_len + 1):
                end = start + length
                if end > len(reader):
                    break
                if any(not word for word in reader[start:end]):
                    continue  # unreadable regions cannot bridge a target
                target = _similar(variant, reader[start:end], fragments)
                if target < (0.40 if has_context else 0.94):
                    continue
                right = _anchor_score(after, reader, end, False, fragments)
                anchors = [v for a, v in ((before, left), (after, right)) if a]
                context = sum(anchors) / len(anchors) if anchors else 0.0
                if has_context and (context < 0.86 or min(anchors) < 0.72):
                    continue
                score = 0.85 * context + 0.15 * target if has_context else target
                candidates[start, end] = max(score, candidates.get((start, end), 0))
    if not candidates:
        return None
    ranked = sorted(candidates.items(), key=lambda v: (-v[1], v[0][1]-v[0][0], v[0][0]))
    (lo, hi), score = ranked[0]
    # Contained windows can represent split/fused token boundaries. Distinct
    # occurrences need a margin even when their repeated words overlap.
    if any(s >= score - 0.04
           and not (a <= lo and hi <= b)
           and not (lo <= a and b <= hi)
           for (a, b), s in ranked[1:]):
        return None
    return lo, hi


def supported_groups(line_text: str, groups: list[tuple[str, list[float], str]],
                     line_id: int, line_evidence: str) -> list[PrintedWord]:
    """Pair independently read crop groups to a line's text without width guesses.

    Split/fused boundaries are reconciled by exact concatenated text. All
    physical groups supporting a token contribute to its envelope. A group
    containing several words deliberately shares geometry; box construction
    accounts for those neighboring words and enforces the buffer limit.
    """
    tokens = locate._norm_words(line_text)
    texts = ["".join(locate._norm_words(g[0])) for g in groups]
    if not tokens or any(not t for t in texts) or "".join(tokens) != "".join(texts):
        return []
    offsets, cursor = [], 0
    for text, rect, key in groups:
        length = len("".join(locate._norm_words(text)))
        offsets.append((cursor, cursor+length, rect, key))
        cursor += length
    out, cursor = [], 0
    for token in tokens:
        end = cursor + len(token)
        hits = [(r, k) for a, b, r, k in offsets if a < end and b > cursor]
        rect = locate._union_rects([fitz.Rect(r) for r, _ in hits])
        keys = sorted({k for _, k in hits})
        out.append(PrintedWord(token, list(rect), line_id,
                               [line_evidence, *keys], True, ":".join(keys)))
        cursor = end
    return out


def supported_windows(line_text, groups, line_id, line_evidence):
    """Certify words by unique exact readings of overlapping physical crops.

    Crop widths never assign identities. Multiword crops share geometry and
    pay the normal neighboring-word buffer cost.
    """
    tokens = locate._norm_words(line_text)
    joined = "".join(tokens)
    offsets = [0]
    for token in tokens:
        offsets.append(offsets[-1]+len(token))
    candidates = {}
    for text, rect, key in groups:
        reading = "".join(locate._norm_words(text))
        if not reading:
            continue
        starts = [i for i,a in enumerate(offsets[:-1])
                  if joined.startswith(reading,a) and a+len(reading) in offsets]
        if len(starts) != 1:
            continue
        lo = starts[0]; hi = offsets.index(offsets[lo]+len(reading))
        for i in range(lo,hi):
            item = PrintedWord(tokens[i],list(rect),line_id,[line_evidence,key],True,key)
            if i not in candidates or fitz.Rect(rect).get_area() < fitz.Rect(candidates[i].rect).get_area():
                candidates[i] = item
    return [candidates.get(i,PrintedWord(token,[0,0,0,0],line_id,[line_evidence],False))
            for i,token in enumerate(tokens)]


def insertion_slot(md, q, reader):
    before, after = _context(md, q)
    if q.kind != "insertion" or not before or not after:
        return None
    slots = [i for i in range(1, len(reader))
             if _anchor_score(before, reader, i, True, False) >= .95
             and _anchor_score(after, reader, i, False, False) >= .95]
    return slots[0] if len(slots) == 1 else None


def place(md: str, q: locate.Query, words: list[PrintedWord], *,
          source: str = "scan") -> PlacementResult:
    reader = [locate._fold_word(w.text) for w in words]
    span = target_span(md, q, reader, fragments=source == "match")
    if span is None:
        slot = insertion_slot(md, q, reader)
        if slot is not None and all(w.supported for w in words[slot-1:slot+1]):
            a, b = words[slot-1:slot+1]
            if a.line == b.line:
                x = (a.rect[0] + b.rect[2]) / 2
                bounds = [(x, min(a.rect[1], b.rect[1]), max(a.rect[3], b.rect[3]))]
            else:
                bounds = [(a.rect[0], a.rect[1], a.rect[3]), (b.rect[2], b.rect[1], b.rect[3])]
            segments = [{"x0": max(0., x-.001), "y0": y0,
                         "x1": min(1., x+.001), "y1": y1} for x, y0, y1 in bounds]
            envelope = {"x0": min(r["x0"] for r in segments), "y0": min(r["y0"] for r in segments),
                        "x1": max(r["x1"] for r in segments), "y1": max(r["y1"] for r in segments)}
            box = {**envelope, "source": source, "segments": segments,
                   "kind": "insertion_boundary", "status": "located"}
            return PlacementResult("located", box, evidence=a.evidence+b.evidence,
                                   kind="insertion_boundary")
        return PlacementResult("unresolved", reason="target_absent_or_ambiguous")
    lo, hi = span
    required = words[lo:hi]
    if not required or not all(w.supported for w in required):
        return PlacementResult("unresolved", reason="word_geometry_unverified")
    by_line: dict[int, list[fitz.Rect]] = {}
    for word in required:
        by_line.setdefault(word.line, []).append(fitz.Rect(word.rect))
    line_ids = sorted(by_line)
    if line_ids != list(range(line_ids[0], line_ids[-1]+1)):
        return PlacementResult("unresolved", reason="noncontiguous_lines")
    rects = [locate._union_rects(by_line[i]) for i in line_ids]
    covered = set()
    for i, word in enumerate(words):
        if not word.supported:
            continue
        r = fitz.Rect(word.rect)
        if any((r & b).get_area() > 1e-12 for b in rects):
            covered.add(i)
    extras = covered - set(range(lo, hi))
    allowed = set(range(max(0, lo-2), lo)) | set(range(hi, min(len(words), hi+2)))
    if not extras <= allowed:
        return PlacementResult("unresolved", reason="buffer_exceeds_two_words")
    before, after = sum(i < lo for i in extras), sum(i >= hi for i in extras)
    env = locate._union_rects(rects)
    def boxdict(r):
        return dict(zip(("x0", "y0", "x1", "y1"), list(r)))
    box = {**boxdict(env), "source": source, "segments": [boxdict(r) for r in rects],
           "status": "buffered" if extras else "located",
           "buffer_before": before, "buffer_after": after}
    evidence = sorted({e for w in required for e in w.evidence})
    return PlacementResult(box["status"], box, evidence=evidence,
                           buffer_before=before, buffer_after=after)
