"""Step 2 of docs/bbox-old-fonts-plan.md: transcription experiments.

Evaluation tooling only; nothing in farsi2epub/ imports this. It runs a fixed
set of transcription "arms" on a few truth pages, caches every API response
on disk, keeps a spend ledger with a hard cap, and scores everything
deterministically (no model ever grades).

    ./venv/bin/python tests/transcription_experiments.py plan     # call plan + cost estimate, $0
    ./venv/bin/python tests/transcription_experiments.py smoke    # one tiny paid call
    ./venv/bin/python tests/transcription_experiments.py run ARM [ARM ...] [--pages ...] [--rep N]
    ./venv/bin/python tests/transcription_experiments.py score    # offline, from the cache, $0

Everything written goes to out/transcription_experiments/ (gitignored),
because outputs contain page text. Nothing is written under books/.

Folding rules (applied to truth and output alike, after Markdown syntax is
stripped and Unicode is NFC-normalised):

  strict  ("as printed"): Arabic yeh/alef-maqsura -> Persian yeh, Arabic kaf
          -> Persian kaf, Arabic-Indic digits -> Persian digits, tatweel and
          invisible controls (ZWSP, ZWJ, LRM, RLM, BOM) removed, whitespace
          runs collapsed. Tokens are whitespace-separated, punctuation stays
          attached, ZWNJ stays inside the token. Word joins count.
  ortho   strict + ZWNJ deleted, diacritics (U+064B-065F, U+0670) and
          heh-with-hamza (U+06C0 -> U+0647) folded, punctuation deleted
          (tokens left empty are dropped). Word joins still count.
  lenient ortho + join-insensitive alignment: one output token may equal the
          concatenation of 2-3 consecutive truth tokens (typewriter join) or
          the reverse (split), at zero cost.

WER = edit cost / number of truth tokens under that folding. A truth token
`⟦؟⟧` (unreadable to the owner) is a wildcard: matched or skipped for free and
not counted in the denominator. Page-number-only lines (digits and dashes)
are removed from both sides before scoring, since the pipeline omits them by
design.

"True errors" are the edit operations of the lenient alignment. Invented
words: output tokens that the lenient alignment leaves unmatched (insertion)
or substitutes, whose character similarity (1 - Levenshtein/max length, on
ortho strings) is < 0.5 to every truth token within 3 truth positions of the
aligned location. `⟦؟⟧` markers are never counted as invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from farsi2epub import config, llm  # noqa: E402

OUT = ROOT / "out" / "transcription_experiments"
CAP_USD = 3.00
PLAN_LIMIT_USD = 2.50

# ---------------------------------------------------------------------------
# Folding and tokenisation
# ---------------------------------------------------------------------------

ZWNJ = "‌"
MARK_UNK = "⟦؟⟧"
_MARKER_RE = re.compile(r"⟦([^⟧]*)⟧")
_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک",
    **{chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)},
    "ـ": None, "​": None, "‍": None, "‎": None,
    "‏": None, "﻿": None, " ": " ",
})
_DIACRITICS = re.compile("[ً-ٰٟ]")
_PAGE_NO_LINE = re.compile(r"^[\s\-–—_()\d۰-۹٠-٩]*$")


def strict_fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_CHAR_MAP)


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def ortho_fold_token(tok: str) -> str:
    tok = tok.replace(ZWNJ, "").replace("ۀ", "ه")
    tok = _DIACRITICS.sub("", tok)
    return "".join(ch for ch in tok if not _is_punct(ch))


def strip_markdown(md: str) -> str:
    """Remove Markdown syntax, keeping the text. One output line per input line."""
    out = []
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"^>\s?", "", s)
        s = re.sub(r"^[-*+]\s+", "", s)
        s = s.replace("**", "").replace("__", "")
        s = s.replace(" --- ", " ").replace("|", " ")
        s = re.sub(r"\[\^\d+\]:?", " ", s)
        out.append(s)
    return "\n".join(out)


@dataclass
class Tok:
    text: str          # strict-folded surface (marker brackets removed for guesses)
    line: int          # 0-based line index in its document
    marker: Optional[str] = None   # None | "unk" | "guess"
    marker_id: int = -1


def _line_tokens(line: str, line_no: int, mark_counter: list) -> list[Tok]:
    toks: list[Tok] = []
    pos = 0
    for m in _MARKER_RE.finditer(line):
        for w in line[pos:m.start()].split():
            toks.append(Tok(w, line_no))
        inner = m.group(1).strip()
        mark_counter[0] += 1
        mid = mark_counter[0]
        if inner in ("؟", "?", ""):
            toks.append(Tok(MARK_UNK, line_no, "unk", mid))
        else:
            inner = inner.rstrip("؟?").strip()
            words = inner.split() or [MARK_UNK]
            for w in words:
                toks.append(Tok(w, line_no, "guess", mid))
        pos = m.end()
    for w in line[pos:].split():
        toks.append(Tok(w, line_no))
    return toks


def tokenize_lines(lines: list[str], drop_page_numbers: bool = True) -> list[Tok]:
    """Strict tokens over a list of printed lines. Blank lines are ignored and
    do not consume a line index; page-number-only lines are dropped."""
    toks: list[Tok] = []
    counter = [0]
    ln = 0
    for raw in lines:
        line = strict_fold(raw).strip()
        if not line:
            continue
        if drop_page_numbers and _PAGE_NO_LINE.match(line):
            continue
        toks.extend(_line_tokens(line, ln, counter))
        ln += 1
    return toks


def tokenize_md(md: str) -> list[Tok]:
    return tokenize_lines(strip_markdown(md).split("\n"))


def count_lines(lines: list[str]) -> int:
    return sum(1 for l in lines if strict_fold(l).strip() and not _PAGE_NO_LINE.match(strict_fold(l).strip()))


def fold_tokens(toks: list[Tok], folding: str) -> list[Tok]:
    if folding == "strict":
        return list(toks)
    out = []
    for t in toks:
        if t.marker == "unk":
            out.append(t)
            continue
        f = ortho_fold_token(t.text)
        if f:
            out.append(Tok(f, t.line, t.marker, t.marker_id))
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


@dataclass
class Op:
    kind: str            # match | sub | del | ins | join | split
    ref: tuple           # ref token indices
    hyp: tuple           # hyp token indices
    gap: int = -1        # for del: hyp index of the token that follows the gap


def align(ref: list[str], hyp: list[str], joins: bool = False,
          wildcard: str = MARK_UNK) -> tuple[int, list[Op]]:
    """Minimum edit alignment. With joins=True, 1 hyp token may equal 2-3
    concatenated ref tokens (join) or the reverse (split) at zero cost.
    A ref token equal to `wildcard` matches anything, or is skipped, free."""
    n, m = len(ref), len(hyp)
    INF = 10 ** 9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B: list[list] = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i][j]
            if d >= INF:
                continue
            # transitions out of (i, j)
            if i < n and j < m:
                if ref[i] == hyp[j] or ref[i] == wildcard:
                    c, k = d, "match"
                else:
                    c, k = d + 1, "sub"
                if c < D[i + 1][j + 1]:
                    D[i + 1][j + 1], B[i + 1][j + 1] = c, (k, i, j, 1, 1)
            if i < n:
                c = d + (0 if ref[i] == wildcard else 1)
                if c < D[i + 1][j]:
                    D[i + 1][j], B[i + 1][j] = c, ("del", i, j, 1, 0)
            if j < m:
                c = d + 1
                if c < D[i][j + 1]:
                    D[i][j + 1], B[i][j + 1] = c, ("ins", i, j, 0, 1)
            if joins and i < n and j < m:
                for kk in (2, 3):
                    if i + kk <= n and "".join(ref[i:i + kk]) == hyp[j] and d < D[i + kk][j + 1]:
                        D[i + kk][j + 1], B[i + kk][j + 1] = d, ("join", i, j, kk, 1)
                    if j + kk <= m and "".join(hyp[j:j + kk]) == ref[i] and d < D[i + 1][j + kk]:
                        D[i + 1][j + kk], B[i + 1][j + kk] = d, ("split", i, j, 1, kk)
    ops: list[Op] = []
    i, j = n, m
    while i or j:
        kind, pi, pj, di, dj = B[i][j]
        ops.append(Op(kind, tuple(range(pi, pi + di)), tuple(range(pj, pj + dj)),
                      gap=pj if kind == "del" else -1))
        i, j = pi, pj
    ops.reverse()
    return D[n][m], ops


FOLDINGS = ("strict", "ortho", "lenient")


def _strs(toks: list[Tok]) -> list[str]:
    return [t.text for t in toks]


def wer(truth: list[Tok], hyp: list[Tok], folding: str) -> dict:
    base = "strict" if folding == "strict" else "ortho"
    r, h = fold_tokens(truth, base), fold_tokens(hyp, base)
    cost, ops = align(_strs(r), _strs(h), joins=(folding == "lenient"))
    n = sum(1 for t in r if t.text != MARK_UNK)
    return {"wer": cost / n if n else 0.0, "errors": cost, "n_truth": n, "n_hyp": len(h)}


@dataclass
class ErrorEvent:
    kind: str            # sub | ins | del
    ref: int             # ref index (or ref gap position for ins)
    hyp: int             # hyp index (sub/ins) or hyp gap (del: index of the following hyp token)


def error_events(ops: list[Op], ref: list[str]) -> list[ErrorEvent]:
    ev = []
    ref_pos = 0
    for op in ops:
        if op.kind == "sub" and ref[op.ref[0]] != MARK_UNK:
            ev.append(ErrorEvent("sub", op.ref[0], op.hyp[0]))
        elif op.kind == "ins":
            ev.append(ErrorEvent("ins", ref_pos, op.hyp[0]))
        elif op.kind == "del" and ref[op.ref[0]] != MARK_UNK:
            ev.append(ErrorEvent("del", op.ref[0], op.gap))
        if op.ref:
            ref_pos = op.ref[-1] + 1
    return ev


def lenient_alignment(truth: list[Tok], hyp: list[Tok]):
    r, h = fold_tokens(truth, "ortho"), fold_tokens(hyp, "ortho")
    _, ops = align(_strs(r), _strs(h), joins=True)
    return r, h, ops, error_events(ops, _strs(r))


def invented_words(truth: list[Tok], hyp: list[Tok], window: int = 3, sim_min: float = 0.5) -> list[int]:
    """Indices (into the ortho-folded hyp) of invented tokens; see module doc."""
    r, h, ops, events = lenient_alignment(truth, hyp)
    rs = _strs(r)
    out = []
    for e in events:
        if e.kind == "del":
            continue
        tok = h[e.hyp]
        if tok.marker == "unk":
            continue
        lo, hi = max(0, e.ref - window), min(len(rs), e.ref + window + 1)
        best = max((char_sim(tok.text, rs[k]) for k in range(lo, hi) if rs[k] != MARK_UNK), default=0.0)
        if best < sim_min:
            out.append(e.hyp)
    return out


def _covered(e: ErrorEvent, flagged_tok: set, flagged_gap: set) -> bool:
    if e.kind in ("sub", "ins"):
        return e.hyp in flagged_tok
    return e.hyp in flagged_gap or e.hyp in flagged_tok or (e.hyp - 1) in flagged_tok


def marker_stats(truth: list[Tok], hyp: list[Tok]) -> dict:
    r, h, ops, events = lenient_alignment(truth, hyp)
    marked = {i for i, t in enumerate(h) if t.marker}
    n_unk = len({t.marker_id for t in h if t.marker == "unk"})
    guess_ids = {t.marker_id for t in h if t.marker == "guess"}
    err_hyp = {e.hyp for e in events if e.kind in ("sub", "ins")}
    guess_right = guess_wrong = 0
    for mid in guess_ids:
        idx = [i for i, t in enumerate(h) if t.marker_id == mid and t.marker == "guess"]
        if any(i in err_hyp for i in idx):
            guess_wrong += 1
        else:
            guess_right += 1
    in_markers = sum(1 for e in events if _covered(e, marked, set()))
    return {"markers_unk": n_unk, "markers_guess": len(guess_ids),
            "guess_correct": guess_right, "guess_wrong": guess_wrong,
            "errors": len(events), "errors_in_markers": in_markers,
            "silent_errors": len(events) - in_markers}


# ---------------------------------------------------------------------------
# T2: two-reading agreement
# ---------------------------------------------------------------------------

def t2_flags(primary: list[Tok], secondary: list[Tok]) -> tuple[set, set]:
    """Flag primary (ortho-folded) tokens the secondary reading disagrees with,
    and primary gaps where the secondary has extra words. Lenient alignment,
    so join/ZWNJ/punctuation differences are not disagreements."""
    p, s = fold_tokens(primary, "ortho"), fold_tokens(secondary, "ortho")
    _, ops = align(_strs(p), _strs(s), joins=True, wildcard="\x00")
    tok, gap = set(), set()
    p_pos = 0
    for op in ops:
        if op.kind in ("sub", "del"):
            tok.update(op.ref)
        elif op.kind == "ins":
            gap.add(p_pos)
        if op.ref:
            p_pos = op.ref[-1] + 1
    for t in [i for i, x in enumerate(p) if x.marker == "unk"]:
        tok.add(t)
    return tok, gap


def t2_stats(truth: list[Tok], primary: list[Tok], secondary: list[Tok]) -> dict:
    ftok, fgap = t2_flags(primary, secondary)
    r, h, ops, events = lenient_alignment(truth, primary)
    caught = sum(1 for e in events if _covered(e, ftok, fgap))
    n = len(h)
    return {"errors": len(events), "caught": caught,
            "recall": caught / len(events) if events else 1.0,
            "flagged_words": len(ftok), "flagged_gaps": len(fgap), "primary_words": n,
            "flag_rate": (len(ftok) + len(fgap)) / n if n else 0.0}


def vote3(primary: list[Tok], s1: list[Tok], s2: list[Tok]) -> tuple[list[Tok], set]:
    """Word-level 2-of-3 vote (deterministic, zero cost). Each secondary is
    aligned to the primary (lenient, ortho-folded). A primary token is replaced
    when both secondaries align a substitution with the same token, and dropped
    when both delete it. Returns (voted tokens, indices of voted tokens on
    which the three readings still do not all agree)."""
    p = fold_tokens(primary, "ortho")

    def amap(sec):
        q = fold_tokens(sec, "ortho")
        _, ops = align(_strs(p), _strs(q), joins=True, wildcard="\x00")
        m = {}
        for op in ops:
            if op.kind in ("match", "join"):
                for i in op.ref:
                    m[i] = ("same", None)
            elif op.kind == "split":
                m[op.ref[0]] = ("same", None)
            elif op.kind == "sub":
                m[op.ref[0]] = ("sub", q[op.hyp[0]])
            elif op.kind == "del":
                m[op.ref[0]] = ("del", None)
        return m

    m1, m2 = amap(s1), amap(s2)
    out, unsure = [], set()
    for i, t in enumerate(p):
        a, b = m1.get(i, ("del", None)), m2.get(i, ("del", None))
        if a[0] == "sub" and b[0] == "sub" and a[1].text == b[1].text:
            out.append(Tok(a[1].text, t.line, a[1].marker, a[1].marker_id))
            continue
        if a[0] == "del" and b[0] == "del":
            continue
        if a[0] != "same" or b[0] != "same":
            unsure.add(len(out))
        out.append(t)
    return out, unsure


# ---------------------------------------------------------------------------
# Line-break accuracy
# ---------------------------------------------------------------------------

def line_stats(truth_lines: list[str], hyp_lines: list[str]) -> dict:
    truth = tokenize_lines(truth_lines)
    hyp = tokenize_lines(hyp_lines)
    r, h = fold_tokens(truth, "ortho"), fold_tokens(hyp, "ortho")
    _, ops = align(_strs(r), _strs(h), joins=True)
    hyp_line_of_ref: dict[int, int] = {}
    for op in ops:
        if op.kind in ("match", "sub", "join", "split"):
            for ri in op.ref:
                hyp_line_of_ref[ri] = h[op.hyp[0]].line
    n_lines = (r[-1].line + 1) if r else 0
    by_line: dict[int, list[int]] = {}
    for i, t in enumerate(r):
        by_line.setdefault(t.line, []).append(i)
    # A split op whose pieces fall on two hyp lines is itself a broken line.
    split_broken = {r[op.ref[0]].line for op in ops if op.kind == "split"
                    and len({h[j].line for j in op.hyp}) > 1}

    def hl(i):
        return hyp_line_of_ref.get(i)

    correct = 0
    for L in range(n_lines):
        idx = by_line.get(L, [])
        aligned = [hl(i) for i in idx if hl(i) is not None]
        if not aligned or L in split_broken:
            continue
        if len(set(aligned)) != 1:
            continue
        hline = aligned[0]
        prev_idx = [hl(i) for k in range(L) for i in by_line.get(k, []) if hl(i) is not None]
        next_idx = [hl(i) for k in range(L + 1, n_lines) for i in by_line.get(k, []) if hl(i) is not None]
        prev_ok = (not prev_idx) or prev_idx[-1] == hline - 1
        next_ok = (not next_idx) or next_idx[0] == hline + 1
        if prev_ok and next_ok:
            correct += 1
    return {"truth_lines": count_lines(truth_lines), "hyp_lines": count_lines(hyp_lines),
            "lines_break_correct": correct, "break_correct_share": correct / n_lines if n_lines else 1.0}


# ---------------------------------------------------------------------------
# Band stitching
# ---------------------------------------------------------------------------

def _line_key(s: str) -> str:
    return "".join(ortho_fold_token(t) for t in strict_fold(s).split())


def stitch_bands(bands: list[list[str]], sim_min: float = 0.6, window: int = 12) -> tuple[list[str], list[str]]:
    """Merge per-band line lists top to bottom. The tail of the merged text is
    aligned with the head of the next band (monotone, unmatched lines allowed
    on either side); a matched pair is kept once (the copy farther from its
    band edge, i.e. the earlier band for the first half of the overlap), and
    an unmatched line inside the overlap is kept. Returns (lines, warnings)."""
    warnings: list[str] = []
    merged: list[str] = [l for l in (bands[0] if bands else []) if l.strip()]
    for bi, band in enumerate(bands[1:], 1):
        band = [l for l in band if l.strip()]
        if not band:
            warnings.append(f"band {bi} empty")
            continue
        t0 = max(0, len(merged) - window)
        A = merged[t0:]
        Bh = band[:window]
        ka, kb = [_line_key(x) for x in A], [_line_key(x) for x in Bh]
        na, nb = len(A), len(Bh)
        # maximise matches; free unmatched prefix of A and suffix of Bh
        S = [[0.0] * (nb + 1) for _ in range(na + 1)]
        P = [[None] * (nb + 1) for _ in range(na + 1)]
        for i in range(na + 1):
            for j in range(nb + 1):
                if i == 0 and j == 0:
                    continue
                best, bp = -1e9, None
                if i > 0 and j > 0 and char_sim(ka[i - 1], kb[j - 1]) >= sim_min:
                    v = S[i - 1][j - 1] + 1
                    if v > best:
                        best, bp = v, "m"
                if i > 0:
                    v = S[i - 1][j] - (0.0 if j == 0 else 0.25)
                    if v > best:
                        best, bp = v, "a"
                if j > 0:
                    v = S[i][j - 1] - 0.25
                    if v > best:
                        best, bp = v, "b"
                S[i][j], P[i][j] = best, bp
        # end anywhere in Bh (rest of band follows), but all of A must be consumed
        jbest = max(range(nb + 1), key=lambda j: (S[na][j], -j))
        pairs = []
        i, j = na, jbest
        while i > 0 or j > 0:
            bp = P[i][j]
            if bp == "m":
                pairs.append(("m", i - 1, j - 1)); i, j = i - 1, j - 1
            elif bp == "a":
                pairs.append(("a", i - 1, None)); i -= 1
            else:
                pairs.append(("b", None, j - 1)); j -= 1
        pairs.reverse()
        matched = [p for p in pairs if p[0] == "m"]
        if not matched:
            warnings.append(f"band {bi}: no overlap match; concatenated")
            merged = merged + band
            continue
        first_a = matched[0][1]
        region = [p for p in pairs if not (p[0] == "a" and p[1] < first_a)]
        out = merged[:t0 + first_a]
        nm = len(matched)
        mi = 0
        for kind, ia, jb in region:
            if kind == "m":
                out.append(A[ia] if mi < (nm + 1) // 2 else Bh[jb])
                mi += 1
            elif kind == "a":
                out.append(A[ia])
            else:
                out.append(Bh[jb])
        merged = out + band[jbest:]
    return merged, warnings


# ---------------------------------------------------------------------------
# Ledger, budget guard and cache
# ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class AccessError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, cap: float = CAP_USD):
        self.path, self.cap = Path(path), cap
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def total(self) -> float:
        return sum(e.get("cost", 0.0) + e.get("uncertain_cost", 0.0) for e in self.entries())

    def guard(self, worst_case: float) -> None:
        t = self.total()
        if t + worst_case > self.cap:
            raise BudgetExceeded(f"ledger ${t:.4f} + worst case ${worst_case:.4f} would pass cap ${self.cap:.2f}")

    def record(self, entry: dict) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
        with self.path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# A stronger model than the repo's MODEL_STRONG, for the per-line follow-up.
# Defined here on purpose (not in farsi2epub/config.py).
MODEL_OPUS = "claude-opus-5-5"
# USD per MTok. Source: the claude-api skill's model table ("Current Models",
# cached 2026-11-03): Claude Opus 5.5 $4.00 in / $20.00 out. The budget guard
# pads prices of models outside config.PRICES by GUARD_MARGIN_EXTRA.
PRICES_EXTRA = {MODEL_OPUS: {"in": 4.0, "out": 20.0}}
GUARD_MARGIN_EXTRA = 1.25
HIGH_RES_PREFIXES = ("claude-sonnet-5", "claude-opus-5")


def price(model: str) -> dict:
    return config.PRICES[model] if model in config.PRICES else PRICES_EXTRA[model]


def guard_price(model: str) -> dict:
    p = price(model)
    if model in config.PRICES:
        return p
    return {k: v * GUARD_MARGIN_EXTRA for k, v in p.items()}


def cost_usd(usage, model: str) -> float:
    p = price(model)
    return usage.input_tokens / 1e6 * p["in"] + usage.output_tokens / 1e6 * p["out"]


def text_tokens_est(s: str) -> int:
    return len(s.encode("utf-8")) // 3 + 50


def request_key(req: dict) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class Image:
    data: bytes
    media_type: str
    width: int
    height: int
    label: str = ""

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def image_block(img: Image) -> dict:
    import base64
    return {"type": "image", "source": {"type": "base64", "media_type": img.media_type,
                                        "data": base64.standard_b64encode(img.data).decode("ascii")}}


def api_tokens(img: Image, model: str) -> int:
    """Visual tokens for this model's tier (repo rule, config.image_tokens)."""
    tiers = config.IMAGE_TIERS if model.startswith(HIGH_RES_PREFIXES) else config.IMAGE_TIERS[1:]
    edge, cap = tiers[0]
    return config._patch_tokens(*config._resized_size(img.width, img.height, edge, cap))


def api_size(img: Image, model: str) -> tuple[int, int]:
    tiers = config.IMAGE_TIERS if model.startswith(HIGH_RES_PREFIXES) else config.IMAGE_TIERS[1:]
    edge, cap = tiers[0]
    return config._resized_size(img.width, img.height, edge, cap)


@dataclass
class Request:
    arm: str
    page: str
    model: str
    system: str
    content: list          # list of Image | str, in order
    schema: type
    max_tokens: int = llm.MAX_OUTPUT_TOKENS
    rep: int = 0
    expected_out: int = 1800
    thinking: Optional[dict] = None   # None = repo convention (thinking_kwargs)
    extra: Optional[dict] = None      # extra API kwargs, e.g. output_config effort

    def thinking_kw(self) -> dict:
        kw = {"thinking": self.thinking} if self.thinking else thinking_kwargs(self.model)
        return {**kw, **(self.extra or {})}

    def key_dict(self) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens, "system": self.system,
                "content": [{"image": c.sha, "mt": c.media_type} if isinstance(c, Image) else {"text": c}
                            for c in self.content],
                "schema": self.schema.model_json_schema(), "thinking": self.thinking_kw(),
                "rep": self.rep, **({"extra": self.extra} if self.extra else {})}

    def input_est(self) -> int:
        n = text_tokens_est(self.system)
        for c in self.content:
            n += api_tokens(c, self.model) if isinstance(c, Image) else text_tokens_est(c)
        return n

    def worst_case(self) -> float:
        p = guard_price(self.model)
        return 1.10 * self.input_est() / 1e6 * p["in"] + self.max_tokens / 1e6 * p["out"]

    def expected(self) -> float:
        p = price(self.model)
        return self.input_est() / 1e6 * p["in"] + self.expected_out / 1e6 * p["out"]


def thinking_kwargs(model: str) -> dict:
    # Same convention as farsi2epub/llm.py: thinking disabled on Sonnet 5.
    return {"thinking": {"type": "disabled"}} if model.startswith("claude-sonnet-5") else {}


def _is_access_error(exc: BaseException) -> bool:
    import anthropic
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return True
    msg = str(exc).lower()
    return isinstance(exc, anthropic.BadRequestError) and ("credit" in msg or "billing" in msg)


class Runner:
    def __init__(self, out: Path = OUT, client_factory: Optional[Callable] = None, cap: float = CAP_USD):
        self.out = Path(out)
        self.cache_dir = self.out / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.out / "ledger.jsonl", cap)
        self._client = None
        self._factory = client_factory or llm.get_client

    @property
    def client(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def cache_path(self, req: Request) -> Path:
        return self.cache_dir / f"{request_key(req.key_dict())}.json"

    def cached(self, req: Request) -> Optional[dict]:
        p = self.cache_path(req)
        return json.loads(p.read_text()) if p.is_file() else None

    def call(self, req: Request) -> dict:
        hit = self.cached(req)
        if hit is not None:
            hit["cache_hit"] = True
            return hit
        key = request_key(req.key_dict())
        content = [image_block(c) if isinstance(c, Image) else {"type": "text", "text": c} for c in req.content]
        last_exc = None
        for attempt in (1, 2):   # at most one retry
            self.ledger.guard(req.worst_case())
            try:
                resp = self.client.messages.parse(
                    model=req.model, max_tokens=req.max_tokens, system=req.system,
                    messages=[{"role": "user", "content": content}],
                    output_format=req.schema, **req.thinking_kw())
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                status = getattr(exc, "status_code", None)
                pre = status in (400, 401, 403, 404, 413, 422, 429)
                self.ledger.record({"arm": req.arm, "page": req.page, "rep": req.rep, "model": req.model,
                                    "key": key, "status": f"error:{type(exc).__name__}:{status}",
                                    "attempt": attempt, "cost": 0.0,
                                    "uncertain_cost": 0.0 if pre else round(req.worst_case(), 6)})
                if _is_access_error(exc):
                    raise AccessError(str(exc)) from exc
                continue
            usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
            cost = cost_usd(resp.usage, req.model)
            parsed = resp.parsed_output
            self.ledger.record({"arm": req.arm, "page": req.page, "rep": req.rep, "model": req.model,
                                "key": key, "status": "ok" if parsed is not None else "unparsed",
                                "attempt": attempt, **usage, "cost": round(cost, 6), "uncertain_cost": 0.0})
            if parsed is None:
                last_exc = RuntimeError("unparseable output")
                continue
            rec = {"arm": req.arm, "page": req.page, "rep": req.rep, "model": req.model,
                   "parsed": parsed.model_dump(), "usage": usage, "cost": cost,
                   "stop_reason": getattr(resp, "stop_reason", None),
                   "images": [{"sha": c.sha, "w": c.width, "h": c.height, "label": c.label,
                               "api_w_h": api_size(c, req.model), "api_tokens_est": api_tokens(c, req.model)}
                              for c in req.content if isinstance(c, Image)],
                   "input_est": req.input_est()}
            self.cache_path(req).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            rec["cache_hit"] = False
            return rec
        raise RuntimeError(f"{req.arm} {req.page}: failed twice: {last_exc}")


# ---------------------------------------------------------------------------
# Prompts and schemas
# ---------------------------------------------------------------------------

_CUR = llm.TRANSCRIBE_SYSTEM
assert "TRANSCRIPTION RULES" in _CUR and "IGNORE COMPLETELY" in _CUR and "MARKDOWN STRUCTURE" in _CUR
_CUR_HEAD, _rest = _CUR.split("TRANSCRIPTION RULES", 1)
_CUR_RULES, _rest = _rest.split("IGNORE COMPLETELY", 1)
_IGNORE, _MD_AND_FIELDS = _rest.split("MARKDOWN STRUCTURE", 1)
IGNORE_SECTION = "IGNORE COMPLETELY" + _IGNORE
CURRENT_RULES_ONLY = _CUR.split("MARKDOWN STRUCTURE", 1)[0]

DIPLOMATIC_RULES = """TRANSCRIPTION RULES (diplomatic: write exactly what is printed)
- Write every word exactly as printed: same letters, same spelling, same punctuation, and the same spacing around punctuation. Do not correct typos, modernize, or standardize spelling.
- Word joins and splits exactly as printed. Typewritten and older print often runs two words together with no space, or leaves a prefix such as می or a suffix such as ها standing apart; copy what you see. Never add a zero-width non-joiner (U+200C) or a space that is not in the print, and never remove one that is.
- Use standard Persian codepoints: always ی (U+06CC) and ک (U+06A9), never Arabic ي or ك. Keep Persian digits (۰۱۲۳۴۵۶۷۸۹) and the digits' order exactly as printed.
- Remove kashida/tatweel stretching: write بهتر even if printed بـــهـــتر.
- NEVER complete, infer from context, or paraphrase text you cannot actually read. A fluent phrase that is not on the page is the worst possible error; a marked gap is cheap for the human reviewer to fix.
  - For each word you cannot read, write ⟦؟⟧ (one marker per unreadable word).
  - For a word you are guessing at, write it inside the marker: ⟦کتاب؟⟧.
  - Mark only real uncertainty: write confidently read words plainly.
- Do not transcribe signatures, stamps or seals, handwriting, or watermarks. Where stamp ink covers printed text, write the printed words you can read and mark the rest.
- Completeness: transcribe every printed line of body text, top to bottom. Never skip a line; if a whole line is unreadable, write one ⟦؟⟧ per word you can see.

"""

READ_RULES = """TRANSCRIPTION RULES (identify each typed word; write joins and spacing as printed)
- The page may be typewritten, photographed, blurred or stained, so letter shapes are crude: dots merge, break or vanish; the medial forms of ب پ ت ث ن ی can look alike; ه ع م and ف ق can blur together; a typewriter often leaves a small gap inside a word after a non-joining letter that is not a word break. Identify each word as the Persian word that was typed. Never write a non-word that merely copies a glyph defect.
- Use the surrounding words only to choose between letter readings that the visible shapes allow. Never write a word, name, number or phrase that the visible letters do not support, and never complete a sentence from what it "should" say.
- When you cannot identify a word from what is visible, write ⟦؟⟧ (one per unreadable word). When you are choosing between readings, write your best reading as ⟦word؟⟧. On a hard page, expect to use many markers: an unmarked wrong word is far worse than a marked right one. Mark each word you would not bet on.
- Word joins and splits exactly as typed: if the typist ran two words together with no space, write them together; if a prefix such as می or a suffix such as ها stands apart, write it apart. Never add a zero-width non-joiner (U+200C). Keep the spacing around punctuation as printed.
- Use standard Persian codepoints: always ی (U+06CC) and ک (U+06A9), never Arabic ي or ك. Persian digits (۰۱۲۳۴۵۶۷۸۹), in the order the typist typed them.
- Remove kashida/tatweel stretching.
- Do not transcribe signatures, stamps or seals, handwriting, or watermarks. Where stamp ink covers printed text, write the printed words you can read and mark the rest.
- Completeness: transcribe every printed line of body text, top to bottom, including headings. Never skip a line.

"""

DIPLOMATIC_SYSTEM = (_CUR_HEAD.replace("into clean Markdown", "into Markdown")
                     + DIPLOMATIC_RULES + IGNORE_SECTION + "MARKDOWN STRUCTURE" + _MD_AND_FIELDS)

LINES_FORMAT = """OUTPUT FORMAT
- `lines`: one string per physically printed line, top to bottom, in reading order. Never merge two printed lines into one string and never split one printed line across two. A heading, a centered title, or the short last line of a paragraph is its own line.
- Plain text: no Markdown, no heading marks, no list marks."""

LINES_SYSTEM = (_CUR_HEAD.split("You receive", 1)[0]
                + "You receive one page as an image and write down its printed lines, exactly as printed.\n\n"
                + DIPLOMATIC_RULES + IGNORE_SECTION + LINES_FORMAT)

BAND_FORMAT = """INPUT AND OUTPUT FORMAT (replaces any Markdown or field instructions)
- The page arrives as {n} overlapping horizontal bands of the same page, top to bottom, at full resolution, each preceded by a label "Band k:". Consecutive bands overlap by a few printed lines.
- Transcribe each band on its own and echo its number as band_index. For each band return `lines`: one string per printed line that is fully visible in that band, top to bottom, including lines that also appear in the neighbouring band. Skip only a line whose letters are cut by the band's top or bottom edge (it is complete in the neighbouring band).
- One string per physically printed line: never merge two printed lines or split one. Plain text: no Markdown."""


def band_system(rules: str, n: int) -> str:
    return rules + "\n" + BAND_FORMAT.format(n=n)


BANDS_CURRENT_SYSTEM = lambda n: band_system(CURRENT_RULES_ONLY.rstrip() + "\n\n", n)  # noqa: E731
BANDS_DIPLOMATIC_SYSTEM = lambda n: band_system(  # noqa: E731
    _CUR_HEAD.split("You receive", 1)[0] + "You receive one page and write down its printed lines.\n\n"
    + DIPLOMATIC_RULES + IGNORE_SECTION.rstrip() + "\n", n)

BANDS_READ_SYSTEM = lambda n: band_system(  # noqa: E731
    _CUR_HEAD.split("You receive", 1)[0] + "You receive one page and write down its printed lines.\n\n"
    + READ_RULES + IGNORE_SECTION.rstrip() + "\n", n)

FOLDED_ADDENDUM = """

ADDITIONAL FIELD printed_lines
- Besides text_md, return `printed_lines`: the same page as one string per physically printed line of body text (headings included), top to bottom, written diplomatically: exactly as printed, word joins and splits as printed, no zero-width non-joiner or space added that is not in the print, no Markdown.
- In printed_lines only, never complete or guess silently: write ⟦؟⟧ for a word you cannot read and ⟦word؟⟧ for a guess. text_md follows the rules above."""

FOLDED_SYSTEM = _CUR + FOLDED_ADDENDUM


class LineReading(BaseModel):
    lines: list[str] = Field(description="One string per physically printed line, top to bottom.")


class BandLines(BaseModel):
    band_index: int = Field(description="The k from the band's 'Band k:' label.")
    lines: list[str] = Field(description="Printed lines fully visible in this band, top to bottom.")


class BandsReading(BaseModel):
    bands: list[BandLines]


class PageWithLines(llm.PageTranscription):
    printed_lines: list[str] = Field(description="Diplomatic reading: one string per printed line, as printed.")


# ---------------------------------------------------------------------------
# Pages and images
# ---------------------------------------------------------------------------

HADPI = "hadpi-articles"
CONTROL = "ghessehaye-majid"
PAGES = {"h1": (HADPI, 1), "h2": (HADPI, 2), "c23": (CONTROL, 23), "c109": (CONTROL, 109)}
HELD_OUT = {(HADPI, 3), (HADPI, 4), (HADPI, 5)}
BAND_OVERLAP_PX = 300


def _check(slug, page):
    if (slug, page) in HELD_OUT:
        raise SystemExit(f"{slug} p{page} is held out")


def book_meta(slug: str) -> dict:
    import yaml
    return yaml.safe_load((ROOT / "books" / slug / "book.yaml").read_text())


def whole_page_image(slug: str, page: int) -> Image:
    """The exact bytes the app transcribes: books/<slug>/pages/hires/NNNN.png."""
    _check(slug, page)
    import fitz
    data = (ROOT / "books" / slug / "pages" / "hires" / f"{page:04d}.png").read_bytes()
    pix = fitz.Pixmap(data)
    return Image(data, "image/png", pix.width, pix.height, "page")


def _band_rows(height: int, width: int, overlap: int = BAND_OVERLAP_PX) -> list[tuple[int, int]]:
    """Fewest bands of equal height, overlapping by `overlap` px, such that no
    band is downsized by the API (sonnet-5 high-res tier)."""
    edge, cap = config.IMAGE_TIERS[0]
    cols = -(-width // config.IMAGE_PATCH_PX)
    max_h = min(edge, (cap // cols) * config.IMAGE_PATCH_PX)
    n = max(1, math.ceil((height - overlap) / (max_h - overlap)))
    bh = math.ceil((height + (n - 1) * overlap) / n)
    while -(-bh // config.IMAGE_PATCH_PX) * cols > cap or bh > edge:
        n += 1
        bh = math.ceil((height + (n - 1) * overlap) / n)
    step = (height - bh) / (n - 1) if n > 1 else 0
    rows = []
    for k in range(n):
        y0 = int(round(k * step))
        rows.append((y0, min(height, y0 + bh)))
    return rows


def band_source(slug: str, page: int):
    """(pixmap, px_per_pt, kind): the embedded photo at native size if it is
    finer than the hi-res render, else the hi-res render itself."""
    _check(slug, page)
    import fitz
    doc = fitz.open(str(ROOT / "books" / slug / "source.pdf"))
    pg = doc[page - 1]
    render_zoom = config.LONG_EDGE_HI / max(pg.rect.width, pg.rect.height)
    best = None
    for info in pg.get_image_info(xrefs=True):
        bw = info["bbox"][2] - info["bbox"][0]
        if bw <= 0 or not info.get("xref"):
            continue
        ppt = info["width"] / bw
        if best is None or info["width"] * info["height"] > best[1]["width"] * best[1]["height"]:
            best = (ppt, info)
    if best and best[0] > render_zoom:
        pix = fitz.Pixmap(doc, best[1]["xref"])
        if pix.alpha or pix.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix, best[0], "embedded_native"
    img = whole_page_image(slug, page)
    return fitz.Pixmap(img.data), render_zoom, "hires_render"


def band_images(slug: str, page: int) -> tuple[list[Image], dict]:
    import fitz
    import numpy as np
    pix, ppt, kind = band_source(slug, page)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rows = _band_rows(pix.height, pix.width)
    out = []
    for k, (y0, y1) in enumerate(rows):
        sub = np.ascontiguousarray(arr[y0:y1])
        bp = fitz.Pixmap(pix.colorspace, pix.width, y1 - y0, sub.tobytes(), False)
        data, mt = bp.tobytes("png"), "image/png"
        if len(data) > 3_700_000:
            data, mt = bp.tobytes("jpg", jpg_quality=95), "image/jpeg"
        out.append(Image(data, mt, pix.width, y1 - y0, f"band{k + 1}:{y0}-{y1}"))
    return out, {"source": kind, "px_per_pt": ppt, "src_w": pix.width, "src_h": pix.height, "rows": rows}


def user_text(slug: str, page: int) -> str:
    """Same user message as llm.transcribe_page."""
    meta = book_meta(slug)
    title = meta.get("title_fa")
    hint = f"کتاب: {title}" if title else "عنوان کتاب نامشخص است."
    return f"صفحهٔ {page} از {meta.get('page_count', page)}. {hint}\nTranscribe this page following the system instructions."


# ---------------------------------------------------------------------------
# Per-line crops from audited line outlines (follow-up arm)
# ---------------------------------------------------------------------------
#
# Each audited line has a frame (centre cx, cy and clockwise angle, y down, in
# the audit image's pixels) and a box [u0, v0, u1, v1] in that frame. A strip
# is resampled straight (bilinear) from the finest source: the PDF's embedded
# photo at native size when it is finer than the audit image, else the audit
# image itself. The geometry is the owner's audit, so this arm measures
# reading alone, not line finding.

LINE_SCALE_CAP = 1.6          # enlargement over the source's native pixels
LINE_MARGIN_V = 0.30          # of the line box height, above and below
LINE_MARGIN_U_PX = 16         # native px, left and right
CONTEXT_MARK_RGB = (220, 0, 0)
EXCLUDE_LINES = {"h1": {0}}   # p1 line 0: audited outline sits on an underline


def load_audit(pid: str) -> dict:
    slug, page = PAGES[pid]
    _check(slug, page)
    return json.loads((ROOT / "books" / slug / "truth" / f"{page:04d}.geometry.audit.json").read_text())


def frame_point(frame: dict, u: float, v: float) -> tuple[float, float]:
    """Line-frame (u along the line, v down) -> image (x right, y down)."""
    a = math.radians(frame["angle_deg"])
    return (frame["cx"] + u * math.cos(a) - v * math.sin(a),
            frame["cy"] + u * math.sin(a) + v * math.cos(a))


def sample_affine(arr, x0: float, y0: float, ux: tuple, vy: tuple, width: int, height: int):
    """Bilinear resample of arr (H, W, C uint8): output pixel (i, j) reads
    source point (x0, y0) + j*ux + i*vy. Points outside the source are white."""
    import numpy as np
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    xs = x0 + jj * ux[0] + ii * vy[0]
    ys = y0 + jj * ux[1] + ii * vy[1]
    H, W, C = arr.shape
    x0f, y0f = np.floor(xs), np.floor(ys)
    fx, fy = (xs - x0f)[..., None], (ys - y0f)[..., None]
    xi, yi = x0f.astype(int), y0f.astype(int)
    pad = np.pad(arr, ((1, 2), (1, 2), (0, 0)), constant_values=255).astype(np.float64)
    xi = np.clip(xi + 1, 0, W + 1)
    yi = np.clip(yi + 1, 0, H + 1)
    out = (pad[yi, xi] * (1 - fx) * (1 - fy) + pad[yi, xi + 1] * fx * (1 - fy)
           + pad[yi + 1, xi] * (1 - fx) * fy + pad[yi + 1, xi + 1] * fx * fy)
    outside = (xs < -0.5) | (ys < -0.5) | (xs > W - 0.5) | (ys > H - 0.5)
    out[outside] = 255
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def png_from_array(arr) -> bytes:
    import fitz
    import numpy as np
    arr = np.ascontiguousarray(arr)
    cs = fitz.csGRAY if arr.shape[2] == 1 else fitz.csRGB
    return fitz.Pixmap(cs, arr.shape[1], arr.shape[0], arr.tobytes(), False).tobytes("png")


def line_source(pid: str):
    """(array, k, ox, oy, src_kind): audit px (x, y) -> source px (k*x + ox, k*y + oy)."""
    import fitz
    import numpy as np
    slug, page = PAGES[pid]
    audit = load_audit(pid)
    zoom = audit["image"]["zoom"]
    pix, ppt, kind = band_source(slug, page)
    if kind == "embedded_native":
        doc = fitz.open(str(ROOT / "books" / slug / "source.pdf"))
        info = max((i for i in doc[page - 1].get_image_info(xrefs=True) if i.get("xref")),
                   key=lambda i: i["width"] * i["height"])
        bx0, by0 = info["bbox"][0], info["bbox"][1]
        k = ppt / zoom
        ox, oy = -bx0 * ppt, -by0 * ppt
    else:
        apath = ROOT / audit["image"]["path"]
        pix = fitz.Pixmap(apath.read_bytes())
        if pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        k, ox, oy, kind = 1.0, 0.0, 0.0, "audit_image"
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr, k, ox, oy, kind


def crop_line(arr, k, ox, oy, line: dict, context_lines: tuple = (), scale_cap: float = LINE_SCALE_CAP,
              pitch: float = 0.0):
    """Straightened, enlarged strip of one audited line. With context_lines
    (the neighbouring audited lines), the strip extends to cover them and the
    target line is marked with red bars in the left and right margins.
    Returns (array, meta)."""
    import numpy as np
    u0, v0, u1, v1 = line["box"]
    bh = v1 - v0
    mv = LINE_MARGIN_V * bh
    mu = LINE_MARGIN_U_PX / k
    top, bot = v0 - mv, v1 + mv
    if context_lines:
        top = min(top, v0 - pitch - mv)
        bot = max(bot, v1 + pitch + mv)
        for nb in context_lines:
            u0 = min(u0, nb["box"][0])
            u1 = max(u1, nb["box"][2])
        mu = 3 * LINE_MARGIN_U_PX / k
    left, right = u0 - mu, u1 + mu
    w_native = (right - left) * k
    edge = config.IMAGE_TIERS[0][0]
    scale = min(scale_cap, (edge - 1) / w_native)
    width = int(w_native * scale)
    height = int((bot - top) * k * scale)
    while config._patch_tokens(width, height) > config.IMAGE_TIERS[0][1]:
        scale *= 0.95
        width, height = int(w_native * scale), int((bot - top) * k * scale)
    step = 1.0 / (k * scale)            # audit px per output px
    a = math.radians(line["frame"]["angle_deg"])
    ux = (math.cos(a) * step * k, math.sin(a) * step * k)
    vy = (-math.sin(a) * step * k, math.cos(a) * step * k)
    sx, sy = frame_point(line["frame"], left, top)
    out = sample_affine(arr, k * sx + ox, k * sy + oy, ux, vy, width, height)
    if context_lines:
        t0 = int((v0 - top) * k * scale)
        t1 = int((v1 - top) * k * scale)
        bar = max(4, int(LINE_MARGIN_U_PX * scale * 0.6))
        for x_lo in (0, width - bar):
            out[max(0, t0):t1, x_lo:x_lo + bar] = CONTEXT_MARK_RGB[:out.shape[2]] if out.shape[2] == 3 else 0
    return out, {"scale": round(scale, 3), "box_h_px": round(bh * k * scale, 1), "w": width, "h": height}


def line_images(pid: str, context: bool = False) -> list[Optional[Image]]:
    """One Image per audited line (None for excluded and page-number lines)."""
    import statistics as st
    audit = load_audit(pid)
    lines = audit["lines"]
    arr, k, ox, oy, kind = line_source(pid)
    cys = [l["frame"]["cy"] for l in lines]
    pitch = st.median([b - a for a, b in zip(cys, cys[1:])])
    truth = [l for l in truth_lines(pid) if l.strip()]
    out: list[Optional[Image]] = []
    for i, line in enumerate(lines):
        if i in EXCLUDE_LINES.get(pid, set()) or _PAGE_NO_LINE.match(strict_fold(truth[i]).strip()):
            out.append(None)
            continue
        ctx = tuple(lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)) if context else ()
        a, meta = crop_line(arr, k, ox, oy, line, ctx, pitch=pitch)
        out.append(Image(png_from_array(a), "image/png", meta["w"], meta["h"],
                         f"line{i}:{kind}:x{meta['scale']}:box{meta['box_h_px']}"))
    return out


PERLINE_RULES = READ_RULES.replace(
    "- Completeness: transcribe every printed line of body text, top to bottom, including headings. Never skip a line.\n",
    "- Write every word of the line, right to left, in order. Do not add words from outside the line.\n")

LINE_SYSTEM_SINGLE = (
    "You are an expert transcriber of Persian (Farsi) typewritten and printed documents. "
    "You receive one straightened, enlarged image of a single printed line cut from a photographed page. "
    "Partial letters of the lines above and below may show at the top and bottom edges: ignore them.\n\n"
    + PERLINE_RULES + "OUTPUT\n- `text`: the printed line, exactly as typed, as one string. No Markdown.")

LINE_SYSTEM_BATCH = (
    "You are an expert transcriber of Persian (Farsi) typewritten and printed documents. "
    "You receive {n} straightened, enlarged images, each of a single printed line cut from a photographed page, "
    "each preceded by a label \"Line k:\". Partial letters of neighbouring lines may show at the top and bottom "
    "edges: ignore them.\n\n"
    + PERLINE_RULES
    + "OUTPUT\n- `lines`: one entry per image, echoing its label number as line_index, with `text`: that image's "
    "printed line exactly as typed. Each entry must come from its own image alone: never move, repeat or "
    "complete words across images, and do not read the images as running prose. No Markdown.")

LINE_SYSTEM_CONTEXT = (
    "You are an expert transcriber of Persian (Farsi) typewritten and printed documents. "
    "You receive {n} straightened, enlarged images cut from a photographed page, each preceded by a label "
    "\"Line k:\". Each image shows three printed lines; the TARGET line is the one between the red bars at the "
    "left and right edges. The lines above and below are there only so you can see the context: never copy "
    "words from them.\n\n"
    + PERLINE_RULES
    + "OUTPUT\n- `lines`: one entry per image, echoing its label number as line_index, with `text`: that "
    "image's TARGET line exactly as typed. Each entry must come from its own target line alone. No Markdown.")


class LineText(BaseModel):
    text: str = Field(description="The printed line, exactly as typed.")


class IndexedLine(BaseModel):
    line_index: int = Field(description="The k from the image's 'Line k:' label.")
    text: str = Field(description="That image's printed line, exactly as typed.")


class IndexedLines(BaseModel):
    lines: list[IndexedLine]


PERLINE_ARMS = {   # arm: (model, mode, context)
    "pl_s1": (config.MODEL_STRONG, "single", False),
    "pl_s": (config.MODEL_STRONG, "batch", False),
    "pl_o": (MODEL_OPUS, "batch", False),
    "pl_o1": (MODEL_OPUS, "single", False),
    "pl_s_ctx": (config.MODEL_STRONG, "batch", True),
    "pl_o_ctx": (MODEL_OPUS, "batch", True),
}
OPUS_EFFORT = "medium"   # thinking cannot be disabled on Opus 5.5; effort is the control


def _model_kwargs(model: str) -> dict:
    return {"extra": {"output_config": {"effort": OPUS_EFFORT}}} if model == MODEL_OPUS else {}


def perline_requests(arm: str, pid: str, rep: int = 0) -> list[tuple[list[int], Request]]:
    """[(audit line indices covered, request)]."""
    model, mode, ctx = PERLINE_ARMS[arm]
    imgs = line_images(pid, context=ctx)
    idx = [i for i, im in enumerate(imgs) if im is not None]
    out = []
    if mode == "single":
        for i in idx:
            out.append(([i], Request(arm, f"{pid}:L{i}", model, LINE_SYSTEM_SINGLE,
                                     [imgs[i], "Transcribe this line following the system instructions."],
                                     LineText, max_tokens=2000, rep=rep, expected_out=150 if model != MODEL_OPUS else 600,
                                     **_model_kwargs(model))))
        return out
    content: list = []
    for k, i in enumerate(idx, 1):
        content += [f"Line {k}:", imgs[i]]
    content.append(f"Transcribe each of the {len(idx)} line images above following the system instructions.")
    system = (LINE_SYSTEM_CONTEXT if ctx else LINE_SYSTEM_BATCH).format(n=len(idx))
    out.append((idx, Request(arm, pid, model, system, content, IndexedLines, max_tokens=12000, rep=rep,
                             expected_out=2500 if model != MODEL_OPUS else 6000, **_model_kwargs(model))))
    return out


def perline_output(runner: "Runner", arm: str, pid: str, rep: int, call: bool = False) -> Optional[dict]:
    """Assemble per-line texts in audit order; None if anything is uncached
    (unless call=True, which pays for missing requests)."""
    n = len(load_audit(pid)["lines"])
    texts = [""] * n
    cost, usage = 0.0, {"input_tokens": 0, "output_tokens": 0}
    missing_idx = []
    for idx, req in perline_requests(arm, pid, rep):
        rec = runner.call(req) if call else runner.cached(req)
        if rec is None:
            return None
        cost += rec["cost"]
        for k2 in usage:
            usage[k2] += rec["usage"][k2]
        p = rec["parsed"]
        if "text" in p:
            texts[idx[0]] = p["text"]
        else:
            got = {e["line_index"]: e["text"] for e in p["lines"]}
            for k, i in enumerate(idx, 1):
                if k in got:
                    texts[i] = got[k]
                else:
                    missing_idx.append(i)
    return {"md": None, "lines": texts, "cost": cost, "usage": usage, "missing_lines": missing_idx}


def truth_tokens_wild(pid: str, wild: frozenset = frozenset()) -> list[Tok]:
    """Truth tokens with the given (non-blank) truth lines turned into wildcards."""
    lines = [l for l in truth_lines(pid) if l.strip()]
    lines = [" ".join(MARK_UNK for _ in l.split()) if i in wild else l for i, l in enumerate(lines)]
    return tokenize_lines(lines)


def per_line_errors(truth: list[Tok], hyp: list[Tok]) -> dict:
    """Lenient errors per truth line (insertions go to the line of the next
    truth token). Returns counts, perfect lines and concentration."""
    r, h, ops, events = lenient_alignment(truth, hyp)
    nlines = (r[-1].line + 1) if r else 0
    graded = {t.line for t in r if t.text != MARK_UNK}
    per = {L: 0 for L in sorted(graded)}
    for e in events:
        ri = min(e.ref, len(r) - 1)
        L = r[ri].line
        if L in per:
            per[L] += 1
    errs = sorted(per.values(), reverse=True)
    total = sum(errs)
    return {"lines": len(per), "perfect_lines": sum(1 for v in errs if v == 0), "errors": total,
            "worst5_share": (sum(errs[:5]) / total) if total else 0.0, "per_line": per}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_request(arm: str, pid: str, rep: int = 0) -> Request:
    slug, page = PAGES[pid]
    ut = user_text(slug, page)
    if arm == "current":
        return Request(arm, pid, config.MODEL_STRONG, llm.TRANSCRIBE_SYSTEM,
                       [whole_page_image(slug, page), ut], llm.PageTranscription, rep=rep)
    if arm == "t4":
        return Request(arm, pid, config.MODEL_STRONG, DIPLOMATIC_SYSTEM,
                       [whole_page_image(slug, page), ut], llm.PageTranscription, rep=rep)
    if arm == "lines":
        return Request(arm, pid, config.MODEL_STRONG, LINES_SYSTEM,
                       [whole_page_image(slug, page), ut], LineReading, rep=rep)
    if arm == "lines_haiku":
        return Request(arm, pid, config.MODEL_FAST, LINES_SYSTEM,
                       [whole_page_image(slug, page), ut], LineReading, rep=rep)
    if arm == "folded":
        return Request(arm, pid, config.MODEL_STRONG, FOLDED_SYSTEM,
                       [whole_page_image(slug, page), ut], PageWithLines, rep=rep, expected_out=3200)
    if arm in ("t1", "t1t4", "t1r", "t1r_think", "t1r_opus"):
        imgs, _ = band_images(slug, page)
        content: list = []
        for k, im in enumerate(imgs, 1):
            content += [f"Band {k}:", im]
        content.append(ut.replace("Transcribe this page", f"Transcribe the {len(imgs)} bands of this page"))
        sysf = {"t1": BANDS_CURRENT_SYSTEM, "t1t4": BANDS_DIPLOMATIC_SYSTEM}.get(arm, BANDS_READ_SYSTEM)
        if arm == "t1r_opus":
            return Request(arm, pid, MODEL_OPUS, sysf(len(imgs)), content, BandsReading, rep=rep,
                           max_tokens=12000, expected_out=4000, **_model_kwargs(MODEL_OPUS))
        if arm == "t1r_think":
            return Request(arm, pid, config.MODEL_STRONG, sysf(len(imgs)), content, BandsReading, rep=rep,
                           max_tokens=16000, expected_out=6000, thinking={"type": "adaptive"})
        return Request(arm, pid, config.MODEL_STRONG, sysf(len(imgs)), content, BandsReading, rep=rep,
                       expected_out=2600)
    raise KeyError(arm)


ARMS = ("current", "t1", "t4", "t1t4", "lines", "folded", "lines_haiku", "t1r", "t1r_think", "t1r_opus")


def output_of(rec: dict, arm: str) -> dict:
    """Normalise a cached response to {'md': str|None, 'lines': list|None, 'band_warnings': [...]}"""
    p = rec["parsed"]
    if arm in ("current", "t4"):
        return {"md": p["text_md"], "lines": None, "confidence": p.get("confidence"), "flags": p.get("flags")}
    if arm in ("lines", "lines_haiku"):
        return {"md": None, "lines": p["lines"]}
    if arm == "folded":
        return {"md": p["text_md"], "lines": p["printed_lines"], "confidence": p.get("confidence")}
    if arm in ("t1", "t1t4", "t1r", "t1r_think", "t1r_opus"):
        bands = sorted(p["bands"], key=lambda b: b["band_index"])
        lines, warns = stitch_bands([b["lines"] for b in bands])
        return {"md": None, "lines": lines, "band_warnings": warns,
                "band_line_counts": [len(b["lines"]) for b in bands]}
    raise KeyError(arm)


def hyp_tokens(out: dict, prefer: str = "md") -> list[Tok]:
    if prefer == "md" and out.get("md") is not None:
        return tokenize_md(out["md"])
    if out.get("lines") is not None:
        return tokenize_lines(out["lines"])
    return tokenize_md(out["md"])


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

def truth_lines(pid: str) -> list[str]:
    slug, page = PAGES[pid]
    if slug == HADPI:
        return (ROOT / "books" / slug / "truth" / f"{page:04d}.lines.txt").read_text(encoding="utf-8").split("\n")
    audit = json.loads((ROOT / "books" / slug / "truth" / f"{page:04d}.geometry.audit.json").read_text())
    return [" ".join(l["tokens"]) for l in audit["lines"]]


def truth_tokens(pid: str) -> list[Tok]:
    slug, page = PAGES[pid]
    if slug == HADPI:
        return tokenize_lines(truth_lines(pid))
    return tokenize_md((ROOT / "books" / slug / "text" / f"{page:04d}.md").read_text(encoding="utf-8"))


def stored_md(pid: str) -> str:
    slug, page = PAGES[pid]
    return (ROOT / "books" / slug / "text" / f"{page:04d}.md").read_text(encoding="utf-8")


def line_geometry(pid: str) -> dict:
    slug, page = PAGES[pid]
    audit = json.loads((ROOT / "books" / slug / "truth" / f"{page:04d}.geometry.audit.json").read_text())
    import statistics as st
    zoom = audit["image"]["zoom"]
    hs = [l["box"][3] - l["box"][1] for l in audit["lines"]]
    cys = [l["frame"]["cy"] for l in audit["lines"]]
    pitch = st.median([b - a for a, b in zip(cys, cys[1:])])
    return {"pitch_pt": pitch / zoom, "box_h_pt": st.median(hs) / zoom}


# ---------------------------------------------------------------------------
# Plan, smoke, run, score
# ---------------------------------------------------------------------------

HADPI_PIDS = ("h1", "h2")
ALL_PIDS = ("h1", "h2", "c23", "c109")

PLAN = [  # (arm, pages, reps)
    ("current", ALL_PIDS, (0,)), ("current", HADPI_PIDS, (1, 2)),
    ("t1", ALL_PIDS, (0,)),
    ("t4", ALL_PIDS, (0,)),
    ("t1t4", ALL_PIDS, (0,)), ("t1t4", HADPI_PIDS, (1, 2)),
    ("lines", ALL_PIDS, (0,)), ("lines", HADPI_PIDS, (1,)),
    ("folded", ALL_PIDS, (0,)),
    ("lines_haiku", ALL_PIDS, (0,)),
]


def cmd_plan(_args) -> None:
    total_exp = total_worst = 0.0
    rows = []
    for arm, pids, reps in PLAN:
        for pid in pids:
            for rep in reps:
                r = build_request(arm, pid, rep)
                total_exp += r.expected()
                total_worst += r.worst_case()
                imgs = [c for c in r.content if isinstance(c, Image)]
                rows.append(f"{arm:12s} {pid:5s} rep{rep}  imgs={len(imgs)} "
                            f"img_tok={sum(api_tokens(i, r.model) for i in imgs):6d} in_est={r.input_est():6d}"
                            f"  exp=${r.expected():.4f} worst=${r.worst_case():.4f}")
    print("\n".join(rows))
    print(f"\n{len(rows)} calls  expected ${total_exp:.3f}  worst-case ${total_worst:.3f}  "
          f"(plan limit ${PLAN_LIMIT_USD:.2f}, cap ${CAP_USD:.2f}); ledger so far ${Runner().ledger.total():.4f}")
    (OUT / "plan.txt").write_text("\n".join(rows) + f"\nexpected total ${total_exp:.3f}\n")


class _Ok(BaseModel):
    ok: bool


def cmd_smoke(_args) -> None:
    runner = Runner()
    req = Request("smoke", "-", config.MODEL_STRONG, "Answer with JSON only.",
                  ["Set ok to true."], _Ok, max_tokens=50, expected_out=10)
    rec = runner.call(req)
    print(json.dumps({k: rec[k] for k in ("parsed", "usage", "cost", "cache_hit")}, ensure_ascii=False))
    print(f"ledger total ${runner.ledger.total():.5f}")


def cmd_run(args) -> None:
    runner = Runner()
    pids = args.pages.split(",") if args.pages else ALL_PIDS
    for arm in args.arms:
        for pid in pids:
            req = build_request(arm, pid, args.rep)
            try:
                rec = runner.call(req)
            except (BudgetExceeded, AccessError) as exc:
                print(f"STOP: {type(exc).__name__}: {exc}")
                return
            print(f"{arm:12s} {pid:5s} rep{args.rep} hit={rec['cache_hit']} "
                  f"in={rec['usage']['input_tokens']} (est {rec['input_est']}) out={rec['usage']['output_tokens']} "
                  f"${rec['cost']:.4f} stop={rec.get('stop_reason')}  ledger ${runner.ledger.total():.4f}")


def cmd_runpl(args) -> None:
    runner = Runner()
    pids = args.pages.split(",") if args.pages else ALL_PIDS
    for arm in args.arms:
        for pid in pids:
            before = runner.ledger.total()
            try:
                out = perline_output(runner, arm, pid, args.rep, call=True)
            except (BudgetExceeded, AccessError, RuntimeError) as exc:
                print(f"STOP: {type(exc).__name__}: {exc}")
                return
            print(f"{arm:9s} {pid:5s} rep{args.rep} in={out['usage']['input_tokens']} out={out['usage']['output_tokens']} "
                  f"${out['cost']:.4f} paid_now=${runner.ledger.total() - before:.4f} missing={out['missing_lines']} "
                  f"ledger ${runner.ledger.total():.4f}", flush=True)


def _cached_runs(runner: Runner, arm: str, pid: str) -> list[tuple[int, dict]]:
    out = []
    for rep in range(6):
        rec = runner.cached(build_request(arm, pid, rep))
        if rec:
            out.append((rep, rec))
    return out


def effective_px(pid: str, arm: str) -> dict:
    slug, page = PAGES[pid]
    g = line_geometry(pid)
    if arm in ("t1", "t1t4", "t1r", "t1r_think", "t1r_opus"):
        imgs, meta = band_images(slug, page)
        scale = min(api_size(i, config.MODEL_STRONG)[0] / i.width for i in imgs)
        ppt = meta["px_per_pt"]
    else:
        img = whole_page_image(slug, page)
        model = config.MODEL_FAST if arm == "lines_haiku" else config.MODEL_STRONG
        w, _ = api_size(img, model)
        import fitz
        doc = fitz.open(str(ROOT / "books" / slug / "source.pdf"))
        pg = doc[page - 1]
        ppt = img.height / max(pg.rect.width, pg.rect.height) if img.height >= img.width else img.width / max(pg.rect.width, pg.rect.height)
        scale = w / img.width
    return {"pitch_px": round(g["pitch_pt"] * ppt * scale, 1), "line_box_px": round(g["box_h_pt"] * ppt * scale, 1),
            "src_px_per_pt": round(ppt, 3), "api_scale": round(scale, 3)}


def score_output(pid: str, out: dict, prefer: str = "md") -> dict:
    truth = truth_tokens(pid)
    hyp = hyp_tokens(out, prefer)
    res = {f: wer(truth, hyp, f) for f in FOLDINGS}
    inv = invented_words(truth, hyp)
    res["invented"] = len(inv)
    _, h, _, _ = lenient_alignment(truth, hyp)
    res["invented_in_markers"] = sum(1 for i in inv if h[i].marker)
    res["markers"] = marker_stats(truth, hyp)
    if out.get("lines") is not None and prefer != "md" or (out.get("md") is None and out.get("lines")):
        res["linebreaks"] = line_stats(truth_lines(pid), out["lines"])
    return res


def cmd_score(_args) -> None:
    runner = Runner()
    results = {"pages": {}, "ledger_total": runner.ledger.total(), "folding_doc": __doc__}
    detail_lines = []
    for pid in ALL_PIDS:
        pr = {}
        if PAGES[pid][0] == HADPI:
            pr["stored"] = [{"rep": "stored", **score_output(pid, {"md": stored_md(pid)})}]
        for arm in ARMS:
            runs = _cached_runs(runner, arm, pid)
            if not runs:
                continue
            rows = []
            for rep, rec in runs:
                out = output_of(rec, arm)
                row = {"rep": rep, "cost": rec["cost"], "usage": rec["usage"],
                       **score_output(pid, out, "md" if out.get("md") is not None else "lines")}
                if arm == "folded":
                    row["printed_lines_score"] = score_output(pid, {"md": None, "lines": out["lines"]}, "lines")
                for k in ("band_warnings", "band_line_counts", "confidence"):
                    if k in out:
                        row[k] = out[k]
                row["effective_px"] = effective_px(pid, arm)
                row["api_images"] = rec.get("images")
                rows.append(row)
                detail_lines.append(_detail(pid, arm, rep, out))
            pr[arm] = rows
        results["pages"][pid] = pr
    results["t2"] = score_t2(runner)
    results["votes"] = score_votes(runner)
    results["perline"] = score_perline(runner)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (OUT / "errors_detail.txt").write_text("\n".join(detail_lines), encoding="utf-8")
    text = summary_table(results) + "\n\n" + perline_table(results["perline"])
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    pl_detail = []
    for pid in ALL_PIDS:
        for arm in ("pl_o", "pl_s", "pl_o_ctx", "t1r_opus"):
            for rep in range(3):
                got = _any_tokens(runner, pid, arm, rep)
                if got:
                    pl_detail.append(_detail_tokens(pid, arm, rep, got[0]))
    (OUT / "errors_detail_perline.txt").write_text("\n".join(pl_detail), encoding="utf-8")


T2_PAIRS = [  # (name, primary (arm, rep), secondary (arm, rep))
    ("t1t4 vs t1t4(rep1) same model+res", ("t1t4", 0), ("t1t4", 1)),
    ("t1t4 vs lines (whole-page res)", ("t1t4", 0), ("lines", 0)),
    ("t1t4 vs lines_haiku (other model)", ("t1t4", 0), ("lines_haiku", 0)),
    ("t1t4 vs current text_md", ("t1t4", 0), ("current", 0)),
    ("t1t4 vs t1 (current rules, bands)", ("t1t4", 0), ("t1", 0)),
    ("current vs lines (D2 setup)", ("current", 0), ("lines", 0)),
    ("current vs folded.printed_lines", ("current", 0), ("folded_lines", 0)),
    ("t1 vs t1t4", ("t1", 0), ("t1t4", 0)),
    ("t4 vs lines", ("t4", 0), ("lines", 0)),
    ("t1r vs t1r(rep1) same model+res", ("t1r", 0), ("t1r", 1)),
    ("t1r vs lines (whole-page res)", ("t1r", 0), ("lines", 0)),
    ("t1r vs lines(rep1) (whole-page res)", ("t1r", 0), ("lines", 1)),
    ("t1r vs current text_md (other prompt+res)", ("t1r", 0), ("current", 0)),
    ("t1r vs t1 (current rules, bands)", ("t1r", 0), ("t1", 0)),
    ("t1r vs t1r_think", ("t1r", 0), ("t1r_think", 0)),
    ("t1r vs lines_haiku (other model)", ("t1r", 0), ("lines_haiku", 0)),
    ("t1 vs t1(rep1)", ("t1", 0), ("t1", 1)),
    ("t1 vs current", ("t1", 0), ("current", 0)),
    ("current vs current(rep1)", ("current", 0), ("current", 1)),
]


VOTES = [  # (name, primary, s1, s2)
    ("vote t1r0|t1r1|current0", ("t1r", 0), ("t1r", 1), ("current", 0)),
    ("vote t1r0|t1r1|t1r2", ("t1r", 0), ("t1r", 1), ("t1r", 2)),
    ("vote t1r0|current0|current1", ("t1r", 0), ("current", 0), ("current", 1)),
    ("vote t1r0|t1-0|current0", ("t1r", 0), ("t1", 0), ("current", 0)),
    ("vote current0|current1|current2", ("current", 0), ("current", 1), ("current", 2)),
]


def score_votes(runner) -> list[dict]:
    rows = []
    for name, a, b, c in VOTES:
        for pid in ALL_PIDS:
            P, S1, S2 = (_toks_for(runner, pid, *x) for x in (a, b, c))
            if P is None or S1 is None or S2 is None:
                continue
            truth = truth_tokens(pid)
            voted, unsure = vote3(P, S1, S2)
            row = {"vote": name, "page": pid, **{f: wer(truth, voted, f) for f in ("ortho", "lenient")},
                   "invented": len(invented_words(truth, voted))}
            r, h, ops, ev = lenient_alignment(truth, voted)
            caught = sum(1 for e in ev if _covered(e, unsure, set()))
            row.update({"errors": len(ev), "caught_by_unsure": caught,
                        "recall": caught / len(ev) if ev else 1.0,
                        "flag_rate": len(unsure) / len(voted) if voted else 0.0, "flagged": len(unsure)})
            rows.append(row)
    return rows


def _toks_for(runner, pid, arm, rep):
    real = "folded" if arm == "folded_lines" else arm
    rec = runner.cached(build_request(real, pid, rep))
    if rec is None:
        return None
    out = output_of(rec, real)
    if arm == "folded_lines":
        return tokenize_lines(out["lines"])
    return hyp_tokens(out, "md")


def score_t2(runner) -> list[dict]:
    rows = []
    for name, (pa, pr), (sa, sr) in T2_PAIRS:
        for pid in ALL_PIDS:
            P, S = _toks_for(runner, pid, pa, pr), _toks_for(runner, pid, sa, sr)
            if P is None or S is None:
                continue
            rows.append({"pair": name, "page": pid, **t2_stats(truth_tokens(pid), P, S)})
    return rows


PL_COMPARE = ("pl_o", "pl_o1", "pl_s", "pl_s1", "pl_o_ctx", "pl_s_ctx", "t1r_opus", "t1r", "current")
PL_T2 = [
    ("pl_o vs t1r (best Sonnet whole-page arm)", ("pl_o", 0), ("t1r", 0)),
    ("pl_o vs current", ("pl_o", 0), ("current", 0)),
    ("pl_o vs pl_o(rep1)", ("pl_o", 0), ("pl_o", 1)),
    ("pl_o vs pl_s (same crops, Sonnet)", ("pl_o", 0), ("pl_s", 0)),
    ("pl_o vs t1r_opus (same model, bands)", ("pl_o", 0), ("t1r_opus", 0)),
    ("t1r_opus vs t1r", ("t1r_opus", 0), ("t1r", 0)),
    ("t1r vs current (earlier best pair)", ("t1r", 0), ("current", 0)),
]


def _any_tokens(runner, pid, arm, rep):
    if arm in PERLINE_ARMS:
        o = perline_output(runner, arm, pid, rep)
        return None if o is None else (tokenize_lines(o["lines"]), o)
    rec = runner.cached(build_request(arm, pid, rep))
    if rec is None:
        return None
    o = output_of(rec, arm)
    return hyp_tokens(o, "md" if o.get("md") is not None else "lines"), {**o, "cost": rec["cost"], "usage": rec["usage"]}


def score_perline(runner) -> dict:
    """Per-line follow-up: all arms scored against truth with EXCLUDE_LINES as
    wildcards (so whole-page and per-line arms are compared on the same words)."""
    rows, t2 = [], []
    for pid in ALL_PIDS:
        truth = truth_tokens_wild(pid, frozenset(EXCLUDE_LINES.get(pid, ())))
        for arm in PL_COMPARE:
            for rep in range(4):
                got = _any_tokens(runner, pid, arm, rep)
                if got is None:
                    continue
                hyp, o = got
                row = {"page": pid, "arm": arm, "rep": rep, "cost": o.get("cost"), "usage": o.get("usage"),
                       **{f: wer(truth, hyp, f) for f in FOLDINGS},
                       "invented": len(invented_words(truth, hyp)), "markers": marker_stats(truth, hyp),
                       "per_line": per_line_errors(truth, hyp)}
                if o.get("lines") is not None:
                    row["linebreaks"] = line_stats([l for l in truth_lines(pid) if l.strip()], o["lines"])
                if o.get("missing_lines"):
                    row["missing_lines"] = o["missing_lines"]
                rows.append(row)
        for name, (pa, pr), (sa, sr) in PL_T2:
            P, S = _any_tokens(runner, pid, pa, pr), _any_tokens(runner, pid, sa, sr)
            if P is None or S is None:
                continue
            t2.append({"pair": name, "page": pid, **t2_stats(truth, P[0], S[0])})
    return {"rows": rows, "t2": t2}


def perline_table(pl: dict) -> str:
    out = ["PER-LINE FOLLOW-UP (truth with excluded lines as wildcards: h1 line 0)",
           "page  arm        rep strict  lenient (err/n)   inv  unk guess(ok/wrong) in-mark perfect-lines worst5  cost"]
    for r in pl["rows"]:
        m, p = r["markers"], r["per_line"]
        cost = f"${r['cost']:.4f}" if r.get("cost") is not None else "-"
        out.append(f"{r['page']:5s} {r['arm']:10s} {r['rep']}  {r['strict']['wer']:.3f}   {r['lenient']['wer']:.3f} "
                   f"({r['lenient']['errors']:3d}/{r['lenient']['n_truth']})  {r['invented']:3d}  {m['markers_unk']:3d} "
                   f"{m['markers_guess']:2d}({m['guess_correct']}/{m['guess_wrong']})      {m['errors_in_markers']:3d}   "
                   f"{p['perfect_lines']:2d}/{p['lines']:2d}      {p['worst5_share']:.2f}   {cost}")
    out.append("")
    out.append("T2 pair (per-line follow-up)                 page  recall (caught/err)  flag rate")
    for t in pl["t2"]:
        out.append(f"{t['pair']:44s} {t['page']:5s} {t['recall']:.2f} ({t['caught']}/{t['errors']})   "
                   f"{t['flag_rate']:.3f} ({t['flagged_words']}+{t['flagged_gaps']}/{t['primary_words']})")
    return "\n".join(out)


def _detail_tokens(pid, arm, rep, hyp) -> str:
    truth = truth_tokens_wild(pid, frozenset(EXCLUDE_LINES.get(pid, ())))
    r, h, ops, ev = lenient_alignment(truth, hyp)
    lines = [f"=== {pid} {arm} rep{rep}: {len(ev)} lenient errors"]
    ref_pos = 0
    for op in ops:
        if op.ref:
            ref_pos = op.ref[0]
        if op.kind in ("match", "join", "split"):
            continue
        rt = " ".join(r[i].text for i in op.ref)
        ht = " ".join(h[j].text + ("*" if h[j].marker else "") for j in op.hyp)
        tl = r[min(ref_pos, len(r) - 1)].line + 1
        lines.append(f"  {op.kind:4s} truthline {tl}: [{rt}] -> [{ht}]")
    return "\n".join(lines)


def _detail(pid, arm, rep, out) -> str:
    truth = truth_tokens(pid)
    hyp = hyp_tokens(out, "md" if out.get("md") is not None else "lines")
    r, h, ops, ev = lenient_alignment(truth, hyp)
    lines = [f"=== {pid} {arm} rep{rep}: {len(ev)} lenient errors"]
    for op in ops:
        if op.kind in ("match", "join", "split"):
            continue
        rt = " ".join(r[i].text for i in op.ref)
        ht = " ".join(h[j].text + ("*" if h[j].marker else "") for j in op.hyp)
        tl = r[op.ref[0]].line + 1 if op.ref else "-"
        lines.append(f"  {op.kind:4s} truthline {tl}: [{rt}] -> [{ht}]")
    return "\n".join(lines)


def summary_table(res: dict) -> str:
    out = ["page  arm          rep  strict  ortho  lenient  err(len) inv  unk guess(ok/wrong) silent  lines(t/h, ok%)  cost   px/pitch"]
    for pid, arms in res["pages"].items():
        for arm, rows in arms.items():
            for r in rows:
                m = r["markers"]
                lb = r.get("linebreaks") or (r.get("printed_lines_score") or {}).get("linebreaks")
                lbs = f"{lb['truth_lines']}/{lb['hyp_lines']}, {100 * lb['break_correct_share']:.0f}%" if lb else "-"
                ep = r.get("effective_px", {}).get("pitch_px", "-")
                out.append(f"{pid:5s} {arm:12s} {str(r['rep']):4s} {r['strict']['wer']:.3f}  {r['ortho']['wer']:.3f}  "
                           f"{r['lenient']['wer']:.3f}   {r['lenient']['errors']:3d}/{r['lenient']['n_truth']:3d}  "
                           f"{r['invented']:3d}  {m['markers_unk']:3d} {m['markers_guess']:3d}({m['guess_correct']}/{m['guess_wrong']})"
                           f"        {m['silent_errors']:3d}    {lbs:16s} ${r.get('cost', 0):.4f} {ep}")
                if arm == "folded":
                    p = r["printed_lines_score"]
                    out.append(f"{pid:5s}  .printed_lines    {p['strict']['wer']:.3f}  {p['ortho']['wer']:.3f}  "
                               f"{p['lenient']['wer']:.3f}   {p['lenient']['errors']:3d}/{p['lenient']['n_truth']:3d}  "
                               f"{p['invented']:3d}  {p['markers']['markers_unk']:3d} {p['markers']['markers_guess']:3d}")
    out.append("")
    out.append("T2 pair                                   page  recall (caught/err)  flag rate (words+gaps/primary)")
    for t in res["t2"]:
        out.append(f"{t['pair']:42s} {t['page']:5s} {t['recall']:.2f} ({t['caught']}/{t['errors']})        "
                   f"{t['flag_rate']:.3f} ({t['flagged_words']}+{t['flagged_gaps']}/{t['primary_words']})")
    out.append("")
    out.append("2-of-3 vote                          page  ortho  lenient (err/n)  inv  unsure-flag recall  flag rate")
    for v in res.get("votes", []):
        out.append(f"{v['vote']:36s} {v['page']:5s} {v['ortho']['wer']:.3f}  {v['lenient']['wer']:.3f} "
                   f"({v['lenient']['errors']}/{v['lenient']['n_truth']})  {v['invented']:3d}  "
                   f"{v['recall']:.2f} ({v['caught_by_unsure']}/{v['errors']})  {v['flag_rate']:.3f} ({v['flagged']})")
    out.append(f"\nledger total ${res['ledger_total']:.4f}")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("smoke")
    pr = sub.add_parser("run")
    pr.add_argument("arms", nargs="+", choices=ARMS)
    pr.add_argument("--pages", default=None, help="comma list of " + ",".join(PAGES))
    pr.add_argument("--rep", type=int, default=0)
    pl = sub.add_parser("runpl")
    pl.add_argument("arms", nargs="+", choices=list(PERLINE_ARMS))
    pl.add_argument("--pages", default=None)
    pl.add_argument("--rep", type=int, default=0)
    sub.add_parser("score")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"plan": cmd_plan, "smoke": cmd_smoke, "run": cmd_run, "runpl": cmd_runpl, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
