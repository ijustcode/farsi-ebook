#!/usr/bin/env python3
"""Arm T: page-trained template aligner (plan §4.3, Step 4 continuation).

Research code (numpy only, no API, deterministic). Learns what every letter
form of the page looks like *from the page itself*, by forced alignment of the
known line text (embedded training, hard EM / Viterbi training), and reads the
word boxes off the final alignment.

Model (one left-to-right HMM per line, states tied across the page):

* **Features.** Each line's own ink (the components arm G assigns to it, so
  neighbouring lines and masked stamps are absent) is rasterised in the line
  frame: columns along the reading direction (step ``COL_SW`` x stroke width),
  ``ROWS`` rows from ``V_UP`` x glyph height above the line's baseline peak to
  ``V_DN`` x glyph height below it. Cell value = ink share, lightly blurred.
  Binarised ink (L1's adaptive threshold) normalises ink density and contrast;
  the blur absorbs small strike-height jitter; the per-line frame (audited
  outline or L1 shear) removes slope and curvature. Scale is per page
  (typewriter pitch is constant; perspective changes it slowly, which the
  stay/skip transitions absorb).
* **Units** are the positional letter forms of the known text:
  (letter, joins previous, joins next), with lam-alef as one ligature unit,
  digits/Latin runs in visual order, punctuation as its own unit. A unit is a
  chain of ``n_k`` column states (self-loop "stay", "advance", "skip one"),
  each with a mean column vector; a shared per-row variance.
* **Gaps.** Between consecutive units a gap block: optional blank (Ba), an
  optional **not-text** state J (stray ink, plan §13 O1: a broad, flat
  emission, entry cost ``J_ENTER``) and optional blank (Bb). Gap usage and
  blank lengths are learned per boundary type (inside a joined piece / after a
  non-joining letter / between words / line edge). In-word and word gaps get
  separate parameters but nothing forces them to differ: on a typewriter they
  coincide (§13 O2) and the letter shapes decide. J is not allowed inside a
  joined piece.
* **Wrong text (§4.4).** A whole word may be bypassed (jump from the J before
  it to the J after it, ``DEL_WORD``) so its ink goes to the not-text state:
  a word whose letters do not match the image is then not placed, and its
  neighbours keep their own evidence. A word's mean emission cost against its
  own templates is its *fit*; a high fit z (robust, page-relative) also
  blocks ``located``.
* **Training.** Initial word intervals come from arm G's best alignment (or
  arm R's warp, or a flat proportional start); letters inside a word are
  split by G's learned width table; each unit's template is the mean of its
  resampled occurrences. Then: align every line (Viterbi) -> re-estimate means
  (shrunk toward the mean text column with ``KAPPA`` columns of prior
  weight), per-row variance, per-unit transition probabilities (shrunk to the
  page's), gap parameters -> align ..., ``ITERS`` rounds or until no
  alignment changes. Unit lengths ``n_k`` are re-estimated once, after the
  first alignment, then fixed (so the objective is a fixed-structure hard-EM
  objective from round 2 on; it is recorded per round). Words with a bad fit
  or bypassed do not feed re-estimation.
* **Confidence.** Forward and backward min-sums give min-marginals: the best
  total cost with a word's start or end more than ``POS_TOL`` letter from the
  chosen one, or with the word bypassed. ``margin`` = that minus the best.
  ``located`` iff margin >= ``LOC_MARGIN`` and fit z <= ``FIT_Z``; else
  ``line`` (the line box) when the line is trusted, ``unresolved`` otherwise.
  Never an approximate guess box (owner decision D1).

Also here: **Rc**, arm R with the line rendered from glyph templates cut out
of the page (from words where R and G already agree) instead of a font.

Every length is relative to measured page quantities (stroke width ``sw``,
glyph height ``hc``, G's letter unit ``unit``).
"""
from __future__ import annotations

import math
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np

import word_placer as WP

VERSION = "word_placer/T-1"

TP = dict(
    COL_SW=0.5,        # column step = this x stroke width
    V_UP=1.25,         # rows from baseline peak - V_UP x hc ...
    V_DN=0.85,         # ... to + V_DN x hc
    ROWS=16,
    PAD_HC=0.5,        # blank columns added at each line end (x hc)
    KAPPA=2.0,         # template mean prior weight (columns) toward the mean text column
    ALPHA=10.0,        # per-unit transition prior weight (counts) toward the page's
    VAR_FLOOR=0.004,   # per-row variance floor (cell values are 0..1)
    JUNK_VAR=6.0,      # not-text emission: zero-mean, this x the row variance
    J_ENTER=8.0,       # entering the not-text state (nats)
    J_EDGE=0.5,        # ... times this at a line end
    DEL_WORD=12.0,     # bypass a whole word (its ink to not-text)
    ITERS=6,           # re-estimation rounds after the initial templates
    N_MIN=2,           # unit length bounds (states)
    N_MAX_UNIT=3.0,    # ... at most this x letter unit
    P_STAY0=0.15, P_SKIP0=0.10,     # initial transitions inside a unit
    P_BLANK0=(0.10, 0.6, 0.8, 0.5),  # initial P(blank) per gap type: join, piece, word, edge
    P_BSTAY0=0.7,
    POS_TOL=0.5,       # alternative edge positions closer than this x letter unit are 'the same'
    LOC_MARGIN=8.0,    # located: margin >= this (nats) ...
    FIT_Z=6.0,         # ... and word fit z <= this ...
    OVER_MAX=1.0,      # ... and the snapped box reaches at most this x letter beyond the aligned interval
    LEARN_Z=3.0,       # words with fit z above this do not feed re-estimation
    AGREE_TOL=0.5,     # agreement arms: both edges within this x letter unit
)

JOIN, PIECE, WORD, EDGE = 0, 1, 2, 3
LAM_ALEF = set("اآأإ")
INF = np.float32(1e30)


# ---------------------------------------------------------------------------
# Units of the known text
# ---------------------------------------------------------------------------
def token_units(token: str) -> list:
    """Units of one token in reading order: list of (key, btype_after) where
    key = (letter key, joins previous, joins next); btype_after is the gap
    type to the next unit of the same token (JOIN or PIECE)."""
    chars = [c for c in token if not unicodedata.category(c).startswith("M") and c != WP.TATWEEL]
    out = []
    prev_j = False
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in WP.ZW:
            prev_j = False
            i += 1
            continue
        if not WP.is_arabic_letter(c):
            j = i
            while j < len(chars) and not WP.is_arabic_letter(chars[j]) and chars[j] not in WP.ZW:
                j += 1
            run = chars[i:j]
            if any(ch.isdigit() or (ch.isascii() and ch.isalpha()) for ch in run):
                run = run[::-1]           # left-to-right run: visual order is reversed in s
            for ch in run:
                out.append([(WP.char_key(ch), 0, 0), PIECE])
            prev_j = False
            i = j
            continue
        key, n = WP.char_key(c), 1
        if c == "ل" and i + 1 < len(chars) and chars[i + 1] in LAM_ALEF:
            key, n = "لا", 2
        last = chars[i + n - 1]
        nxt = chars[i + n] if i + n < len(chars) else None
        joins = (key != "لا" and last not in WP.NONJOIN and nxt is not None and nxt not in WP.ZW
                 and WP.is_arabic_letter(nxt))
        out.append([(key, int(prev_j), int(joins)), JOIN if joins else PIECE])
        prev_j = joins
        i += n
    if not out:
        out = [[("?", 0, 0), PIECE]]
    return [tuple(x) for x in out]


def g_width(gmodel, key) -> float:
    """Width hint (px) of a unit from arm G's learned table."""
    k, _jp, jn = key
    if k == "لا":
        return gmodel.width(("ل", "J")) + gmodel.width(("ا", "E"))
    return gmodel.width((k, "J" if jn else "E"))


# ---------------------------------------------------------------------------
# Line features
# ---------------------------------------------------------------------------
@dataclass
class TLine:
    li: object                  # WP.LineInk
    tokens: list
    order: list                 # visual word -> logical token
    X: np.ndarray               # T x R
    s_lo: float
    cs: float
    ukeys: list                 # per unit (visual order)
    uword: list                 # per unit: visual word index
    btype: list                 # per unit: gap type after it (last unit: EDGE)
    trusted: bool = True
    tag: str = ""

    @property
    def T(self):
        return self.X.shape[0]

    @property
    def nwords(self):
        return len(self.order)


def _blur(X):
    k = np.array([0.25, 0.5, 0.25])
    Y = X.copy()
    Y[1:-1] = k[0] * X[:-2] + k[1] * X[1:-1] + k[2] * X[2:]
    Z = Y.copy()
    Z[:, 1:-1] = k[0] * Y[:, :-2] + k[1] * Y[:, 1:-1] + k[2] * Y[:, 2:]
    return Z


def line_features(li, hc: float, sw: float, cs: float | None = None):
    cs = cs or max(1.0, TP["COL_SW"] * sw)
    R = TP["ROWS"]
    if not len(li.u):
        return np.zeros((0, R)), 0.0, cs
    s = li.uref - li.u
    v = li.v - li.base_v
    v_lo, v_hi = -TP["V_UP"] * hc, TP["V_DN"] * hc
    rh = (v_hi - v_lo) / R
    pad = TP["PAD_HC"] * hc
    s_lo = float(s.min()) - pad
    T = int(math.ceil((float(s.max()) + pad - s_lo) / cs)) + 1
    sel = (v >= v_lo) & (v < v_hi)
    ci = np.clip(((s[sel] - s_lo) / cs).astype(int), 0, T - 1)
    ri = np.clip(((v[sel] - v_lo) / rh).astype(int), 0, R - 1)
    X = np.bincount(ci * R + ri, minlength=T * R).reshape(T, R).astype(np.float64) / (cs * rh)
    return _blur(np.clip(X, 0.0, 1.0)), s_lo, cs


def make_line(li, tokens, hc, sw, trusted=True, tag="", cs=None) -> TLine:
    X, s_lo, cs = line_features(li, hc, sw, cs)
    order = WP.visual_order(list(tokens))
    uk, uw, bt = [], [], []
    for w, ti in enumerate(order):
        us = token_units(tokens[ti])
        for k, (key, b) in enumerate(us):
            uk.append(key)
            uw.append(w)
            bt.append(b if k < len(us) - 1 else WORD)
    if bt:
        bt[-1] = EDGE
    return TLine(li, list(tokens), order, X, s_lo, cs, uk, uw, bt, trusted, tag)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _resample(seg: np.ndarray, n: int) -> np.ndarray:
    m = len(seg)
    if m == n:
        return seg.copy()
    if m == 1:
        return np.repeat(seg, n, 0)
    x = (np.arange(n) + 0.5) * m / n - 0.5
    x = np.clip(x, 0, m - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0)[:, None]
    return seg[i0] * (1 - f) + seg[i1] * f


@dataclass
class TModel:
    R: int
    tmpl: dict                       # key -> (n x R) mean
    occ: dict                        # key -> occurrences used
    prior: np.ndarray                # mean text column
    var: np.ndarray                  # per-row variance
    trans: dict                      # key -> (p_stay, p_adv, p_skip)
    trans_g: tuple
    gap: dict                        # type -> [p_blank, p_bstay]
    n_hint: dict = field(default_factory=dict)   # key -> n for unseen units
    cs: float = 1.0
    history: list = field(default_factory=list)

    def n_of(self, key):
        if key in self.tmpl:
            return len(self.tmpl[key])
        return self.n_hint.get(key, max(TP["N_MIN"], 4))

    def mean_of(self, key):
        if key in self.tmpl:
            return self.tmpl[key]
        return np.repeat(self.prior[None, :], self.n_of(key), 0)

    def tr(self, key):
        return self.trans.get(key, self.trans_g)


def initial_model(lines: list, intervals: list, gwidth, unit_px: float) -> TModel:
    """lines: [TLine]; intervals: per line, per visual word (s0, s1) or None
    (initial word positions); gwidth(key) -> px width hint."""
    R = TP["ROWS"]
    cs = lines[0].cs if lines else 1.0
    segs: dict = {}
    textcols = []
    for L, iv in zip(lines, intervals):
        if iv is None or L.T == 0:
            continue
        for w in range(L.nwords):
            if iv[w] is None:
                continue
            us = [k for k in range(len(L.ukeys)) if L.uword[k] == w]
            a = (iv[w][0] - L.s_lo) / L.cs
            b = (iv[w][1] - L.s_lo) / L.cs
            if b - a < len(us):
                continue
            wts = np.array([max(1.0, gwidth(L.ukeys[k])) for k in us])
            edges = a + np.r_[0.0, np.cumsum(wts)] / wts.sum() * (b - a)
            for k, e0, e1 in zip(us, edges[:-1], edges[1:]):
                t0, t1 = int(round(e0)), max(int(round(e0)) + 1, int(round(e1)))
                t0, t1 = max(0, t0), min(L.T, t1)
                if t1 > t0:
                    segs.setdefault(L.ukeys[k], []).append(L.X[t0:t1])
                    textcols.append(L.X[t0:t1])
    allc = np.concatenate(textcols) if textcols else np.concatenate([L.X for L in lines if L.T])
    prior = allc.mean(0)
    var = np.maximum(TP["VAR_FLOOR"], 0.5 * allc.var(0))
    nmax = max(TP["N_MIN"] + 1, int(round(TP["N_MAX_UNIT"] * unit_px / cs)))
    tmpl, occ, n_hint = {}, {}, {}
    keys = sorted({k for L in lines for k in L.ukeys}, key=str)
    for key in keys:
        n_hint[key] = int(np.clip(round(gwidth(key) / cs), TP["N_MIN"], nmax))
    for key, ss in segs.items():
        n = int(np.clip(round(float(np.median([len(x) for x in ss]))), TP["N_MIN"], nmax))
        acc = np.zeros((n, R))
        for x in ss:
            acc += _resample(x, n)
        tmpl[key] = (acc + TP["KAPPA"] * prior) / (len(ss) + TP["KAPPA"])
        occ[key] = len(ss)
    ps, pk = TP["P_STAY0"], TP["P_SKIP0"]
    gap = {t: [TP["P_BLANK0"][t], TP["P_BSTAY0"]] for t in (JOIN, PIECE, WORD, EDGE)}
    return TModel(R, tmpl, occ, prior, var, {}, (ps, 1 - ps - pk, pk), gap, n_hint, cs)


# ---------------------------------------------------------------------------
# State graph of one line
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    S: int
    M: np.ndarray        # S x R means (blank / junk rows zero)
    junk: np.ndarray     # S bool
    unit_state: np.ndarray   # S: unit index or -1
    Ctr: np.ndarray      # 5 x S: cost of entering s from s-k
    init: np.ndarray
    endc: np.ndarray
    jumps: list          # (src, dst, cost, word)
    first: list          # per unit: first state
    last: list           # per unit: last state
    blocks: list         # per block b (before unit b; b = U is the end): (Ba, J, Bb)


def _nl(p):
    return -math.log(max(1e-6, p))


def build_graph(L: TLine, m: TModel) -> Graph:
    U = len(L.ukeys)
    R = m.R
    sizes = [m.n_of(k) for k in L.ukeys]
    S = 3 * (U + 1) + sum(sizes)
    M = np.zeros((S, R))
    junk = np.zeros(S, bool)
    ust = np.full(S, -1)
    Ctr = np.full((5, S), float(INF))
    init = np.full(S, float(INF))
    endc = np.full(S, float(INF))
    first, last, blocks = [], [], []
    JE = TP["J_ENTER"]

    def btype_of_block(b):
        return EDGE if b == 0 or b == U else L.btype[b - 1]

    pos = 0
    prev_adv = 0.0
    for b in range(U + 1):
        t = btype_of_block(b)
        pb, pbs = m.gap[t]
        stay, ex = _nl(pbs), _nl(1 - pbs)
        Ba, J, Bb = pos, pos + 1, pos + 2
        blocks.append((Ba, J, Bb))
        junk[J] = True
        je = JE * (TP["J_EDGE"] if t == EDGE else 1.0)
        jok = t != JOIN
        Ctr[0, Ba] = stay
        Ctr[0, Bb] = stay
        Ctr[0, J] = 0.0
        Ctr[1, Bb] = 0.0 if jok else INF                    # J -> Bb
        if jok:
            Ctr[1, J] = ex + je                             # Ba -> J
        if b > 0:
            Ctr[1, Ba] = prev_adv + _nl(pb)                 # last of unit b-1 -> Ba
            if jok:
                Ctr[2, J] = prev_adv + je                   # last -> J
        else:
            init[Ba] = 0.0
            init[J] = je
        if b == U:
            endc[Ba] = endc[J] = endc[Bb] = 0.0
            if U:
                endc[last[-1]] = 0.0
            break
        pos += 3
        key = L.ukeys[b]
        n = sizes[b]
        p_s, p_a, p_k = m.tr(key)
        f = pos
        first.append(f)
        last.append(f + n - 1)
        M[f:f + n] = m.mean_of(key)
        ust[f:f + n] = b
        Ctr[0, f:f + n] = _nl(p_s)
        if n > 1:
            Ctr[1, f + 1:f + n] = _nl(p_a)
        if n > 2:
            Ctr[2, f + 2:f + n] = _nl(p_k)
        Ctr[1, f] = ex if jok else INF                      # Bb -> first
        Ctr[2, f] = 0.0 if jok else INF                     # J -> first
        Ctr[3, f] = ex                                      # Ba -> first
        if b > 0:
            Ctr[4, f] = prev_adv + _nl(1 - pb)              # last of previous unit -> first (no gap)
        else:
            init[f] = 0.0
        prev_adv = _nl(p_a)
        pos += n
    # whole-word bypass: J before the word's first unit -> J after its last
    jumps = []
    for w in range(L.nwords):
        us = [k for k in range(U) if L.uword[k] == w]
        if not us:
            continue
        jumps.append((blocks[us[0]][1], blocks[us[-1] + 1][1], TP["DEL_WORD"], w))
    return Graph(S, M, junk, ust, Ctr.astype(np.float32), init.astype(np.float32), endc.astype(np.float32),
                 jumps, first, last, blocks)


def emissions(L: TLine, g: Graph, m: TModel) -> np.ndarray:
    """T x S emission costs (negative log Gaussian densities)."""
    iv = 1.0 / m.var
    X = L.X
    const = 0.5 * float(np.sum(np.log(2 * math.pi * m.var)))
    xx = (X * X) @ iv
    E = 0.5 * (xx[:, None] - 2 * X @ (g.M * iv).T + ((g.M * g.M) @ iv)[None, :]) + const
    if g.junk.any():
        E[:, g.junk] = (0.5 * xx / TP["JUNK_VAR"] + const + 0.5 * m.R * math.log(TP["JUNK_VAR"]))[:, None]
    return E.astype(np.float32)


# ---------------------------------------------------------------------------
# Batched Viterbi (forward with back-pointers, and backward min-sums)
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    lines: list
    graphs: list
    E: np.ndarray        # L x Tmax x Smax
    Ctr: np.ndarray      # L x 5 x Smax
    init: np.ndarray
    endc: np.ndarray
    Tl: np.ndarray
    jl: np.ndarray
    js: np.ndarray
    jd: np.ndarray
    jc: np.ndarray


def make_batch(lines, m: TModel) -> Batch:
    gs = [build_graph(L, m) for L in lines]
    Ln = len(lines)
    Smax = max(g.S for g in gs)
    Tmax = max(L.T for L in lines)
    E = np.zeros((Ln, Tmax, Smax), np.float32)
    Ctr = np.full((Ln, 5, Smax), INF, np.float32)
    init = np.full((Ln, Smax), INF, np.float32)
    endc = np.full((Ln, Smax), INF, np.float32)
    jl, js, jd, jc = [], [], [], []
    for i, (L, g) in enumerate(zip(lines, gs)):
        E[i, :L.T, :g.S] = emissions(L, g, m)
        E[i, :, g.S:] = INF
        Ctr[i, :, :g.S] = g.Ctr
        init[i, :g.S] = g.init
        endc[i, :g.S] = g.endc
        for a, b, c, _w in g.jumps:
            jl.append(i); js.append(a); jd.append(b); jc.append(c)
    return Batch(lines, gs, E, Ctr, init, endc, np.array([L.T for L in lines]),
                 np.array(jl, int), np.array(js, int), np.array(jd, int), np.array(jc, np.float32))


def forward(B: Batch):
    Ln, Tmax, Smax = B.E.shape
    F = np.empty((Tmax, Ln, Smax), np.float32)
    BP = np.zeros((Tmax, Ln, Smax), np.int8)
    F[0] = np.minimum(B.init + B.E[:, 0], INF)
    cand = np.empty((Ln, Smax), np.float32)
    for t in range(1, Tmax):
        prev = F[t - 1]
        best = prev + B.Ctr[:, 0]
        arg = np.zeros((Ln, Smax), np.int8)
        for k in range(1, 5):
            cand[:, :k] = INF
            np.add(prev[:, :-k], B.Ctr[:, k, k:], out=cand[:, k:])
            mk = cand < best
            best[mk] = cand[mk]
            arg[mk] = k
        if len(B.jl):
            cj = prev[B.jl, B.js] + B.jc
            cur = best[B.jl, B.jd]
            mk = cj < cur
            best[B.jl[mk], B.jd[mk]] = cj[mk]
            arg[B.jl[mk], B.jd[mk]] = 5
        F[t] = np.minimum(best + B.E[:, t], INF)
        BP[t] = arg
    return F, BP


def backward(B: Batch):
    Ln, Tmax, Smax = B.E.shape
    Bk = np.empty((Tmax, Ln, Smax), np.float32)
    cand = np.empty((Ln, Smax), np.float32)
    for t in range(Tmax - 1, -1, -1):
        if t == Tmax - 1:
            cur = np.full((Ln, Smax), INF, np.float32)
        else:
            G = np.minimum(B.E[:, t + 1] + Bk[t + 1], INF)
            cur = G + B.Ctr[:, 0]
            for k in range(1, 5):
                cand[:, -k:] = INF
                np.add(G[:, k:], B.Ctr[:, k, k:], out=cand[:, :-k])
                np.minimum(cur, cand, out=cur)
            if len(B.jl):
                cj = G[B.jl, B.jd] + B.jc
                np.minimum.at(cur, (B.jl, B.js), cj)
        endl = B.Tl - 1 == t
        cur[endl] = B.endc[endl]
        cur[B.Tl - 1 < t] = INF
        Bk[t] = np.minimum(cur, INF)
    return Bk


def backtrack(B: Batch, F, BP, i):
    L, g = B.lines[i], B.graphs[i]
    T = L.T
    fin = F[T - 1, i, :g.S] + B.endc[i, :g.S]
    s = int(np.argmin(fin))
    total = float(fin[s])
    jumpmap = {}
    for k in range(len(B.jl)):
        if B.jl[k] == i:
            jumpmap[int(B.jd[k])] = int(B.js[k])
    path = np.empty(T, int)
    jumped = []
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t == 0:
            break
        a = int(BP[t, i, s])
        if a == 5:
            src = jumpmap[s]
            jumped.append((src, s))
            s = src
        else:
            s -= a
    return path, total, jumped


# ---------------------------------------------------------------------------
# Alignment read-out
# ---------------------------------------------------------------------------
@dataclass
class LineAlign:
    total: float
    path: np.ndarray
    ucols: list             # per unit: (t0, t1) inclusive, or None
    words: list             # per visual word: dict(t0, t1, deleted, fit, margin)
    cost_cols: np.ndarray   # per column emission cost on the path


def read_alignment(B: Batch, i, path, total, F=None, Bk=None) -> LineAlign:
    L, g = B.lines[i], B.graphs[i]
    U = len(L.ukeys)
    us = g.unit_state[path]
    ucols = [None] * U
    for t, u in enumerate(us):
        if u >= 0:
            ucols[u] = (ucols[u][0], t) if ucols[u] else (t, t)
    ec = B.E[i, np.arange(L.T), path].astype(np.float64)
    words = []
    for w in range(L.nwords):
        uu = [k for k in range(U) if L.uword[k] == w]
        cols = [ucols[k] for k in uu]
        if not uu or any(c is None for c in cols):
            words.append(dict(t0=None, t1=None, deleted=True, fit=None, margin=0.0))
            continue
        t0, t1 = cols[0][0], cols[-1][1]
        inw = (us >= uu[0]) & (us <= uu[-1])
        fit = float(ec[inw].mean()) if inw.any() else None
        words.append(dict(t0=t0, t1=t1, deleted=False, fit=fit, margin=None))
    return LineAlign(total, path, ucols, words, ec)


def word_margins(B: Batch, i, al: LineAlign, F, Bk, tol_cols: float):
    """Min-marginal margins for every word: best cost with its start or end
    more than tol away, or the word bypassed, minus the best cost."""
    L, g = B.lines[i], B.graphs[i]
    T, S = L.T, g.S
    Fi = F[:T, i, :S].astype(np.float64)
    Bi = Bk[:T, i, :S].astype(np.float64)
    Ei = B.E[i, :T, :S].astype(np.float64)
    Ci = B.Ctr[i, :, :S].astype(np.float64)
    total = al.total
    U = len(L.ukeys)
    jumps = {w: (a, b, c) for a, b, c, w in g.jumps}
    for w, d in enumerate(al.words):
        uu = [k for k in range(U) if L.uword[k] == w]
        if not uu:
            continue
        f, l = g.first[uu[0]], g.last[uu[-1]]
        # end of the word at t: in state l at t, leaving at t+1
        Mend = np.full(T, np.inf)
        succ = np.full(T - 1, np.inf)
        for k in range(1, 5):
            if l + k < S and Ci[k, l + k] < 1e29:
                succ = np.minimum(succ, Ci[k, l + k] + Ei[1:, l + k] + Bi[1:, l + k])
        Mend[:-1] = Fi[:-1, l] + succ
        Mend[-1] = Fi[-1, l] + float(B.endc[i, l])
        # start of the word at t: entering f at t
        Mst = np.full(T, np.inf)
        pre = np.full(T - 1, np.inf)
        for k in range(1, 5):
            if f - k >= 0 and Ci[k, f] < 1e29:
                pre = np.minimum(pre, Fi[:-1, f - k] + Ci[k, f])
        Mst[1:] = pre + Ei[1:, f] + Bi[1:, f]
        Mst[0] = float(B.init[i, f]) + Ei[0, f] + Bi[0, f]
        alts = []
        if not d["deleted"]:
            t = np.arange(T)
            far_e = np.abs(t - d["t1"]) > tol_cols
            far_s = np.abs(t - d["t0"]) > tol_cols
            if far_e.any():
                alts.append(Mend[far_e].min())
            if far_s.any():
                alts.append(Mst[far_s].min())
            if w in jumps:
                a, b, c = jumps[w]
                alts.append(float(np.min(Fi[:-1, a] + c + Ei[1:, b] + Bi[1:, b])))
            d["margin"] = float(min(alts) - total) if alts else 1e9
        else:
            d["margin"] = float(min(Mend.min(), Mst.min()) - total)


# ---------------------------------------------------------------------------
# Re-estimation
# ---------------------------------------------------------------------------
def reestimate(m: TModel, B: Batch, aligns: list, use_word, resize: bool):
    R = m.R
    sums, cnts, lens = {}, {}, {}
    tr_cnt = {}
    gap_cnt = {t: [0, 0, 0, 0] for t in (JOIN, PIECE, WORD, EDGE)}   # blank used, blank not used, blank stays, blank exits
    res2 = np.zeros(R)
    nres = 0
    obs = []
    for i, al in enumerate(aligns):
        L, g = B.lines[i], B.graphs[i]
        if al is None or not L.trusted:
            continue
        path = al.path
        U = len(L.ukeys)
        ok_unit = np.array([use_word(i, L.uword[k]) for k in range(U)], bool) if U else np.zeros(0, bool)
        us = g.unit_state[path]
        for k in range(U):
            if al.ucols[k] is None:
                continue
            key = L.ukeys[k]
            f = g.first[k]
            n = g.last[k] - f + 1
            t0, t1 = al.ucols[k]
            st = path[t0:t1 + 1] - f
            Xs = L.X[t0:t1 + 1]
            obs.append((key, st, Xs))         # every placed unit enters the variance
            if not ok_unit[k]:
                continue
            if key not in sums:
                sums[key] = np.zeros((n, R)); cnts[key] = np.zeros(n)
            if sums[key].shape[0] == n:
                np.add.at(sums[key], st, Xs)
                np.add.at(cnts[key], st, 1)
            lens.setdefault(key, []).append(t1 - t0 + 1)
            d = np.diff(st)
            c = tr_cnt.setdefault(key, [0, 0, 0])
            c[0] += int(np.sum(d == 0)); c[1] += int(np.sum(d == 1)); c[2] += int(np.sum(d == 2))
        # gaps
        blank = np.zeros(g.S, bool)
        for Ba, J, Bb in g.blocks:
            blank[Ba] = blank[Bb] = True
        for b, (Ba, J, Bb) in enumerate(g.blocks):
            if 0 < b < U and not (ok_unit[b - 1] and ok_unit[b]):
                continue
            t = EDGE if b == 0 or b == U else L.btype[b - 1]
            sel = (path == Ba) | (path == Bb)
            nb = int(sel.sum())
            if b in (0, U):
                used = nb > 0
            else:
                used = nb > 0 or bool(np.any(path == J))
            gap_cnt[t][0 if used else 1] += 1
            if nb:
                runs = int(np.sum(np.diff(np.r_[0, sel.astype(int)]) == 1))
                gap_cnt[t][2] += nb - runs
                gap_cnt[t][3] += runs
        bc = blank[path]
        res2 += (L.X[bc] ** 2).sum(0); nres += int(bc.sum())
        jc = g.junk[path]                 # not-text columns: zero mean, JUNK_VAR x variance
        res2 += (L.X[jc] ** 2).sum(0) / TP["JUNK_VAR"]; nres += int(jc.sum())
    # means
    new = {}
    for key, s_ in sums.items():
        c = cnts[key]
        new[key] = (s_ + TP["KAPPA"] * m.prior) / (c[:, None] + TP["KAPPA"])
    for key in new:
        m.tmpl[key] = new[key]
        m.occ[key] = len(lens.get(key, []))
    for key, st, Xs in obs:              # residuals against the updated means
        mu = m.mean_of(key)
        if mu.shape[0] > int(st.max()):
            res2 += ((Xs - mu[st]) ** 2).sum(0); nres += len(st)
    if nres:
        m.var = np.maximum(TP["VAR_FLOOR"], res2 / nres)
    # transitions
    tot = np.array([0.0, 0.0, 0.0])
    for c in tr_cnt.values():
        tot += c
    if tot.sum() > 0:
        pg = (tot + 1.0) / (tot.sum() + 3.0)
        m.trans_g = tuple(float(x) for x in pg)
        a = TP["ALPHA"]
        for key, c in tr_cnt.items():
            c = np.array(c, float)
            p = (c + a * pg) / (c.sum() + a)
            m.trans[key] = tuple(float(x) for x in p)
    for t, (u, nu, stays, exits) in gap_cnt.items():
        if u + nu >= 4:
            m.gap[t][0] = float(np.clip((u + 0.5) / (u + nu + 1.0), 0.02, 0.98))
        if stays + exits >= 4:
            m.gap[t][1] = float(np.clip((stays + 0.5) / (stays + exits + 1.0), 0.05, 0.98))
    if resize:
        nmax = max(TP["N_MIN"] + 1, int(round(TP["N_MAX_UNIT"] * m.n_hint_unit / m.cs)))
        for key, ls in lens.items():
            n = int(np.clip(round(float(np.median(ls)) * (1.0 - m.tr(key)[0] + m.tr(key)[2])), TP["N_MIN"], nmax))
            if key in m.tmpl and n != len(m.tmpl[key]):
                m.tmpl[key] = _resample(m.tmpl[key], n)


# ---------------------------------------------------------------------------
# Training and decoding
# ---------------------------------------------------------------------------
def align_all(m: TModel, lines: list, margins: bool, unit_px: float):
    live = [k for k, L in enumerate(lines) if L.T > 0 and L.ukeys]
    aligns = [None] * len(lines)
    if not live:
        return aligns, 0.0, None, live
    B = make_batch([lines[k] for k in live], m)
    F, BP = forward(B)
    Bk = backward(B) if margins else None
    tot = 0.0
    out = []
    for i in range(len(live)):
        path, total, _ = backtrack(B, F, BP, i)
        al = read_alignment(B, i, path, total)
        if margins:
            word_margins(B, i, al, F, Bk, TP["POS_TOL"] * unit_px / B.lines[i].cs)
        out.append(al)
        tot += total
    for k, al in zip(live, out):
        aligns[k] = al
    return aligns, tot, B, live


def fit_z(lines, aligns):
    """Robust page-relative z of each word's mean emission cost."""
    fits = [d["fit"] for al in aligns if al is not None for d in al.words if not d["deleted"] and d["fit"] is not None]
    if len(fits) < 5:
        return
    med = float(np.median(fits))
    mad = float(np.median(np.abs(np.array(fits) - med))) * 1.4826 + 1e-6
    for al in aligns:
        if al is None:
            continue
        for d in al.words:
            d["z"] = None if d["deleted"] or d["fit"] is None else (d["fit"] - med) / mad


def train(lines: list, m: TModel, unit_px: float, iters: int | None = None, learn: bool = True):
    """Embedded training; returns (aligns of the final round, history)."""
    iters = TP["ITERS"] if iters is None else iters
    m.n_hint_unit = unit_px
    hist = []
    prev_paths = None
    rounds = iters if learn else 0
    for it in range(rounds + 1):
        last = it == rounds
        res = align_all(m, lines, margins=last, unit_px=unit_px)
        aligns, tot = res[0], res[1]
        fit_z(lines, aligns)
        paths = [None if a is None else a.path.copy() for a in aligns]
        changed = None
        if prev_paths is not None:
            changed = int(sum(int(np.sum(a != b)) for a, b in zip(paths, prev_paths) if a is not None and b is not None and len(a) == len(b)))
        ndel = sum(d["deleted"] for a in aligns if a is not None for d in a.words)
        hist.append(dict(round=it, objective=round(tot, 2), changed_cols=changed, deleted_words=ndel,
                         units=len(m.tmpl), var_mean=round(float(m.var.mean()), 5),
                         trans_g=[round(x, 3) for x in m.trans_g],
                         gap={t: [round(a, 3), round(b, 3)] for t, (a, b) in m.gap.items()}))
        if last:
            m.history = hist
            return aligns, hist
        if changed == 0:
            # converged: one more pass with margins
            res = align_all(m, lines, margins=True, unit_px=unit_px)
            aligns = res[0]
            fit_z(lines, aligns)
            m.history = hist
            return aligns, hist

        def use_word(i, w, aligns=aligns):
            d = aligns[i].words[w] if aligns[i] is not None else None
            if d is None or d["deleted"]:
                return False
            z = d.get("z")
            return it == 0 or z is None or z <= TP["LEARN_Z"]
        live = [k for k, a in enumerate(aligns) if a is not None]
        B = res[2]
        reestimate(m, B, [aligns[k] for k in live],
                   lambda i, w: use_word(live[i], w), resize=(it == 0))
        prev_paths = paths if it > 0 else None
    m.history = hist
    return aligns, hist


def word_results(L: TLine, al, sw: float, unit: float, hc: float):
    """Per logical token: dict(status, quad, line_quad, margin, z, reason) and
    the raw interval (s) per logical token."""
    li = L.li
    n = len(L.tokens)
    res = [None] * n
    ivs = [None] * n
    if al is None:
        for ti in range(n):
            res[ti] = dict(status="unresolved", quad=None, line_quad=None, margin=None, z=None, reason="T: no line")
        return res, ivs
    vis_iv = []
    for d in al.words:
        if d["deleted"]:
            vis_iv.append(None)
        else:
            vis_iv.append((L.s_lo + d["t0"] * L.cs, L.s_lo + (d["t1"] + 1) * L.cs))
    txt = [x for x in vis_iv if x is not None]
    lq = WP.line_quad(li, min(a for a, _ in txt), max(b for _, b in txt), sw) if txt else (
        WP.line_quad(li, float(li.s0[0]), float(li.s1[-1]), sw) if li.M else None)
    quads = WP.snap_words(li, vis_iv, sw, unit, hc)
    for w, d in enumerate(al.words):
        ti = L.order[w]
        ivs[ti] = vis_iv[w]
        if not L.trusted:
            res[ti] = dict(status="unresolved", quad=None, line_quad=None, margin=d.get("margin"), z=d.get("z"),
                           reason="line assignment not trusted")
            continue
        if d["deleted"]:
            res[ti] = dict(status="line", quad=None, line_quad=lq, margin=0.0, z=None,
                           reason="T: word does not match the image (bypassed)")
            continue
        z = d.get("z")
        shape = box_shape(li, quads[w], vis_iv[w], unit, hc)
        d["shape"] = shape
        other_ok = (quads[w] is not None and (z is None or z <= TP["FIT_Z"])
                    and (shape is None or shape["over"] <= TP["OVER_MAX"]))
        ok = other_ok and d["margin"] is not None and d["margin"] >= TP["LOC_MARGIN"]
        if ok:
            res[ti] = dict(status="located", quad=quads[w], line_quad=lq, margin=d["margin"], z=z, reason="T",
                           other_ok=True, shape=shape)
        else:
            why = ("fit" if (z is not None and z > TP["FIT_Z"]) else
                   "ink beyond the aligned columns" if (shape and shape["over"] > TP["OVER_MAX"]) else "margin")
            res[ti] = dict(status="line", quad=quads[w], line_quad=lq, margin=d["margin"], z=z,
                           reason=f"T: word position not trusted ({why})", other_ok=other_ok, shape=shape)
    return res, ivs


def box_shape(li, quad, iv, unit, hc):
    """How far the snapped box reaches beyond the aligned interval (letters,
    the larger side) and its height (x glyph height): ink attached to a word
    beyond its own columns, or far above/below the glyph zone, is likely not
    the word (overlays, underlines, marks)."""
    if quad is None or iv is None:
        return None
    import word_placer_v as V
    a, b = V.quad_s_extent(li, quad)
    xs = np.array([p[0] for p in quad]); ys = np.array([p[1] for p in quad])
    _, vv = li.geom.uv(xs, ys)
    return dict(over=round(max(0.0, iv[0] - a, b - iv[1]) / max(1.0, unit), 3),
                short=round(max(0.0, a - iv[0], iv[1] - b) / max(1.0, unit), 3),
                h=round(float(vv.max() - vv.min()) / max(1.0, hc), 3))


def every_box(res):
    """Diagnostic arm: every non-bypassed T box shown as located."""
    out = []
    for x in res:
        if x.get("quad") is not None and x["status"] in ("located", "line") and "bypassed" not in x.get("reason", ""):
            out.append(dict(x, status="located"))
        else:
            out.append(dict(x))
    return out


def agree(li, a_res, b_res, unit, need_a_located=False):
    """Located only where both boxes exist and agree on both edges within
    AGREE_TOL letter (independent sources); else the line box."""
    import word_placer_v as V
    tol = TP["AGREE_TOL"] * unit
    out = []
    for a, b in zip(a_res, b_res):
        ok = False
        qa, qb = a.get("quad"), b.get("quad")
        bypassed = "bypassed" in a.get("reason", "")
        if qa is not None and qb is not None and not bypassed and (a["status"] == "located" or not need_a_located):
            ea, eb = V.quad_s_extent(li, qa), V.quad_s_extent(li, qb)
            ok = abs(ea[0] - eb[0]) <= tol and abs(ea[1] - eb[1]) <= tol
        if ok:
            out.append(dict(a, status="located", reason="agree"))
        elif a.get("status") == "unresolved" and b.get("status") == "unresolved":
            out.append(dict(a))
        else:
            lq = a.get("line_quad") or b.get("line_quad")
            out.append(dict(status="line" if lq else "unresolved", quad=qa or qb, line_quad=lq, margin=a.get("margin"),
                            z=a.get("z"), reason="sources disagree or one is missing"))
    return out


# ---------------------------------------------------------------------------
# Rc: arm R rendered from glyph templates cut out of the page
# ---------------------------------------------------------------------------
def rc_profile(L: TLine, m: TModel, gap_cols: dict):
    """Synthetic column-occupancy profile of the line from the page's own
    templates (reading order), with token id per column (logical)."""
    occ, tok = [], []
    for k, key in enumerate(L.ukeys):
        mu = m.mean_of(key)
        o = (mu > 0.3).mean(1)
        occ.append(o); tok.append(np.full(len(o), L.order[L.uword[k]]))
        if k < len(L.ukeys) - 1:
            gcol = int(round(gap_cols[L.btype[k]]))
            if gcol > 0:
                occ.append(np.zeros(gcol)); tok.append(np.full(gcol, -1))
    return np.concatenate(occ), np.concatenate(tok)


def rc_align(L: TLine, m: TModel, gap_cols: dict, sw: float):
    """As word_placer_r.align_line_r, with the rendered profile replaced by
    the page-template profile. Returns per logical token (s0, s1) or None."""
    import word_placer_r as Rr
    li = L.li
    b = max(2.0, Rr.RP["BIN_SW"] * sw)
    sp = Rr.strip_profile(li, b)
    if sp is None or not L.tokens:
        return None
    q, s_lo, b = sp
    occ, tok = rc_profile(L, m, gap_cols)
    # profile at template resolution -> resample to the strip's bins
    n = len(q)
    nz = occ[occ > 0]
    r = Rr.resample(occ / max(1e-9, np.quantile(nz, 0.95) if len(nz) else 1.0), n)
    tok_r = tok[np.clip(((np.arange(n) + 0.5) * len(tok) / n).astype(int), 0, len(tok) - 1)]
    _cost, path = Rr.dtw(np.clip(q, 0, 1), np.clip(r, 0, 1))
    mapped = np.zeros(n); cnt = np.zeros(n)
    for i, j in path:
        mapped[j] += i; cnt[j] += 1
    mapped = mapped / np.maximum(cnt, 1)
    spans = {}
    for j in range(n):
        if tok_r[j] >= 0:
            a = spans.setdefault(int(tok_r[j]), [j, j])
            a[0] = min(a[0], j); a[1] = max(a[1], j)
    ids = sorted(spans, key=lambda k: spans[k][0])
    cuts = [float(np.interp((spans[k1][1] + spans[k2][0]) / 2.0, np.arange(n), mapped)) for k1, k2 in zip(ids, ids[1:])]
    edges = [0.0] + cuts + [float(n)]
    out = [None] * len(L.tokens)
    for k, tid in enumerate(ids):
        if 0 <= tid < len(L.tokens):
            out[tid] = (s_lo + edges[k] * b, s_lo + edges[k + 1] * b)
    return out


# ---------------------------------------------------------------------------
# Template contact sheet (scratch / out only: templates are page images)
# ---------------------------------------------------------------------------
def contact_sheet(m: TModel, path: str, scale: int = 4, label: bool = True):
    import fitz
    keys = sorted(m.tmpl, key=lambda k: (-m.occ.get(k, 0), str(k)))
    cellw = max(len(m.tmpl[k]) for k in keys) * scale + 8
    cellh = m.R * scale + 22
    ncol = 12
    nrow = (len(keys) + ncol - 1) // ncol
    doc = fitz.open()
    W, H = ncol * cellw + 10, nrow * cellh + 10
    pg = doc.new_page(width=W, height=H)
    img = np.full((H, W), 255, np.uint8)
    for n, k in enumerate(keys):
        r, c = divmod(n, ncol)
        mu = m.tmpl[k]
        tile = np.clip(255 - 255 * mu.T, 0, 255).astype(np.uint8)          # R x n
        tile = np.repeat(np.repeat(tile, scale, 0), scale, 1)[:, ::-1]     # s runs right to left on the page
        y0, x0 = 5 + r * cellh + 18, 5 + c * cellw + (cellw - tile.shape[1]) // 2
        img[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile
    pg.insert_image(pg.rect, pixmap=fitz.Pixmap(fitz.csGRAY, W, H, img.tobytes(), False))
    if label:
        for n, k in enumerate(keys):
            r, c = divmod(n, ncol)
            form = {(0, 0): "iso", (0, 1): "ini", (1, 1): "med", (1, 0): "fin"}[(k[1], k[2])]
            try:
                pg.insert_text((5 + c * cellw + 2, 5 + r * cellh + 12), f"{form} n{m.occ.get(k, 0)}", fontsize=8)
            except Exception:  # noqa: BLE001
                pass
    pg.get_pixmap().save(path)
    doc.close()
    return keys
