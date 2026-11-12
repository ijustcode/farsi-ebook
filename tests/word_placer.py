#!/usr/bin/env python3
"""Word placement inside found lines (plan §4.2, §4.3 arm G, §4.4; Step 4).

Research code only: production (``farsi2epub/``) never imports ``tests/``.
numpy + PyMuPDF only, no API, deterministic.

Inputs per page: a grey image, the page's lines (either the Step 3 line finder
``tests/line_finder.py`` (arm L1) or given line geometries such as audited line
frames) and the text: either one string per printed line (line-broken) or a
paragraph stream (Markdown, not line-broken).

1. **Line ink** (``line_ink``). The ink kept by L1 (masked stamps/signatures
   removed, text under them restored) is split into connected components and
   every component goes to the nearest line centre within ``ASSIGN_TOL`` x
   line spacing. A line's pixels are mapped into its own frame (u along the
   line, v across it). The baseline zone is the peak of the v profile; the
   column profile of the *core band* around it (dots, descender tails and
   neighbouring lines' ink are outside it) gives the **ink runs**: maximal
   column intervals with ink. Reading coordinate s runs right to left.
2. **Text pieces**. Each word is cut after every non-joining letter (ا د ذ ر
   ز ژ و ...), at ZWNJ and around non-Arabic characters: the parts of a word
   that are printed as separate ink. Each letter has a joining *form* (joins
   the next letter or not), the key of the width table.
3. **Arm G** (``align_line``): exact dynamic programming over (pieces done,
   runs consumed, was the last run text). Moves:
   * a group: 1 piece over 1..``B_MAX`` runs (dots, broken strokes), or
     2..``A_MAX`` pieces in one run (touching letters, *fused words*: the
     cut is then placed by width and snapped to the weakest ink column);
   * **not text**: skip one run, cost rising with its ink mass relative to a
     letter piece, lower outside the fitted text block (§4.3, §13 O1);
   * delete a whole word (text word with no ink, §4.4).
   Costs are negative log likelihoods: width residual of the group against the
   learned piece widths (variance growing with letter count; **width leads**),
   gap width given the boundary type (in-word vs word, learned per page: on
   tight typewriter pages the two distributions coincide and gaps only refine,
   §13 O2), fusion and split rates.
   Forward-backward min-sums give, for every word edge, the cost of the best
   segmentation that puts that edge elsewhere (more than ``POS_TOL`` letter
   away) or deletes the word: the **margin** (§4.3 confidence).
4. **Width table** (``WidthModel``): per (letter, form) widths with ridge
   shrinkage to a per-letter and a global width, plus a piece offset; start
   uniform, then align -> re-estimate -> align (``ITERS``), from the page's
   own image and text only. Optional ``classes``: quantise the table into K
   width classes on a common pitch when BIC prefers it (§4.3 typewriter).
5. **Status** (owner decision D1): ``located`` when the margin is at least
   ``LOC_MARGIN`` and the width residual is plausible; ``line`` (box = the
   line) when the line is trusted but the word is not; ``unresolved`` when the
   line assignment itself is not trusted. Never an approximate guess box.
6. **Line text** (§4.2, ``assign_text_lines``): sequence alignment of text
   lines to detected lines by length (characters vs ink extent, page scale
   searched on a grid), with detected lines allowed to carry no text, text
   lines allowed to be missing, vertical spacing checked, and a per-line
   margin. ``distribute_words`` does the same for unbroken text (tokens to
   lines), then refines each line boundary by arm G's own cost.

Every length is relative to measured page quantities: stroke width ``sw``,
glyph height ``hc``, line spacing ``S`` (from L1) and the page's own letter
unit ``unit`` (ink extent per letter, measured from the text and the ink).
"""
from __future__ import annotations

import math
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np

import line_finder as LF

VERSION = "word_placer/G-1"

PARAMS = dict(
    # line ink
    ASSIGN_TOL=0.6,        # component -> nearest line centre within this x S
    ASSIGN_REACH=1.0,      # ... and within the line's x-range extended by this x S
    DOT_Q=0.01,            # detached ink joins a word only if its centre lies within the line's core-attached
    DOT_PAD=0.5,           # ink v-range (DOT_Q..1-DOT_Q quantiles) widened by this x hc
    RULE_RUN=4.0,          # an ink row along the line at least this x glyph height long ...
    RULE_GAP=0.3,          # ... (gaps up to this x glyph height bridged: typed underscores) ...
    RULE_BELOW=0.1,        # ... lying below the baseline zone by this x glyph height is a rule, not text
    CORE_UP=0.60,          # core band: this x hc above the baseline peak ...
    CORE_DN=0.30,          # ... and this x hc below it
    # arm G moves
    A_MAX=4,               # at most this many pieces fused in one run
    B_MAX=4,               # at most this many runs in one piece
    POS_TOL=0.5,           # alternative edge positions closer than this x letter unit are 'the same'
    # costs (nats)
    SKIP_BASE=1.0,         # not-text: fixed cost per skipped run
    SKIP_MASS=4.0,         # ... plus this x (run mass / median piece mass) ...
    SKIP_MASS_POW=1.0,     # ... to this power ...
    SKIP_H=10.0,           # ... plus this x max(0, run height / glyph height - SKIP_H0): tall = a letter
    SKIP_H0=0.45,
    SKIP_OUT=0.25,         # ... times this outside the fitted text block (by > EDGE_OUT x S)
    EDGE_OUT=0.5,
    SPLIT_BASE=1.0,        # one piece over several runs: per extra run
    SPLIT_SW=2.0,          # ... plus 0.5 (gap / (SPLIT_SW x sw))^2
    DEL_WORD=12.0,         # delete a text word (no ink)
    DEL_PER_PIECE=1.0,
    FUSE_INIT=(0.30, 0.05),  # initial P(fused) for (in-word, word) boundaries
    FUSE_CLIP=(0.01, 0.6),
    GAP_SD_MIN=0.15,       # floor of the log-gap sd
    # width model
    ITERS=3,               # align -> re-estimate rounds after the uniform start
    SD0_INIT=0.6, SD1_INIT=0.45,  # initial width sd: sqrt(SD0^2 + SD1^2 n) x unit
    SD0_MIN=0.25, SD1_MIN=0.10,   # floors (x unit)
    RIDGE_LETTER=0.35,     # prior sd of a letter's deviation from the global width (x unit)
    RIDGE_FORM=0.20,       # prior sd of a form's deviation from its letter (x unit)
    LEARN_MARGIN=2.0,      # only pieces of words with margin >= this feed re-estimation
    CLASSES_K=3,           # 'classes' width mode: K classes on a common pitch
    # status
    LOC_MARGIN=6.0,        # located: margin >= this (nats; smallest reaching 98% precision on synthetic dev B, clean) ...
    LOC_Z=2.5,             # ... and |width z| <= this
    FUSE_VALLEY=0.35,      # a fused word cut is trusted only if its column ink <= this x median column ink
    # line assignment (§4.2)
    LA_SD_REL=0.10,        # length sd, share of the predicted extent ...
    LA_SD_ABS=1.0,         # ... plus this x unit
    LA_SKIP_DET=4.0,       # detected line without text
    LA_SKIP_DET_NONTEXT=0.5,  # ... if L1 already classed it non-text
    LA_SKIP_TEXT=14.0,     # text line with no detected line
    LA_CLOSE=0.6,          # two text lines closer than this x S: forbidden
    LA_NEAR=0.85,          # closer than this x S: LA_NEAR_COST
    LA_NEAR_COST=3.0,
    LA_MARGIN=3.0,         # a text line's assignment is trusted at margin >= this
    LA_SCALES=41,          # scale grid size (0.6 .. 1.6 x the guess)
    # unbroken text (§4.2 variant)
    GAPCOUNT_ETA=2.0,      # word-gap counts in the line-text DPs: DISABLED (>1 never passes). Tried at 0.6 on synthetic dev: book naskh B 78->31%, D worse; rejected
    GAPCOUNT_SD=1.0,       # gap-count sd: GAPCOUNT_SD + GAPCOUNT_REL x tokens
    GAPCOUNT_REL=0.12,
    DW_STRETCH=0.22,       # a non-final justified line may be this much wider than its natural width
    DW_SD=0.06,            # log-width sd outside the allowance
    DW_PARA_CROSS=8.0,     # a line holding the end of one paragraph and the start of the next
    DW_SKIP_TOKEN=6.0,     # a token placed on no line
    DW_REFINE=1,           # boundary refinement window (tokens) with arm G cost
    DW_MARGIN=3.0,         # boundary tokens trusted only when moving them costs >= this
)
P = PARAMS

NONJOIN = set("اآأإٱدذرزژوؤءةۀ")   # letters that never join the next one
ZW = "‌‍‎‏​"
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
TATWEEL = "ـ"


def is_arabic_letter(ch: str) -> bool:
    o = ord(ch)
    if not any(a <= o <= b for a, b in ARABIC_RANGES):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("L")


def char_key(ch: str) -> str:
    """Width-table identity of a character (digits and Latin letters pooled)."""
    if ch.isdigit():
        return "0"
    if ch.isascii() and ch.isalpha():
        return "a" if ch.islower() else "A"
    if unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("S"):
        return ch
    return {"ي": "ی", "ك": "ک", "ى": "ی", "ة": "ه"}.get(ch, ch)


def letters(token: str) -> float:
    n = sum(1 for c in token if not unicodedata.category(c).startswith("M") and c not in ZW and c != TATWEEL)
    return float(max(n, 1))


@dataclass
class Piece:
    keys: list            # (char_key, form) per printed letter
    word: int             # index of the word in the line (visual order)
    first: bool           # first piece of its word
    last: bool            # last piece of its word

    @property
    def n(self):
        return len(self.keys)


def token_pieces(token: str) -> list[list[tuple[str, str]]]:
    """Split one token into separately printed pieces of (key, form)."""
    chars = [c for c in token if not unicodedata.category(c).startswith("M") and c != TATWEEL]
    pieces, cur = [], []
    for i, c in enumerate(chars):
        if c in ZW:
            if cur:
                pieces.append(cur)
            cur = []
            continue
        if not is_arabic_letter(c):
            if cur:
                pieces.append(cur)
            pieces.append([(char_key(c), "E")])
            cur = []
            continue
        nxt = next((d for d in chars[i + 1:]), None)
        joins = (c not in NONJOIN and nxt is not None and nxt not in ZW and is_arabic_letter(nxt))
        cur.append((char_key(c), "J" if joins else "E"))
        if not joins:
            pieces.append(cur)
            cur = []
    if cur:
        pieces.append(cur)
    return pieces or [[("?", "E")]]


def visual_order(tokens: list[str]) -> list[int]:
    """Right-to-left visual order of logical tokens: runs of LTR tokens
    (Latin, and digits/numbers following Latin, bidi W7) are reversed."""
    import synth_pages as S
    flags = S.ltr_flags(tokens)
    order, i = [], 0
    while i < len(tokens):
        if flags[i]:
            j = i
            while j < len(tokens) and flags[j]:
                j += 1
            order += list(range(j - 1, i - 1, -1))
            i = j
        else:
            order.append(i)
            i += 1
    return order


def line_pieces(tokens: list[str]) -> tuple[list[Piece], list[int]]:
    """Pieces of a line in visual (reading-direction) order and, for each
    visual word, its logical token index."""
    order = visual_order(tokens)
    pieces = []
    for w, ti in enumerate(order):
        ps = token_pieces(tokens[ti])
        for k, keys in enumerate(ps):
            pieces.append(Piece(keys, w, k == 0, k == len(ps) - 1))
    return pieces, order


# ---------------------------------------------------------------------------
# Line geometries: page (x, y) <-> line frame (u along the line, v across)
# ---------------------------------------------------------------------------
class ShearGeom:
    """L1 line: u = x, v = y - centre(x) (vertical shear, exactly invertible)."""

    def __init__(self, L):
        self.L = L
        self.x0, self.x1 = float(L.x0), float(L.x1)

    def centre_y(self, x):
        return self.L.y_at(x)

    def uv(self, x, y):
        x = np.asarray(x, np.float64)
        return x, np.asarray(y, np.float64) - self.L.y_at(x)

    def xy(self, u, v):
        u = np.asarray(u, np.float64)
        return u, np.asarray(v, np.float64) + self.L.y_at(u)


class FrameGeom:
    """A rotated straight frame (audited line outlines)."""

    def __init__(self, cx, cy, angle_deg, u0, u1):
        self.cx, self.cy = float(cx), float(cy)
        a = math.radians(angle_deg)
        self.c, self.s = math.cos(a), math.sin(a)
        self.u0, self.u1 = float(u0), float(u1)
        xa, _ = self.xy(np.array([u0, u1]), np.array([0.0, 0.0]))
        self.x0, self.x1 = float(xa.min()), float(xa.max())

    def centre_y(self, x):
        x = np.asarray(x, np.float64)
        return self.cy + (x - self.cx) * self.s / max(1e-9, self.c)

    def uv(self, x, y):
        dx = np.asarray(x, np.float64) - self.cx
        dy = np.asarray(y, np.float64) - self.cy
        return dx * self.c + dy * self.s, -dx * self.s + dy * self.c

    def xy(self, u, v):
        u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
        return self.cx + u * self.c - v * self.s, self.cy + u * self.s + v * self.c


# ---------------------------------------------------------------------------
# Page evidence (from L1) and line ink
# ---------------------------------------------------------------------------
@dataclass
class PageEvidence:
    gray: np.ndarray
    lf: dict                    # line_finder.find_lines result (keep_arrays=True)
    W: int
    H: int
    sw: float
    hc: float
    S: float
    cc: object                  # components of the kept + restored text ink
    order: np.ndarray           # runs of cc sorted by label
    starts: np.ndarray          # first run index of each component in `order`
    edges: dict
    seconds: float = 0.0


def page_evidence(gray: np.ndarray) -> PageEvidence:
    t0 = time.perf_counter()
    lf = LF.find_lines(gray, use_mask=True, keep_arrays=True)
    m = lf["measurements"]
    ink = lf["line_ink"].copy()
    hc = float(m["glyph_height_px"])
    # rules touching text (typed underlines): horizontal ink runs at least
    # RULE_RUN x glyph height long are not text; L1 keeps them because they
    # touch text components (§4.1 conservative masking)
    c0 = LF.components(ink)
    long_ = (c0.e - c0.s) >= P["RULE_RUN"] * hc
    rule_px = 0
    if long_.any():
        for r_, a_, b_ in zip(c0.r[long_], c0.s[long_], c0.e[long_]):
            ink[r_, a_:b_] = False
            rule_px += int(b_ - a_)
    cc = LF.components(ink)
    order = np.argsort(cc.lab, kind="stable")
    starts = np.searchsorted(cc.lab[order], np.arange(cc.n + 1))
    return PageEvidence(gray, lf, m["width"], m["height"], float(m["stroke_px"]), float(m["glyph_height_px"]),
                        float(m["line_spacing_px"]), cc, order, starts, lf["edges"],
                        seconds=time.perf_counter() - t0)


def assign_components(ev: PageEvidence, geoms: list) -> list[np.ndarray]:
    """Components -> nearest line centre (vertical distance at the
    component's centroid) within ASSIGN_TOL x S and the line's x-range."""
    cc = ev.cc
    if not geoms or cc.n == 0:
        return [np.zeros(0, np.int64) for _ in geoms]
    cx, cy = cc.cx, cc.cy
    reach = P["ASSIGN_REACH"] * ev.S
    D = np.full((len(geoms), cc.n), np.inf)
    for k, g in enumerate(geoms):
        ok = (cx >= g.x0 - reach) & (cx <= g.x1 + reach)
        D[k, ok] = np.abs(cy[ok] - g.centre_y(cx[ok]))
    best = np.argmin(D, axis=0)
    bd = D[best, np.arange(cc.n)]
    good = bd <= P["ASSIGN_TOL"] * ev.S
    return [np.nonzero(good & (best == k))[0] for k in range(len(geoms))]


def comp_pixels(ev: PageEvidence, comps: np.ndarray):
    """Pixel centres of the given components and, per pixel, the position of
    its component in ``comps``."""
    cc = ev.cc
    if not len(comps):
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
    parts = [ev.order[ev.starts[c]:ev.starts[c + 1]] for c in comps]
    idx = np.concatenate(parts)
    owner = np.repeat(np.arange(len(comps)), [len(p) for p in parts])
    r, s, e = cc.r[idx], cc.s[idx], cc.e[idx]
    ln = e - s
    tot = int(ln.sum())
    xs = np.repeat(s, ln) + (np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln))
    ys = np.repeat(r, ln)
    return xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5, np.repeat(owner, ln)


def rule_pixels(u, v, min_len, gap=2.0):
    """Pixels on runs along the line (constant v row of the line frame) at
    least min_len long, gaps of one pixel bridged: typed underlines and rules
    touching the text, which L1's mask keeps (§4.1)."""
    ui = np.floor(u).astype(np.int64); vi = np.floor(v).astype(np.int64)
    o = np.lexsort((ui, vi))
    us, vs = ui[o], vi[o]
    brk = np.r_[True, (vs[1:] != vs[:-1]) | (us[1:] - us[:-1] > gap)]
    rid = np.cumsum(brk) - 1
    first = np.nonzero(brk)[0]
    last = np.r_[first[1:], len(us)] - 1
    length = us[last] - us[first] + 1
    out = np.zeros(len(u), bool)
    out[o] = length[rid] >= min_len
    return out


@dataclass
class LineInk:
    geom: object
    u: np.ndarray               # pixel u (all band pixels of the line)
    v: np.ndarray
    uref: float                 # s = uref - u (reading coordinate, right to left)
    base_v: float
    s0: np.ndarray              # runs (core band) in s, ascending
    s1: np.ndarray
    mass: np.ndarray            # all-band pixels in each run's columns
    colprof: np.ndarray         # core column profile indexed by int s
    in_block: np.ndarray        # per run: inside the fitted text block
    y_mid: float
    comp: np.ndarray = None     # per pixel: component (index into the line's component list)
    core: np.ndarray = None     # per pixel: inside the core band
    height: np.ndarray = None   # per run: vertical extent of all ink in its columns

    @property
    def M(self):
        return len(self.s0)


def line_ink(ev: PageEvidence, geom, comps: np.ndarray) -> LineInk:
    xs, ys, owner = comp_pixels(ev, comps)
    u, v = geom.uv(xs, ys)
    if len(u):
        vb0 = np.floor(v - v.min()).astype(int)
        pr0 = LF.smooth1d(np.bincount(vb0).astype(np.float64), max(0.5, 0.05 * ev.hc))
        b0 = float(np.argmax(pr0)) + float(v.min()) + 0.5
        rp = rule_pixels(u, v, P["RULE_RUN"] * ev.hc, P["RULE_GAP"] * ev.hc)
        rp &= v > b0 + P["RULE_BELOW"] * ev.hc
        u, v, owner = u[~rp], v[~rp], owner[~rp]
    if not len(u):
        z = np.zeros(0)
        return LineInk(geom, u, v, 0.0, 0.0, z, z, z, z, np.zeros(0, bool), float(geom.centre_y((geom.x0 + geom.x1) / 2)),
                       owner, np.zeros(0, bool))
    # baseline zone: peak of the v profile
    vb = np.floor(v - v.min()).astype(int)
    prof = LF.smooth1d(np.bincount(vb).astype(np.float64), max(0.5, 0.05 * ev.hc))
    base_v = float(np.argmax(prof)) + float(v.min()) + 0.5
    core = (v >= base_v - P["CORE_UP"] * ev.hc) & (v <= base_v + P["CORE_DN"] * ev.hc)
    uref = float(np.ceil(u.max())) + 1.0
    s = uref - u
    si = np.floor(s).astype(int)
    n = int(si.max()) + 2
    colc = np.bincount(si[core], minlength=n).astype(np.float64) if core.any() else np.zeros(n)
    cola = np.bincount(si, minlength=n).astype(np.float64)
    on = colc > 0
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    st = np.nonzero(d == 1)[0]
    en = np.nonzero(d == -1)[0]
    cs = np.r_[0.0, np.cumsum(cola)]
    mass = cs[en] - cs[st]
    s0 = st.astype(np.float64); s1 = en.astype(np.float64)
    # text-block membership of each run (fitted edges from L1)
    xm, _ = geom.xy(uref - (s0 + s1) / 2, np.full(len(s0), base_v))
    ym = geom.centre_y(xm)
    inb = np.ones(len(s0), bool)
    tol = P["EDGE_OUT"] * ev.S
    for key, sgn in (("right", 1), ("left", -1)):
        e = ev.edges.get(key) if ev.edges else None
        if e:
            ex = e["a"] + e["b"] * ym
            inb &= (sgn * (xm - ex)) <= tol
    ymid = float(geom.centre_y((geom.x0 + geom.x1) / 2))
    ri = np.searchsorted(s0, s, side="right") - 1
    okr = (ri >= 0) & (s < s1[np.clip(ri, 0, None)])
    vmin = np.full(len(s0), np.inf); vmax = np.full(len(s0), -np.inf)
    np.minimum.at(vmin, ri[okr], v[okr]); np.maximum.at(vmax, ri[okr], v[okr])
    height = np.where(np.isfinite(vmin), vmax - vmin + 1.0, 0.0)
    return LineInk(geom, u, v, uref, base_v, s0, s1, mass, colc, inb, ymid, owner, core, height)


# ---------------------------------------------------------------------------
# Width model
# ---------------------------------------------------------------------------
class WidthModel:
    def __init__(self, unit: float, sw: float, hc: float | None = None):
        self.unit = float(unit)
        self.sw = float(sw)
        self.hc = hc
        self.g = self.unit          # global width per letter
        self.dc: dict = {}          # letter deviation
        self.ef: dict = {}          # (letter, form) deviation
        self.beta = 0.0             # piece offset (ink narrower than advances)
        self.sd0 = P["SD0_INIT"] * unit
        self.sd1 = P["SD1_INIT"] * unit
        x_in = math.log(max(1.0, 0.5 * unit) + sw)
        x_w = math.log(max(1.0, 1.0 * unit) + sw)
        self.gap = {0: [x_in, 0.6], 1: [x_w, 0.6]}
        self.pfuse = {0: P["FUSE_INIT"][0], 1: P["FUSE_INIT"][1]}
        self.classes = None         # optional {(letter, form): class width}
        self.history: list = []

    def width(self, key) -> float:
        if self.classes is not None and key in self.classes:
            return self.classes[key]
        return self.g + self.dc.get(key[0], 0.0) + self.ef.get(key, 0.0)

    def piece_mu(self, p: Piece) -> float:
        return sum(self.width(k) for k in p.keys) + self.beta

    def piece_var(self, p: Piece) -> float:
        return self.sd0 ** 2 + self.sd1 ** 2 * p.n

    def gap_cost(self, t: int, gap: np.ndarray) -> np.ndarray:
        m, s = self.gap[t]
        x = np.log(np.maximum(gap, 0.0) + self.sw)
        return 0.5 * ((x - m) / s) ** 2 + math.log(s) - math.log(max(1e-6, 1 - self.pfuse[t]))

    def fuse_cost(self, t: int) -> float:
        return -math.log(max(1e-6, self.pfuse[t]))

    # --- re-estimation from aligned observations ---
    def refit(self, obs_pieces, obs_gaps, obs_fuse, mode: str = "free"):
        """obs_pieces: list of (Piece, measured extent); obs_gaps: list of
        (type, gap px); obs_fuse: {type: [fused, total]}."""
        if len(obs_pieces) >= 8:
            letters_ = sorted({k[0] for p, _ in obs_pieces for k in p.keys})
            forms_ = sorted({k for p, _ in obs_pieces for k in p.keys})
            li = {c: i for i, c in enumerate(letters_)}
            fi = {k: i for i, k in enumerate(forms_)}
            nL, nF = len(letters_), len(forms_)
            X = np.zeros((len(obs_pieces), 2 + nL + nF))
            y = np.zeros(len(obs_pieces))
            for r, (p, E) in enumerate(obs_pieces):
                X[r, 0] = p.n
                X[r, 1] = 1.0
                for k in p.keys:
                    X[r, 2 + li[k[0]]] += 1
                    X[r, 2 + nL + fi[k]] += 1
                y[r] = E
            w_ = np.ones(len(y))
            for _ in range(3):   # robust: downweight residuals beyond 2.5 sd
                sig2 = np.mean((y - X @ self._vec(letters_, forms_)) ** 2) if self.history else (self.sd0 ** 2 + self.sd1 ** 2 * 3)
                lamL = sig2 / (P["RIDGE_LETTER"] * self.unit) ** 2
                lamF = sig2 / (P["RIDGE_FORM"] * self.unit) ** 2
                R = np.diag(np.r_[0.0, 0.0, np.full(nL, lamL), np.full(nF, lamF)])
                Xw = X * w_[:, None]
                theta = np.linalg.solve(X.T @ Xw + R + 1e-9 * np.eye(X.shape[1]), Xw.T @ y)
                res = y - X @ theta
                sd = max(1e-6, float(np.sqrt(np.average(res ** 2, weights=w_))))
                w_ = np.where(np.abs(res) <= 2.5 * sd, 1.0, 2.5 * sd / np.maximum(np.abs(res), 1e-9))
                self.g, self.beta = float(theta[0]), float(theta[1])
                self.dc = {c: float(theta[2 + li[c]]) for c in letters_}
                self.ef = {k: float(theta[2 + nL + fi[k]]) for k in forms_}
                self.history.append(1)
            # variance: res^2 ~ sd0^2 + sd1^2 n (non-negative least squares, floored)
            n = X[:, 0]
            A = np.stack([np.ones(len(n)), n], 1)
            r2 = np.minimum(res ** 2, (4 * sd) ** 2)
            c0, c1 = np.linalg.lstsq(A, r2, rcond=None)[0]
            self.sd0 = max(P["SD0_MIN"] * self.unit, math.sqrt(max(c0, 0.0)))
            self.sd1 = max(P["SD1_MIN"] * self.unit, math.sqrt(max(c1, 0.0)))
            self.classes = None
            if mode == "classes":
                self._quantise(obs_pieces, forms_)
        for t in (0, 1):
            g = np.array([x for tt, x in obs_gaps if tt == t], np.float64)
            if len(g) >= 6:
                x = np.log(np.maximum(g, 0.0) + self.sw)
                med = float(np.median(x))
                mad = float(np.median(np.abs(x - med))) * 1.4826
                self.gap[t] = [med, max(P["GAP_SD_MIN"], mad)]
            f, tot = obs_fuse.get(t, [0, 0])
            if tot >= 6:
                lo, hi = P["FUSE_CLIP"]
                self.pfuse[t] = min(hi, max(lo, (f + 0.5) / (tot + 1.0)))

    def _vec(self, letters_, forms_):
        return np.r_[self.g, self.beta, [self.dc.get(c, 0.0) for c in letters_], [self.ef.get(k, 0.0) for k in forms_]]

    def _quantise(self, obs_pieces, forms_):
        """K width classes on a common pitch: k-means of the free widths
        (weighted by use), class widths refitted jointly; kept only if BIC
        prefers it to the free table."""
        K = P["CLASSES_K"]
        cnt = {}
        for p, _ in obs_pieces:
            for k in p.keys:
                cnt[k] = cnt.get(k, 0) + 1
        keys = [k for k in forms_ if cnt.get(k, 0) > 0]
        if len(keys) < 2 * K:
            return
        w = np.array([self.width(k) for k in keys])
        wt = np.array([cnt[k] for k in keys], np.float64)
        cs = np.quantile(w, np.linspace(0.15, 0.85, K))
        for _ in range(50):
            lab = np.argmin(np.abs(w[:, None] - cs[None, :]), 1)
            new = np.array([np.average(w[lab == j], weights=wt[lab == j]) if (lab == j).any() else cs[j] for j in range(K)])
            if np.allclose(new, cs):
                break
            cs = new
        X = np.zeros((len(obs_pieces), K + 1)); y = np.zeros(len(obs_pieces))
        kl = dict(zip(keys, lab))
        for r, (p, E) in enumerate(obs_pieces):
            for k in p.keys:
                X[r, kl.get(k, int(np.argmin(np.abs(cs - self.width(k)))))] += 1
            X[r, -1] = 1
            y[r] = E
        th = np.linalg.lstsq(X, y, rcond=None)[0]
        rss_c = float(np.sum((y - X @ th) ** 2))
        Xf = np.array([[sum(self.width(k) for k in p.keys) + self.beta] for p, _ in obs_pieces])[:, 0]
        rss_f = float(np.sum((y - Xf) ** 2))
        n = len(y)
        bic_c = n * math.log(max(rss_c, 1e-9) / n) + (K + 1) * math.log(n)
        bic_f = n * math.log(max(rss_f, 1e-9) / n) + (len(keys) + 2) * math.log(n)
        self.class_report = dict(widths=[round(float(x), 2) for x in th[:K]], offset=round(float(th[-1]), 2),
                                 bic_classes=round(bic_c, 1), bic_free=round(bic_f, 1), used=bic_c < bic_f)
        if bic_c < bic_f:
            self.classes = {k: float(th[kl[k]]) for k in keys}
            self.beta = float(th[-1])

    def snapshot(self) -> dict:
        return dict(unit=round(self.unit, 2), g=round(self.g, 2), beta=round(self.beta, 2),
                    sd0=round(self.sd0, 2), sd1=round(self.sd1, 2),
                    gap={t: [round(math.exp(m) - self.sw, 2), round(s, 3)] for t, (m, s) in self.gap.items()},
                    pfuse={t: round(v, 3) for t, v in self.pfuse.items()},
                    classes=getattr(self, "class_report", None) if self.classes is not None else None)


# ---------------------------------------------------------------------------
# Arm G: alignment of pieces to ink runs in one line
# ---------------------------------------------------------------------------
INF = 1e18


@dataclass
class LineAlignment:
    cost: float
    words: list                 # per visual word: dict(s0, s1, status_hint, margin, z, deleted, fused)
    groups: list                # (piece_lo, piece_hi, run_lo, run_hi) on the best path; skips as (None, None, j, j+1)
    skipped_runs: list
    n_pieces: int
    obs_pieces: list = field(default_factory=list)
    obs_gaps: list = field(default_factory=list)
    obs_fuse: dict = field(default_factory=dict)


def skip_costs(li: LineInk, mref: float, nottext: bool, hc: float = None) -> np.ndarray:
    if not nottext:
        return np.full(li.M, INF)
    rel = (li.mass / max(1.0, mref)) ** P["SKIP_MASS_POW"]
    c = P["SKIP_BASE"] + P["SKIP_MASS"] * rel
    if hc and li.height is not None:
        c = c + P["SKIP_H"] * np.maximum(0.0, li.height / hc - P["SKIP_H0"])
    return np.where(li.in_block, c, c * P["SKIP_OUT"])


def align_line(li: LineInk, pieces: list[Piece], model: WidthModel, mref: float, *, nottext: bool = True,
               want_margins: bool = True) -> LineAlignment:
    M, N = li.M, len(pieces)
    nwords = (pieces[-1].word + 1) if pieces else 0
    if N == 0:
        return LineAlignment(0.0, [], [], list(range(M)), 0)
    if M == 0:
        return LineAlignment(DEL_ALL(pieces), [dict(deleted=True, s0=None, s1=None, margin=0.0, z=0.0, fused=False)
                                                 for _ in range(nwords)], [], [], N)
    s0, s1 = li.s0, li.s1
    gaps = s0[1:] - s1[:-1]                       # gap after run j (j = 0..M-2)
    skc = skip_costs(li, mref, nottext, model.hc)
    C = np.r_[0.0, np.cumsum(np.minimum(skc, 1e12))]   # prefix skip costs
    mu = np.array([model.piece_mu(p) for p in pieces])
    var = np.array([model.piece_var(p) for p in pieces])
    cmu = np.r_[0.0, np.cumsum(mu)]
    cvar = np.r_[0.0, np.cumsum(var)]
    btype = np.array([1 if pieces[i].last else 0 for i in range(N)])   # boundary after piece i
    fuse_c = np.array([model.fuse_cost(int(t)) for t in btype])
    cfuse = np.r_[0.0, np.cumsum(fuse_c)]
    gapc = {t: np.r_[model.gap_cost(t, gaps), INF] for t in (0, 1)}  # index j-1 -> gap before run j
    sp = P["SPLIT_BASE"] + 0.5 * (gaps / (P["SPLIT_SW"] * model.sw)) ** 2
    csplit = np.r_[0.0, np.cumsum(sp)]
    word_first = [i for i in range(N) if pieces[i].first]
    word_last = {pieces[i].word: i for i in range(N) if pieces[i].last}
    wfirst_of = {pieces[i].word: i for i in range(N) if pieces[i].first}
    del_cost = {w: P["DEL_WORD"] + P["DEL_PER_PIECE"] * (word_last[w] - wfirst_of[w] + 1) for w in word_last}
    shapes = [(a, 1) for a in range(1, P["A_MAX"] + 1)] + [(1, b) for b in range(2, P["B_MAX"] + 1)]
    jj = np.arange(M + 1)

    def group_cost(i, a, b):
        """Vector over j (runs consumed before the group, 0..M-b): cost of
        pieces i..i+a-1 over runs j..j+b-1."""
        if i + a > N:
            return None
        js = np.arange(0, M - b + 1)
        E = s1[js + b - 1] - s0[js]
        m_ = cmu[i + a] - cmu[i]
        v_ = cvar[i + a] - cvar[i] + (a - 1) * (0.5 * model.unit) ** 2
        c = 0.5 * (E - m_) ** 2 / v_ + 0.5 * math.log(2 * math.pi * v_)
        if a > 1:
            c = c + (cfuse[i + a - 1] - cfuse[i])
        if b > 1:
            c = c + (csplit[js + b - 1] - csplit[js])
        return js, c

    GC = {}
    for i in range(N):
        for a, b in shapes:
            r = group_cost(i, a, b)
            if r is not None:
                GC[(i, a, b)] = r

    # ---- forward ----
    F0 = np.full((N + 1, M + 1), INF)
    F1 = np.full((N + 1, M + 1), INF)
    F1[0, :] = C            # lead skips (start state counts as 'skipped')
    for i in range(1, N + 1):
        for a, b in shapes:
            ip = i - a
            if ip < 0 or (ip, a, b) not in GC:
                continue
            js, c = GC[(ip, a, b)]
            if ip == 0:
                prev = F1[0, js]
            else:
                t = int(btype[ip - 1])
                g = gapc[t][np.maximum(js - 1, 0)]
                g = np.where(js >= 1, g, INF)
                prev = np.minimum(F0[ip, js] + g, F1[ip, js])
            F0[i, js + b] = np.minimum(F0[i, js + b], prev + c)
        # deletion of a whole word ending at piece i-1
        if pieces[i - 1].last:
            w = pieces[i - 1].word
            ip = wfirst_of[w]
            F0[i] = np.minimum(F0[i], F0[ip] + del_cost[w])
            F1[i] = np.minimum(F1[i], F1[ip] + del_cost[w])
        # skips within row i: F1[i][j+1] = min(F0[i][j], F1[i][j]) + skc[j]
        X = np.minimum(F0[i], F1[i])
        acc = np.minimum.accumulate(X[:-1] - C[:-1])
        F1[i, 1:] = np.minimum(F1[i, 1:], acc + C[1:])
    total = float(min(F0[N, M], F1[N, M]))

    # ---- backward ----
    B0 = np.full((N + 1, M + 1), INF)
    B1 = np.full((N + 1, M + 1), INF)
    Bg0 = np.full((N + 1, M + 1), INF)
    Bg1 = np.full((N + 1, M + 1), INF)
    Cend = C[M] - C            # trailing skips from j to the end
    B0[N] = Cend; B1[N] = Cend
    for i in range(N - 1, -1, -1):
        for a, b in shapes:
            if (i, a, b) not in GC:
                continue
            js, c = GC[(i, a, b)]
            nxt = B0[i + a, js + b] + c
            Bg1[i, js] = np.minimum(Bg1[i, js], nxt)
            if i >= 1:
                t = int(btype[i - 1])
                g = np.where(js >= 1, gapc[t][np.maximum(js - 1, 0)], INF)
                Bg0[i, js] = np.minimum(Bg0[i, js], nxt + g)
        D0 = np.full(M + 1, INF); D1 = np.full(M + 1, INF)
        if pieces[i].first:
            w = pieces[i].word
            il = word_last[w] + 1
            D0 = B0[il] + del_cost[w]
            D1 = B1[il] + del_cost[w]
        Y1 = np.minimum(Bg1[i], D1)
        # B1[i][j] = min_{j' >= j} (Y1[j'] + C[j'] - C[j])
        rev = np.minimum.accumulate((Y1 + C)[::-1])[::-1]
        B1[i] = rev - C
        sk = np.r_[skc + B1[i, 1:], INF]
        B0[i] = np.minimum(np.minimum(Bg0[i], D0), sk)
    # consistency: B1[0,0] == total (up to float error)

    # ---- best path (greedy on exact cost-to-go) ----
    path = []
    i, j, f = 0, 0, 1
    eps = 1e-6 * max(1.0, abs(total))
    while not (i == N and j == M):
        here = (B1 if f else B0)[i, j]
        moved = False
        # group moves
        for a, b in shapes:
            if (i, a, b) not in GC or j + b > M:
                continue
            js, c = GC[(i, a, b)]
            if j > js[-1]:
                continue
            cc = c[j]
            if f == 0:
                if i == 0 or j == 0:
                    continue
                cc += gapc[int(btype[i - 1])][j - 1]
            if abs(cc + B0[i + a, j + b] - here) <= eps:
                path.append(("g", i, i + a, j, j + b))
                i, j, f = i + a, j + b, 0
                moved = True
                break
        if moved:
            continue
        if i < N and pieces[i].first:
            w = pieces[i].word
            il = word_last[w] + 1
            if abs(del_cost[w] + (B1 if f else B0)[il, j] - here) <= eps:
                path.append(("d", i, il, j, j))
                i = il
                continue
        if j < M and abs(skc[j] + B1[i, j + 1] - here) <= eps:
            path.append(("s", i, i, j, j + 1))
            j, f = j + 1, 1
            continue
        if i == N and j < M:          # trailing skips (cost already in Cend)
            path.append(("s", i, i, j, j + 1))
            j, f = j + 1, 1
            continue
        raise RuntimeError("best-path recovery failed")

    # ---- word boxes on the best path ----
    piece_span = {}           # piece -> (s_start, s_end, fused_left, fused_right)
    groups, skipped = [], []
    obs_p, obs_g, obs_f = [], [], {0: [0, 0], 1: [0, 0]}
    last_text_end, last_piece = None, None
    for kind, ia, ib, ja, jb in path:
        if kind == "s":
            skipped.append(ja)
            last_text_end = None
            continue
        if kind == "d":
            continue
        groups.append((ia, ib, ja, jb))
        Sa, Sb = float(s0[ja]), float(s1[jb - 1])
        a = ib - ia
        if a == 1:
            piece_span[ia] = (Sa, Sb, False, False)
            obs_p.append((pieces[ia], Sb - Sa))
        else:
            # fused pieces: cuts by width share, snapped to the weakest core column nearby
            ws = mu[ia:ib] / max(1e-9, mu[ia:ib].sum())
            cuts = Sa + np.cumsum(ws)[:-1] * (Sb - Sa)
            edges = [Sa]
            for k, x in enumerate(cuts):
                half = 0.5 * model.unit
                lo, hi = int(max(Sa + 1, x - half)), int(min(Sb - 1, x + half))
                if hi > lo and hi < len(li.colprof):
                    seg = li.colprof[lo:hi + 1]
                    xs_ = int(lo + np.argmin(seg))
                    edges.append(float(xs_) + 0.5)
                else:
                    edges.append(float(x))
            edges.append(Sb)
            for k in range(a):
                piece_span[ia + k] = (edges[k], edges[k + 1], k > 0, k < a - 1)
            for k in range(ia, ib - 1):
                obs_f[int(btype[k])][0] += 1
                obs_f[int(btype[k])][1] += 1
        if last_piece is not None and last_text_end is not None and ia == last_piece + 1:
            t = int(btype[ia - 1])
            obs_g.append((t, Sa - last_text_end))
            obs_f[t][1] += 1
        last_text_end, last_piece = Sb, ib - 1
    # ---- margins (min-marginals) ----
    words = []
    unit = model.unit
    for w in range(nwords):
        p0, p1 = wfirst_of[w], word_last[w]
        if p0 not in piece_span or p1 not in piece_span:
            words.append(dict(deleted=True, s0=None, s1=None, margin=0.0, z=0.0, fused=False))
            continue
        ws0, we1 = piece_span[p0][0], piece_span[p1][1]
        fused_l, fused_r = piece_span[p0][2], piece_span[p1][3]
        tol = P["POS_TOL"] * unit
        margin = INF
        if want_margins:
            # end alternatives: nodes (p1+1, j, flag 0): piece p1 ends at run j-1
            Ecost = F0[p1 + 1, 1:] + B0[p1 + 1, 1:]
            far = np.abs(s1 - we1) > tol
            if far.any():
                margin = min(margin, float(Ecost[far].min()) - total)
            # start alternatives: piece p0 begins at run j
            Scost = np.minimum(F0[p0, :M] + Bg0[p0, :M], F1[p0, :M] + Bg1[p0, :M])
            far = np.abs(s0 - ws0) > tol
            if far.any():
                margin = min(margin, float(Scost[far].min()) - total)
            # deletion alternative
            Dw = min(float(np.min(F0[p0] + B0[p1 + 1])), float(np.min(F1[p0] + B1[p1 + 1]))) + del_cost[w]
            margin = min(margin, Dw - total)
        # width z of the word (sum over its pieces)
        m_ = cmu[p1 + 1] - cmu[p0] + 0.0
        v_ = cvar[p1 + 1] - cvar[p0]
        # include expected in-word gaps
        E = we1 - ws0
        gin = math.exp(model.gap[0][0]) - model.sw
        nin = sum(1 for k in range(p0, p1) if True)
        z = (E - (m_ + nin * max(0.0, gin))) / math.sqrt(v_ + nin * (0.5 * gin) ** 2 + 1e-9)
        fused = fused_l or fused_r
        valley_ok = True
        if fused:
            for x, isf in ((ws0, fused_l), (we1, fused_r)):
                if isf:
                    k = int(x)
                    med = float(np.median(li.colprof[li.colprof > 0])) if (li.colprof > 0).any() else 1.0
                    if 0 <= k < len(li.colprof) and li.colprof[k] > P["FUSE_VALLEY"] * med:
                        valley_ok = False
        words.append(dict(deleted=False, s0=ws0, s1=we1, margin=float(margin), z=float(z), fused=bool(fused),
                          valley_ok=bool(valley_ok)))
    # observations for re-estimation: only pieces of confident words
    conf_words = {w for w, d in enumerate(words) if not d["deleted"] and d["margin"] >= P["LEARN_MARGIN"]}
    obs_p = [(p, E) for p, E in obs_p if p.word in conf_words]
    return LineAlignment(total, words, groups, skipped, N, obs_p, obs_g, obs_f)


def DEL_ALL(pieces):
    nw = (pieces[-1].word + 1) if pieces else 0
    return P["DEL_WORD"] * nw + P["DEL_PER_PIECE"] * len(pieces)


# ---------------------------------------------------------------------------
# Boxes in page coordinates
# ---------------------------------------------------------------------------
def _quad_from(li: LineInk, ua, ub, v0, v1, pad):
    us = np.array([ua - pad, ub + pad, ub + pad, ua - pad])
    vs = np.array([v0 - pad, v0 - pad, v1 + pad, v1 + pad])
    x, y = li.geom.xy(us, vs)
    return [(float(a), float(b)) for a, b in zip(x, y)]


def snap_words(li: LineInk, intervals: list, sw: float, unit: float, hc: float | None = None) -> list:
    """Word boxes snapped to ink. ``intervals``: per word (s_a, s_b) on the
    core-band runs, or None. Each component of the line goes to the word
    holding most of its core pixels (components without core pixels, such as
    dots: the word containing their centre, if within half a letter); a
    component whose core pixels fall in two words (fused words) is split at
    the cut. Ink in no word (skipped runs, stray marks) is in no box.
    Returns per word a quad (page px) or None."""
    out = [None] * len(intervals)
    live = [(k, a, b) for k, iv in enumerate(intervals) if iv is not None for a, b in [iv]]
    if not live or not len(li.u):
        return out
    s = li.uref - li.u
    starts = np.array([a for _, a, _ in live]); ends = np.array([b for _, _, b in live])
    o = np.argsort(starts)
    starts, ends = starts[o], ends[o]
    ks = [live[i][0] for i in o]
    def which(x):
        i = np.searchsorted(starts, x, side="right") - 1
        ok = (i >= 0) & (x <= ends[np.clip(i, 0, None)] + 0.5)
        return np.where(ok, i, -1)
    pw = which(s)                          # per pixel, word by its own column (core interval)
    ncomp = int(li.comp.max()) + 1 if len(li.comp) else 0
    # majority word of each component's core pixels
    comp_word = np.full(ncomp, -1)
    split = np.zeros(ncomp, bool)
    cm = li.core & (pw >= 0)
    if cm.any():
        key = li.comp[cm] * (len(ks) + 1) + pw[cm]
        cnt = np.bincount(key, minlength=ncomp * (len(ks) + 1)).reshape(ncomp, len(ks) + 1)
        comp_word = np.where(cnt.sum(1) > 0, np.argmax(cnt, 1), -1)
        tot = cnt.sum(1)
        second = np.sort(cnt, 1)[:, -2] if cnt.shape[1] >= 2 else np.zeros(ncomp)
        split = (tot > 0) & (second >= 0.15 * tot)
    # components with no core pixels in any word: centroid column
    has_core = np.bincount(li.comp[li.core], minlength=ncomp) > 0 if li.core.any() else np.zeros(ncomp, bool)
    cs = np.bincount(li.comp, s, minlength=ncomp) / np.maximum(1, np.bincount(li.comp, minlength=ncomp))
    nocore = (comp_word < 0) & ~has_core
    if hc and has_core.any():
        # detached ink (dots, marks) only within the vertical range of the
        # line's own core-attached ink (font-independent glyph zone)
        cv = np.bincount(li.comp, li.v, minlength=ncomp) / np.maximum(1, np.bincount(li.comp, minlength=ncomp))
        vin = li.v[has_core[li.comp]]
        lo_, hi_ = np.quantile(vin, P["DOT_Q"]), np.quantile(vin, 1 - P["DOT_Q"])
        nocore &= (cv >= lo_ - P["DOT_PAD"] * hc) & (cv <= hi_ + P["DOT_PAD"] * hc)
    if nocore.any():
        cw = which(cs[nocore])
        # allow half a letter of slack around the word
        miss = cw < 0
        if miss.any():
            x = cs[nocore][miss]
            d_lo = np.abs(x[:, None] - starts[None, :]); d_hi = np.abs(x[:, None] - ends[None, :])
            d = np.minimum(d_lo, d_hi)
            nb = np.argmin(d, 1)
            cw[miss] = np.where(d[np.arange(len(nb)), nb] <= 0.5 * unit, nb, -1)
        comp_word[np.nonzero(nocore)[0]] = cw
    pix_word = np.where(split[li.comp], pw, comp_word[li.comp])
    pad = 0.0
    for i, k in enumerate(ks):
        sel = pix_word == i
        if sel.sum() < 2:
            continue
        uu, vv = li.u[sel], li.v[sel]
        out[k] = _quad_from(li, float(uu.min()) - 0.5, float(uu.max()) + 0.5, float(vv.min()) - 0.5,
                            float(vv.max()) + 0.5, pad)   # pixel centres +- 0.5: the exact ink bounds
    return out


def line_quad(li: LineInk, s_a: float, s_b: float, sw: float):
    """The line box over reading interval [s_a, s_b] (status 'line')."""
    ua, ub = li.uref - s_b, li.uref - s_a
    sel = (li.u >= ua - 0.5) & (li.u <= ub + 0.5)
    if sel.sum() >= 3:
        v0, v1 = float(np.quantile(li.v[sel], 0.002)), float(np.quantile(li.v[sel], 0.998))
    else:
        v0, v1 = li.base_v - 3 * sw, li.base_v + 2 * sw
    return _quad_from(li, ua, ub, v0, v1, 0.5 * sw)


# ---------------------------------------------------------------------------
# Page-level placement (line-broken text on given line geometries)
# ---------------------------------------------------------------------------
@dataclass
class LineJob:
    tokens: list               # logical tokens
    ink: LineInk | None        # None: no detected line for this text line
    trusted: bool = True       # line assignment trusted
    la_margin: float | None = None


def initial_unit(jobs: list[LineJob]) -> float:
    rs = []
    for jb in jobs:
        if jb.ink is None or jb.ink.M == 0 or not jb.tokens:
            continue
        n = sum(letters(t) for t in jb.tokens) + 0.6 * (len(jb.tokens) - 1)
        ext = float(jb.ink.s1[-1] - jb.ink.s0[0])
        ink_cols = float(np.sum(jb.ink.s1 - jb.ink.s0))
        rs.append(min(ext, ink_cols * 1.6) / max(1.0, n))
    return float(np.median(rs)) if rs else 15.0


def place_jobs(ev_sw: float, jobs: list[LineJob], *, nottext: bool = True, width_mode: str = "free",
               iters: int | None = None, model: WidthModel | None = None, hc: float | None = None) -> tuple[list[list[dict]], WidthModel, dict]:
    """Arm G on every line job. Returns per job a list of per-logical-token
    results dict(status, quad, line_quad, margin, z, reason), the width model
    and diagnostics."""
    iters = P["ITERS"] if iters is None else iters
    unit = initial_unit(jobs)
    if model is None:
        model = WidthModel(unit, ev_sw, hc)
    masses = [jb.ink.mass for jb in jobs if jb.ink is not None and jb.ink.M]
    mref = float(np.median(np.concatenate(masses))) if masses else 100.0
    prepared = []
    for jb in jobs:
        pcs, order = line_pieces(jb.tokens) if jb.tokens else ([], [])
        prepared.append((pcs, order))
    diag = dict(unit0=round(unit, 2), mref=round(mref, 1), rounds=[])
    aligns = [None] * len(jobs)
    for it in range(iters + 1):
        op, og, of = [], [], {0: [0, 0], 1: [0, 0]}
        for k, (jb, (pcs, order)) in enumerate(zip(jobs, prepared)):
            if jb.ink is None or not pcs:
                aligns[k] = None
                continue
            al = align_line(jb.ink, pcs, model, mref, nottext=nottext, want_margins=True)
            aligns[k] = al
            if jb.trusted:
                op += al.obs_pieces; og += al.obs_gaps
                for t in (0, 1):
                    of[t][0] += al.obs_fuse[t][0]; of[t][1] += al.obs_fuse[t][1]
        diag["rounds"].append(dict(model=model.snapshot(), n_obs=len(op),
                                   cost=round(sum(a.cost for a in aligns if a is not None), 1)))
        if it < iters:
            model.refit(op, og, of, mode=width_mode)
    out = []
    for jb, (pcs, order), al in zip(jobs, prepared, aligns):
        res = [None] * len(jb.tokens)
        if al is None:
            for ti in range(len(jb.tokens)):
                res[ti] = dict(status="unresolved", quad=None, line_quad=None, margin=None, z=None,
                               reason="no detected line for this text line")
            out.append(res)
            continue
        li = jb.ink
        txt = [d for d in al.words if not d["deleted"]]
        if txt:
            la, lb = min(d["s0"] for d in txt), max(d["s1"] for d in txt)
        elif li.M:
            la, lb = float(li.s0[0]), float(li.s1[-1])
        else:
            la = lb = None
        lq = line_quad(li, la, lb, ev_sw) if la is not None else None
        quads = snap_words(li, [None if d["deleted"] else (d["s0"], d["s1"]) for d in al.words], ev_sw, model.unit,
                           model.hc)
        for w, d in enumerate(al.words):
            ti = order[w]
            if not jb.trusted:
                res[ti] = dict(status="unresolved", quad=None, line_quad=None, margin=d.get("margin"), z=d.get("z"),
                               reason="line assignment not trusted")
                continue
            if d["deleted"]:
                res[ti] = dict(status="line", quad=None, line_quad=lq, margin=0.0, z=None,
                               reason="no ink matched this word (deleted)")
                continue
            other_ok = abs(d["z"]) <= P["LOC_Z"] and d.get("valley_ok", True) and quads[w] is not None
            ok = d["margin"] >= P["LOC_MARGIN"] and other_ok
            q = quads[w]
            if q is None:
                ok = False
            if ok:
                res[ti] = dict(status="located", quad=q, line_quad=lq, margin=d["margin"], z=d["z"], reason="",
                               other_ok=True)
            else:
                why = ("margin" if d["margin"] < P["LOC_MARGIN"] else
                       "width" if abs(d["z"]) > P["LOC_Z"] else "fused cut not at a valley")
                res[ti] = dict(status="line", quad=q, line_quad=lq, margin=d["margin"], z=d["z"],
                               reason=f"word position not trusted ({why})", other_ok=other_ok)
        out.append(res)
    diag["model"] = model.snapshot()
    return out, model, diag


# ---------------------------------------------------------------------------
# §4.2 line text: text lines -> detected lines
# ---------------------------------------------------------------------------
def robust_extent(li: LineInk, mref: float) -> float:
    """Core ink extent with small isolated end runs trimmed (not text, likely)."""
    if li.M == 0:
        return 0.0
    a, b = 0, li.M - 1
    gaps = li.s0[1:] - li.s1[:-1]
    mg = float(np.median(gaps)) if len(gaps) else 0.0
    while a < b and li.mass[a] < 0.3 * mref and (li.s0[a + 1] - li.s1[a]) > 2 * mg + 1:
        a += 1
    while b > a and li.mass[b] < 0.3 * mref and (li.s0[b] - li.s1[b - 1]) > 2 * mg + 1:
        b -= 1
    return float(li.s1[b] - li.s0[a])


def wide_gap_counts(dets: list, mref: float) -> tuple[np.ndarray, dict]:
    """Per detected line, the number of gaps between core runs that are wider
    than the page's word-gap threshold (Otsu on log gap widths over all lines,
    inside each line's robust extent). Returns (counts or None, info): None
    when the page's gaps are not bimodal (between-class share of variance
    < GAPCOUNT_ETA), e.g. tight typewriter print, where gap counts mean nothing."""
    allg = []
    per = []
    for d in dets:
        if d.M < 2:
            per.append(np.zeros(0)); continue
        g = d.s0[1:] - d.s1[:-1]
        per.append(g)
        allg.append(g)
    if not allg:
        return None, dict(eta=0.0)
    x = np.log(np.concatenate(allg) + 1.0)
    if len(x) < 20:
        return None, dict(eta=0.0)
    xs = np.sort(x)
    best, thr = -1.0, None
    tot_var = float(np.var(xs)) + 1e-12
    for k in range(5, len(xs) - 5):
        a, b = xs[:k], xs[k:]
        bv = len(a) * len(b) / len(xs) ** 2 * (a.mean() - b.mean()) ** 2
        if bv > best:
            best, thr = bv, (xs[k - 1] + xs[k]) / 2
    eta = best / tot_var
    info = dict(eta=round(float(eta), 3), threshold_px=round(float(math.exp(thr) - 1.0), 1))
    if eta < P["GAPCOUNT_ETA"]:
        return None, info
    counts = np.array([int(np.sum(np.log(g + 1.0) > thr)) for g in per])
    return counts, info


def text_units(tokens: list[str]) -> float:
    return sum(letters(t) for t in tokens) + 0.6 * max(0, len(tokens) - 1)


def assign_text_lines(dets: list[LineInk], kinds: list[str], texts: list[list[str]], S: float, mref: float) -> dict:
    """Monotone alignment of text lines (top to bottom) to detected lines.

    Returns dict(assign=[det index or None per text line], margin=[...],
    det_text=[text index or None per detected line], scale=...)."""
    m, n = len(texts), len(dets)
    X = np.array([robust_extent(d, mref) for d in dets])
    Y = pair_dy(dets)
    U = np.array([text_units(t) for t in texts])
    if m == 0 or n == 0:
        return dict(assign=[None] * m, margin=[0.0] * m, det_text=[None] * n, scale=None)
    guess = float(np.median(X[X > 0]) / max(1e-9, np.median(U))) if (X > 0).any() else 15.0
    skipD = np.array([P["LA_SKIP_DET"] if k == "text" else P["LA_SKIP_DET_NONTEXT"] for k in kinds])
    gc, ginfo = wide_gap_counts(dets, mref)
    GC = 0.0
    if gc is not None:
        nt = np.array([max(0, len(t) - 1) for t in texts], np.float64)
        gsd = P["GAPCOUNT_SD"] + P["GAPCOUNT_REL"] * nt
        GC = 0.5 * ((nt[:, None] - gc[None, :]) / gsd[:, None]) ** 2
    best = None
    for sc in guess * np.exp(np.linspace(math.log(0.6), math.log(1.6), P["LA_SCALES"])):
        pred = sc * U
        sd = P["LA_SD_REL"] * pred[:, None] + P["LA_SD_ABS"] * sc
        Mc = 0.5 * ((X[None, :] - pred[:, None]) / sd) ** 2 + np.log(sd) - math.log(P["LA_SD_ABS"] * sc) + GC
        r = _la_dp(Mc, Y, skipD, S)
        if best is None or r["cost"] < best["cost"] - 1e-9:
            best = dict(r, scale=float(sc))
    best["gapcount"] = ginfo
    return best


def pair_dy(dets: list) -> np.ndarray:
    """dY[a, b] = vertical distance from line a down to line b, measured at a
    common x (the middle of their x-overlap, or between them): on a sloped
    page a short line and a long one differ by tan(slope) x their x offset
    when each is measured at its own centre."""
    n = len(dets)
    D = np.zeros((n, n))
    for a in range(n):
        ga = dets[a].geom
        for b in range(n):
            if a == b:
                continue
            gb = dets[b].geom
            lo, hi = max(ga.x0, gb.x0), min(ga.x1, gb.x1)
            x = (lo + hi) / 2
            D[a, b] = float(gb.centre_y(x)) - float(ga.centre_y(x))
    return D


def _spacing_cost(dy: float, S: float) -> float:
    d = dy / max(1e-9, S)
    if d < P["LA_CLOSE"]:
        return 1e6
    if d < P["LA_NEAR"]:
        return P["LA_NEAR_COST"]
    return 0.0


def _la_dp(Mc: np.ndarray, Y: np.ndarray, skipD: np.ndarray, S: float) -> dict:
    """DP over (text line i matched to detected line j); forward + backward
    for per-line margins. Skipped detected lines cost skipD; a missing text
    line costs LA_SKIP_TEXT (at most 3 consecutive)."""
    m, n = Mc.shape
    cs = np.r_[0.0, np.cumsum(skipD)]          # skips of dets [a, b) = cs[b]-cs[a]
    ST = P["LA_SKIP_TEXT"]
    dY = (Y if Y.ndim == 2 else (Y[None, :] - Y[:, None])) / max(1e-9, S)   # dY[jp, j]: from jp down to j
    SP = np.where(dY < P["LA_CLOSE"], 1e6, np.where(dY < P["LA_NEAR"], P["LA_NEAR_COST"], 0.0))
    tri = np.triu(np.ones((n, n), bool), 1)            # jp < j
    # transition cost from (ip, jp) to (i, j), excluding the miss term: skips between + spacing
    Tr = np.where(tri, (cs[None, :n] - cs[1:, None]) + SP, INF)   # [jp, j]
    Fm = np.full((m, n), INF)
    for i in range(m):
        best = i * ST + cs[:n]
        for ip in range(max(0, i - 3), i):
            c = np.min(Fm[ip][:, None] + Tr, axis=0) + (i - ip - 1) * ST
            best = np.minimum(best, c)
        Fm[i] = best + Mc[i]
    Bm = np.full((m, n), INF)
    for i in range(m - 1, -1, -1):
        best = (m - 1 - i) * ST + (cs[n] - cs[1:])
        for inx in range(i + 1, min(m, i + 4)):
            c = np.min(Tr + (Mc[inx] + Bm[inx])[None, :], axis=1) + (inx - i - 1) * ST
            best = np.minimum(best, c)
        Bm[i] = best
    T = Fm + Bm
    none_cost = m * ST + cs[n]
    total = min(float(T.min()), none_cost)
    assign, margin = [], []
    for i in range(m):
        j = int(np.argmin(T[i]))
        bi = float(T[i, j])
        if bi <= total + 1e-6:
            alt = np.delete(T[i], j)
            miss_alt = total - Mc[i, j] + ST + skipD[j]
            a2 = min(float(alt.min()) if len(alt) else INF, miss_alt)
            assign.append(j)
            margin.append(a2 - total)
        else:
            assign.append(None)
            margin.append(bi - total)
    used = set()
    for i, j in enumerate(assign):
        if j is not None and j in used:
            assign[i] = None
            margin[i] = 0.0
        elif j is not None:
            used.add(j)
    det_text = [None] * n
    for i, j in enumerate(assign):
        if j is not None:
            det_text[j] = i
    return dict(cost=total, assign=assign, margin=margin, det_text=det_text)


# ---------------------------------------------------------------------------
# §4.2 variant: unbroken text (paragraphs) -> detected lines
# ---------------------------------------------------------------------------
def distribute_words(dets: list[LineInk], kinds: list[str], paragraphs: list[list[str]], S: float, mref: float,
                     block_width: float | None = None) -> dict:
    """Assign a paragraph token stream to detected lines (top to bottom).

    DP over (tokens consumed k, detected line j): line j takes tokens k..k'-1
    (cost: its ink extent against the tokens' natural width, a justified
    non-final line may be up to DW_STRETCH wider), or carries no text. A line
    should not hold the end of one paragraph and the start of the next.
    Returns dict(spans=[(k0, k1) or None per detected line], scale)."""
    toks, para_end = [], []
    for p in paragraphs:
        for i, t in enumerate(p):
            toks.append(t)
            para_end.append(i == len(p) - 1)
    K, n = len(toks), len(dets)
    X = np.array([robust_extent(d, mref) for d in dets])
    Y = np.array([d.y_mid for d in dets])
    if K == 0 or n == 0:
        return dict(spans=[None] * n, tokens=toks, para_end=para_end, scale=None, cost=0.0)
    u = np.array([letters(t) for t in toks])
    cu = np.r_[0.0, np.cumsum(u)]
    pe = np.array(para_end)
    cpe = np.r_[0, np.cumsum(pe)]
    skipD = np.array([P["LA_SKIP_DET"] if k == "text" else P["LA_SKIP_DET_NONTEXT"] for k in kinds])
    tl = X[X > 0]
    guess_tokens_per_line = max(1.0, K / max(1, (tl > 0.3 * np.median(tl)).sum()))
    guess = float(np.median(tl)) / max(1e-9, float(np.mean(u)) * guess_tokens_per_line + 0.6 * (guess_tokens_per_line - 1))
    Tmax = int(min(K, max(6, 2.5 * guess_tokens_per_line)))
    gc, ginfo = wide_gap_counts(dets, mref)
    best = None
    coarse = guess * np.exp(np.linspace(math.log(0.6), math.log(1.6), 11))
    for sc in coarse:
        r = _dw_dp(X, Y, skipD, cu, cpe, pe, K, n, sc, Tmax, S, gc)
        if best is None or r["cost"] < best["cost"] - 1e-9:
            best = dict(r, scale=float(sc))
    step = math.log(1.6 / 0.6) / 10
    for f in np.linspace(-step, step, 9):
        sc = best["scale"] * math.exp(f)
        r = _dw_dp(X, Y, skipD, cu, cpe, pe, K, n, sc, Tmax, S, gc)
        if r["cost"] < best["cost"] - 1e-9:
            best = dict(r, scale=float(sc))
    best.update(tokens=toks, para_end=para_end, gapcount=ginfo)
    return best


def _dw_dp(X, Y, skipD, cu, cpe, pe, K, n, sc, Tmax, S, gc=None):
    """F[j, k]: best cost after detected lines 0..j-1 with k tokens placed.
    Line j takes tokens k0..k1-1 (vectorised over all k0 and lengths), or
    carries no text; a token may also be placed on no line (DW_SKIP_TOKEN)."""
    F = np.full((n + 1, K + 1), INF)
    F[0, 0] = 0.0
    k0s = np.arange(K + 1)[:, None]
    ts = np.arange(1, Tmax + 1)[None, :]
    k1 = k0s + ts                                         # [k0, t]
    valid = k1 <= K
    k1c = np.minimum(k1, K)
    units = (cu[k1c] - cu[k0s]) + 0.6 * (ts - 1)
    final = pe[np.maximum(k1c - 1, 0)]
    inner = cpe[np.maximum(k1c - 1, 0)] - cpe[k0s]
    hi = np.where(final, 0.03, P["DW_STRETCH"])
    back = {}
    for j in range(n):
        nat = sc * units
        r = np.log(max(X[j], 1.0) / np.maximum(nat, 1.0))
        ex = np.where(r > hi, r - hi, np.where(r < -0.03, -0.03 - r, 0.0))
        c = 0.5 * (ex / P["DW_SD"]) ** 2 + P["DW_PARA_CROSS"] * inner
        if gc is not None:
            nt = (ts - 1).astype(np.float64)
            c = c + 0.5 * ((nt - gc[j]) / (P["GAPCOUNT_SD"] + P["GAPCOUNT_REL"] * nt)) ** 2
        tot = np.where(valid, F[j][:, None] + c, INF)    # [k0, t]
        # scatter-min into k1
        flat_k1 = k1c[valid]; flat_v = tot[valid]; flat_k0 = np.broadcast_to(k0s, k1.shape)[valid]
        best = np.full(K + 1, INF); arg = np.full(K + 1, -1)
        o = np.lexsort((flat_v, flat_k1))
        fk1, fv, fk0 = flat_k1[o], flat_v[o], flat_k0[o]
        first = np.r_[True, fk1[1:] != fk1[:-1]]
        best[fk1[first]] = fv[first]; arg[fk1[first]] = fk0[first]
        skip = F[j] + skipD[j]
        take = best < skip
        F[j + 1] = np.where(take, best, skip)
        back[j] = (np.where(take, arg, -2))              # -2: line j skipped (k unchanged)
        # tokens placed on no line, at this row
        tok = np.full(K + 1, -1)
        for k in range(K):
            if F[j + 1, k] + P["DW_SKIP_TOKEN"] < F[j + 1, k + 1]:
                F[j + 1, k + 1] = F[j + 1, k] + P["DW_SKIP_TOKEN"]
                tok[k + 1] = k
        back[(j, "tok")] = tok
    cost = float(F[n, K])
    spans = [None] * n
    j, k = n, K
    while j > 0:
        tok = back[(j - 1, "tok")]
        while k > 0 and tok[k] >= 0 and F[j, k] == F[j, tok[k]] + P["DW_SKIP_TOKEN"]:
            k = int(tok[k])
        a_ = int(back[j - 1][k])
        if a_ >= 0:
            spans[j - 1] = (a_, k)
            k = a_
        j -= 1
    return dict(cost=cost, spans=spans)


# ---------------------------------------------------------------------------
# Page pipelines
# ---------------------------------------------------------------------------
def l1_lines(ev: PageEvidence, text_only: bool = False):
    """L1 lines (all kinds unless text_only) in top-to-bottom order, as
    (geometry, kind)."""
    out = []
    for L in ev.lf["lines"]:
        if text_only and L.kind != "text":
            continue
        if L.x1 - L.x0 < 1:
            continue
        out.append((ShearGeom(L), L.kind))
    if out:
        xref = float(np.median([(g.x0 + g.x1) / 2 for g, _ in out if g.x1 - g.x0 > 0.3 * ev.W] or
                               [(g.x0 + g.x1) / 2 for g, _ in out]))
        out.sort(key=lambda gk: float(gk[0].centre_y(min(max(xref, gk[0].x0), gk[0].x1))) +
                 float(gk[0].centre_y(xref)) * 0.0 + _slope_at(gk[0], xref))
    return out


def _slope_at(g, xref):
    """y of the line at xref, extrapolated linearly from its nearest end
    (lines are ordered at one common x, so slope does not reorder them)."""
    x = min(max(xref, g.x0), g.x1)
    if x == xref:
        return 0.0
    h = max(1.0, 0.25 * (g.x1 - g.x0))
    k = (float(g.centre_y(x + h)) - float(g.centre_y(x - h))) / (2 * h) if g.x1 - g.x0 > 2 else 0.0
    return k * (xref - x)


def page_mref(inks: list[LineInk]) -> float:
    ms = [li.mass for li in inks if li.M]
    return float(np.median(np.concatenate(ms))) if ms else 100.0


def flatten(results_per_line, texts):
    return [[r for r in res] for res in results_per_line]


def place_given_lines(ev: PageEvidence, geoms: list, texts: list[list[str]], *, nottext=True,
                      width_mode="free") -> dict:
    """Experiment A: text line i is printed on geometry i (no assignment)."""
    comps = assign_components(ev, geoms)
    inks = [line_ink(ev, g, c) for g, c in zip(geoms, comps)]
    jobs = [LineJob(list(t), li, True, None) for t, li in zip(texts, inks)]
    res, model, diag = place_jobs(ev.sw, jobs, nottext=nottext, width_mode=width_mode, hc=ev.hc)
    return dict(results=res, model=model, diag=diag, inks=inks, assign=list(range(len(texts))))


def place_assigned_lines(ev: PageEvidence, dets: list, kinds: list, texts: list[list[str]], *, nottext=True,
                         width_mode="free") -> dict:
    """Experiment B: text lines aligned to detected lines by length (§4.2)."""
    comps = assign_components(ev, dets)
    inks = [line_ink(ev, g, c) for g, c in zip(dets, comps)]
    mref = page_mref(inks)
    la = assign_text_lines(inks, kinds, texts, ev.S, mref)
    jobs = []
    for i, t in enumerate(texts):
        j = la["assign"][i]
        trusted = j is not None and la["margin"][i] >= P["LA_MARGIN"]
        jobs.append(LineJob(list(t), inks[j] if j is not None else None, trusted, la["margin"][i]))
    res, model, diag = place_jobs(ev.sw, jobs, nottext=nottext, width_mode=width_mode, hc=ev.hc)
    return dict(results=res, model=model, diag=diag, inks=inks, assign=la["assign"], la=la, kinds=kinds)


def place_unbroken(ev: PageEvidence, dets: list, kinds: list, paragraphs: list[list[str]], *, nottext=True,
                   width_mode="free", refine=True) -> dict:
    """Unbroken text (Markdown paragraphs): tokens distributed over detected
    lines, line boundaries refined with arm G cost, then arm G per line.
    Returns per-token results in stream order (``results_flat``)."""
    comps = assign_components(ev, dets)
    inks = [line_ink(ev, g, c) for g, c in zip(dets, comps)]
    mref = page_mref(inks)
    dw = distribute_words(inks, kinds, paragraphs, ev.S, mref)
    toks = dw["tokens"]
    spans = list(dw["spans"])
    used = [j for j, sp in enumerate(spans) if sp is not None and sp[1] > sp[0]]
    # provisional width model from the first distribution
    jobs = [LineJob(toks[spans[j][0]:spans[j][1]], inks[j]) for j in used]
    _, model, _ = place_jobs(ev.sw, jobs, nottext=nottext, width_mode=width_mode, hc=ev.hc)
    boundary_margin = {}
    if refine and len(used) >= 2:
        R = P["DW_REFINE"]
        for a_, b_ in zip(used, used[1:]):
            k0, k1 = spans[a_]
            _, k2 = spans[b_]
            if spans[b_][0] != k1:
                continue
            costs = {}
            for d in range(-R, R + 1):
                cut = k1 + d
                if cut <= k0 or cut >= k2:
                    continue
                c = 0.0
                for j, (x0, x1) in ((a_, (k0, cut)), (b_, (cut, k2))):
                    pcs, _ = line_pieces(toks[x0:x1])
                    c += align_line(inks[j], pcs, model, mref, nottext=nottext, want_margins=False).cost
                costs[cut] = c
            if not costs:
                continue
            best = min(costs, key=costs.get)
            spans[a_] = (k0, best)
            spans[b_] = (best, k2)
            others = [v for k, v in costs.items() if k != best]
            boundary_margin[best] = (min(others) - costs[best]) if others else INF
    jobs, owner = [], []
    for j in used:
        k0, k1 = spans[j]
        jobs.append(LineJob(toks[k0:k1], inks[j], True, None))
        owner.append((j, k0, k1))
    res, model, diag = place_jobs(ev.sw, jobs, nottext=nottext, width_mode=width_mode, hc=ev.hc)
    flat = [dict(status="unresolved", quad=None, line_quad=None, margin=None, z=None, reason="token on no line", det=None)
            for _ in toks]
    for (j, k0, k1), rr in zip(owner, res):
        for t, r in enumerate(rr):
            k = k0 + t
            r = dict(r, det=j)
            # boundary tokens whose line is ambiguous are never shown
            amb = False
            for cut, mg in boundary_margin.items():
                if mg < P["DW_MARGIN"] and (k == cut or k == cut - 1):
                    amb = True
            if amb and r["status"] != "unresolved":
                r = dict(r, status="unresolved", quad=None, line_quad=None,
                         reason="line of this token is ambiguous (line boundary)")
            flat[k] = r
    return dict(results_flat=flat, tokens=toks, spans=spans, model=model, diag=diag, inks=inks, kinds=kinds,
                boundary_margin=boundary_margin)
