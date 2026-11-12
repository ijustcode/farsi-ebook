#!/usr/bin/env python3
"""Classical local line finder for photographed / warped pages (plan §4.1, Step 3 arm L1).

Evaluation code only: production (``farsi2epub/``) never imports ``tests/``.
numpy + PyMuPDF only, no API, deterministic.

Pipeline (every length is expressed in measured page quantities: stroke width
``sw``, median glyph-component height ``hc`` and line spacing ``S``):

1. **Measure** (``measure``): background (paper) field from block upper
   percentiles, its gradient, stroke width (median horizontal ink run), median
   component height, line spacing (autocorrelation of narrow-strip row
   profiles, so slope does not blur it), share of ink in oversized components.
   The local slope field is added after line finding (``slope_field``).
2. **Binarise** (``binarise``): divide by the paper field, then Sauvola
   (window from the measured glyph height; a first pass with a page-relative
   window supplies that height).
3. **Mask non-text ink** (``mask_nontext``), conservatively, by size and
   isolation alone:
   * big: a component at least ``BIG_H`` spacings tall and ``BIG_W`` wide, or
     ``TALL_H`` tall at any width (stamps, signatures, margin bars);
   * rule: a thin horizontal component or a collinear chain of thin pieces at
     least ``RULE_LEN`` spacings long (underlines, typed underscores). A rule
     that touches text is part of a text component and is not masked (a
     pixel-level rule cut was tried and removed: it deleted the baseline joins
     of long bold words);
   * isolated: a component entirely outside the text block by ``ISO_OUT``
     spacings with no other ink within ``ISO_CLEAR`` spacings (margin marks).
   Anything near the block at glyph size is never masked (it may be
   punctuation or a narrow letter). Big-component pixels that fall inside a
   text line found without them, next to that line's own ink, are restored
   (text under a stamp is text).
4. **Text-block edges** (``fit_block_edges``): right (line start, RTL) and left
   edges fitted from all lines together as straight lines in y, robustly
   (asymmetric trimming: indented or short lines fall inside the block and are
   not used; ink protruding beyond the fit is flagged).
5. **Lines by local following** (``find_lines``): overlapping vertical strips
   ``STRIP_W`` spacings wide; per-strip smoothed row profiles; prominent
   peaks; peaks linked across strips by order-preserving (monotone) dynamic
   programming against each track's predicted position (lines cannot cross),
   tolerating gaps of up to ``GAP_MAX`` spacings (verse hemistichs, holes);
   collinear x-disjoint tracks joined; then every component is assigned to the
   nearest track and each line's centre curve is re-measured from its own ink
   only. Output per line: a bending centre curve (dense polyline), an upper
   and lower offset (band), its polygon and convex hull, ink extent, and the
   shear rectification ``to_strip``/``from_strip``/``rectify``.

Rectification is a vertical shear along the centre curve: strip column
u = x - x0, strip row v = y - c(x) + up. It is exactly invertible, keeps page x
(word boundaries found in the strip are page x), and shears glyphs by at most
tan(slope) (0.06 at 3.5 deg).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

VERSION = "line_finder/L1-1"

# ---------------------------------------------------------------------------
# Free parameters (all relative to measured quantities unless noted)
# ---------------------------------------------------------------------------
PARAMS = dict(
    BG_BLOCK_FRAC=1 / 24,   # paper-field block, fraction of min(H, W) (before text size is known)
    BG_PCT=90,              # paper = this percentile of each block
    SAUVOLA_K=0.25,         # Sauvola k on the paper-normalised image (R = 0.5)
    SAUVOLA_W1_FRAC=1 / 40,  # first-pass window, fraction of min(H, W)
    SAUVOLA_W_HC=2.5,       # second-pass window = this x median component height
    PAPER_MAX=0.88,         # ink must also be darker than this share of the local paper
    BIG_H=2.2, BIG_W=1.5,   # big component: >= BIG_H x S tall and >= BIG_W x S wide
    TALL_H=3.0,             # ... or >= TALL_H x S tall at any width
    RULE_THIN_SW=3.0,       # rule piece: height <= RULE_THIN_SW x stroke ...
    RULE_THIN_HC=0.5,       # ... and <= RULE_THIN_HC x glyph height
    RULE_ASPECT=2.5,        # ... and width >= RULE_ASPECT x height
    RULE_LEN=1.5,           # rule (piece or chain) length >= RULE_LEN x S
    RULE_GAP=0.25,          # chain link: horizontal gap <= RULE_GAP x S
    ISO_OUT=1.0,            # isolated: outside the text block by >= ISO_OUT x S
    ISO_CLEAR=1.0,          # ... and no other component within ISO_CLEAR x S
    ISO_MAX_H=1.5,          # ... and no taller than ISO_MAX_H x S (else it is 'big'/'tall')
    BLOCK_QUANT=0.005,      # text block x-range: ink column quantiles (each side)
    STRIP_W=2.5,            # strip width x S
    STRIP_STEP=0.5,         # strip step, fraction of strip width
    PROF_SIGMA=0.10,        # profile smoothing sigma x S
    PEAK_SEP=0.5,           # min peak separation x S
    PEAK_PROM=0.30,         # min prominence, fraction of the peak
    PEAK_FLOOR=0.10,        # min peak, fraction of the page's median strong peak
    MATCH_TOL=0.30,         # max |observed - predicted| when linking, x S
    GAP_MAX=5.0,            # longest gap a track may bridge, x S
    SLOPE_NEIGH=5,          # predicted slope = median over this many nearest established tracks
    JOIN_TOL=0.25,          # collinear join of x-disjoint tracks, x S
    DUP_TOL=0.30,           # tracks overlapping in x and within this x S are one line
    CURVE_QUAD_LEN=8.0,     # centre curve: quadratic when the line is >= this x S long (else straight)
    CURVE_REJ=0.15,         # centre curve: drop strip peaks off the fit by > this x S
    ASSIGN_TOL=0.60,        # component -> line: |dy| <= ASSIGN_TOL x S
    LINE_MIN_MASS=0.02,     # drop a line holding < this share of the median line ink ...
    LINE_MIN_COMPS=2,       # ... unless it has at least this many glyph-sized components
    BAND_FRAC=0.04,         # band edge: own-ink row profile falls below this share of its peak ...
    BAND_GAP=0.15,          # ... for more than BAND_GAP x S rows (dots/marks across a short gap stay in)
    BAND_MAX=0.8,
    RULE_BAND_HC=0.75,      # a found line whose band is thinner than this x glyph height is a rule, not text           # band never wider than BAND_MAX x S on either side
    OVERLAY_CLOSE=0.15,     # close gaps up to this x S in a big component before filling its holes
    OVERLAY_IN=0.9,         # a line with >= this share of its ink inside a big component's filled interior is not text
    BAND_NEIGH=0.5,         # band edge never past this share of the distance to a neighbour
    EDGE_TOL=0.5,
    EDGE_LONG=0.5,          # only lines at least this share of the median line length vote for the block edges
    EDGE_MAX_DEG=10.0,      # block edges steeper than this from vertical are not considered
    MARGIN_OUT=0.8,         # a line with >= this share of its extent beyond a block edge (+EDGE_TOL) is a margin mark           # block-edge inlier tolerance x S
    RESTORE_REACH=1.0,      # restore big-component pixels within this x S of the line's own ink
)
P = PARAMS


# ---------------------------------------------------------------------------
# Small numpy helpers
# ---------------------------------------------------------------------------
def load_gray(path) -> np.ndarray:
    import fitz
    pm = fitz.Pixmap(str(path))
    if pm.alpha:
        pm = fitz.Pixmap(pm, 0)
    if pm.n != 1:
        pm = fitz.Pixmap(fitz.csGRAY, pm)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width).astype(np.float32)


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edges clipped (integral image)."""
    a = a.astype(np.float64)
    H, W = a.shape
    c = np.zeros((H + 1, W + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) + r + 1, 0, H)
    x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) + r + 1, 0, W)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    n = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s / n


def smooth1d(v: np.ndarray, sigma: float) -> np.ndarray:
    v = np.asarray(v, np.float64)
    if sigma <= 0.3:
        return v.copy()
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def resize_bilinear(a: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = a.shape
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    tmp = np.stack([np.interp(xs, np.arange(w), a[i]) for i in range(h)])
    return np.stack([np.interp(ys, np.arange(h), tmp[:, j]) for j in range(W)], axis=1)


def convex_hull(pts):
    """Andrew's monotone chain; returns hull points counter-clockwise (image coords)."""
    P_ = sorted(set((float(x), float(y)) for x, y in pts))
    if len(P_) <= 2:
        return P_
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P_:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P_):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


# ---------------------------------------------------------------------------
# Connected components (8-connected) by run-length union-find, vectorised
# ---------------------------------------------------------------------------
@dataclass
class Components:
    r: np.ndarray       # per run: row
    s: np.ndarray       # per run: start col
    e: np.ndarray       # per run: end col (exclusive)
    lab: np.ndarray     # per run: component id
    n: int
    area: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    cx: np.ndarray
    cy: np.ndarray

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0

    def paint(self, shape, keep: np.ndarray) -> np.ndarray:
        """Boolean image of the components where keep[comp] is True."""
        out = np.zeros(shape, bool)
        sel = keep[self.lab]
        r, s, e = self.r[sel], self.s[sel], self.e[sel]
        if not len(r):
            return out
        ln = e - s
        tot = int(ln.sum())
        base = np.repeat(r * shape[1] + s, ln)
        off = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln)
        out.ravel()[base + off] = True
        return out


def components(mask: np.ndarray) -> Components:
    H, W = mask.shape
    m = np.zeros((H, W + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1)
    rs, ss = np.nonzero(d == 1)
    re_, es = np.nonzero(d == -1)
    # np.nonzero is row-major, so starts and ends pair up in order
    r, s, e = rs.astype(np.int64), ss.astype(np.int64), es.astype(np.int64)
    n = len(r)
    if n == 0:
        z = np.zeros(0)
        return Components(r, s, e, np.zeros(0, np.int64), 0, z, z, z, z, z, z, z)
    K = W + 4
    kstart = r * K + s
    kend = r * K + e
    # runs a (row r) and b (row r+1) touch (8-conn) iff b.s <= a.e and b.e >= a.s
    lo = np.searchsorted(kend, (r + 1) * K + s, side="left")
    hi = np.searchsorted(kstart, (r + 1) * K + e, side="right") - 1
    hi = np.minimum(hi, n - 1)
    ok = (hi >= lo) & (lo < n)
    ok &= np.where(lo < n, r[np.minimum(lo, n - 1)] == r + 1, False)
    a_idx = np.nonzero(ok)[0]
    cnt = (hi[a_idx] - lo[a_idx] + 1)
    A = np.repeat(a_idx, cnt)
    B = np.repeat(lo[a_idx], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    lab = np.arange(n)
    if len(A):
        while True:
            m_ = np.minimum(lab[A], lab[B])
            new = lab.copy()
            np.minimum.at(new, A, m_)
            np.minimum.at(new, B, m_)
            while True:
                nn = new[new]
                if np.array_equal(nn, new):
                    break
                new = nn
            if np.array_equal(new, lab):
                break
            lab = new
    uniq, lab = np.unique(lab, return_inverse=True)
    nc = len(uniq)
    ln = (e - s).astype(np.float64)
    area = np.bincount(lab, ln, nc)
    x0 = np.full(nc, np.iinfo(np.int64).max); np.minimum.at(x0, lab, s)
    x1 = np.zeros(nc, np.int64); np.maximum.at(x1, lab, e)
    y0 = np.full(nc, np.iinfo(np.int64).max); np.minimum.at(y0, lab, r)
    y1 = np.zeros(nc, np.int64); np.maximum.at(y1, lab, r + 1)
    cy = np.bincount(lab, ln * (r + 0.5), nc) / area
    cx = np.bincount(lab, ln * (s + e) / 2.0, nc) / area
    return Components(r, s, e, lab, nc, area, x0, x1, y0, y1, cx, cy)


# ---------------------------------------------------------------------------
# 1-2. Measurement and binarisation
# ---------------------------------------------------------------------------
def paper_field(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    H, W = gray.shape
    B = max(16, int(round(min(H, W) * P["BG_BLOCK_FRAC"])))
    nby, nbx = math.ceil(H / B), math.ceil(W / B)
    pad = np.pad(gray, ((0, nby * B - H), (0, nbx * B - W)), mode="edge")
    blocks = pad.reshape(nby, B, nbx, B).transpose(0, 2, 1, 3).reshape(nby, nbx, B * B)
    g = np.percentile(blocks, P["BG_PCT"], axis=2)
    # upper envelope over 3x3 blocks (a block under a stamp is not paper), then smooth
    gp = np.pad(g, 1, mode="edge")
    mx = np.max(np.stack([gp[i:i + nby, j:j + nbx] for i in range(3) for j in range(3)]), axis=0)
    gp = np.pad(mx, 1, mode="edge")
    sm = np.mean(np.stack([gp[i:i + nby, j:j + nbx] for i in range(3) for j in range(3)]), axis=0)
    bg = resize_bilinear(sm, H, W)
    bg = np.maximum(bg, 1.0)
    med = float(np.median(sm))
    info = dict(block_px=B, paper_median=round(med, 2),
                background_gradient=round(float((np.percentile(sm, 95) - np.percentile(sm, 5)) / max(med, 1e-6)), 4))
    return bg, info


def sauvola(norm: np.ndarray, win: int) -> np.ndarray:
    r = max(2, win // 2)
    m = box_mean(norm, r)
    s = np.sqrt(np.maximum(box_mean(norm * norm, r) - m * m, 0.0))
    T = m * (1.0 + P["SAUVOLA_K"] * (s / 0.5 - 1.0))
    return (norm < T) & (norm < P["PAPER_MAX"])


def stroke_width(cc: Components) -> float:
    """Mode of the horizontal ink-run length histogram (vertical strokes; the
    median is pulled up by horizontal strokes and joins)."""
    ln = (cc.e - cc.s).astype(np.int64)
    if not len(ln):
        return 1.0
    h = np.bincount(np.minimum(ln, 200))[1:].astype(np.float64)
    h = np.convolve(np.pad(h, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return float(np.argmax(h) + 1)


def glyph_height(cc: Components, sw: float) -> float:
    sel = (cc.area >= 2 * sw * sw) & (cc.h >= 1.5 * sw)
    return float(np.median(cc.h[sel])) if sel.any() else max(4.0, 4 * sw)


def text_block_x(cc: Components, keep: np.ndarray, W: int) -> tuple[float, float]:
    """x-range holding all but BLOCK_QUANT of the kept ink on each side."""
    sel = keep[cc.lab]
    if not sel.any():
        return 0.0, float(W)
    col = np.zeros(W + 1)
    np.add.at(col, cc.s[sel], 1.0)
    np.add.at(col, cc.e[sel], -1.0)
    col = np.cumsum(col)[:W]
    c = np.cumsum(col)
    tot = c[-1]
    q = P["BLOCK_QUANT"]
    return float(np.searchsorted(c, q * tot)), float(np.searchsorted(c, (1 - q) * tot) + 1)


def line_spacing(ink: np.ndarray, xa: float, xb: float, hc: float) -> tuple[float, dict]:
    """Line pitch from the summed autocorrelation of narrow-strip row profiles."""
    H, W = ink.shape
    ns = 8
    edges = np.linspace(xa, xb, ns + 1).astype(int)
    lo, hi = max(2, int(1.0 * hc)), min(H // 3, int(10 * hc) + 2)
    acs = np.zeros(hi)
    nfft = 1 << int(math.ceil(math.log2(2 * H)))
    for i in range(ns):
        p = ink[:, edges[i]:edges[i + 1]].sum(1).astype(np.float64)
        p = smooth1d(p, max(1.0, 0.1 * hc))
        p -= p.mean()
        f = np.fft.rfft(p, nfft)
        ac = np.fft.irfft(f * np.conj(f), nfft)[:hi]
        if ac[0] > 0:
            acs += ac / ac[0]
    seg = acs[lo:hi]
    if not len(seg):
        return 3.0 * hc, dict(method="fallback")
    top = float(seg.max())
    lag = int(np.argmax(seg)) + lo
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.5 * top:
            lag = i + lo
            break
    # sub-sample peak
    if 1 <= lag < hi - 1:
        cm, c0, cp = acs[lag - 1], acs[lag], acs[lag + 1]
        den = cm - 2 * c0 + cp
        off = 0.5 * (cm - cp) / den if abs(den) > 1e-12 else 0.0
        lagf = lag + max(-0.5, min(0.5, off))
    else:
        lagf = float(lag)
    return float(lagf), dict(method="strip_autocorrelation", strips=ns, peak=round(float(acs[lag] / ns), 3))


# ---------------------------------------------------------------------------
# 3. Masking
# ---------------------------------------------------------------------------
def mask_nontext(cc: Components, S: float, sw: float, hc: float, block: tuple[float, float]):
    """Returns (masked bool per component, reason per component)."""
    n = cc.n
    reason = np.array([""] * n, dtype=object)
    h, w = cc.h.astype(np.float64), cc.w.astype(np.float64)
    big = ((h >= P["BIG_H"] * S) & (w >= P["BIG_W"] * S)) | (h >= P["TALL_H"] * S)
    reason[big] = "big"
    # rules: thin horizontal pieces, alone or chained
    thin = (~big) & (h <= min(P["RULE_THIN_SW"] * sw, P["RULE_THIN_HC"] * hc)) & (w >= P["RULE_ASPECT"] * h)
    rule = np.zeros(n, bool)
    idx = np.nonzero(thin)[0]
    if len(idx):
        order = idx[np.argsort(cc.x0[idx], kind="stable")]
        parent = {int(i): int(i) for i in order}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        xs0, xs1 = cc.x0[order], cc.x1[order]
        cys = cc.cy[order]
        for ii, a in enumerate(order):
            j = ii + 1
            while j < len(order) and xs0[j] <= xs1[ii] + P["RULE_GAP"] * S:
                if abs(cys[j] - cys[ii]) <= 1.5 * sw:
                    ra, rb = find(int(a)), find(int(order[j]))
                    if ra != rb:
                        parent[ra] = rb
                j += 1
        groups: dict = {}
        for a in order:
            groups.setdefault(find(int(a)), []).append(int(a))
        for g in groups.values():
            gx0 = min(cc.x0[i] for i in g); gx1 = max(cc.x1[i] for i in g)
            if gx1 - gx0 >= P["RULE_LEN"] * S:
                rule[g] = True
    reason[rule] = "rule"
    # isolated marks outside the block
    xa, xb = block
    out_gap = np.maximum(xa - cc.x1, cc.x0 - xb)
    cand = np.nonzero((out_gap >= P["ISO_OUT"] * S) & (h <= P["ISO_MAX_H"] * S) & ~big & ~rule)[0]
    iso = np.zeros(n, bool)
    clear = P["ISO_CLEAR"] * S
    for i in cand:
        dx = np.maximum(0, np.maximum(cc.x0 - cc.x1[i], cc.x0[i] - cc.x1))
        dy = np.maximum(0, np.maximum(cc.y0 - cc.y1[i], cc.y0[i] - cc.y1))
        near = (dx <= clear) & (dy <= clear)
        near[i] = False
        # its own fragments (a cross drawn as two strokes) do not count as text nearby
        near &= ~(np.maximum(cc.x0 - xb, xa - cc.x1) >= P["ISO_OUT"] * S)
        if not near.any():
            iso[i] = True
    reason[iso] = "isolated"
    return big | rule | iso, reason


# ---------------------------------------------------------------------------
# 5. Line following
# ---------------------------------------------------------------------------
def strip_layout(xa: float, xb: float, S: float):
    sw_ = max(8.0, P["STRIP_W"] * S)
    step = max(4.0, sw_ * P["STRIP_STEP"])
    span = xb - xa
    n = max(1, int(math.ceil(max(0.0, span - sw_) / step)) + 1)
    starts = xa + np.arange(n) * (span - sw_) / max(1, n - 1) if n > 1 else np.array([xa])
    return [(float(s), float(s + sw_)) for s in starts]


def find_peaks(p: np.ndarray, sep: float, prom_frac: float, floor: float):
    """Local maxima with prominence >= prom_frac x peak, separated by >= sep."""
    n = len(p)
    if n < 3:
        return []
    d = np.diff(p)
    cand = np.nonzero((np.concatenate([[1.0], d]) > 0) & (np.concatenate([d, [-1.0]]) <= 0))[0]
    peaks = [int(i) for i in cand if p[i] >= floor]
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        # prominence relative to neighbouring peaks
        worst, wi = None, None
        for k, i in enumerate(peaks):
            lv = p[peaks[k - 1]:i + 1].min() if k > 0 else p[:i + 1].min()
            rv = p[i:peaks[k + 1] + 1].min() if k + 1 < len(peaks) else p[i:].min()
            prom = p[i] - max(lv, rv)
            bad = prom < prom_frac * p[i]
            close = (k > 0 and i - peaks[k - 1] < sep) or (k + 1 < len(peaks) and peaks[k + 1] - i < sep)
            if bad or close:
                score = prom / max(p[i], 1e-9)
                if worst is None or score < worst or (score == worst and p[i] < p[peaks[wi]]):
                    worst, wi = score, k
        if wi is not None:
            # drop the weakest offender, but if it is "close" and stronger than its
            # neighbour, drop the neighbour instead
            i = peaks[wi]
            nb = [k for k in (wi - 1, wi + 1) if 0 <= k < len(peaks) and abs(peaks[k] - i) < sep]
            if nb:
                k2 = min(nb, key=lambda k: p[peaks[k]])
                if p[peaks[k2]] < p[i]:
                    wi = k2
            peaks.pop(wi)
            changed = True
    return peaks


def _subpix(p, i):
    if 0 < i < len(p) - 1:
        cm, c0, cp = p[i - 1], p[i], p[i + 1]
        den = cm - 2 * c0 + cp
        if abs(den) > 1e-12:
            return i + max(-0.5, min(0.5, 0.5 * (cm - cp) / den))
    return float(i)


@dataclass
class Track:
    pts: list = field(default_factory=list)     # [(strip index, x centre, y, strength)]
    lost: int = 0

    def last(self):
        return self.pts[-1]

    def slope(self):
        if len(self.pts) < 2:
            return None
        q = self.pts[-4:]
        xs = np.array([a[1] for a in q]); ys = np.array([a[2] for a in q])
        if np.ptp(xs) < 1:
            return None
        return float(np.polyfit(xs, ys, 1)[0])


def monotone_assign(pred: list, obs: list, tol: float):
    """Order-preserving matching of predicted to observed y (both ascending).
    Cost: (d/tol)^2 per match, 1 per unmatched item on either side."""
    m, n = len(pred), len(obs)
    INF = 1e18
    D = np.full((m + 1, n + 1), INF)
    bt = np.zeros((m + 1, n + 1), np.int8)
    D[0, :] = np.arange(n + 1); D[:, 0] = np.arange(m + 1)
    bt[0, 1:] = 2; bt[1:, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best, b = D[i - 1, j] + 1.0, 1
            if D[i, j - 1] + 1.0 < best:
                best, b = D[i, j - 1] + 1.0, 2
            d = abs(pred[i - 1] - obs[j - 1])
            if d <= tol:
                c = D[i - 1, j - 1] + (d / tol) ** 2
                if c < best:
                    best, b = c, 3
            D[i, j], bt[i, j] = best, b
    pairs = []
    i, j = m, n
    while i > 0 or j > 0:
        b = bt[i, j]
        if b == 3:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def link_tracks(strips, peaks_per_strip, S):
    """Follow peaks from the right edge (line starts, RTL) leftwards."""
    tol = P["MATCH_TOL"] * S
    step = (strips[0][1] - strips[0][0]) * P["STRIP_STEP"] if strips else S
    max_lost = max(1, int(math.ceil(P["GAP_MAX"] * S / max(step, 1.0))))
    order = list(range(len(strips)))[::-1]
    active: list[Track] = []
    done: list[Track] = []
    for k in order:
        xc = (strips[k][0] + strips[k][1]) / 2
        obs = peaks_per_strip[k]          # [(y, strength)] ascending y
        # predictions
        act = sorted(active, key=lambda t: t.last()[2])
        slopes = [t.slope() if len(t.pts) >= 3 else None for t in act]
        good = [(t.last()[2], s) for t, s in zip(act, slopes) if s is not None]
        preds = []
        for i, t in enumerate(act):
            # lines are locally parallel: predict with the median slope of the
            # established tracks nearest in y (itself included), not with the
            # track's own recent slope alone (a stray mark can bend it)
            if good:
                near = sorted(good, key=lambda g: abs(g[0] - t.last()[2]))[:P["SLOPE_NEIGH"]]
                s = float(np.median([g[1] for g in near]))
            else:
                s = slopes[i] if slopes[i] is not None else 0.0
            preds.append(t.last()[2] + s * (xc - t.last()[1]))
        # keep predictions ordered (lines do not cross)
        pairs = monotone_assign(preds, [o[0] for o in obs], tol) if act and obs else []
        used_t, used_o = set(), set()
        for i, j in pairs:
            act[i].pts.append((k, xc, obs[j][0], obs[j][1]))
            act[i].lost = 0
            used_t.add(i); used_o.add(j)
        still = []
        for i, t in enumerate(act):
            if i not in used_t:
                t.lost += 1
                if t.lost > max_lost:
                    done.append(t)
                    continue
            still.append(t)
        for j, o in enumerate(obs):
            if j not in used_o:
                still.append(Track(pts=[(k, xc, o[0], o[1])]))
        active = still
    return done + active


@dataclass
class Line:
    xs: np.ndarray            # centre-curve sample x (ascending)
    ys: np.ndarray            # centre-curve y at xs
    x0: float = 0.0           # ink extent
    x1: float = 0.0
    up: float = 0.0           # band above / below the centre curve (px)
    dn: float = 0.0
    up_x: np.ndarray | None = None   # per-sample band (clipped by neighbours)
    dn_x: np.ndarray | None = None
    mass: float = 0.0
    ncomp: int = 0
    comps: list = field(default_factory=list)
    kind: str = "text"
    flags: list = field(default_factory=list)
    track_strips: int = 0

    def y_at(self, x):
        x = np.asarray(x, np.float64)
        y = np.interp(x, self.xs, self.ys)
        if len(self.xs) >= 2:
            sl = (self.ys[1] - self.ys[0]) / max(1e-9, self.xs[1] - self.xs[0])
            sr = (self.ys[-1] - self.ys[-2]) / max(1e-9, self.xs[-1] - self.xs[-2])
            y = np.where(x < self.xs[0], self.ys[0] + sl * (x - self.xs[0]), y)
            y = np.where(x > self.xs[-1], self.ys[-1] + sr * (x - self.xs[-1]), y)
        return y

    # --- rectification (vertical shear along the centre curve) ---
    def to_strip(self, x, y):
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
        return x - self.x0, y - self.y_at(x) + self.up

    def from_strip(self, u, v):
        u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
        x = u + self.x0
        return x, v - self.up + self.y_at(x)

    def strip_shape(self):
        return int(math.ceil(self.up + self.dn)) + 1, int(math.ceil(self.x1 - self.x0)) + 1

    def rectify(self, gray: np.ndarray, fill: float = 255.0) -> np.ndarray:
        """Straight strip image (rows v = 0..up+dn, columns u = 0..x1-x0), bilinear."""
        hh, ww = self.strip_shape()
        u = np.arange(ww, dtype=np.float64)
        v = np.arange(hh, dtype=np.float64)
        X, Y = self.from_strip(u[None, :] + 0 * v[:, None], v[:, None] + 0 * u[None, :])
        H, W = gray.shape
        x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int)
        fx, fy = X - x0, Y - y0
        out = np.zeros_like(X)
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                wgt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                val = np.where(ok, gray[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)], fill)
                out += wgt * val
        return out

    def polygon(self, n: int | None = None):
        """Band polygon around the bending centre curve, clockwise from the top-left."""
        xs = self.sample_x(n)
        yc = self.y_at(xs)
        up = np.interp(xs, self.xs, self.up_x) if self.up_x is not None else np.full(len(xs), self.up)
        dn = np.interp(xs, self.xs, self.dn_x) if self.dn_x is not None else np.full(len(xs), self.dn)
        top = list(zip(xs, yc - up))
        bot = list(zip(xs[::-1], (yc + dn)[::-1]))
        return [(float(a), float(b)) for a, b in top + bot]

    def sample_x(self, n=None):
        if n is None:
            n = max(2, min(64, len(self.xs)))
        return np.linspace(self.x0, self.x1, n)

    def baseline(self, n=None):
        """Centre curve, right -> left (reading order start first)."""
        xs = self.sample_x(n)[::-1]
        return [(float(x), float(y)) for x, y in zip(xs, self.y_at(xs))]

    def angle_deg(self):
        xs = self.sample_x(8)
        k = np.polyfit(xs, self.y_at(xs), 1)[0] if self.x1 - self.x0 > 1 else 0.0
        return float(math.degrees(math.atan(k)))


def _line_profile_peaks(cc, comp_ids, xa, xb, H, S, strips):
    """Per-strip peak of the line's own ink profile (its centre/baseline zone)."""
    sel = np.isin(cc.lab, comp_ids)
    r, s, e = cc.r[sel], cc.s[sel], cc.e[sel]
    out = []
    if not len(r):
        return out
    y_lo, y_hi = int(r.min()), int(r.max()) + 1
    for (a, b) in strips:
        if b < xa or a > xb:
            continue
        ss, ee = np.maximum(s, a), np.minimum(e, b)
        ok = ee > ss
        if not ok.any():
            continue
        prof = np.bincount(r[ok] - y_lo, (ee - ss)[ok], y_hi - y_lo).astype(np.float64)
        if prof.sum() < 1:
            continue
        sm = smooth1d(prof, P["PROF_SIGMA"] * S)
        i = int(np.argmax(sm))
        out.append(((a + b) / 2, y_lo + _subpix(sm, i) + 0.5, float(sm[i]), float(prof.sum())))
    return out


def find_lines(gray: np.ndarray, *, use_mask: bool = True, keep_arrays: bool = False) -> dict:
    """Run the full front end on one grey page image (float or uint8, HxW)."""
    t0 = time.perf_counter()
    gray = np.asarray(gray, np.float32)
    H, W = gray.shape
    bg, binfo = paper_field(gray)
    norm = np.clip(gray / bg, 0.0, 1.5)
    # pass 1: page-relative window -> glyph size
    w1 = max(9, int(min(H, W) * P["SAUVOLA_W1_FRAC"]) | 1)
    ink1 = sauvola(norm, w1)
    cc1 = components(ink1)
    sw1 = stroke_width(cc1)
    hc1 = glyph_height(cc1, sw1)
    # pass 2: window from the glyph height
    w2 = max(9, int(P["SAUVOLA_W_HC"] * hc1) | 1)
    ink = sauvola(norm, w2)
    cc = components(ink)
    sw = stroke_width(cc)
    hc = glyph_height(cc, sw)
    allk = np.ones(cc.n, bool)
    xa0, xb0 = text_block_x(cc, allk, W)
    S, sinfo = line_spacing(ink, xa0, xb0, hc)
    # masking
    if use_mask:
        masked, reason = mask_nontext(cc, S, sw, hc, (xa0, xb0))
    else:
        masked, reason = np.zeros(cc.n, bool), np.array([""] * cc.n, dtype=object)
    keep = ~masked
    xa, xb = text_block_x(cc, keep, W)
    if use_mask:
        # isolation was judged against the unmasked block; re-judge once against the cleaned block
        m2, r2 = mask_nontext(cc, S, sw, hc, (xa, xb))
        masked, reason = m2, r2
        keep = ~masked
        xa, xb = text_block_x(cc, keep, W)
    big_ink = float(cc.area[reason == "big"].sum()) / max(1.0, float(cc.area.sum()))
    line_ink = cc.paint((H, W), keep)

    big_ids0 = np.nonzero(reason == "big")[0]
    interiors = filled_interiors(cc, big_ids0, H, W, P["OVERLAY_CLOSE"] * S) if len(big_ids0) else None
    lines, dbg = _follow(cc, keep, line_ink, xa, xb, S, sw, hc, H, W, interiors)

    # restore big-component pixels inside found text lines, next to the line's own ink
    restored_img = np.zeros((H, W), bool)
    big_ids = np.nonzero(reason == "big")[0]
    if use_mask and len(big_ids) and lines:
        restored_img = _restore_big(cc, big_ids, lines, S, H, W)
        line_ink = line_ink | restored_img
    restored = int(restored_img.sum())
    masked_px = (cc.paint((H, W), masked) & ~line_ink) if use_mask else np.zeros((H, W), bool)
    edges = fit_block_edges(lines, S, cc)
    if margin_lines(lines, edges, S):
        edges = fit_block_edges(lines, S, cc)
    slopes = slope_field(lines, H, W)
    meas = dict(
        width=W, height=H, stroke_px=round(sw, 2), glyph_height_px=round(hc, 2),
        line_spacing_px=round(S, 2), spacing_info=sinfo, text_block_x=[round(xa, 1), round(xb, 1)],
        oversized_ink_share=round(big_ink, 4), sauvola_windows=[w1, w2], ink_share=round(float(ink.mean()), 4),
        **binfo, slope_field=slopes,
    )
    masked_summary = {}
    for rsn in ("big", "rule", "isolated"):
        ids = np.nonzero(reason == rsn)[0]
        masked_summary[rsn] = dict(components=int(len(ids)), ink_px=int(cc.area[ids].sum()))
    masked_summary["restored_px"] = int(restored)
    masked_summary["masked_px_final"] = int(masked_px.sum())
    out = dict(version=VERSION, params=dict(PARAMS), use_mask=use_mask, measurements=meas,
               masked=masked_summary, edges=edges,
               lines=lines, debug=dbg, seconds=round(time.perf_counter() - t0, 3))
    if keep_arrays:
        out.update(ink=ink, masked_px=masked_px, line_ink=line_ink, cc=cc, reason=reason, keep=keep)
    return out


def _restore_big(cc, big_ids, lines, S, H, W) -> np.ndarray:
    """Pixels of big (masked) components that lie inside a text line's band,
    within RESTORE_REACH x S of that line's ink, growing along the line (text
    under or touching a stamp is text). Band: up to half way to the
    neighbouring text line, at most BAND_MAX x S."""
    img = np.zeros((H, W), bool)
    big_px = cc.paint((H, W), np.isin(np.arange(cc.n), big_ids))
    reach = P["RESTORE_REACH"] * S
    text = [L for L in lines if L.kind == "text"]
    cap = P["BAND_MAX"] * S
    for i, L in enumerate(text):
        lo_x, hi_x = L.x0, L.x1
        for _ in range(6):
            xs = np.arange(max(0, int(lo_x - reach)), min(W, int(math.ceil(hi_x + reach))))
            if not len(xs):
                break
            yc = L.y_at(xs)
            up = np.full(len(xs), cap); dn = np.full(len(xs), cap)
            if i > 0:
                up = np.minimum(up, 0.5 * np.abs(yc - text[i - 1].y_at(xs)))
            if i + 1 < len(text):
                dn = np.minimum(dn, 0.5 * np.abs(text[i + 1].y_at(xs) - yc))
            got_lo, got_hi = None, None
            for x, y, u_, d_ in zip(xs, yc, up, dn):
                a, b = max(0, int(y - u_)), min(H, int(math.ceil(y + d_)))
                if b > a:
                    col = big_px[a:b, x]
                    if col.any():
                        img[a:b, x] |= col
                        got_lo = x if got_lo is None else got_lo
                        got_hi = x
            if got_lo is None:
                break
            # grow along the line only through restored ink adjacent to the current extent
            new_lo, new_hi = min(lo_x, got_lo), max(hi_x, got_hi + 1)
            if new_lo == lo_x and new_hi == hi_x:
                break
            lo_x, hi_x = new_lo, new_hi
    return img


def _follow(cc, keep, line_ink, xa, xb, S, sw, hc, H, W, interiors=None):
    strips = strip_layout(xa, xb, S)
    sig = P["PROF_SIGMA"] * S
    profs = []
    for a, b in strips:
        prof = line_ink[:, int(a):int(math.ceil(b))].sum(1).astype(np.float64)
        profs.append(smooth1d(prof, sig))
    allmax = []
    raw = []
    for p in profs:
        pk = find_peaks(p, P["PEAK_SEP"] * S, P["PEAK_PROM"], 0.0)
        raw.append(pk)
        allmax += [p[i] for i in pk]
    strong = float(np.median([v for v in allmax if v >= np.percentile(allmax, 50)])) if allmax else 1.0
    floor = P["PEAK_FLOOR"] * strong
    peaks = []
    for p in profs:
        pk = find_peaks(p, P["PEAK_SEP"] * S, P["PEAK_PROM"], floor)
        peaks.append([(_subpix(p, i) + 0.5, float(p[i])) for i in pk])
    tracks = link_tracks(strips, peaks, S)
    tracks = _merge_duplicates(tracks, S)
    tracks = _join_collinear(tracks, S)
    # provisional curves from tracks
    prov = []
    for t in tracks:
        pts = sorted(t.pts, key=lambda a: a[1])
        xs = np.array([a[1] for a in pts]); ys = np.array([a[2] for a in pts])
        prov.append((xs, ys, len(pts)))
    prov = _fill_slopes(prov, S)
    # assign components to the nearest provisional curve
    ids = np.nonzero(keep)[0]
    ncur = len(prov)
    if ncur == 0:
        return [], dict(strips=len(strips), tracks=0)
    cy = cc.cy[ids]; cx = cc.cx[ids]
    Y = np.stack([_eval_curve(c, cx) for c in prov])          # ncur x ncomp
    # a curve only claims components within its x-span plus the gap allowance
    span_ok = np.stack([(cx >= c[0].min() - 0.75 * P["STRIP_W"] * S) & (cx <= c[0].max() + 0.75 * P["STRIP_W"] * S)
                        for c in prov])
    dist = np.where(span_ok, np.abs(Y - cy[None, :]), np.inf)
    best = np.argmin(dist, axis=0)
    bd = dist[best, np.arange(len(ids))]
    ok = bd <= P["ASSIGN_TOL"] * S
    members: dict = {}
    for ci, b, o in zip(ids, best, ok):
        if o:
            members.setdefault(int(b), []).append(int(ci))
    orphans = [int(c) for c, o in zip(ids, ok) if not o]
    lines = []
    for k, (xs, ys, npts, *_r) in enumerate(prov):
        comps = members.get(k, [])
        if not comps:
            continue
        comps = np.array(comps)
        mass = float(cc.area[comps].sum())
        x0 = float(cc.x0[comps].min()); x1 = float(cc.x1[comps].max())
        pk = _line_profile_peaks(cc, comps, x0, x1, H, S, strips)
        if len(pk) >= 1:
            pxs = np.array([a[0] for a in pk]); pys = np.array([a[1] for a in pk])
            wts = np.array([a[3] for a in pk])
            L = Line(xs=pxs, ys=pys)
            if len(pxs) >= 2:
                # robust weighted polynomial: quadratic for long lines (a bow),
                # straight otherwise; points off by > CURVE_REJ x S are dropped
                deg = 2 if (len(pxs) >= 5 and x1 - x0 >= P["CURVE_QUAD_LEN"] * S) else 1
                ok = np.ones(len(pxs), bool)
                for _ in range(3):
                    c = np.polyfit(pxs[ok], pys[ok], deg, w=np.sqrt(wts[ok]))
                    res_ = np.abs(pys - np.polyval(c, pxs))
                    new_ok = res_ <= max(2.0, P["CURVE_REJ"] * S)
                    if new_ok.sum() < deg + 2 or (new_ok == ok).all():
                        break
                    ok = new_ok
                xs_d = np.linspace(min(x0, pxs.min()), max(x1, pxs.max()), max(2, min(32, int((x1 - x0) / (0.5 * S)) + 2)))
                L = Line(xs=xs_d, ys=np.polyval(c, xs_d))
        else:
            L = Line(xs=xs, ys=ys)
        if len(L.xs) == 1:
            sl = _eval_slope(prov[k])
            L.xs = np.array([L.xs[0] - S, L.xs[0] + S]); L.ys = np.array([L.ys[0] - sl * S, L.ys[0] + sl * S])
        L.x0, L.x1, L.mass, L.ncomp, L.comps, L.track_strips = x0, x1, mass, len(comps), comps.tolist(), npts
        lines.append(L)
    lines.sort(key=lambda L: float(L.y_at((L.x0 + L.x1) / 2)))
    _classify(lines, cc, S, sw, hc, interiors)
    _bands(lines, cc, S)
    for L in lines:
        if L.kind == "text" and (L.up + L.dn) < P["RULE_BAND_HC"] * hc:
            L.kind = "rule"
            L.flags.append("band thinner than a glyph: underline or rule")
    dbg = dict(strips=len(strips), tracks=len(tracks), peak_floor=round(floor, 2),
               orphan_components=len(orphans), orphan_ink_px=int(cc.area[orphans].sum()) if orphans else 0)
    return lines, dbg


def _eval_curve(c, x):
    xs, ys = c[0], c[1]
    sl = c[3]
    if len(xs) == 1:
        return ys[0] + sl * (x - xs[0])
    y = np.interp(x, xs, ys)
    y = np.where(x < xs[0], ys[0] + c[4] * (x - xs[0]), y)
    y = np.where(x > xs[-1], ys[-1] + c[5] * (x - xs[-1]), y)
    return y


def _eval_slope(c):
    return c[3]


def _fill_slopes(prov, S):
    """Each curve gets an end slope; short curves borrow from neighbours."""
    own = []
    for xs, ys, n in prov:
        if len(xs) >= 2 and np.ptp(xs) > 0:
            own.append(float(np.polyfit(xs, ys, 1)[0]))
        else:
            own.append(None)
    mids = [float(np.mean(ys)) for xs, ys, n in prov]
    good = [i for i, s in enumerate(own) if s is not None and len(prov[i][0]) >= 3]
    out = []
    for i, (xs, ys, n) in enumerate(prov):
        s = own[i]
        if s is None or len(xs) < 3:
            if good:
                near = sorted(good, key=lambda g: abs(mids[g] - mids[i]))[:3]
                s_nb = float(np.median([own[g] for g in near]))
                s = s_nb if s is None else s_nb
            elif s is None:
                s = 0.0
        if len(xs) >= 3:
            q = min(3, len(xs))
            sl = float(np.polyfit(xs[:q], ys[:q], 1)[0]); sr = float(np.polyfit(xs[-q:], ys[-q:], 1)[0])
        else:
            sl = sr = s
        out.append((xs, ys, n, s, sl, sr))
    return out


def _track_fits(tracks, S):
    """Robust straight fit (a, b) per track: least squares, one round of
    outlier rejection (> 0.15 S); tracks with < 3 points take the median slope
    of the 5 nearest longer tracks."""
    raw = []
    for t in tracks:
        pts = sorted(t.pts, key=lambda a: a[1])
        xs = np.array([a[1] for a in pts]); ys = np.array([a[2] for a in pts])
        raw.append((xs, ys))
    fits = [None] * len(raw)
    for i, (xs, ys) in enumerate(raw):
        if len(xs) >= 3:
            b_, a_ = np.polyfit(xs, ys, 1)
            res = np.abs(ys - (a_ + b_ * xs))
            ok = res <= max(2.0, 0.15 * S)
            if ok.sum() >= 3 and not ok.all():
                b_, a_ = np.polyfit(xs[ok], ys[ok], 1)
            fits[i] = (float(a_), float(b_))
    longs = [(float(np.mean(raw[i][1])), fits[i][1]) for i in range(len(raw)) if fits[i] is not None]
    for i, (xs, ys) in enumerate(raw):
        if fits[i] is None:
            ym = float(np.mean(ys))
            b_ = float(np.median([g[1] for g in sorted(longs, key=lambda g: abs(g[0] - ym))[:5]])) if longs else 0.0
            fits[i] = (float(np.mean(ys - b_ * xs)), b_)
    return raw, fits


def _merge_duplicates(tracks, S):
    """Tracks that share x-range and run within DUP_TOL x S of each other are
    one line (a track lost for a few strips restarts beside itself)."""
    tol = P["DUP_TOL"] * S
    changed = True
    while changed:
        changed = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a = sorted(tracks[i].pts, key=lambda p: p[1]); b = sorted(tracks[j].pts, key=lambda p: p[1])
                xa = np.array([p[1] for p in a]); ya = np.array([p[2] for p in a])
                xb = np.array([p[1] for p in b]); yb = np.array([p[2] for p in b])
                lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
                if hi < lo:
                    continue
                xs = np.unique(np.concatenate([xa[(xa >= lo) & (xa <= hi)], xb[(xb >= lo) & (xb <= hi)]]))
                if np.max(np.abs(np.interp(xs, xa, ya) - np.interp(xs, xb, yb))) <= tol:
                    best = {}
                    for p in a + b:
                        if p[0] not in best or p[3] > best[p[0]][3]:
                            best[p[0]] = p
                    tracks[i].pts = list(best.values())
                    tracks.pop(j)
                    changed = True
                    break
            if changed:
                break
    return tracks


def _join_collinear(tracks, S):
    """Join x-disjoint tracks whose robust fits meet in the gap (long gaps,
    holes under stamps); never across a third track."""
    tol = P["JOIN_TOL"] * S
    tracks = [t for t in tracks if t.pts]
    changed = True
    while changed:
        changed = False
        raw, fits = _track_fits(tracks, S)
        best = None
        for i, (xi, yi) in enumerate(raw):
            for j, (xj, yj) in enumerate(raw):
                if i == j or xi.max() >= xj.min():
                    continue       # i must be entirely left of j
                gap = xj.min() - xi.max()
                if gap > 3 * P["GAP_MAX"] * S:
                    continue
                mid = (xi.max() + xj.min()) / 2
                yi_m = fits[i][0] + fits[i][1] * mid
                yj_m = fits[j][0] + fits[j][1] * mid
                d = abs(yi_m - yj_m)
                if d > tol:
                    continue
                ymid = (yi_m + yj_m) / 2
                blocked = False
                for k, (xk, yk) in enumerate(raw):
                    if k in (i, j):
                        continue
                    if xk.min() <= mid <= xk.max() and abs(np.interp(mid, xk, yk) - ymid) < 0.5 * S:
                        blocked = True
                        break
                if not blocked and (best is None or d < best[0]):
                    best = (d, i, j)
        if best:
            _, i, j = best
            tracks[i].pts += tracks[j].pts
            tracks.pop(j)
            changed = True
    return tracks


def filled_interiors(cc, big_ids, H, W, close_r: float = 0.0) -> np.ndarray:
    """Big components with their holes filled (a stamp ring's inside)."""
    out = np.zeros((H, W), bool)
    for i in big_ids:
        x0, y0, x1, y1 = int(cc.x0[i]), int(cc.y0[i]), int(cc.x1[i]), int(cc.y1[i])
        sel = cc.lab == i
        pad = max(1, int(round(close_r)))
        m = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), bool)
        for r, s_, e in zip(cc.r[sel], cc.s[sel], cc.e[sel]):
            m[r - y0 + pad, s_ - x0 + pad:e - x0 + pad] = True
        # close small breaks (ink dropout in a ring) before filling holes
        mc = box_mean(m.astype(np.float32), pad) > 0 if close_r > 0 else m
        bgc = components(~mc)
        border = (bgc.x0 == 0) | (bgc.y0 == 0) | (bgc.x1 == m.shape[1]) | (bgc.y1 == m.shape[0])
        holes = bgc.paint(m.shape, ~border) if bgc.n else np.zeros_like(m)
        out[y0:y1, x0:x1] |= (m | holes)[pad:pad + y1 - y0, pad:pad + x1 - x0]
    return out


def _classify(lines, cc, S, sw, hc, interiors=None):
    if not lines:
        return
    med = float(np.median([L.mass for L in lines]))
    for L in lines:
        if interiors is not None and L.comps:
            # a line whose own ink lies (>= 90%) inside a stamp's filled interior is the stamp's text
            ids = np.array(L.comps)
            cx = np.clip(cc.cx[ids].astype(int), 0, interiors.shape[1] - 1)
            cy = np.clip(cc.cy[ids].astype(int), 0, interiors.shape[0] - 1)
            inside = float(cc.area[ids][interiors[cy, cx]].sum()) / max(1.0, float(cc.area[ids].sum()))
            if inside >= P["OVERLAY_IN"]:
                L.kind = "overlay"
                L.flags.append("inside a masked stamp/signature")
                continue
        glyphs = sum(1 for c in L.comps if cc.h[c] >= 0.5 * hc)
        if L.mass < P["LINE_MIN_MASS"] * med and glyphs < P["LINE_MIN_COMPS"]:
            L.kind = "noise"
            L.flags.append("too little ink")


def _bands(lines, cc, S):
    """Band above/below each centre curve from the line's own ink offsets,
    never past BAND_NEIGH of the distance to the neighbouring text line."""
    text = [L for L in lines if L.kind == "text"]
    for L in lines:
        sel = np.isin(cc.lab, L.comps)
        r, s, e = cc.r[sel], cc.s[sel], cc.e[sel]
        if len(r):
            ln = (e - s).astype(np.float64)
            off = (r + 0.5) - L.y_at((s + e) / 2.0)
            R = int(math.ceil(P["BAND_MAX"] * S))
            ok = np.abs(off) <= R
            prof = np.bincount(np.rint(off[ok]).astype(int) + R, ln[ok], 2 * R + 1).astype(np.float64)
            prof = smooth1d(prof, max(0.5, 0.03 * S))
            c = int(np.argmax(prof[max(0, R - int(0.3 * S)):R + int(0.3 * S) + 1])) + max(0, R - int(0.3 * S))
            thr = P["BAND_FRAC"] * prof[c]
            gap = max(2, int(P["BAND_GAP"] * S))
            def walk(step):
                i, last, miss = c, c, 0
                while 0 <= i + step < len(prof):
                    i += step
                    if prof[i] >= thr:
                        last, miss = i, 0
                    else:
                        miss += 1
                        if miss > gap:
                            break
                return last
            top, bot = walk(-1), walk(1)
            L.up, L.dn = float(R - top + 1.0), float(bot - R + 1.0)
        else:
            L.up = L.dn = 0.3 * S
        L.up_x = np.full(len(L.xs), L.up); L.dn_x = np.full(len(L.xs), L.dn)
    for i, L in enumerate(text):
        yc = L.y_at(L.xs)
        for nb, side in ((text[i - 1] if i > 0 else None, "up"), (text[i + 1] if i + 1 < len(text) else None, "dn")):
            if nb is None:
                continue
            ov = (L.xs >= nb.x0 - 0.5 * S) & (L.xs <= nb.x1 + 0.5 * S)
            dist = np.abs(nb.y_at(L.xs) - yc)
            lim = P["BAND_NEIGH"] * dist
            if side == "up":
                L.up_x = np.where(ov, np.minimum(L.up_x, lim), L.up_x)
            else:
                L.dn_x = np.where(ov, np.minimum(L.dn_x, lim), L.dn_x)


# ---------------------------------------------------------------------------
# 4. Block edges, slope field
# ---------------------------------------------------------------------------
def _robust_edge(ys, xs, side, tol, use=None):
    """Fit x = a + b*y to line ends by exhaustive pairwise consensus
    (deterministic RANSAC): the straight edge through two line ends that has
    the most line ends within tol, ties broken by the outermost edge (the block
    edge, not an indent); then least squares on its inliers. Only lines in
    `use` (long lines) vote; residuals are returned for all."""
    ys = np.asarray(ys, float); xs = np.asarray(xs, float)
    n = len(xs)
    use = np.ones(n, bool) if use is None else np.asarray(use, bool)
    idx = np.nonzero(use)[0]
    if len(idx) < 3:
        return None
    sgn = 1.0 if side == "right" else -1.0
    v = sgn * xs
    best = None
    max_b = math.tan(math.radians(P["EDGE_MAX_DEG"]))
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            if abs(ys[j] - ys[i]) < 1e-6:
                continue
            b_ = (v[j] - v[i]) / (ys[j] - ys[i])
            if abs(b_) > max_b:
                continue
            a_ = v[i] - b_ * ys[i]
            res = v[idx] - (a_ + b_ * ys[idx])
            inl = np.abs(res) <= tol
            key = (int(inl.sum()), float(np.median(a_ + b_ * ys[idx])))
            if best is None or key > best[0]:
                best = (key, a_, b_)
    if best is None:
        return None
    _, a_, b_ = best
    for _ in range(3):
        res = v[idx] - (a_ + b_ * ys[idx])
        inl = idx[np.abs(res) <= tol]
        if len(inl) < 2:
            break
        A = np.stack([np.ones(len(inl)), ys[inl]], 1)
        a_, b_ = np.linalg.lstsq(A, v[inl], rcond=None)[0]
    res = v - (a_ + b_ * ys)
    ok = np.abs(res[idx]) <= tol
    return dict(a=float(sgn * a_), b=float(sgn * b_), inliers=int(ok.sum()), n=int(len(idx)),
                residual_sd_px=round(float(np.std(res[idx][ok])) if ok.any() else 0.0, 2), res=res)


def fit_block_edges(lines, S, cc=None) -> dict:
    text = [L for L in lines if L.kind == "text"]
    tol = P["EDGE_TOL"] * S
    out = dict(tolerance_px=round(tol, 2), right=None, left=None)
    if len(text) < 3:
        return out
    yr = [float(L.y_at(L.x1)) for L in text]; xr = [L.x1 for L in text]
    yl = [float(L.y_at(L.x0)) for L in text]; xl = [L.x0 for L in text]
    ext = np.array([L.x1 - L.x0 for L in text])
    use = ext >= P["EDGE_LONG"] * float(np.median(ext))
    R = _robust_edge(yr, xr, "right", tol, use)
    Lf = _robust_edge(yl, xl, "left", tol, use)
    for key, fit, ys, xs in (("right", R, yr, xr), ("left", Lf, yl, xl)):
        if fit is None:
            continue
        res = fit.pop("res")
        out[key] = dict(fit, angle_deg=round(math.degrees(math.atan(fit["b"])), 3))
        for L, y, x, rr in zip(text, ys, xs, res):
            edge = fit["a"] + fit["b"] * y
            st = "flush" if abs(rr) <= tol else ("protrudes" if rr > tol else "inside")
            setattr(L, f"edge_{key}", float(edge))
            setattr(L, f"edge_{key}_status", st)
            setattr(L, f"edge_{key}_offset_px", float(rr))
    # fitted start/end: a flush or indented line keeps its own ink end; a line
    # whose ink protrudes beyond the edge is re-measured from its components
    # that lie within the edge (+ tolerance): the protruding ink (speck,
    # margin mark) is cut off. With no components inside, the edge itself.
    for L in text:
        for key in ("right", "left"):
            st = getattr(L, f"edge_{key}_status", None)
            edge = getattr(L, f"edge_{key}", None)
            own = L.x1 if key == "right" else L.x0
            val = own
            if st == "protrudes" and cc is not None and L.comps:
                ids = np.array(L.comps)
                if key == "right":
                    ok = cc.x1[ids] <= edge + tol
                    val = float(cc.x1[ids][ok].max()) if ok.any() else float(edge)
                else:
                    ok = cc.x0[ids] >= edge - tol
                    val = float(cc.x0[ids][ok].min()) if ok.any() else float(edge)
            setattr(L, f"fit_{key}", float(val))
    return out


def margin_lines(lines, edges, S) -> int:
    """A text candidate with >= MARGIN_OUT of its extent beyond a fitted block
    edge (+ EDGE_TOL) is a margin mark, not a text line (punch holes, page-edge
    shadow, ticks after line ends, scanner logos). Returns how many changed."""
    n = 0
    tol = P["EDGE_TOL"] * S
    for L in lines:
        if L.kind != "text" or L.x1 <= L.x0:
            continue
        out = 0.0
        for key, side in (("right", 1), ("left", -1)):
            e = edges.get(key)
            if not e:
                continue
            y = float(L.y_at((L.x0 + L.x1) / 2))
            edge = e["a"] + e["b"] * y + side * tol
            if side > 0:
                out += max(0.0, L.x1 - max(L.x0, edge))
            else:
                out += max(0.0, min(L.x1, edge) - L.x0)
        if out >= P["MARGIN_OUT"] * (L.x1 - L.x0):
            L.kind = "margin"
            L.flags.append("outside the fitted text block")
            n += 1
    return n


def slope_field(lines, H, W) -> dict:
    text = [L for L in lines if L.kind == "text" and L.x1 - L.x0 > 0.2 * W]
    if not text:
        return dict(n=0)
    ang = np.array([L.angle_deg() for L in text])
    ys = np.array([float(L.y_at((L.x0 + L.x1) / 2)) for L in text])
    k3 = max(1, len(ang) // 3)
    o = np.argsort(ys)
    a = ang[o]
    # within-line bend: max deviation of the centre curve from its chord, px
    bends = []
    for L in text:
        xs = L.sample_x(16); yc = L.y_at(xs)
        chord = yc[0] + (yc[-1] - yc[0]) * (xs - xs[0]) / max(1e-9, xs[-1] - xs[0])
        bends.append(float(np.max(np.abs(yc - chord))))
    return dict(n=int(len(ang)), deg_top_mid_bottom=[round(float(np.median(a[:k3])), 2),
                                                    round(float(np.median(a[len(a) // 2 - k3 // 2:len(a) // 2 + k3 // 2 + 1])), 2),
                                                    round(float(np.median(a[-k3:])), 2)],
                deg_range=[round(float(a.min()), 2), round(float(a.max()), 2)],
                bend_px_median=round(float(np.median(bends)), 2), bend_px_max=round(float(np.max(bends)), 2))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def line_record(L: Line, W: int, H: int) -> dict:
    poly = L.polygon()
    rec = dict(kind=L.kind, flags=list(L.flags),
               x0=round(L.x0, 1), x1=round(L.x1, 1), up=round(L.up, 1), dn=round(L.dn, 1),
               angle_deg=round(L.angle_deg(), 3), mass_px=int(L.mass), components=L.ncomp,
               track_strips=L.track_strips,
               centre_curve=[[round(x, 1), round(y, 1)] for x, y in zip(L.xs, L.ys)],
               baseline_rtl=[[round(x, 1), round(y, 1)] for x, y in L.baseline(16)],
               polygon=[[round(x, 1), round(y, 1)] for x, y in poly],
               hull=[[round(x, 1), round(y, 1)] for x, y in convex_hull(poly)],
               rectification=dict(model="vertical shear along centre_curve", u="x - x0",
                                  v="y - centre(x) + up", strip_size=list(L.strip_shape())))
    for key in ("right", "left"):
        if hasattr(L, f"edge_{key}_status"):
            rec[f"edge_{key}"] = dict(status=getattr(L, f"edge_{key}_status"),
                                      edge_px=round(getattr(L, f"edge_{key}"), 1),
                                      offset_px=round(getattr(L, f"edge_{key}_offset_px"), 1),
                                      fitted_px=round(getattr(L, f"fit_{key}"), 1))
    return rec


def to_json(res: dict) -> dict:
    W, H = res["measurements"]["width"], res["measurements"]["height"]
    return dict({k: v for k, v in res.items() if k in ("version", "params", "use_mask", "measurements", "masked",
                                                       "edges", "debug", "seconds")},
                lines=[line_record(L, W, H) for L in res["lines"]])


def regions(res: dict, *, text_only: bool = True) -> list[dict]:
    """Scorer regions (tests/real_truth_eval.py): one convex polygon per line,
    normalised to the page."""
    W, H = res["measurements"]["width"], res["measurements"]["height"]
    out = []
    for L in res["lines"]:
        if text_only and L.kind != "text":
            continue
        hull = convex_hull(L.polygon())
        if len(hull) >= 3:
            out.append({"quad": [[x / W, y / H] for x, y in hull]})
    return out
