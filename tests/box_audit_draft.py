#!/usr/bin/env python3
"""Text-constrained drafting of word rectangles for the box-audit packet.

Evaluation tooling only (production never imports ``tests/``). numpy + PyMuPDF,
no API, no new dependencies.

The draft is a *proposal for a human auditor*, not a locator. It knows the
printed text in advance and only has to find where it is:

1. ``binarize``: local-mean threshold (uneven phone lighting), speck removal.
2. Lines: the text block is cut into vertical strips; adjacent strips' row
   profiles are cross-correlated in sliding windows to estimate how far a line
   moves up or down from one strip to the next (slope that changes down the
   page). Summing the strips along those tracks gives a straightened profile
   whose peaks are line candidates.
3. When the number of printed lines is known (hadpi text truth), candidates are
   aligned to the text lines by dynamic programming on line width versus
   character count (fixed-pitch typewriter), skipping candidates that are not
   text (stamp, signature). When it is not known (book scans), every candidate
   in the text block is a line and the transcription's words are distributed
   over the lines by a second DP (ink width versus character count, paragraph
   ends forced to line ends).
4. Words: inside each straightened line, the K-1 boundaries for K known words
   are chosen among the white column gaps by a DP that rewards wide gaps and
   penalises word widths far from their character-proportional share.

Every guess that the evidence does not settle is flagged ``uncertain`` with a
reason, so the audit tool can show it first. Coordinates returned here are
pixels of the image given; ``box_audit`` converts them to line frames.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

import fitz
import numpy as np

DRAFT_VERSION = "box_audit_draft/1"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_gray(path) -> np.ndarray:
    pm = fitz.Pixmap(str(path))
    if pm.alpha:
        pm = fitz.Pixmap(pm, 0)
    if pm.n != 1:
        pm = fitz.Pixmap(fitz.csGRAY, pm)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width).astype(np.float32)


def box_sum(a: np.ndarray, ry: int, rx: int) -> np.ndarray:
    """Sum over a (2ry+1) x (2rx+1) window, edges clipped."""
    a = a.astype(np.float64)
    H, W = a.shape
    c = np.zeros((H + 1, W + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(H) - ry, 0, H)
    y1 = np.clip(np.arange(H) + ry + 1, 0, H)
    x0 = np.clip(np.arange(W) - rx, 0, W)
    x1 = np.clip(np.arange(W) + rx + 1, 0, W)
    return c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]


def binarize(gray: np.ndarray, k: float = 0.18, r: int | None = None) -> np.ndarray:
    H, W = gray.shape
    r = r or max(15, int(round(min(H, W) / 60)))
    ones = np.ones_like(gray)
    mean = box_sum(gray, r, r) / box_sum(ones, r, r)
    ink = gray < mean * (1.0 - k)
    # Also require darker than a global paper estimate, so flat paper noise
    # inside large blank areas does not fire.
    paper = float(np.percentile(gray, 90))
    ink &= gray < paper * 0.82
    # Remove specks: keep a pixel if at least 3 of its 3x3 neighbourhood is ink.
    nb = box_sum(ink, 1, 1)
    return ink & (nb >= 3)


def smooth1d(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return v.astype(np.float64)
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    return np.convolve(np.pad(v.astype(np.float64), r, mode="edge"), kern, mode="valid")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def tokens_of(text: str) -> list[str]:
    return text.split()


def char_weight(token: str) -> float:
    """Visible width units of a token: base letters count 1, combining marks 0."""
    n = 0.0
    for ch in token:
        cat = unicodedata.category(ch)
        if cat.startswith("M") or ch in "‌‍‎‏":
            continue
        n += 1.0
    return max(n, 1.0)


# ---------------------------------------------------------------------------
# Line tracks
# ---------------------------------------------------------------------------
@dataclass
class Track:
    xs: np.ndarray          # strip centres (px)
    ys: np.ndarray          # line centre y at each strip centre (px)
    strength: float = 0.0
    x0: float = 0.0         # ink extent along the track
    x1: float = 0.0
    mass: float = 0.0
    thick: float = 0.0
    rule: bool = False
    reasons: list = field(default_factory=list)

    def y_at(self, x):
        return np.interp(x, self.xs, self.ys)


def text_block_x(ink: np.ndarray, y0: int = 0, y1: int | None = None) -> tuple[int, int]:
    col = ink[y0:y1].sum(0).astype(np.float64)
    col = smooth1d(col, 15)
    thr = 0.12 * float(np.percentile(col, 95))
    act = np.nonzero(col > thr)[0]
    if not len(act):
        return 0, ink.shape[1]
    return int(act[0]), int(act[-1]) + 1


def estimate_spacing(profile: np.ndarray, lo: int = 15, hi: int = 300) -> float:
    """Typical distance between neighbouring text lines (px).

    Autocorrelation alone locks onto a multiple of the spacing when lines come
    in pairs (numbered articles), so the smallest lag whose autocorrelation is
    a clear local maximum wins, and is cross-checked against the median
    distance between profile peaks."""
    p = profile - profile.mean()
    hi = min(hi, len(p) // 3)
    ac = np.array([float(np.dot(p[:-d], p[d:])) for d in range(lo, hi)])
    top = float(ac.max())
    lag = int(np.argmax(ac)) + lo
    for i in range(1, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.25 * top:
            lag = i + lo
            break
    sm = smooth1d(profile, max(1.0, lag / 10))
    pk = [y for y in range(1, len(sm) - 1) if sm[y] > sm[y - 1] and sm[y] >= sm[y + 1] and sm[y] > 0.2 * sm.max()]
    d = np.diff(pk)
    d = d[(d > 0.4 * lag) & (d < 1.6 * lag)]
    if len(d) >= 5:
        med = float(np.median(d))
        if abs(med - lag) > 0.2 * lag:
            return med
    return float(lag)


def strip_profiles(ink, xa, xb, nstrips, sigma):
    edges = np.linspace(xa, xb, nstrips + 1).astype(int)
    P = np.stack([smooth1d(ink[:, edges[i]:edges[i + 1]].sum(1), sigma) for i in range(nstrips)])
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, P, edges


def shift_field(P: np.ndarray, spacing: float, maxd: int) -> np.ndarray:
    """D[k, y]: vertical shift from strip k to k+1 for a line at y in strip k."""
    K, H = P.shape
    half = int(1.5 * spacing)
    D = np.zeros((K - 1, H))
    for k in range(K - 1):
        a, b = P[k], P[k + 1]
        ds = np.arange(-maxd, maxd + 1)
        corr = np.zeros((len(ds), H))
        ea = np.cumsum(np.concatenate([[0.0], a * a]))
        eb = np.cumsum(np.concatenate([[0.0], b * b]))
        idx = np.arange(H)
        lo = np.clip(idx - half, 0, H)
        hi = np.clip(idx + half + 1, 0, H)
        for i, d in enumerate(ds):
            bs = np.zeros(H)
            if d >= 0:
                bs[:H - d] = b[d:]
            else:
                bs[-d:] = b[:H + d]
            c = np.cumsum(np.concatenate([[0.0], a * bs]))
            corr[i] = c[hi] - c[lo]
        na = ea[hi] - ea[lo]
        nb = eb[hi] - eb[lo]
        norm = np.sqrt(na * nb) + 1e-9
        ncorr = corr / norm
        j = np.argmax(ncorr, axis=0)
        best = ncorr[j, idx]
        # subpixel
        jm = np.clip(j - 1, 0, len(ds) - 1)
        jp = np.clip(j + 1, 0, len(ds) - 1)
        cm, cp = ncorr[jm, idx], ncorr[jp, idx]
        den = cm - 2 * best + cp
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(np.abs(den) > 1e-9, 0.5 * (cm - cp) / den, 0.0)
        off = np.nan_to_num(off)
        off = np.clip(off, -0.5, 0.5)
        dk = ds[j] + off
        energy = np.minimum(na, nb)
        valid = (best > 0.5) & (energy > 0.05 * np.percentile(energy, 95))
        if valid.sum() >= 2:
            dk = np.interp(idx, idx[valid], dk[valid])
        else:
            dk = np.zeros(H)
        D[k] = smooth1d(dk, spacing)
    return D


def straighten_map(D: np.ndarray, ref: int, H: int) -> np.ndarray:
    """Y[k, y]: y in strip k of the line passing through y in strip ref."""
    K = D.shape[0] + 1
    Y = np.zeros((K, H))
    base = np.arange(H, dtype=np.float64)
    Y[ref] = base
    cur = base.copy()
    for k in range(ref, K - 1):
        cur = cur + np.interp(cur, base, D[k])
        Y[k + 1] = cur
    cur = base.copy()
    for k in range(ref - 1, -1, -1):
        # invert y_{k+1} = y_k + D[k](y_k) by fixed point
        y = cur.copy()
        for _ in range(4):
            y = cur - np.interp(y, base, D[k])
        Y[k] = y
        cur = y
    return Y


def find_tracks(ink: np.ndarray, region: tuple[int, int, int, int] | None = None, nstrips: int = 12):
    """Line candidates as tracks. region = (x0, y0, x1, y1) limits the search."""
    H, W = ink.shape
    rx0, ry0, rx1, ry1 = region or (0, 0, W, H)
    sub = np.zeros_like(ink)
    sub[ry0:ry1, rx0:rx1] = ink[ry0:ry1, rx0:rx1]
    xa, xb = text_block_x(sub, ry0, ry1)
    centre = sub[:, xa + (xb - xa) // 3: xb - (xb - xa) // 3].sum(1).astype(np.float64)
    spacing = estimate_spacing(smooth1d(centre, 2))
    sigma = max(1.5, spacing / 8)
    centres, P, edges = strip_profiles(sub, xa, xb, nstrips, sigma)
    strip_w = (xb - xa) / nstrips
    maxd = int(min(spacing * 0.42, strip_w * math.tan(math.radians(5)) + 3))
    D = shift_field(P, spacing, max(2, maxd))
    ref = nstrips // 2
    Y = straighten_map(D, ref, H)
    base = np.arange(H, dtype=np.float64)
    S = np.zeros(H)
    for k in range(nstrips):
        S += np.interp(Y[k], base, P[k], left=0, right=0)
    S = smooth1d(S, max(1.0, spacing / 10))
    peaks = [y for y in range(1, H - 1) if S[y] > S[y - 1] and S[y] >= S[y + 1]]
    thr = 0.015 * float(np.percentile(S[ry0:ry1], 99))
    peaks = [y for y in peaks if S[y] > thr]
    # Suppress peaks that are too close (keep the stronger), then valleys must
    # dip clearly between neighbours.
    peaks.sort(key=lambda y: -S[y])
    kept: list[int] = []
    for y in peaks:
        if all(abs(y - q) > 0.45 * spacing for q in kept):
            kept.append(y)
    kept.sort()
    tracks = []
    for y in kept:
        ys = np.array([float(np.interp(y, base, Y[k])) for k in range(nstrips)])
        tracks.append(Track(xs=centres.copy(), ys=ys, strength=float(S[y])))
    refine_tracks(tracks, P, spacing)
    info = dict(spacing=spacing, xa=xa, xb=xb, nstrips=nstrips, S=S, Y=Y, P=P, edges=edges)
    return tracks, info


def refine_tracks(tracks: list[Track], P: np.ndarray, spacing: float) -> None:
    """Re-measure each track strip by strip and replace it by a straight
    line (weighted least squares). Tracks seen in fewer than three strips take
    the slope of their nearest well-measured neighbours."""
    K, H = P.shape
    win = 0.4 * spacing
    strip_ref = np.array([np.percentile(P[k][P[k] > 0], 90) if (P[k] > 0).any() else 1.0 for k in range(K)])
    fits = []
    for t in tracks:
        xs_ok, ys_ok, wts = [], [], []
        for k in range(K):
            y0 = int(max(0, t.ys[k] - win)); y1 = int(min(H - 1, t.ys[k] + win))
            seg = P[k, y0:y1 + 1]
            if len(seg) < 3:
                continue
            j = int(np.argmax(seg))
            if seg[j] < 0.25 * strip_ref[k] or j in (0, len(seg) - 1):
                continue
            cm, c0, cp = seg[j - 1], seg[j], seg[j + 1]
            den = cm - 2 * c0 + cp
            off = 0.5 * (cm - cp) / den if abs(den) > 1e-9 else 0.0
            xs_ok.append(t.xs[k]); ys_ok.append(y0 + j + max(-0.5, min(0.5, off))); wts.append(seg[j])
        fits.append((np.array(xs_ok), np.array(ys_ok), np.array(wts)))
    slopes = [None] * len(tracks)
    for i, (xs, ys, w) in enumerate(fits):
        if len(xs) >= 3 and xs.max() - xs.min() > 0.2 * (tracks[i].xs[-1] - tracks[i].xs[0]):
            keep = np.ones(len(xs), bool)
            for _ in range(3):
                b, a = np.polyfit(xs[keep], ys[keep], 1, w=np.sqrt(w[keep]))
                res = np.abs(ys - (a + b * xs))
                new = res < max(3.0, 0.15 * spacing)
                if new.sum() < 3 or (new == keep).all():
                    break
                keep = new
            if abs(b) < math.tan(math.radians(6)):
                slopes[i] = b
    ys_mid = [float(np.median(t.ys)) for t in tracks]
    good = [i for i, b in enumerate(slopes) if b is not None]
    for i, t in enumerate(tracks):
        xs, ys, w = fits[i]
        b = slopes[i]
        if b is None:
            if good:
                near = sorted(good, key=lambda g: abs(ys_mid[g] - ys_mid[i]))[:2]
                b = float(np.mean([slopes[g] for g in near]))
            else:
                b = 0.0
            t.reasons.append("slope taken from neighbouring lines")
        if len(xs):
            a = float(np.average(ys - b * xs, weights=w))
        else:
            a = float(np.mean(t.ys - b * t.xs))
            t.reasons.append("line position not re-measured")
        t.ys = a + b * t.xs


def band_columns(ink, track: Track, up: float, down: float, x0: int, x1: int):
    """Straightened band: rows -up..down around the track, columns x0..x1."""
    H, W = ink.shape
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    xs = np.arange(x0, x1)
    yc = np.rint(track.y_at(xs)).astype(int)
    offs = np.arange(-int(up), int(down) + 1)
    rows = yc[None, :] + offs[:, None]
    valid = (rows >= 0) & (rows < H)
    rows = np.clip(rows, 0, H - 1)
    band = ink[rows, xs[None, :]] & valid
    return band, offs, xs


def line_extent(ink, track: Track, spacing: float, xa: int, xb: int):
    """Ink extent (x0, x1), ink mass and ink thickness of a candidate line.

    Stray marks at either end (margin specks, a stamp's edge) are dropped when
    they are small compared with the line's typical glyph run and separated
    from it by more than half a line spacing."""
    band, offs, xs = band_columns(ink, track, 0.32 * spacing, 0.32 * spacing, xa - spacing, xb + spacing)
    col = band.sum(0)
    on = col >= 2
    rr = [r for r in runs(on) if r[1] - r[0] >= 2]
    if not rr:
        return None
    gap_lim = 2.5 * spacing
    groups = [[rr[0]]]
    for r in rr[1:]:
        if r[0] - groups[-1][-1][1] > gap_lim:
            groups.append([r])
        else:
            groups[-1].append(r)
    mass = [sum(int(col[a:b].sum()) for a, b in g) for g in groups]
    g = list(groups[int(np.argmax(mass))])
    rmass = [int(col[a:b].sum()) for a, b in g]
    med = float(np.median(rmass))
    # chunks: runs closer than half a spacing belong together
    chunks = [[g[0]]]
    for r in g[1:]:
        if r[0] - chunks[-1][-1][1] > 0.5 * spacing:
            chunks.append([r])
        else:
            chunks[-1].append(r)
    def cmass(c):
        return sum(int(col[a:b].sum()) for a, b in c)
    while len(chunks) > 1:
        if cmass(chunks[0]) < 0.6 * med and cmass(chunks[0]) <= cmass(chunks[-1]):
            chunks.pop(0)
        elif cmass(chunks[-1]) < 0.6 * med:
            chunks.pop()
        else:
            break
    g = [r for c in chunks for r in c]
    x0 = float(xs[g[0][0]]); x1 = float(xs[g[-1][1] - 1] + 1)
    run_list = [(float(xs[a]), float(xs[b - 1] + 1), float(col[a:b].sum())) for a, b in g]
    # thickness: rows of the straightened band carrying ink in the extent
    c0, c1 = g[0][0], g[-1][1]
    tall, toffs, _ = band_columns(ink, track, 0.45 * spacing, 0.45 * spacing, x0, x1)
    rp = tall.sum(1).astype(np.float64)
    thick = float((rp > 0.12 * rp.max()).sum()) if rp.max() > 0 else 0.0
    return x0, x1, float(sum(int(col[a:b].sum()) for a, b in g)), thick, run_list


def measure_tracks(ink, tracks: list[Track], info: dict) -> None:
    sp = info["spacing"]
    for t in tracks:
        e = line_extent(ink, t, sp, info["xa"], info["xb"])
        if e is None:
            t.x0, t.x1, t.mass, t.thick, t.runs = info["xa"], info["xa"] + 1.0, 0.0, 0.0, []
        else:
            t.x0, t.x1, t.mass, t.thick, t.runs = e
        # a long thin stroke is an underline/rule, not a text line
        t.rule = t.thick < 0.16 * sp and (t.x1 - t.x0) > 4 * sp


def trim_to_text(t: Track, expected: float, spacing: float) -> str | None:
    """Drop ink chunks at either end of a line (margin marks, specks, stamp
    edges) when the line is clearly wider than its known text. A chunk is a
    run group separated from the rest by more than 0.6 line spacings."""
    runs_ = getattr(t, "runs", None) or []
    if len(runs_) < 2 or (t.x1 - t.x0) < 1.15 * expected:
        return None
    chunks = [[runs_[0]]]
    for r in runs_[1:]:
        if r[0] - chunks[-1][-1][1] > 0.6 * spacing:
            chunks.append([r])
        else:
            chunks[-1].append(r)
    if len(chunks) < 2:
        return None
    total = sum(r[2] for r in runs_)
    best = (abs(math.log((t.x1 - t.x0) / expected)), 0, len(chunks))
    for a in range(len(chunks)):
        for b in range(a + 1, len(chunks) + 1):
            keep = chunks[a:b]
            m = sum(r[2] for c in keep for r in c)
            if m < 0.7 * total:
                continue
            w = keep[-1][-1][1] - keep[0][0][0]
            sc = abs(math.log(max(1.0, w) / expected))
            if sc < best[0] - 0.1:
                best = (sc, a, b)
    _, a, b = best
    if (a, b) == (0, len(chunks)):
        return None
    keep = chunks[a:b]
    t.x0, t.x1 = keep[0][0][0], keep[-1][-1][1]
    t.runs = [r for c in keep for r in c]
    return "stray ink at the line end was left out"


# ---------------------------------------------------------------------------
# Matching tracks to known text lines
# ---------------------------------------------------------------------------
def _match_cost(t: Track, chars: float, pitch: float, sigma: float) -> float:
    r = math.log(max(t.x1 - t.x0, 1.0) / (pitch * chars))
    c = min(12.0, (r / sigma) ** 2)
    if getattr(t, "rule", False):
        c += 10.0
    return c


def _skip_cost(t: Track, med: float) -> float:
    if getattr(t, "rule", False):
        return 0.05
    return 1.2 * min(3.0, t.strength / med)


def align_known_lines(tracks: list[Track], chars: list[float], gap_after: list[bool] | None = None,
                      spacing: float | None = None, sigma: float = 0.22):
    """Monotone alignment of N text lines to M candidates.

    Returns (assignment, pitch, costs): assignment[i] is a candidate index or
    None (text line with no candidate). Pass 1 fits the character pitch with a
    plain DP; pass 2 adds a spacing term (consecutive printed lines are about
    one spacing apart unless the text marks a vertical gap)."""
    M, N = len(tracks), len(chars)
    if M == 0:
        return [None] * N, 0.0, [None] * N
    med = float(np.median([t.strength for t in tracks]))
    INF = 1e18
    best = None
    for pitch in np.exp(np.linspace(math.log(4), math.log(60), 90)):
        dp = np.full((N + 1, M + 1), INF)
        dp[0, 0] = 0.0
        for j in range(M + 1):
            for i in range(N + 1):
                cur = dp[i, j]
                if cur >= INF:
                    continue
                if j < M:
                    dp[i, j + 1] = min(dp[i, j + 1], cur + _skip_cost(tracks[j], med))
                if i < N:
                    dp[i + 1, j] = min(dp[i + 1, j], cur + 8.0)
                if i < N and j < M:
                    dp[i + 1, j + 1] = min(dp[i + 1, j + 1], cur + _match_cost(tracks[j], chars[i], pitch, sigma))
        if best is None or dp[N, M] < best[0]:
            best = (dp[N, M], pitch)
    pitch = best[1]
    gap_after = gap_after or [False] * N
    yc = [float(t.y_at((t.x0 + t.x1) / 2)) for t in tracks]
    sp = spacing or float(np.median(np.diff(sorted(yc)))) if M > 1 else 1.0
    skip_pre = np.concatenate([[0.0], np.cumsum([_skip_cost(t, med) for t in tracks])])
    MISS = 8.0
    # state: (i lines done, last matched candidate index + 1; 0 = none yet)
    dp = np.full((N + 1, M + 1), INF)
    bk: dict = {}
    dp[0, 0] = 0.0
    for i in range(N):
        for last in range(M + 1):
            cur = dp[i, last]
            if cur >= INF:
                continue
            # text line i missing
            if cur + MISS < dp[i + 1, last]:
                dp[i + 1, last] = cur + MISS
                bk[(i + 1, last)] = (last, None)
            for j in range(last, M):
                c = cur + (skip_pre[j] - skip_pre[last]) + _match_cost(tracks[j], chars[i], pitch, sigma)
                if last > 0:
                    g = (yc[j] - yc[last - 1]) / sp
                    allowed = 2.6 if gap_after[i - 1] else 1.45
                    if g > allowed:
                        c += min(4.0, 3.0 * (g - allowed))
                    if g < 0.55:
                        c += 4.0
                if c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = c
                    bk[(i + 1, j + 1)] = (last, j)
    # finish: skip remaining candidates
    fin = [dp[N, last] + (skip_pre[M] - skip_pre[last]) for last in range(M + 1)]
    last = int(np.argmin(fin))
    assign: list[int | None] = [None] * N
    i = N
    while i > 0:
        prev, j = bk[(i, last)]
        assign[i - 1] = j
        last = prev
        i -= 1
    costs = [(_match_cost(tracks[a], chars[i], pitch, sigma) if a is not None else None) for i, a in enumerate(assign)]
    return assign, float(pitch), costs


# ---------------------------------------------------------------------------
# Word segmentation inside a line
# ---------------------------------------------------------------------------
BREAK_MARGIN = 1.5      # line_fit_cost units
SOFT_REASONS = {"boundary ambiguous"}
AMBIGUITY_MARGIN = 1.0   # DP cost units; set on synthetic typewriter pages (box_audit synth-eval)


def choose_boundaries(L: int, gaps: list[tuple[int, int]], weights: list[float], overlay=None,
                      soft=None, gap_ref: float | None = None):
    """Choose len(weights)-1 word boundaries among white gaps.

    ``gaps`` are (start, end) column offsets measured from the right end of
    the line (reading order). Returns (bounds, margins, virtual): bounds are
    (start, end) per word; margins[i] is how much worse the best segmentation
    is that puts boundary i in a different gap (small = ambiguous); virtual[i]
    is True when boundary i had to be placed without a white gap.

    Proportional "virtual" boundaries are always offered: cheap inside
    ``overlay`` (columns crossed by stamp/signature ink, where real gaps are
    hidden), expensive elsewhere. This stops one hidden gap from shifting
    every later word by one. ``soft[i]`` marks boundary i as optional
    (inside one word at a ZWNJ): a white gap may or may not be printed there,
    so its proportional boundary costs nothing."""
    k = len(weights)
    soft = soft or [False] * max(0, k - 1)
    if k == 1:
        return [(0, L)], [], [], 1.0, 0.0
    ws = sorted((e - s for s, e in gaps), reverse=True)
    pitch_ink = max(1.0, (L - sum(ws[:k - 1])) / sum(weights))
    cand = sorted(set(gaps))
    virtual = {}
    tot = sum(weights)
    acc = 0.0
    for bi, w in enumerate(weights[:-1]):
        acc += w
        q = int(L * acc / tot)
        if (q, q) not in cand:
            cand.append((q, q))
            hidden = overlay is not None and bool(overlay[min(L - 1, max(0, q))])
            if soft[bi]:
                virtual[(q, q)] = 0.0
            else:
                virtual[(q, q)] = -0.5 if hidden else (-4.0 if len(gaps) < k - 1 else -6.0)
    cand.sort()
    Mc = len(cand)
    INF = 1e18

    def wcost(i, s, e):
        r = math.log(max(1.0, e - s) / (pitch_ink * weights[i]))
        sig = 0.45 if weights[i] > 2 else 0.8
        return (r / sig) ** 2

    def gscore(j):
        if cand[j] in virtual:
            return virtual[cand[j]]
        s, e = cand[j]
        return 2.5 * math.log1p((e - s) / max(1.0, 0.25 * pitch_ink)) - offset

    # With gap_ref (a page-wide word-gap threshold), a boundary in a gap
    # narrower than that threshold costs instead of earning: totals for
    # different word counts become comparable.
    offset = 2.5 * math.log1p(gap_ref / max(1.0, 0.25 * pitch_ink)) if gap_ref else 0.0

    fw = np.full((k - 1, Mc), INF)
    bk = np.full((k - 1, Mc), -1, dtype=int)
    for j in range(Mc):
        fw[0, j] = wcost(0, 0, cand[j][0]) - gscore(j)
    for i in range(1, k - 1):
        for j in range(i, Mc):
            best, arg = INF, -1
            for jp in range(i - 1, j):
                if fw[i - 1, jp] >= INF or cand[jp][1] > cand[j][0]:
                    continue
                c = fw[i - 1, jp] + wcost(i, cand[jp][1], cand[j][0])
                if c < best:
                    best, arg = c, jp
            if arg >= 0:
                fw[i, j] = best - gscore(j)
                bk[i, j] = arg
    bw = np.full((k - 1, Mc), INF)
    for j in range(Mc):
        bw[k - 2, j] = wcost(k - 1, cand[j][1], L)
    for i in range(k - 3, -1, -1):
        for j in range(Mc):
            best = INF
            for jn in range(j + 1, Mc):
                if bw[i + 1, jn] >= INF or cand[j][1] > cand[jn][0]:
                    continue
                c = wcost(i + 1, cand[j][1], cand[jn][0]) - gscore(jn) + bw[i + 1, jn]
                if c < best:
                    best = c
            bw[i, j] = best
    tot = fw + bw
    last = int(np.argmin(tot[k - 2]))
    best_total = float(tot[k - 2, last])
    chosen = [last]
    for i in range(k - 2, 0, -1):
        chosen.append(int(bk[i, chosen[-1]]))
    chosen.reverse()
    margins = []
    centre = np.array([(a + b) / 2 for a, b in cand])
    for i, j in enumerate(chosen):
        row = tot[i].copy()
        # alternatives that move the boundary by less than half a character
        # are the same boundary
        row[np.abs(centre - centre[j]) < 0.6 * pitch_ink] = INF
        margins.append(float(row.min() - best_total) if row.min() < INF else 99.0)
    bounds, prev = [], 0
    for j in chosen:
        bounds.append((prev, cand[j][0]))
        prev = cand[j][1]
    bounds.append((prev, L))
    return bounds, margins, [cand[j] in virtual for j in chosen], pitch_ink, best_total


def token_parts(tokens: list[str]):
    """Tokens split at ZWNJ. Old typesetting often prints a ZWNJ as a visible
    gap; the boundary between parts of one token is optional (soft)."""
    parts, owner, soft = [], [], []
    for ti, t in enumerate(tokens):
        ps = [x for x in t.split("\u200c") if x] or [t]
        for pi, x in enumerate(ps):
            if parts:
                soft.append(pi > 0)
            parts.append(x)
            owner.append(ti)
    return parts, owner, soft


def line_gap_model(ink, track: Track, spacing: float, up: float, down: float):
    """(L, gaps from the right, hidden mask from the right) for a line."""
    core = 0.30 * spacing
    band, offs, xs = band_columns(ink, track, core, core, track.x0, track.x1)
    col = band.sum(0)
    on = col >= 1
    for a, b in runs(on):
        if b - a <= 2 and col[a:b].sum() <= 3 and a > 0 and b < len(col):
            on[a:b] = False
    L = len(col)
    gaps = [(L - b, L - a) for a, b in runs(~on) if a > 0 and b < L]
    full, foffs, fxs = band_columns(ink, track, up, down, track.x0, track.x1)
    edge = (foffs <= -up + 2) | (foffs >= down - 2)
    crossing = full[edge].any(0).astype(np.float64)
    density = smooth1d(crossing, max(3.0, spacing / 2) / 2)[::-1]
    return L, gaps, density > 0.12, float(on.sum())


def _vertical_run(rowp: np.ndarray, offs: np.ndarray, core: float):
    """Rows of the word: the run of inked rows (gaps of up to 3 rows allowed)
    carrying the most ink near the line centre."""
    on = rowp > 0
    if not on.any():
        return None, None
    rr = runs(on)
    merged = [list(rr[0])]
    for a, b in rr[1:]:
        if a - merged[-1][1] <= 3:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    def score(r):
        a, b = r
        w = np.exp(-0.5 * (offs[a:b] / max(1.0, core)) ** 2)
        return float((rowp[a:b] * w).sum())
    best = max(merged, key=score)
    return best[0], best[1] - 1


def segment_words(ink, track: Track, x0: float, x1: float, tokens: list[str], spacing: float,
                  up: float, down: float):
    """Return per-token [x0, x1, top_off, bot_off] (page px; offsets relative to
    the track) in reading order (right to left), per-word reasons and line
    reasons."""
    k = len(tokens)
    core = 0.30 * spacing
    band, offs, xs = band_columns(ink, track, core, core, x0, x1)
    col = band.sum(0)
    on = col >= 1
    # a speck (at most 2 columns, at most 3 ink pixels) does not close a gap
    for a, b in runs(on):
        if b - a <= 2 and col[a:b].sum() <= 3 and a > 0 and b < len(col):
            on[a:b] = False
    L = len(col)
    gaps = [(L - b, L - a) for a, b in runs(~on) if a > 0 and b < L]
    parts, owner, soft = token_parts(tokens)
    weights = [char_weight(t) for t in parts]
    line_reasons: list[str] = []
    word_reasons = [[] for _ in range(k)]
    if k > 1 and len(gaps) < k - 1:
        line_reasons.append(f"only {len(gaps)} white gaps for {k} words")
    # Ink crossing the half-way rows to the neighbouring lines: stamp,
    # signature, underline or handwriting, not this line's type.
    full, foffs, fxs = band_columns(ink, track, up, down, x0, x1)
    edge = (foffs <= -up + 2) | (foffs >= down - 2)
    crossing = full[edge].any(0).astype(np.float64)
    win = max(3.0, spacing / 2)
    density = smooth1d(crossing, win / 2)[::-1]                # indexed from the right
    hidden_rl = density > 0.12     # gaps may be hidden: proportional boundaries are cheap
    overlay_rl = density > 0.40    # flagged: printed glyphs alone stay below this
    pb, pm, pv, pitch_ink, _ = choose_boundaries(L, gaps, weights, hidden_rl, soft)
    # merge parts back into tokens; keep only the boundaries between tokens
    bounds, margins, virtual = [], [], []
    for pi, (a, b) in enumerate(pb):
        if pi == 0 or owner[pi] != owner[pi - 1]:
            bounds.append([a, b])
        else:
            bounds[-1][1] = b
        if pi > 0 and owner[pi] != owner[pi - 1]:
            margins.append(pm[pi - 1])
            virtual.append(pv[pi - 1])
    bounds = [tuple(x) for x in bounds]
    tok_weights = [sum(weights[pi] for pi in range(len(parts)) if owner[pi] == ti) for ti in range(k)]
    for i, m in enumerate(margins):
        if virtual[i]:
            why = "boundary guessed: no white gap"
        elif m < AMBIGUITY_MARGIN:
            why = "boundary ambiguous"
        else:
            continue
        for wi in (i, i + 1):
            if why not in word_reasons[wi]:
                word_reasons[wi].append(why)
    for wi, (s0, e0) in enumerate(bounds):
        seg = overlay_rl[s0:e0]
        if len(seg) and seg.mean() > 0.2:
            word_reasons[wi].append("other ink (stamp, signature, underline) crosses this word")
    # Pixel extents (right-to-left index -> page x)
    xr = xs[0] + L  # right edge (exclusive) in page px
    out = []
    for wi, (s, e) in enumerate(bounds):
        px1 = xr - s
        px0 = xr - e
        # tighten to ink columns
        c0 = int(px0 - xs[0]); c1 = int(px1 - xs[0])
        sub = col[c0:c1]
        nz = np.nonzero(sub)[0]
        if len(nz):
            px0 = xs[0] + c0 + nz[0]
            px1 = xs[0] + c0 + nz[-1] + 1
        fsub = full[:, c0:c1]
        rowp = fsub.sum(1).astype(np.float64)
        top, bot = _vertical_run(rowp, foffs, core)
        if top is None:
            top, bot = int(np.argmin(np.abs(foffs + core))), int(np.argmin(np.abs(foffs - core)))
            word_reasons[wi].append("no ink found")
        out.append([float(px0), float(px1), float(foffs[top]), float(foffs[bot] + 1)])
    hs = [b[3] - b[2] for b in out]
    if len(hs) >= 3:
        mh = float(np.median(hs))
        for wi, h in enumerate(hs):
            if h > 1.6 * mh:
                word_reasons[wi].append("box much taller than its neighbours")
    for wi, (s0, e0) in enumerate(bounds):
        r = math.log(max(1.0, e0 - s0) / (pitch_ink * tok_weights[wi]))
        if r > math.log(2.2) and tok_weights[wi] > 1:
            word_reasons[wi].append("box much wider than its text")
    strong = sum(1 for r in word_reasons if any(x not in SOFT_REASONS for x in r))
    soft = sum(1 for r in word_reasons if r)
    if strong:
        # one hidden or invented boundary can shift every later word by one
        line_reasons.append(f"{strong} word(s) under other ink or with guessed boundaries; later words may be shifted")
    elif k >= 3 and soft >= 0.5 * k:
        line_reasons.append(f"{soft} of {k} word boundaries ambiguous")
    return out, word_reasons, line_reasons


# ---------------------------------------------------------------------------
# Distributing an un-broken transcription over detected lines
# ---------------------------------------------------------------------------
def otsu_1d(values) -> float:
    v = np.sort(np.asarray(values, float))
    if len(v) < 4:
        return float(v.mean()) if len(v) else 0.0
    best, thr = -1.0, float(v.mean())
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            continue
        a, b = v[:i], v[i:]
        sc = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if sc > best:
            best, thr = sc, (v[i - 1] + v[i]) / 2
    return float(thr)


def distribute_tokens(line_inkw: list[float], paragraphs: list[list[str]], clear_gaps=None,
                      sigma_w: float = 0.12, sigma_n: float = 0.8, zwnj_gap: float | None = None,
                      short=None):
    """See ``_distribute``. When ``zwnj_gap`` is None, the share of ZWNJs that
    are printed as a visible gap (0, 0.5 or 1) is chosen by the best fit."""
    if clear_gaps is None or zwnj_gap is not None:
        return _distribute(line_inkw, paragraphs, clear_gaps, sigma_w, sigma_n, zwnj_gap or 0.0, short)
    best = None
    for g in (0.0, 0.5, 1.0):
        r = _distribute(line_inkw, paragraphs, clear_gaps, sigma_w, sigma_n, g, short)
        if r is not None and (best is None or r[3] < best[3]):
            best = r + (g,)
    return best


SHORT_LINE_PENALTY = 25.0


def _distribute(line_inkw, paragraphs, clear_gaps, sigma_w, sigma_n, zwnj_gap, short=None):
    """Assign consecutive tokens to lines. ``paragraphs`` is a list of token
    lists; a paragraph always starts on a new line. ``line_inkw`` is the ink
    width (non-gap columns) of each line; ``clear_gaps`` (optional) the number
    of unambiguous word gaps seen on each line. Returns (per-line token lists,
    per-line log width ratio, px per character) or None."""
    toks = [t for p in paragraphs for t in p]
    T, Lc = len(toks), len(line_inkw)
    starts = set()
    pos = 0
    for p in paragraphs:
        starts.add(pos)
        pos += len(p)
    w = np.array([char_weight(t) for t in toks])
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cz = np.concatenate([[0.0], np.cumsum([t.count("\u200c") for t in toks])])
    alpha = sum(line_inkw) / max(1.0, cw[-1])
    INF = 1e18
    dp = np.full((Lc + 1, T + 1), INF)
    bk = np.full((Lc + 1, T + 1), -1, dtype=int)
    dp[0, 0] = 0
    for l in range(Lc):
        for t0 in range(T + 1):
            if dp[l, t0] >= INF:
                continue
            for t1 in range(t0 + 1, T + 1):
                if t1 - 1 > t0 and (t1 - 1) in starts:
                    break
                exp = alpha * (cw[t1] - cw[t0])
                r = math.log(max(1.0, line_inkw[l]) / max(1.0, exp))
                c = (r / sigma_w) ** 2
                if clear_gaps is not None:
                    c += ((t1 - t0 - 1 + zwnj_gap * (cz[t1] - cz[t0]) - clear_gaps[l]) / sigma_n) ** 2
                if short is not None and short[l] and t1 < T and t1 not in starts:
                    c += SHORT_LINE_PENALTY   # a short line ends its paragraph
                v = dp[l, t0] + c
                if v < dp[l + 1, t1]:
                    dp[l + 1, t1] = v
                    bk[l + 1, t1] = t0
                if exp > line_inkw[l] * 2.5:
                    break
    if dp[Lc, T] >= INF:
        return None
    total = float(dp[Lc, T])
    out, costs = [], []
    t = T
    for l in range(Lc, 0, -1):
        t0 = bk[l, t]
        out.append(toks[t0:t])
        costs.append(math.log(max(1.0, line_inkw[l - 1]) / max(1.0, alpha * (cw[t] - cw[t0]))))
        t = t0
    out.reverse(); costs.reverse()
    return out, costs, alpha, total


# ---------------------------------------------------------------------------
# Page drafts
# ---------------------------------------------------------------------------
def flag_competing_candidates(tracks, assign, chars, pitch, spacing):
    """A matched line is uncertain when an unmatched neighbouring candidate
    would have fitted its text almost as well."""
    used = {a for a in assign if a is not None}
    out = [[] for _ in assign]
    for i, j in enumerate(assign):
        if j is None:
            continue
        cj = _match_cost(tracks[j], chars[i], pitch, 0.22)
        yj = float(np.median(tracks[j].ys))
        for jj in (j - 1, j + 1):
            if jj < 0 or jj >= len(tracks) or jj in used:
                continue
            if abs(float(np.median(tracks[jj].ys)) - yj) > 1.3 * spacing:
                continue
            if _match_cost(tracks[jj], chars[i], pitch, 0.22) <= cj + 2.0 and tracks[jj].strength > 0.2 * tracks[j].strength:
                out[i].append("a neighbouring unmatched ink line could be this line")
    return out


def _bands(tracks_in_order, spacing):
    """Up/down half-heights for each line: half the distance to its
    neighbours, bounded."""
    ys = [float(np.median(t.ys)) for t in tracks_in_order]
    out = []
    for i, y in enumerate(ys):
        up = (y - ys[i - 1]) / 2 if i > 0 else 0.6 * spacing
        dn = (ys[i + 1] - y) / 2 if i + 1 < len(ys) else 0.6 * spacing
        out.append((max(0.3 * spacing, min(0.75 * spacing, up)), max(0.3 * spacing, min(0.75 * spacing, dn))))
    return out


def _interp_track(prev: Track | None, nxt: Track | None, spacing: float, xs, xa, xb) -> Track:
    if prev is not None and nxt is not None:
        ys = (prev.ys + nxt.ys) / 2
    elif prev is not None:
        ys = prev.ys + spacing
    elif nxt is not None:
        ys = nxt.ys - spacing
    else:
        ys = np.zeros(len(xs))
    t = Track(xs=np.array(xs, float), ys=np.array(ys, float))
    t.x0, t.x1 = float(xa), float(xb)
    return t


def draft_known_lines(gray: np.ndarray, line_texts: list[str], gap_after: list[bool],
                      region=None) -> dict:
    """Draft for a page whose printed lines are known (one string per line)."""
    ink = binarize(gray)
    tracks, info = find_tracks(ink, region=region)
    measure_tracks(ink, tracks, info)
    sp = info["spacing"]
    chars = [float(len(t)) for t in line_texts]
    assign, pitch, costs = align_known_lines(tracks, chars, gap_after, sp)
    competing = flag_competing_candidates(tracks, assign, chars, pitch, sp)
    line_tracks: list[Track] = []
    line_reasons: list[list[str]] = []
    for i, a in enumerate(assign):
        reasons = list(competing[i])
        if a is None:
            prev = next((tracks[assign[k]] for k in range(i - 1, -1, -1) if assign[k] is not None), None)
            nxt = next((tracks[assign[k]] for k in range(i + 1, len(assign)) if assign[k] is not None), None)
            t = _interp_track(prev, nxt, sp, tracks[0].xs if tracks else [info["xa"], info["xb"]], info["xa"], info["xb"])
            # guess the extent from the character count, right-aligned to the block
            w = min(info["xb"] - info["xa"], pitch * chars[i])
            t.x1 = float(info["xb"]); t.x0 = float(info["xb"] - w)
            reasons.append("line not found in the image; position interpolated")
        else:
            t = tracks[a]
            note = trim_to_text(t, pitch * chars[i], sp)
            if note and note not in t.reasons:
                t.reasons.append(note)
            r = math.log(max(1.0, t.x1 - t.x0) / (pitch * chars[i]))
            if abs(r) > 0.3:
                reasons.append(f"line width {'wider' if r > 0 else 'narrower'} than its text by {abs(math.exp(r) - 1):.0%}")
        line_tracks.append(t)
        line_reasons.append(reasons)
    bands = _bands(line_tracks, sp)
    lines = []
    for i, text in enumerate(line_texts):
        t = line_tracks[i]
        up, dn = bands[i]
        toks = tokens_of(text)
        boxes, wreasons, lreasons = segment_words(ink, t, t.x0, t.x1, toks, sp, up, dn)
        lines.append(dict(text=text, tokens=toks, track=t, up=up, down=dn, words=boxes,
                          word_reasons=wreasons, reasons=line_reasons[i] + lreasons,
                          notes=list(t.reasons), candidate=assign[i]))
    return dict(lines=lines, info=dict(spacing=sp, pitch=pitch, xa=info["xa"], xb=info["xb"],
                                       candidates=len(tracks), assign=assign,
                                       unused=[j for j in range(len(tracks)) if j not in assign]),
                tracks=tracks)


# ---------------------------------------------------------------------------
# Book pages: transcription Markdown, not line-broken
# ---------------------------------------------------------------------------
import re as _re

_FOOTNOTE_REF = _re.compile(r"\[\^[^\]]*\]")


def markdown_paragraphs(md: str) -> list[dict]:
    """Paragraphs of printed words from a page's transcription Markdown.

    Markup is removed (heading hashes, bold/italic asterisks, footnote
    references, verse fences); each verse hemistich becomes its own paragraph
    because it is printed on its own line or half-line. Footnote definitions
    are kept as paragraphs (they are printed text)."""
    out = []
    in_verse = False
    block: list[str] = []
    kind = "prose"

    def flush():
        nonlocal block
        text = " ".join(block).strip()
        if text:
            out.append(dict(kind=kind, tokens=text.split()))
        block = []

    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            flush()
            in_verse = not in_verse
            kind = "verse" if in_verse else "prose"
            continue
        if in_verse:
            for hemi in line.split(" --- "):
                block = [hemi]
                flush()
            continue
        if not line:
            flush()
            kind = "prose"
            continue
        if line.startswith("#"):
            flush()
            kind = "heading"
            block = [line.lstrip("#").strip()]
            flush()
            kind = "prose"
            continue
        line = _FOOTNOTE_REF.sub("", line).replace("**", "").replace("__", "")
        line = _re.sub(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", line)
        block.append(line)
    flush()
    return out


def line_ink_width(ink, track: Track, spacing: float) -> float:
    band, offs, xs = band_columns(ink, track, 0.30 * spacing, 0.30 * spacing, track.x0, track.x1)
    return float((band.sum(0) >= 1).sum())


def line_fit_cost(model, tokens, alpha, tau, sigma_w: float = 0.12) -> float:
    """How well ``tokens`` explain one detected line: ink width against the
    page's ink-per-character, plus the best word segmentation cost with gaps
    judged against the page-wide word-gap threshold."""
    L, gaps, hidden, inkw = model
    if not tokens:
        return 1e9
    parts, owner, soft = token_parts(tokens)
    weights = [char_weight(t) for t in parts]
    exp = alpha * sum(weights)
    c = (math.log(max(1.0, inkw) / max(1.0, exp)) / sigma_w) ** 2
    if len(parts) > 1:
        c += choose_boundaries(L, gaps, weights, hidden, soft, gap_ref=tau)[4]
    return c


def refine_distribution(per_line, models, alpha, paragraphs, tau, window: int = 3, short=None):
    """Re-decide every line break within +-window tokens of the first guess,
    jointly (a DP over break positions), scoring each line by
    ``line_fit_cost``. Paragraph starts stay line starts."""
    toks = [t for p in paragraphs for t in p]
    T, Lc = len(toks), len(per_line)
    para_start = set()
    pos = 0
    for p in paragraphs:
        para_start.add(pos)
        pos += len(p)
    init = [0]
    for tl in per_line:
        init.append(init[-1] + len(tl))
    cand = []
    for l in range(Lc + 1):
        if l == 0:
            cand.append([0])
        elif l == Lc:
            cand.append([T])
        elif init[l] in para_start:
            cand.append([init[l]])
        else:
            cand.append([b for b in range(max(1, init[l] - window), min(T - 1, init[l] + window) + 1)])
    memo = {}

    def cost(l, a, b):
        if b <= a or any(x in para_start for x in range(a + 1, b)):
            return 1e9
        key = (l, a, b)
        if key not in memo:
            memo[key] = line_fit_cost(models[l], toks[a:b], alpha, tau)
            if short is not None and short[l] and b < T and b not in para_start:
                memo[key] += SHORT_LINE_PENALTY
        return memo[key]

    best = {0: (0.0, None)}
    hist = [best]
    for l in range(Lc):
        nxt = {}
        for b in cand[l + 1]:
            for a, (ca, _) in hist[-1].items():
                c = ca + cost(l, a, b)
                if b not in nxt or c < nxt[b][0]:
                    nxt[b] = (c, a)
        hist.append(nxt)
    if T not in hist[-1] or hist[-1][T][0] >= 1e9:
        return per_line
    brk = [T]
    for l in range(Lc, 0, -1):
        brk.append(hist[l][brk[-1]][1])
    brk.reverse()
    return [toks[brk[l]:brk[l + 1]] for l in range(Lc)]


def draft_unbroken(gray: np.ndarray, md: str, region=None, header_band: float = 0.12) -> dict:
    """Draft for a book page: detect every printed line in the text block,
    then distribute the transcription's words over them."""
    ink = binarize(gray)
    H, W = ink.shape
    tracks, info = find_tracks(ink, region=region)
    measure_tracks(ink, tracks, info)
    sp = info["spacing"]
    regions = []
    # Running head: everything above the lowest full-width horizontal rule in
    # the top part of the page (this book prints the head between two rules).
    xa, xb = info["xa"], info["xb"]
    top = int(header_band * H)
    fill = ink[:top, xa:xb].mean(1) if top > 0 else np.zeros(0)
    rule_rows = np.nonzero(fill > 0.85)[0]
    body = list(range(len(tracks)))
    if len(rule_rows):
        y_lo, y_hi = int(rule_rows.min()), int(rule_rows.max())
        head = [i for i, t in enumerate(tracks) if float(np.median(t.ys)) < y_hi + 0.2 * sp]
        regions.append(dict(kind="running_head", rect_px=[float(xa), float(max(0, y_lo - 0.6 * sp)), float(xb), float(y_hi + 3)]))
        body = [i for i in body if i not in head]
    # Other rules / tiny marks are not text lines.
    body = [i for i in body if not tracks[i].rule and tracks[i].x1 - tracks[i].x0 > 0.6 * sp]
    lt = [tracks[i] for i in body]
    paras = markdown_paragraphs(md)
    inkw, gapsets = [], []
    for t in lt:
        band, offs, xs = band_columns(ink, t, 0.30 * sp, 0.30 * sp, t.x0, t.x1)
        on = band.sum(0) >= 1
        inkw.append(float(on.sum()))
        gapsets.append([b - a for a, b in runs(~on) if a > 0 and b < len(on)])
    tau = otsu_1d([g for gs in gapsets for g in gs])
    clear = [sum(1 for g in gs if g > tau) for gs in gapsets]
    # A line whose left end stops well short of the text block's left margin
    # is the last line of a paragraph (or a centred verse/heading line).
    left = float(np.percentile([t.x0 for t in lt], 20)) if lt else 0.0
    short = [t.x0 > left + 1.5 * sp for t in lt]
    dist = distribute_tokens(inkw, [p["tokens"] for p in paras], clear, short=short)
    lines = []
    if dist is None:
        raise RuntimeError("cannot distribute the transcription over the detected lines")
    per_line, costs, alpha, _total, zg = dist
    bands = _bands(lt, sp)
    models = [line_gap_model(ink, t, sp, up, dn) for t, (up, dn) in zip(lt, bands)]
    per_line = refine_distribution(per_line, models, alpha, [p["tokens"] for p in paras], tau, short=short)
    # Sensitivity: would moving one word across a line break fit about as well?
    alt_flags = [[] for _ in lt]
    para_start = set()
    pos = 0
    for p in paras:
        para_start.add(pos)
        pos += len(p["tokens"])
    starts_idx = []
    acc = 0
    for toks in per_line:
        starts_idx.append(acc)
        acc += len(toks)

    def lcost(l, toks, ends_para=False):
        c = line_fit_cost(models[l], toks, alpha, tau)
        return c + (SHORT_LINE_PENALTY if short[l] and not ends_para else 0.0)

    for l in range(len(lt) - 1):
        a, b = per_line[l], per_line[l + 1]
        if starts_idx[l + 1] in para_start:
            continue  # a paragraph break is not negotiable
        base = lcost(l, a) + lcost(l + 1, b)
        alts = []
        if len(a) > 1:
            alts.append(lcost(l, a[:-1]) + lcost(l + 1, [a[-1]] + b))
        if len(b) > 1:
            alts.append(lcost(l, a + [b[0]]) + lcost(l + 1, b[1:]))
        if alts and min(alts) - base < BREAK_MARGIN:
            alt_flags[l].append("line break uncertain: the last word may belong to the next line")
            alt_flags[l + 1].append("line break uncertain: the first word may belong to the previous line")
    kinds = []
    for l, toks in enumerate(per_line):
        kinds.append(next((p["kind"] for p in paras if toks and toks[0] in p["tokens"]), "prose"))
    for l, toks in enumerate(per_line):
        t = lt[l]
        up, dn = bands[l]
        reasons = list(alt_flags[l])
        if abs(costs[l]) > 0.25:
            reasons.append(f"line ink {'wider' if costs[l] > 0 else 'narrower'} than its assigned words by {abs(math.exp(costs[l]) - 1):.0%}")
        boxes, wreasons, lreasons = segment_words(ink, t, t.x0, t.x1, toks, sp, up, dn)
        lines.append(dict(text=" ".join(toks), tokens=toks, track=t, up=up, down=dn, words=boxes,
                          word_reasons=wreasons, reasons=reasons + lreasons, notes=list(t.reasons),
                          candidate=body[l]))
    for r in regions:
        x0, y0, x1, y1 = r["rect_px"]
        r["rect_px"] = [max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)]
    return dict(lines=lines, regions=regions,
                info=dict(spacing=sp, alpha=alpha, gap_threshold=tau, zwnj_gap=zg, xa=info["xa"], xb=info["xb"], candidates=len(tracks),
                          body=body, paragraphs=len(paras)),
                tracks=tracks)
