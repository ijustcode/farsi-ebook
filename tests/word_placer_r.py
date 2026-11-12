#!/usr/bin/env python3
"""Arm R: render the known line text and warp it onto the line (plan §4.3,
Hassner/Wolf/Dershowitz-style transcript alignment). numpy + PyMuPDF, $0.

Per line: the known tokens are laid out as one right-to-left line with the
synthetic generator's shaping path (PyMuPDF ``insert_htmlbox``; each token
in its own id-coloured span, so the rendered word boundaries are exact).
The rendered column ink profile is resampled to the photographed line's ink
extent and aligned to the line's column ink profile by one-dimensional
dynamic time warping with a Sakoe-Chiba band (bounded local stretch) and a
step penalty. The rendered gaps between words are mapped through the warp;
boxes are snapped to the line's ink exactly as arm G's are.

The face is chosen per page from the installed candidates by the lowest mean
DTW cost over the page's lines (a run-time choice from image and text only).
Statuses: ``R`` located for every word with a box; ``RG`` located only where
R and arm G agree on both edges within half a letter.
"""
from __future__ import annotations

import os

import numpy as np

import word_placer as WP

RP = dict(
    SIZE_PT=40.0,            # render size (the profile is resampled anyway)
    BIN_SW=1.0,              # profile bin = this x stroke width (>= 2 px)
    BAND=0.15,               # warp band: |i - j| <= BAND x length + 3 bins
    STEP_PEN=0.15,           # cost of a non-diagonal step (local stretch)
    FONT_LINES=10,           # lines used to choose the face
)


def font_candidates():
    """Every installed Persian-capable face the generator knows (first
    choice and alternatives of the typewriter, modern and book classes)."""
    import synth_pages as S
    out, seen = [], set()
    for cls in ("typewriter", "modern_naskh", "book_naskh"):
        for name, path in S.FONT_CANDIDATES.get(cls, []):
            if path in seen or not os.path.exists(path):
                continue
            try:
                ok = S.font_status(path)["persian_letters"]
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                seen.add(path)
                out.append((cls, name, path))
    return out


_RENDER_CACHE: dict = {}


def render_tokens(tokens: list[str], font_path: str):
    """Rendered line: (occupancy per column right->left as float 0/1,
    token index per column or -1). Cached."""
    key = (tuple(tokens), font_path)
    if key in _RENDER_CACHE:
        return _RENDER_CACHE[key]
    import fitz
    import synth_pages as S
    W, H = 6000.0, 3.0 * RP["SIZE_PT"]
    doc = fitz.open()
    pg = doc.new_page(width=W, height=H)
    css = S._css(font_path, "F", RP["SIZE_PT"], 1.2, "right")
    arch = fitz.Archive(os.path.dirname(font_path))
    pg.insert_htmlbox(fitz.Rect(10, 5, W - 10, H - 5), S._block_html(tokens, 0), css=css, archive=arch, scale_low=1)
    ids = S.render_idmap(pg)          # token id + 1 per pixel, 0 elsewhere (anti-aliasing off)
    doc.close()
    occ = (ids > 0)
    cols = occ.any(axis=0)
    if not cols.any():
        _RENDER_CACHE[key] = (np.zeros(0), np.zeros(0, int))
        return _RENDER_CACHE[key]
    x0, x1 = np.nonzero(cols)[0][[0, -1]]
    sub = ids[:, x0:x1 + 1]
    occf = (sub > 0).mean(axis=0)
    tok = np.full(sub.shape[1], -1)
    for c in range(sub.shape[1]):
        v = sub[:, c][sub[:, c] > 0]
        if len(v):
            tok[c] = int(np.bincount(v).argmax()) - 1
    res = (occf[::-1].astype(np.float64), tok[::-1])     # reading direction (right -> left)
    _RENDER_CACHE[key] = res
    return res


def strip_profile(li: WP.LineInk, b: float):
    """Column ink occupancy of the line in reading coordinate s, binned by b,
    over the line's robust extent (small isolated end runs trimmed)."""
    s = li.uref - li.u
    if li.M == 0:
        return None
    mref = float(np.median(li.mass)) if li.M else 1.0
    a, z = 0, li.M - 1
    gaps = li.s0[1:] - li.s1[:-1]
    mg = float(np.median(gaps)) if len(gaps) else 0.0
    while a < z and li.mass[a] < 0.3 * mref and (li.s0[a + 1] - li.s1[a]) > 2 * mg + 1:
        a += 1
    while z > a and li.mass[z] < 0.3 * mref and (li.s0[z] - li.s1[z - 1]) > 2 * mg + 1:
        z -= 1
    s_lo, s_hi = float(li.s0[a]), float(li.s1[z])
    n = max(4, int(np.ceil((s_hi - s_lo) / b)))
    sel = (s >= s_lo) & (s < s_hi)
    bi = np.clip(((s[sel] - s_lo) / b).astype(int), 0, n - 1)
    vb = np.floor(li.v[sel]).astype(int)
    # occupancy: share of distinct rows with ink in each bin, normalised
    occ = np.zeros(n)
    if len(bi):
        key = bi * 100000 + (vb - vb.min())
        uk = np.unique(key)
        occ = np.bincount(uk // 100000, minlength=n).astype(np.float64)
        occ /= max(1.0, np.quantile(occ[occ > 0], 0.95) if (occ > 0).any() else 1.0)
    return np.clip(occ, 0, 1.0), s_lo, b


def resample(prof: np.ndarray, n: int) -> np.ndarray:
    if len(prof) == 0:
        return np.zeros(n)
    x = (np.arange(n) + 0.5) * len(prof) / n - 0.5
    return np.interp(x, np.arange(len(prof)), prof)


def dtw(q: np.ndarray, r: np.ndarray):
    """Banded DTW, steps diagonal (cost d) and horizontal/vertical (d + pen).
    Returns (mean cost per step, path as list of (i, j))."""
    n, m = len(q), len(r)
    band = int(RP["BAND"] * max(n, m)) + 3
    pen = RP["STEP_PEN"]
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jc = int(round(i * m / n))
        lo, hi = max(1, jc - band), min(m, jc + band)
        di = np.abs(q[i - 1] - r[lo - 1:hi])
        diag = D[i - 1, lo - 1:hi]
        up = D[i - 1, lo:hi + 1] + pen
        best = np.minimum(diag, up) + di
        row = D[i]
        prev = row[lo - 1]
        for k, j in enumerate(range(lo, hi + 1)):
            v = min(best[k], prev + pen + di[k])
            row[j] = v
            prev = v
    # backtrack
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        cands = []
        if i > 1 and j > 1:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 1:
            cands.append((D[i - 1, j] + pen, i - 1, j))
        if j > 1:
            cands.append((D[i, j - 1] + pen, i, j - 1))
        _, i, j = min(cands)
        path.append((i - 1, j - 1))
    path.reverse()
    return float(D[n, m]) / max(1, len(path)), path


def align_line_r(li: WP.LineInk, tokens: list[str], font_path: str, sw: float):
    """Word intervals in s (logical token order) and the DTW cost."""
    b = max(2.0, RP["BIN_SW"] * sw)
    sp = strip_profile(li, b)
    if sp is None or not tokens:
        return None, None
    q, s_lo, b = sp
    # logical tokens: the HTML engine applies bidi itself; colour ids = logical indices
    occ, tok = render_tokens(list(tokens), font_path)
    if not len(occ):
        return None, None
    n = len(q)
    r = resample(occ / max(1e-9, np.quantile(occ[occ > 0], 0.95) if (occ > 0).any() else 1.0), n)
    tok_r = tok[np.clip(((np.arange(n) + 0.5) * len(tok) / n).astype(int), 0, len(tok) - 1)]
    cost, path = dtw(np.clip(q, 0, 1), np.clip(r, 0, 1))
    mapped = np.zeros(n); cnt = np.zeros(n)
    for i, j in path:
        mapped[j] += i; cnt[j] += 1
    mapped = mapped / np.maximum(cnt, 1)
    spans = {}
    for j in range(n):
        if tok_r[j] >= 0:
            a = spans.setdefault(int(tok_r[j]), [j, j])
            a[0] = min(a[0], j); a[1] = max(a[1], j)
    ids = sorted(spans, key=lambda k: spans[k][0])       # in reading (right -> left) order
    cuts = []
    for k1, k2 in zip(ids, ids[1:]):
        g = (spans[k1][1] + spans[k2][0]) / 2.0
        cuts.append(float(np.interp(g, np.arange(n), mapped)))
    edges = [0.0] + cuts + [float(n)]
    out = [None] * len(tokens)
    for k, tid in enumerate(ids):
        if 0 <= tid < len(tokens):
            out[tid] = (s_lo + edges[k] * b, s_lo + edges[k + 1] * b)
    return out, cost


def choose_font(lines, sw, exclude: str | None = None):
    """lines: [(li, tokens)] -> (font_path, {name: mean cost}). ``exclude``:
    a font file basename never used (synthetic pages: the page's own face, so
    the test matches real pages, where no identical face exists)."""
    costs = {}
    for cls, name, path in font_candidates():
        if exclude and os.path.basename(path) == os.path.basename(exclude):
            continue
        cs = []
        for li, toks in lines[:RP["FONT_LINES"]]:
            _, c = align_line_r(li, toks, path, sw)
            if c is not None:
                cs.append(c)
        if cs:
            costs[(cls, name, path)] = float(np.mean(cs))
    if not costs:
        return None, {}
    best = min(costs, key=costs.get)
    return best[2], {f"{k[0]}:{k[1]}": round(v, 4) for k, v in costs.items()}
