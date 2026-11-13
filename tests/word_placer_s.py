#!/usr/bin/env python3
"""Arm S and the agreement rules (Step 4 continuation, owner request).

**Arm S** is the audit tool's draft word splitter (``tests/box_audit_draft.py``
``segment_words``: text-constrained choice of white gaps by character weight,
proportional cuts where gaps are hidden) driven with *external* lines (the
audited line frames for experiment A, L1's lines for B) and the same line
text as every other arm. ``box_audit_draft`` is imported, not edited: its
``segment_words`` takes a ``Track`` (centre y as a function of x), an x range,
the tokens, the line spacing and the half-band heights, so the only new code
here is the adapter that builds a Track from a line geometry, measures the
line's x extent with the draft's own ``line_extent`` (end specks dropped by
its rule) and its half-bands with ``_bands``, and turns the per-word
[x0, x1, top, bottom] back into quads. The draft binarises the page with its
own ``binarize``. S shows every box (it has no confidence): status
``located`` for every word with a box.

**Agreement** (defined once, a priori, used by every agreement arm):
two arms agree on a word when both show a box for it and both box edges along
the line (the box's extent projected on the line's own axis, in the line
frame used by every arm) are within ``AGREE_TOL`` = 0.5 letter unit (arm G's
per-page ink extent per letter) of each other. Boxes compared are each arm's
*unthresholded* box: S (every box), R (every box), G (its best segmentation,
whatever its margin), T (every box T did not bypass). An agreement arm shows
the box of the first agreeing arm in the fixed order T, R, G, S, as
``located``; otherwise ``line`` (or ``unresolved`` when the line is not
trusted). Majority rules: ``M2`` / ``M3`` = at least 2 / 3 of {S, R, G, T}
mutually agree with one of them (the anchor: the first arm in that order
with enough agreeing partners).
"""
from __future__ import annotations

import numpy as np

import box_audit_draft as bd
import word_placer as WP

AGREE_TOL = 0.5
ORDER = ("T", "R", "G", "S")


def track_of(geom, n: int = 24) -> "bd.Track":
    xs = np.linspace(float(geom.x0), float(geom.x1), n)
    ys = np.asarray(geom.centre_y(xs), np.float64)
    return bd.Track(xs=xs, ys=ys)


def s_page(gray: np.ndarray, line_items: list, spacing: float) -> dict:
    """line_items: [(key, geom, tokens)] in top-to-bottom order. Returns
    {key: per logical token quad (page px) or None}."""
    ink = bd.binarize(gray)
    tracks = []
    for key, geom, toks in line_items:
        t = track_of(geom)
        e = bd.line_extent(ink, t, spacing, int(geom.x0), int(geom.x1))
        if e is None:
            t.x0, t.x1 = float(geom.x0), float(geom.x1)
        else:
            t.x0, t.x1 = e[0], e[1]
        tracks.append(t)
    bands = bd._bands(tracks, spacing) if tracks else []
    out = {}
    for (key, geom, toks), t, (up, dn) in zip(line_items, tracks, bands):
        order = WP.visual_order(list(toks))
        vis = [toks[i] for i in order]
        quads = [None] * len(toks)
        if t.x1 - t.x0 >= 2 and vis:
            boxes, _wr, _lr = bd.segment_words(ink, t, t.x0, t.x1, vis, spacing, up, dn)
            for w, (x0, x1, dt, db) in enumerate(boxes):
                if w < len(order) and x1 > x0:
                    quads[order[w]] = [(x0, float(t.y_at(x0)) + dt), (x1, float(t.y_at(x1)) + dt),
                                       (x1, float(t.y_at(x1)) + db), (x0, float(t.y_at(x0)) + db)]
        out[key] = quads
    return out


def s_results(quads: list, line_quad) -> list:
    return [dict(status="located", quad=q, line_quad=line_quad, margin=None, z=None, reason="S")
            if q is not None else dict(status="line" if line_quad else "unresolved", quad=None, line_quad=line_quad,
                                       margin=None, z=None, reason="S: no box")
            for q in quads]


def _extent(li, q):
    import word_placer_v as V
    return V.quad_s_extent(li, q)


def boxes_of(res: dict) -> dict:
    """Unthresholded box per arm for one word (None when the arm shows none
    or bypassed the word)."""
    out = {}
    for name, x in res.items():
        q = x.get("quad") if x else None
        if q is not None and "bypassed" in (x.get("reason") or ""):
            q = None
        out[name] = q
    return out


def agree_pair(li, qa, qb, unit) -> bool:
    if qa is None or qb is None:
        return False
    ea, eb = _extent(li, qa), _extent(li, qb)
    tol = AGREE_TOL * unit
    return abs(ea[0] - eb[0]) <= tol and abs(ea[1] - eb[1]) <= tol


def agreement_line(li, per_arm: dict, unit: float, rule) -> list:
    """per_arm: {arm: [result dict per logical token]} for one line.
    rule: tuple of arm names (all must agree) or ("M", k) (k of ORDER).
    Returns result dicts per token."""
    names = [n for n in ORDER if n in per_arm]
    n_tok = len(next(iter(per_arm.values())))
    out = []
    for ti in range(n_tok):
        res = {n: per_arm[n][ti] for n in names}
        q = boxes_of(res)
        any_x = next((res[n] for n in names if res[n]), {})
        lq = next((res[n].get("line_quad") for n in names if res[n] and res[n].get("line_quad")), None)
        trusted = not all((res[n] or {}).get("status") == "unresolved" for n in names)
        shown = None
        if rule[0] == "M":
            k = rule[1]
            for a in names:
                if q[a] is None:
                    continue
                partners = [b for b in names if b != a and agree_pair(li, q[a], q[b], unit)]
                if len(partners) + 1 >= k:
                    shown = (a, q[a])
                    break
        else:
            arms = [a for a in ORDER if a in rule]
            if all(q.get(a) is not None for a in arms) and all(
                    agree_pair(li, q[arms[0]], q[b], unit) for b in arms[1:]):
                shown = (arms[0], q[arms[0]])
        if shown and trusted:
            out.append(dict(status="located", quad=shown[1], line_quad=lq, margin=None, z=None,
                            reason=f"agree ({shown[0]})"))
        elif not trusted:
            out.append(dict(status="unresolved", quad=None, line_quad=None, margin=None, z=None,
                            reason=any_x.get("reason", "line assignment not trusted")))
        else:
            out.append(dict(status="line" if lq else "unresolved", quad=None, line_quad=lq, margin=None, z=None,
                            reason="arms disagree or a box is missing"))
    return out


# ---------------------------------------------------------------------------
# Shared vertical extent (follow-up 1): one rule for every arm
# ---------------------------------------------------------------------------
VP = dict(
    BAND_Q=0.0,        # line band: the line's own ink v quantiles BAND_Q .. 1 - BAND_Q (0: its full extent; 0.005 clipped the tallest strokes)
)


def line_band(li):
    """(v_lo, v_hi) of a line's own ink: all pixels of the components
    assigned to this line (nearest line centre, word_placer.assign_components),
    robust to a few far pixels of another line's tall stroke."""
    if not len(li.v):
        return None
    return float(np.quantile(li.v, VP["BAND_Q"])), float(np.quantile(li.v, 1 - VP["BAND_Q"]))


def vertical_box(li, quad, band=None):
    """The same horizontal extent, re-measured vertically: the box spans all
    ink of this line (every pixel of its assigned components, detached dots
    and marks included) inside the box's horizontal span, bounded by the line
    band. Returns a quad in page px (None stays None)."""
    if quad is None or not len(li.u):
        return quad
    band = band or line_band(li)
    a, b = _extent(li, quad)            # reading coordinate s
    s = li.uref - li.u
    sel = (s >= a - 0.5) & (s <= b + 0.5) & (li.v >= band[0]) & (li.v <= band[1])
    if sel.sum() < 2:
        return quad
    v0, v1 = float(li.v[sel].min()) - 0.5, float(li.v[sel].max()) + 0.5
    import word_placer as WP_
    return WP_._quad_from(li, li.uref - b, li.uref - a, v0, v1, 0.0)


def vertical_line_comp(li, results, sw, unit, hc):
    """Shared vertical rule, component version (the one adopted): every arm's
    boxes on one line keep their horizontal extent; the vertical extent is
    that of the ink the word owns, decided once for all words of the line by
    word_placer.snap_words (components to the word holding most of their core
    pixels; detached dots and marks to the word containing their centre, when
    they lie within the line's glyph zone: its core-attached ink's v range
    widened by DOT_PAD x glyph height, which keeps the neighbouring lines'
    ink out), bounded by the line's own ink band.
    Tried and rejected on synthetic development pages (see the run log):
    (a) the pixel version above (``vertical_box``: all of the line's pixels in
    the word's span) takes other words' tails and dots that cross the span
    (nastaliq T 168 -> 151 of 172 correct); with the band at the 0.5% / 99.5%
    quantiles it also clipped the tallest strokes (clean dev T 90.3% -> 85.1%);
    (b) the detached-ink gate set to the bare core-attached range (no DOT_PAD)
    drops dots above lines without tall letters (tight-leading naskh T 541 ->
    478). For T, R and G this rule reproduces their own boxes; it changes S,
    whose draft routine measured height from its own band."""
    import word_placer as WP_
    ivs = []
    for x in results:
        q = x.get("quad") if x else None
        ivs.append(_extent(li, q) if q is not None else None)
    if not any(iv is not None for iv in ivs) or not len(li.u):
        return results
    band = line_band(li)
    owned = WP_.snap_words(li, ivs, sw, unit, hc)
    out = []
    for x, iv, qo in zip(results, ivs, owned):
        if iv is None or qo is None:
            out.append(x)
            continue
        xs = np.array([p[0] for p in qo]); ys = np.array([p[1] for p in qo])
        _, vv = li.geom.uv(xs, ys)
        v0, v1 = max(float(vv.min()), band[0] - 0.5), min(float(vv.max()), band[1] + 0.5)
        out.append(dict(x, quad=WP_._quad_from(li, li.uref - iv[1], li.uref - iv[0], v0, v1, 0.0)))
    return out


# ---------------------------------------------------------------------------
# Follow-up 2a: rules between T alone and S&T (defined before scoring)
# ---------------------------------------------------------------------------
GP = dict(
    OVER_WIDTH=1.5,    # T guard: aligned width > this x the width arm G's table predicts ...
    OVER_ABS=1.0,      # ... plus this many letter units -> over-extended
)


def t_guard_line(L, al, gwidth, unit):
    """Per logical token of a line: True when arm T's box is guarded out:
    (1) a word next to (in visual order) a word T bypassed: a bypass lets the
    neighbour stretch over the bypassed word's ink (p66 line 28 word 5);
    (2) over-extension: the aligned width exceeds OVER_WIDTH x the width G's
    letter table predicts for the word's units, plus OVER_ABS letters."""
    n = len(L.tokens)
    out = [False] * n
    if al is None:
        return out
    W = al.words
    for w, d in enumerate(W):
        ti = L.order[w]
        nb = any(W[v]["deleted"] for v in (w - 1, w + 1) if 0 <= v < len(W))
        over = False
        if not d["deleted"] and d.get("t0") is not None:
            pred = sum(gwidth(k) for k, u in zip(L.ukeys, L.uword) if u == w)
            width = (d["t1"] - d["t0"] + 1) * L.cs
            over = width > GP["OVER_WIDTH"] * pred + GP["OVER_ABS"] * unit
        out[ti] = bool(nb or over)
    return out


def apply_guard(res, guard):
    out = []
    for x, g in zip(res, guard):
        if g and x.get("status") == "located":
            out.append(dict(x, status="line" if x.get("line_quad") else "unresolved",
                            reason="T guard: next to a bypassed word or over-extended"))
        else:
            out.append(dict(x))
    return out


def t_any_agree(li, t_all, others: dict, unit):
    """T's box, located when at least one of the other arms (S, G, R)
    agrees with it on both side edges (agree_pair)."""
    out = []
    names = list(others)
    for ti, x in enumerate(t_all):
        q = x.get("quad")
        ok = (q is not None and "bypassed" not in (x.get("reason") or "")
              and x.get("status") in ("located", "line")
              and any(agree_pair(li, q, (others[n][ti] or {}).get("quad"), unit) for n in names))
        if ok:
            out.append(dict(x, status="located", reason="T agrees with another arm"))
        elif x.get("status") == "unresolved":
            out.append(dict(x))
        else:
            out.append(dict(x, status="line" if x.get("line_quad") else "unresolved",
                            reason="no other arm agrees with T"))
    return out
