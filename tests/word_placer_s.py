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
