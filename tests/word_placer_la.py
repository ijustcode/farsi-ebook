#!/usr/bin/env python3
"""Template-scored line-text assignment (Step 4 follow-up 2; plan §4.2).

The §4.2 length assignment (``word_placer.assign_text_lines``) matches text
lines to detected lines by characters versus ink extent. It is right on most
lines but cannot tell a running head, a stamp line or a block of equally long
lines apart, so whole blocks stay untrusted (low margin) or shift by one.

Here the page's own letter templates (arm T, trained on the lines the length
assignment already trusts) score each candidate pairing directly: align text
line i to detected line j with the T model and compare the cost with the
line's *null* cost (every column blank or not-text). The gain
``null_j - align_ij`` is large when the text is printed on that line and near
zero or negative otherwise (words are bypassed, ink goes to not-text). A
monotone dynamic programme picks the pairing with the largest total gain;
detected lines and text lines may stay unmatched. A text line is trusted only
when (1) its gain per text letter is at least ``GAIN_REL`` x the page's
median over matched pairs (a line holding only part of the text, or other
text, explains much less of it), (2) the best total with that line moved
elsewhere or unmatched is lower by at least ``MARGIN_REL`` x its gain and
``MARGIN_ABS`` nats, and (3) the detected line is one L1 classes as text. A
line failing any test stays untrusted: its words are ``unresolved`` (gate G2:
a box is never shown on a line that may be non-text).

Candidates: each text line against the detected lines within ``WINDOW`` of
its length-assignment position (or of the position interpolated between its
assigned neighbours).
"""
from __future__ import annotations

import math

import numpy as np

import word_placer as WP
import word_placer_t as T

LP = dict(
    WINDOW=3,          # candidate detected lines: +- this around the length-assignment position
    MARGIN_REL=0.25,   # trust: margin >= this x the pair's gain ...
    MARGIN_ABS=20.0,   # ... and >= this (nats) ...
    GAIN_REL=0.5,      # ... and gain per text letter >= this x the page's median over matched pairs
                       # (a partial or wrong line explains at most part of the text)
    TRAIN_ITERS=2,     # T rounds for the scoring model (trusted lines of the length assignment)
    DEF_K=3.0,         # --la T2: a trusted pairing's deficit per letter against the free loop (free_cost) must be
    DEF_ABS=5.0,       # <= page median + DEF_K x MAD + DEF_ABS (over the page's trusted pairings); set on synthetic dev
    CONFIRM_FRAC=0.5,  # --la T2: a trusted pairing needs at least this share of the text's letters aligned
                       # (not bypassed) on that ink. Requiring every word (1.0) cost a quarter of the
                       # located words on synthetic hard pages (words under stamps are bypassed by design)
)


def null_cost(L: T.TLine, m: T.TModel) -> float:
    """Cost of the line's columns when nothing aligns: each column blank or
    not-text, whichever is cheaper (transition costs ignored, a lower bound
    in favour of 'no text')."""
    iv = 1.0 / m.var
    const = 0.5 * float(np.sum(np.log(2 * math.pi * m.var)))
    xx = (L.X * L.X) @ iv
    blank = 0.5 * xx + const
    junk = 0.5 * xx / T.TP["JUNK_VAR"] + const + 0.5 * m.R * math.log(T.TP["JUNK_VAR"])
    return float(np.minimum(blank, junk).sum())


def pair_gains(ev, inks, texts, cands, m: T.TModel) -> np.ndarray:
    """G[i, j] = null_j - align(text i on line j); -inf where not a candidate."""
    mt, n = len(texts), len(inks)
    G = np.full((mt, n), -np.inf)
    nulls = {}
    batch, where = [], []
    for i, js in enumerate(cands):
        if not texts[i]:
            continue
        for j in js:
            li = inks[j]
            if not li.M:
                continue
            L = T.make_line(li, list(texts[i]), ev.hc, ev.sw, cs=m.cs)
            if L.T == 0 or not L.ukeys:
                continue
            if j not in nulls:
                nulls[j] = null_cost(L, m)
            batch.append(L)
            where.append((i, j))
    for k0 in range(0, len(batch), 40):
        part = batch[k0:k0 + 40]
        B = T.make_batch(part, m)
        F, _BP = T.forward(B)
        for k, L in enumerate(part):
            g = B.graphs[k]
            tot = float(np.min(F[L.T - 1, k, :g.S] + B.endc[k, :g.S]))
            i, j = where[k0 + k]
            G[i, j] = nulls[j] - tot
    return G


def monotone_best(G: np.ndarray):
    """Maximum-total monotone matching (unmatched allowed, gain <= 0 never
    matched) with, per text line, the margin: best total minus the best total
    with that line not on its chosen detected line."""
    mt, n = G.shape
    P = np.where(np.isfinite(G) & (G > 0), G, -np.inf)
    A = np.zeros((mt + 1, n + 1))
    for i in range(1, mt + 1):
        for j in range(1, n + 1):
            A[i, j] = max(A[i - 1, j], A[i, j - 1], A[i - 1, j - 1] + P[i - 1, j - 1])
    Bs = np.zeros((mt + 2, n + 2))
    for i in range(mt, 0, -1):
        for j in range(n, 0, -1):
            Bs[i, j] = max(Bs[i + 1, j], Bs[i, j + 1], Bs[i + 1, j + 1] + P[i - 1, j - 1])
    total = A[mt, n]
    # value with text i matched to j: A[i-1, j-1] + P + Bs[i+1, j+1] (1-based)
    V = np.full((mt, n), -np.inf)
    for i in range(mt):
        for j in range(n):
            if np.isfinite(P[i, j]):
                V[i, j] = A[i, j] + P[i, j] + Bs[i + 2, j + 2]
    assign, margin = [], []
    for i in range(mt):
        j = int(np.argmax(V[i])) if np.isfinite(V[i]).any() else None
        if j is None or V[i, j] < total - 1e-6:
            assign.append(None)
            margin.append(0.0)
            continue
        others = np.delete(V[i], j)
        alt_pair = float(others.max()) if len(others) and np.isfinite(others).any() else -np.inf
        # i unmatched: best over split points k of A[i, k] + Bs[i+2, k+1]
        alt_none = max(A[i, k] + Bs[i + 2, k + 1] for k in range(n + 1))
        assign.append(j)
        margin.append(float(total - max(alt_pair, alt_none)))
    return assign, margin, float(total)


def candidates(base_assign, n, mt):
    pos = []
    known = [(i, j) for i, j in enumerate(base_assign) if j is not None]
    for i in range(mt):
        j = base_assign[i]
        if j is None:
            prev = max([(a, b) for a, b in known if a < i], default=None)
            nxt = min([(a, b) for a, b in known if a > i], default=None)
            if prev and nxt:
                j = int(round(prev[1] + (nxt[1] - prev[1]) * (i - prev[0]) / max(1, nxt[0] - prev[0])))
            elif prev:
                j = prev[1] + (i - prev[0])
            elif nxt:
                j = nxt[1] - (nxt[0] - i)
            else:
                j = i
        W = LP["WINDOW"]
        pos.append(list(range(max(0, j - W), min(n, j + W + 1))))
    return pos


def confirm_pairs(ev, inks, texts, pairs, m: T.TModel, unit: float) -> dict:
    """G2 follow-up: for each (i, j) pairing, align text i to line j with the
    page templates and report whether at least CONFIRM_FRAC of its letters
    aligned (were not bypassed): the text is positively on this ink."""
    lines, keys = [], []
    for i, j in pairs:
        L = T.make_line(inks[j], list(texts[i]), ev.hc, ev.sw, cs=m.cs)
        if L.T and L.ukeys:
            lines.append(L); keys.append((i, j))
    out = {}
    if not lines:
        return out
    m.n_hint_unit = getattr(m, "n_hint_unit", unit)
    aligns, _tot, _B, live = T.align_all(m, lines, margins=False, unit_px=unit)
    for (i, j), al, L in zip(keys, aligns, lines):
        if al is None:
            out[(i, j)] = False
            continue
        nl = [sum(1 for k, u in zip(L.ukeys, L.uword) if u == w) for w in range(L.nwords)]
        kept = sum(n for n, d in zip(nl, al.words) if not d["deleted"])
        out[(i, j)] = kept >= LP["CONFIRM_FRAC"] * max(1, sum(nl))
    return out


def template_assign(ev, r, kinds, texts, m: T.TModel, confirm: bool = False):
    """New assignment for an arm-G placement result ``r`` (length assignment)
    -> dict(assign, margin (trusted lines LA_MARGIN, else 0), gains, raw_margin, trusted).

    confirm=True (``--la T2``, G2 follow-up): every trusted pairing must also
    pass confirm_pairs (most of the text's letters aligned on that ink) and the
    free-loop test (its alignment cost per letter within DEF_K MAD + DEF_ABS of
    the page median above what an unconstrained loop over the page's letters
    achieves on that ink: the text, not just any text, is printed there), and
    the 'length assignment agrees' route also requires an L1 text line."""
    inks = r["inks"]
    cands = candidates(r["assign"], len(inks), len(texts))
    G = pair_gains(ev, inks, texts, cands, m)
    assign, margin, total = monotone_best(G)
    nlet = [max(1.0, sum(WP.letters(t) for t in toks)) for toks in texts]
    per = [G[i, j] / nlet[i] for i, j in enumerate(assign) if j is not None]
    floor = LP["GAIN_REL"] * float(np.median(per)) if per else np.inf
    trusted = []
    for i, (j, mg) in enumerate(zip(assign, margin)):
        ok = (j is not None and G[i, j] > 0 and kinds[j] == "text" and G[i, j] / nlet[i] >= floor
              and mg >= max(LP["MARGIN_ABS"], LP["MARGIN_REL"] * G[i, j]))
        # the length assignment already trusted this same pairing: two sources agree
        both = (j is not None and r["assign"][i] == j and r["la"]["margin"][i] >= WP.P["LA_MARGIN"] and G[i, j] > 0
                and (kinds[j] == "text" or not confirm))
        trusted.append(bool(ok or both))
    confirmed = {}
    deficit = {}
    if confirm:
        pairs = [(i, j) for i, (j, t) in enumerate(zip(assign, trusted)) if t]
        confirmed = confirm_pairs(ev, inks, texts, pairs, m, r["model"].unit)
        trusted = [t and confirmed.get((i, assign[i]), False) for i, t in enumerate(trusted)]
        # "some text of this print" hypothesis: the given text must explain the ink nearly as
        # well as the free loop over the page's letters does (page-relative threshold)
        frees = {}
        for i, j in pairs:
            if j not in frees:
                frees[j] = free_cost(T.make_line(inks[j], ["x"], ev.hc, ev.sw, cs=m.cs), m)
            Lf = T.make_line(inks[j], list(texts[i]), ev.hc, ev.sw, cs=m.cs)
            align = null_cost(Lf, m) - G[i, j]
            deficit[i] = (align - frees[j]) / nlet[i]
        vals = np.array([deficit[i] for i, _ in pairs if trusted[i]]) if pairs else np.zeros(0)
        if len(vals):
            med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))) * 1.4826
            th = med + LP["DEF_K"] * mad + LP["DEF_ABS"]
            trusted = [t and (i not in deficit or deficit[i] <= th) for i, t in enumerate(trusted)]
    return dict(assign=assign, margin=[WP.P["LA_MARGIN"] if t else 0.0 for t in trusted], raw_margin=margin,
                trusted=trusted, gains=G, total=total, det_text=[None] * len(inks),
                unconfirmed=sum(1 for v in confirmed.values() if not v),
                deficit={int(k): round(float(v), 2) for k, v in deficit.items()})


def place_with_assignment(ev, r, kinds, texts, la, *, nottext=True):
    """Arm G re-run with a given assignment (same as place_assigned_lines
    after its own assignment step)."""
    inks = r["inks"]
    jobs = []
    for i, t in enumerate(texts):
        j = la["assign"][i]
        trusted = la["trusted"][i]
        jobs.append(WP.LineJob(list(t), inks[j] if j is not None else None, trusted, la["margin"][i]))
    res, model, diag = WP.place_jobs(ev.sw, jobs, nottext=nottext, hc=ev.hc)
    det_text = [None] * len(inks)
    for i, j in enumerate(la["assign"]):
        if j is not None:
            det_text[j] = i
    la = dict(la, det_text=det_text)
    return dict(results=res, model=model, diag=diag, inks=inks, assign=la["assign"], la=la, kinds=kinds)


# ---------------------------------------------------------------------------
# G2 follow-up: perturbed line text for synthetic development
# ---------------------------------------------------------------------------
def perturb_lines(texts, seed, n_invent=1, n_repeat=1, n_drop=1, pool=None):
    """Seeded perturbation of line-broken text as a reader might produce it:
    invented lines (words from ``pool``, text not printed on the page),
    repeated lines (a line given twice in a row) and dropped lines. Returns
    (texts, src) with src[l][t] = truth (li, wi) or ('x', l, t) for words not
    printed (invented lines, the second copy of a repeated line)."""
    import random
    rnd = random.Random(seed)
    lines = [(li, list(t)) for li, t in enumerate(texts)]
    idx = [k for k, (_, t) in enumerate(lines) if t]
    drop = set(rnd.sample(idx, min(n_drop, len(idx))))
    rep = set(rnd.sample([k for k in idx if k not in drop], min(n_repeat, len(idx) - len(drop))))
    out = []
    for k, (li, t) in enumerate(lines):
        if k in drop:
            continue
        out.append([(li, wi, w) for wi, w in enumerate(t)])
        if k in rep:
            out.append([(None, None, w) for w in t])
    pool = pool or [w for t in texts for w in t]
    for _ in range(n_invent):
        n = rnd.randint(3, 8)
        pos = rnd.randint(0, len(out))
        out.insert(pos, [(None, None, w) for w in rnd.sample(pool, min(n, len(pool)))])
    texts2 = [[w for _, _, w in row] for row in out]
    src = [[(li, wi) if li is not None else ("x", l, t) for t, (li, wi, _w) in enumerate(row)] for l, row in enumerate(out)]
    return texts2, src


# ---------------------------------------------------------------------------
# G2 follow-up: "any text here" hypothesis (free loop over the page's units)
# ---------------------------------------------------------------------------
def free_cost(L: T.TLine, m: T.TModel, min_occ: int = 2) -> float:
    """Viterbi cost of a line's columns under an unconstrained loop over the
    page's learned letter templates (units seen at least ``min_occ`` times),
    with blank and not-text states: how well *some* text in this print
    explains the ink. A given text can never beat it by much (it is one of
    the loop's paths, apart from the unit-choice costs); the deficit
    align(text) - free is small when the text is printed there and large when
    it is not."""
    import math
    keys = [k for k in sorted(m.tmpl, key=str) if m.occ.get(k, 0) >= min_occ]
    if not keys or L.T == 0:
        return float("inf")
    tot = float(sum(m.occ[k] for k in keys))
    R = m.R
    first, last, means, stay, adv, skip, entry = [], [], [], [], [], [], []
    pos = 0
    for k in keys:
        mu = m.tmpl[k]
        n = len(mu)
        ps, pa, pk = m.tr(k)
        first.append(pos); last.append(pos + n - 1)
        means.append(mu)
        stay += [T._nl(ps)] * n; adv += [T._nl(pa)] * n; skip += [T._nl(pk)] * n
        entry.append(-math.log(m.occ[k] / tot))
        pos += n
    S = pos + 2                      # + blank, junk
    B, J = pos, pos + 1
    M = np.zeros((S, R)); M[:pos] = np.concatenate(means)
    junk = np.zeros(S, bool); junk[J] = True
    g = T.Graph(S, M, junk, None, None, None, None, [], [], [], [])
    E = T.emissions(L, g, m).astype(np.float64)
    stay, adv, skip = np.array(stay), np.array(adv), np.array(skip)
    first, last, entry = np.array(first), np.array(last), np.array(entry)
    isfirst = np.zeros(pos, bool); isfirst[first] = True
    pb = m.gap[T.WORD][1]
    bstay, bexit = T._nl(pb), T._nl(1 - pb)
    INF = 1e30
    F = np.full(S, INF)
    F[first] = entry + E[0, first]; F[B] = E[0, B]; F[J] = T.TP["J_ENTER"] + E[0, J]
    for t in range(1, L.T):
        prev = F
        new = np.full(S, INF)
        u = prev[:pos] + stay
        a = np.full(pos, INF); a[1:] = prev[:pos - 1] + adv[1:]
        a[isfirst] = INF
        k2 = np.full(pos, INF); k2[2:] = prev[:pos - 2] + skip[2:]
        k2[isfirst] = INF; k2[np.r_[first + 1][first + 1 < pos]] = INF
        endbest = min(float(np.min(prev[last] + adv[last])), prev[B] + bexit, prev[J])
        en = np.full(pos, INF); en[first] = endbest + entry
        new[:pos] = np.minimum(np.minimum(u, a), np.minimum(k2, en))
        endu = float(np.min(prev[last] + adv[last]))
        new[B] = min(prev[B] + bstay, endu, prev[J])
        new[J] = min(prev[J], endu + T.TP["J_ENTER"], prev[B] + T.TP["J_ENTER"])
        F = new + E[t]
    return float(min(np.min(F[last]), F[B], F[J]))


# ---------------------------------------------------------------------------
# D15 follow-up: detected lines that look like non-document marks
# ---------------------------------------------------------------------------
def det_line_stats(ev, inks, kinds):
    """Run-time statistics per detected line (no truth): glyph height of its
    ink runs relative to the page's glyph height, ink density, share of the
    line's extent covered by L1's oversized (masked) components, and whether
    it lies below the lowest line of ordinary glyph statistics."""
    lf = ev.lf
    cc, reason = lf.get("cc"), lf.get("reason")
    big = np.nonzero(reason == "big")[0] if reason is not None else np.zeros(0, int)
    out = []
    for li, kind in zip(inks, kinds):
        if not li.M:
            out.append(dict(h_rel=0.0, dens=0.0, big=0.0, kind=kind, y=li.y_mid))
            continue
        h = float(np.median(li.height)) / max(1.0, ev.hc)
        dens = len(li.u) / max(1.0, float(np.sum(li.s1 - li.s0)) * ev.hc)
        xs, ys = li.geom.xy(li.uref - np.r_[li.s0, li.s1], np.full(2 * li.M, li.base_v))
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        cov = 0.0
        if len(big):
            ym = li.y_mid
            for c in big:
                if cc.y0[c] - 0.25 * ev.S <= ym <= cc.y1[c] + 0.25 * ev.S:
                    cov += max(0.0, min(x1, cc.x1[c]) - max(x0, cc.x0[c]))
            cov = min(1.0, cov / max(1.0, x1 - x0))
        out.append(dict(h_rel=round(h, 3), dens=round(dens, 3), big=round(cov, 3), kind=kind, y=round(li.y_mid, 1)))
    return out


MP = dict(
    MARK_BIG=0.8,      # a detected line whose extent is at least this share covered by L1's oversized components ...
    MARK_H_LO=0.75,    # ... and whose run height is outside [MARK_H_LO, MARK_H_HI] x the page's typical
    MARK_H_HI=1.3,     # text-line run height is a non-document mark line: no text is ever paired with it
)


def mark_lines(ev, inks, kinds):
    """Detected lines that look like stamps, signatures or logos (run time,
    no truth). Set on synthetic dev pages: blocks 18 of 22 non-text lines and
    1 of 323 text lines."""
    st = det_line_stats(ev, inks, kinds)
    ref = [x["h_rel"] for x in st if x["big"] < 0.5 and x["h_rel"] > 0]
    med = float(np.median(ref)) if ref else 1.0
    return {j for j, x in enumerate(st) if x["big"] >= MP["MARK_BIG"]
            and (x["h_rel"] < MP["MARK_H_LO"] * med or x["h_rel"] > MP["MARK_H_HI"] * med)}, st


# D15 follow-up, second rule (--la TZ). TM above was set on synthetic dev and, run once on the
# real development pages, blocked nothing (hadpi p2's Sonnet breach pairs a 7-letter line with a
# detected line below the text block, 2.3x the page's glyph height, no oversized component).
# TZ adds the zone below the text block. Rule (third version, all on synthetic dev: the first also
# marked every line with oversized coverage >= 0.8 and ended the block at the first break in a chain
# of ordinary lines; the second required oversized coverage < 0.35 for a line to be ordinary; both
# blocked text lines under stamps, s201 and s202, and were dropped):
#   normal line: L1 kind text, run height within [MARK_H_LO, MARK_H_HI] x the page's typical,
#     ink density within [MZ_D_LO, MZ_D_HI] x typical (oversized coverage ignored: text under a
#     stamp keeps its glyph statistics);
#   text block bottom: the lowest normal line with another normal line at most MZ_GAP x the
#     page's line spacing above it (one isolated normal-looking mark does not extend the block);
#   mark line: TM's mark line (mark_lines above), or more than MZ_BELOW x spacing below the block
#     bottom and not normal. The zone above the block is left alone (running heads are text);
#   exemption at pairing time: a trusted pairing with a mark line survives only when the
#     template gain per letter is >= MZ_EXEMPT x the page's median over pairings and the line's
#     height and density are ordinary (density stands in for stroke width: L1 has no per-line
#     stroke measurement).
MZ = dict(MZ_D_LO=0.6, MZ_D_HI=1.6, MZ_GAP=1.8, MZ_BELOW=0.5, MZ_EXEMPT=0.9,
          MZ_ZH_LO=0.5, MZ_ZH_HI=1.6)
# --la TZ2 (after TZ was run once on untouched synthetic seeds 401-402: it blocked the last line of a
# clean book_naskh page, s401, whose run height was 0.74 x the page's typical; naskh lines without
# tall letters vary that much). TZ2 is TZ with a wider height band for the zone below the block
# only: a line there is a mark when its height is outside [MZ_ZH_LO, MZ_ZH_HI] x typical or its
# density outside [MZ_D_LO, MZ_D_HI] x typical. Checked on fresh seeds 601-602, never on 401-402 again.


def zone_mark_lines(ev, inks, kinds, wide=False):
    """-> (mark set, normal set (height and density ordinary, whatever the oversized coverage),
    block bottom y or None, stats). Run time, no truth."""
    st = det_line_stats(ev, inks, kinds)
    ref = [x for x in st if x["kind"] == "text" and x["big"] < 0.5 and x["h_rel"] > 0]
    mh = float(np.median([x["h_rel"] for x in ref])) if ref else 1.0
    md = float(np.median([x["dens"] for x in ref])) if ref else 1.0
    normal = {j for j, x in enumerate(st) if x["kind"] == "text"
              and MP["MARK_H_LO"] * mh <= x["h_rel"] <= MP["MARK_H_HI"] * mh
              and MZ["MZ_D_LO"] * md <= x["dens"] <= MZ["MZ_D_HI"] * md}
    S = float(ev.S)
    ys = sorted(st[j]["y"] for j in normal)
    yb = None
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] <= MZ["MZ_GAP"] * S:
            yb = ys[k]
    marks, _ = mark_lines(ev, inks, kinds)
    marks = set(marks)
    if wide:
        zone_ok = {j for j, x in enumerate(st) if x["kind"] == "text"
                   and MZ["MZ_ZH_LO"] * mh <= x["h_rel"] <= MZ["MZ_ZH_HI"] * mh
                   and MZ["MZ_D_LO"] * md <= x["dens"] <= MZ["MZ_D_HI"] * md}
    else:
        zone_ok = normal
    for j, x in enumerate(st):
        if yb is not None and x["y"] > yb + MZ["MZ_BELOW"] * S and j not in zone_ok:
            marks.add(j)
    return marks, normal, yb, st


def guard_zone(ev, inks, kinds, texts, la, wide=False):
    """Apply TZ (wide=True: TZ2) to a template assignment in place -> report dict."""
    marks, normal, yb, st = zone_mark_lines(ev, inks, kinds, wide=wide)
    G = la["gains"]
    nlet = [max(1.0, sum(WP.letters(t) for t in toks)) for toks in texts]
    per = [G[i, j] / nlet[i] for i, j in enumerate(la["assign"]) if j is not None and G[i, j] > 0]
    med = float(np.median(per)) if per else np.inf
    blocked, exempt = [], []
    for i, j in enumerate(la["assign"]):
        if j is None or j not in marks or not la["trusted"][i]:
            continue
        if j in normal and G[i, j] / nlet[i] >= MZ["MZ_EXEMPT"] * med:
            exempt.append(i)
            continue
        la["trusted"][i] = False
        la["margin"][i] = 0.0
        blocked.append(i)
    return dict(mark_lines=sorted(marks), block_bottom=yb, blocked=blocked, exempt=exempt)
