"""Strip-first restructuring of a text-audit draft (used by tests/text_audit.py).

The draft builder takes its line structure from the readers' own line breaks, so where two
readers broke a printed line differently the draft carries fragments (``⟦text|∅؟⟧`` lines),
disputed-break marks and whole-line alternatives. This module rebuilds such places from the
printed strips the line finder found, so that the owner is offered one printed line at a time:

* ``refine``: a marker whose alternatives are several words long is aligned word by word and
  replaced by plain words where every reader agrees and small markers where they differ.
* ``consolidate``: runs of draft lines whose structure is in doubt (a disputed break, no strip,
  or a line that may be empty) are flattened and cut again to fit the strips between the
  neighbouring trusted lines (typewritten text: ink width is proportional to character count).

Nothing is re-read and no alternative wording is invented: every alternative of the draft can
still be composed from the small markers (the one exception: a lone ``…`` piece is offered as
"could not be read"). Pure Python, no I/O.
"""
from __future__ import annotations

import math
import re
import statistics
from difflib import SequenceMatcher

GAP_ABSENT, GAP_UNREAD = "∅", "…"
BREAK_PENALTY = 0.3       # cutting where the draft had no line end, or keeping a draft line end uncut
SIMILAR = 0.6

_DROP = re.compile("[ ‌‎‏ـً-ْٰء]")   # space, ZWNJ, marks, kashida, diacritics, hamza
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ي": "ی", "ك": "ک",
                       "ۀ": "ه", "ة": "ه", "ؤ": "و", "ئ": "ی"})


def norm(s: str) -> str:
    return _DROP.sub("", s).translate(_FOLD)


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# tokens: a line as space-separated words and markers
# ---------------------------------------------------------------------------
def is_mark(t) -> bool:
    return isinstance(t, dict) and t["t"] == "mark"


def is_brk(t) -> bool:
    return isinstance(t, dict) and t["t"] == "brk"


def tokens_of(segs: list):
    """Segments -> tokens (words as str, markers as dicts), or None when the line is not made
    of single-space-separated words and markers (then it is left exactly as it is)."""
    segs = list(segs)
    if segs and segs[-1]["t"] == "text" and segs[-1]["v"] == " " and len(segs) > 1:
        segs = segs[:-1]                      # "⟦↵؟⟧ " : a break mark and nothing else
    toks, last = [], len(segs) - 1
    for k, s in enumerate(segs):
        if s["t"] != "text":
            toks.append(s)
            continue
        parts = s["v"].split(" ")
        if k > 0:
            if parts[0] != "":
                return None
            parts = parts[1:]
        if k < last:
            if not parts or parts[-1] != "":
                return None
            parts = parts[:-1]
        if any(p == "" for p in parts):
            return None
        toks.extend(parts)
    return toks


def segs_of(tokens: list) -> list:
    out = []
    for k, t in enumerate(tokens):
        if k:
            out.append({"t": "text", "v": " "})
        out.append({"t": "text", "v": t} if isinstance(t, str) else t)
    merged = []
    for s in out:
        if s["t"] == "text" and merged and merged[-1]["t"] == "text":
            merged[-1] = {"t": "text", "v": merged[-1]["v"] + s["v"]}
        else:
            merged.append(dict(s) if s["t"] == "text" else s)
    return merged


def real_alts(m: dict) -> list:
    return [a for a in m["alts"] if a not in (GAP_ABSENT, "")]


def has_gap(t) -> bool:
    return is_mark(t) and any(a in (GAP_ABSENT, "") for a in t["alts"])


def simple_optional(t) -> bool:
    return has_gap(t) and len(real_alts(t)) == 1 and real_alts(t)[0] != GAP_UNREAD


def tok_len(t) -> float:
    if isinstance(t, str):
        return float(len(t))
    if not is_mark(t):
        return 0.0
    if not t["alts"]:
        return 4.0
    real = real_alts(t)                    # "nothing here" is not a width: the words are on the page for some reader
    return sum(len(a) for a in real) / len(real) if real else 0.0


def span_len(tokens: list) -> float:
    return sum(tok_len(t) for t in tokens) + max(0, len(tokens) - 1)


# ---------------------------------------------------------------------------
# word-level alignment of a marker's alternatives
# ---------------------------------------------------------------------------
_GAP_COST = 0.55
_MOVES = [(k, l) for k in (1, 2, 3) for l in (1, 2, 3)]


def _pair(a: list, b: list) -> list:
    """Monotone pairing of two word lists: [(a indices, b indices)], joins of up to 3 words allowed
    when the joined letters agree."""
    n, m = len(a), len(b)
    na, nb = [norm(x) for x in a], [norm(x) for x in b]
    INF = 1e18
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    P = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i][j]
            if d >= INF:
                continue
            cands = []
            for k, l in _MOVES:
                if i + k > n or j + l > m:
                    continue
                if k == 1 and l == 1:
                    if a[i] == b[j]:
                        c = 0.0
                    else:                      # two readings of one word; unlike words pair only as a last resort
                        r = sim(na[i], nb[j])
                        c = 0.02 + 0.98 * (1 - r) if r >= SIMILAR else 1.0
                else:
                    if "".join(na[i:i + k]) != "".join(nb[j:j + l]):
                        continue               # a join question only: the same letters, spaced differently
                    c = 0.03 * (k + l)         # the smallest join that explains the difference
                cands.append((i + k, j + l, c))
            if i < n:
                cands.append((i + 1, j, _GAP_COST))
            if j < m:
                cands.append((i, j + 1, _GAP_COST))
            for ni, nj, c in cands:
                if d + c < D[ni][nj] - 1e-12:
                    D[ni][nj], P[ni][nj] = d + c, (i, j)
    out, i, j = [], n, m
    while i or j:
        pi, pj = P[i][j]
        out.append((list(range(pi, i)), list(range(pj, j))))
        i, j = pi, pj
    out.reverse()
    return out


def align_alts(seqs: list) -> list:
    """Column blocks for several word lists: [[piece of seq 0, piece of seq 1, ...], ...] in
    reading order, each piece a string of that reading's words for the block ('' = none)."""
    pv = max(range(len(seqs)), key=lambda k: (len(seqs[k]), -k))
    P = seqs[pv]
    n = len(P)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = {}
    for k, s in enumerate(seqs):
        if k == pv:
            continue
        pairs[k] = _pair(P, s)
        for ia, _ in pairs[k]:
            for x in ia[1:]:
                parent[find(x)] = find(ia[0])
    groups: dict = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    blocks = sorted(groups.values(), key=lambda g: g[0])
    for g in blocks:                       # a block must be contiguous: fill holes
        if g != list(range(g[0], g[-1] + 1)):
            return [[" ".join(s) for s in seqs]]
    block_of = {x: bi for bi, g in enumerate(blocks) for x in g}
    cols = []                              # (sort key, {seq index: [words]})
    main = [dict() for _ in blocks]
    ins = [dict() for _ in range(len(blocks) + 1)]
    for bi, g in enumerate(blocks):
        main[bi][pv] = [P[x] for x in g]
    for k, segs in pairs.items():
        s = seqs[k]
        for ia, ib in segs:
            if ia:
                main[block_of[ia[0]]].setdefault(k, []).extend(s[x] for x in ib)
            else:
                nxt = len(blocks)
                # the slot is before the next block that this pairing reaches
                for ia2, _ in segs[segs.index((ia, ib)) + 1:]:
                    if ia2:
                        nxt = block_of[ia2[0]]
                        break
                ins[nxt].setdefault(k, []).extend(s[x] for x in ib)
    for bi in range(len(blocks) + 1):
        if ins[bi]:
            cols.append(ins[bi])
        if bi < len(blocks):
            cols.append(main[bi])
    return [[" ".join(c.get(k, [])) for k in range(len(seqs))] for c in cols]


def _pattern(pieces: list) -> tuple:
    first = {}
    return tuple(first.setdefault(p, i) for i, p in enumerate(pieces))


def refine(mark: dict, new_id) -> tuple:
    """A marker -> (tokens, mapping, counts). ``mapping`` is {alternative index: [(new marker id,
    alternative index there), ...]} or None when the marker is left as it is; ``counts[i][k]`` is
    how many words of alternative k have been used up to and including output token i."""
    alts = mark["alts"]
    if mark.get("choice") is not None or len(alts) < 2 or has_gap(mark) or GAP_UNREAD in alts:
        return [mark], None, None
    seqs = [a.split(" ") for a in alts]
    if any("" in s for s in seqs) or all(len(s) == 1 for s in seqs):
        return [mark], None, None
    cols = align_alts(seqs)
    merged = []                            # adjacent differing columns with the same agreement pattern join
    for pieces in cols:
        same = len(set(pieces)) == 1
        if merged and not same and not merged[-1][1] and _pattern(merged[-1][0]) == _pattern(pieces):
            prev = merged[-1][0]
            merged[-1] = ([" ".join(x for x in (p, q) if x) for p, q in zip(prev, pieces)], False)
        else:
            merged.append((list(pieces), same))
    # a one-word scrap that only some readers have ("…", a stray bracket) joins a differing neighbour
    k = 0
    while k < len(merged):
        pieces, same = merged[k]
        scrap = not same and "" in pieces and max(len(p.split(" ")) for p in pieces if p) == 1
        nb = k - 1 if k and not merged[k - 1][1] else (k + 1 if k + 1 < len(merged) and not merged[k + 1][1] else None)
        if scrap and nb is not None:
            lo, hi = min(k, nb), max(k, nb)
            merged[lo:hi + 1] = [([" ".join(x for x in (p, q) if x) for p, q in zip(merged[lo][0], merged[hi][0])], False)]
            k = lo
            continue
        k += 1
    if len(merged) == 1:
        return [mark], None, None
    # the pieces of each alternative must compose back to that alternative, exactly
    for k, a in enumerate(alts):
        if " ".join(p[k] for p, _ in merged if p[k]) != a:
            return [mark], None, None
    out, mapping, counts = [], {k: [] for k in range(len(alts))}, []
    run = [0] * len(alts)
    for pieces, same in merged:
        if same:
            for w in pieces[0].split(" "):
                out.append(w)
                run = [x + 1 for x in run]
                counts.append(tuple(run))
            continue
        run = [x + (len(p.split(" ")) if p else 0) for x, p in zip(run, pieces)]
        counts.append(tuple(run))
        distinct = sorted(set(pieces), key=lambda p: (p == "", p))
        sub = {"t": "mark", "id": new_id(), "alts": [p if p else GAP_ABSENT for p in distinct], "choice": None}
        out.append(sub)
        for k, p in enumerate(pieces):
            mapping[k].append((sub["id"], distinct.index(p)))
    return out, mapping, counts


def refine_tokens(tokens: list, new_id, mappings: dict | None = None) -> list:
    out = []
    for t in tokens:
        if is_mark(t):
            toks, mp, _ = refine(t, new_id)
            if mp is not None and mappings is not None:
                mappings[t["id"]] = mp
            out.extend(toks)
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# zones: draft lines whose structure is in doubt
# ---------------------------------------------------------------------------
def may_be_empty(tokens: list) -> bool:
    return all(has_gap(t) or is_brk(t) for t in tokens)


def _absorb_fragments(tokens: list, ends: list) -> tuple:
    """``[frag|∅] [frag|∅] [tail|whole|whole|∅]`` -> one marker of whole-line readings: a reader
    that broke the printed line into pieces is put back together beside the readers that did not."""
    toks, ends = list(tokens), list(ends)
    k = 0
    while k < len(toks):
        if not simple_optional(toks[k]):
            k += 1
            continue
        j = k
        while j < len(toks) and simple_optional(toks[j]):
            j += 1
        if j < len(toks) and has_gap(toks[j]) and len(real_alts(toks[j])) >= 2:
            frag = " ".join(real_alts(t)[0] for t in toks[k:j])
            nf = norm(frag)
            ns, c = [], 0
            for t in toks[k:j]:
                c += len(real_alts(t)[0].split(" "))
                ns.append(c)
            alts, kept, cuts = [], 0, []
            for a in real_alts(toks[j]):
                na = norm(a)
                if len(na) >= len(nf) and sim(na[:len(nf)], nf) >= SIMILAR:
                    alts.append(a)
                    kept += 1
                else:
                    alts.append(frag + " " + a)
                    cuts.append((alts[-1], ns))
            if kept:
                m = dict(toks[j], alts=sorted(set(alts)), choice=None, _cuts=cuts)
                toks[k:j + 1] = [m]
                ends[k:j + 1] = [ends[j]]
                k += 1
                continue
        k = j
    return toks, ends


def _merge_optionals(tokens: list, ends: list, new_id) -> tuple:
    """``[X|∅] [Y|∅]`` with X and Y alike (the same words put on different lines by two readers)
    -> ``[X|Y]``."""
    toks, ends = list(tokens), list(ends)
    k = 0
    while k + 1 < len(toks):
        a, b = toks[k], toks[k + 1]
        if simple_optional(a) and simple_optional(b):
            x, y = real_alts(a)[0], real_alts(b)[0]
            if sim(norm(x), norm(y)) >= SIMILAR:
                if x == y:
                    new = x.split(" ")
                else:
                    new = [{"t": "mark", "id": new_id(), "alts": sorted({x, y}), "choice": None}]
                if k:
                    ends[k - 1] = True          # the words may belong to either neighbouring line
                toks[k:k + 2] = new
                ends[k:k + 2] = [False] * (len(new) - 1) + [True]
                k += len(new)
                continue
        k += 1
    return toks, ends


def _refine_keep_ends(tokens: list, ends: list, new_id, mappings=None) -> tuple:
    """Refine every marker. A marker that may be absent and has several readings loses its "absent"
    alternative first: inside a zone it only said that one reader put these words on another line."""
    out_t, out_e = [], []
    for t, e in zip(tokens, ends):
        if is_mark(t):
            if has_gap(t) and t.get("choice") is None and len(real_alts(t)) >= 2 and GAP_UNREAD not in t["alts"]:
                t = dict(t, alts=real_alts(t))
            cuts = t.get("_cuts") or []
            if cuts:
                t = {k: v for k, v in t.items() if k != "_cuts"}
            toks, mp, counts = refine(t, new_id)
            if mp is not None and mappings is not None:
                mappings[t["id"]] = mp
            flags = [False] * (len(toks) - 1) + [e]
            if counts:                      # a reader's own line ends inside an absorbed reading
                for alt, ns in cuts:
                    k = t["alts"].index(alt)
                    for x in ns:
                        hit = [q for q, c in enumerate(counts) if c[k] == x]
                        if hit:
                            flags[hit[-1]] = True
            out_t.extend(toks)
            out_e.extend(flags)
            continue
        out_t.append(t)
        out_e.append(e)
    return out_t, out_e


def _segment(tokens: list, ends: list, strips: list, ratio: float, full: float):
    """Cut tokens into consecutive groups, each on its own strip (in order; strips may be left
    out at a cost). Returns [(strip index, [tokens])] or None when it cannot be done."""
    n, m = len(tokens), len(strips)
    if not n or not m:
        return None
    lens = [tok_len(t) for t in tokens]
    cum = [0.0]
    for x in lens:
        cum.append(cum[-1] + x)
    nend = [0]
    for e in ends:
        nend.append(nend[-1] + (1 if e else 0))
    skip = [2.0 if w >= 0.5 * full else 0.8 for w in strips]
    INF = 1e18
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    P = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for j in range(m):
        for i in range(n + 1):
            d = D[i][j]
            if d >= INF:
                continue
            if d + skip[j] < D[i][j + 1]:
                D[i][j + 1], P[i][j + 1] = d + skip[j], (i, j, False)
            for i2 in range(i + 1, n + 1):
                chars = cum[i2] - cum[i] + (i2 - i - 1)
                c = min(1.5, abs(math.log(max(1.0, strips[j]) / max(1.0, ratio * chars))))
                inside = nend[i2 - 1] - nend[i]                 # draft line ends kept uncut
                c += BREAK_PENALTY * inside
                if i2 < n and not ends[i2 - 1]:
                    c += BREAK_PENALTY
                if d + c < D[i2][j + 1]:
                    D[i2][j + 1], P[i2][j + 1] = d + c, (i, j, True)
    if D[n][m] >= INF:
        return None
    out, i, j = [], n, m
    while j:
        pi, pj, used = P[i][j]
        if used:
            out.append((pj, tokens[pi:i]))
        i, j = pi, pj
    out.reverse()
    return out


def consolidate(lines: list, crops: list, new_id) -> tuple:
    """``lines``: [{"segs", "crop", "blank"}] in draft order (crop = strip id or None).
    Returns (new lines [{"segs", "crop", "src": [draft line indices], "reflowed": bool}],
    mappings {old marker id: refine mapping}, report)."""
    order = {c["id"]: k for k, c in enumerate(crops)}
    width = [max(1.0, float(c.get("ink_w") or (c["x1"] - c["x0"]))) for c in crops]
    toks = [None if l.get("blank") else tokens_of(l["segs"]) for l in lines]
    mappings: dict = {}
    report = dict(zones=[], refined_marks=0)

    def doubt(i):
        t = toks[i]
        if t is None:
            return False
        return any(is_brk(x) for x in t) or lines[i]["crop"] is None or may_be_empty(t)
    trusted = [(span_len(toks[i]), width[order[lines[i]["crop"]]]) for i in range(len(lines))
               if toks[i] and not doubt(i) and lines[i]["crop"] in order]
    ratio = statistics.median(w / max(1.0, c) for c, w in trusted) if trusted else None
    full = statistics.median(sorted(width)[len(width) // 2:]) if width else 1.0
    out, i, n = [], 0, len(lines)
    while i < n:
        if toks[i] is None or not doubt(i):
            segs = lines[i]["segs"]
            if toks[i] is not None:
                segs = segs_of(refine_tokens(toks[i], new_id, mappings))
            out.append(dict(segs=segs, crop=lines[i]["crop"], src=[i], reflowed=False))
            i += 1
            continue
        j = i
        while j < n and toks[j] is not None and doubt(j):
            j += 1
        zone = list(range(i, j))
        prev = next((order[lines[k]["crop"]] for k in range(i - 1, -1, -1) if lines[k]["crop"] in order), None)
        nxt = next((order[lines[k]["crop"]] for k in range(j, n) if lines[k]["crop"] in order), None)
        own = [order[lines[k]["crop"]] for k in zone if lines[k]["crop"] in order]
        lo = prev + 1 if prev is not None else 0
        hi = nxt if nxt is not None else (max(own) + 1 if own else lo)
        strips = list(range(lo, hi))
        flat, ends = [], []
        for k in zone:
            t = [x for x in toks[k] if not is_brk(x)]
            flat.extend(t)
            ends.extend([False] * (len(t) - 1) + [True] * (1 if t else 0))
        flat, ends = _absorb_fragments(flat, ends)
        flat, ends = _refine_keep_ends(flat, ends, new_id)
        flat, ends = _merge_optionals(flat, ends, new_id)
        flat, ends = _refine_keep_ends(flat, ends, new_id)
        groups = _segment(flat, ends, [width[s] for s in strips], ratio, full) if (ratio and strips) else None
        if groups is None:
            # no strip to go by: keep the draft's lines, only refine their markers
            for k in zone:
                segs = lines[k]["segs"]
                if not any(is_brk(x) for x in toks[k]):
                    segs = segs_of(refine_tokens(toks[k], new_id, mappings))
                out.append(dict(segs=segs, crop=lines[k]["crop"], src=[k], reflowed=False))
            report["zones"].append(dict(lines=[k + 1 for k in zone], strips=len(strips), result="kept"))
        else:
            for sj, g in groups:
                out.append(dict(segs=segs_of(g), crop=crops[strips[sj]]["id"], src=zone, reflowed=True))
            report["zones"].append(dict(lines=[k + 1 for k in zone], strips=len(strips), result=f"{len(groups)} lines"))
        i = j
    report["refined_marks"] = len(mappings)
    return out, mappings, report
