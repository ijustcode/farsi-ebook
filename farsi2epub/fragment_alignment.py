"""Opt-in experiment: approximate crop-text to line-text geometry alignment.

Production does not import this module. All inferred words remain unsupported.
The detector cannot supply strings: observations must be independently read
physical crops, never text synthesized from the target or its proposed box.
"""
from dataclasses import dataclass, replace
from math import inf

import fitz

from . import locate

EXPERIMENT_VERSION = 1
MIN_CHARS = 4
MAX_EDIT_RATE = .25
MIN_MARGIN = .12
MIN_WORD_COVERAGE = .75


@dataclass(frozen=True)
class Alignment:
    start: int
    end: int
    distance: int
    edit_rate: float
    margin: float
    matched: tuple[int, ...]


def normalize(text):
    return ''.join(locate._norm_words(text))


def closest_substring(fragment, line):
    """Semiglobal unit-cost Levenshtein, with free line prefix/suffix.

    Return (alignment, reason). Traceback records only line characters paired
    with crop characters; deleted line characters do not count as observed.
    A competing occurrence must be worse by MIN_MARGIN. Nearby endpoints for
    the same occurrence are boundary uncertainty, not independent matches.
    """
    a, b = normalize(fragment), normalize(line)
    if len(a) < MIN_CHARS or not b or '\ufffd' in fragment or '\ufffd' in line:
        return None, 'short_or_unreadable'
    m, n = len(a), len(b)
    # Local crops/lines only: avoid accidental unbounded page-to-page DP.
    if m > 512 or n > 2048:
        return None, 'input_too_long'
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        dp[i][0] = i
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j-1] + (a[i-1] != b[j-1]),
                           dp[i-1][j]+1, dp[i][j-1]+1)

    def trace(end):
        i, j, paired = m, end, []
        while i:
            if j and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
                paired.append(j-1)
                i, j = i-1, j-1
            elif dp[i][j] == dp[i-1][j]+1:
                i -= 1
            else:
                j -= 1
        return j, tuple(reversed(paired))

    # Include poor competitors too, so a margin does not inflate just because
    # a second location narrowly failed the acceptance threshold.
    candidates = []
    for end in range(1, n+1):
        start, paired = trace(end)
        if paired:
            candidates.append((dp[m][end], abs(end-start-m), start, end, paired))
    if not candidates:
        return None, 'no_match'
    candidates.sort()
    distance, _, start, end, paired = candidates[0]
    rate = distance/m
    if rate > MAX_EDIT_RATE:
        return None, 'edit_distance'
    runner = inf
    for cost, _, lo, hi, _ in candidates[1:]:
        overlap = max(0, min(end, hi)-max(start, lo))
        if overlap < .5 * min(end-start, hi-lo):
            runner = min(runner, cost/m)
    margin = min(1., runner-rate)
    if margin < MIN_MARGIN:
        return None, 'ambiguous_occurrence'
    return Alignment(start, end, distance, rate, margin, paired), 'accepted'


def infer_words(words, observations, regions, *, allow_edits=True):
    """Return new words, changed word indices and auditable diagnostics.

    Observations contain text, normalized crop rect, line and evidence key.
    The input words/regions come from the frozen baseline page map. No query is
    accepted here. Conflicting geometry abstains; supported words are immutable.
    """
    result, changed, diagnostics = list(words), set(), []
    for line in sorted({w.line for w in words}):
        indices = [i for i, w in enumerate(words) if w.line == line]
        row = [words[i] for i in indices]
        tokens = [normalize(w.text) for w in row]
        if not all(tokens):
            continue  # anonymous ink is not recognized text
        offsets = [0]
        for token in tokens:
            offsets.append(offsets[-1]+len(token))
        candidates = []
        for ob in (o for o in observations if o['line'] == line):
            alignment, reason = closest_substring(ob['text'], ''.join(tokens))
            entry = {'line': line, 'key': ob['key'], 'text': ob['text'],
                     'rect': ob['rect'], 'reason': reason}
            diagnostics.append(entry)
            if alignment is None:
                continue
            entry.update(start=alignment.start, end=alignment.end,
                         distance=alignment.distance, edit_rate=alignment.edit_rate,
                         margin=alignment.margin, matched=list(alignment.matched))
            if not allow_edits and alignment.distance:
                entry['reason'] = 'exact_only_ablation'
                continue
            # Supported groups constrain physical reading order. Shared group
            # rectangles do not certify internal word boundaries.
            rect = fitz.Rect(ob['rect'])
            for j, w in enumerate(row):
                if not w.supported:
                    continue
                wr = fitz.Rect(w.rect)
                if (rect.x0 >= wr.x1 and alignment.start >= offsets[j+1] or
                        rect.x1 <= wr.x0 and alignment.end <= offsets[j]):
                    entry['reason'] = 'supported_anchor_conflict'
            if entry['reason'] == 'accepted':
                candidates.append((alignment, ob, entry))
        # Disjoint physical crops must preserve RTL text order; overlapping
        # crops cannot claim disjoint text. No arbitrary winner on conflicts.
        rejected = set()
        for i, (a, oa, _) in enumerate(candidates):
            ra = fitz.Rect(oa['rect'])
            for j in range(i+1, len(candidates)):
                b, ob, _ = candidates[j]
                rb = fitz.Rect(ob['rect'])
                gap_tolerance = .002  # independently padded adjacent crops
                conflict = ((ra.x0 >= rb.x1-gap_tolerance and a.end > b.start+1)
                            or (rb.x0 >= ra.x1-gap_tolerance and b.end > a.start+1))
                overlap = (ra & rb).get_area()/max(1e-12, min(ra.get_area(), rb.get_area()))
                if overlap > .5 and (a.end <= b.start or b.end <= a.start):
                    conflict = True
                if conflict:
                    rejected.update((i, j))
        for i in rejected:
            candidates[i][2]['reason'] = 'neighbor_geometry_conflict'
        for j, w in enumerate(row):
            if w.supported:
                continue
            required = set(range(offsets[j], offsets[j+1]))
            hits = [(a, ob) for i, (a, ob, _) in enumerate(candidates)
                    if i not in rejected and required.intersection(a.matched)]
            covered = set().union(*(required.intersection(a.matched) for a, _ in hits))
            if len(covered)/len(required) < MIN_WORD_COVERAGE:
                continue
            # Never infer sub-word x coordinates from character widths. Keep
            # physical crop extents; include adjacent detected pieces where a
            # matched fragment may have omitted a word boundary or detached ink.
            rect = locate._union_rects([fitz.Rect(ob['rect']) for _, ob in hits])
            ink = [r for r in regions if r['line'] == line]
            selected = [k for k, r in enumerate(ink)
                        if (fitz.Rect(r['rect']) & rect).get_area() > 1e-10]
            refs = []
            if selected:
                selected = range(max(0, min(selected)-1), min(len(ink), max(selected)+2))
                rect |= locate._union_rects([fitz.Rect(ink[k]['rect']) for k in selected])
                refs = [ink[k]['id'] for k in selected]
            index = indices[j]
            result[index] = replace(w, rect=list(rect), supported=False,
                                    evidence=sorted({ob['key'] for _, ob in hits} | set(refs)),
                                    group='levenshtein_estimate_v1')
            if result[index].rect != w.rect:
                changed.add(index)
    return result, changed, diagnostics
