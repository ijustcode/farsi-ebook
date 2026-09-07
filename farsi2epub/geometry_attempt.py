"""Best-attempt placement on actual ink regions, separate from verified evidence.

Every detected region has geometry before recognition. Text-to-region inference
may be approximate; it never promotes a word to independently supported status.
"""
from bisect import bisect_right
from dataclasses import replace
from difflib import SequenceMatcher
import re

import fitz
import numpy as np

ATTEMPT_VERSION = 1

from . import locate
from .page_map import PrintedWord, PlacementResult, place, target_span


def _union(rects):
    return list(locate._union_rects([fitz.Rect(r) for r in rects]))


def _tokens(text):
    return [(m.start(), m.end(), locate._fold_word(m.group()))
            for m in re.finditer(r'\S+', text) if locate._fold_word(m.group())]


def _partition(tokens, regions, anchors=()):
    """Monotone allocation to whole ink regions; no invented rectangle edges.

    Character mass supplies a prior when recognition cannot identify a boundary.
    Verified single-word geometry supplies local anchors that override that prior.
    Shared/fused regions may legitimately support several approximate tokens.
    """
    if not tokens or not regions:
        return []
    widths = [max(1e-8, r['rect'][2]-r['rect'][0]) for r in regions]
    ink = np.cumsum([0., *widths])
    chars = np.cumsum([0., *[max(1, len(t)) for t in tokens]])
    points = [(0., 0.), (float(chars[-1]), float(ink[-1]))]
    for index, word in anchors:
        hits = [j for j,r in enumerate(regions)
                if (fitz.Rect(r['rect']) & fitz.Rect(word.rect)).get_area() > 1e-10]
        if hits:
            points += [(float(chars[index]), float(ink[min(hits)])),
                       (float(chars[index+1]), float(ink[max(hits)+1]))]
    monotone = [(0., 0.)]
    for x,y in sorted(set(points)):
        if x > monotone[-1][0] and y >= monotone[-1][1]:
            monotone.append((x,y))
    if monotone[-1][0] < chars[-1]:
        monotone.append((float(chars[-1]),float(ink[-1])))
    bounds = np.interp(chars, [p[0] for p in monotone], [p[1] for p in monotone])
    out = []
    for a,b in zip(bounds,bounds[1:]):
        hits = [j for j in range(len(regions)) if ink[j] < b-1e-10 and ink[j+1] > a+1e-10]
        if not hits:
            hits = [min(len(regions)-1, max(0, bisect_right(ink, (a+b)/2)-1))]
        # Include one adjacent ink fragment at either edge: estimated text
        # boundaries should favor target coverage over clipping a detached part.
        hits = list(range(max(0,min(hits)-1),min(len(regions),max(hits)+2)))
        out.append((_union([regions[j]['rect'] for j in hits]), [regions[j]['id'] for j in hits]))
    return out


def complete_geometry(words, regions):
    """Give every recognized token geometry while preserving all verified words."""
    result = []
    for line in sorted({r['line'] for r in regions} | {w.line for w in words}):
        row = [w for w in words if w.line == line]
        ink = [r for r in regions if r['line'] == line]
        if not row:
            row = [PrintedWord('�', r['rect'], line, [r['id']], False) for r in ink]
        # A shared supported group cannot anchor individual internal boundaries.
        anchors = [(i,w) for i,w in enumerate(row) if w.supported and
                   sum(v.rect == w.rect for v in row) == 1]
        assigned = _partition([locate._fold_word(w.text) or '�' for w in row], ink, anchors)
        for i,w in enumerate(row):
            if w.supported or not assigned:
                result.append(w)
            else:
                rect, refs = assigned[i]
                result.append(replace(w, rect=rect, evidence=list(dict.fromkeys([e for e in w.evidence if not e.startswith('ink:')]+refs)), group='inferred:' + ':'.join(refs)))
    return result


class PageProjection:
    """Correction-independent alignment from original Markdown offsets to ink."""
    def __init__(self, md, words, regions):
        self.tokens = _tokens(md)
        self.words = words
        self.regions = regions
        self.projected = []
        if not self.tokens or not regions:
            return
        known = [w for w in words if locate._fold_word(w.text)]
        char_width = sum(w.rect[2]-w.rect[0] for w in known)/max(1,sum(len(locate._fold_word(w.text)) for w in known))
        stream = []
        for line in sorted({r['line'] for r in regions}):
            row = [w for w in words if w.line == line and locate._fold_word(w.text)]
            if row:
                stream.extend(row)
            else:
                stream.extend(PrintedWord('�'*max(1, round((r['rect'][2]-r['rect'][0])/max(char_width,.003))),
                                          r['rect'], line, [r['id']], False) for r in regions if r['line']==line)
        if not stream:
            return
        reading = [locate._fold_word(w.text) or w.text for w in stream]
        text = ''.join(t[2] for t in self.tokens)
        printed = ''.join(reading)
        a = np.cumsum([0,*[len(t[2]) for t in self.tokens]])
        b = np.cumsum([0,*[len(t) for t in reading]])
        # Long unique agreement anchors survive errors and unread lines locally.
        points = [(0,0),(len(text),len(printed))]
        for block in SequenceMatcher(None,text,printed,autojunk=False).get_matching_blocks():
            snippet = text[block.a:block.a+block.size]
            if block.size >= 10 and text.count(snippet)==1 and printed.count(snippet)==1:
                points.extend([(block.a,block.b),(block.a+block.size,block.b+block.size)])
        points = sorted(set(points))
        monotone = [points[0]]
        for x,y in points[1:]:
            if x>monotone[-1][0] and y>=monotone[-1][1]:
                monotone.append((x,y))
        positions = np.interp((a[:-1]+a[1:])/2,[p[0] for p in monotone],[p[1] for p in monotone])
        for token,pos in zip(self.tokens,positions):
            j = min(len(stream)-1,max(0,bisect_right(b,pos)-1))
            self.projected.append(replace(stream[j],text=token[2],supported=False))
        # Within each inferred line, use all physical groups and retained exact
        # identity anchors. This avoids mapping an unread line to one giant box.
        for line in sorted({w.line for w in self.projected}):
            indices = [i for i,w in enumerate(self.projected) if w.line==line]
            ink = [r for r in regions if r['line']==line]
            row = [self.projected[i] for i in indices]
            anchors = []
            for j,w in enumerate(row):
                exact = [v for v in words if v.line==line and v.supported and locate._fold_word(v.text)==w.text]
                if len(exact)==1 and sum(v.text==w.text for v in row)==1:
                    anchors.append((j,exact[0]))
            for i,(rect,refs) in zip(indices,_partition([w.text for w in row],ink,anchors)):
                self.projected[i] = replace(self.projected[i],rect=rect,evidence=refs)

    def span(self, span):
        if not self.projected:
            return []
        lo,hi = span
        if lo==hi:
            after = next((i for i,t in enumerate(self.tokens) if t[0]>=lo),len(self.tokens))
            return self.projected[max(0,after-1):min(len(self.tokens),after+1)]
        selected = [w for t,w in zip(self.tokens,self.projected) if t[0]<hi and t[1]>lo]
        if not selected:
            # Standalone punctuation attaches to its preceding printed word.
            previous = [i for i,t in enumerate(self.tokens) if t[1]<=lo]
            selected = [self.projected[previous[-1] if previous else 0]]
        return selected


def _estimated(selected, reason, *, insertion=False, alternatives=0, context=None, target_range=None):
    if not selected:
        return PlacementResult('unresolved',reason='no_detected_geometry')
    by_line = {}
    for word in selected:
        r=fitz.Rect(word.rect)
        if not r.is_empty:
            by_line.setdefault(word.line,[]).append(word.rect)
    if not by_line:
        return PlacementResult('unresolved',reason='no_detected_geometry')
    rects = [_union(by_line[line]) for line in sorted(by_line)]
    kind = 'words'
    if insertion:
        # Boundary marker, including page-start/end and wrapped insertion sites.
        kind = 'insertion_boundary'
        if len(selected)==2 and selected[0].line==selected[1].line:
            x=(selected[0].rect[0]+selected[1].rect[2])/2
            bounds=[(x,min(w.rect[1] for w in selected),max(w.rect[3] for w in selected))]
        else:
            bounds=[(w.rect[2] if insertion=='start' or j>0 else w.rect[0], w.rect[1],w.rect[3]) for j,w in enumerate(selected)]
        rects = [[max(0,x-.001),y0,min(1,x+.001),y1] for x,y0,y1 in bounds]
    def boxdict(r): return dict(zip(('x0','y0','x1','y1'),r))
    before = after = 0
    if context is not None and target_range is not None:
        lo,hi = target_range
        extras = [i for i,w in enumerate(context) if not lo<=i<hi and any((fitz.Rect(w.rect)&fitz.Rect(r)).get_area()>1e-10 for r in rects)]
        before,after = sum(i<lo for i in extras),sum(i>=hi for i in extras)
    box = {**boxdict(_union(rects)), 'source':'scan', 'status':'estimated',
           'segments':[boxdict(r) for r in rects], 'kind':kind,
           'buffer_before':before,'buffer_after':after,'alternatives':alternatives}
    return PlacementResult('estimated',box,reason=reason,
                           evidence=sorted({e for w in selected for e in w.evidence}),kind=kind,
                           buffer_before=before,buffer_after=after)


def best_attempt(md, q, words, projection, strict=None):
    """Preserve supported placements, then prefer a useful labeled approximation."""
    strict = strict or place(md,q,words)
    if strict.box is not None:
        return strict
    reader = [locate._fold_word(w.text) for w in words]
    span = target_span(md,q,reader)
    if span:
        return _estimated(words[span[0]:span[1]],strict.reason,context=words,target_range=span)
    if q.span is not None and 0<=q.span[0]<=q.span[1]<=len(md):
        result = _estimated(projection.span(q.span),'aligned to page text; ' + strict.reason,
                            insertion=('start' if q.span[0]==0 else 'end' if q.span[0]>=len(md) else 'between') if q.kind=='insertion' and q.span[0]==q.span[1] else False)
        if result.box:
            return result
    # Finding-only text has no authoritative Markdown position. Rank occurrences
    # explicitly; tied occurrences remain visible as a best guess, never verified.
    query = locate._norm_words(q.text)
    pool = projection.projected or [w for w in words if locate._fold_word(w.text)]
    if query and pool:
        target = ''.join(query)
        candidates = []
        for i in range(len(pool)):
            for size in range(max(1,len(query)-2),min(len(pool)-i,len(query)+3)+1):
                text = ''.join(locate._fold_word(w.text) for w in pool[i:i+size])
                score = SequenceMatcher(None,target,text,autojunk=False).ratio()
                candidates.append((score,i,size))
        candidates.sort(key=lambda c:(-c[0],abs(c[2]-len(query)),c[1]))
        score,i,size = candidates[0]
        if score >= .45:
            alternatives = sum(abs(s-score)<.025 and (j+length<=i or j>=i+size) for s,j,length in candidates[1:])
            reason = 'multiple possible occurrences; best guess' if alternatives else 'closest text match; best guess'
            return _estimated(pool[i:i+size],reason,alternatives=alternatives)
    if any(t in q.text for t in ('پاورقی','پایان صفحه','یادداشت')) and projection.regions:
        last = max(r['line'] for r in projection.regions)
        selected = [PrintedWord('',r['rect'],last,[r['id']],False) for r in projection.regions if r['line']==last]
        return _estimated(selected,'finding describes a footnote; approximate bottom-line region')
    return strict
