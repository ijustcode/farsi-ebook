"""Offline paired experiment; never writes to the source book or calls a VLM.

Run from the repository after activating venv. Outputs contain personal book
excerpts and remain gitignored. --collect freezes inputs once; later runs replay
the frozen inputs without source-book dependencies (except review page images).
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from farsi2epub import fragment_alignment, locate, review
from farsi2epub.fragment_alignment import EXPERIMENT_VERSION, infer_words
from farsi2epub.geometry_attempt import PageProjection, best_attempt
from farsi2epub.page_map import PrintedWord, target_span
from farsi2epub.scan import PlacementService, digest
from farsi2epub.workspace import Workspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def tree_hash(root):
    return digest({str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()})


class Collector(PlacementService):
    """Replay current crop generation into scratch; observations are read-only."""
    def __init__(self, ws, scratch):
        super().__init__(ws, mode='offline', max_cost=0, reader=lambda *a: 1/0)
        self.source_root = self.root
        self.root = scratch
        self.rendered, self.observations = {}, {}

    def _observation(self, key):
        path = self.source_root / 'readings' / f'{key}.json'
        ob = json.loads(path.read_text()) if path.exists() else None
        return ob if self._valid(ob) and ob.get('key') == key else None

    def _render(self, page, rect, factor=1.):
        png = super()._render(page, rect, factor)
        nr = locate._rect_to_fracs(fitz.Rect(rect) & page.rect, page.rect)
        self.rendered.setdefault(self._observation_key(png), []).append(
            [nr[k] for k in ('x0','y0','x1','y1')])
        return png

    def _read(self, pngs, page_no, allow_api):
        assert not allow_api
        observations, keys = super()._read(pngs, page_no, False)
        for ob, key, png in zip(observations, keys, pngs):
            rect = self.rendered[key].pop(0)
            if ob is not None and len(ob['lines']) == 1:
                assert ob['image_sha256'] == hashlib.sha256(png).hexdigest(), 'observation image drift'
                self.observations[digest([key, rect])] = {'key': key, 'text': ob['lines'][0],
                                          'rect': rect,
                                          'observation_sha256': sha(self.source_root/'readings'/f'{key}.json'),
                                          'image_sha256': ob['image_sha256']}
        return observations, keys


def query(row):
    q = dict(row['query'])
    q['span'] = tuple(q['span']) if q['span'] is not None else None
    q['alts'] = tuple(q['alts'])
    return locate.Query(**q)


def collect(args):
    path = args.out/'inputs.json'
    if path.exists():
        raise ValueError('Frozen inputs already exist; choose a new output directory to recollect')
    ws = Workspace(args.slug)
    baseline = json.loads(args.baseline.read_text())
    assert len(baseline['corrections']) == 111
    assert Counter(r['status'] for r in baseline['corrections']) == dict(located=12, buffered=11, estimated=86, unresolved=2)
    fingerprints = {name: tree_hash(ws.root/name) for name in ('text', 'locate_evidence')}
    source_hash = sha(ws.pdf_path)
    collector = Collector(ws, args.out/'scratch')
    pages = []
    try:
        for n in sorted({r['page'] for r in baseline['corrections']}):
            started = time.monotonic()
            rows = [r for r in baseline['corrections'] if r['page'] == n]
            _, issues, hunks, md, _ = review._page_box_inputs(ws, n)
            specs = review._box_specs(issues, hunks, md)
            assert {r['key'] for r in rows} == {s[0] for s in specs}
            current_queries = {s[0]: asdict(s[1]) for s in specs}
            assert all(digest(r['query']) == digest(current_queries[r['key']]) for r in rows)
            assert all(r['md_sha256'] == hashlib.sha256(md.encode()).hexdigest()
                       and r['sidecar_sha256'] == sha(ws.text_dir/f'{n:04d}.json') for r in rows)
            page_key = collector._page_key(n)
            map_path = collector.source_root/'pages'/f'{page_key}.json'
            frozen_map = json.loads(map_path.read_text())
            words = [PrintedWord(**w) for w in frozen_map['words']]
            projection = PageProjection(md, words, frozen_map['regions'])
            for r in rows:
                assert best_attempt(md, query(r), words, projection).box == r['box'], (n, r['key'], 'baseline drift')
            # Previous words preserve existing support and determine the same
            # bounded retry paths; all writes below go to experiment scratch.
            write(collector.root/'pages'/f'{page_key}.json', frozen_map)
            collector.observations.clear()
            collector.rendered.clear()
            collector._build_map(n, '', [], False)
            with fitz.open(ws.pdf_path) as doc:
                page = doc[n-1]
                lines = locate._scan_page_lines(page)
                # Evidence-only compatibility with f21e161's raw crop layout:
                # individual detected pieces (standard/hires) and 2/3-piece
                # windows with +/-1 point line padding. Never import/replay an
                # old locator. A byte-identical rendered image key is required.
                for line in lines:
                    for length in (1, 2, 3):
                        clips = []
                        for start in range(len(line.words)-length+1):
                            rect = locate._union_rects(line.words[start:start+length])
                            rect.y0 = max(page.rect.y0, line.rect.y0-1)
                            rect.y1 = min(page.rect.y1, line.rect.y1+1)
                            clips.append(rect)
                        for factor in ((1., 1.5) if length == 1 else (1.,)):
                            collector._read([collector._render(page, rect, factor) for rect in clips], n, False)
                norm_lines = [locate._rect_to_fracs(line.rect, page.rect) for line in lines]
                recovered = []
                for ob in collector.observations.values():
                    r = ob['rect']
                    hits = [i for i, line in enumerate(norm_lines)
                            if r[1] <= (line['y0']+line['y1'])/2 <= r[3]]
                    if len(hits) != 1:
                        continue
                    line = norm_lines[hits[0]]
                    if r[2]-r[0] >= .92*(line['x1']-line['x0']):
                        continue  # line readings cannot certify their own geometry
                    recovered.append({**ob, 'line': hits[0]})
                image_path = args.out/'images'/f'{n:04d}.png'
                image_path.parent.mkdir(parents=True, exist_ok=True)
                scale = 1568/max(page.rect.width, page.rect.height)
                page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(image_path)
            pages.append({'page': n, 'md': md, 'map': frozen_map, 'map_sha256': sha(map_path),
                          'observations': recovered, 'image_sha256': sha(image_path)})
            print(f'page {n}: {len(recovered)} cached crop readings; baseline reproduced; {time.monotonic()-started:.1f}s', flush=True)
        assert fingerprints == {name: tree_hash(ws.root/name) for name in fingerprints}
        assert source_hash == sha(ws.pdf_path)
        write(path, {'schema_version': 1, 'baseline_commit': 'bc37742d81d70ab0e7e6b55e2d14bfe7605b214c',
                     'baseline_file_sha256': sha(args.baseline), 'baseline': baseline,
                     'source_sha256': source_hash, 'source_trees': fingerprints,
                     'slug': args.slug, 'pages': pages, 'paid_cost': collector.budget.spent})
    finally:
        collector.close()


def geometry(box):
    return None if box is None else (box.get('kind', 'words'), box['segments'])


def area(box):
    return sum((s['x1']-s['x0'])*(s['y1']-s['y0']) for s in box['segments']) if box else 0


def treatment(row, words, inferred, changed, md, projection, span):
    result = {k: row[k] for k in ('status', 'box', 'reason', 'evidence', 'kind', 'buffer_before', 'buffer_after')}
    reason = 'baseline_preserved'
    if row['status'] == 'estimated' and row['kind'] != 'insertion_boundary' and span:
        required = range(*span)
        if any(i in changed for i in required) and all(words[i].supported or inferred[i].group == 'levenshtein_estimate_v1' for i in required):
            result = best_attempt(md, query(row), inferred, projection).to_dict()
            result['reason'] = 'experimental fragment alignment; approximate, not independently verified'
            assert result['status'] == 'estimated' and result['box'] is not None
            reason = 'fragment_evidence'
        else:
            reason = 'insufficient_fragment_coverage'
    elif row['status'] == 'estimated':
        reason = 'target_identity_or_insertion_preserved'
    return result, reason


def evaluate(args):
    started = time.monotonic()
    inputs = json.loads((args.out/'inputs.json').read_text())
    rows, diagnostics, total_words, changed_words = [], [], 0, 0
    for page in inputs['pages']:
        n, md = page['page'], page['md']
        words = [PrintedWord(**w) for w in page['map']['words']]
        regions = page['map']['regions']
        inferred, changed, diag = infer_words(words, page['observations'], regions)
        exact_words, exact_changed, _ = infer_words(words, page['observations'], regions, allow_edits=False)
        total_words += len(words)
        changed_words += len(changed)
        assert all(a == b for a, b in zip(words, inferred) if a.supported)
        assert [w.supported for w in words] == [w.supported for w in inferred]
        diagnostics.extend({'page': n, **d} for d in diag)
        projection = PageProjection(md, inferred, regions)
        exact_projection = PageProjection(md, exact_words, regions)
        for r in (r for r in inputs['baseline']['corrections'] if r['page'] == n):
            span = target_span(md, query(r), [locate._fold_word(w.text) for w in words])
            candidate, reason = treatment(r, words, inferred, changed, md, projection, span)
            exact, _ = treatment(r, words, exact_words, exact_changed, md, exact_projection, span)
            changed_box = geometry(candidate['box']) != geometry(r['box'])
            rows.append({'page': n, 'key': r['key'], 'query': r['query'],
                         'md_sha256': r['md_sha256'], 'sidecar_sha256': r['sidecar_sha256'],
                         'baseline': {k: r[k] for k in candidate}, 'candidate': candidate, 'exact_only': exact,
                         'changed': changed_box, 'decision': reason,
                         'target_span': span, 'area_ratio': area(candidate['box'])/area(r['box']) if r['box'] else None})
        print(f'page {n}: {len(changed)} word geometries changed', flush=True)
    assert len(rows) == 111 and len({(r['page'],r['key']) for r in rows}) == 111
    assert all(r['baseline'] == r['candidate'] for r in rows if r['baseline']['status'] in ('located', 'buffered'))
    counts = {name: dict(Counter(r[name]['status'] for r in rows)) for name in ('baseline', 'exact_only', 'candidate')}
    report = {'experiment_version': EXPERIMENT_VERSION, 'inputs_sha256': sha(args.out/'inputs.json'),
              'implementation_sha256': sha(Path(fragment_alignment.__file__)),
              'parameters': {name: getattr(fragment_alignment,name) for name in
                             ('MIN_CHARS','MAX_EDIT_RATE','MIN_MARGIN','MIN_WORD_COVERAGE')},
              'counts': counts, 'changed_boxes': sum(r['changed'] for r in rows),
              'exact_only_changed_boxes': sum(geometry(r['baseline']['box']) != geometry(r['exact_only']['box']) for r in rows),
              'candidate_differs_from_exact_only': sum(geometry(r['candidate']['box']) != geometry(r['exact_only']['box']) for r in rows),
              'changed_word_geometries': changed_words, 'total_word_records': total_words,
              'observation_decisions': dict(Counter(d['reason'] for d in diagnostics)),
              'accepted_edit_distances': dict(Counter(d['distance'] for d in diagnostics if d['reason'] == 'accepted')),
              'recovered_observations': sum(len(p['observations']) for p in inputs['pages']),
              'correction_decisions': dict(Counter(r['decision'] for r in rows)),
              'paid_cost': 0, 'elapsed_seconds': time.monotonic()-started,
              'accuracy': 'Not measured: changed boxes require independent image review.',
              'corrections': rows}
    write(args.out/'comparison.json', report)
    write(args.out/'alignment_diagnostics.json', diagnostics)
    print(json.dumps({k:v for k,v in report.items() if k != 'corrections'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--slug', default='bachehaye-ghali-fresh30-20260906-123256')
    parser.add_argument('--baseline', type=Path, default=Path('out/geometry_best_attempt/after.json'))
    parser.add_argument('--out', type=Path, default=Path('out/levenshtein_fragment_final'))
    args = parser.parse_args()
    if args.collect:
        collect(args)
    evaluate(args)


if __name__ == '__main__':
    main()
