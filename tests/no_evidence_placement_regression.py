"""Step 0: a scanned page with zero read lines shows no estimate, only a reason.

Offline and synthetic: no API key, no books/ workspace. Partially read pages
must keep today's coverage-first estimates; embedded-text pages are untouched.
"""
from pathlib import Path
import json
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from click.testing import CliRunner
from farsi2epub import locate, review
from farsi2epub.cli import main
from farsi2epub.geometry_attempt import PageProjection, best_attempt
from farsi2epub.scan import PlacementService, no_evidence_reason


def _workspace(root):
    with fitz.open() as doc:
        doc.new_page(width=100, height=100)
        doc.save(root/'source.pdf')
    return SimpleNamespace(root=root, pdf_path=root/'source.pdf', slug='synthetic')


LINES = [locate._ScanLine(fitz.Rect(10, y, 90, y+10), [fitz.Rect(55, y, 90, y+10), fitz.Rect(10, y, 45, y+10)])
         for y in (10, 50)]
MD = 'alpha beta gamma delta'


def _reader_for(readable_top):
    """Fake reader: the top line is legible only when `readable_top`; the
    bottom line is never legible (so it is never persisted)."""
    def render(page, rect, factor=1):
        if rect.y0 >= 30 or not readable_top:
            return b'?' + str(tuple(round(v, 2) for v in rect)).encode()
        return b'alpha beta' if rect.width > 70 else b'alpha' if rect.x0 > 50 else b'beta'
    def reader(images, *args):
        rows = [dict(region_index=i, lines=[im.decode()] if not im.startswith(b'?') else ['x'],
                     legible=not im.startswith(b'?'), complete=True) for i, im in enumerate(images)]
        return rows, {}, .001
    return render, reader


def _poll(n, until=10.):
    start = time.time()
    while True:
        payload = review._boxes_payload(review._REFINER.ws, n)
        if not payload.get('pending'):
            return payload
        assert time.time()-start < until, 'payload stayed pending'
        time.sleep(.05)


def run():
    # Pure rule: only "detected lines, none read" on a scan page qualifies.
    assert no_evidence_reason({'detected_lines': 19, 'read_lines': 0,
                               'acquisition_reasons': ['reader_credit_unavailable']}) \
        == 'no evidence: reader_credit_unavailable'
    assert no_evidence_reason({'detected_lines': 10, 'read_lines': 0, 'acquisition_reasons': ['budget_exhausted']}) \
        == 'no evidence: budget_exhausted'
    assert no_evidence_reason({'detected_lines': 3, 'read_lines': 0, 'acquisition_reasons': []}) \
        == 'no evidence: no_reading_stored'
    assert no_evidence_reason({'detected_lines': 3, 'read_lines': 1, 'acquisition_reasons': ['budget_exhausted']}) is None
    assert no_evidence_reason({'status': 'blank_nontext', 'detected_lines': 0, 'read_lines': 0}) is None
    assert no_evidence_reason({'source': 'match', 'detected_lines': 4, 'read_lines': 0}) is None
    assert no_evidence_reason({}) is None and no_evidence_reason(None) is None

    with tempfile.TemporaryDirectory() as directory:
        ws = _workspace(Path(directory))
        q = locate.Query('gamma', (MD.index('gamma'), MD.index('gamma')+5))
        # 1. Detected lines, zero readings (reader credit refused for every call).
        render, _ = _reader_for(False)
        def refused(*_a):
            raise RuntimeError('budget_exhausted')
        with patch.object(locate, '_scan_page_lines', return_value=LINES), \
             patch.object(PlacementService, '_render', side_effect=render):
            service = PlacementService(ws, reader=refused)
            _words, summary = service.ensure_page_map(1)
            assert summary['detected_lines'] == 2 and summary['read_lines'] == 0
            assert summary['acquisition_reasons'] == ['budget_exhausted']
            assert service.refine(1, MD, [q], [None]) == [None]
            result = service.results(1, MD, [q], [None])[0]
            assert result.status == 'unresolved' and result.box is None
            assert result.reason == 'no evidence: budget_exhausted', result.reason
            assert not list((ws.root/'locate_evidence'/'boxes').glob('*.json'))
            service.close()
            # A box cached before Step 0 (same key: the map revision did not
            # change) must not be replayed as an approximate box.
            offline = PlacementService(ws, mode='offline')
            words, summary = offline.ensure_page_map(1)
            stale = best_attempt(MD, q, words, PageProjection(MD, words, json.loads(
                (ws.root/'locate_evidence'/'pages'/f'{offline._page_key(1)}.json').read_text())['regions']))
            assert stale.status == 'estimated'  # what the old engine would have shown
            key = offline._key(1, MD, q)
            (ws.root/'locate_evidence'/'boxes').mkdir(parents=True, exist_ok=True)
            (ws.root/'locate_evidence'/'boxes'/f'{key}.json').write_text(json.dumps({'key': key, 'result': stale.to_dict()}))
            assert offline.results(1, MD, [q], [None])[0].status == 'pending'
            assert offline.replay(1, MD, [q], [None]) == [None]
            assert offline.results(1, MD, [q], [None])[0].reason == 'no evidence: budget_exhausted'
            # Initial render and /boxes/<page> polling agree: banner + chip.
            issues = [{'type': 'wrong_word', 'description': 'd', 'snippet': 'gamma'}]
            inputs = lambda _ws, _n: ({}, [dict(i) for i in issues], [], MD, 'no_suggestion')
            with patch.object(review, '_page_box_inputs', side_effect=inputs), \
                 patch.object(review, '_image_path_for', return_value=None):
                fresh = PlacementService(ws, mode='offline')
                review._REFINER = fresh
                try:
                    view = review._page_view(ws, 1)
                    assert view['evidence_banner'] and view['evidence_banner']['reason'] == 'no evidence: budget_exhausted'
                    payload = _poll(1)
                    assert payload['boxes'] == [] and payload['evidence_banner'] == view['evidence_banner']
                    chips = [c for c in list(payload['hunk_scores'].values()) + list(payload['issue_scores'].values()) if c]
                    assert chips and all(c['label'] == 'no evidence' and c['cls'] == 'pm-none'
                                         and c['status'] == 'unresolved' and 'budget exhausted' in c['title']
                                         for c in chips), chips
                    html = review._render_index(ws, [1], [], bbox_refine=True)
                    assert 'id="evidence-banner-1" role="status">' in html  # visible, not hidden
                    assert '>no evidence</span>' in html and 'qc-box-estimated' not in html.split('</style>')[1].split('<script')[0]
                finally:
                    review._REFINER = None
                    fresh.close()
            offline.close()

    with tempfile.TemporaryDirectory() as directory:
        # 2. Partial readings: unchanged coverage-first estimate on the unread line.
        ws = _workspace(Path(directory))
        render, reader = _reader_for(True)
        with patch.object(locate, '_scan_page_lines', return_value=LINES), \
             patch.object(PlacementService, '_render', side_effect=render):
            service = PlacementService(ws, reader=reader)
            _words, summary = service.ensure_page_map(1)
            assert summary['detected_lines'] == 2 and summary['read_lines'] == 1, summary
            assert no_evidence_reason(summary) is None
            q = locate.Query('gamma', (MD.index('gamma'), MD.index('gamma')+5))
            box = service.refine(1, MD, [q], [None])[0]
            assert box is not None and box['status'] == 'estimated', box
            # It is still persisted and replayed offline exactly as before.
            service.close()
            offline = PlacementService(ws, mode='offline', reader=lambda *a: 1/0)
            assert offline.results(1, MD, [q], [None])[0].box == box
            offline.close()
            with patch.object(review, '_REFINER', SimpleNamespace(page_map_summary=lambda n: summary)):
                assert review._evidence_banner('synthetic', 1, summary) is None

    # 3. Embedded-text ("match") boxes bypass the rule entirely.
    match_box = {'x0': .1, 'y0': .1, 'x1': .2, 'y1': .2, 'source': 'match', 'status': 'located'}
    with tempfile.TemporaryDirectory() as directory:
        ws = _workspace(Path(directory))
        service = PlacementService(ws, mode='offline')
        with patch.object(PlacementService, 'page_map_summary', side_effect=AssertionError('read page map')):
            assert service.results(1, MD, [locate.Query('alpha')], [match_box])[0].status == 'located'
        service.close()

    # 4. Banner text and its resume command are real CLI invocations.
    banner = review._evidence_banner('my-book', 3, {'detected_lines': 10, 'read_lines': 0,
                                                     'acquisition_reasons': ['reader_credit_unavailable']})
    assert banner['command'] == 'farsi2epub geometry my-book --pages 3 --bbox-mode auto'
    assert 'no credit' in banner['text'] and '0 of 10 detected lines' in banner['text']
    assert banner['command'] in banner['hint'] and '--pages 3 --bbox-mode auto' in banner['hint']
    budget = review._evidence_banner('my-book', 3, {'detected_lines': 10, 'read_lines': 0,
                                                     'acquisition_reasons': ['budget_exhausted']})
    assert '--bbox-max-cost' in budget['hint']
    fake = SimpleNamespace(meta={'page_count': 5, 'page_range': None}, pages_done=lambda: [1, 2, 3])
    runner = CliRunner()
    with patch('farsi2epub.cli._load_workspace', return_value=fake), \
         patch('farsi2epub.cli._geometry_pages') as geometry:
        result = runner.invoke(main, banner['command'].split()[1:])
        assert result.exit_code == 0, result.output
        assert geometry.call_args.args[1:3] == ([3], 'auto')
    print('NO-EVIDENCE PLACEMENT REGRESSIONS PASSED')


if __name__ == '__main__':
    run()
