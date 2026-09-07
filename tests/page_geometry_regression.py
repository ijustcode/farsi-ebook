"""Correction-independent acquisition, resumability and CLI scope regressions."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from click.testing import CliRunner
from farsi2epub import locate
from farsi2epub.scan import PlacementService
from farsi2epub.cli import main
from farsi2epub.page_map import _aligned_exact_span, PrintedWord, place


def run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with fitz.open() as doc:
            doc.new_page(width=100, height=100)
            doc.save(root/'source.pdf')
        ws = SimpleNamespace(root=root, pdf_path=root/'source.pdf')
        lines = [locate._ScanLine(fitz.Rect(10,10,90,20), [fitz.Rect(55,10,90,20), fitz.Rect(10,10,45,20)])]
        calls = []
        def render(page, rect, factor=1):
            return b'alpha beta' if rect.width > 70 else b'alpha' if rect.x0 > 50 else b'beta'
        def reader(images, *args):
            calls.append(images)
            return [dict(region_index=i, lines=[im.decode()], legible=True, complete=True) for i,im in enumerate(images)], {}, .001
        with patch.object(locate, '_scan_page_lines', return_value=lines), patch.object(PlacementService, '_render', side_effect=render):
            blocked = PlacementService(ws, reader=reader, max_cost=0)
            words, summary = blocked.ensure_page_map(1)
            assert not calls and summary['status'] == 'acquisition_blocked'
            assert summary['verified_words'] == 0
            blocked.close()
            service = PlacementService(ws, reader=reader)
            key_before = service._key(1,'alpha',locate.Query('alpha'))
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: service.ensure_page_map(1), range(2)))
            assert all(r[1]['status']=='complete' for r in results)
            assert results[0][1]['verified_words'] == 2
            assert len(calls) == 1, calls  # identical full group reuses the line image
            assert key_before != service._key(1,'alpha',locate.Query('alpha'))
            offline = PlacementService(ws, mode='offline', reader=lambda *a: 1/0)
            assert offline.ensure_page_map(1) == results[0]
            service.close(); offline.close()
    # A later request failure preserves an earlier line's verified geometry.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with fitz.open() as doc:
            doc.new_page(width=100, height=100); doc.save(root/'source.pdf')
        ws = SimpleNamespace(root=root, pdf_path=root/'source.pdf')
        lines = [locate._ScanLine(fitz.Rect(10,y,90,y+10), [fitz.Rect(65,y,90,y+10), fitz.Rect(35,y,60,y+10), fitz.Rect(10,y,30,y+10)]) for y in (10,50)]
        def render(page, rect, factor=1):
            tokens = ('alpha','beta','epsilon') if rect.y0 < 30 else ('gamma','delta','zeta')
            return (('line ' if rect.width > 80 else '') + (' '.join(tokens) if rect.width > 70 else ' '.join(tokens[:2]) if rect.width > 40 else tokens[0] if rect.x0 > 60 else tokens[1] if rect.x0 > 30 else tokens[2])).encode()
        def failing_reader(images, *args):
            if any(b'gamma' in im or b'delta' in im for im in images if not im.startswith(b'line ')):
                raise RuntimeError('simulated transport failure')
            return [dict(region_index=i, lines=[im.decode().removeprefix('line ')], legible=True, complete=True) for i,im in enumerate(images)], {}, .001
        with patch.object(locate, '_scan_page_lines', return_value=lines), patch.object(PlacementService, '_render', side_effect=render):
            service = PlacementService(ws, reader=failing_reader)
            words, summary = service.ensure_page_map(1)
            assert summary['status'] == 'acquisition_blocked'
            assert all(w.supported for w in words if w.line == 0)
            assert service.budget.uncertain > 0
            service.close()
            offline = PlacementService(ws, mode='offline')
            replayed, _ = offline.ensure_page_map(1)
            assert [w for w in replayed if w.supported] == [w for w in words if w.supported]
            offline.close()
    runner = CliRunner()
    for command in ('geometry', 'transcribe'):
        help_result = runner.invoke(main, [command, '--help'])
        assert help_result.exit_code == 0, help_result.output
        assert all(option in help_result.output for option in ('--bbox-mode','--bbox-model','--bbox-max-cost'))
    fake = SimpleNamespace(meta={'page_count':3}, pages_done=lambda:[1,2,3])
    with patch('farsi2epub.cli._load_workspace', return_value=fake), patch('farsi2epub.cli._geometry_pages') as geometry:
        result = runner.invoke(main, ['transcribe','old','--pages','2','--qc','skip','--bbox-mode','offline'])
        assert result.exit_code == 0, result.exception
        assert geometry.call_args.args[1] == [2]
    md = 'beginning uniquely anchored words damaged context target trailing uniquely anchored sentence ends'
    start = md.index('target')
    reader = 'beginning uniquely anchored words � target trailing uniquely anchored sentence ends'.split()
    assert _aligned_exact_span(md, locate.Query('target',(start,start+6)),reader) == (5,6)
    assert _aligned_exact_span(md, locate.Query('target',(start,start+6)),['target']) is None
    # Screenshot phrase: matching text cannot compensate for missing ink evidence.
    words = [PrintedWord('آقا',[.75,.6,.8,.63],0,[],False), PrintedWord('گفت',[.65,.6,.74,.63],0,['group'])]
    query = locate.Query('«آقا» گفت:', (0,len('«آقا» گفت:')), (), 'finding')
    assert place('«آقا» گفت:', query, words).reason == 'word_geometry_unverified'
    words[0].supported = True
    assert place('«آقا» گفت:', query, words).box is not None
    assert place('آقا گفت آقا گفت',locate.Query('آقا گفت',None,(),'finding'),words).reason == 'ambiguous_occurrence'
    print('PAGE GEOMETRY REGRESSIONS PASSED')

if __name__ == '__main__':
    run()
