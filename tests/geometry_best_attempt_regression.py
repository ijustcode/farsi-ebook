"""Coverage-first geometry must not masquerade as independently verified placement."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import fitz
from farsi2epub import locate, review
from farsi2epub.page_map import PrintedWord, place
from farsi2epub.geometry_attempt import complete_geometry, PageProjection, best_attempt
from farsi2epub.scan import PlacementService, _cached_result, _valid_rect


def run():
    regions=[{'id':str(i),'line':0,'rect':[x,.1,x+.15,.2]} for i,x in enumerate((.75,.45,.15))]
    words=[PrintedWord(t,[0,0,0,0],0,[],False) for t in ('alpha','beta','gamma')]
    words[0]=PrintedWord('alpha',regions[0]['rect'],0,['independent'])
    mapped=complete_geometry(words,regions)
    assert mapped[0]==words[0]
    assert all(_valid_rect(w.rect) for w in mapped)
    assert sum(w.supported for w in mapped)==1
    projection=PageProjection('alpha beta gamma',mapped,regions)
    q=locate.Query('beta',(6,10))
    result=best_attempt('alpha beta gamma',q,mapped,projection)
    assert result.status=='estimated' and result.box['source']=='scan'
    assert place('alpha beta gamma',q,mapped).box is None
    assert _cached_result(result.to_dict())==result
    # Wider context remains visible, and its actual buffer is retained.
    wide=[PrintedWord(t,[.1,.1,.9,.2],0,['wide']) for t in 'one two three four five six'.split()]
    q=locate.Query('one')
    result=best_attempt('',q,wide,PageProjection('',wide,regions))
    assert result.status=='estimated' and result.buffer_after==5
    assert _cached_result(result.to_dict()) is not None
    data=result.to_dict(); data['status']='located'
    assert _cached_result(data) is None
    # Unread lines can receive an approximate Markdown alignment using ink.
    unknown=[PrintedWord('�',[.1,.1,.9,.2],0,[],False)]
    projected=PageProjection('alpha beta gamma',unknown,regions)
    for text,start,end in [('alpha',0,5),('beta',6,10),('gamma',11,16)]:
        result=best_attempt('alpha beta gamma',locate.Query(text,(start,end)),unknown,projected)
        assert result.status=='estimated' and result.box is not None
    # Ambiguous finding-only snippets are labeled guesses, not certified matches.
    md='same word same word'
    repeated=[PrintedWord(t,[.8-i*.15,.1,.9-i*.15,.2],0,[],False) for i,t in enumerate(md.split())]
    rp=PageProjection(md,repeated,regions)
    result=best_attempt(md,locate.Query('same word',None,(),'finding'),repeated,rp)
    assert result.status=='estimated' and result.box['alternatives']>0
    # Punctuation attaches to the preceding word; no fabricated blank rectangle.
    md='alpha , beta'
    pp=PageProjection(md,mapped,regions)
    assert best_attempt(md,locate.Query(',',(6,7)),mapped,pp).box is not None
    # Start/end insertions are boundary markers on the correct RTL edge.
    for span,edge in [((0,0),'right'),((16,16),'left')]:
        result=best_attempt('alpha beta gamma',locate.Query('',span,(),'insertion'),unknown,projected)
        assert result.kind=='insertion_boundary'
        word=projected.projected[0 if edge=='right' else -1]
        expected=word.rect[2 if edge=='right' else 0]
        assert abs((result.box['x0']+result.box['x1'])/2-expected)<1e-6
    assert best_attempt('',locate.Query('absent'),[],PageProjection('',[],[])).box is None
    # No API, no readings, no correction query: every detected region persists.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        with fitz.open() as doc:
            doc.new_page(width=100,height=100);doc.save(root/'source.pdf')
        ws=SimpleNamespace(root=root,pdf_path=root/'source.pdf')
        lines=[locate._ScanLine(fitz.Rect(10,10,90,20),[fitz.Rect(60,10,90,20),fitz.Rect(10,10,40,20)])]
        service=PlacementService(ws,mode='offline',reader=lambda *a:1/0)
        with patch.object(locate,'_scan_page_lines',return_value=lines):
            with patch("farsi2epub.scan.DERIVATION_VERSION", 6):
                service.ensure_page_map(1)
            words,summary=service.ensure_page_map(1)
            assert service.page_map_summary(1) == summary  # migration persists current inventory
            assert summary['geometry_complete'] and summary['detected_word_geometry']==summary['detected_words']==2
            assert summary['verified_words']==0
            q=locate.Query('beta',(6,10))
            first=service.replay(1,'alpha beta',[q],[None])[0]
            assert first['status']=='estimated'
            second=PlacementService(ws,mode='offline',reader=lambda *a:1/0)
            assert second.results(1,'alpha beta',[q],[None])[0].box==first
            second.close()
        service.close()
    # UI accepts estimates and exposes their qualification on initial and polling paths.
    hunk={}; estimated=best_attempt('alpha beta gamma',locate.Query('beta',(6,10)),mapped,projection)
    boxes=review._attach_boxes([('h1',q,None,hunk,None)],[estimated.box],[estimated])
    assert boxes[0]['status']=='estimated' and hunk['placement']['label'].startswith('approximate')
    print('GEOMETRY BEST-ATTEMPT REGRESSIONS PASSED')

if __name__=='__main__': run()
