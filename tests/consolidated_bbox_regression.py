"""Offline regression checks for the consolidated two-path placement service."""
from pathlib import Path
import json
import sys
import tempfile
import httpx
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from click.testing import CliRunner
from farsi2epub import locate, review, llm
from farsi2epub.cli import main
from farsi2epub.page_map import PrintedWord, place, supported_groups
from farsi2epub.scan import PlacementService, Budget
import bbox_metrics as metric


def run():
    # Detached upper ink can enter a retry crop without changing accepted
    # geometry or crossing halfway into the neighboring line's whitespace.
    lines = [locate._ScanLine(fitz.Rect(10,20,90,40), []),
             locate._ScanLine(fitz.Rect(10,50,90,70), []),
             locate._ScanLine(fitz.Rect(10,80,90,100), [])]
    original = tuple(lines[1].rect)
    crop = PlacementService._retry_line_clip(fitz.Rect(0,0,100,120), lines, 1)
    assert crop.y0 == 45 and crop.y1 == 75
    assert tuple(lines[1].rect) == original
    first = PlacementService._retry_line_clip(fitz.Rect(0,0,100,120), lines, 0)
    assert first.y0 == 12 and first.y1 == 45
    # Independent crop readings, not equal counts, certify word geometry.
    groups = [("alpha", [.7,.1,.9,.2], "a"), ("beta gamma", [.2,.1,.65,.2], "b")]
    words = supported_groups("alpha beta gamma", groups, 0, "line")
    assert len(words) == 3
    assert not supported_groups("alpha wrong gamma", groups, 0, "line")
    result = place("alpha beta gamma", locate.Query("beta", (6,10)), words)
    assert result.status == "buffered" and result.buffer_after == 1
    assert result.box["source"] == "scan"
    # Far-away position hints do not exist in the occurrence API.
    repeated = words + [PrintedWord(w.text,w.rect,1,w.evidence) for w in words]
    assert place("",locate.Query("beta"),repeated).status == "unresolved"
    fused = supported_groups("one two three four five six",[("one two three four five six",[.1,.1,.9,.2],"all")],0,"line")
    assert place("",locate.Query("one"),fused).box is None
    assert locate._fold_word("۱۴۰۳") == locate._fold_word("١٤٠٣") == "1403"
    overlapping = [PrintedWord("آرام",[.7-i*.2,.1,.8-i*.2,.2],0) for i in range(3)]
    assert place("",locate.Query("آرام آرام"),overlapping).status == "unresolved"
    insertion_md = "one two three four five six seven eight nine ten eleven twelve"
    offset = insertion_md.index("seven")
    iq = locate.Query("",(offset,offset),("eight",),"insertion")
    iwords = [PrintedWord(w,[.95-i*.07,.1,.99-i*.07,.2],0) for i,w in enumerate(insertion_md.split())]
    assert place(insertion_md,iq,iwords).kind == "insertion_boundary"
    # All words in a long correction remain in the contract.
    text="one two three four five six seven"
    hunks=review._derive_hunks(text,"changed")
    specs=review._box_specs([],hunks,text)
    assert specs[0][1].text == text
    assert "bbox" not in llm.QCIssue.model_fields
    assert "estimate its bounding box" not in llm.QC_SYSTEM
    issue={"snippet":"missing", "bbox":[0,0,1000,1000]}
    specs=review._box_specs([issue],[],"")
    assert specs[0][2] is None
    assert review._attach_boxes(specs,[None]) == [] and issue["box"] is None
    assert issue["placement"]["label"] == "unresolved"
    # Geometric metric: union segments, full coverage, fractional excess.
    line={"line_id":"l", "words":[{"word_id":str(i),"rect":[i*.15,.1,i*.15+.1,.2],"text":str(i)} for i in range(5)]}
    page={"status":"audited","lines":[line]}
    target={"status":"audited","word_ids":["2"]}
    exact={"x0":.3,"y0":.1,"x1":.4,"y1":.2}
    assert metric.grade(page,target,exact)["tight"]
    duplicate={**exact,"segments":[exact,exact]}
    assert metric.grade(page,target,duplicate)["target_coverage"] == [1.0]
    wide={"x0":.15,"y0":.1,"x1":.4,"y1":.2}
    assert metric.grade(page,target,wide)["excess"] > .99
    assert metric.grade(page,target,wide)["buffered"]
    assert not metric.grade(page,target,None)["full"]
    sliver={**exact,"x1":.34}
    assert not metric.grade(page,target,sliver)["full"]
    # Concurrent reservations never exceed the cap.
    budget=Budget(.1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims=list(pool.map(lambda _:budget.reserve(.03),range(8)))
    assert sum(claims)==3
    budget.settle(.03,None)
    assert not budget.reserve(.03)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);pdf=root/'source.pdf'
        with fitz.open() as doc:
            doc.new_page();doc.save(pdf)
        ws=SimpleNamespace(root=root,pdf_path=pdf)
        counter=[]
        def reader(pngs,model,page):
            counter.append(len(pngs))
            return [{"region_index":i,"lines":["alpha"],"legible":True,"complete":True} for i,_ in enumerate(pngs)],{},.001
        service=PlacementService(ws,reader=reader,max_cost=1)
        # Concurrent same-key readers share physical evidence across instances.
        other=PlacementService(ws,reader=reader,max_cost=1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(lambda svc:svc._read([b'png'],1,True),[service,other]))
        assert counter==[1],counter
        assert values[0][0][0]["lines"]==["alpha"]
        offline=PlacementService(ws,mode="offline",reader=lambda *a:(_ for _ in ()).throw(AssertionError("API")))
        assert offline._read([b'png'],1,False)[0][0]["lines"]==["alpha"]
        assert offline._read([b'missing'],1,False)[0]==[None]
        q=locate.Query("alpha")
        corrupt_key=offline._key(1,"alpha",q)
        corrupt=root/'locate_evidence'/'boxes'/f'{corrupt_key}.json'
        corrupt.parent.mkdir(parents=True,exist_ok=True)
        corrupt.write_text(json.dumps({'key':corrupt_key,'result':{'status':'located','box':None}}))
        assert offline.results(1,"alpha",[q],[None])[0].status == 'pending'

        assert offline.pending(1,"alpha",[q],[None])
        assert offline.refine(1,"alpha",[q],[None])==[None]
        assert not offline.pending(1,"alpha",[q],[None])
        assert offline.results(1,"alpha",[q],[None])[0].status=="unresolved"
        key=offline._key(1,"alpha",q)
        assert key!=offline._key(1,"changed",q)
        # Source replacement invalidates keys in the same running process.
        with fitz.open() as doc:
            doc.new_page(width=800);doc.save(root/'other.pdf')
        (root/'other.pdf').replace(pdf)
        assert key!=offline._key(1,"alpha",q)
    # Exercise the entire scan engine with independent line and group reads.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with fitz.open() as doc:
            doc.new_page(width=100, height=100); doc.save(root/'source.pdf')
        ws = SimpleNamespace(root=root, pdf_path=root/'source.pdf')
        lines = [locate._ScanLine(fitz.Rect(10,10,90,20),
                 [fitz.Rect(65,10,90,20),fitz.Rect(35,10,60,20),fitz.Rect(10,10,30,20)])]
        def render(page, rect, factor=1.):
            if rect.width > 70: return b'alpha beta gamma'
            return b'alpha' if rect.x0 > 60 else b'beta' if rect.x0 > 30 else b'gamma'
        calls = []
        def reader(images, *_):
            calls.append(len(images))
            return [{"region_index":i,"lines":[v.decode()],"legible":True,"complete":True}
                    for i,v in enumerate(images)], {}, .001
        q = locate.Query("beta", (6,10))
        service = PlacementService(ws, reader=reader)
        with patch.object(locate, '_scan_page_lines', return_value=lines), patch.object(PlacementService, '_render', side_effect=render):
            result = service.refine(1,"alpha beta gamma",[q],[None])[0]
            assert result['source'] == 'scan' and result['x0'] == .35
            paid = sum(calls)
            # Re-derive from physical observations under changed Markdown.
            offline = PlacementService(ws, mode='offline', reader=lambda *a:1/0)
            again = offline.refine(1,"alpha beta gamma!",[q],[None])[0]
            assert again == result and sum(calls) == paid
            # Boundary marker has independently recognized adjacent anchors.
            boundary = locate.Query("",(6,6),(),"insertion")
            placed = service.refine(1,"alpha beta gamma",[boundary],[None])[0]
            assert placed['kind'] == 'insertion_boundary'
            service.close(); offline.close()
    # A pending poll must schedule work even when there is no rectangle.
    fake = SimpleNamespace(pending=lambda *a:True, schedule=lambda *a: calls.append('scheduled'))
    with patch.object(review,'_REFINER',fake), patch.object(review,'_page_box_inputs',return_value=({},[{'snippet':'absent'}],[],"",None)), patch.object(review,'_locate_queries_cached',return_value=(None,)):
        assert review._boxes_payload(ws,1) == {'pending':True}
        assert calls[-1] == 'scheduled'
    # PyMuPDF word coordinates precede page rotation; the display does not.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'rotated.pdf'
        with fitz.open() as doc:
            page=doc.new_page(width=200,height=100)
            page.insert_font(fontname='persian',fontfile='assets/fonts/Vazirmatn-Regular.ttf')
            page.insert_text((20,30),'سلام',fontname='persian')
            page.set_rotation(90); doc.save(path)
        with fitz.open(path) as doc:
            page=doc[0]; expected=locate._rect_to_fracs(fitz.Rect(page.get_text('words')[0][:4])*page.rotation_matrix,page.rect)
        box=locate.locate_queries(path,1,'سلام',[locate.Query('سلام')])[0]
        assert box is not None
        assert all(abs(box[k]-expected[k])<1e-6 for k in expected)
    # A rejected request cannot consume actual inference spend; cancellation
    # also prevents a dispatch. Transport failures remain conservatively reserved.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        with fitz.open() as doc:
            doc.new_page();doc.save(root/'source.pdf')
        ws=SimpleNamespace(root=root,pdf_path=root/'source.pdf')
        def rejected(*args):
            response=httpx.Response(400,request=httpx.Request('POST','https://api.anthropic.com/v1/messages'))
            raise llm.anthropic.BadRequestError('rejected',response=response,body={'error':{'message':'test rejection'}})
        service=PlacementService(ws,reader=rejected,max_cost=1)
        try:service._read([b'pixels'],1,True)
        except llm.anthropic.BadRequestError:pass
        else:raise AssertionError('expected rejection')
        assert service.budget.snapshot() == {'limit':1,'spent':0.,'reserved':0.,'uncertain':0.}
        assert service._failures['reader']['status_code'] == 400
        assert len(list((root/'locate_evidence'/'runs').glob('*.json'))) == 1
        service.close()
        assert service._read([b'never-dispatch'],1,True)[0] == [None]
        # Unreadability and clipping remain diagnostic failures, never durable
        # raw evidence that an offline replay can mistake for a supported read.
        def unreadable(*args):
            return [{'region_index':0,'lines':['uncertain'],
                     'legible':False,'complete':False}], {}, .001
        service=PlacementService(ws,reader=unreadable,max_cost=1)
        assert service._read([b'unreadable'],1,True)[0] == [None]
        failure=service._failures['regions'][0]
        assert failure['legible'] is False and failure['complete'] is False
        assert not (service.root/'readings'/f"{failure['key']}.json").exists()
        assert service._read([b'unreadable'],1,False)[0] == [None]
        service.close()
    help_result=CliRunner().invoke(main,["review","--help"])
    assert help_result.exit_code==0
    assert all(flag in help_result.output for flag in ("--bbox-mode","--bbox-model","--bbox-max-cost"))
    # Legacy switches reach the same new foreground contract and announce migration.
    with patch('farsi2epub.cli._load_workspace',return_value=ws), patch.object(review,'read_server_state',return_value=None), patch.object(review,'run_review') as start:
        for flag,mode in (('--bbox-refine','auto'),('--no-bbox-refine','offline')):
            response=CliRunner().invoke(main,['review','unused',flag,'--bbox-max-cost','1.25'])
            assert response.exit_code == 0, response.output
            assert 'Deprecated' in response.output
            assert start.call_args.kwargs['bbox_mode'] == mode
            assert start.call_args.kwargs['bbox_max_cost'] == 1.25
        response=CliRunner().invoke(main,['review','unused','--bbox-refine-model','claude-sonnet-5'])
        assert response.exit_code == 0 and 'deprecated' in response.output
        assert start.call_args.kwargs['bbox_model'] == 'claude-sonnet-5'
    conflict=CliRunner().invoke(main,['review','unused','--bbox-mode','auto','--no-bbox-refine'])
    assert conflict.exit_code == 2 and 'Conflicting' in conflict.output
    old=CliRunner().invoke(main,["review","unused","--bbox-refine-algorithm","legacy_v1"])
    assert old.exit_code==2 and "offline" in old.output
    bad=CliRunner().invoke(main,["review","unused","--bbox-max-cost","nan"])
    assert bad.exit_code==2
    assert "hasScanBox" not in review._PAGE_TEMPLATE
    assert "legend-model" not in review._PAGE_TEMPLATE
    print("CONSOLIDATED BBOX REGRESSIONS PASSED")


if __name__=="__main__":
    run()
