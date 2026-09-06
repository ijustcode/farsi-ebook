"""Capture full-passage boxes against fixed manifest identities, without truth.

Historical control is offline only and is explicitly a full-query re-run, not
an exact snapshot of previously capped production queries. Use saved v5 reports
for the latter. Neither engine sees annotations or target word rectangles.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from farsi2epub import locate
from farsi2epub.scan import PlacementService
from farsi2epub.workspace import Workspace, DEFAULT_BOOKS_ROOT
from farsi2epub.config import MODEL_STRONG


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--engine',choices=['consolidated','historical_full_query'],default='consolidated')
    p.add_argument('--split',choices=['dev','heldout','all'],default='dev')
    p.add_argument('--books-root',type=Path,default=DEFAULT_BOOKS_ROOT)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    boxes={}; states={}; services={}
    for item in manifest['pages']:
        if args.split!='all' and item['split']!=args.split: continue
        ws=Workspace(item['slug'],args.books_root)
        if item['slug'] not in services:
            if hashlib.sha256(ws.pdf_path.read_bytes()).hexdigest()!=item['image']['source']['pdf_sha256']:
                raise ValueError('PDF changed since manifest export')
            if args.engine=='consolidated': svc=PlacementService(ws,mode='offline')
            else:
                from historical import locate as old_locate, review as old_review
                svc=old_review._ScanBoxRefiner(ws,MODEL_STRONG,offline=True)
            services[item['slug']]=svc
        svc=services[item['slug']]
        md=(ws.text_dir/f"{item['page']:04d}.md").read_text()
        if hashlib.sha256(md.encode()).hexdigest()!=item['markdown_sha256']:
            raise ValueError('Markdown changed since manifest export')
        cases=[c for c in manifest['cases'] if c['page_id']==item['page_id']]
        queries=[]
        for c in cases:
            h=c['hunk']; span=h['markdown_span']
            cls=locate.Query if args.engine=='consolidated' else old_locate.Query
            kwargs={'kind':c['kind']} if args.engine=='consolidated' else {}
            queries.append(cls(h['old'],tuple(span) if span is not None else None,
                               (h['new'],) if h['new'].strip() else (),**kwargs))
        engine=locate if args.engine=='consolidated' else old_locate
        initial=engine.locate_queries(ws.pdf_path,item['page'],md,queries)
        located=svc.replay(item['page'],md,queries,initial)
        if args.engine!='consolidated': located=svc.apply_cached(item['page'],md,queries,initial)
        for c,b in zip(cases,located):
            boxes[c['case_id']]=b
            states[c['case_id']]=b.get('status','located') if b else 'unresolved'
        print(f"{item['slug']} p{item['page']}: {sum(b is not None for b in located)}/{len(cases)}",flush=True)
    for svc in services.values():
        if hasattr(svc,'close'): svc.close()
    report={'run':{'engine':args.engine,'mode':'offline','cost_usd':0,
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'target_contract':'full_passage','split':args.split,'audited_accuracy':False,
        'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            ([Path('farsi2epub')/name for name in ('locate.py','page_map.py','scan.py','llm.py')] if args.engine=='consolidated' else
             [Path('tests/historical')/name for name in ('locate.py','placement.py','review.py','llm.py')])}},
        'boxes':boxes,'states':states,'counts':dict(Counter(states.values()))}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report['counts'])


if __name__=='__main__': main()
