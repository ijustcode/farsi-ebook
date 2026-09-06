"""Export detector geometry on development pages without treating it as truth.

Kraken is an optional research dependency, not installed into the CLI runtime.
Audited line matching can be added once the independent page annotations freeze.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import fitz
from farsi2epub import locate
from farsi2epub.workspace import DEFAULT_BOOKS_ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--detector",choices=["projection","kraken"],required=True)
    p.add_argument("--limit",type=int,default=6)
    p.add_argument("--page-id",action="append",default=[],
                   help="Explicit development page identity; repeat to evaluate reviewed pages")
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    pages=[page for page in manifest["pages"] if page["split"]=="dev"]
    # Fixed development-page identity, independent of any detector output.
    if args.page_id:
        selected=set(args.page_id)
        if len(selected)!=len(args.page_id) or selected-{p['page_id'] for p in pages}:
            raise ValueError('page IDs must be unique development pages')
        pages=sorted([page for page in pages if page['page_id'] in selected],key=lambda v:v['page_id'])
    else:
        pages=sorted(pages,key=lambda v:hashlib.sha256(v["page_id"].encode()).hexdigest())[:args.limit]
    out=[]
    for item in pages:
        started=time.monotonic()
        image=args.manifest.parent/item["image"]["path"]
        if hashlib.sha256(image.read_bytes()).hexdigest()!=item["image"]["sha256"]:
            raise ValueError("source image changed")
        if args.detector=="projection":
            with fitz.open(DEFAULT_BOOKS_ROOT/item["slug"]/'source.pdf') as doc:
                page=doc[item["page"]-1]
                rects=[list(locate._rect_to_fracs(r.rect,page.rect).values()) for r in locate._scan_page_lines(page)]
            version="projection_v1"
        else:
            from PIL import Image
            from kraken import blla
            from importlib.metadata import version as package_version
            result=blla.segment(Image.open(image),text_direction="horizontal-rl",device="cpu")
            w,h=item["image"]["width_px"],item["image"]["height_px"]
            rects=[]
            for line in result.lines:
                xs,ys=zip(*line.boundary)
                rects.append([min(xs)/w,min(ys)/h,max(xs)/w,max(ys)/h])
            version=package_version("kraken")
        out.append({"page_id":item["page_id"],"slug":item["slug"],"page":item["page"],
                    "image_sha256":item["image"]["sha256"],"rects":rects,
                    "seconds":time.monotonic()-started})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"detector":args.detector,"version":version,
        "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "promotion_eligible":False,"reason":"independent line truth not yet audited","pages":out},indent=2))
    print(json.dumps({"detector":args.detector,"pages":len(out),"lines":[len(v['rects']) for v in out]}))


if __name__=="__main__":main()
