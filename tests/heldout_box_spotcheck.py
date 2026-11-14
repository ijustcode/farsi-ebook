#!/usr/bin/env python3
"""Held-out word placement, hadpi p3-p5: spot-check of shown boxes (owner decision 2026-11-14).

Pages 3-5 have owner text truth but no audited word boxes, so "located and
correct" cannot be computed there. Instead the frozen configuration (L1 lines,
line assignment ``--la T``, rule TA, shared vertical rule) is run once, a
random sample of the boxes it would show is drawn with a fixed seed, and the
owner marks each one right or wrong. That gives precision directly; coverage
(share of words that get a box) needs no truth; their product estimates
"located and correct".

    ./venv/bin/python tests/heldout_box_spotcheck.py build          # run the placer once, draw the sample ($0)
    ./venv/bin/python tests/heldout_box_spotcheck.py serve [--port 8772] [--open]
    ./venv/bin/python tests/heldout_box_spotcheck.py report

Two line texts are placed: the owner's truth and the cached Opus band reading
(``t1r_opus``, run 1). The tool does not say which text a box came from. A box
is RIGHT when it sits on the named word and its left and right edges are
within about half a letter of the word's edges; height does not matter (D12).
For a token the reader wrote as two words joined, the box should span both.

Nothing is tuned here and nothing is sent anywhere. ``build`` refuses to run a
second time: the sample is the test. Files (private, they contain page text):
``books/hadpi-articles/truth/box_spotcheck.sample.json`` and ``.marks.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

SLUG = "hadpi-articles"
PAGES = (3, 4, 5)
PID = {3: "h3", 4: "h4", 5: "h5"}
SOURCES = ("truth", "t1r_opus")
RULE = "TA"
PER_PAGE_PER_SOURCE = 40
TRUTH = ROOT / "books" / SLUG / "truth"
SAMPLE = TRUTH / "box_spotcheck.sample.json"
MARKS = TRUTH / "box_spotcheck.marks.json"
OUT = ROOT / "out" / "word_placer" / "heldout_spotcheck.json"
VERDICTS = ("right", "wrong", "cannot_tell")


def _atomic(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def truth_lines(page: int) -> list[list[str]]:
    st = json.loads((TRUTH / f"{page:04d}.text_audit.json").read_text())
    raw = (TRUTH / f"{page:04d}.lines.txt").read_bytes()
    if st.get("status") != "audited" or not st["export"].get("final") \
            or st["export"]["sha256"] != hashlib.sha256(raw).hexdigest():
        raise SystemExit(f"page {page}: text truth is not a final audited export")
    return [l.split() for l in raw.decode("utf-8").split("\n") if l.strip()]


def opus_lines(page: int) -> list[list[str]]:
    """Cached reading, loaded with the frozen experiment script (never fetched)."""
    import heldout_score
    TE = heldout_score.load_frozen()
    rec = TE.Runner(cap=TE.CAP_HELDOUT_USD, ledger_name=TE.LEDGER_HELDOUT).cached(
        TE.build_request("t1r_opus", PID[page], 0))
    if rec is None:
        raise SystemExit(f"no cached t1r_opus reading for page {page}; refusing to call the API")
    return [l.split() for l in TE.output_of(rec, "t1r_opus")["lines"] if l and l.strip()]


def cmd_build(args) -> None:
    if SAMPLE.exists() and not args.force:
        raise SystemExit(f"{SAMPLE} exists: the sample is drawn once. (--force redraws it and discards the marks.)")
    os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
    import line_finder as LF
    import real_truth_eval as rt
    import word_placer as WP
    import word_placer_eval as E
    E.T_OPTS.update(vert="comp", la="T")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    items, summary = [], {}
    for page in PAGES:
        img = TRUTH / f"{page:04d}.page.png"
        gray = LF.load_gray(img)
        H, W = gray.shape
        tp = rt.TruthPage(SLUG, page, "typewriter_photo", "heldout_no_boxes", W, H, img, "", img, "", [], [])
        ev = WP.page_evidence(gray)
        texts = {"truth": truth_lines(page), "t1r_opus": opus_lines(page)}
        boxes = {}
        for source in SOURCES:
            lines = texts[source]
            pg = dict(name=f"{SLUG}/{page}", gray=gray, truth_page=tp, texts=lines, kind="heldout")
            arms, info = E.run_T(pg, "C", ev=ev, texts=lines, src=None)
            shown = []
            for (li, ti), x in sorted(arms[RULE].items()):
                if x["status"] == "located" and x.get("segments"):
                    q = [[round(a * W, 1), round(b * H, 1)] for a, b in x["segments"][0]["quad"]]
                    shown.append({"line": li, "token": ti, "text": lines[li][ti], "quad": q})
            n_words = sum(len(l) for l in lines)
            summary[f"{page}/{source}"] = {
                "text_lines": len(lines), "words": n_words, "boxes_shown": len(shown),
                "coverage": round(len(shown) / n_words, 4),
                "every_box_arm_shown": sum(1 for x in arms["Tall"].values() if x["status"] == "located"),
                "line_assignment": {k: v for k, v in (info.get("la_template") or {}).items()},
                "detected_lines": len(WP.l1_lines(ev))}
            rng = random.Random(int(hashlib.sha256(f"heldout-spotcheck/{page}/{source}".encode()).hexdigest()[:12], 16))
            boxes[source] = rng.sample(shown, min(PER_PAGE_PER_SOURCE, len(shown)))
            print(f"  p{page} {source:9s} words {n_words}  boxes shown {len(shown)} ({100 * len(shown) / n_words:.1f}%)", flush=True)
        merged: list[dict] = []
        for source in SOURCES:
            for b in boxes[source]:
                xs = [p[0] for p in b["quad"]]
                twin = next((m for m in merged if m["text"] == b["text"]
                             and abs(min(p[0] for p in m["quad"]) - min(xs)) <= 2
                             and abs(max(p[0] for p in m["quad"]) - max(xs)) <= 2
                             and abs(m["quad"][0][1] - b["quad"][0][1]) <= 6), None)
                if twin:
                    twin["sources"].append(source)
                else:
                    merged.append({"page": page, "W": W, "H": H, "sources": [source], **b})
        random.Random(page).shuffle(merged)
        items += merged
    for k, it in enumerate(items):
        it["id"] = k
    _atomic(SAMPLE, {"schema": "heldout_box_spotcheck/1", "rule": RULE, "la": "T", "code": head,
                     "per_page_per_source": PER_PAGE_PER_SOURCE, "summary": summary, "items": items})
    if MARKS.exists():
        MARKS.unlink()
    print(f"sample: {len(items)} boxes to mark -> {SAMPLE}")
    print("now: ./venv/bin/python tests/heldout_box_spotcheck.py serve --open")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def load() -> tuple[dict, dict]:
    if not SAMPLE.exists():
        raise SystemExit("no sample yet: run `build` first")
    sample = json.loads(SAMPLE.read_text())
    marks = json.loads(MARKS.read_text()) if MARKS.exists() else {}
    return sample, marks


def compute(sample: dict, marks: dict) -> dict:
    res = {"marked": len(marks), "of": len(sample["items"]), "by": {}}
    for source in SOURCES:
        rows = {}
        for scope in ("all",) + tuple(str(p) for p in PAGES):
            its = [it for it in sample["items"] if source in it["sources"] and scope in ("all", str(it["page"]))]
            v = [marks.get(str(it["id"])) for it in its]
            right, wrong, unsure = v.count("right"), v.count("wrong"), v.count("cannot_tell")
            n = right + wrong
            keys = [k for k in sample["summary"] if k.endswith("/" + source) and scope in ("all", k.split("/")[0])]
            words = sum(sample["summary"][k]["words"] for k in keys)
            shown = sum(sample["summary"][k]["boxes_shown"] for k in keys)
            lo, hi = wilson(right, n)
            rows[scope] = {"sampled": len(its), "right": right, "wrong": wrong, "cannot_tell": unsure,
                           "precision": round(right / n, 4) if n else None, "precision_95": [round(lo, 4), round(hi, 4)],
                           "words": words, "boxes_shown": shown, "coverage": round(shown / words, 4),
                           "located_and_correct_est": round(shown / words * right / n, 4) if n else None}
        res["by"][source] = rows
    return res


def cmd_report(_args) -> None:
    sample, marks = load()
    res = compute(sample, marks)
    print(f"Held-out box spot-check, {SLUG} p3-p5, rule {sample['rule']}, line assignment {sample['la']}, "
          f"code {sample['code']}: {res['marked']} of {res['of']} boxes marked")
    for source, rows in res["by"].items():
        print(f"\nline text: {source}")
        for scope, r in rows.items():
            prec = "-" if r["precision"] is None else (f"{100 * r['precision']:.1f}% "
                    f"(95% interval {100 * r['precision_95'][0]:.0f}-{100 * r['precision_95'][1]:.0f}%)")
            est = "-" if r["located_and_correct_est"] is None else f"{100 * r['located_and_correct_est']:.1f}%"
            print(f"  {scope:4s} coverage {r['boxes_shown']}/{r['words']} = {100 * r['coverage']:.1f}%   "
                  f"precision {r['right']}/{r['right'] + r['wrong']} = {prec}   "
                  f"cannot tell {r['cannot_tell']}   located and correct (estimate) {est}")
    if res["marked"] == res["of"]:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"summary": sample["summary"], "result": res, "code": sample["code"]}, indent=1))
        print(f"\nwrote {OUT}")
    else:
        print("\n(not complete: numbers are provisional and nothing was written to out/)")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

PAGE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Box spot-check</title>
<style>
body{font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0;background:#f4f2ee;color:#1d1d1f}
header{padding:10px 18px;background:#fff;border-bottom:1px solid #d8d4cc;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
main{padding:14px 18px;max-width:1500px;margin:auto}
.word{font:44px/1.3 "Vazirmatn","Geeza Pro",Tahoma,sans-serif;direction:rtl;text-align:center;margin:6px 0 10px}
svg{display:block;width:100%;background:#fff;border:1px solid #d8d4cc;border-radius:6px;margin-bottom:10px}
.btns{display:flex;gap:10px;justify-content:center;margin:12px 0}
button{font:inherit;padding:10px 22px;border-radius:8px;border:1px solid #9a958a;background:#fff;cursor:pointer}
button.on{outline:3px solid #1d1d1f}
#right{background:#dff3e2}#wrong{background:#f8dcd9}#unsure{background:#eee}
.rule{color:#555;max-width:900px;margin:0 auto 6px;text-align:center}
kbd{border:1px solid #aaa;border-radius:4px;padding:0 5px;background:#fff}
</style></head><body>
<header><b>Box spot-check, pages 3–5</b><span id="pos"></span><span id="done"></span>
<span>Keys: <kbd>R</kbd> right · <kbd>W</kbd> wrong · <kbd>C</kbd> cannot tell · <kbd>←</kbd> back · <kbd>→</kbd> next</span></header>
<main>
<p class="rule">Is the box on this word, with its left and right edges at the word's edges (within about half a letter)?
Height does not matter. If the text below is two words typed as one, the box should span both.</p>
<div class="word" id="word"></div>
<svg id="zoom" preserveAspectRatio="xMidYMid meet"></svg>
<svg id="strip" preserveAspectRatio="xMidYMid meet"></svg>
<div class="btns"><button id="right">Right (R)</button><button id="wrong">Wrong (W)</button>
<button id="unsure">Cannot tell (C)</button><button id="back">← Back</button><button id="next">Next →</button></div>
<p class="rule" id="finished"></p>
</main><script>
let S=null,i=0;
const $=id=>document.getElementById(id);
function view(svg,it,x0,y0,w,h){
  x0=Math.max(0,x0);y0=Math.max(0,y0);w=Math.min(w,it.W-x0);h=Math.min(h,it.H-y0);
  svg.setAttribute('viewBox',`${x0} ${y0} ${w} ${h}`);
  svg.style.aspectRatio=`${w} / ${h}`;
  const pts=it.quad.map(p=>p.join(',')).join(' ');
  svg.innerHTML=`<image href="/img/${it.page}.png" x="0" y="0" width="${it.W}" height="${it.H}"/>
   <polygon points="${pts}" fill="rgba(255,0,170,.10)" stroke="#e6007e" stroke-width="${Math.max(1.5,h/140)}"/>`;
}
function show(){
  const it=S.items[i],xs=it.quad.map(p=>p[0]),ys=it.quad.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),bh=Math.max(y1-y0,30),cx=(x0+x1)/2,cy=(y0+y1)/2;
  const zw=Math.max((x1-x0)+10*bh,16*bh),zh=3.2*bh;
  view($('zoom'),it,cx-zw/2,cy-zh/2,zw,zh);
  view($('strip'),it,0,cy-3.5*bh,it.W,7*bh);
  $('word').textContent=it.text;
  $('pos').textContent=`Box ${i+1} of ${S.items.length} · page ${it.page}`;
  const n=Object.keys(S.marks).length;$('done').textContent=`${n} marked`;
  const m=S.marks[it.id];
  $('right').className=m==='right'?'on':'';$('wrong').className=m==='wrong'?'on':'';$('unsure').className=m==='cannot_tell'?'on':'';
  $('finished').textContent=n===S.items.length?'All boxes are marked. You can close this page; tell Claude you are done.':'';
}
async function mark(v){
  const it=S.items[i];
  const r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:it.id,verdict:v})});
  if(!r.ok){$('finished').textContent='Save failed; is the server still running?';return}
  S.marks[it.id]=v;
  if(i<S.items.length-1)i++;show();
}
function go(d){i=Math.min(S.items.length-1,Math.max(0,i+d));show()}
$('right').onclick=()=>mark('right');$('wrong').onclick=()=>mark('wrong');$('unsure').onclick=()=>mark('cannot_tell');
$('back').onclick=()=>go(-1);$('next').onclick=()=>go(1);
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
  if(k==='r')mark('right');else if(k==='w')mark('wrong');else if(k==='c')mark('cannot_tell');
  else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1)});
fetch('/api/state').then(r=>r.json()).then(s=>{S=s;
  const first=S.items.findIndex(it=>!(it.id in S.marks));i=first<0?0:first;show()});
</script></body></html>"""


def make_handler(sample_path: Path, marks_path: Path, truth_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D401
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if ctype == "image/png" else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                sample = json.loads(sample_path.read_text())
                marks = json.loads(marks_path.read_text()) if marks_path.exists() else {}
                items = [{k: it[k] for k in ("id", "page", "W", "H", "text", "quad")} for it in sample["items"]]
                return self._send(200, json.dumps({"items": items, "marks": marks}, ensure_ascii=False).encode("utf-8"),
                                  "application/json")
            if path.startswith("/img/") and path.endswith(".png") and path[5:-4].isdigit() and int(path[5:-4]) in PAGES:
                return self._send(200, (truth_dir / f"{int(path[5:-4]):04d}.page.png").read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/mark":
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                ids = {it["id"] for it in json.loads(sample_path.read_text())["items"]}
                if req["id"] not in ids or req["verdict"] not in VERDICTS:
                    raise ValueError("bad mark")
            except Exception as exc:  # noqa: BLE001
                return self._send(400, str(exc).encode(), "text/plain")
            marks = json.loads(marks_path.read_text()) if marks_path.exists() else {}
            marks[str(req["id"])] = req["verdict"]
            _atomic(marks_path, marks)
            self._send(200, b"{}", "application/json")
    return Handler


def cmd_serve(args) -> None:
    load()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(SAMPLE, MARKS, TRUTH))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"box spot-check at {url} (Ctrl-C to stop)", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8772)
    s.add_argument("--open", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()
    {"build": cmd_build, "serve": cmd_serve, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
