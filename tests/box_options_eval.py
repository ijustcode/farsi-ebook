#!/usr/bin/env python3
"""Choose the box-placement engine for review: evaluation of
``docs/box-placement-eval-plan.md`` sections 1-4.

    ./venv/bin/python tests/box_options_eval.py selftest     # judge on constructed cases ($0)
    ./venv/bin/python tests/box_options_eval.py spans        # E1 targets, written once ($0)
    ./venv/bin/python tests/box_options_eval.py v0 [--jobs 6] # judge validated on the owner's Step 5 marks ($0)
    ./venv/bin/python tests/box_options_eval.py a1 [--jobs 4] # new engine, rules ST and TA (one paid p66 reading)
    ./venv/bin/python tests/box_options_eval.py a0           # current engine on p66, offline replay ($0)
    ./venv/bin/python tests/box_options_eval.py a2 smoke     # PAID (Google): synthetic page, two wordings
    ./venv/bin/python tests/box_options_eval.py a2 run       # PAID (Google): frozen wording, audited pages
    ./venv/bin/python tests/box_options_eval.py score        # E1 table and the decision rule ($0)
    ./venv/bin/python tests/box_options_eval.py status

Everything here was written before the run it judges (rule D25). Outputs go
to ``out/box_options/`` (gitignored; holds page text). hadpi-articles p3-p5
are held out and refused everywhere.

THE JUDGE (plan section 2, owner's standard of 2026-11-14, height ignored, D12).
Units: each truth line has a frame (centre, angle) and each truth word a
horizontal extent [u0, u1] along the line in that frame; a word's letter is
its width / its letters (``real_truth_eval.letters``), as in
``real_truth_eval.grade_target_h1``.

* A segment is ON the truth line whose band holds its vertical centre. The
  band of a line is its centre (middle of its outline across the line, in its
  frame) +/- half the page's median line spacing; the segment's centre goes to
  the nearest line centre, and to no line when it is farther than half a
  spacing from every line (off text, e.g. a stamp or the margin above the
  text). A truth line excluded from grading (owner text flag) is still a line:
  a segment there is on a line with no target word.
* NO_BOX: no segment.
* WRONG_LINE: some segment is on a line with no target word, or off text.
* MISS: on some line holding target words, no segment, or the segments' hull
  along the line does not reach the target's outer edges to within half a
  letter (left edge: letter of the leftmost target word; right edge: of the
  rightmost).
* Excess (CHOICE, the plan says "whole truth words"): on each target line and
  each side, the number of non-target words of that line beyond the target
  that the hull covers WHOLE, i.e. reaches that word's far edge to within half
  of its letter. A partly covered word does not count.
* WIDE: covers as above, but the excess exceeds two words on some side of
  some target line.
* RIGHT: covers, at most two whole extra words on each side of every target
  line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "out" / "box_options"
DEAD = "http://127.0.0.1:9"
PAGES = {"h1": ("hadpi-articles", 1), "h2": ("hadpi-articles", 2), "c23": ("ghessehaye-majid", 23),
         "c109": ("ghessehaye-majid", 109), "p66": ("bachehaye_ghali", 66)}
CLASS = {"h1": "hard", "h2": "hard", "c23": "clean", "c109": "clean", "p66": "old"}
CLASSES = ("hard", "clean", "old")
HELD_OUT = {("hadpi-articles", 3), ("hadpi-articles", 4), ("hadpi-articles", 5)}
LABELS = ("RIGHT", "WIDE", "MISS", "WRONG_LINE", "NO_BOX")

SPANS = OUT / "spans.json"
SPANS_PER_PAGE = 100
SPAN_SEED_TAG = "box-options-e1/2026-11-15"
CONTEXT_WORDS = 4

# A1 line text: cached gg_carbon band readings (run 0, and run 1 where it exists);
# p66 has none cached and gets one new reading (same request as arm gg_carbon).
A1_RUNS = {"h1": (0, 1), "h2": (0, 1), "c23": (0,), "c109": (0,), "p66": (0,)}
A1_RULES = ("ST", "TA")

# Spend caps (plan section 3), checked against the ledgers before every attempt.
CAP_SMOKE = 0.15
CAP_E1 = 0.60
LEDGER_SMOKE = OUT / "ledger_a2_smoke.jsonl"
LEDGER_E1 = OUT / "ledger_e1.jsonl"
CACHE = OUT / "cache"

# A2 (Gemini box_2d). Convention from Google's documentation (read 2026-11-15:
# https://aistudio.google.com/docs/image-understanding, object detection;
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection):
# box_2d = [ymin, xmin, ymax, xmax], integers normalised to 0-1000 relative to
# the image's height (y) and width (x), origin top left. Image resolution
# (https://aistudio.google.com/docs/media-resolution): per-item "resolution"
# with a token budget (ultra_high 2240 tokens on Gemini 3/4); whether pixels are
# kept at any level is not documented, so the hi-res page image is sent as is.
from farsi2epub.gemini_boxes import A2_MODEL, A2_RESOLUTION, A2_THINKING  # noqa: E402  (one copy, production)
A2_SMOKE_PAGE = "out/synth/typewriter_memo_tw_photo_medium_s11"   # closest synthetic match to hadpi
A2_SMOKE_SPANS = 40
A2_SMOKE_MAX_OUT = 6000
A2_SMOKE_PASS = {"right_min": 0.60, "wrong_line_max": 0.10}
A2_RUN_REPS = (0, 1)
A2_RUN_MAX_OUT = {50: 8000, 100: 12000}
# Batch size for `a2 run`, fixed before the smoke: 50 passages per call, unless
# the full run projected from the smoke's measured token usage per passage
# (better wording) would exceed the E1 cap left after the p66 reading; then 100
# (one call per page and run). Cost-only rule; quality plays no part.
A2_BATCH_DEFAULT, A2_BATCH_LARGE = 50, 100
A2_FREEZE = OUT / "a2_freeze.json"

from farsi2epub.gemini_boxes import A2_WORDING_W1  # noqa: E402
A2_WORDING_W2 = A2_WORDING_W1 + (
    "\nThe page is Persian, written right to left: within a line the first target word is the rightmost one, and a "
    "passage that wraps continues at the right end of the next line. Make each box tight around the ink of the "
    "target words on that line: its right edge at the start of the first target word on that line and its left "
    "edge at the end of the last one; do not extend a box to the end of the line unless the target words reach "
    "it. For each box also write the target words it contains, as printed.")
A2_WORDINGS = {"W1": A2_WORDING_W1, "W2": A2_WORDING_W2}


def refuse_held_out(slug: str, page: int) -> None:
    if (slug, int(page)) in HELD_OUT:
        raise SystemExit(f"{slug} p{page} is held out: refused")


for _s, _p in PAGES.values():
    refuse_held_out(_s, _p)


def _atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dead_api() -> None:
    os.environ["ANTHROPIC_BASE_URL"] = DEAD


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
def page_to_frame(frame: dict, x: float, y: float) -> tuple[float, float]:
    a = math.radians(frame["angle_deg"])
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - frame["cx"], y - frame["cy"]
    return dx * c + dy * s, -dx * s + dy * c


@dataclass
class JWord:
    key: object
    u0: float
    u1: float
    letters: float
    text: str = ""

    @property
    def letter(self) -> float:
        return (self.u1 - self.u0) / max(1.0, self.letters)


@dataclass
class JLine:
    index: int
    frame: dict
    vc: float                      # centre across the line, in the frame
    words: list = field(default_factory=list)


@dataclass
class JPage:
    W: float
    H: float
    lines: list
    spacing: float
    word_line: dict = field(default_factory=dict)   # word key -> line index

    def finish(self) -> "JPage":
        self.word_line = {w.key: L.index for L in self.lines for w in L.words}
        return self


def median_spacing(lines: list) -> float:
    ys = sorted(L.frame["cy"] + L.vc * math.cos(math.radians(L.frame["angle_deg"])) for L in lines)
    d = [b - a for a, b in zip(ys, ys[1:]) if b - a > 1e-6]
    return statistics.median(d) if d else 1e9


def poly_of(seg: dict, W: float, H: float) -> list:
    if "quad" in seg:
        return [(float(x) * W, float(y) * H) for x, y in seg["quad"]]
    x0, y0, x1, y1 = seg["rect"]
    return [(x0 * W, y0 * H), (x1 * W, y0 * H), (x1 * W, y1 * H), (x0 * W, y1 * H)]


def line_of(jp: JPage, poly: list):
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    best, bd = None, None
    for L in jp.lines:
        d = abs(page_to_frame(L.frame, cx, cy)[1] - L.vc)
        if bd is None or d < bd:
            best, bd = L, d
    if best is None or bd > jp.spacing / 2:
        return None
    return best


def judge(segments: list, target: list, jp: JPage) -> tuple[str, dict]:
    """segments: [{"quad": [[x,y]*4]} | {"rect": [x0,y0,x1,y1]}] normalised to the page;
    target: word keys of jp. Returns (label, detail)."""
    if not segments:
        return "NO_BOX", {}
    polys = [poly_of(s, jp.W, jp.H) for s in segments]
    tlines = {}
    for k in target:
        tlines.setdefault(jp.word_line[k], []).append(k)
    on = {li: [] for li in tlines}
    for poly in polys:
        L = line_of(jp, poly)
        if L is None:
            return "WRONG_LINE", {"why": "segment off text"}
        if L.index not in on:
            return "WRONG_LINE", {"why": f"segment on line {L.index} without target words"}
        on[L.index].append(poly)
    detail = {"lines": [], "box_w": 0.0, "target_w": 0.0}
    miss = wide = False
    tset = set(target)
    for li, keys in sorted(tlines.items()):
        L = jp.lines[li]
        tw = sorted([w for w in L.words if w.key in tset], key=lambda w: w.u0)
        t0, t1 = tw[0].u0, tw[-1].u1
        detail["target_w"] += t1 - t0
        if not on[li]:
            miss = True
            detail["lines"].append({"line": li, "segments": 0})
            continue
        us = [page_to_frame(L.frame, x, y)[0] for poly in on[li] for x, y in poly]
        a0, a1 = min(us), max(us)
        detail["box_w"] += a1 - a0
        el, er = (t0 - a0) / tw[0].letter, (a1 - t1) / tw[-1].letter      # > 0: box beyond the target
        covers = el >= -0.5 and er >= -0.5
        left = [w for w in L.words if w.key not in tset and (w.u0 + w.u1) / 2 < t0]
        right = [w for w in L.words if w.key not in tset and (w.u0 + w.u1) / 2 > t1]
        xl = sum(1 for w in left if a0 <= w.u0 + 0.5 * w.letter)
        xr = sum(1 for w in right if a1 >= w.u1 - 0.5 * w.letter)
        detail["lines"].append({"line": li, "edge_letters": [round(el, 2), round(er, 2)], "excess_words": [xl, xr]})
        if not covers:
            miss = True
        elif max(xl, xr) > 2:
            wide = True
    detail["width_ratio"] = round(detail["box_w"] / detail["target_w"], 3) if detail["target_w"] > 0 else None
    if miss:
        return "MISS", detail
    return ("WIDE" if wide else "RIGHT"), detail


def jpage_from_truth(tp) -> JPage:
    lines = []
    for L in tp.lines:
        words = [JWord((w.li, w.wi), w.fbox[0], w.fbox[2], w.letters, w.text) for w in L.words]
        lines.append(JLine(L.index, L.frame, (L.box[1] + L.box[3]) / 2, words))
    return JPage(tp.W, tp.H, lines, median_spacing(lines)).finish()


def frame_from_quad(quad_px: list) -> tuple[dict, float, float]:
    """Frame (centre, angle) of a line quad, any corner order; plus the quad's
    extent across the line (v0, v1) in that frame."""
    pts = sorted(quad_px, key=lambda p: p[0])
    l = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    r = ((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)
    ang = math.degrees(math.atan2(r[1] - l[1], r[0] - l[0]))
    fr = {"cx": sum(p[0] for p in quad_px) / 4, "cy": sum(p[1] for p in quad_px) / 4, "angle_deg": ang}
    vs = [page_to_frame(fr, x, y)[1] for x, y in quad_px]
    return fr, min(vs), max(vs)


def jpage_from_synth(truth: dict) -> JPage:
    import real_truth_eval as rt
    W, H = truth["image"]["width"], truth["image"]["height"]
    words = {w["id"]: w for w in truth["words"]}
    lines = []
    for L in truth["lines"]:
        bl = sorted(((x * W, y * H) for x, y in L["baseline"]), key=lambda p: p[0])
        ang = math.degrees(math.atan2(bl[-1][1] - bl[0][1], bl[-1][0] - bl[0][0]))
        fr = {"cx": sum(p[0] for p in bl) / len(bl), "cy": sum(p[1] for p in bl) / len(bl), "angle_deg": ang}
        jw, vcs = [], []
        for wid in L["word_ids"]:
            w = words[wid]
            x0, y0, x1, y1 = w["ink_aabb"]
            corners = [(x0 * W, y0 * H), (x1 * W, y0 * H), (x1 * W, y1 * H), (x0 * W, y1 * H)]
            fb = [page_to_frame(fr, x, y) for x, y in corners]
            jw.append(JWord(wid, min(p[0] for p in fb), max(p[0] for p in fb), rt.letters(w["text"]), w["text"]))
            vcs.append((min(p[1] for p in fb) + max(p[1] for p in fb)) / 2)
        lines.append(JLine(L["index"], fr, statistics.median(vcs) if vcs else 0.0, jw))
    return JPage(W, H, lines, median_spacing(lines)).finish()


# ---------------------------------------------------------------------------
# selftest: constructed cases
# ---------------------------------------------------------------------------
def _toy_page(angle: float = 0.0) -> JPage:
    """3 lines, 6 words each, word width 100 px with 4 letters (letter 25 px),
    gaps 20 px; lines 100 px apart."""
    lines = []
    for li in range(3):
        fr = {"cx": 400.0, "cy": 200.0 + 100 * li, "angle_deg": angle}
        ws = [JWord((li, wi), -360 + 120 * wi, -260 + 120 * wi, 4.0) for wi in range(6)]
        lines.append(JLine(li, fr, 0.0, ws))
    return JPage(1000.0, 1000.0, lines, 100.0).finish()


def _toy_seg(jp: JPage, li: int, u0: float, u1: float, dv: float = 0.0, h: float = 30.0) -> dict:
    fr = jp.lines[li].frame
    a = math.radians(fr["angle_deg"])
    c, s = math.cos(a), math.sin(a)

    def P(u, v):
        return ((fr["cx"] + u * c - v * s) / jp.W, (fr["cy"] + u * s + v * c) / jp.H)
    v0, v1 = dv - h / 2, dv + h / 2
    return {"quad": [P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)]}


def cmd_selftest(_a=None) -> None:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {name}: {got} (want {want})")
    for ang in (0.0, 3.0):
        jp = _toy_page(ang)
        w = lambda li, wi: jp.lines[li].words[wi]  # noqa: E731
        print(f"angle {ang}")
        T = [(1, 2), (1, 3)]
        t0, t1 = w(1, 2).u0, w(1, 3).u1
        check("exact box", judge([_toy_seg(jp, 1, t0, t1)], T, jp)[0], "RIGHT")
        check("0.4 letter short each side", judge([_toy_seg(jp, 1, t0 + 10, t1 - 10)], T, jp)[0], "RIGHT")
        check("0.6 letter short on the left", judge([_toy_seg(jp, 1, t0 + 15, t1)], T, jp)[0], "MISS")
        check("0.6 letter short on the right", judge([_toy_seg(jp, 1, t0, t1 - 15)], T, jp)[0], "MISS")
        check("+2 whole words each side", judge([_toy_seg(jp, 1, w(1, 0).u0, w(1, 5).u1)], T, jp)[0], "RIGHT")
        T1 = [(1, 3)]
        check("+3 whole words left", judge([_toy_seg(jp, 1, w(1, 0).u0, w(1, 3).u1)], T1, jp)[0], "WIDE")
        check("+2.5 words left (partial third)", judge([_toy_seg(jp, 1, w(1, 0).u0 + 50, w(1, 3).u1)], T1, jp)[0],
              "RIGHT")
        check("+3 words, third short by 0.4 letter", judge([_toy_seg(jp, 1, w(1, 0).u0 + 10, w(1, 3).u1)], T1, jp)[0],
              "WIDE")
        check("box on the line above", judge([_toy_seg(jp, 0, t0, t1)], T, jp)[0], "WRONG_LINE")
        check("extra segment on a wrong line", judge([_toy_seg(jp, 1, t0, t1), _toy_seg(jp, 2, t0, t1)], T, jp)[0],
              "WRONG_LINE")
        check("segment off text (far below)", judge([_toy_seg(jp, 2, t0, t1, dv=80)], [(2, 2)], jp)[0], "WRONG_LINE")
        check("centre 40 px off (inside band)", judge([_toy_seg(jp, 1, t0, t1, dv=40)], T, jp)[0], "RIGHT")
        check("tall box ignored (height)", judge([_toy_seg(jp, 1, t0, t1, h=180)], T, jp)[0], "RIGHT")
        T2 = [(0, 4), (0, 5), (1, 0), (1, 1)]
        s0 = _toy_seg(jp, 0, w(0, 4).u0, w(0, 5).u1)
        s1 = _toy_seg(jp, 1, w(1, 0).u0, w(1, 1).u1)
        check("wrapped target, both lines", judge([s0, s1], T2, jp)[0], "RIGHT")
        check("wrapped target, one line only", judge([s0], T2, jp)[0], "MISS")
        check("no box", judge([], T, jp)[0], "NO_BOX")
        check("two segments on one line, hull covers", judge([_toy_seg(jp, 1, t0, t0 + 50),
                                                             _toy_seg(jp, 1, t1 - 50, t1)], T, jp)[0], "RIGHT")
        check("rect form", judge([{"rect": [0.0, 0.0, 0.0, 0.0]}], T, jp)[0], "WRONG_LINE")
    print("selftest:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------
def span_lengths() -> list[int]:
    import word_placer_passage as PS
    return [k for k, w in PS.SPAN_WEIGHTS.items() for _ in range(w)]   # as step5_compare.cmd_smoke


def draw_spans(order: list, n: int, seed: int) -> list[list]:
    """step5_compare smoke generator: length from the review-like weights, start uniform."""
    lengths = span_lengths()
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.choice(lengths)
        i0 = rnd.randrange(0, max(1, len(order) - k))
        out.append(list(range(i0, min(len(order), i0 + k))))
    return out


def truth_page(pid: str):
    import real_truth_eval as rt
    slug, n = PAGES[pid]
    refuse_held_out(slug, n)
    return rt.load_truth(ROOT, slug, n)


def cmd_spans(_a) -> None:
    if SPANS.exists():
        raise SystemExit(f"{SPANS} exists: the spans are drawn once")
    import real_truth_eval as rt
    import word_placer_eval as E
    rows = []
    for pid, (slug, n) in PAGES.items():
        tp = truth_page(pid)
        allw = [(w.li, w.wi) for L in tp.lines for w in L.words]
        order = [(w.li, w.wi) for w in tp.graded_words() if rt._fold(w.text)]
        for k, ids in enumerate(draw_spans(order, SPANS_PER_PAGE, E.hash_seed(f"{slug}/{n}", SPAN_SEED_TAG))):
            keys = [order[i] for i in ids]
            a, b = allw.index(keys[0]), allw.index(keys[-1])
            rows.append({"sid": f"{pid}-{k:03d}", "pid": pid, "slug": slug, "page": n,
                         "words": [list(x) for x in keys], "text": " ".join(tp.word(*x).text for x in keys),
                         "before": " ".join(tp.word(*x).text for x in allw[max(0, a - CONTEXT_WORDS):a]),
                         "after": " ".join(tp.word(*x).text for x in allw[b + 1:b + 1 + CONTEXT_WORDS]),
                         "n_words": len(keys), "lines": sorted({x[0] for x in keys})})
        print(f"  {pid}: {SPANS_PER_PAGE} spans, words {len(order)} graded, "
              f"lengths {sorted(len(r['words']) for r in rows if r['pid'] == pid)[-5:]} (longest)")
    _atomic(SPANS, {"schema": "box_options_spans/1", "seed_tag": SPAN_SEED_TAG, "spans": rows})
    print(f"wrote {len(rows)} spans -> {SPANS}")


def spans() -> list[dict]:
    if not SPANS.exists():
        raise SystemExit("run `spans` first")
    return _load(SPANS)["spans"]


# ---------------------------------------------------------------------------
# V0: the judge against the owner's Step 5 marks ($0)
# ---------------------------------------------------------------------------
V0_DIR = OUT / "v0_pages"
V0_RESULT = OUT / "v0_result.json"
V0_PASS = {"agreement_min": 0.90, "not_right_called_right_max": 0.05}


def _v0_page(pid: str) -> str:
    _dead_api()
    out = V0_DIR / f"{pid}.json"
    if out.exists():
        return f"{pid} cached"
    import line_finder as LF
    import step5_compare as S5
    TE = S5._harness()
    p = next(x for x in S5.population()["pages"] if x["pid"] == pid)
    md = (ROOT / "books" / p["slug"] / "text" / f"{p['page']:04d}.md").read_text(encoding="utf-8")
    if S5._sha(md.encode("utf-8")) != p["md_sha256"]:
        raise SystemExit(f"{pid}: page text changed since the Step 5 freeze")
    lines = S5.reading(TE, TE.Runner(cap=S5.CAP_USD, ledger_name=S5.LEDGER), pid)
    lines = [l.split() for l in lines]
    gray = LF.load_gray(S5.hires(p["slug"], p["page"]))
    H, W = gray.shape
    failed = None
    try:
        arm, _info = S5.run_engine(f"{p['slug']}/{p['page']}", gray, lines)
    except Exception as exc:  # noqa: BLE001  as in step5_compare._build_page
        arm, failed = {}, f"{type(exc).__name__}: {exc}"
    keys, _ = S5.reading_keys(lines)
    words = {}
    for (li, ti), x in arm.items():
        q = x["segments"][0]["quad"] if x["status"] == "located" and x["segments"] else None
        lq = (x.get("_line") or [{}])[0].get("quad")
        words[f"{li}:{ti}"] = {"status": x["status"], "quad": q, "line_quad": lq, "text": lines[li][ti]}
    corr = {}
    for c in p["corrections"]:
        pl = S5.place_correction(arm, lines, md, c["span"], c["kind"], c["text"])
        if failed:
            pl = {"status": "unresolved", "segments": [], "tokens": []}
        corr[c["key"]] = {"status": pl["status"], "tokens": [list(keys[j]) for j in pl["tokens"]],
                          "quads": [[[round(a * W, 1), round(b * H, 1)] for a, b in sg["quad"]] for sg in pl["segments"]]}
    _atomic(out, {"pid": pid, "W": W, "H": H, "engine_failed": failed, "words": words, "corrections": corr})
    return f"{pid}{' ENGINE FAILED' if failed else ''} words {len(words)}"


def jpage_standin(page: dict) -> JPage:
    """Stand-in truth: the new engine's located words (S∧T) on their lines."""
    import real_truth_eval as rt
    W, H = page["W"], page["H"]
    by_line = {}
    for k, x in page["words"].items():
        li, ti = map(int, k.split(":"))
        by_line.setdefault(li, []).append((ti, x))
    lines, idx = [], {}
    for li in sorted(by_line):
        lq = next((x["line_quad"] for _, x in by_line[li] if x.get("line_quad")), None)
        if lq is None:
            continue
        fr, v0, v1 = frame_from_quad([(a * W, b * H) for a, b in lq])
        ws = []
        for ti, x in sorted(by_line[li]):
            if x["status"] == "located" and x["quad"]:
                us = [page_to_frame(fr, a * W, b * H)[0] for a, b in x["quad"]]
                ws.append(JWord((li, ti), min(us), max(us), rt.letters(x["text"]), x["text"]))
        idx[li] = len(lines)
        lines.append(JLine(len(lines), fr, (v0 + v1) / 2, ws))
    jp = JPage(W, H, lines, median_spacing(lines)).finish()
    return jp


def cmd_v0(a) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import step5_compare as S5
    pop = S5.population()
    V0_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p["pid"] for p in pop["pages"] if not (V0_DIR / f"{p['pid']}.json").exists()]
    if todo:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for msg in ex.map(_v0_page, todo):
                print("  " + msg, flush=True)
    sample = _load(S5.SAMPLE)["items"]
    marks = _load(S5.MARKS)
    pages = {p["pid"]: _load(V0_DIR / f"{p['pid']}.json") for p in pop["pages"]}
    jps = {pid: jpage_standin(pg) for pid, pg in pages.items()}
    # determinism: rebuilt new-engine boxes against the ones the owner marked
    same = diff = 0
    for it in sample:
        if it["engine"] != "new":
            continue
        q = pages[it["pid"]]["corrections"][it["key"]]["quads"]
        ok = len(q) == len(it["quads"]) and all(abs(x - y) <= 0.11 for qa, qb in zip(q, it["quads"])
                                                 for pa, pb in zip(qa, qb) for x, y in zip(pa, pb))
        same += ok
        diff += not ok
    print(f"rebuild check: {same} new-engine boxes identical to the marked ones, {diff} differ")
    owner_vals = ("right", "wide", "miss", "wrong_line", "cannot_tell")
    conf = {e: {j: {o: 0 for o in owner_vals} for j in LABELS} for e in ("current", "new")}
    rows, why_not = [], {}
    for it in sample:
        pg = pages[it["pid"]]
        c = pg["corrections"][it["key"]]
        toks = [tuple(t) for t in c["tokens"]]
        mark = marks.get(str(it["id"]))
        jp = jps[it["pid"]]
        reason = None
        if pg["engine_failed"]:
            reason = "engine failed on the page"
        elif not toks:
            reason = "correction has no reading words"
        elif not all(pg["words"].get(f"{li}:{ti}", {}).get("status") == "located" for li, ti in toks):
            reason = "not every target word located"
        elif any((li, ti) not in jp.word_line for li, ti in toks):
            reason = "target line has no line quad"
        if reason:
            why_not[reason] = why_not.get(reason, 0) + 1
            continue
        W, H = it["W"], it["H"]
        segs = [{"quad": [[x / W, y / H] for x, y in q]} for q in it["quads"]]
        lab, det = judge(segs, toks, jp)
        conf[it["engine"]][lab][mark] += 1
        rows.append({"id": it["id"], "pid": it["pid"], "key": it["key"], "engine": it["engine"],
                     "status": it["status"], "owner": mark, "judge": lab, "detail": det})
    print(f"\nV0: {len(rows)} of {len(sample)} marked boxes judged; not judged: {why_not}")
    for e in ("current", "new"):
        print(f"\n  {e} engine: judge label (rows) x owner mark (columns)")
        print("    " + " " * 11 + "".join(f"{o:>12s}" for o in owner_vals))
        for j in LABELS:
            print(f"    {j:11s}" + "".join(f"{conf[e][j][o]:12d}" for o in owner_vals))
    use = [r for r in rows if r["owner"] != "cannot_tell"]
    agree = sum((r["judge"] == "RIGHT") == (r["owner"] == "right") for r in use)
    notright = [r for r in use if r["owner"] != "right"]
    nr_right = sum(r["judge"] == "RIGHT" for r in notright)
    ag = agree / len(use) if use else 0.0
    sh = nr_right / len(notright) if notright else 0.0
    ok = ag >= V0_PASS["agreement_min"] and sh <= V0_PASS["not_right_called_right_max"]
    by_eng = {}
    for e in ("current", "new"):
        u = [r for r in use if r["engine"] == e]
        by_eng[e] = {"judged": len(u), "agree": sum((r["judge"] == "RIGHT") == (r["owner"] == "right") for r in u),
                     "owner_not_right": sum(r["owner"] != "right" for r in u),
                     "owner_not_right_judged_right": sum(r["owner"] != "right" and r["judge"] == "RIGHT" for r in u)}
    print(f"\n  agreement on right / not right: {agree}/{len(use)} = {ag:.1%} (owner 'cannot tell' left out)")
    print(f"  owner 'not right' judged RIGHT: {nr_right}/{len(notright)} = {sh:.1%}")
    print(f"  by engine: {by_eng}")
    print(f"  criteria {V0_PASS}: {'PASS' if ok else 'FAIL'}")
    _atomic(V0_RESULT, {"judged": len(rows), "boxes": len(sample), "not_judged": why_not,
                        "rebuild_identical": same, "rebuild_differ": diff, "confusion": conf,
                        "agreement": [agree, len(use)], "not_right_judged_right": [nr_right, len(notright)],
                        "by_engine": by_eng, "pass": ok, "criteria": V0_PASS, "rows": rows})
    print(f"wrote {V0_RESULT}")


# ---------------------------------------------------------------------------
# Google calls: cache, ledger, caps (A2 and the p66 reading)
# ---------------------------------------------------------------------------
class CapReached(RuntimeError):
    pass


def ledger_total(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(e.get("cost", 0.0) + e.get("uncertain_cost", 0.0)
               for e in (json.loads(l) for l in path.read_text().splitlines() if l.strip()))


def _ledger(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}, ensure_ascii=False) + "\n")


def body_key(body: dict, rep: int) -> str:
    """Cache identity: the body with every image replaced by its SHA-256, plus the run number."""
    def strip(node):
        if isinstance(node, dict):
            if node.get("type") == "image" and "data" in node:
                return {**{k: v for k, v in node.items() if k != "data"},
                        "sha256": hashlib.sha256(node["data"].encode("ascii")).hexdigest()}
            return {k: strip(v) for k, v in node.items()}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node
    return hashlib.sha256(json.dumps({"body": strip(body), "rep": rep}, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def worst_case(body: dict, image_tokens: int = 2240) -> float:
    from farsi2epub import config
    p = config.gemini_price()
    text = sum(len(it.get("text", "").encode("utf-8")) for it in body["input"] if it["type"] == "text")
    n_img = sum(1 for it in body["input"] if it["type"] == "image")
    inp = text // 2 + 200 + n_img * image_tokens
    return 1.10 * inp / 1e6 * p["in"] + body["generation_config"]["max_output_tokens"] / 1e6 * p["out"]


def google_call(body: dict, schema, *, rep: int, ledger: Path, cap: float, what: str, attempts: int = 2):
    """One cached request. Returns the cache record. Raises CapReached before an
    attempt that could pass the cap; RuntimeError when it failed."""
    from farsi2epub import config, hard_reader
    import transcription_experiments as TE
    key = body_key(body, rep)
    cp = CACHE / f"{key}.json"
    if cp.exists():
        rec = _load(cp)
        rec["cache_hit"] = True
        return rec
    api_key = hard_reader.gemini_key()
    if not api_key:
        raise SystemExit("no Gemini key in the environment or .env")
    price = config.gemini_price()
    last = "no attempt"
    for attempt in range(1, attempts + 1):
        wc = worst_case(body)
        tot = ledger_total(ledger)
        if tot + wc > cap:
            raise CapReached(f"ledger ${tot:.4f} + worst case ${wc:.4f} would pass cap ${cap:.2f} ({what})")
        t0 = time.time()
        try:
            status, data = hard_reader._gemini_post(body, api_key)
        except OSError as exc:
            last = f"transport {type(exc).__name__}"
            _ledger(ledger, {"what": what, "key": key, "rep": rep, "attempt": attempt, "status": last,
                             "cost": 0.0, "uncertain_cost": round(wc, 6)})
            continue
        if status >= 400:
            msg = TE.redact(json.dumps(data)[:300], api_key)
            last = f"http {status}: {msg}"
            pre = status in (400, 401, 403, 404, 413, 422, 429)
            _ledger(ledger, {"what": what, "key": key, "rep": rep, "attempt": attempt, "status": f"http:{status}",
                             "detail": msg, "cost": 0.0, "uncertain_cost": 0.0 if pre else round(wc, 6)})
            if status in (401, 403) or "billing" in msg.lower() or "quota" in msg.lower():
                raise SystemExit(f"Google refused access: {msg}")
            if status in (400, 404, 413, 422):
                break
            time.sleep(5)
            continue
        parsed, usage, problem = TE.GoogleAdapter.parse(data)
        u = data.get("usage") or {}
        cost = usage["input_tokens"] / 1e6 * price["in"] + usage["output_tokens"] / 1e6 * price["out"]
        _ledger(ledger, {"what": what, "key": key, "rep": rep, "attempt": attempt,
                         "status": "ok" if parsed is not None else f"unparsed:{problem}", **usage,
                         "raw_usage": {k: v for k, v in u.items() if isinstance(v, (int, float))},
                         "price": price, "cost": round(cost, 6), "uncertain_cost": 0.0,
                         "secs": round(time.time() - t0, 1)})
        if parsed is None:
            last = f"unparsed: {problem}"
            continue
        try:
            parsed = schema.model_validate(parsed).model_dump()
        except Exception as exc:  # noqa: BLE001
            last = f"schema: {str(exc)[:200]}"
            continue
        rec = {"what": what, "rep": rep, "model": body["model"], "parsed": parsed, "usage": usage, "cost": cost,
               "price": price, "raw_status": data.get("status")}
        CACHE.mkdir(parents=True, exist_ok=True)
        _atomic(cp, rec)
        rec["cache_hit"] = False
        return rec
    raise RuntimeError(f"{what}: failed: {last}")


# ---------------------------------------------------------------------------
# A1: the frozen new engine, rules ST and TA, Gemini line text
# ---------------------------------------------------------------------------
A1_DIR = OUT / "a1"
P66_READING = OUT / "p66_reading.json"


def p66_reading_request():
    import transcription_experiments as TE
    TE.PAGES["p66"] = PAGES["p66"]
    req = TE.build_request("gg_carbon", "p66", 0)       # arm gg_carbon exactly: same bands, prompt, schema
    _url, body = TE.GoogleAdapter.build(req)
    return TE, req, body


def p66_lines(call: bool) -> list[str] | None:
    TE, req, body = p66_reading_request()
    key = body_key(body, 0)
    if not (CACHE / f"{key}.json").exists() and not call:
        return None
    rec = google_call(body, TE.BandsReading, rep=0, ledger=LEDGER_E1, cap=CAP_E1, what="p66 gg_carbon reading")
    if not P66_READING.exists():
        _atomic(P66_READING, {"key": key, "usage": rec["usage"], "cost": rec["cost"], "cache_hit": rec["cache_hit"]})
    o = TE.output_of(rec, "gg_carbon")
    print(f"  p66 reading: hit={rec['cache_hit']} usage={rec['usage']} ${rec['cost']:.4f} lines={len(o['lines'])} "
          f"warnings={o.get('band_warnings')}")
    return [l for l in o["lines"] if l and l.strip()]


def a1_lines(pid: str, rep: int) -> list[list[str]]:
    if pid == "p66":
        ls = p66_lines(call=False)
        if ls is None:
            raise SystemExit("p66 reading missing")
    else:
        import word_placer_e2e as X
        ls = X.model_lines("gg_carbon", pid, rep)
    return [l.split() for l in ls]


def truth_md(tp) -> tuple[str, dict]:
    """Truth words as text, one line per printed line; char span per word key."""
    parts, spans_, pos = [], {}, 0
    for L in tp.lines:
        toks = []
        for w in L.words:
            spans_[(w.li, w.wi)] = (pos, pos + len(w.text))
            toks.append(w.text)
            pos += len(w.text) + 1
        if not toks:
            pos += 1
        parts.append(" ".join(toks))
    md = "\n".join(parts)
    for k, (s, e) in spans_.items():
        assert md[s:e] == tp.word(*k).text
    return md, spans_


def _a1_job(job: tuple[str, int]) -> str:
    _dead_api()
    pid, rep = job
    out = A1_DIR / f"{pid}_r{rep}.json"
    if out.exists():
        return f"{pid} r{rep} cached"
    import real_truth_eval as rt
    import step5_compare as S5
    import word_placer_eval as E
    slug, n = PAGES[pid]
    refuse_held_out(slug, n)
    page = E.real_page(slug, n)
    tp = page["truth_page"]
    lines = a1_lines(pid, rep)
    E.T_OPTS.update(vert="comp", la="T")               # as step5_compare.run_engine (frozen configuration)
    import word_placer as WP
    failed = None
    try:
        ev = WP.page_evidence(page["gray"])
        pg = dict(name=f"{slug}/{n}", gray=page["gray"], truth_page=tp, texts=lines, kind="step5")
        arms, info = E.run_T(pg, "C", ev=ev, texts=lines, src=None)
    except Exception as exc:  # noqa: BLE001
        arms, info, failed = {r: {} for r in A1_RULES}, {}, f"{type(exc).__name__}: {exc}"
    md, wspan = truth_md(tp)
    keys, _ = S5.reading_keys(lines)
    W, H = tp.W, tp.H
    res = {}
    for rule in A1_RULES:
        arm = arms.get(rule, {})
        # page letter unit from this rule's located words (for H3; no truth used)
        lets = []
        for (li, ti), x in arm.items():
            if x["status"] == "located" and x["segments"] and x.get("_line"):
                fr, _, _ = frame_from_quad([(a * W, b * H) for a, b in x["_line"][0]["quad"]])
                us = [page_to_frame(fr, a * W, b * H)[0] for a, b in x["segments"][0]["quad"]]
                lets.append((max(us) - min(us)) / max(1.0, rt.letters(lines[li][ti])))
        rows = {}
        for sp in spans():
            if sp["pid"] != pid:
                continue
            ks = [tuple(k) for k in sp["words"]]
            span = (wspan[ks[0]][0], wspan[ks[-1]][1])
            pl = S5.place_correction(arm, lines, md, span, "replacement", "")
            words = []
            for j in pl["tokens"]:
                x = arm.get(keys[j])
                words.append({"tok": list(keys[j]), "text": lines[keys[j][0]][keys[j][1]],
                              "status": x["status"] if x else None,
                              "tline": x.get("_tline") if x else None,
                              "quad": x["segments"][0]["quad"] if x and x["status"] == "located" and x["segments"] else None,
                              "line_quad": (x.get("_line") or [{}])[0].get("quad") if x else None})
            rows[sp["sid"]] = {"status": pl["status"], "segments": pl["segments"], "reason": pl.get("reason", ""),
                               "words": words}
        res[rule] = {"unit_px": statistics.median(lets) if lets else None,
                     "words_located": sum(1 for x in arm.values() if x["status"] == "located"), "spans": rows}
    _atomic(out, {"pid": pid, "rep": rep, "W": W, "H": H, "engine_failed": failed,
                  "text_lines": len(lines), "words": sum(len(l) for l in lines),
                  "la_template": info.get("la_template"), "g2_boxes_on_nontext": info.get("g2_boxes_on_nontext"),
                  "rules": res})
    st = {r: sum(v["status"] == "located" for v in res[r]["spans"].values()) for r in A1_RULES}
    return f"{pid} r{rep}{' ENGINE FAILED ' + failed if failed else ''} located spans {st}"


def cmd_a1(a) -> None:
    from concurrent.futures import ProcessPoolExecutor
    spans()
    if p66_lines(call=False) is None:
        print("p66 has no cached Gemini band reading: one paid call (arm gg_carbon request)")
        p66_lines(call=True)
    A1_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(pid, r) for pid, reps in A1_RUNS.items() for r in reps]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for msg in ex.map(_a1_job, jobs):
            print("  " + msg, flush=True)


# ---------------------------------------------------------------------------
# A0: the current engine on p66, offline replay
# ---------------------------------------------------------------------------
A0_PATH = OUT / "a0.json"


def cmd_a0(_a) -> None:
    if A0_PATH.exists():
        print(f"{A0_PATH} exists")
        return
    _dead_api()
    import shutil
    import tempfile
    import real_truth_eval as rt
    evidence = {}
    for pid, (slug, n) in PAGES.items():
        tp = truth_page(pid)
        tmp = Path(tempfile.mkdtemp(prefix="box_options_a0_"))
        try:
            pi = rt.build_page_input(ROOT, tp, tmp)
            eng = rt.EngineOffline()
            svc = eng._service(pi)
            try:
                svc.ensure_page_map(n, allow_api=False)
                sm = svc.page_map_summary(n) or {}
                spent = svc.budget.spent
            finally:
                svc.close()
            evidence[pid] = {k: sm.get(k) for k in ("status", "detected_lines", "read_lines", "verified_words",
                                                    "acquisition_reasons", "source")}
            evidence[pid]["spent"] = spent
            print(f"  evidence {pid}: {evidence[pid]}")
            if pid != "p66":
                continue
            targets = [rt.Target(sp["sid"], [tuple(k) for k in sp["words"]], sp["text"])
                       for sp in spans() if sp["pid"] == pid]
            qs = eng._queries(pi, targets, pi.md)
            placed = eng.place(pi, targets)
            rows = {}
            for tg, q, pl in zip(targets, qs, placed):
                rows[tg.tid] = {"status": pl["status"], "segments": pl["segments"], "reason": pl.get("reason", ""),
                                "query": None if q is None else {"text": q.text, "span": list(q.span)}}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if any(v["spent"] for v in evidence.values()):
        raise SystemExit("A0 replay spent money; refusing to write")
    cnt = {}
    for r in rows.values():
        cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    noq = sum(r["query"] is None for r in rows.values())
    print(f"  p66 A0: statuses {cnt}; spans without a faithful query (first/last word not 1:1 in text/0066.md): {noq}")
    _atomic(A0_PATH, {"query_construction": "real_truth_eval._EngineBase._queries: Query(md[a:b], (a, b), (), "
                      "'replacement') over the workspace Markdown, a and b from the 1:1 folded alignment of the "
                      "span's first and last truth word; no alternatives (a bare span has no replacement text)",
                      "evidence": evidence, "spans": rows})
    print(f"wrote {A0_PATH}")


# ---------------------------------------------------------------------------
# A2: Gemini box_2d
# ---------------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402


from farsi2epub.gemini_boxes import Answer1, Box1, Passage1, passage_list  # noqa: E402,F401


class Box2(BaseModel):
    words: str
    box_2d: list[int]


class Passage2(BaseModel):
    id: int
    boxes: list[Box2]


class Answer2(BaseModel):
    passages: list[Passage2]


SCHEMAS = {"W1": Answer1, "W2": Answer2}


def a2_body(wording: str, png: bytes, items: list[dict], max_out: int) -> dict:
    from farsi2epub import hard_reader
    body = hard_reader.gemini_items_body(A2_MODEL, [A2_WORDINGS[wording], (png, "image/png"), passage_list(items)],
                                         SCHEMAS[wording], max_out, thinking_level=A2_THINKING)
    assert body["input"][1]["resolution"] == A2_RESOLUTION
    return body


def boxes_to_segments(boxes: list[dict], order: str) -> list[dict]:
    segs = []
    for b in boxes:
        v = b.get("box_2d") or []
        if len(v) != 4:
            continue
        if order == "yxyx":
            y0, x0, y1, x1 = v
        else:
            x0, y0, x1, y1 = v
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        segs.append({"rect": [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]})
    return segs


def synth_spans() -> tuple[dict, JPage, list[dict]]:
    import real_truth_eval as rt
    import word_placer_eval as E
    truth = _load(ROOT / f"{A2_SMOKE_PAGE}.truth.json")
    jp = jpage_from_synth(truth)
    words = sorted(truth["words"], key=lambda w: w["reading_index"])
    order = [w["id"] for w in words if rt._fold(w["text"])]
    allw = [w["id"] for w in words]
    text = {w["id"]: w["text"] for w in words}
    items = []
    for k, ids in enumerate(draw_spans(order, A2_SMOKE_SPANS, E.hash_seed(A2_SMOKE_PAGE, "box-options-a2-smoke"))):
        ks = [order[i] for i in ids]
        a, b = allw.index(ks[0]), allw.index(ks[-1])
        items.append({"sid": f"s{k:02d}", "words": ks, "text": " ".join(text[x] for x in ks),
                      "before": " ".join(text[x] for x in allw[max(0, a - CONTEXT_WORDS):a]),
                      "after": " ".join(text[x] for x in allw[b + 1:b + 1 + CONTEXT_WORDS])})
    return truth, jp, items


def a2_answers(rec: dict, n: int) -> dict:
    """passage number (1-based) -> boxes; duplicates: first kept."""
    out = {}
    for p in rec["parsed"]["passages"]:
        if 1 <= p["id"] <= n and p["id"] not in out:
            out[p["id"]] = p["boxes"]
    return out


def smoke_score(rec: dict, jp: JPage, items: list[dict], order: str) -> dict:
    ans = a2_answers(rec, len(items))
    cnt = {l: 0 for l in LABELS}
    widths = []
    for k, it in enumerate(items, 1):
        lab, det = judge(boxes_to_segments(ans.get(k, []), order), it["words"], jp)
        cnt[lab] += 1
        if det.get("width_ratio") is not None and lab != "WRONG_LINE":
            widths.append(det["width_ratio"])
    n = len(items)
    return {"counts": cnt, "right_rate": cnt["RIGHT"] / n, "wrong_line_rate": cnt["WRONG_LINE"] / n,
            "median_width_ratio": statistics.median(widths) if widths else None,
            "answered": len(ans), "boxes": sum(len(v) for v in ans.values())}


def cmd_a2_smoke(_a) -> None:
    truth, jp, items = synth_spans()
    png = (ROOT / truth["image"]["path"]).read_bytes()
    res = {}
    for wd in ("W1", "W2"):
        body = a2_body(wd, png, items, A2_SMOKE_MAX_OUT)
        try:
            rec = google_call(body, SCHEMAS[wd], rep=0, ledger=LEDGER_SMOKE, cap=CAP_SMOKE, what=f"a2 smoke {wd}")
        except CapReached as exc:
            print(f"STOP: {exc}")
            break
        except RuntimeError as exc:
            print(f"{wd}: FAILED: {exc}")
            res[wd] = {"failed": str(exc)}
            continue
        res[wd] = {"usage": rec["usage"], "cost": rec["cost"], "cache_hit": rec["cache_hit"],
                   "documented_yxyx": smoke_score(rec, jp, items, "yxyx"),
                   "alternative_xyxy": smoke_score(rec, jp, items, "xyxy")}
        r = res[wd]
        print(f"{wd}: hit={rec['cache_hit']} usage={rec['usage']} ${rec['cost']:.4f}")
        for o in ("documented_yxyx", "alternative_xyxy"):
            s = r[o]
            print(f"   {o:17s} {s['counts']}  RIGHT {s['right_rate']:.1%}  WRONG_LINE {s['wrong_line_rate']:.1%}  "
                  f"median width/target {s['median_width_ratio']}  answered {s['answered']}/{len(items)} boxes {s['boxes']}")
    ok = {wd: r for wd, r in res.items() if "failed" not in r}
    if not ok:
        print("no wording answered")
        return
    # convention: the ordering that scores more RIGHT across both wordings
    conv = max(("documented_yxyx", "alternative_xyxy"),
               key=lambda o: (sum(r[o]["counts"]["RIGHT"] for r in ok.values()), o == "documented_yxyx"))
    best = max(ok, key=lambda wd: (ok[wd][conv]["counts"]["RIGHT"], -ok[wd][conv]["counts"]["WRONG_LINE"], wd == "W1"))
    s = ok[best][conv]
    viable = s["right_rate"] >= A2_SMOKE_PASS["right_min"] and s["wrong_line_rate"] <= A2_SMOKE_PASS["wrong_line_max"]
    # batch size rule (fixed before the smoke): cost only
    u = ok[best]["usage"]
    per_call_in = u["input_tokens"]
    per_passage_out = u["output_tokens"] / len(items)
    from farsi2epub import config
    pr = config.gemini_price()
    left = CAP_E1 - ledger_total(LEDGER_E1)
    calls = lambda b: len(PAGES) * len(A2_RUN_REPS) * math.ceil(SPANS_PER_PAGE / b)  # noqa: E731
    proj = lambda b: calls(b) * (per_call_in / 1e6 * pr["in"]) + len(PAGES) * len(A2_RUN_REPS) * SPANS_PER_PAGE * \
        per_passage_out / 1e6 * pr["out"]  # noqa: E731
    batch = A2_BATCH_DEFAULT if proj(A2_BATCH_DEFAULT) <= left else A2_BATCH_LARGE
    freeze = {"wording": best, "convention": conv, "smoke": res, "viable": viable, "criteria": A2_SMOKE_PASS,
              "batch": batch, "projected_run_cost": {str(b): round(proj(b), 4) for b in (A2_BATCH_DEFAULT, A2_BATCH_LARGE)},
              "e1_cap_left_at_freeze": round(left, 4),
              "reps": list(A2_RUN_REPS) if viable else [0], "page": A2_SMOKE_PAGE, "spans": len(items)}
    if A2_FREEZE.exists():
        old = _load(A2_FREEZE)
        if (old["wording"], old["convention"], old["batch"]) != (best, conv, batch):
            raise SystemExit(f"{A2_FREEZE} exists with a different choice; refusing to overwrite")
        print(f"(freeze already written: {A2_FREEZE})")
    else:
        _atomic(A2_FREEZE, freeze)
    print(f"convention read: {conv}; better wording {best}: RIGHT {s['right_rate']:.1%}, WRONG_LINE "
          f"{s['wrong_line_rate']:.1%}; criteria {A2_SMOKE_PASS}: {'PASS (viable)' if viable else 'FAIL (not viable)'}")
    print(f"batch for the run: {batch} passages per call (projected {freeze['projected_run_cost']}, E1 cap left ${left:.4f}); "
          f"reps {freeze['reps']} -> {A2_FREEZE}")


A2_DIR = OUT / "a2"


def cmd_a2_run(_a) -> None:
    if not A2_FREEZE.exists():
        raise SystemExit("run `a2 smoke` first (no wording frozen)")
    fz = _load(A2_FREEZE)
    wd, batch, reps = fz["wording"], fz["batch"], fz["reps"]
    sp = spans()
    stop = None
    for rep in reps:
        for pid, (slug, n) in PAGES.items():
            refuse_held_out(slug, n)
            out = A2_DIR / f"{pid}_r{rep}.json"
            if out.exists():
                print(f"  {pid} r{rep}: done")
                continue
            png = (ROOT / "books" / slug / "pages" / "hires" / f"{n:04d}.png").read_bytes()
            items = [s for s in sp if s["pid"] == pid]
            rows, calls = {}, []
            for b0 in range(0, len(items), batch):
                chunk = items[b0:b0 + batch]
                body = a2_body(wd, png, chunk, A2_RUN_MAX_OUT[batch])
                try:
                    rec = google_call(body, SCHEMAS[wd], rep=rep, ledger=LEDGER_E1, cap=CAP_E1,
                                      what=f"a2 run {pid} r{rep} b{b0 // batch}")
                except CapReached as exc:
                    stop = str(exc)
                    break
                except RuntimeError as exc:
                    calls.append({"batch": b0 // batch, "failed": str(exc)})
                    for it in chunk:
                        rows[it["sid"]] = {"boxes": [], "failed": True}
                    print(f"  {pid} r{rep} b{b0 // batch}: FAILED {exc}")
                    continue
                ans = a2_answers(rec, len(chunk))
                for k, it in enumerate(chunk, 1):
                    rows[it["sid"]] = {"boxes": ans.get(k, []), "answered": k in ans}
                calls.append({"batch": b0 // batch, "usage": rec["usage"], "cost": rec["cost"],
                              "cache_hit": rec["cache_hit"], "answered": len(ans)})
                print(f"  {pid} r{rep} b{b0 // batch}: hit={rec['cache_hit']} usage={rec['usage']} "
                      f"${rec['cost']:.4f} answered {len(ans)}/{len(chunk)}  E1 ledger ${ledger_total(LEDGER_E1):.4f}",
                      flush=True)
            if stop:
                break
            _atomic(out, {"pid": pid, "rep": rep, "wording": wd, "convention": fz["convention"], "batch": batch,
                          "calls": calls, "spans": rows})
        if stop:
            break
    if stop:
        print(f"STOP (cap): {stop}")
    print(f"E1 ledger total ${ledger_total(LEDGER_E1):.4f} of ${CAP_E1:.2f}")


def cmd_a2(a) -> None:
    {"smoke": cmd_a2_smoke, "run": cmd_a2_run}[a.what](a)


# ---------------------------------------------------------------------------
# score: the E1 table and the decision rule (plan section 4)
# ---------------------------------------------------------------------------
E1_RESULT = OUT / "e1_result.json"
PAID_CALLS = {"A0": "Anthropic line/word-group readings (several)", "A1": 1, "A2": 1}


def _a2_segments(row: dict, conv: str) -> list:
    return boxes_to_segments(row.get("boxes") or [], conv)


def _seg_centre(seg: dict, W: float, H: float) -> tuple[float, float]:
    poly = poly_of(seg, W, H)
    return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)


def _on_quad(pt, quad_px) -> tuple[bool, dict]:
    fr, v0, v1 = frame_from_quad(quad_px)
    v = page_to_frame(fr, *pt)[1]
    return v0 <= v <= v1, fr


def h2_ok(a1row: dict, a2segs: list, W: float, H: float) -> bool:
    lq = []
    for w in a1row["words"]:
        q = w.get("line_quad")
        if q and q not in lq:
            lq.append(q)
    if not lq or not a2segs:
        return False
    for s in a2segs:
        c = _seg_centre(s, W, H)
        if not any(_on_quad(c, [(a * W, b * H) for a, b in q])[0] for q in lq):
            return False
    return True


def h3_agree(a1row: dict, a2segs: list, unit: float | None, W: float, H: float) -> bool:
    if a1row["status"] != "located" or not a2segs or not unit:
        return False
    groups = {}
    for w in a1row["words"]:
        if w["quad"] is None or w.get("line_quad") is None:
            return False
        groups.setdefault(json.dumps(w["line_quad"]), []).append(w["quad"])
    frames = {g: _on_quad((0, 0), [(a * W, b * H) for a, b in json.loads(g)])[1] for g in groups}
    a2by = {g: [] for g in groups}
    for s in a2segs:
        c = _seg_centre(s, W, H)
        hit = [g for g in groups if _on_quad(c, [(a * W, b * H) for a, b in json.loads(g)])[0]]
        if not hit:
            return False
        a2by[hit[0]].append(poly_of(s, W, H))
    for g, quads in groups.items():
        fr = frames[g]
        if not a2by[g]:
            return False
        ua = [page_to_frame(fr, a * W, b * H)[0] for q in quads for a, b in q]
        ub = [page_to_frame(fr, x, y)[0] for poly in a2by[g] for x, y in poly]
        if abs(min(ua) - min(ub)) > 0.5 * unit or abs(max(ua) - max(ub)) > 0.5 * unit:
            return False
    return True


def candidate_outputs():
    """{candidate name: {(pid, sid): (segments, status)}} per available run combination."""
    sp = spans()
    fz = _load(A2_FREEZE) if A2_FREEZE.exists() else None
    a1 = {}
    for pid, reps in A1_RUNS.items():
        for r in reps:
            f = A1_DIR / f"{pid}_r{r}.json"
            if f.exists():
                a1[(pid, r)] = _load(f)
    a2 = {}
    for pid in PAGES:
        for r in A2_RUN_REPS:
            f = A2_DIR / f"{pid}_r{r}.json"
            if f.exists():
                a2[(pid, r)] = _load(f)
    a0 = _load(A0_PATH)["spans"] if A0_PATH.exists() else None
    conv = {"documented_yxyx": "yxyx", "alternative_xyxy": "xyxy"}[fz["convention"]] if fz else None
    W = {pid: truth_page(pid).W for pid in PAGES}
    Hh = {pid: truth_page(pid).H for pid in PAGES}
    cands = {}      # name -> run label -> {sid: (segs, status)}

    def put(name, run, sid, segs, status):
        cands.setdefault(name, {}).setdefault(run, {})[sid] = (segs, status)
    a2runs = sorted({r for (_, r) in a2})
    for s in sp:
        pid, sid = s["pid"], s["sid"]
        w_, h_ = W[pid], Hh[pid]
        if a0 is not None and pid == "p66":
            r0 = a0[sid]
            put("A0", "-", sid, r0["segments"], r0["status"])
        for r in a2runs:
            if (pid, r) in a2:
                row = a2[(pid, r)]["spans"].get(sid, {"boxes": []})
                segs = _a2_segments(row, conv)
                put("A2", f"a2r{r}", sid, segs, "box" if segs else "none")
        for rule in A1_RULES:
            for r1 in A1_RUNS[pid]:
                if (pid, r1) not in a1:
                    continue
                d = a1[(pid, r1)]
                row = d["rules"][rule]["spans"][sid]
                unit = d["rules"][rule]["unit_px"]
                run1 = f"a1r{r1}"
                put(f"A1-{rule}", run1, sid, row["segments"], row["status"])
                loc = row["status"] == "located"
                if a0 is not None and pid == "p66":
                    r0 = a0[sid]
                    put(f"H0-{rule}", run1, sid, row["segments"] if loc else r0["segments"],
                        "a1_located" if loc else "a0:" + r0["status"])
                for r in a2runs:
                    if (pid, r) not in a2:
                        continue
                    segs2 = _a2_segments(a2[(pid, r)]["spans"].get(sid, {"boxes": []}), conv)
                    run = f"{run1}+a2r{r}"
                    put(f"H1-{rule}", run, sid, row["segments"] if loc else segs2,
                        "a1_located" if loc else ("a2" if segs2 else "none"))
                    if loc:
                        put(f"H2-{rule}", run, sid, row["segments"], "a1_located")
                    elif h2_ok(row, segs2, w_, h_):
                        put(f"H2-{rule}", run, sid, segs2, "a2_on_a1_line")
                    else:
                        put(f"H2-{rule}", run, sid, row["segments"] if row["status"] == "line" else [],
                            "a1_line" if row["status"] == "line" else "none")
                    if h3_agree(row, segs2, unit, w_, h_):
                        put(f"H3-{rule}", run, sid, segs2, "a2_agrees")
                    else:
                        put(f"H3-{rule}", run, sid, row["segments"], "a1:" + row["status"])
    return cands


def tally(rows: list[tuple[str, str, dict]]) -> dict:
    """rows: (label, status, detail)."""
    n = len(rows)
    cnt = {l: sum(1 for r in rows if r[0] == l) for l in LABELS}
    loc = [r for r in rows if r[1] in ("located", "a1_located")]
    widths = [r[2].get("width_ratio") for r in rows if r[0] in ("RIGHT", "WIDE", "MISS") and r[2].get("width_ratio")]
    return {"n": n, **cnt, "right_rate": cnt["RIGHT"] / n if n else None,
            "covered": cnt["RIGHT"] + cnt["WIDE"], "covered_rate": (cnt["RIGHT"] + cnt["WIDE"]) / n if n else None,
            "wl_miss_rate": (cnt["WRONG_LINE"] + cnt["MISS"]) / n if n else None,
            "located": len(loc), "located_right": sum(1 for r in loc if r[0] == "RIGHT"),
            "located_covered": sum(1 for r in loc if r[0] in ("RIGHT", "WIDE")),
            "median_width_ratio": round(statistics.median(widths), 3) if widths else None}


def decide(table: dict, metric: str) -> dict:
    """Plan section 4 applied mechanically. metric 'right' (RIGHT) or 'covered'
    (RIGHT + WIDE, the plan's reading of E1 when V0 fails)."""
    M = metric.upper()
    rk, lk = f"{metric}_rate", f"located_{metric}"
    count = (lambda t: t["RIGHT"]) if metric == "right" else (lambda t: t["covered"])
    out = {"metric": metric}
    print(f"\nDECISION RULE (plan section 4) on {M}")
    rule_info = {}
    for rule in A1_RULES:
        runs = table.get(f"A1-{rule}", {})
        c0 = sum(count(runs["a1r0"][c]) for c in CLASSES if c in runs.get("a1r0", {}))
        ok, precs = True, {}
        for run, e in runs.items():
            for c in CLASSES:
                if c in e:
                    t = e[c]
                    precs[f"{run}/{c}"] = [t[lk], t["located"]]
                    if not t["located"] or t[lk] / t["located"] < 0.97:
                        ok = False
        rule_info[rule] = {"count_run0": c0, "located_by_class": precs, "meets_97": ok}
        print(f"  A1-{rule}: {M} (run 0, all 500 spans) {c0}; located {M}/located {precs}; >= 97% in every class "
              f"and run: {ok}")
    er = [r for r in A1_RULES if rule_info[r]["meets_97"]]
    if not er:
        chosen, why = "ST", "neither rule meets 97% in every class: S∧T by the plan"
    else:
        chosen = max(er, key=lambda r: (rule_info[r]["count_run0"], r == "ST"))
        why = f"rules meeting 97%: {er}; the higher {M} count"
    print(f"  located rule chosen: {chosen} ({why})")
    out["located_rule"] = {"chosen": chosen, "why": why, "info": rule_info}

    def worst(name):
        res = {}
        for c in CLASSES:
            vals = [e[c] for e in table.get(name, {}).values() if c in e]
            if vals:
                res[c] = {"rate": min(v[rk] for v in vals), "wl_miss_rate": max(v["wl_miss_rate"] for v in vals),
                          "median_width_ratio": max((v["median_width_ratio"] or 0) for v in vals), "runs": len(vals)}
        return res
    names = ["A2", f"A1-{chosen}", f"H1-{chosen}", f"H2-{chosen}", f"H3-{chosen}"]
    rows = {}
    print(f"  per class, lower run: {M} rate / WRONG_LINE+MISS rate / median width ratio (widest run)")
    for nm in names + ["A0", f"H0-{chosen}"]:
        w = worst(nm)
        if not w:
            continue
        rows[nm] = w
        print(f"   {nm:8s} " + "  ".join(f"{c}: {w[c]['rate']:.1%} / {w[c]['wl_miss_rate']:.1%} / "
                                         f"{w[c]['median_width_ratio']} ({w[c]['runs']} run(s))" for c in CLASSES if c in w))
    out["by_class_lower_run"] = rows
    complete = {nm: w for nm, w in rows.items() if all(c in w for c in CLASSES)}
    print("  A0 and H0 are measured on p66 only (no current-engine reader evidence on the other pages): not in the "
          "three-class mean; they enter step 3.")
    elig = {nm: w for nm, w in complete.items() if all(w[c]["wl_miss_rate"] <= 0.03 for c in CLASSES)}
    for nm, w in complete.items():
        print(f"   trust floor {nm:8s}: {'eligible' if nm in elig else 'NOT eligible'} "
              f"(worst class WRONG_LINE+MISS {max(w[c]['wl_miss_rate'] for c in CLASSES):.1%})")
    calls = {"A2": 1, f"A1-{chosen}": 1, f"H1-{chosen}": 2, f"H2-{chosen}": 2, f"H3-{chosen}": 2}
    winner = None
    if elig:
        mean = {nm: sum(w[c]["rate"] for c in CLASSES) / 3 for nm, w in elig.items()}
        best = max(mean.values())
        tied = [nm for nm in mean if best - mean[nm] <= 0.02 + 1e-12]
        winner = min(tied, key=lambda nm: (calls[nm], -mean[nm]))   # none needs an Anthropic call
        for nm in sorted(mean, key=mean.get, reverse=True):
            print(f"   mean {M} {nm:8s} {mean[nm]:.1%}{'  (within 2 points of the best)' if nm in tied else ''}")
        print(f"  WINNER: {winner} (tie rule: fewer paid calls per page, then no Anthropic call; tied {tied})")
        out["mean"], out["tied"] = mean, tied
    else:
        print("  no candidate is eligible: A0 stays; the best candidate is shown in the demo as an option (plan 4.4)")
    out["eligible"], out["winner"] = sorted(elig), winner
    todo = []
    a0r = rows.get("A0", {}).get("old", {}).get("rate")
    if winner:
        p66w = rows[winner]["old"]["rate"]
        if a0r is None:
            todo.append("condition 1: A0 on p66 not measured")
        else:
            c1 = p66w >= a0r
            print(f"  step 3, condition 1 (at least as good as A0 where A0 is measured, p66): winner {p66w:.1%} vs "
                  f"A0 {a0r:.1%}: {'met' if c1 else 'NOT met'}")
            out["step3_cond1"] = {"winner_p66": p66w, "a0_p66": a0r, "met": c1}
        if winner == "A2" or winner.startswith(("H1", "H2", "H3")):
            todo.append("condition 2: E2 (A2 boxes for the corrections of 20 seeded Step 5 pages, judged by the V0 "
                        "stand-in truth, against the owner's A0 marks on the same corrections)")
        else:
            todo.append("condition 2 for an A1-only winner: Step 5 marks (A1 161 vs A0 170 right) already lose; the "
                        "plan names no further test for it")
    h0 = rows.get(f"H0-{chosen}", {}).get("old", {}).get("rate")
    if h0 is not None and a0r is not None:
        print(f"  p66: H0-{chosen} {h0:.1%} vs A0 {a0r:.1%} (H0's real-correction evidence is the post-hoc 181 vs 170)")
        out["p66_h0_vs_a0"] = [h0, a0r]
    for t in todo:
        print(f"  still to check: {t}")
    out["still_to_check"] = todo
    return out


def cmd_score(_a) -> None:
    sp = spans()
    jps = {pid: jpage_from_truth(truth_page(pid)) for pid in PAGES}
    cands = candidate_outputs()
    judged = {}      # name -> run -> sid -> (label, status, detail)
    for name, runs in cands.items():
        for run, rows in runs.items():
            for s in sp:
                if s["sid"] not in rows:
                    continue
                segs, st = rows[s["sid"]]
                lab, det = judge(segs, [tuple(k) for k in s["words"]], jps[s["pid"]])
                judged.setdefault(name, {}).setdefault(run, {})[s["sid"]] = (lab, st, det)
    pid_of = {s["sid"]: s["pid"] for s in sp}
    table = {}
    print("E1 (100 spans per page; class hard = h1+h2, clean = c23+c109, old print = p66)")
    print("columns: counts; RIGHT rate; covered (RIGHT+WIDE) rate; WRONG_LINE+MISS rate; located boxes and how many "
          "RIGHT / covered; median box width / target width\n")
    hdr = f"{'candidate':9s} {'run':14s} {'scope':5s} {'n':>4s} {'RIGHT':>5s} {'WIDE':>5s} {'MISS':>5s} {'WL':>4s} " \
          f"{'NOBOX':>5s} {'RIGHT%':>7s} {'COV%':>6s} {'WL+M%':>6s} {'loc':>4s} {'locR':>4s} {'locC':>4s} {'width':>6s}"
    print(hdr)
    for name in sorted(judged):
        for run in sorted(judged[name]):
            res = judged[name][run]
            entry = {}
            scopes = [(pid, [pid]) for pid in PAGES] + [(c, [p for p in PAGES if CLASS[p] == c]) for c in CLASSES]
            for scope, pids in scopes:
                rows = [v for sid, v in res.items() if pid_of[sid] in pids]
                # a run counts for a scope only when it covers every span of every page in it
                if not rows or len(rows) != SPANS_PER_PAGE * len(pids):
                    continue
                t = tally(rows)
                entry[scope] = t
                print(f"{name:9s} {run:14s} {scope:5s} {t['n']:4d} {t['RIGHT']:5d} {t['WIDE']:5d} {t['MISS']:5d} "
                      f"{t['WRONG_LINE']:4d} {t['NO_BOX']:5d} {t['right_rate']:7.1%} {t['covered_rate']:6.1%} "
                      f"{t['wl_miss_rate']:6.1%} {t['located']:4d} {t['located_right']:4d} {t['located_covered']:4d} "
                      f"{str(t['median_width_ratio']):>6s}")
            table.setdefault(name, {})[run] = entry
        print()
    v0 = _load(V0_RESULT) if V0_RESULT.exists() else None
    v0_pass = bool(v0 and v0["pass"])
    print(f"V0: {'PASS' if v0_pass else 'FAIL' if v0 else 'not run'}: E1 is read on "
          f"{'RIGHT' if v0_pass else 'covered (RIGHT+WIDE) and not wrong line, plus box width (plan section 2)'}")
    primary = decide(table, "right" if v0_pass else "covered")
    secondary = None if v0_pass else decide(table, "right")
    if secondary is not None:
        print("  (the RIGHT-based outcome above is for reference only: V0 failed)")
    _atomic(E1_RESULT, {"table": table, "v0_pass": v0_pass, "primary": primary, "reference_right": secondary,
                        "labels": {name: {run: {sid: v[0] for sid, v in r.items()} for run, r in runs.items()}
                                   for name, runs in judged.items()}})
    print(f"\nwrote {E1_RESULT}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def cmd_status(_a) -> None:
    print(f"spans: {'yes' if SPANS.exists() else 'no'}")
    print(f"v0 pages cached: {len(list(V0_DIR.glob('*.json'))) if V0_DIR.exists() else 0}; result: {V0_RESULT.exists()}")
    print(f"p66 reading cached: {P66_READING.exists()}")
    print(f"a1 runs: {sorted(p.stem for p in A1_DIR.glob('*.json')) if A1_DIR.exists() else []}")
    print(f"a0: {A0_PATH.exists()}")
    print(f"a2 freeze: {_load(A2_FREEZE)['wording'] + ' ' + _load(A2_FREEZE)['convention'] if A2_FREEZE.exists() else 'no'}")
    print(f"a2 runs: {sorted(p.stem for p in A2_DIR.glob('*.json')) if A2_DIR.exists() else []}")
    print(f"cache files: {len(list(CACHE.glob('*.json'))) if CACHE.exists() else 0}")
    for name, path, cap in (("synthetic smoke", LEDGER_SMOKE, CAP_SMOKE), ("E1", LEDGER_E1, CAP_E1)):
        ent = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
        print(f"ledger {name}: {len(ent)} attempts, ${ledger_total(path):.4f} of cap ${cap:.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("spans")
    v = sub.add_parser("v0")
    v.add_argument("--jobs", type=int, default=6)
    a1 = sub.add_parser("a1")
    a1.add_argument("--jobs", type=int, default=4)
    sub.add_parser("a0")
    a2 = sub.add_parser("a2")
    a2.add_argument("what", choices=("smoke", "run"))
    sub.add_parser("score")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    {"selftest": cmd_selftest, "spans": cmd_spans, "v0": cmd_v0, "a1": cmd_a1, "a0": cmd_a0, "a2": cmd_a2,
     "score": cmd_score, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
