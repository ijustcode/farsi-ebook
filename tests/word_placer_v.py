#!/usr/bin/env python3
"""Arm V on line crops (plan §4.3 control arm, Step 4 continuation).

The model never judges a box. It is shown ONE straightened, enlarged printed
line and the known word list of that line, and asked where the words are;
Python grades. Two formulations:

* **V-mark** (``mark``): arm G's candidate cut positions (every gap between
  core ink runs, plus G's width-placed cuts inside fused runs, plus one just
  outside each end of the line) are drawn as small numbered ticks in a white
  margin ABOVE the text (never over ink), numbered right to left. The model
  names, for every listed word, the mark at its right edge and at its left
  edge ("R-L"), "R-?" / "?-L" when an edge has no visible boundary (touching
  words), or "absent" when the word is not printed on the line. Ink outside
  every word's span is not text (stray marks).
* **V-coord** (``coord``): a pixel ruler (ticks every 10 px, labels every
  100 px) in the top margin; the model gives each word's right and left x.

Every answer becomes an interval along the line, and boxes are snapped to the
line's own ink exactly as arm G's are (``word_placer.snap_words``).

Statuses (owner decision D1): ``V`` located when the answer is complete and
consistent; ``VG`` located only when V and arm G's best segmentation agree on
both edges within half a letter (independent sources); ``H`` (hybrid) keeps G
where G is located and asks V only for lines with a word G did not locate.

API use goes through the committed ``tests/transcription_experiments.py``
``Runner`` (disk cache keyed by request, ledger with a hard-cap guard, one
retry, stop on credit/access errors) with this module's own output directory
and cap. Only line crops of the frozen truth pages and synthetic pages are
ever built; held-out pages are refused by the truth loader.
"""
from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

import word_placer as WP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "word_placer" / "api"
CAP_USD = 1.00
PLAN_LIMIT_USD = 0.85

MODEL_SONNET = "claude-sonnet-5"
MODEL_OPUS = "claude-opus-5-5"
OPUS_EFFORT = "low"          # thinking cannot be disabled on Opus 5.5; effort is the control

VP = dict(
    MAX_W=2400,              # crop width cap (px): under 2576 so the API does not downsize
    SCALE_MAX=2.0,           # enlarge at most this much
    SCALE_HC=56.0,           # ... aiming at a glyph height of this many crop px
    PAD_U=0.6,               # crop extends this x hc beyond the line's ink on each side
    PAD_V=0.35,              # ... and this x hc above/below its 1-99% ink rows
    MARGIN=58,               # white margin on top for marks / ruler (crop px)
    LABEL_SEP=17,            # labels closer than this (crop px) go to another row
    LABEL_ROWS=3,
    MARK_MIN_SEP=5,          # candidate cuts closer than this (crop px) are merged (keep the wider gap)
    BATCH=6,                 # lines per request
    AGREE_TOL=0.5,           # VG: edges agree within this x letter unit
)


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------
@dataclass
class Crop:
    img: np.ndarray          # uint8 H x W, the straightened line with the white margin on top
    scale: float
    u_lo: float
    v_lo: float
    uref: float
    margin: int
    marks: list              # [(number, s position, kind)] for V-mark; [] for coord
    text_top: int            # first crop row of the text area

    def s_to_x(self, s):
        return (self.uref - np.asarray(s, float) - self.u_lo) * self.scale

    def x_to_s(self, x):
        return self.uref - (np.asarray(x, float) / self.scale + self.u_lo)


def _bilinear(gray, X, Y, fill=255.0):
    H, W = gray.shape
    x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0
    out = np.zeros_like(X, dtype=np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            out += w * np.where(ok, gray[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)], fill)
    return out


def line_strip(gray: np.ndarray, li: WP.LineInk, hc: float) -> Crop:
    """Straightened, enlarged strip of one line (any geometry: L1 shear or
    an audited frame), white margin on top."""
    if not len(li.u):
        raise ValueError("line has no ink")
    u_lo = float(li.u.min()) - VP["PAD_U"] * hc
    u_hi = float(li.u.max()) + VP["PAD_U"] * hc
    v_lo = float(np.quantile(li.v, 0.01)) - VP["PAD_V"] * hc
    v_hi = float(np.quantile(li.v, 0.99)) + VP["PAD_V"] * hc
    scale = min(VP["SCALE_MAX"], VP["SCALE_HC"] / max(1.0, hc), VP["MAX_W"] / max(1.0, u_hi - u_lo))
    scale = max(scale, 0.5)
    Wc = int(math.ceil((u_hi - u_lo) * scale))
    Hc = int(math.ceil((v_hi - v_lo) * scale))
    uu = u_lo + (np.arange(Wc) + 0.5) / scale
    vv = v_lo + (np.arange(Hc) + 0.5) / scale
    U, V = np.meshgrid(uu, vv)
    X, Y = li.geom.xy(U, V)
    strip = _bilinear(np.asarray(gray, np.float64), X, Y)
    m = VP["MARGIN"]
    img = np.full((Hc + m, Wc), 255, np.uint8)
    img[m:] = np.clip(strip, 0, 255).astype(np.uint8)
    return Crop(img, scale, u_lo, v_lo, li.uref, m, [], m)


def candidate_marks(li: WP.LineInk, al, crop: Crop) -> list:
    """Candidate cuts in reading coordinate s: line ends, every gap between
    core runs (mid-gap), and G's width-placed cuts inside fused runs.
    Marks closer than MARK_MIN_SEP crop px are merged, keeping the wider gap.
    Numbered 1.. from right to left (reading order)."""
    cand = []
    if li.M:
        cand.append((float(li.s0[0]) - 2.0, 99.0, "end"))
        for j in range(li.M - 1):
            g = float(li.s0[j + 1] - li.s1[j])
            cand.append(((float(li.s1[j]) + float(li.s0[j + 1])) / 2, g, "gap"))
        cand.append((float(li.s1[-1]) + 2.0, 99.0, "end"))
    for c in (getattr(al, "fused_cuts", None) or []):
        cand.append((float(c), 0.0, "fused"))
    cand.sort()
    sep = VP["MARK_MIN_SEP"] / crop.scale
    kept = []
    for c in cand:
        if kept and c[0] - kept[-1][0] < sep:
            if c[1] > kept[-1][1]:
                kept[-1] = c
            continue
        kept.append(c)
    return [(i + 1, s, k) for i, (s, g, k) in enumerate(kept)]


def render_png(crop: Crop, mode: str) -> bytes:
    """Draw marks (mode 'mark') or a ruler (mode 'coord') in the top margin,
    never over the text area. Returns PNG bytes (greyscale + blue overlay)."""
    import fitz
    H, W = crop.img.shape
    doc = fitz.open()
    pg = doc.new_page(width=W, height=H)
    pm = fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(crop.img).tobytes(), False)
    pg.insert_image(pg.rect, pixmap=pm)
    blue = (0.0, 0.15, 0.85)
    m = crop.margin
    if mode == "mark":
        rows_last = [-1e9] * VP["LABEL_ROWS"]
        for num, s, kind in crop.marks:
            x = float(crop.s_to_x(s))
            lab = str(num)
            wlab = 6.2 * len(lab) + 2
            row = next((r for r in range(VP["LABEL_ROWS"]) if x - wlab / 2 - rows_last[r] >= 3), None)
            if row is None:
                row = int(np.argmin(rows_last))
            rows_last[row] = x + wlab / 2
            y_lab = 12 + 15 * row
            col = (0.8, 0.0, 0.0) if kind == "fused" else blue
            pg.draw_line(fitz.Point(x, y_lab + 2), fitz.Point(x, m - 2), color=col, width=1.0)
            pg.insert_text(fitz.Point(x - wlab / 2 + 1, y_lab), lab, fontsize=11, color=col)
    else:
        for x in range(0, W, 10):
            big = x % 100 == 0
            mid = x % 50 == 0
            pg.draw_line(fitz.Point(x + 0.5, m - (16 if big else 10 if mid else 5)), fitz.Point(x + 0.5, m - 1),
                         color=blue, width=1.0 if big else 0.6)
            if big:
                pg.insert_text(fitz.Point(x + 2, 14), str(x), fontsize=11, color=blue)
    pix = pg.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    data = pix.tobytes("png")
    doc.close()
    return data


def text_area_untouched(png: bytes, crop: Crop) -> bool:
    """True when the rendered overlay left the text area exactly as cropped."""
    import fitz
    pix = fitz.Pixmap(png)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(int)
    g = crop.img[crop.margin:].astype(int)
    t = a[crop.margin:]
    return bool(np.all(np.abs(t - g[..., None]) <= 1))


# ---------------------------------------------------------------------------
# Prompts, schema, parsing
# ---------------------------------------------------------------------------
class VLine(BaseModel):
    line: int
    words: list[str]


class VAnswer(BaseModel):
    lines: list[VLine]


SYSTEM_COMMON = """You locate printed words in images. Each image shows ONE printed line of Persian text (written right to left), straightened and enlarged, with a white margin on top that carries {aid}.

For each line you are given the words that should be printed on it, numbered in reading order (right to left). The list was transcribed by a person or a machine and MAY CONTAIN ERRORS: a word may be misspelled, or not printed on this line at all. Persian words often contain internal gaps (after letters such as ا د ذ ر ز ژ و), so one word can span several separate pieces of ink; neighbouring words can also touch with almost no space between them. Ink that belongs to no listed word (a stray dot or speck, a fragment of a stamp or signature, a mark in the margin) must be left outside every word.

{task}

Answer with one entry per image, "line" = the line number given with that image, and "words" = one string per listed word, in the listed order. Look closely at the letters; do not divide the line by proportions."""

MARK_AID = "small numbered tick marks (blue; red ones are guesses inside touching ink). The marks are candidate cut positions between pieces of ink, numbered from RIGHT to LEFT"
MARK_TASK = """For every listed word give the mark at its RIGHT edge (where the word begins) and the mark at its LEFT edge (where it ends) as "R-L", for example "4-9". Adjacent words normally share a mark: the left mark of one word is the right mark of the next. If one edge of a word has no suitable mark (it touches its neighbour), write "?" for that edge, e.g. "4-?". If a listed word is not printed on this line, write "absent"."""
COORD_AID = "a pixel ruler: tick labels are x positions in pixels from the LEFT edge of the image"
COORD_TASK = """For every listed word give the x position (pixels, read off the ruler) of its RIGHT ink edge and of its LEFT ink edge as "R-L", for example "1840-1712" (R is larger than L). If an edge cannot be seen because the word touches its neighbour, give your best reading of where one word ends and the next begins. If a listed word is not printed on this line, write "absent"."""


def system_prompt(mode: str) -> str:
    return SYSTEM_COMMON.format(aid=MARK_AID if mode == "mark" else COORD_AID,
                                task=MARK_TASK if mode == "mark" else COORD_TASK)


def line_prompt(k: int, tokens: list[str], crop: Crop, mode: str) -> str:
    words = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tokens))
    extra = f" (marks 1-{len(crop.marks)})" if mode == "mark" else f" (image width {crop.img.shape[1]} px)"
    return f"Line {k}{extra}. Listed words ({len(tokens)}):\n{words}"


_PAIR = re.compile(r"^\s*(\?|\d+)\s*[-–—:]\s*(\?|\d+)\s*$")


def parse_word(ans: str) -> tuple:
    """'R-L' -> (R or None, L or None, present). 'absent' -> (None, None, False).
    Anything unparseable -> (None, None, None)."""
    a = (ans or "").strip().lower()
    if a.startswith("absent") or a in ("none", "missing", "not present"):
        return None, None, False
    m = _PAIR.match(a)
    if not m:
        return None, None, None
    r = None if m.group(1) == "?" else int(m.group(1))
    l_ = None if m.group(2) == "?" else int(m.group(2))
    return r, l_, True


def answer_intervals(answers: list[str], crop: Crop, mode: str, n_words: int) -> list:
    """Per listed word: dict(interval=(s_a, s_b) or None, present, why)."""
    out = []
    marks = {n: s for n, s, _ in crop.marks}
    for i in range(n_words):
        a = answers[i] if i < len(answers) else ""
        r, l_, pres = parse_word(a)
        if pres is False:
            out.append(dict(interval=None, present=False, why="model: word not on this line"))
            continue
        if pres is None:
            out.append(dict(interval=None, present=None, why="unparseable answer"))
            continue
        if r is None or l_ is None:
            out.append(dict(interval=None, present=True, why="model: edge without a visible boundary"))
            continue
        if mode == "mark":
            if r not in marks or l_ not in marks:
                out.append(dict(interval=None, present=True, why="mark number not on the image"))
                continue
            sa, sb = marks[r], marks[l_]
        else:
            if not (0 <= l_ <= crop.img.shape[1] and 0 <= r <= crop.img.shape[1]):
                out.append(dict(interval=None, present=True, why="x outside the image"))
                continue
            sa, sb = float(crop.x_to_s(r)), float(crop.x_to_s(l_))
        if sb <= sa:
            out.append(dict(interval=None, present=True, why="right edge not right of left edge"))
            continue
        out.append(dict(interval=(sa, sb), present=True, why=""))
    # words whose interval overlaps a neighbour's by more than a mark spacing are inconsistent
    for i in range(len(out) - 1):
        a, b = out[i]["interval"], out[i + 1]["interval"]
        if a and b and b[0] < a[1] - 1.0:
            out[i]["why"] = out[i + 1]["why"] = "overlaps its neighbour"
            out[i]["interval"] = out[i + 1]["interval"] = None
    return out


# ---------------------------------------------------------------------------
# Snapping and status rules
# ---------------------------------------------------------------------------
def snapped(li: WP.LineInk, intervals: list, sw: float, unit: float, hc: float):
    quads = WP.snap_words(li, intervals, sw, unit, hc)
    return quads


def quad_s_extent(li: WP.LineInk, quad):
    if quad is None:
        return None
    xs = np.array([p[0] for p in quad]); ys = np.array([p[1] for p in quad])
    u, _ = li.geom.uv(xs, ys)
    return float(li.uref - u.max()), float(li.uref - u.min())


def v_results(li, tokens, ans_iv, sw, unit, hc, line_quad):
    """Arm V statuses for one line from parsed answer intervals."""
    quads = snapped(li, [a["interval"] for a in ans_iv], sw, unit, hc)
    res = []
    for a, q in zip(ans_iv, quads):
        if a["interval"] is not None and q is not None:
            res.append(dict(status="located", quad=q, line_quad=line_quad, margin=None, z=None, reason="V"))
        else:
            res.append(dict(status="line", quad=q, line_quad=line_quad, margin=None, z=None,
                            reason=f"V: {a['why'] or 'no ink in the answered span'}"))
    return res


def vg_results(li, v_res, g_res, unit):
    """Located only where V and G agree on both edges within AGREE_TOL letter."""
    out = []
    tol = VP["AGREE_TOL"] * unit
    for v, g in zip(v_res, g_res):
        ok = False
        if v["status"] == "located" and g.get("quad") is not None:
            ev, eg = quad_s_extent(li, v["quad"]), quad_s_extent(li, g["quad"])
            ok = abs(ev[0] - eg[0]) <= tol and abs(ev[1] - eg[1]) <= tol
        if ok:
            out.append(dict(v, reason="V and G agree"))
        elif v["status"] == "located" or g.get("quad") is not None:
            out.append(dict(status="line", quad=v.get("quad") or g.get("quad"), line_quad=v.get("line_quad") or g.get("line_quad"),
                            margin=None, z=None, reason="V and G disagree or one is missing"))
        else:
            out.append(dict(v))
    return out


def route_line(g_res_line: list) -> bool:
    """Hybrid routing rule: ask V for a line when any of its words is not
    located by G (G already carries the margin threshold in its status)."""
    return any(r["status"] != "located" for r in g_res_line)


def hybrid_results(g_res_line, v_res_line):
    if v_res_line is None:
        return [dict(r) for r in g_res_line]
    return [dict(g) if g["status"] == "located" else dict(v) for g, v in zip(g_res_line, v_res_line)]


# ---------------------------------------------------------------------------
# Requests and the plan
# ---------------------------------------------------------------------------
def te():
    import transcription_experiments as TE
    return TE


def runner(cap=CAP_USD, client_factory=None, out=OUT):
    TE = te()
    return TE.Runner(out=out, client_factory=client_factory, cap=cap)


def model_kwargs(model: str) -> dict:
    return {"extra": {"output_config": {"effort": OPUS_EFFORT}}} if model == MODEL_OPUS else {}


def make_image(png: bytes, w: int, h: int, label: str):
    TE = te()
    return TE.Image(png, "image/png", w, h, label)


def build_requests(page_label: str, lines: list, mode: str, model: str, arm: str, rep: int = 0,
                   batch: int | None = None) -> list:
    """lines: [(line_no, tokens, crop, png)] -> [(line_nos, Request)]."""
    TE = te()
    batch = batch or VP["BATCH"]
    out = []
    sysp = system_prompt(mode)
    for k in range(0, len(lines), batch):
        grp = lines[k:k + batch]
        content = []
        for ln, toks, crop, png in grp:
            content.append(line_prompt(ln, toks, crop, mode))
            content.append(make_image(png, crop.img.shape[1], crop.img.shape[0], f"{page_label}:l{ln}"))
        n_words = sum(len(t) for _, t, _, _ in grp)
        max_tokens = 6000 if model == MODEL_OPUS else 2000
        req = TE.Request(arm=arm, page=page_label, model=model, system=sysp, content=content, schema=VAnswer,
                         max_tokens=max_tokens, rep=rep, expected_out=40 + 8 * n_words + (900 if model == MODEL_OPUS else 0),
                         **model_kwargs(model))
        out.append(([ln for ln, _, _, _ in grp], req))
    return out


def parse_response(rec: dict) -> dict:
    """line number -> list of word answer strings."""
    out = {}
    for L in (rec.get("parsed") or {}).get("lines", []):
        out[int(L["line"])] = list(L.get("words") or [])
    return out
