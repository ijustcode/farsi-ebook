#!/usr/bin/env python3
"""Phase 2 smoke test (docs/line-finder-integration-plan.md 4.2): band rows cut
between detected lines (`hard_reader.line_rows`) against today's fixed bands.

Offline, $0, no API. Images are the hard reader's own band source
(`hard_reader.band_source`: the embedded photo at native size when finer than
the hi-res render), so rows are in that image's pixel coordinates.

(a) Five frozen truth pages (hadpi p1-p2, majid p23/p109, bachehaye p66) against
    the audited line outlines (`books/<slug>/truth/NNNN.geometry.audit.json`,
    loaded by tests/real_truth_eval.py and scaled to the band source image), for
    lines_per_band 1, 3, "same" and for the fixed bands.
(b) Whole-book robustness on every non-blank page (sidecar is_blank false) of
    bachehaye_ghali, ghessehaye-majid and jameeshenasi_khodemani (a third
    scanned printed book with a source.pdf), plus hadpi-articles p1-p2 only.

hadpi-articles p3-p5 are the held-out test: refused here, never opened.

Thresholds are the plan's, fixed before any run. Prints PASS/FAIL per
criterion, writes out/line_finder/line_rows_smoke.json, exits 1 on any FAIL.

  ./venv/bin/python tests/line_rows_smoke.py [--part a|b|all] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from farsi2epub import config, hard_reader as HR, line_finder as LF, render  # noqa: E402

OUT = ROOT / "out" / "line_finder" / "line_rows_smoke.json"
LPB = (1, 3, "same")
ARMS = ("fixed",) + tuple(f"lines_{v}" for v in LPB)
HELD_OUT = {("hadpi-articles", 3), ("hadpi-articles", 4), ("hadpi-articles", 5)}
BOOKS = (("bachehaye_ghali", None), ("ghessehaye-majid", None), ("jameeshenasi_khodemani", None),
         ("hadpi-articles", (1, 2)))
FALLBACK_MAX = 0.10       # plan 4.2: FAIL above 10% of non-blank prose pages in any book
SECONDS_MEDIAN_MAX = 10.0  # plan 4.2: FAIL above 10 s median
_NOT_PROSE = HR._UNSUPPORTED  # verse " --- ", table "|", code fence (same test the hard reader uses)


def refuse(slug: str, page: int) -> None:
    if (slug, int(page)) in HELD_OUT:
        raise SystemExit(f"refusing {slug} p{page}: held-out test page")


def source_gray(slug: str, page: int) -> tuple[np.ndarray, str]:
    """Grey band-source image of one page (same image band_images cuts)."""
    import fitz
    refuse(slug, page)
    ws = ROOT / "books" / slug
    pdf = ws / "source.pdf"
    hp = ws / "pages" / "hires" / f"{page:04d}.png"
    hires = hp.read_bytes() if hp.is_file() else render.render_page(pdf, page, config.LONG_EDGE_HI)
    pix, kind = HR.band_source(pdf, page, hires)
    g = pix if pix.n == 1 else fitz.Pixmap(fitz.csGRAY, pix)
    return np.frombuffer(g.samples, np.uint8).reshape(g.height, g.width).astype(np.float32), kind


def truth_to_source(slug: str, page: int, kind: str, truth_w: int, src_w: int, src_h: int):
    """Map truth-image px (frame: the full PDF page) to band-source px. For an
    embedded photo the photo's own placement matrix is inverted (the photo
    need not cover the page, and may be slightly rotated)."""
    import fitz
    doc = fitz.open(str(ROOT / "books" / slug / "source.pdf"))
    pg = doc[page - 1]
    k = pg.rect.width / truth_w                       # truth px -> pt
    if kind == "hires_render":
        s = src_w / pg.rect.width
        return lambda x, y: (x * k * s, y * k * s)
    best = max((i for i in pg.get_image_info(xrefs=True) if i.get("xref") and i["bbox"][2] > i["bbox"][0]),
               key=lambda i: i["width"] * i["height"])     # band_source's choice
    assert (best["width"], best["height"]) == (src_w, src_h), "band source is not the largest image"
    inv = ~fitz.Matrix(best["transform"])
    def f(x, y):
        p = fitz.Point(x * k, y * k) * inv              # unit square of the image
        return p.x * src_w, p.y * src_h
    return f


def cover_ok(rows, H) -> bool:
    if not rows or rows[0][0] != 0 or rows[-1][1] != H:
        return False
    if any(not (0 <= a < b <= H) for a, b in rows):
        return False
    return all(rows[i - 1][0] < rows[i][0] <= rows[i - 1][1] and rows[i - 1][1] < rows[i][1]
               for i in range(1, len(rows)))


def oversize(rows, W) -> int:
    return sum(not HR._band_fits(b - a, W) for a, b in rows)


class CachedFinder:
    """Runs the real finder once per page (timed); later calls reuse the result."""

    def __init__(self):
        self.res = self.exc = None
        self.seconds = None

    def __call__(self, g):
        if self.res is None and self.exc is None:
            t0 = time.perf_counter()
            try:
                self.res = LF.find_lines(g)
            except Exception as exc:          # noqa: BLE001 - recorded, then re-raised to line_rows
                self.exc = exc
            self.seconds = time.perf_counter() - t0
        if self.exc is not None:
            raise self.exc
        return self.res


def arm_rows(arm: str, g: np.ndarray, finder) -> tuple[list, dict]:
    H, W = g.shape
    if arm == "fixed":
        return HR.band_rows(H, W), {"cut": "fixed", "reason": "baseline"}
    v = arm.split("_", 1)[1]
    return HR.line_rows(g, int(v) if v.isdigit() else v, finder=finder)


# ---------------------------------------------------------------------------
# (a) truth pages
# ---------------------------------------------------------------------------
def part_a() -> dict:
    import real_truth_eval as rt
    pages, agg = {}, {a: dict(pages=0, cover_ok=0, lines=0, whole=0, crossed=0, crossed_not_whole=0,
                              oversize=0, deterministic=0, bands=0, fallback=0) for a in ARMS}
    for slug, page in rt.frozen_keys(ROOT):
        refuse(slug, page)
        t = rt.load_truth(ROOT, slug, page)
        g, kind = source_gray(slug, page)
        H, W = g.shape
        to_src = truth_to_source(slug, page, kind, t.W, W, H)
        spans = []
        for L in t.lines:
            ys = [to_src(*p)[1] for p in L.quad]
            spans.append((L.index, min(ys), max(ys), L.excluded))
        finder = CachedFinder()
        prow = dict(source=kind, size=[W, H], truth_size=[t.W, t.H], truth_lines=len(spans),
                    truth_y_range=[round(min(s[1] for s in spans), 1), round(max(s[2] for s in spans), 1)], arms={})
        for arm in ARMS:
            rows, info = arm_rows(arm, g, finder)
            rows2, _ = arm_rows(arm, g, CachedFinder() if arm != "fixed" else None)
            cuts = sorted({y for a, b in rows for y in (a, b) if 0 < y < H})
            crossed, cnw, not_whole = [], [], []
            for idx, top, bot, _ex in spans:
                whole = any(a <= top and bot <= b for a, b in rows)
                if not whole:
                    not_whole.append(idx)
                if any(top < y < bot for y in cuts):
                    crossed.append(idx)
                    if not whole:
                        cnw.append(idx)
            # diagnostic only (no criterion): cut rows holding more than 5 dark pixels (< 0.6 x median grey)
            dark = 0.6 * float(np.median(g))
            ink_rows = sum(int((g[y] < dark).sum()) > 5 for y in cuts)
            r = dict(cut=info.get("cut"), reason=info.get("reason"), bands=len(rows), rows=[list(x) for x in rows],
                     cuts=len(cuts), cut_rows_with_ink=ink_rows,
                     cover_ok=cover_ok(rows, H), crossed=crossed, crossed_not_whole=cnw, not_whole=not_whole,
                     oversize=oversize(rows, W), deterministic=rows == rows2,
                     text_lines=info.get("text_lines"), gaps=info.get("gaps"),
                     lines_per_band=info.get("lines_per_band_resolved"))
            prow["arms"][arm] = r
            a = agg[arm]
            a["pages"] += 1
            a["cover_ok"] += r["cover_ok"]
            a["lines"] += len(spans)
            a["whole"] += len(spans) - len(not_whole)
            a["crossed"] += len(crossed)
            a["crossed_not_whole"] += len(cnw)
            a["oversize"] += r["oversize"]
            a["deterministic"] += r["deterministic"]
            a["bands"] += len(rows)
            a["fallback"] += arm != "fixed" and r["cut"] != "lines"
        prow["finder_seconds"] = round(finder.seconds, 2) if finder.seconds is not None else None
        # mapping sanity check: each truth line's centre row lies inside some found text line's band
        if finder.res is not None:
            ext = [(min(q[1] for q in L.polygon()), max(q[1] for q in L.polygon()))
                   for L in finder.res["lines"] if L.kind == "text"]
            prow["truth_centres_in_found_line"] = sum(any(a <= (top + bot) / 2 <= b for a, b in ext)
                                                      for _, top, bot, _ in spans)
        pages[f"{slug}/p{page}"] = prow
    return dict(pages=pages, summary=agg)


def print_a(rep) -> list[tuple[str, bool]]:
    print("\n(a) Truth pages (audited line outlines, scaled to the band source image)")
    for key, p in rep["pages"].items():
        print(f"  {key:26s} {p['source']:15s} {p['size'][0]}x{p['size'][1]}  truth lines {p['truth_lines']}"
              f"  finder {p['finder_seconds']}s  truth y {p['truth_y_range']}"
              f"  truth centres inside a found line {p.get('truth_centres_in_found_line')}/{p['truth_lines']}")
        for arm, r in p["arms"].items():
            print(f"    {arm:11s} cut={r['cut']:5s} bands={r['bands']:2d} lpb={r['lines_per_band']}"
                  f" text_lines={r['text_lines']} gaps={r['gaps']} cover={'ok' if r['cover_ok'] else 'NO'}"
                  f" crossed={len(r['crossed'])} crossed_not_whole={len(r['crossed_not_whole'])}"
                  f" not_whole={len(r['not_whole'])} oversize={r['oversize']} det={'yes' if r['deterministic'] else 'NO'}"
                  f" cut_rows_with_ink={r['cut_rows_with_ink']}/{r['cuts']}"
                  + (f" reason={r['reason']}" if r['cut'] != 'lines' and arm != 'fixed' else ""))
    print(f"\n  {'arm':11s} {'cover':>7s} {'crossed':>8s} {'crossed&not-whole':>18s} {'whole':>9s} {'oversize':>8s}"
          f" {'determ.':>8s} {'bands':>6s} {'fallback':>8s}")
    verdicts = []
    for arm, a in rep["summary"].items():
        print(f"  {arm:11s} {a['cover_ok']}/{a['pages']:<5d} {a['crossed']:>8d} {a['crossed_not_whole']:>18d}"
              f" {a['whole']:>4d}/{a['lines']:<4d} {a['oversize']:>8d} {a['deterministic']}/{a['pages']:<6d}"
              f" {a['bands']:>6d} {a['fallback']:>8d}")
        if arm == "fixed":
            continue
        verdicts += [(f"(a) {arm}: bands cover every row in order on 100% of pages", a["cover_ok"] == a["pages"]),
                     (f"(a) {arm}: truth lines crossed by a cut and not whole in a band = 0", a["crossed_not_whole"] == 0),
                     (f"(a) {arm}: every truth line whole inside a band (100%)", a["whole"] == a["lines"]),
                     (f"(a) {arm}: bands over the no-resize size = 0", a["oversize"] == 0),
                     (f"(a) {arm}: same input twice gives the same rows", a["deterministic"] == a["pages"])]
    return verdicts


# ---------------------------------------------------------------------------
# (b) whole books
# ---------------------------------------------------------------------------
def book_pages(slug: str, only) -> list[tuple[int, bool]]:
    """(page, prose) for every non-blank page with a sidecar."""
    out = []
    for js in sorted((ROOT / "books" / slug / "text").glob("[0-9][0-9][0-9][0-9].json")):
        page = int(js.stem)
        if only is not None and page not in only:
            continue
        if (slug, page) in HELD_OUT:
            continue
        side = json.loads(js.read_text(encoding="utf-8"))
        if side.get("is_blank"):
            continue
        md = js.with_suffix(".md")
        text = md.read_text(encoding="utf-8") if md.is_file() else ""
        prose = not any(_NOT_PROSE.search(s.strip()) for s in text.split("\n") if s.strip())
        out.append((page, prose))
    return out


def part_b(limit: int | None = None) -> dict:
    books = {}
    for slug, only in BOOKS:
        plist = book_pages(slug, only)
        if limit:
            plist = plist[:limit]
        rows_out, secs = [], []
        print(f"  {slug}: {len(plist)} non-blank pages", flush=True)
        for page, prose in plist:
            refuse(slug, page)
            rec = dict(page=page, prose=prose, exception=None, arms={})
            try:
                g, kind = source_gray(slug, page)
                H, W = g.shape
                rec.update(source=kind, size=[W, H])
                finder = CachedFinder()
                for arm in ARMS:
                    rows, info = arm_rows(arm, g, finder)
                    rec["arms"][arm] = dict(cut=info.get("cut"), reason=info.get("reason"), bands=len(rows),
                                            cover_ok=cover_ok(rows, H), oversize=oversize(rows, W),
                                            text_lines=info.get("text_lines"), gaps=info.get("gaps"),
                                            splits=info.get("splits"))
                rec["finder_seconds"] = round(finder.seconds, 3) if finder.seconds is not None else None
                rec["finder_error"] = repr(finder.exc) if finder.exc is not None else None
                if finder.seconds is not None:
                    secs.append(finder.seconds)
            except Exception as exc:              # noqa: BLE001 - counted as an exception
                rec["exception"] = f"{type(exc).__name__}: {exc}"
            rows_out.append(rec)
        books[slug] = dict(pages=rows_out, summary=summarise_book(rows_out, secs))
        s = books[slug]["summary"]
        print(f"    done: exceptions {s['exceptions']}, finder median {s['finder_seconds_median']} s", flush=True)
    return books


def _reason_key(r: str | None) -> str:
    if not r:
        return "?"
    return re.sub(r"\(\d+( px)?\)", "(n)", r)[:80]


def summarise_book(rows, secs) -> dict:
    s = dict(pages=len(rows), prose_pages=sum(r["prose"] for r in rows),
             exceptions=sum(r["exception"] is not None for r in rows),
             finder_errors=sum(bool(r.get("finder_error")) for r in rows),
             finder_seconds_median=round(statistics.median(secs), 2) if secs else None,
             finder_seconds_max=round(max(secs), 2) if secs else None, arms={})
    for arm in ARMS:
        ok = [r for r in rows if r["exception"] is None]
        a = [r["arms"][arm] for r in ok]
        fb = [r for r in ok if r["arms"][arm]["cut"] != "lines"] if arm != "fixed" else []
        fbp = [r for r in fb if r["prose"]]
        reasons: dict = {}
        for r in fb:
            k = _reason_key(r["arms"][arm]["reason"])
            reasons[k] = reasons.get(k, 0) + 1
        s["arms"][arm] = dict(cover_ok=sum(x["cover_ok"] for x in a), oversize_pages=sum(x["oversize"] > 0 for x in a),
                              checked=len(a), bands=sum(x["bands"] for x in a),
                              fallback=len(fb), fallback_prose=len(fbp),
                              fallback_prose_share=round(len(fbp) / s["prose_pages"], 4) if s["prose_pages"] else None,
                              fallback_share=round(len(fb) / len(rows), 4) if rows else None,
                              fallback_pages=[r["page"] for r in fb], reasons=reasons)
    return s


def print_b(books) -> list[tuple[str, bool]]:
    print("\n(b) Whole-book robustness (band source image)")
    verdicts = []
    for slug, b in books.items():
        s = b["summary"]
        print(f"  {slug}: {s['pages']} non-blank pages ({s['prose_pages']} prose), exceptions {s['exceptions']},"
              f" finder errors {s['finder_errors']}, finder seconds median {s['finder_seconds_median']} max {s['finder_seconds_max']}")
        for arm, a in s["arms"].items():
            line = (f"    {arm:11s} cover {a['cover_ok']}/{a['checked']}  pages with an oversize band {a['oversize_pages']}"
                    f"  bands {a['bands']}")
            if arm != "fixed":
                line += (f"  fallback {a['fallback']}/{s['pages']} (prose {a['fallback_prose']}/{s['prose_pages']}"
                         f" = {100 * (a['fallback_prose_share'] or 0):.1f}%)  reasons {a['reasons']}")
            print(line)
        verdicts.append((f"(b) {slug}: exceptions = 0", s["exceptions"] == 0))
        for arm, a in s["arms"].items():
            if arm == "fixed":
                continue
            verdicts.append((f"(b) {slug} {arm}: covering and size invariants on 100% of pages",
                             a["cover_ok"] == s["pages"] and a["oversize_pages"] == 0 and a["checked"] == s["pages"]))
            verdicts.append((f"(b) {slug} {arm}: fallback <= 10% of non-blank prose pages",
                             (a["fallback_prose_share"] or 0.0) <= FALLBACK_MAX))
        verdicts.append((f"(b) {slug}: line-finder median <= 10 s per page",
                         s["finder_seconds_median"] is not None and s["finder_seconds_median"] <= SECONDS_MEDIAN_MAX))
    return verdicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("a", "b", "all"), default="all")
    ap.add_argument("--limit", type=int, help="first N pages per book in (b) (a quick look; not the smoke)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    rep = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), finder_version=LF.VERSION,
               thresholds=dict(fallback_prose_max=FALLBACK_MAX, finder_seconds_median_max=SECONDS_MEDIAN_MAX),
               lines_per_band=list(LPB), limit=a.limit)
    verdicts = []
    if a.part in ("a", "all"):
        rep["truth_pages"] = part_a()
        verdicts += print_a(rep["truth_pages"])
    if a.part in ("b", "all"):
        rep["books"] = part_b(a.limit)
        verdicts += print_b(rep["books"])
    rep["verdicts"] = [dict(check=c, passed=bool(v)) for c, v in verdicts]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for c, v in verdicts:
        print(f"  {'PASS' if v else 'FAIL'}  {c}")
    fails = sum(not v for _, v in verdicts)
    print(f"\n{len(verdicts) - fails} PASS, {fails} FAIL; wrote {a.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
