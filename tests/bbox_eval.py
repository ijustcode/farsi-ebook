"""Human-countable bbox accuracy contact sheet for the review UI's overlays.

Builds, for every page with a pending QC suggestion, the exact same locate
queries and boxes the review UI does (reusing review._build_boxes and its
hunk-derivation path), then renders one self-contained HTML sheet with one
row per finding/hunk box: page number, tier chip, the RTL query snippet, and
a generous crop of the page around the box with the box outline drawn on it.
A human scrolls once and counts misses per tier — this is how the scan tier's
historical miss rate gets re-measured after VLM refinement.

Usage:
    ./venv/bin/python tests/bbox_eval.py <slug> [--pages N-M] [--no-refine] [--out PATH]

With refinement on (the default), review._REFINER is pointed at the book's
existing locate_vlm.json disk cache, so cached (page, query) pairs cost
nothing; any uncached scan boxes are refined via the API (cost is reported).
--no-refine produces the plain-tier baseline sheet for before/after
comparison.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
from pathlib import Path
from typing import Optional

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farsi2epub import llm, review  # noqa: E402
from farsi2epub.config import MODEL_STRONG  # noqa: E402
from farsi2epub.review import (  # noqa: E402
    _build_boxes,
    _cap_words,
    _read_sidecar,
    _ScanBoxRefiner,
)
from farsi2epub.workspace import Workspace, parse_pages_spec  # noqa: E402

# Tier chip colors, matching the review UI's box CSS exactly.
TIER_COLORS = {
    "match": "#3a9ff0",
    "layout": "#2fb6a8",
    "scan": "#7ac65c",
    "scan_vlm": "#2e8b57",
    "model": "#f0a03a",
}
TIER_ORDER = ["match", "layout", "scan", "scan_vlm", "model"]
MAX_ROWS = 120


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _page_specs(ws: Workspace, n: int) -> Optional[tuple[list[dict], list[dict], str]]:
    """(issues, hunks, text) for page `n`, derived exactly the way
    review._page_view does for a pending QC suggestion (shared helper); None
    when the page has no pending suggestion (review builds no boxes for it
    either)."""
    _sidecar, issues, hunks, text, panel_kind = review._page_box_inputs(ws, n)
    if panel_kind is None:
        return None
    return issues, hunks, text


def _hunk_query_text(h: dict) -> str:
    """The display snippet for a hunk row: the same capped text
    review._build_boxes uses as the locating query."""
    if h["ws_only"]:
        return _cap_words(h["ctx_before"] + h["old"] + h["ctx_after"])
    return _cap_words(h["old"])


def _crop_png(page: fitz.Page, box: dict, color_hex: str) -> bytes:
    """PNG crop of the page around `box` (0-1 fractions) with the box outline
    drawn on it. Margin is generous (~2 box widths of context) so a human can
    instantly judge whether the box contains the snippet."""
    pr = page.rect
    bx0 = pr.x0 + box["x0"] * pr.width
    by0 = pr.y0 + box["y0"] * pr.height
    bx1 = pr.x0 + box["x1"] * pr.width
    by1 = pr.y0 + box["y1"] * pr.height
    bw = max(bx1 - bx0, 1.0)
    bh = max(by1 - by0, 1.0)
    pad_x = max(bw * 1.0, pr.width * 0.10)
    pad_y = max(bh * 2.5, pr.height * 0.05)
    clip = fitz.Rect(
        max(pr.x0, bx0 - pad_x),
        max(pr.y0, by0 - pad_y),
        min(pr.x1, bx1 + pad_x),
        min(pr.y1, by1 + pad_y),
    )
    scale = max(1.5, min(4.0, 1000.0 / max(clip.width, 1.0)))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    # Box corners in crop pixel coordinates.
    px0 = int(round((bx0 - clip.x0) * scale))
    py0 = int(round((by0 - clip.y0) * scale))
    px1 = int(round((bx1 - clip.x0) * scale))
    py1 = int(round((by1 - clip.y0) * scale))
    t = max(2, int(round(pix.width * 0.004)))
    rgb = _hex_to_rgb(color_hex)

    def _edge(x0: int, y0: int, x1: int, y1: int) -> None:
        r = fitz.IRect(
            max(0, x0), max(0, y0), min(pix.width, x1), min(pix.height, y1)
        )
        if not r.is_empty:
            pix.set_rect(r, rgb)

    _edge(px0 - t, py0 - t, px1 + t, py0)  # top
    _edge(px0 - t, py1, px1 + t, py1 + t)  # bottom
    _edge(px0 - t, py0, px0, py1)  # left
    _edge(px1, py0, px1 + t, py1)  # right
    return pix.tobytes("png")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bbox accuracy contact sheet (one row per finding/hunk box)."
    )
    ap.add_argument("slug", help="book workspace slug under books/")
    ap.add_argument("--pages", default=None, help='page spec, e.g. "40-60" or "39,41-45"')
    ap.add_argument(
        "--no-refine",
        action="store_true",
        help="skip scan_vlm refinement (plain-tier baseline sheet)",
    )
    ap.add_argument("--out", default=None, help="output HTML path")
    args = ap.parse_args()

    ws = Workspace.load(args.slug)
    out_path = Path(args.out) if args.out else ws.out_dir / "bbox_eval.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = ws.pages_done()
    if args.pages:
        page_count = int(ws.meta.get("page_count") or (max(done) if done else 0))
        wanted = set(parse_pages_spec(args.pages, page_count))
        done = [n for n in done if n in wanted]

    qc_pages = [n for n in done if _read_sidecar(ws, n).get("qc")]
    if not qc_pages:
        print(f"No pages with QC findings for '{args.slug}'.")
        return 1

    # Refinement wiring: same _ScanBoxRefiner review uses, same disk cache
    # (books/<slug>/locate_vlm.json). Cached entries are free; uncached scan
    # boxes call the API only when a key resolves (otherwise _build_boxes
    # degrades to the plain scan boxes, exactly like the review server).
    cache_path = ws.root / "locate_vlm.json"

    def _load_cache() -> dict:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    cache_before = _load_cache()
    if args.no_refine:
        review._REFINER = None
    else:
        review._REFINER = _ScanBoxRefiner(ws, MODEL_STRONG)
        llm.load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "note: no ANTHROPIC_API_KEY — cached scan_vlm boxes still apply, "
                "uncached scan boxes stay unrefined."
            )

    rows: list[dict] = []  # {page, source, snippet, kind, box}
    tier_counts: dict[str, int] = {}
    no_box_count = 0
    pages_pending: list[int] = []
    pages_resolved = 0

    for n in qc_pages:
        specs = _page_specs(ws, n)
        if specs is None:
            pages_resolved += 1
            continue
        pages_pending.append(n)
        issues, hunks, text = specs
        # The render path is cache-only now; the eval drives the API-calling
        # refine() itself (same specs => same cache keys) so uncached scan
        # boxes still get refined here, then _build_boxes picks them up from
        # the warmed cache.
        if not args.no_refine and review._REFINER is not None:
            box_specs = review._box_specs(issues, hunks, text)
            queries = [s[1] for s in box_specs]
            located = list(
                review._locate_queries_cached(
                    str(ws.pdf_path), n, text, tuple(queries)
                )
            )
            try:
                review._REFINER.refine(n, text, queries, located)
            except Exception as exc:
                print(f"bbox refine p{n}: eval refine error: {exc}", file=sys.stderr)
        _build_boxes(ws, n, issues, hunks, text)  # sets "box" on hunks/issues

        linked = {h["issue_idx"] for h in hunks if h["issue_idx"] is not None}
        for h in hunks:
            box = h.get("box")
            snippet = _hunk_query_text(h)
            kind = "finding-edit" if h.get("edit_only") else "hunk"
            if box is None:
                no_box_count += 1
                continue
            tier_counts[box["source"]] = tier_counts.get(box["source"], 0) + 1
            rows.append(
                {"page": n, "source": box["source"], "snippet": snippet, "kind": kind, "box": box}
            )
        for i, iss in enumerate(issues):
            if i in linked:
                continue
            box = iss.get("box")
            snippet = _cap_words(iss.get("snippet") or "")
            if box is None:
                no_box_count += 1
                continue
            tier_counts[box["source"]] = tier_counts.get(box["source"], 0) + 1
            rows.append(
                {"page": n, "source": box["source"], "snippet": snippet, "kind": "issue", "box": box}
            )

    # Refinement cost actually incurred by this run (new cache entries).
    cache_after = _load_cache()
    new_keys = set(cache_after) - set(cache_before)
    extra_cost = sum((cache_after[k] or {}).get("cost", 0.0) for k in new_keys)

    shown = rows[:MAX_ROWS]
    truncated = len(rows) - len(shown)

    doc = fitz.open(str(ws.pdf_path))
    try:
        for r in shown:
            page = doc[r["page"] - 1]
            color = TIER_COLORS.get(r["source"], "#888888")
            png = _crop_png(page, r["box"], color)
            r["img"] = base64.standard_b64encode(png).decode("ascii")
    finally:
        doc.close()

    mode = "baseline (no refine)" if args.no_refine else "refined"
    tier_summary = " · ".join(
        f'{t}: {tier_counts.get(t, 0)}' for t in TIER_ORDER if tier_counts.get(t)
    )
    chips_css = "\n".join(
        f".chip-{t} {{ background: {c}; }}" for t, c in TIER_COLORS.items()
    )

    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>bbox eval — {html.escape(args.slug)} ({html.escape(mode)})</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; background: #16161a; color: #e8e8ea; }}
header {{ padding: 1rem 1.5rem; border-bottom: 1px solid #333; position: sticky; top: 0; background: #16161aee; backdrop-filter: blur(4px); }}
header h1 {{ font-size: 1.1rem; margin: 0 0 0.4rem; }}
header p {{ margin: 0.15rem 0; font-size: 0.9rem; color: #b8b8c0; }}
.chip {{ display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; color: #0d0d0f; font-size: 0.8rem; font-weight: 700; vertical-align: middle; }}
{chips_css}
.row {{ display: grid; grid-template-columns: 340px 1fr; gap: 1rem; padding: 1rem 1.5rem; border-bottom: 1px solid #2a2a30; align-items: center; }}
.row .meta {{ font-size: 0.9rem; color: #b8b8c0; }}
.row .meta .pageno {{ font-size: 1rem; font-weight: 700; color: #e8e8ea; margin-inline-end: 0.6rem; }}
.row .snippet {{ direction: rtl; text-align: right; font-size: 1.35rem; line-height: 2.1rem; margin-top: 0.6rem; color: #f2f2f4; font-family: "Vazirmatn", "Iranian Sans", "Tahoma", sans-serif; word-break: break-word; }}
.row .kind {{ font-size: 0.75rem; color: #7a7a85; margin-top: 0.4rem; }}
.row img {{ max-width: 100%; height: auto; border: 1px solid #333; border-radius: 4px; background: #fff; }}
</style>
</head>
<body>
<header>
<h1>bbox eval — {html.escape(args.slug)} — {html.escape(mode)}</h1>
<p>Boxes per tier: {html.escape(tier_summary) if tier_summary else "none"} — findings/hunks with no box: {no_box_count}</p>
<p>{len(pages_pending)} page(s) with pending QC suggestions produced boxes; {pages_resolved} QC page(s) already resolved (review builds no boxes for those either).</p>
<p>Judge each row: does the outlined box contain the snippet on the right-hand crop? Count misses per tier.</p>
"""
    )
    if truncated > 0:
        parts.append(
            f"<p><strong>Showing the first {MAX_ROWS} of {len(rows)} rows; "
            f"{truncated} row(s) omitted (tier totals above cover all rows).</strong></p>"
        )
    if not args.no_refine and (new_keys or extra_cost):
        parts.append(
            f"<p>Refinement API cost incurred by this run: ${extra_cost:.4f} "
            f"({len(new_keys)} new cache entries).</p>"
        )
    parts.append("</header>\n")

    for r in shown:
        color = TIER_COLORS.get(r["source"], "#888888")
        parts.append(
            f"""<div class="row">
<div>
<div class="meta"><span class="pageno">p{r['page']}</span><span class="chip chip-{html.escape(r['source'])}">{html.escape(r['source'])}</span></div>
<div class="snippet">{html.escape(r['snippet'])}</div>
<div class="kind">{html.escape(r['kind'])} — box drawn in <span style="color:{color}">{html.escape(r['source'])}</span></div>
</div>
<div><img src="data:image/png;base64,{r['img']}" alt="p{r['page']} crop"></div>
</div>
"""
        )
    parts.append("</body>\n</html>\n")

    out_path.write_text("".join(parts), encoding="utf-8")

    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")
    print(f"mode: {mode}")
    print(
        "rows: "
        + (", ".join(f"{t}={tier_counts.get(t, 0)}" for t in TIER_ORDER))
        + f", total={len(rows)}, shown={len(shown)}, no-box={no_box_count}"
    )
    if not args.no_refine:
        print(
            f"refinement API cost this run: ${extra_cost:.4f} "
            f"({len(new_keys)} new cache entries)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
