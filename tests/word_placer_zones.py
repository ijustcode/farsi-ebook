#!/usr/bin/env python3
"""Ignore zones for marks that are not the document's text (owner decision
D15: stamps, signatures, handwriting, scanner watermarks). Derived by one
rule for every page, written to a private, additive sidecar next to the
truth freeze record; the audited truth files are never touched.

Rule (the same on every page):
  A. everything below the last audited (graded) text line: from that line's
     lowest audited point plus half the page's line spacing to the bottom of
     the page, full width;
  B. the bounding box, padded by a quarter line spacing, of every component
     the line finder masks as oversized ("big": stamps, signatures, logos).

  ./venv/bin/python tests/word_placer_zones.py derive [--keys slug/page ...]
  ./venv/bin/python tests/word_placer_zones.py show

Zone B can overlap the last typed lines where a signature or stamp touches
them (hadpi p1: 4 audited words, p2: 7). Those words stay graded: the zones
are used only to classify detected lines that hold no audited line (G2), never
to drop truth words. Each zone records ``audited_words_inside``.

Sidecar: books/<slug>/truth/ignore_zones.derived.json (private, gitignored
with books/). Held-out hadpi pages are refused by the truth loader.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import line_finder as LF  # noqa: E402
import real_truth_eval as rt  # noqa: E402

SCHEMA = "farsi2epub.ignore-zones-derived/v1"
RULE = "A: below the last graded line + 0.5 line spacing, full width; B: L1 'big' masked components, padded 0.25 spacing"
KEYS = [("hadpi-articles", 1), ("hadpi-articles", 2), ("ghessehaye-majid", 23), ("ghessehaye-majid", 109),
        ("bachehaye_ghali", 66)]


def derive_page(slug, page):
    t = rt.load_truth(ROOT, slug, page)
    gray = LF.load_gray(t.image_path)
    H, W = gray.shape
    lf = LF.find_lines(gray, use_mask=True, keep_arrays=True)
    S = float(lf["measurements"]["line_spacing_px"])
    graded = [L for L in t.lines if not L.excluded]
    spacing = statistics.median([b.frame["cy"] - a.frame["cy"] for a, b in zip(t.lines, t.lines[1:])] or [S])
    zones = []
    if graded:
        yb = max(p[1] for L in graded for w in L.words for p in w.quad) + 0.5 * spacing
        if yb < H:
            zones.append(dict(rect=[0.0, yb / H, 1.0, 1.0], source="A below the last audited line"))
    cc, reason = lf["cc"], lf["reason"]
    pad = 0.25 * S
    for c in np.nonzero(reason == "big")[0]:
        x0, x1 = max(0.0, cc.x0[c] - pad), min(W, cc.x1[c] + 1 + pad)
        y0, y1 = max(0.0, cc.y0[c] - pad), min(H, cc.y1[c] + 1 + pad)
        zones.append(dict(rect=[x0 / W, y0 / H, x1 / W, y1 / H], source="B oversized component"))
    # report (never clip: the rule stays as stated) audited words whose centre falls in a zone; such
    # words keep being graded, and G2 counts only detected lines that hold no audited line
    for z in zones:
        x0, y0, x1, y1 = z["rect"][0] * W, z["rect"][1] * H, z["rect"][2] * W, z["rect"][3] * H
        z["audited_words_inside"] = sum(
            1 for L in t.lines for w in L.words
            if x0 <= sum(p[0] for p in w.quad) / 4 <= x1 and y0 <= sum(p[1] for p in w.quad) / 4 <= y1)
    return dict(zones=zones, image_sha256=t.image_sha256, audit_sha256=t.audit_sha256,
                derived_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--keys", nargs="*", default=None)
    sub.add_parser("show")
    a = ap.parse_args(argv)
    keys = [k for k in KEYS if not a.__dict__.get("keys") or f"{k[0]}/{k[1]}" in a.keys]
    for slug, page in keys:
        path = rt.extra_zones_path(ROOT, slug)
        doc = json.loads(path.read_text()) if path.exists() else dict(schema=SCHEMA, rule=RULE, pages={})
        if a.cmd == "derive":
            doc["pages"][str(page)] = derive_page(slug, page)
            doc["rule"] = RULE
            path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        rec = doc["pages"].get(str(page), {})
        z = rec.get("zones", [])
        a_ = [x for x in z if x["source"].startswith("A")]
        b_ = [x for x in z if x["source"].startswith("B")]
        area = sum((x["rect"][2] - x["rect"][0]) * (x["rect"][3] - x["rect"][1]) for x in z)
        print(f"  {slug}/{page}: zone A {'from y=' + format(a_[0]['rect'][1], '.3f') if a_ else 'none'}; "
              f"{len(b_)} oversized-component zones; total area (with overlaps) {area:.1%} of the page; audited words"
              f" with centre inside: {sum(x.get('audited_words_inside', 0) for x in z)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
