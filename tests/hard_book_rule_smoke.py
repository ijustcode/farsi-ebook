"""Smoke test for the book-level hard-page rule (transcribe.hard_book_decision).

$0, offline: replays stored first-pass sidecars; needs processed books under
books/ and the cached `current` readings of hadpi from
out/transcription_experiments/cache/.

Criteria, fixed before the rule was written (2026-11-14):
  1. hadpi-articles triggers the rule under EVERY combination of cached
     first-pass confidences (the stored sidecars, and each cached repeat of the
     production call, including the run where page 3 reported 0.75).
  2. No other workspace triggers it.
Criterion 2 revised by owner decision D21 (2026-11-15, line lowered from one
half to one fifth so that old-font printed books are re-read): exactly the
workspaces in OLD_FONT trigger, no other.
PASS only if both hold. hadpi p3-p5 are held out: only their first-pass
confidence is used, once, as the final check of this trigger; nothing is tuned.
"""
from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from farsi2epub import transcribe  # noqa: E402

HADPI = "hadpi-articles"
SKIP = {"hadpi-articles-truth"}      # built from the owner's truth, not a first pass
OLD_FONT = {"bachehaye_ghali", "jafarian-factory-sale"}   # meant to trigger since D21


def sidecars(slug: str) -> dict[int, dict]:
    out = {}
    for f in sorted(glob.glob(str(ROOT / "books" / slug / "text" / "[0-9][0-9][0-9][0-9].json"))):
        out[int(Path(f).stem)] = json.loads(Path(f).read_text(encoding="utf-8"))
    return out


def main() -> int:
    ok = True
    books = sorted(p.parent.name for p in (ROOT / "books").glob("*/text")
                   if p.parent.name not in SKIP and not p.parent.name.endswith("-live"))   # -live: copies made for live runs
    print("1. other workspaces: only the old-font ones may trigger")
    for slug in books:
        if slug == HADPI:
            continue
        d = transcribe.hard_book_decision(sidecars(slug))
        print(f"   {slug:<46} fired {d['fired']:>3} of {d['counted']:>3}  hard={d['hard']}")
        ok &= d["hard"] == (slug in OLD_FONT)
    print("2. hadpi must trigger under every cached set of first-pass confidences")
    base = sidecars(HADPI)
    reps: dict[int, list] = {n: [None] for n in base}      # None = stored sidecar
    for f in glob.glob(str(ROOT / "out/transcription_experiments/cache/*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if r.get("arm") == "current" and str(r.get("page", "")).startswith("h") and r["page"][1:].isdigit():
            reps[int(r["page"][1:])].append((r["parsed"]["confidence"], r["parsed"].get("flags") or []))
    combos = hard = 0
    worst = None
    for choice in itertools.product(*(reps[n] for n in sorted(reps))):
        sides = {}
        for n, c in zip(sorted(reps), choice):
            s = dict(base[n])
            s.pop("first_pass_review", None)
            s.pop("hard_reader", None)
            if c is not None:
                s["confidence"], s["flags"] = c
            sides[n] = s
        d = transcribe.hard_book_decision(sides)
        combos += 1
        hard += d["hard"]
        if worst is None or d["fired"] < worst["fired"]:
            worst = d
    print(f"   {hard} of {combos} combinations trigger; fewest pages fired: {worst['fired']} of {worst['counted']}")
    ok &= hard == combos and combos > 1
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
