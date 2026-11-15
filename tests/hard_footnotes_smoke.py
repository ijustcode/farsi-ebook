"""Smoke test: footnotes carried through the hard-page reader's Markdown.

$0, offline. Needs books/bachehaye_ghali (first-pass pages with footnotes) and
the cached Opus readings of the development pages.

Criteria, fixed before the code was written (2026-11-15):
  A. Round trip. Every bachehaye_ghali page whose stored Markdown has footnotes
     and no verse or table is turned into a simulated reader output in the form
     the reader used on the one live sample (p64): the reference as a separate
     digit after its word, the definition as a bottom line "n ـ text", body
     wrapped to printed-line lengths. lines_to_markdown must give back, on at
     least 95% of pages: the same reference numbers, each after the same word;
     the same definitions with the same words; no leftover digit token.
  B. No invention. On the cached Opus readings of hadpi p1-p2 and majid
     p23/p109 (no footnotes), the output must contain no footnote at all.
PASS only if both hold. This tests the mechanism against an assumed reader
format; real readings of footnote pages need a paid call.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from farsi2epub import hard_reader as HR  # noqa: E402

FA = "۰۱۲۳۴۵۶۷۸۹"
REF = re.compile(r"\[\^(\d+)\]")
DEF = re.compile(r"^\[\^(\d+)\]:\s*(.*)$")


def fa(n: str) -> str:
    return "".join(FA[int(c)] for c in n)


def simulate(md: str, width: int = 11) -> list[str]:
    body, defs = [], []
    for line in md.split("\n"):
        s = line.strip()
        m = DEF.match(s)
        if m:
            defs.append(f"{fa(m.group(1))} ـ {REF.sub('', m.group(2)).strip()}")
        elif s:
            s = re.sub(r"^(#{1,3}|>|[-*+])\s+", "", s).replace("**", "")
            body.append(REF.sub(lambda r: f" {fa(r.group(1))} ", s).split())
    lines = []
    for words in body:
        lines += [" ".join(words[i:i + width]) for i in range(0, len(words), width)]
    return lines + defs


def facts(md: str):
    refs, defs = [], {}
    for line in md.split("\n"):
        s = line.strip()
        m = DEF.match(s)
        if m:
            defs[m.group(1)] = HR._fold("".join(REF.sub("", m.group(2)).split()))
            continue
        for w in s.split():
            for r in REF.finditer(w):
                refs.append((r.group(1), HR._fold(w[:r.start()])))
    return refs, defs


def main() -> int:
    ok_pages = total = 0
    for f in sorted(glob.glob(str(ROOT / "books/bachehaye_ghali/text/[0-9][0-9][0-9][0-9].md"))):
        md = Path(f).read_text(encoding="utf-8")
        if not REF.search(md) or re.search(r"^(```|\|)| --- ", md, re.M):
            continue
        refs, defs = facts(md)
        if not defs or {n for n, _ in refs} != set(defs):
            continue          # the first pass itself is inconsistent: not a round-trip case
        total += 1
        out, _, report = HR.lines_to_markdown(simulate(md), md)
        refs2, defs2 = facts(out)
        stray = [w for l in out.split("\n") if not DEF.match(l.strip()) for w in l.split() if w.strip("()") in {fa(n) for n in defs}]
        good = refs2 == refs and defs2 == defs and not stray
        ok_pages += good
        if not good:
            print(f"   differs: {Path(f).name} refs {len(refs)}->{len(refs2)} defs {len(defs)}->{len(defs2)} stray {stray} {report.get('footnotes')}")
    share = ok_pages / total if total else 0.0
    print(f"A. round trip: {ok_pages} of {total} pages ({share:.0%}); need 95%")
    import transcription_experiments as TE
    invented = n = 0
    for f in glob.glob(str(ROOT / "out/transcription_experiments/cache/*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if r.get("arm") != "t1r_opus" or r.get("page") not in ("h1", "h2", "c23", "c109"):
            continue
        lines, _, _ = HR.plain_lines(TE.output_of(r, "t1r_opus")["lines"])
        out, _, _ = HR.lines_to_markdown(lines, TE.stored_md(r["page"]))
        n += 1
        invented += bool(REF.search(out))
    print(f"B. no invention: {invented} of {n} cached readings gained a footnote; need 0")
    ok = total > 0 and share >= 0.95 and n > 0 and invented == 0
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
