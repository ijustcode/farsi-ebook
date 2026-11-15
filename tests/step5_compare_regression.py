#!/usr/bin/env python3
"""Step 5 comparison: the pure parts (correction -> words of the reading, paired test). Synthetic, no books/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5_compare as S

FAILS = 0


def check(name, cond):
    global FAILS
    if not cond:
        FAILS += 1
    print(("ok   " if cond else "FAIL ") + name)


md = "## سرفصل\n\nاین یک جملهٔ ساده است، و ادامه دارد.\n"
toks = S.md_tokens(md)
check("markdown marks fold away", [t[2] for t in toks][0] == S._fold("سرفصل"))
i = md.index("ساده")
check("replacement takes the words under the span", S.target_tokens(toks, (i, i + 4), "replacement") == [4])
check("punctuation-only span takes the word it is attached to",
      S.target_tokens(toks, (md.index("،"), md.index("،") + 1), "replacement") == [5])
check("insertion takes the word on each side", S.target_tokens(toks, (i - 1, i - 1), "insertion") == [3, 4])

a = ["a", "b", "c", "d", "e"]
check("equal runs map one to one", S.align_words(a, ["x", "a", "b", "c", "d", "e"]) == {k: [k + 1] for k in range(5)})
m = S.align_words(a, ["a", "b", "X", "Y", "d", "e"])
check("unequal replacement maps to the whole block", m[2] == [2, 3] and m[3] == [4])
check("a deleted word has no counterpart", 2 not in S.align_words(a, ["a", "b", "d", "e"]))

rd = [S._fold(w) for w in "روزی روزگاری مردی بود که کتاب می‌خواند و روزی دیگر رفت".split()]
idx, why = S.find_text("مردی بود که", rd)
check("span-less text is found", idx == [2, 3, 4] and why == "")
idx, why = S.find_text("روزی", rd)
check("a repeated text is refused", idx is None and "more than once" in why)
idx, why = S.find_text("چیزی کاملاً متفاوت اینجا", rd)
check("an absent text is refused", idx is None)
check("empty text is refused", S.find_text("", rd)[0] is None)

arm = {(0, 0): {"status": "located", "segments": [{"quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
       (0, 1): {"status": "located", "segments": [{"quad": [[1, 0], [2, 0], [2, 1], [1, 1]]}]},
       (0, 2): {"status": "unresolved", "segments": []}}
lines = [["الف", "ب", "ج"]]
r = S.place_correction(arm, lines, "الف ب ج", (0, 5), "replacement", "")
check("all words located -> located box with one segment per word", r["status"] == "located" and len(r["segments"]) == 2)
r = S.place_correction(arm, lines, "الف ب ج", (4, 7), "replacement", "")
check("an unresolved word -> no box", r["status"] == "unresolved" and not r["segments"])
r = S.place_correction(arm, lines, "الف ب ج د", (8, 9), "replacement", "")
check("a word missing from the reading -> no box, with the reason", r["status"] == "unresolved" and "counterpart" in r["reason"])

check("exact paired test", abs(S._mcnemar(5, 0) - 0.0625) < 1e-9 and S._mcnemar(0, 0) == 1.0 and S._mcnemar(3, 3) == 1.0)
lo, hi = S._wilson(50, 100)
check("interval", 0.40 < lo < 0.41 and 0.59 < hi < 0.60)

print(f"\n{'PASS' if not FAILS else 'FAIL'}: {FAILS} failure(s)")
sys.exit(1 if FAILS else 0)
