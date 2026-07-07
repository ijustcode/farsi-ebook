"""Conservative, deterministic normalization for Persian markdown text.

Only handles things that are unambiguous and safe to change automatically:
character-set unification (Arabic -> Persian letterforms), kashida removal,
whitespace collapsing, and spacing around Persian punctuation / guillemets.

Deliberately does NOT touch: digits (of any script), ZWNJ insertion/repair
inside words, or newlines (paragraph/block structure is preserved exactly
as produced by the transcription step).
"""

from __future__ import annotations

import re

# -- character mapping -------------------------------------------------

_ARABIC_YE = "ي"  # ي
_PERSIAN_YE = "ی"  # ی
_ARABIC_KAF = "ك"  # ك
_PERSIAN_KAF = "ک"  # ک
_KASHIDA = "ـ"  # ـ (tatweel)
_ZWNJ = "‌"

# Persian punctuation that never wants a space before it and always wants
# exactly one space after it (unless at end of line, or immediately before
# a closing quote/paren).
_SIMPLE_PUNCT = "،؛؟"

# Punctuation that can also appear *inside* numbers (decimal separators,
# ratios, times such as ۱۲:۳۰ or 3.14). We only fix spacing around these
# when neither neighbour is a digit, so numbers are never touched.
_DIGIT_RISKY_PUNCT = ".:"

# Characters used inside regex character classes below; escaped once here
# so call sites never need to worry about "]" closing a class early.
_CLOSERS = re.escape("»\")”’]}")
_DIGITS = "0-9٠-٩۰-۹"


def normalize_persian(text: str) -> str:
    """Apply conservative, deterministic normalization to `text`.

    - Maps Arabic ي/ك to Persian ی/ک.
    - Strips kashida (tatweel, U+0640).
    - Collapses runs of horizontal whitespace (spaces/tabs) to a single
      space; newlines are never touched.
    - Trims spaces touching ZWNJ (U+200C) without repairing ZWNJ usage
      inside words.
    - Removes space before ، ؛ ؟ . : and ensures exactly one space after
      (skipped for . and : when adjacent to a digit, and never forced
      right before a closing quote/paren).
    - Collapses « text » spacing so there is no space touching the marks
      from the inside.
    """
    if not text:
        return text

    out = text
    out = out.replace(_ARABIC_YE, _PERSIAN_YE)
    out = out.replace(_ARABIC_KAF, _PERSIAN_KAF)
    out = out.replace(_KASHIDA, "")

    # Collapse runs of spaces/tabs (never newlines) to a single space.
    out = re.sub(r"[ \t]+", " ", out)

    # Trim spaces touching ZWNJ (do not otherwise repair ZWNJ placement).
    out = re.sub(r" +" + _ZWNJ, _ZWNJ, out)
    out = re.sub(_ZWNJ + r" +", _ZWNJ, out)

    # -- simple punctuation: ، ؛ ؟ ------------------------------------
    out = re.sub(r"[ \t]+([" + _SIMPLE_PUNCT + "])", r"\1", out)
    out = re.sub(
        r"([" + _SIMPLE_PUNCT + r"])[ \t]*(?=[^\s" + _CLOSERS + "])",
        r"\1 ",
        out,
    )

    # -- digit-risky punctuation: . : ----------------------------------
    out = re.sub(
        r"(?<![" + _DIGITS + r"])[ \t]+(?=[" + re.escape(_DIGIT_RISKY_PUNCT) + "])",
        "",
        out,
    )
    out = re.sub(
        r"([" + re.escape(_DIGIT_RISKY_PUNCT) + r"])(?![" + _DIGITS + r"])[ \t]*"
        r"(?=[^\s" + _DIGITS + _CLOSERS + "])",
        r"\1 ",
        out,
    )

    # -- guillemets: no inner space -------------------------------------
    out = re.sub(r"«[ \t]+", "«", out)
    out = re.sub(r"[ \t]+»", "»", out)

    # Trailing horizontal whitespace at end of a line is always safe to
    # trim (never changes meaning or block structure).
    out = re.sub(r"[ \t]+$", "", out, flags=re.MULTILINE)

    return out
