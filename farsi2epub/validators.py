"""Deterministic quality validators for page transcriptions.

LLM self-reported confidence is poorly calibrated, so escalation and review
decisions combine it with checks the model can't game:

- Embedded-text cross-check (digital PDFs): PyMuPDF's extracted Persian text
  has broken ordering (visual-order line segments) and broken lam-alef
  ligatures, but its words survive as character bags. Comparing char-sorted
  word multisets is order- and ligature-insensitive yet sharply discriminates
  real transcriptions from skipped/hallucinated text (measured on book2:
  self 1.0, adjacent page 0.23, half-corrupted 0.27).
- Script sanity (all pages): Arabic-only codepoints that should be Persian,
  unexpected Latin ratio, degenerate repetition.
- Heading cross-check: the model's own `headings` field, and (digital PDFs)
  bold/oversized lines from the text layer, must each have a matching `#`
  line in text_md (word-bag compared, so broken ligatures don't false-flag).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

# Thresholds
CONF_ESCALATE = 0.85          # fast-model confidence below this -> escalate
CONF_REVIEW = 0.70            # strong-model confidence below this -> human review
# The embedded-text oracle is strict when present: a faithful transcription of
# a digital page scores ~0.99; 0.97 already means several wrong words
# (measured on book2 p7: Haiku with visible word errors scored 0.9695).
EMBED_SIM_ESCALATE = 0.97     # embedded-text similarity below this -> escalate
EMBED_SIM_REVIEW = 0.90       # ... below this after escalation -> human review
LEN_RATIO_BOUNDS = (0.60, 1.60)   # transcription/embedded length outside -> suspicious
LATIN_RATIO_MAX = 0.15        # more Latin than this in a Persian book is suspicious
MIN_EMBEDDED_WORDS = 25       # below this the embedded-text oracle is unreliable

ESCALATE_FLAGS = {"illegible", "two_column", "table"}
REVIEW_FLAGS = {"illegible"}

_ARABIC_ONLY = "يك"  # should be ی / ک in Persian body text
# ZWNJ, hair space, soft hyphen: treated as word separators so می‌رود == می رود
_JOINERS = re.compile(r"[‌ ­]")
_DIACRITICS = re.compile(r"[ً-ْٰ]")
_PUNCT_DIGITS = re.compile(r"[\d۰-۹.,:;!?()\[\]{}«»،؛؟\-—_*#>`^\"'|/\\]+")
_LATIN = re.compile(r"[A-Za-z]")
_PERSIAN_ARABIC = re.compile(r"[؀-ۿ]")

HEADING_BAG_SIM = 0.85  # word-bag cosine above this counts as "the same heading"
_HEADING_LINE = re.compile(r"^#{1,3} (.+)$", re.MULTILINE)


def _normalize(text: str) -> str:
    text = _JOINERS.sub(" ", text)
    text = _DIACRITICS.sub("", text).replace("ـ", "")  # kashida
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه").replace("ة", "ه")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ی")
    return _PUNCT_DIGITS.sub(" ", text)


def word_bag(text: str) -> Counter:
    """Multiset of char-sorted words (length >= 2) after normalization.

    Char-sorting each word makes the comparison immune to the intra-word
    character reordering PDF extractors produce (e.g. کلاس -> کالس).
    """
    words = [w for w in _normalize(text).split() if len(w) >= 2]
    return Counter("".join(sorted(w)) for w in words)


def bag_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def heading_lines(text_md: str) -> list[str]:
    """The text of every `#`/`##`/`###` line in text_md, marks stripped."""
    return [m.strip() for m in _HEADING_LINE.findall(text_md)]


def missing_headings(headings: list[str], text_md: str) -> list[str]:
    """Declared `headings` with no matching heading line in text_md.

    A heading counts as matched when its word bag is >= HEADING_BAG_SIM
    cosine-similar to some heading line's word bag (ligature/ZWNJ immune).
    """
    line_bags = [word_bag(h) for h in heading_lines(text_md)]
    return [
        h for h in headings
        if not any(bag_cosine(word_bag(h), lb) >= HEADING_BAG_SIM for lb in line_bags)
    ]


def missing_embedded_headings(candidates: list[str], text_md: str) -> list[str]:
    """PDF-text-layer `candidates` (see render.extract_heading_candidates) with
    no matching heading line in text_md. Candidates with an empty word bag are
    skipped (nothing to compare)."""
    line_bags = [word_bag(h) for h in heading_lines(text_md)]
    missing = []
    for c in candidates:
        cb = word_bag(c)
        if not cb:
            continue
        if not any(bag_cosine(cb, lb) >= HEADING_BAG_SIM for lb in line_bags):
            missing.append(c)
    return missing


def evaluate(
    text_md: str,
    confidence: float,
    flags: list[str],
    embedded_text: Optional[str],
    headings: Optional[list[str]] = None,
    heading_candidates: Optional[list[str]] = None,
) -> dict:
    """Score one page transcription. Returns a validators dict for the sidecar."""
    issues: list[str] = []
    total_chars = len(text_md)

    # Script sanity ------------------------------------------------------
    latin_count = len(_LATIN.findall(text_md))
    persian_count = len(_PERSIAN_ARABIC.findall(text_md))
    letters = latin_count + persian_count
    latin_ratio = (latin_count / letters) if letters else 0.0
    if latin_ratio > LATIN_RATIO_MAX and "non_persian" not in flags:
        issues.append("high_latin_ratio")

    arabic_codepoints = sum(text_md.count(ch) for ch in _ARABIC_ONLY)
    # A few are fine (Arabic quotations); a body full of them means wrong codepoints.
    if persian_count and arabic_codepoints / max(persian_count, 1) > 0.05:
        issues.append("arabic_codepoints")

    # Degenerate repetition (model looped)
    if total_chars > 200:
        words = text_md.split()
        if words:
            top = Counter(words).most_common(1)[0]
            if len(top[0]) > 2 and top[1] / len(words) > 0.25:
                issues.append("repetition")

    # Embedded-text cross-check ------------------------------------------
    embedded_similarity = None
    length_ratio = None
    if embedded_text:
        emb_bag = word_bag(embedded_text)
        if sum(emb_bag.values()) >= MIN_EMBEDDED_WORDS:
            txt_bag = word_bag(text_md)
            embedded_similarity = round(bag_cosine(txt_bag, emb_bag), 4)
            emb_words = sum(emb_bag.values())
            txt_words = sum(txt_bag.values())
            length_ratio = round(txt_words / emb_words, 3) if emb_words else None
            if embedded_similarity < EMBED_SIM_ESCALATE:
                issues.append("embedded_mismatch")
            if length_ratio is not None and not (LEN_RATIO_BOUNDS[0] <= length_ratio <= LEN_RATIO_BOUNDS[1]):
                issues.append("length_anomaly")

    # Heading cross-checks -------------------------------------------------
    if headings and missing_headings(headings, text_md):
        issues.append("missing_heading")
    if heading_candidates and missing_embedded_headings(heading_candidates, text_md):
        issues.append("embedded_heading_missing")

    # Composite quality score --------------------------------------------
    score = confidence
    if embedded_similarity is not None:
        score = 0.5 * score + 0.5 * embedded_similarity
    score -= 0.1 * len([i for i in issues if i not in ("embedded_mismatch",)])
    score = max(0.0, min(1.0, score))

    return {
        "quality_score": round(score, 4),
        "issues": issues,
        "embedded_similarity": embedded_similarity,
        "length_ratio": length_ratio,
        "latin_ratio": round(latin_ratio, 4),
        "arabic_codepoints": arabic_codepoints,
    }


def needs_escalation(confidence: float, flags: list[str], validators: dict, is_blank: bool) -> bool:
    if is_blank:
        return False
    if confidence < CONF_ESCALATE:
        return True
    if ESCALATE_FLAGS & set(flags):
        return True
    if validators["issues"]:
        return True
    return False


def needs_review(confidence: float, flags: list[str], validators: dict, is_blank: bool) -> bool:
    if is_blank:
        return False
    if confidence < CONF_REVIEW:
        return True
    if REVIEW_FLAGS & set(flags):
        return True
    sim = validators.get("embedded_similarity")
    if sim is not None and sim < EMBED_SIM_REVIEW:
        return True
    review_issues = {"repetition", "arabic_codepoints", "missing_heading", "embedded_heading_missing"}
    if review_issues & set(validators["issues"]):
        return True
    return False
