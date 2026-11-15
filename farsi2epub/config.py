"""Project-wide constants and cost estimation for farsi2epub."""

from __future__ import annotations

MODEL_FAST = "claude-haiku-4-5"
MODEL_STRONG = "claude-sonnet-5"
# Second reader for hard pages only (farsi2epub/hard_reader.py): pages with no
# embedded-text oracle that still trip the review gate after the normal pass.
# Measured on held-out typewritten pages at 2.5% word error against 13% for
# the normal pass (docs/bbox-old-fonts-plan.md, Step 2). About $0.09 a page.
MODEL_HARD = "claude-opus-5-5"
HARD_PAGE_COST_USD = 0.09
# Default hard-page reader since 2026-11-15 (owner decision D22): Gemini 4
# Carbon on the same bands and instruction. Held-out typewritten pages: 2.1%
# word error against 2.5% for Opus, at about a quarter of the price; as printed
# (ZWNJ, spacing) it is worse than Opus. Opus stays selectable.
MODEL_HARD_GEMINI = "gemini-4-carbon"
HARD_READER_DEFAULT = "gemini"
# First (whole-page, Markdown) reader. Gemini since 2026-11-15 (owner decision
# D23) when a Gemini key is present, otherwise MODEL_STRONG. Development pages,
# same instruction and image: typewritten hadpi p1-p2 3.3-4.6% word error
# against 17-40% for Sonnet; clean majid pages 1.0-1.2% against 0-0.9%, strict
# 1.0-1.5% against 0.3-2.1% (it keeps ZWNJ in this mode); about half the price.
MODEL_FIRST_DEFAULT = MODEL_HARD_GEMINI
HARD_READERS = {"gemini": MODEL_HARD_GEMINI, "opus": MODEL_HARD}
# Google's listed prices (USD per MTok, read 2026-11-12): $2/$10 until
# 2026-11-30, then $4/$20; thinking is billed as output.
GEMINI_PRICE_CHANGE = "2026-12-01"
GEMINI_PRICES = ({"in": 2.0, "out": 10.0}, {"in": 4.0, "out": 20.0})


def gemini_price(today: str | None = None) -> dict:
    import datetime
    today = today or datetime.date.today().isoformat()
    return GEMINI_PRICES[0] if today < GEMINI_PRICE_CHANGE else GEMINI_PRICES[1]


def hard_page_cost(reader: str) -> float:
    """Rough cost of one hard-page reading, for messages and budget checks."""
    if reader == "opus":
        return HARD_PAGE_COST_USD
    return 0.025 * gemini_price()["in"] / GEMINI_PRICES[0]["in"]
# Book-level rule: hard pages come in whole documents, and the first reader's
# confidence moves between runs (a held-out typewritten page reported 0.75).
# When at least this share of a book's pages without embedded text trip the
# review gate, the rest of them are re-read too. Stored first passes,
# 2026-11-14: the typewritten document 4-5 of 5 pages, eight other books 0-29%.
# One half at first (D20); one fifth by owner decision on 2026-11-15 (D21),
# after he read re-read pages of an old-font printed book and found them much
# better: that takes in the two books at 27% and 29% and leaves the rest (0-4%).
HARD_BOOK_SHARE = 0.2
HARD_BOOK_MIN_PAGES = 3

# USD per million tokens (MTok).
PRICES = {
    MODEL_FAST: {"in": 1.0, "out": 5.0},
    MODEL_STRONG: {"in": 2.0, "out": 10.0},
    MODEL_HARD: {"in": 4.0, "out": 20.0},
}

# Long-edge pixel targets when rendering PDF pages to images.
# 2576 px is claude-sonnet-5's maximum vision resolution (no API downscaling);
# 1568 px was the pre-high-res ceiling and remains the economical option.
LONG_EDGE_STD = 1568
LONG_EDGE_HI = 2576

# Vision input is billed in visual tokens, not encoded bytes. Rule from the
# Claude platform docs (build-with-claude/vision, "Resolution and token cost",
# and vision-coordinates, "How Claude resizes and pads images"; read
# 2026-11-11): an image costs ceil(w/28) * ceil(h/28) tokens after the API
# shrinks it to the largest aspect-preserving size that fits its tier's
# long-edge and visual-token limits. Tiers are (max long edge, max tokens):
# high-resolution (Claude 4.7 and later, incl. claude-sonnet-5) and standard.
IMAGE_PATCH_PX = 28
IMAGE_TIERS = ((2576, 4784), (1568, 1568))
MAX_IMAGE_TOKENS = max(tokens for _, tokens in IMAGE_TIERS)


def _patch_tokens(width: int, height: int) -> int:
    return -(-width // IMAGE_PATCH_PX) * -(-height // IMAGE_PATCH_PX)


def _resized_size(width: int, height: int, max_edge: int, max_tokens: int) -> tuple[int, int]:
    """The documented reference resize (vision-coordinates), before padding."""
    def fits(w, h):
        return (-(-w // IMAGE_PATCH_PX) * IMAGE_PATCH_PX <= max_edge
                and -(-h // IMAGE_PATCH_PX) * IMAGE_PATCH_PX <= max_edge
                and _patch_tokens(w, h) <= max_tokens)

    if fits(width, height):
        return width, height
    if height > width:
        h, w = _resized_size(height, width, max_edge, max_tokens)
        return w, h
    aspect = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect), 1)


def image_tokens(width: int, height: int) -> int:
    """Upper bound on the visual tokens one image costs, on any model tier.

    The maximum over both tiers, so the bound holds whichever tier the
    configured model is on, and never exceeds MAX_IMAGE_TOKENS.
    """
    if width < 1 or height < 1:
        return MAX_IMAGE_TOKENS
    return min(MAX_IMAGE_TOKENS, max(_patch_tokens(*_resized_size(width, height, edge, tokens))
                                     for edge, tokens in IMAGE_TIERS))


# Resolution modes. "hi" is the default: a std-res Sonnet pass was measured
# making character-level errors that the hi-res pass avoided (bidi-reversed
# dotted abbreviations like ه.ق -> ق.ه, word transpositions, lost diacritics)
# on ordinary book print. "std" costs ~30% less per page and escalates failing
# pages to hi-res, so it suits crisp large-print sources or very long books
# where the per-page saving matters (transcribe --res std).
RES_HI = "hi"
RES_STD = "std"
LONG_EDGE_BY_RES = {RES_HI: LONG_EDGE_HI, RES_STD: LONG_EDGE_STD}

# Measured token usage per page on MODEL_STRONG, by input resolution.
# A 2576px page image costs up to ~4784 image tokens vs ~1568 at std res;
# totals below include the system prompt and were measured on real pages.
TOKENS_IN = {RES_HI: 6800, RES_STD: 3200}
TOKENS_OUT = 1000

# Buffer for metadata proposal, occasional retries, and verbose pages.
ESTIMATE_BUFFER = 1.10


def _cost_per_page(model: str, tokens_in: int, tokens_out: int) -> float:
    prices = PRICES[model]
    return (tokens_in / 1_000_000) * prices["in"] + (tokens_out / 1_000_000) * prices["out"]


def estimate_cost(page_count: int, resolution: str = RES_HI) -> float:
    """Estimate the USD cost of transcribing `page_count` pages.

    Assumes one pass per page on MODEL_STRONG at the given resolution (the
    measured-best default: Haiku first passes failed validation on >90% of
    sampled pages, making the two-pass model ladder both worse and more
    expensive; and hi-res input avoids character-level misreads that the
    std-res pass exhibited), plus a 10% buffer.
    """
    return page_count * _cost_per_page(MODEL_STRONG, TOKENS_IN[resolution], TOKENS_OUT) * ESTIMATE_BUFFER
