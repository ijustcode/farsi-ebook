"""Project-wide constants and cost estimation for farsi2epub."""

from __future__ import annotations

MODEL_FAST = "claude-haiku-4-5"
MODEL_STRONG = "claude-sonnet-5"

# USD per million tokens (MTok).
PRICES = {
    MODEL_FAST: {"in": 1.0, "out": 5.0},
    MODEL_STRONG: {"in": 2.0, "out": 10.0},
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
