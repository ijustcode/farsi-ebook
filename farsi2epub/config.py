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
