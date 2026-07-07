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
LONG_EDGE_STD = 1568
LONG_EDGE_HI = 2576

# Assumed token usage per page (single pass on MODEL_STRONG, the default).
STRONG_TOKENS_IN = 3200
STRONG_TOKENS_OUT = 1000

# Buffer for metadata proposal, occasional retries, and verbose pages.
ESTIMATE_BUFFER = 1.10


def _cost_per_page(model: str, tokens_in: int, tokens_out: int) -> float:
    prices = PRICES[model]
    return (tokens_in / 1_000_000) * prices["in"] + (tokens_out / 1_000_000) * prices["out"]


def estimate_cost(page_count: int) -> float:
    """Estimate the USD cost of transcribing `page_count` pages.

    Assumes one pass per page on MODEL_STRONG (the measured-best default:
    Haiku first passes failed validation on >90% of sampled pages, making
    the two-pass ladder both worse and more expensive), plus a 10% buffer.
    """
    return page_count * _cost_per_page(MODEL_STRONG, STRONG_TOKENS_IN, STRONG_TOKENS_OUT) * ESTIMATE_BUFFER
