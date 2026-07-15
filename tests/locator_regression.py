"""Regression checks for legacy RTL PDF text-layer phrase location.

Usage: source venv/bin/activate && python tests/locator_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farsi2epub.locate import _fold_word, _locate_match  # noqa: E402


class _Page:
    rect = fitz.Rect(0, 0, 420, 595)


def main() -> int:
    # Presentation-form glyphs must survive normalization. Previously they
    # folded to an empty string, leaving punctuation as the only line token.
    assert _fold_word("ﺧﻮﺍﻫﺮﺑﺮﺍﺩﺭ") == "خواهربرادر"

    # This mirrors the problematic PDF's extraction: three visible words are
    # seven fragments, and one fragment has RTL-reversed character order.
    fragments = [
        (fitz.Rect(238, 82, 246, 102), "خ"),
        (fitz.Rect(196, 82, 236, 102), "واهربرادر"),
        (fitz.Rect(168, 82, 194, 102), "یریش"),
        (fitz.Rect(162, 82, 166, 102), "ب"),
        (fitz.Rect(151, 82, 160, 102), "ود"),
        (fitz.Rect(146, 82, 151, 102), "ی"),
        (fitz.Rect(141, 82, 146, 102), "م"),
        (fitz.Rect(136, 82, 141, 102), "،"),
    ]
    box = _locate_match(_Page(), fragments, "خواهربرادر شیری بودیم", None)
    assert box is not None
    assert box["source"] == "match"
    assert abs(box["x0"] - 141 / 420) < 1e-9
    assert abs(box["x1"] - 246 / 420) < 1e-9
    print("ALL LOCATOR REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
