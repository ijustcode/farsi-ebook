"""Regression checks for legacy RTL PDF text-layer phrase location.

Usage: source venv/bin/activate && python tests/locator_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farsi2epub.locate import Query, _fold_word, _locate_match, locate_queries  # noqa: E402


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

    # Real image-only scan regression. This page has zero PDF text words; the
    # phrase appears on the first short body line beneath the opening paragraph.
    root = Path(__file__).resolve().parent.parent / "books" / "bachehaye_ghali"
    if (root / "source.pdf").is_file():
        md = (root / "text" / "0008.md").read_text(encoding="utf-8")
        scan_box = locate_queries(
            root / "source.pdf", 8, md, [Query("بداالله دنبال خر بود")]
        )[0]
        assert scan_box is not None
        assert scan_box["source"] == "scan"
        assert 0.44 < scan_box["y0"] < 0.51
        assert scan_box["x1"] - scan_box["x0"] < 0.60

        # A finding with no verifier bbox still gets deterministic geometry.
        md21 = (root / "text" / "0021.md").read_text(encoding="utf-8")
        no_model_box = locate_queries(
            root / "source.pdf", 21, md21, [Query("کشیدبه طرف کاهدان")]
        )[0]
        assert no_model_box is not None and no_model_box["source"] == "scan"

    # Keep the original digital-PDF failure fixed as the scan tier evolves.
    digital_root = Path(__file__).resolve().parent.parent / "books" / "boof-e-koor"
    if (digital_root / "source.pdf").is_file():
        md46 = (digital_root / "text" / "0046.md").read_text(encoding="utf-8")
        digital_box = locate_queries(
            digital_root / "source.pdf",
            46,
            md46,
            [Query("خواهربرادر شیری بودیم")],
        )[0]
        assert digital_box is not None and digital_box["source"] == "match"
        assert 0.33 < digital_box["x0"] < 0.35
        assert 0.58 < digital_box["x1"] < 0.60
    print("ALL LOCATOR REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
