"""Heading regression test: single-page EPUB builds from real transcriptions.

For each golden case (a page that historically lost its headings), build a
one-page test workspace from the CURRENT transcription under books/<slug>/,
then assert the expected headings survive end to end:

  1. in the page markdown, as a heading line of the expected level;
  2. in the built EPUB's XHTML, inside the corresponding <hN> tag.

A negative control tampers a copy (deletes the heading line) and asserts the
EPUB-level check then FAILS — proving the test can actually catch a miss.

Matching uses validators.word_bag cosine (>= 0.85) so ZWNJ/ligature variants
don't false-flag. No API calls; runs on whatever is currently transcribed.

Usage:  source venv/bin/activate && python tests/headings_regression.py
Test workspaces are (re)created under books/test-headings-* (gitignored).
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farsi2epub import epub as epub_mod  # noqa: E402
from farsi2epub import validators  # noqa: E402
from farsi2epub.workspace import Workspace  # noqa: E402

# (source slug, page, [(level_marks, heading text), ...])
CASES = [
    ("book2", 3, [("##", "اطلاعات تکمیلی"), ("###", "تمرین دوم"), ("###", "تمرین سوم")]),
    ("jamee", 4, [("#", "فهرست مطالب")]),
]

_TAG_RE = re.compile(r"<[^>]+>")


def _bags_match(a: str, b: str) -> bool:
    return validators.bag_cosine(validators.word_bag(a), validators.word_bag(b)) >= 0.85


def _make_test_workspace(src_slug: str, page: int, test_slug: str, tamper_level_text: tuple[str, str] | None = None) -> Workspace:
    """One-page workspace copied from books/<src_slug>/, optionally with the
    given heading line deleted from the markdown (negative control)."""
    src = Workspace.load(src_slug)
    ws = Workspace(test_slug)
    if ws.root.is_dir():
        shutil.rmtree(ws.root)
    ws.text_dir.mkdir(parents=True)
    ws.out_dir.mkdir(parents=True)

    text = src.page_md_path(page).read_text(encoding="utf-8")
    if tamper_level_text is not None:
        marks, heading = tamper_level_text
        lines = [
            ln for ln in text.split("\n")
            if not (ln.strip().startswith(f"{marks} ") and _bags_match(ln.strip()[len(marks) + 1:], heading))
        ]
        assert len(lines) < len(text.split("\n")), "tamper found nothing to delete"
        text = "\n".join(lines)
    ws.page_md_path(page).write_text(text, encoding="utf-8")
    shutil.copyfile(src.page_meta_path(page), ws.page_meta_path(page))

    ws.save_meta({
        "slug": test_slug,
        "title_fa": f"تست سرصفحه ({src_slug} ص {page})",
        "author_fa": "regression test",
        "title_en": test_slug,
        "language": "fa",
        "source_type": src.meta.get("source_type"),
        "page_count": page,
        "chapters": None,
    })
    return ws


def _md_heading_level(ws: Workspace, page: int, heading: str) -> str | None:
    """The '#'-marks of the markdown heading line matching `heading`, or None."""
    text = ws.page_md_path(page).read_text(encoding="utf-8")
    for line in text.split("\n"):
        m = re.match(r"^(#{1,3}) (.+)$", line.strip())
        if m and _bags_match(m.group(2), heading):
            return m.group(1)
    return None


def _epub_heading_level(epub_path: Path, heading: str) -> str | None:
    """The hN tag ('#'-marks form) containing `heading` in the EPUB's chapter
    XHTML, or None. Ignores the injected chapter-title <h1> prefix by matching
    on text, not position."""
    with zipfile.ZipFile(epub_path) as zf:
        for name in zf.namelist():
            if "chap_" not in name or not name.endswith(".xhtml"):
                continue
            content = zf.read(name).decode("utf-8")
            for m in re.finditer(r"<h([123])[^>]*>(.*?)</h\1>", content, re.DOTALL):
                inner = _TAG_RE.sub("", m.group(2))
                if _bags_match(inner, heading):
                    return "#" * int(m.group(1))
    return None


def main() -> int:
    failures: list[str] = []

    for src_slug, page, expected in CASES:
        ws = _make_test_workspace(src_slug, page, f"test-headings-{src_slug}")
        out_path = epub_mod.build_epub(ws)
        print(f"\n[{src_slug} p{page}] -> {out_path}")
        for marks, heading in expected:
            md_level = _md_heading_level(ws, page, heading)
            epub_level = _epub_heading_level(out_path, heading)
            ok = md_level == marks and epub_level == marks
            status = "OK  " if ok else "FAIL"
            print(f"  {status} {marks} {heading}  (md: {md_level}, epub: {epub_level})")
            if not ok:
                failures.append(f"{src_slug} p{page}: {heading} expected {marks}, md={md_level}, epub={epub_level}")

    # Negative control: delete a heading and require the check to fail.
    nc_slug, nc_page, nc_marks, nc_heading = "book2", 3, "##", "اطلاعات تکمیلی"
    ws = _make_test_workspace(nc_slug, nc_page, "test-headings-negctl", tamper_level_text=(nc_marks, nc_heading))
    out_path = epub_mod.build_epub(ws)
    still_there = _epub_heading_level(out_path, nc_heading)
    print(f"\n[negative control] deleted '{nc_marks} {nc_heading}' -> epub finds: {still_there}")
    if still_there is None:
        print("  OK   tampered heading is absent from the EPUB, so the checks above are load-bearing")
    else:
        failures.append("negative control: deleted heading still found — the test proves nothing")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL HEADING REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
