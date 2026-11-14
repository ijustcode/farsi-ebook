"""Build a book workspace from the owner's line-by-line text truth (hadpi-articles).

    ./venv/bin/python tests/truth_book.py [--force]
    ./venv/bin/farsi2epub build hadpi-articles-truth

Reads ``books/hadpi-articles/truth/000N.lines.txt`` (pages 1-5; pages 3-5 must
be final audited exports) and writes a separate workspace
``books/hadpi-articles-truth/`` with one Markdown page per printed page, so the
model-transcribed workspace that the experiments use as a baseline stays as it
is. No word is changed: lines are only joined into paragraphs, page-number lines
are dropped, headings are marked, and an unreadable mark becomes ``[؟]``. Makes
no API call and never writes under ``books/hadpi-articles/``.

Paragraph rule (checked by eye on the five pages): a new block starts at a line
beginning with ماده / فصل / تبصره or a numbered item ``N)``; a block ends after
a line ending in ``:`` and after a short line ending in ``.``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC_SLUG = "hadpi-articles"
DST_SLUG = "hadpi-articles-truth"
PAGES = (1, 2, 3, 4, 5)
AUDIT_TOOL_PAGES = (3, 4, 5)
UNREADABLE = "⟦؟⟧"
UNREADABLE_OUT = "[؟]"

_PAGE_NO = re.compile(r"^-\s*[۰-۹0-9]+\s*-$")
_ARTICLE = re.compile(r"^ماده\s*[۰-۹]+\s*\)")
_CHAPTER = re.compile(r"^فصل\s+\S+$")
_NOTE = re.compile(r"^تبصره\b")
_ITEM = re.compile(r"^[۰-۹]+\s*\)")
SHORT = 0.85     # a line under this share of the page's typical full line is "short"


def page_blocks(lines: list[str], first_page: bool) -> list[tuple[str, str]]:
    """[(kind, text)] with kind in h1/h2/h3/p. Words are never altered."""
    lines = [l.strip() for l in lines if l.strip() and not _PAGE_NO.match(l.strip())]
    full = statistics.median(sorted(len(l) for l in lines)[len(lines) // 2:])
    blocks: list[tuple[str, str]] = []
    cur: list[str] = []
    closed = True

    def flush():
        nonlocal cur
        if cur:
            blocks.append(("p", " ".join(cur)))
            cur = []

    i = 0
    if first_page:                      # the two title lines of the document
        blocks.append(("h1", lines[0]))
        blocks.append(("h2", lines[1]))
        i = 2
    while i < len(lines):
        line = lines[i]
        if _CHAPTER.match(line):
            flush()
            blocks.append(("h2", line))
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt and not _ARTICLE.match(nxt) and len(nxt) < 0.4 * full:   # chapter subtitle
                blocks.append(("h3", nxt))
                i += 1
            closed = True
        elif _ARTICLE.match(line) and line.endswith(":"):
            flush()
            blocks.append(("h3", line))
            closed = True
        else:
            if closed or _ARTICLE.match(line) or _NOTE.match(line) or _ITEM.match(line):
                flush()
            cur.append(line)
            closed = line.endswith(":") or (line.endswith(".") and len(line) < SHORT * full)
        i += 1
    flush()
    return blocks


def to_markdown(blocks: list[tuple[str, str]]) -> str:
    mark = {"h1": "# ", "h2": "## ", "h3": "### ", "p": ""}
    return "\n\n".join(mark[k] + t.replace(UNREADABLE, UNREADABLE_OUT) for k, t in blocks) + "\n"


def words_of(text: str) -> list[str]:
    return text.replace(UNREADABLE_OUT, UNREADABLE).replace("#", " ").split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild an existing truth workspace")
    args = ap.parse_args()
    src = ROOT / "books" / SRC_SLUG
    dst = ROOT / "books" / DST_SLUG
    truth = src / "truth"
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} exists; pass --force to rebuild it")
    for page in AUDIT_TOOL_PAGES:
        st = json.loads((truth / f"{page:04d}.text_audit.json").read_text())
        sha = hashlib.sha256((truth / f"{page:04d}.lines.txt").read_bytes()).hexdigest()
        if st.get("status") != "audited" or not st["export"].get("final") or st["export"]["sha256"] != sha:
            raise SystemExit(f"page {page}: truth is not a final audited export")
    (dst / "text").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "source.pdf", dst / "source.pdf")

    import yaml
    meta = yaml.safe_load((src / "book.yaml").read_text(encoding="utf-8"))
    first = [l.strip() for l in (truth / "0001.lines.txt").read_text(encoding="utf-8").split("\n") if l.strip()]
    meta.update({"slug": DST_SLUG, "title_fa": f"{first[0]} {first[1]}", "title_en": "asasnameh-truth",
                 "source_type": "scanned", "page_count": len(PAGES), "chapters": None})
    (dst / "book.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")

    report = {}
    for page in PAGES:
        raw = (truth / f"{page:04d}.lines.txt").read_text(encoding="utf-8")
        lines = raw.split("\n")
        blocks = page_blocks(lines, first_page=(page == 1))
        md = to_markdown(blocks)
        kept = [w for l in lines if l.strip() and not _PAGE_NO.match(l.strip()) for w in l.split()]
        if words_of(md) != kept:
            raise SystemExit(f"page {page}: Markdown words differ from the truth words; refusing")
        (dst / "text" / f"{page:04d}.md").write_text(md, encoding="utf-8")
        side = {"page": page, "model_used": "owner_truth", "source": f"books/{SRC_SLUG}/truth/{page:04d}.lines.txt",
                "truth_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "is_blank": False, "confidence": 1.0, "quality_score": 1.0, "needs_review": False,
                "flags": ["illegible"] if UNREADABLE in raw or "…" in raw else [],
                "headings": [t for k, t in blocks if k != "p"],
                "starts_mid_paragraph": False, "continues_next": False, "cost_usd": 0.0,
                "reviewed": "owner_truth"}
        (dst / "text" / f"{page:04d}.json").write_text(json.dumps(side, ensure_ascii=False, indent=1), encoding="utf-8")
        report[page] = {"words": len(kept), "blocks": len(blocks), "headings": len(side["headings"]),
                        "unreadable": raw.count(UNREADABLE) + raw.count("…")}
    print(json.dumps(report, indent=1))
    print(f"wrote {dst}; now: ./venv/bin/farsi2epub build {DST_SLUG}")


if __name__ == "__main__":
    main()
