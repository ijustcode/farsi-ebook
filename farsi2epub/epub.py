"""EPUB 3 builder: assemble reviewed page transcriptions into a final EPUB.

Pipeline:
  1. Load transcribed, non-blank pages in page order.
  2. Normalize each page's Persian text (see `normalize.py`).
  3. Parse each page's markdown into a small internal block model.
  4. Stitch cross-page paragraph splits (continues_next / starts_mid_paragraph).
  5. Split the resulting block stream into chapters (explicit book.yaml
     chapters, or "# " headings).
  6. Render each chapter's blocks to XHTML (headings, paragraphs, verse,
     blockquotes, footnotes/endnotes, tables, lists, inline bold/italic).
  7. Assemble an EPUB 3 package with ebooklib: RTL metadata, embedded
     Vazirmatn fonts, a cover, nav + NCX tables of contents.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Optional

from ebooklib import epub

from .normalize import normalize_persian
from .workspace import Workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"

_FN_REF_RE = re.compile(r"\[\^([^\]]+)\]")
_FN_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:\s?(.*)$")
_LIST_BULLET_RE = re.compile(r"^[-*]\s+(.*)$")
_LIST_NUM_RE = re.compile(r"^\d+\.\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")


# ---------------------------------------------------------------------------
# Loading pages
# ---------------------------------------------------------------------------


def _load_pages(ws: Workspace) -> list[dict]:
    """Load all transcribed, non-blank pages in page order.

    Each entry is {"n": int, "sidecar": dict, "text": str (normalized md)}.
    """
    pages = []
    for n in ws.pages_done():
        sidecar = json.loads(ws.page_meta_path(n).read_text(encoding="utf-8"))
        if sidecar.get("is_blank"):
            continue
        raw_text = ws.page_md_path(n).read_text(encoding="utf-8")
        if not raw_text.strip():
            continue
        pages.append({"n": n, "sidecar": sidecar, "text": normalize_persian(raw_text)})

    if not pages:
        raise RuntimeError(
            f"No transcribed, non-blank pages found for workspace '{ws.slug}'. "
            f"Run `farsi2epub transcribe {ws.slug}` first."
        )
    pages.sort(key=lambda p: p["n"])
    return pages


# ---------------------------------------------------------------------------
# Markdown -> block model (per page)
# ---------------------------------------------------------------------------


def _parse_page_markdown(page_n: int, text: str) -> tuple[list[dict], dict[str, str]]:
    """Parse one page's normalized markdown into a list of blocks plus a map
    of footnote label -> definition text found on that page.

    A "paragraph" is exactly one non-empty line (never hard-wrapped, per the
    transcription contract), so every plain line that isn't part of a
    special construct becomes its own paragraph block.
    """
    lines = text.split("\n")
    blocks: list[dict] = []
    footnote_defs: dict[str, str] = {}
    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        if stripped == "":
            i += 1
            continue

        if stripped == "```verse":
            i += 1
            verse_lines = []
            while i < n and lines[i].strip() != "```":
                verse_lines.append(lines[i].strip())
                i += 1
            i += 1  # skip closing fence (tolerate missing fence at EOF)
            blocks.append({"type": "verse", "page": page_n, "lines": verse_lines})
            continue

        if stripped.startswith("## "):
            blocks.append({"type": "h2", "page": page_n, "text": stripped[3:].strip()})
            i += 1
            continue

        if stripped.startswith("# "):
            blocks.append({"type": "h1", "page": page_n, "text": stripped[2:].strip()})
            i += 1
            continue

        fn_match = _FN_DEF_RE.match(stripped)
        if fn_match:
            footnote_defs[fn_match.group(1)] = fn_match.group(2)
            i += 1
            continue

        if stripped.startswith(">"):
            bq_lines = []
            while i < n and lines[i].strip().startswith(">"):
                content = lines[i].strip()[1:]
                if content.startswith(" "):
                    content = content[1:]
                if content:
                    bq_lines.append(content)
                i += 1
            blocks.append({"type": "blockquote", "page": page_n, "lines": bq_lines})
            continue

        if _TABLE_ROW_RE.match(raw_line):
            rows = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                row = lines[i].strip().strip("|")
                cells = [c.strip() for c in row.split("|")]
                rows.append(cells)
                i += 1
            rows = [r for r in rows if not all(_TABLE_SEP_CELL_RE.match(c) for c in r)]
            blocks.append({"type": "table", "page": page_n, "rows": rows})
            continue

        mb = _LIST_BULLET_RE.match(stripped)
        mn = _LIST_NUM_RE.match(stripped)
        if mb or mn:
            ordered = bool(mn)
            pattern = _LIST_NUM_RE if ordered else _LIST_BULLET_RE
            items = []
            while i < n:
                s2 = lines[i].strip()
                mm = pattern.match(s2)
                if not mm:
                    break
                items.append(mm.group(1))
                i += 1
            blocks.append({"type": "list", "page": page_n, "ordered": ordered, "items": items})
            continue

        # Plain paragraph: exactly one line.
        blocks.append({"type": "p", "page": page_n, "frags": [(page_n, stripped)]})
        i += 1

    return blocks, footnote_defs


# ---------------------------------------------------------------------------
# Cross-page paragraph stitching
# ---------------------------------------------------------------------------


def _stitch_pages(pages: list[dict]) -> tuple[list[dict], dict[int, dict[str, str]], int]:
    """Parse every page and join cross-page paragraph splits.

    Returns (all_blocks, footnote_defs_by_page, joins_made).
    """
    all_blocks: list[dict] = []
    footnote_defs_by_page: dict[int, dict[str, str]] = {}
    joins_made = 0
    prev_n: Optional[int] = None
    prev_sidecar: Optional[dict] = None

    for page in pages:
        n = page["n"]
        sidecar = page["sidecar"]
        blocks, fn_defs = _parse_page_markdown(n, page["text"])
        footnote_defs_by_page[n] = fn_defs

        can_join = (
            all_blocks
            and prev_sidecar is not None
            and prev_n is not None
            and n == prev_n + 1
            and bool(prev_sidecar.get("continues_next"))
            and bool(sidecar.get("starts_mid_paragraph"))
            and all_blocks[-1]["type"] == "p"
            and blocks
            and blocks[0]["type"] == "p"
        )

        if can_join:
            merged = {"type": "p", "frags": all_blocks[-1]["frags"] + blocks[0]["frags"]}
            all_blocks[-1] = merged
            all_blocks.extend(blocks[1:])
            joins_made += 1
        else:
            all_blocks.extend(blocks)

        prev_n = n
        prev_sidecar = sidecar

    return all_blocks, footnote_defs_by_page, joins_made


# ---------------------------------------------------------------------------
# Chapter splitting
# ---------------------------------------------------------------------------


def _block_page(block: dict) -> int:
    if block["type"] == "p":
        return block["frags"][0][0]
    return block["page"]


def _split_by_headings(blocks: list[dict], book_title: str) -> list[tuple[str, list[dict]]]:
    chapters: list[tuple[str, list[dict]]] = []
    current_title = book_title
    current_blocks: list[dict] = []

    for b in blocks:
        if b["type"] == "h1":
            chapters.append((current_title, current_blocks))
            current_title = b["text"]
            current_blocks = [b]
        else:
            current_blocks.append(b)
    chapters.append((current_title, current_blocks))

    return [c for c in chapters if c[1]]


def _split_by_page_numbers(
    blocks: list[dict], chapters_meta: list[dict], book_title: str
) -> list[tuple[str, list[dict]]]:
    starts = sorted(chapters_meta, key=lambda c: c["start_page"])
    buckets: list[list[dict]] = [[] for _ in starts]
    front: list[dict] = []

    for b in blocks:
        p = _block_page(b)
        chosen = None
        for idx, c in enumerate(starts):
            if c["start_page"] <= p:
                chosen = idx
        if chosen is None:
            front.append(b)
        else:
            buckets[chosen].append(b)

    result: list[tuple[str, list[dict]]] = []
    if front:
        result.append((book_title, front))
    for c, bucket in zip(starts, buckets):
        if bucket:
            result.append((c.get("title") or book_title, bucket))
    return result


# ---------------------------------------------------------------------------
# Inline + block rendering to XHTML
# ---------------------------------------------------------------------------


def _escape(s: str) -> str:
    return html_lib.escape(s, quote=False)


def _inline(s: str) -> str:
    """Escape HTML and render **bold** / *italic*."""
    s = _escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    return s


class _ChapterRenderer:
    """Renders one chapter's blocks to an XHTML body fragment, collecting
    footnote references into a per-chapter numbered endnotes section."""

    def __init__(self, footnote_defs_by_page: dict[int, dict[str, str]], chapter_idx: int):
        self._defs_by_page = footnote_defs_by_page
        self._chapter_idx = chapter_idx
        self._counter = 0
        self.entries: list[tuple[str, int, str]] = []  # (fid, number, def_html)

    def _resolve_footnotes(self, rendered_html: str, page: int) -> str:
        def repl(m: re.Match) -> str:
            label = m.group(1)
            self._counter += 1
            num = self._counter
            fid = f"c{self._chapter_idx}-{num}"
            def_raw = self._defs_by_page.get(page, {}).get(label, "")
            def_html = _inline(def_raw) if def_raw else ""
            self.entries.append((fid, num, def_html))
            return (
                f'<a id="fnref-{fid}" href="#fn-{fid}" epub:type="noteref" class="fnref">'
                f"<sup>{num}</sup></a>"
            )

        return _FN_REF_RE.sub(repl, rendered_html)

    def _text(self, page: int, raw: str) -> str:
        return self._resolve_footnotes(_inline(raw), page)

    def render(self, blocks: list[dict]) -> str:
        parts: list[str] = []
        for b in blocks:
            t = b["type"]
            if t == "h1":
                parts.append(f"<h1>{self._text(b['page'], b['text'])}</h1>")
            elif t == "h2":
                parts.append(f"<h2>{self._text(b['page'], b['text'])}</h2>")
            elif t == "p":
                frag_html = [self._text(pg, txt) for pg, txt in b["frags"]]
                parts.append(f"<p>{' '.join(frag_html)}</p>")
            elif t == "verse":
                parts.append(self._render_verse(b))
            elif t == "blockquote":
                inner = "".join(f"<p>{self._text(b['page'], ln)}</p>" for ln in b["lines"])
                parts.append(f"<blockquote>{inner}</blockquote>")
            elif t == "table":
                parts.append(self._render_table(b))
            elif t == "list":
                parts.append(self._render_list(b))
        parts.append(self._render_endnotes())
        return "".join(parts)

    def _render_verse(self, b: dict) -> str:
        lines_html = []
        for line in b["lines"]:
            if not line:
                continue
            if " --- " in line:
                left, right = line.split(" --- ", 1)
                left_h = self._text(b["page"], left.strip())
                right_h = self._text(b["page"], right.strip())
                lines_html.append(
                    f'<p class="v"><span class="m1">{left_h}</span>'
                    f'<span class="m2">{right_h}</span></p>'
                )
            else:
                full_h = self._text(b["page"], line.strip())
                lines_html.append(f'<p class="v"><span class="m-full">{full_h}</span></p>')
        return '<div class="verse">' + "".join(lines_html) + "</div>"

    def _render_table(self, b: dict) -> str:
        rows = b["rows"]
        if not rows:
            return ""
        head, *rest = rows
        thead = "".join(f"<th>{self._text(b['page'], c)}</th>" for c in head)
        tbody = "".join(
            "<tr>" + "".join(f"<td>{self._text(b['page'], c)}</td>" for c in r) + "</tr>"
            for r in rest
        )
        return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"

    def _render_list(self, b: dict) -> str:
        tag = "ol" if b["ordered"] else "ul"
        items = "".join(f"<li>{self._text(b['page'], it)}</li>" for it in b["items"])
        return f"<{tag}>{items}</{tag}>"

    def _render_endnotes(self) -> str:
        if not self.entries:
            return ""
        li_html = "".join(
            f'<li id="fn-{fid}">{def_html} '
            f'<a href="#fnref-{fid}" class="fnback" epub:type="backlink">↩</a></li>'
            for fid, _num, def_html in self.entries
        )
        return (
            '<section class="endnotes" epub:type="footnotes">'
            "<hr/><ol>" + li_html + "</ol></section>"
        )


# ---------------------------------------------------------------------------
# CSS + fonts
# ---------------------------------------------------------------------------

_CSS = """\
@font-face {
  font-family: "Vazirmatn";
  font-weight: normal;
  font-style: normal;
  src: url("../fonts/Vazirmatn-Regular.ttf");
}
@font-face {
  font-family: "Vazirmatn";
  font-weight: bold;
  font-style: normal;
  src: url("../fonts/Vazirmatn-Bold.ttf");
}
body {
  font-family: "Vazirmatn", serif;
  text-align: justify;
  line-height: 1.9;
}
h1, h2 {
  font-family: "Vazirmatn", serif;
  font-weight: bold;
  text-align: center;
}
h1 { font-size: 1.6em; margin: 1.5em 0 1em; }
h2 { font-size: 1.25em; margin: 1.3em 0 0.8em; }
p { margin: 0 0 1em; }
.verse { text-align: center; margin: 1.5em 0; }
.verse .v { display: flex; justify-content: space-between; gap: 2em; margin: 0.3em 0; text-align: right; }
.verse .m1, .verse .m2 { flex: 1 1 0; }
.verse .m2 { text-align: left; }
.verse .m-full { flex: 1 1 100%; text-align: center; }
blockquote {
  margin: 1em 2em;
  padding-inline-start: 1em;
  border-inline-start: 3px solid #999;
  color: #444;
  font-style: italic;
}
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
table th, table td { border: 1px solid #999; padding: 0.4em 0.6em; text-align: right; }
.endnotes { margin-top: 2em; padding-top: 1em; border-top: 1px solid #999; font-size: 0.85em; }
.endnotes ol { padding-inline-start: 1.5em; }
a.fnref sup { font-size: 0.7em; }
a.fnback { text-decoration: none; margin-inline-start: 0.3em; }
ul, ol { margin: 0 0 1em; padding-inline-start: 1.5em; }
"""


def _build_css_item() -> epub.EpubItem:
    return epub.EpubItem(
        uid="style_main",
        file_name="style/style.css",
        media_type="text/css",
        content=_CSS,
    )


def _build_font_items() -> list[epub.EpubItem]:
    items = []
    for uid, fname, media in (
        ("font_regular", "Vazirmatn-Regular.ttf", "font/ttf"),
        ("font_bold", "Vazirmatn-Bold.ttf", "font/ttf"),
    ):
        path = FONTS_DIR / fname
        items.append(
            epub.EpubItem(
                uid=uid,
                file_name=f"fonts/{fname}",
                media_type=media,
                content=path.read_bytes(),
            )
        )
    ofl_path = FONTS_DIR / "OFL.txt"
    if ofl_path.is_file():
        items.append(
            epub.EpubItem(
                uid="font_license",
                file_name="fonts/OFL.txt",
                media_type="text/plain",
                content=ofl_path.read_bytes(),
            )
        )
    return items


# ---------------------------------------------------------------------------
# OPF post-processing safety net (page-progression-direction)
# ---------------------------------------------------------------------------


def _ensure_rtl_spine_in_opf(epub_path: Path) -> None:
    """Verify the built EPUB's OPF spine carries page-progression-direction
    ="rtl"; patch the zip in place if ebooklib didn't write it for some
    reason (defensive -- normally `book.set_direction("rtl")` covers this)."""
    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_name = None
        for name in zf.namelist():
            if name.endswith(".opf"):
                opf_name = name
                break
        if opf_name is None:
            return
        opf_data = zf.read(opf_name).decode("utf-8")

    if 'page-progression-direction="rtl"' in opf_data:
        return

    patched = re.sub(
        r"<spine([^>]*)>",
        lambda m: f'<spine{m.group(1)} page-progression-direction="rtl">'
        if "page-progression-direction" not in m.group(1)
        else m.group(0),
        opf_data,
        count=1,
    )
    if patched == opf_data:
        return  # couldn't find a <spine> tag to patch; leave as-is

    tmp_path = epub_path.with_suffix(".tmp.epub")
    with zipfile.ZipFile(epub_path, "r") as zin, zipfile.ZipFile(
        tmp_path, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == opf_name:
                data = patched.encode("utf-8")
            zout.writestr(item, data)
    tmp_path.replace(epub_path)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "book"


def _run_epubcheck(epub_path: Path) -> None:
    exe = shutil.which("epubcheck")
    if not exe:
        return
    try:
        result = subprocess.run(
            [exe, str(epub_path)], capture_output=True, text=True, timeout=120
        )
    except Exception as exc:  # pragma: no cover - best-effort diagnostics only
        print(f"epubcheck: failed to run ({exc})")
        return
    print("--- epubcheck ---")
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"epubcheck exit code: {result.returncode}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_epub(ws: Workspace) -> Path:
    meta = ws.meta
    slug = meta.get("slug") or ws.slug
    title_fa = meta.get("title_fa") or slug
    author_fa = meta.get("author_fa") or "Unknown"
    title_en = meta.get("title_en")

    pages = _load_pages(ws)
    all_blocks, footnote_defs_by_page, joins_made = _stitch_pages(pages)

    chapters_meta = meta.get("chapters")
    if isinstance(chapters_meta, list) and chapters_meta:
        chapters = _split_by_page_numbers(all_blocks, chapters_meta, title_fa)
    else:
        chapters = _split_by_headings(all_blocks, title_fa)

    if not chapters:
        raise RuntimeError("No content blocks found; cannot build chapters.")

    book = epub.EpubBook()
    book.set_identifier(f"farsi2epub-{slug}")
    book.set_title(title_fa)
    book.set_language("fa")
    book.add_author(author_fa)
    book.set_direction("rtl")

    css_item = _build_css_item()
    book.add_item(css_item)
    font_items = _build_font_items()
    for f in font_items:
        book.add_item(f)

    # Cover.
    cover_added = False
    cover_path = ws.page_image_path(1)
    if cover_path.is_file():
        book.set_cover("images/cover.png", cover_path.read_bytes(), create_page=True)
        cover_added = True

    chapter_items = []
    total_footnotes = 0
    for idx, (chap_title, blocks) in enumerate(chapters, start=1):
        renderer = _ChapterRenderer(footnote_defs_by_page, idx)
        body_html = renderer.render(blocks)
        total_footnotes += len(renderer.entries)

        first_is_h1 = bool(blocks) and blocks[0]["type"] == "h1"
        prefix = "" if first_is_h1 else f"<h1>{_escape(chap_title)}</h1>"

        file_name = f"text/chap_{idx:02d}.xhtml"
        item = epub.EpubHtml(title=chap_title, file_name=file_name, lang="fa")
        item.direction = "rtl"
        item.content = prefix + body_html
        item.add_link(href="../style/style.css", rel="stylesheet", type="text/css")
        book.add_item(item)
        chapter_items.append(item)

    nav = epub.EpubNav()
    nav.direction = "rtl"
    nav.add_item(css_item)
    book.add_item(nav)
    book.add_item(epub.EpubNcx())

    if cover_added:
        cover_page = book.get_item_with_id("cover")
        if cover_page is not None:
            cover_page.direction = "rtl"
            cover_page.is_linear = True
            cover_page.add_item(css_item)

    book.toc = tuple(chapter_items)

    spine: list[Any] = []
    if cover_added:
        spine.append("cover")
    spine.append("nav")
    spine.extend(chapter_items)
    book.spine = spine

    out_dir = ws.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = _safe_filename(title_en or slug) + ".epub"
    out_path = out_dir / out_name

    epub.write_epub(str(out_path), book, {})
    _ensure_rtl_spine_in_opf(out_path)

    pages_used = [p["n"] for p in pages]
    size_kb = out_path.stat().st_size / 1024

    print("EPUB build summary")
    print(f"  Chapters:        {len(chapters)}")
    print(
        f"  Pages used:      {len(pages_used)} "
        f"({pages_used[0]}-{pages_used[-1]})" if pages_used else "  Pages used:      0"
    )
    print(f"  Paragraph joins: {joins_made}")
    print(f"  Footnotes:       {total_footnotes}")
    print(f"  Output:          {out_path} ({size_kb:.1f} KB)")

    _run_epubcheck(out_path)

    return out_path
