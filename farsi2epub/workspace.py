"""On-disk workspace layout for a single book being converted."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOOKS_ROOT = PROJECT_ROOT / "books"


class Workspace:
    """Represents the on-disk working directory for one book (`books/<slug>/`)."""

    def __init__(self, slug: str, books_root: Path = DEFAULT_BOOKS_ROOT):
        self.slug = slug
        self.books_root = Path(books_root)
        self._meta: Optional[dict] = None

    # -- paths -----------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.books_root / self.slug

    @property
    def pdf_path(self) -> Path:
        return self.root / "source.pdf"

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def hires_dir(self) -> Path:
        return self.pages_dir / "hires"

    @property
    def text_dir(self) -> Path:
        return self.root / "text"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def meta_path(self) -> Path:
        return self.root / "book.yaml"

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        pdf_path: str | Path,
        slug: str,
        books_root: Path = DEFAULT_BOOKS_ROOT,
    ) -> "Workspace":
        """Create a new workspace, copying `pdf_path` in as source.pdf."""
        ws = cls(slug, books_root=books_root)
        for d in (ws.root, ws.pages_dir, ws.hires_dir, ws.text_dir, ws.review_dir, ws.out_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(pdf_path), str(ws.pdf_path))
        return ws

    @classmethod
    def load(cls, slug: str, books_root: Path = DEFAULT_BOOKS_ROOT) -> "Workspace":
        ws = cls(slug, books_root=books_root)
        if not ws.root.is_dir() or not ws.meta_path.is_file():
            raise FileNotFoundError(
                f"No workspace found for slug '{slug}' under {ws.books_root}. "
                f"Run `farsi2epub analyze <pdf> --slug {slug}` first."
            )
        return ws

    # -- metadata ------------------------------------------------------

    @property
    def meta(self) -> dict:
        if self._meta is None:
            if self.meta_path.is_file():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._meta = yaml.safe_load(f) or {}
            else:
                self._meta = {}
        return self._meta

    def save_meta(self, data: dict) -> None:
        self._meta = data
        with open(self.meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    # -- page file paths -------------------------------------------------

    @staticmethod
    def _pad(n: int) -> str:
        return f"{n:04d}"

    def page_image_path(self, n: int) -> Path:
        return self.pages_dir / f"{self._pad(n)}.png"

    def page_hires_path(self, n: int) -> Path:
        return self.hires_dir / f"{self._pad(n)}.png"

    def page_md_path(self, n: int) -> Path:
        return self.text_dir / f"{self._pad(n)}.md"

    def page_meta_path(self, n: int) -> Path:
        return self.text_dir / f"{self._pad(n)}.json"

    def page_orig_path(self, n: int) -> Path:
        """Backup of the pre-review transcription (written on first human edit)."""
        return self.text_dir / f"{self._pad(n)}.orig.md"

    def pages_done(self) -> list[int]:
        """Page numbers that have both a transcribed .md and a .json sidecar."""
        done = []
        if not self.text_dir.is_dir():
            return done
        for md_file in self.text_dir.glob("*.md"):
            if not md_file.stem.isdigit():
                continue  # e.g. NNNN.orig.md review backups (stem "NNNN.orig")
            n = int(md_file.stem)
            if self.page_meta_path(n).is_file():
                done.append(n)
        return sorted(done)


def parse_pages_spec(spec: Optional[str], page_count: int) -> list[int]:
    """Parse a page spec like "5", "3-10", "3-", "-20", "1,3-5" into a sorted
    list of unique 1-based page numbers, clamped to [1, page_count].

    A falsy `spec` (None or "") yields all pages 1..page_count.
    """
    if not spec:
        return list(range(1, page_count + 1))

    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str) if start_str.strip() else 1
            end = int(end_str) if end_str.strip() else page_count
        else:
            start = end = int(part)
        start = max(1, start)
        end = min(page_count, end)
        for n in range(start, end + 1):
            pages.add(n)
    return sorted(pages)
