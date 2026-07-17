"""Claude vision calls: page transcription and book-metadata proposal."""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

import anthropic
import pydantic
from pydantic import BaseModel, Field

from .config import PRICES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_OUTPUT_TOKENS = 8000

T = TypeVar("T")


def _call_with_retry(fn: Callable[[], T], *, page_no: int, what: str, attempts: int = 3, backoff_s: float = 2.0) -> T:
    """Call fn() up to `attempts` times, retrying only on transient/parse failures.

    Retries on anthropic.APIStatusError with status_code in {429, 500, 502, 503, 529},
    anthropic.APIConnectionError, and pydantic.ValidationError (malformed JSON from
    the model's raw output). Everything else (401/403/400 etc.) propagates
    immediately on first occurrence. Re-raises the last exception once attempts
    are exhausted.
    """
    last_exc: BaseException = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except anthropic.APIStatusError as exc:
            if exc.status_code not in {429, 500, 502, 503, 529}:
                raise
            last_exc = exc
        except anthropic.APIConnectionError as exc:
            last_exc = exc
        except pydantic.ValidationError as exc:
            last_exc = exc
        if attempt < attempts:
            print(f"  page {page_no}: {what} attempt {attempt} failed ({last_exc}); retrying...", file=sys.stderr)
            time.sleep(backoff_s * attempt)
    raise last_exc


class PageTranscription(BaseModel):
    is_blank: bool = Field(description="True if the page has no body text (blank, or pure image/decoration).")
    text_md: str = Field(description="The transcribed body text as Markdown. Empty string when is_blank.")
    headings: list[str] = Field(description="Verbatim headings emitted on this page, in order.")
    flags: list[str] = Field(
        description='Zero or more of: "image", "table", "illegible", "two_column", "marginalia", "non_persian", "decorative_only".'
    )
    starts_mid_paragraph: bool = Field(
        description="True if the first body line continues a paragraph cut off on the previous page."
    )
    continues_next: bool = Field(
        description="True if the last paragraph is cut off mid-sentence at the bottom of the page."
    )
    confidence: float = Field(
        description="Honest 0.0-1.0 estimate that the transcription is both character-accurate AND complete. Below 0.8 means parts were hard to read, or you were unable to fit or transcribe all visible body text."
    )


class BookMetadata(BaseModel):
    title_fa: Optional[str] = Field(description="Book title in Persian, exactly as printed on the title page.")
    author_fa: Optional[str] = Field(description="Author name in Persian, or null if not visible.")
    title_en: str = Field(
        description="Short Latin transliteration or translation of the title, lowercase, words separated by hyphens, usable as a filename slug."
    )
    publisher_fa: Optional[str] = Field(description="Publisher in Persian, or null.")


TRANSCRIBE_SYSTEM = """You are an expert transcriber of Persian (Farsi) books. You receive one page of a book as an image and transcribe its body text into clean Markdown for conversion into an ebook. Accuracy is paramount: the reader will see your output instead of the printed page.

TRANSCRIPTION RULES
- Transcribe the Persian text exactly as printed: same words, same orthography, same punctuation («», ،, ؛, ؟). Preserve diacritics (تشدید، حرکات) only where clearly printed; never add your own.
- Use standard Persian codepoints: always ی (U+06CC) and ک (U+06A9), never Arabic ي or ك.
- Use zero-width non-joiner (U+200C) for detached affixes as Persian orthography requires: می‌رود، کتاب‌ها، بزرگ‌تر، فعّالیت‌های.
- Keep Persian digits (۰۱۲۳۴۵۶۷۸۹) as printed.
- Remove kashida/tatweel stretching: write بهتر even if printed بـــهـــتر.
- Do not modernize spelling, correct perceived typos, translate, or summarize. Transcribe.
- Completeness is as important as accuracy: transcribe every line of body text and every dialogue exchange visible on the page, however long the paragraph or how many speakers trade lines. Never omit, truncate, or silently drop a paragraph, verse line, or dialogue turn — including ones that continue for many lines or wrap around the page.

IGNORE COMPLETELY (never transcribe):
- Running headers and footers: a running header is a short line at the extreme top or bottom margin repeating the book or chapter title on page after page, plus page numbers wherever they appear. A title in the middle of the page, or sitting directly above new content, is NOT a running header — it is a real heading and must be transcribed.
- Decoration: floral ornaments, frames, rules, corner flourishes, background images or watermarks behind the text. But text INSIDE a decorative box, banner, or ornamented strip is a heading, not decoration: transcribe the text, drop the ornament.
- Photographs and illustrations: do not describe them; just add the flag "image".

MARKDOWN STRUCTURE
- Headings use exactly three levels:
  - `# ...` — chapter or page-level titles: framed/ornamented chapter openers (combine label and title on one line, e.g. `# درس سیزدهم: یادحسین (ع)`) and standalone titles naming the whole page (e.g. فهرست مطالب on a table-of-contents page).
  - `## ...` — section titles within a chapter, including boxed or banner-style strips (e.g. اطلاعات تکمیلی inside a decorated frame).
  - `### ...` — minor standalone labels heading a short list or exercise block (e.g. تمرین دوم).
- CRITICAL heading invariant: every string you list in the `headings` field MUST also appear in text_md as a `#`, `##`, or `###` line, and every heading line in text_md must be listed in `headings`. The field mirrors the markdown; it is never a substitute for emitting the heading. Never emit a heading as a plain or **bold** paragraph.
- Body paragraphs separated by one blank line. Never hard-wrap: each paragraph is a single line of output, no matter how long.
- Poetry/verse: emit a fenced block with language tag `verse`; one verse line (بیت) per output line, the two hemistichs (مصراع) separated by ` --- `:
  ```verse
  مصراع اول --- مصراع دوم
  ```
  A line with a single centered hemistich has no separator.
- Free verse / modern poetry (شعر نو — short lines, centered or staggered, no hemistich pairs): also a ```verse block, one printed line per output line, no ` --- ` separators. Never flatten poem lines into a prose paragraph.
- Footnotes: mark references in the text as [^1] and put definitions at the very end of the page as `[^1]: متن پانوشت`.
- Lists as Markdown lists. Simple tables as Markdown tables (also add flag "table").
- Quranic verses, prayers, or Arabic passages: transcribe as printed (Arabic orthography allowed there); if visually set off from the body, use a `>` blockquote.

PAGE-BOUNDARY JUDGMENT
- starts_mid_paragraph: true when the first body line visibly continues a sentence from the previous page (no paragraph indentation, begins mid-sentence).
- continues_next: true when the page's last paragraph is cut off mid-sentence at the bottom.

FIELDS
- headings: the exact heading strings you emitted as `#`/`##`/`###` lines in text_md (without the # marks), in order; empty list if none. This list and the heading lines in text_md must match one-to-one.
- is_blank: true for pages with no body text at all; then text_md must be "".
- flags: subset of "image", "table", "illegible", "two_column", "marginalia", "non_persian", "decorative_only".
- confidence: honest 0.0-1.0 estimate covering both character accuracy and completeness. Use values below 0.8 whenever print quality, scan blur, or unusual typography made you unsure of any words, or whenever you could not confirm you transcribed every visible line of body text."""

METADATA_SYSTEM = """You are looking at the opening page(s) of a Persian (Farsi) PDF. Determine whether they form a cover or title page (صفحهٔ عنوان) and, only if so, extract the book's bibliographic metadata exactly as printed.

Critical: many PDFs are excerpts that begin mid-book. If these pages are body pages — a lesson or chapter opening, running prose, a table of contents — they are NOT a title page: return null for title_fa, author_fa, and publisher_fa. Never report a chapter or lesson heading (e.g. درس سیزدهم) as the book title, and never guess an author that is not explicitly printed as the author.

title_en: a short lowercase Latin transliteration of the title with hyphens between words (a filename slug), e.g. jame-shenasi-khodemani; if title_fa is null, derive it from whatever the pages suggest the work is, or use "unknown"."""


def load_env() -> None:
    """Load ANTHROPIC_API_KEY from a project-root .env file if not already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def get_client() -> anthropic.Anthropic:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in your environment "
            "or put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the project root."
        )
    return anthropic.Anthropic()


def cost_of(usage, model: str) -> float:
    prices = PRICES.get(model)
    if prices is None:
        return 0.0
    return (usage.input_tokens / 1_000_000) * prices["in"] + (usage.output_tokens / 1_000_000) * prices["out"]


def _image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("ascii"),
        },
    }


def transcribe_page(
    client: anthropic.Anthropic,
    png_bytes: bytes,
    model: str,
    page_no: int,
    page_count: int,
    title_hint: Optional[str] = None,
    extra_hint: Optional[str] = None,
) -> tuple[PageTranscription, dict, float]:
    """Transcribe one page image. Returns (result, usage_dict, cost_usd).

    `extra_hint` is appended to the user message; used for corrective retries
    (e.g. listing headings a previous attempt omitted from text_md).
    """
    hint = f"کتاب: {title_hint}" if title_hint else "عنوان کتاب نامشخص است."
    user_text = (
        f"صفحهٔ {page_no} از {page_count}. {hint}\n"
        "Transcribe this page following the system instructions."
    )
    if extra_hint:
        user_text += f"\n{extra_hint}"
    kwargs: dict = {}
    if model.startswith("claude-sonnet-5"):
        # Perception task, not reasoning: keep Sonnet's default adaptive thinking off.
        kwargs["thinking"] = {"type": "disabled"}

    def _call():
        response = client.messages.parse(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=TRANSCRIBE_SYSTEM,
            messages=[{"role": "user", "content": [_image_block(png_bytes), {"type": "text", "text": user_text}]}],
            output_format=PageTranscription,
            **kwargs,
        )
        if response.parsed_output is None:
            raise RuntimeError(f"Model {model} returned unparseable output for page {page_no}")
        return response

    response = _call_with_retry(_call, page_no=page_no, what="transcribe")
    result = response.parsed_output
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return result, usage, cost_of(response.usage, model)


class QCIssue(BaseModel):
    type: str = Field(
        description='One of: "missing_heading", "wrong_heading_level", "wrong_word", "zwnj_error", '
        '"heh_boundary", "footnote_marker", "digit_error", "punctuation", "missing_text", '
        '"extra_text", "verse_structure", "table_structure", "other".'
    )
    description: str = Field(
        description="Short explanation of the problem written entirely in natural Persian (Farsi)."
    )
    snippet: str = Field(description="The affected text as it currently appears in the transcription (verbatim excerpt).")
    bbox: Optional[list[int]] = Field(
        default=None,
        description="Approximate bounding box [x0, y0, x1, y1] of the affected region on the page image, "
        "in 0-1000 normalized coordinates (origin top-left, x rightward, y downward). "
        "null if the issue cannot be localized.",
    )


class QCReport(BaseModel):
    verdict: str = Field(description='"pass" if the transcription is faithful, "fail" if it has any real issues.')
    issues: list[QCIssue] = Field(description="All real issues found; empty when verdict is pass.")
    suggested_text_md: Optional[str] = Field(
        description="When verdict is fail: the complete corrected page Markdown, preserving everything "
        "that was already right. null when verdict is pass."
    )


QC_SYSTEM = """You are a meticulous quality-control verifier for Persian (Farsi) book transcriptions. You receive one page of a book as an image, followed by the current Markdown transcription of that page. Compare them character by character and report real discrepancies.

VERIFY IN PARTICULAR
- Completeness: check that every paragraph, dialogue exchange, and verse line visible in the image is present in the transcription — read the whole page image top to bottom and confirm nothing was skipped, even a single missing line or a dialogue turn. Report any gap as a "missing_text" issue, with the surrounding text as the snippet and a bbox for the missing region when you can localize it.
- Headings: every chapter/page title, boxed or banner section title (even inside decorative frames), and bold standalone label on the page must appear as a `#` / `##` / `###` line. Report missing or wrongly-leveled headings. Running headers at the extreme top/bottom margin and page numbers are correctly omitted — do not report those.
- Words containing ه at a joining boundary (e.g. علاقه‌مند، خانه‌ها): verify the exact letters and the zero-width non-joiner (U+200C) usage against the image.
- Footnotes: every superscript marker printed in the body must appear as [^n] in the text, with a matching [^n]: definition at the end of the page.
- Persian codepoints (ی/ک never ي/ك), digits as printed, punctuation («», ،, ؛, ؟), diacritics only where printed.
- Poetry must be in ```verse blocks (one بیت per line, hemistichs separated by ` --- `); tables as Markdown tables; body paragraphs never hard-wrapped.

RULES
- Only report actual discrepancies against the image; do not restyle, modernize, or "improve" faithful text.
- Minor stylistic judgment calls are not issues. Uncertainty about a blurry word is an issue only if the transcription is likely wrong.
- Write every issue `description` entirely in natural Persian (Farsi), using Persian sentence structure and punctuation. Do not write English prose or repeat the English issue `type` in the description. Use Persian terms such as «نیم‌فاصله» instead of English technical labels. Text quoted from the transcription may remain exactly as it appears.
- Quotation marks: Persian guillemets «…» are the standard Persian quotation marks; differences or conversions between «…», "…", '…' or other quote styles are NEVER an issue — do not report them and do not change quote characters in suggested corrections.
- verdict "pass" requires zero real issues; otherwise "fail" with every issue listed.
- suggested_text_md: only when verdict is "fail" — the full corrected page Markdown. Change ONLY what is wrong; keep all correct text byte-for-byte identical.
- bbox: for every issue, estimate its bounding box on the page image — the box containing the affected text, as [x0, y0, x1, y1] scaled 0-1000 (origin top-left, x rightward, y downward). A loose box covering the right line(s) is fine. Use null only when the issue has no specific location on the page (e.g. missing_text at an unknown spot)."""


def qc_verify_page(
    client: anthropic.Anthropic,
    png_bytes: bytes,
    text_md: str,
    model: str,
    page_no: int,
) -> tuple[QCReport, dict, float]:
    """Verify one page's transcription against its image. Returns (report, usage_dict, cost_usd)."""
    user_text = (
        f"صفحهٔ {page_no}. Current transcription of this page:\n\n{text_md}\n\n"
        "Verify this transcription against the page image per the system instructions."
    )
    kwargs: dict = {}
    if model.startswith("claude-sonnet-5"):
        kwargs["thinking"] = {"type": "disabled"}
    def _call():
        response = client.messages.parse(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=QC_SYSTEM,
            messages=[{"role": "user", "content": [_image_block(png_bytes), {"type": "text", "text": user_text}]}],
            output_format=QCReport,
            **kwargs,
        )
        if response.parsed_output is None:
            raise RuntimeError(f"QC model {model} returned unparseable output for page {page_no}")
        return response

    response = _call_with_retry(_call, page_no=page_no, what="qc")
    report = response.parsed_output
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return report, usage, cost_of(response.usage, model)


def propose_metadata(
    client: anthropic.Anthropic, png_pages: list[bytes], model: str
) -> tuple[Optional[BookMetadata], float]:
    """Propose book metadata from the first page image(s)."""
    content: list[dict] = [_image_block(b) for b in png_pages]
    content.append({"type": "text", "text": "Extract the book metadata from these opening pages."})
    response = client.messages.parse(
        model=model,
        max_tokens=1000,
        system=METADATA_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_format=BookMetadata,
    )
    return response.parsed_output, cost_of(response.usage, model)
