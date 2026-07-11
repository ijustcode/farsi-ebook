# farsi2epub

Convert Farsi (Persian) PDF books into clean, right-to-left EPUB 3 ebooks. PDF text extraction is notoriously unreliable for Persian — broken word ordering, mangled ligatures, lost ZWNJ — so instead of extracting text, farsi2epub renders each page to an image and has a Claude vision model transcribe it. Every page is then checked by deterministic validators, optionally reviewed by an LLM QC pass and a human in a local web UI, and finally assembled into an RTL EPUB with embedded Vazirmatn fonts. It works on scanned and digital PDFs alike.

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (transcription and QC call Claude models)
- Optional: `epubcheck` on your PATH — if present, it runs automatically after each build to validate the EPUB

## Installation

Clone the repo, create a virtual environment at `venv/`, and install the package in editable mode:

```bash
git clone <repo-url> farsi-ebook
cd farsi-ebook
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Then provide your API key, either in your environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

or in a `.env` file at the project root:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

## Processing a Farsi PDF

The pipeline is four commands, run in order. Each one operates on a per-book workspace under `books/<slug>/`, so you can stop and resume at any point. The examples below use a book slugged `my-book`.

### 1. Analyze

```bash
farsi2epub analyze my-book.pdf --slug my-book
```

This creates the workspace at `books/my-book/`, classifies the PDF (digital, scanned, or mixed), counts pages, and prints a cost estimate for the default hi-res transcription plus a cheaper economy estimate. If you only want part of the book, pass `--pages` (e.g. `--pages 3-120`) and the range is saved for later steps.

### 2. Transcribe

```bash
farsi2epub transcribe my-book
```

Each page is rendered to an image and transcribed by Claude, several pages at a time. Pages that already have output are skipped, so re-running the command resumes an interrupted job — use `--force` to redo pages. When it finishes (in a terminal) it offers to run QC right away.

Useful options:

- `--max-cost 5.0` — abort if the estimated cost exceeds this many dollars
- `--pages 10-50` — transcribe only these pages
- `--concurrency 4` — pages transcribed in parallel (default 4)
- `--res std` — the economy path: ~30% cheaper per page; pages that fail validation are automatically retried at hi-res, so quality degrades safely

After this step, `books/my-book/text/` holds one Markdown file per page plus a JSON sidecar with quality scores and flags.

### 3. QC and review

```bash
farsi2epub qc my-book            # auto: LLM verifier pass over risky pages
farsi2epub review my-book        # local web UI for human correction
```

Auto QC runs an LLM verifier over the pages most likely to have problems and attaches suggested corrections — it never overwrites the transcription on its own. `review` then opens a local web app showing each flagged page image next to its editable Markdown, with the QC suggestion as a diff you can apply, edit, or reject. Only the worst flagged pages are surfaced by default, so review stays quick.

Both steps are optional — you can go straight to `build` — but they catch the errors that matter most.

### 4. Build

```bash
farsi2epub build my-book
```

This stitches the per-page Markdown into chapters, normalizes the Persian text conservatively, and writes the finished EPUB to `books/my-book/out/`. If `epubcheck` is installed, the result is validated automatically.

## Cost notes

- The default is **Sonnet at hi-res input** (2576 px page images). This was measured, not guessed: cheaper models failed validation on most pages, and standard-resolution input produced character-level errors (reversed dotted abbreviations, word transpositions, lost diacritics) that hi-res avoided.
- `transcribe --res std` is the supported economy path (~30% cheaper per page). It suits crisp, large-print sources or very long books, and failing pages escalate to hi-res automatically.
- `analyze` prints an up-front estimate for both modes, and `--max-cost` on `transcribe` acts as a hard safety cap mid-run.

## Workspace layout

Everything for one book lives under `books/<slug>/`:

```
books/my-book/
├── source.pdf        # copy of the input PDF
├── book.yaml         # metadata: title, author, page count/range, optional chapters
├── pages/            # rendered page images (hires/ subfolder for 2576px renders)
├── text/
│   ├── 0001.md       # transcribed Markdown, one file per page
│   └── 0001.json     # sidecar: model, confidence, quality score, flags, cost
└── out/              # the finished EPUB
```

A page counts as done when both its `.md` and `.json` exist; `transcribe` skips done pages unless you pass `--force`.

### Chapters

By default, chapters are split wherever a top-level `# ` heading appears in the transcription. If the book's headings are unreliable, you can list chapters explicitly in `book.yaml` and they take precedence:

```yaml
chapters:
  - title: "فصل اول"
    start_page: 9
  - title: "فصل دوم"
    start_page: 34
```

## How it works

Each pipeline stage communicates only through the workspace files, so any stage can be rerun independently. Transcription uses structured output with detailed Persian typography rules (correct ی/ک codepoints, ZWNJ, verse and footnote formatting). Quality is scored deterministically by comparing the transcription against the PDF's embedded text where available, and low-scoring pages are escalated to a higher-resolution pass before being flagged for human review. The build step joins paragraphs across page breaks, splits chapters, and assembles an RTL EPUB 3 with embedded Vazirmatn fonts.
