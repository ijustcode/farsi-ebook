# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`farsi2epub` converts Farsi (Persian) PDF books into RTL EPUB 3 ebooks using Claude vision models to transcribe page images. It is a CLI (`click`-based) installed as an editable package.

## Commands

A virtual environment exists at `venv/`; activate it before any Python command:

```bash
source venv/bin/activate
pip install -e .          # editable install (entry point: farsi2epub)
```

The pipeline is four sequential CLI steps, each operating on a per-book workspace identified by a slug:

```bash
farsi2epub analyze <pdf> [--slug s] [--pages 3-10] [--force]   # create workspace, classify PDF, estimate cost
farsi2epub transcribe <slug> [--pages ...] [--force] [--max-cost N] [--concurrency 4] [--model ...] [--qc auto|manual|skip|ask]
farsi2epub qc <slug> [--mode auto|manual] [--all] [--yes]      # auto = LLM verifier pass (suggest-only), manual = review UI
farsi2epub review <slug>     # local web UI for human correction of flagged pages
farsi2epub build <slug>      # assemble EPUB into books/<slug>/out/
```

There is no test suite or linter configured. `epubcheck` is run automatically after `build` if it's on PATH (optional).

Transcription requires `ANTHROPIC_API_KEY`, read from the environment or from `.env` at the project root (see `llm.load_env`).

## Workspace layout (the on-disk contract)

Everything revolves around `books/<slug>/` (gitignored), managed by `workspace.Workspace`:

- `source.pdf` — copied input
- `book.yaml` — metadata: title/author (LLM-proposed on first transcribe), `page_count`, optional `page_range`, optional `chapters: [{title, start_page}]` list that overrides heading-based chapter splitting at build time
- `pages/NNNN.png` — standard-res renders (long edge 1568 px); `pages/hires/` — 2576 px renders used on escalation
- `text/NNNN.md` — transcribed Markdown per page; `text/NNNN.json` — sidecar with model used, confidence, quality score, flags, `needs_review`, `starts_mid_paragraph`/`continues_next`, cost
- `out/` — final EPUB

A page counts as "done" when both its `.md` and `.json` exist (`Workspace.pages_done`); `transcribe` skips done pages unless `--force`.

## Pipeline architecture

Each module is one stage; they communicate only through the workspace files above, so stages can be rerun independently:

1. **`render.py`** — PyMuPDF rasterization plus `classify_pdf` (digital/scanned/mixed by embedded-text density) and `extract_embedded_text` (the validation oracle for digital PDFs).
2. **`transcribe.py` + `llm.py`** — per-page vision transcription via `client.messages.parse` with Pydantic `output_format` (`PageTranscription`). Pages are independent and run in a thread pool; cross-page paragraph joins are deferred to build time using the `continues_next`/`starts_mid_paragraph` sidecar fields. The elaborate Persian transcription rules (ی/ک codepoints, ZWNJ, verse blocks with ` --- ` hemistich separators, footnote syntax) live in `TRANSCRIBE_SYSTEM` in `llm.py` — the build step's parser depends on this output format. **Headings contract**: every string in the structured `headings` field must also appear in `text_md` as a `#`/`##`/`###` line; `validators.evaluate` enforces this (`missing_heading` issue), `transcribe.py` auto-promotes matching bold lines and retries once with a corrective hint on failure.
3. **`validators.py`** — deterministic quality scoring because LLM self-confidence is poorly calibrated. Core trick: compare char-sorted word multisets (cosine similarity) between the transcription and PyMuPDF's embedded text, which is immune to the extractor's broken word ordering and ligatures. Thresholds at the top of the file were measured empirically; a faithful digital-page transcription scores ~0.99, so 0.97 is already the escalation cutoff. Failing pages escalate to a hi-res render on the strong model; still-failing pages get `needs_review: true`.
4. **`qc.py`** — post-transcription QC. Auto mode runs an LLM verifier (`llm.qc_verify_page`) over risk-selected pages, writing findings + a suggested correction into the sidecar `qc` key (suggest-only — never overwrites `text_md`; the human accepts/edits/rejects in the review UI). Maintains `qc_history.json` at the project root (gitignored): append-only issue events with feature tags, from which per-feature lift weights bias future risk selection and review ordering. Schemas/contracts live in `TODO.md`.
5. **`review.py`** — stdlib `http.server` + jinja2 single-page web app (no external assets) showing page image beside editable Markdown, validator/QC findings, and the QC suggestion diff with an Apply button. Review budget: only the worst `ceil(total/5)` flagged pages are surfaced (bypass with `budget_all`); the rest are auto-accepted with `review_skipped: true`. Human edits and suggestion outcomes are recorded as QC history events.
6. **`epub.py`** — parses page Markdown into a block model, stitches cross-page paragraphs, splits chapters (by `book.yaml` `chapters` page numbers, else by `# ` headings), renders XHTML, and assembles an RTL EPUB with embedded Vazirmatn fonts (`assets/fonts/`). `normalize.py` runs first: conservative-only normalization (Arabic→Persian letterforms, kashida removal, punctuation spacing) — it deliberately never touches digits, ZWNJ placement, or newlines.

## Model choices (measured, don't casually revert)

- Default transcription model is Sonnet (`MODEL_STRONG` in `config.py`), not Haiku: measured Haiku first-pass failure was >90%, making the Haiku→Sonnet escalation ladder both worse and more expensive. Haiku is still used for the cheap metadata proposal and available via `--model`.
- Default input resolution is **hi-res (2576px, Sonnet 5's vision maximum)**, not the 1568px standard render: a std-res Sonnet pass was measured making character-level errors hi-res avoided (bidi-reversed dotted abbreviations like ه.ق → ق.ه, word transpositions, lost diacritics) — errors the word-bag validators are structurally blind to. `transcribe --res std` is the documented economy path (~30% cheaper/page; suits crisp large-print sources or very long books) and failing pages escalate to Sonnet + hi-res, so it degrades safely. Sidecars record `resolution` used per page.
- Sonnet is called with `thinking: {"type": "disabled"}` for transcription — it's a perception task, and adaptive thinking adds cost without accuracy.
- Cost is estimated in `config.estimate_cost` and tracked per page in sidecars; `--max-cost` aborts a run mid-flight.
