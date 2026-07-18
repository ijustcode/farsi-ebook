# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

`farsi2epub` converts Farsi (Persian) PDF books into RTL EPUB 3 ebooks using Claude vision models to transcribe page images. It is a CLI (`click`-based) installed as an editable package.

## Commands

A virtual environment exists at `venv/`; activate it before any Python command:

```bash
source venv/bin/activate
pip install -e .          # editable install (entry point: farsi2epub)
```

### Shell completion

click 8 auto-generates tab completion for the `farsi2epub` entry point. Add to `~/.zshrc` (or `~/.bashrc`, using `bash_source`):

```bash
eval "$(_FARSI2EPUB_COMPLETE=zsh_source farsi2epub)"
```

The CLI stages operate on a per-book workspace identified by a slug; QC and human review are optional:

```bash
farsi2epub analyze <pdf> [--slug s] [--pages 3-10] [--force]   # create workspace, classify PDF, estimate cost
farsi2epub transcribe <slug> [--pages ...] [--force] [--max-cost N] [--concurrency 4] [--model ...] [--qc auto|manual|skip|ask] [--yes]  # --yes skips the auto-QC cost prompt
farsi2epub qc <slug> [--mode auto|manual] [--all] [--yes] [--force] [--pages ...]  # auto = LLM verifier pass (suggest-only), manual = review UI
farsi2epub review <slug> [--all] [--reset] [--background|--status|--stop] [--no-bbox-refine] [--bbox-refine-model ...]  # local web UI for human correction; -b detaches the server
farsi2epub build <slug>      # assemble EPUB into books/<slug>/out/
```

`epubcheck` is run automatically after `build` if it's on PATH (optional).

### Tests

No pytest/linter is configured. `tests/` holds three standalone scripts (run with `./venv/bin/python`, no API key needed except where noted); they operate on real transcribed books under `books/`, so they need a book already processed:

```bash
./venv/bin/python tests/locator_regression.py             # locate.py tiers + VLM strip-alignment; one live llm.read_strips check skipped without a key
./venv/bin/python tests/headings_regression.py            # golden pages: headings survive md → EPUB XHTML, incl. negative control
./venv/bin/python tests/bbox_eval.py <slug> [--no-refine] # renders an HTML contact sheet of review-UI boxes for human miss-counting; uncached refinement bills the API
```

Transcription, auto-QC, and optional VLM bbox refinement require `ANTHROPIC_API_KEY`, read from the environment or from `.env` at the project root (see `llm.load_env`). Review remains usable offline without bbox refinement.

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
4. **`qc.py`** — post-transcription QC. Auto mode runs an LLM verifier (`llm.qc_verify_page`) over risk-selected pages, writing findings + a suggested correction into the sidecar `qc` key (suggest-only — never overwrites `text_md`; the human decides per correction in the review UI). Each finding carries an optional `bbox` ([x0,y0,x1,y1] in 0–1000 image coordinates) estimated by the verifier. Pages with a pending suggestion are skipped on re-runs unless `qc --force`. Maintains `qc_history.json` at the project root (gitignored): append-only issue events with feature tags, from which per-feature lift weights bias future risk selection and review ordering. Schemas/contracts live in `TODO.md`.
5. **`review.py`** — stdlib `http.server` + jinja2 single-page web app (no external assets) showing page image beside editable Markdown, validator/QC findings, and the QC suggestion broken into per-correction **hunks** (whitespace-preserving token diff via `_derive_hunks`; context-anchored so hunks apply independently in any order — do not replace with a plain `split()` diff, it fails round-trip on real data). Each hunk gets Approve/Edit/Reject/Undo; page data is embedded as `<script type="application/json">` payloads (`_json_for_script` escapes `<`). Findings the verifier flagged without a usable fix — no suggestion, a suggestion identical to the current text, or an issue not linked to any hunk — still become an **edit-only** correction (synthesized by `_derive_finding_hunks`, id offset `100000`): Approve/Reject are disabled and only Edit is offered, prefilled with the flagged snippet; the Accept path records these as per-finding `human_edit` events. Located boxes are overlaid on the page image by a **tiered locator** in `locate.py` (`locate_queries`, memoized in `review._locate_queries_cached` with `page_md` in the key so an Accept invalidates stale boxes): Tier A **"match"** (blue) fuzzy-scores normalized word windows against `page.get_text('words')` — immune to RTL intra-word reordering (char-sorted equality) and Arabic↔Persian letterform variance (folding), gated per page on ≥30% Arabic-block words; Tier B **"layout"** (teal) maps the snippet's markdown char-offset onto the PDF's cumulative per-line char counts, so it works even on non-Unicode **cipher** text layers whose line rects are still real geometry (gated on a 0.5–2.0 pdf/md char-count ratio); Tier C **"scan"** (green) binarizes the rendered page, detects printed line and word-gap geometry, and maps the Markdown offset onto those ink regions, so image-only scans need no OCR or font model. Scan-sourced boxes are then upgraded to **"scan_vlm"** (dark green) via `locate.refine_scan_boxes`: a clean strip crop of the hit line(s) ±1 is transcribed line-by-line in one batched VLM call per page (`llm.read_strips`, Sonnet default, `review --bbox-refine-model` overrides), the query — plus the hunk's corrected text as a `Query.alts` alternate, since a wrong-word snippet is exactly what the image does *not* say — is re-aligned onto the detected word rects (line ID becomes exact, within-line placement a word-index lookup). Wrapped matches preserve ordered per-line `segments` plus a compatibility union envelope; the review UI renders those segments as one SVG overlay with complementary ripped edges at each RTL line break and no connector. Failures widen once to ±2 lines, then keep the plain scan box. Refinement never blocks the render path: `GET /` serves instantly using only already-cached results (`_ScanBoxRefiner.apply_cached`), a 4-worker background pipeline warms the cache in review order from server startup, and the browser polls `GET /boxes/<page>` (~2.5s, gated on a `BBOX_REFINE` flag; a muted RTL chip marks pages still pending) to swap upgraded boxes + payload geometry in place. Every outcome, including failures, is cached per book in versioned `locate_vlm.json` keys based on page text (Accept invalidates; negative entries never re-bill; geometry-version changes intentionally re-refine once); `--no-bbox-refine` or a missing API key skips refinement and review stays fully offline-usable. The locating query is capped to ~5 words (`_cap_words`) and the layout/scan tiers narrow each hit line to the covered word-rect sub-run (`_line_word_extent`/`_scan_line_extent`), so a box stays a zoomable fraction of the line rather than spanning it. Each tier self-gates per page; falling through all three yields the verifier's **"model"** `bbox` estimate (orange, dashed). Clicking any segment or its linked hunk/issue smooth-zooms to `segments[0]` (the upper-line beginning in Persian reading order), pulses the complete logical overlay, and marks the correction active; while zoomed, click-drag pans the image and a plain click (or Escape / background-click / re-click) zooms back out. Box plumbing stays 0–1 page fractions server-side → CSS percentages/SVG viewBox coordinates on `.img-wrap`. Selection surfaces pages flagged by either check: `needs_review` (validator- or QC-set) plus unreviewed pages whose validator `issues` stayed below the needs_review threshold. Review budget: only the worst `ceil(total/5)` flagged pages are surfaced (`review --all` bypasses); the rest are auto-accepted with `review_skipped: true`. A single **Accept** button persists the textarea to `text/NNNN.md` — the same file `build` reads — backing up the pre-edit text once to `text/NNNN.orig.md` (`Workspace.pages_done` ignores non-numeric stems for this reason) and reporting the per-correction tally (approved/edited/rejected/undecided). Per-hunk outcomes are recorded as QC history events; `review --reset` undoes a book's human decisions (keeps text edits) and strips its human events from history.
6. **`epub.py`** — parses page Markdown into a block model, stitches cross-page paragraphs, splits chapters (by `book.yaml` `chapters` page numbers, else by `# ` headings), renders XHTML, and assembles an RTL EPUB with embedded Vazirmatn fonts (`assets/fonts/`). `normalize.py` runs first: conservative-only normalization (Arabic→Persian letterforms, kashida removal, punctuation spacing) — it deliberately never touches digits, ZWNJ placement, or newlines.

## Model choices (measured, don't casually revert)

- Default transcription model is Sonnet (`MODEL_STRONG` in `config.py`), not Haiku: measured Haiku first-pass failure was >90%, making the Haiku→Sonnet escalation ladder both worse and more expensive. Haiku is still used for the cheap metadata proposal and available via `--model`.
- Default input resolution is **hi-res (2576px, Sonnet 5's vision maximum)**, not the 1568px standard render: a std-res Sonnet pass was measured making character-level errors hi-res avoided (bidi-reversed dotted abbreviations like ه.ق → ق.ه, word transpositions, lost diacritics) — errors the word-bag validators are structurally blind to. `transcribe --res std` is the documented economy path (~30% cheaper/page; suits crisp large-print sources or very long books) and failing pages escalate to Sonnet + hi-res, so it degrades safely. Sidecars record `resolution` used per page.
- Sonnet is called with `thinking: {"type": "disabled"}` for transcription — it's a perception task, and adaptive thinking adds cost without accuracy.
- Cost is estimated in `config.estimate_cost` and tracked per page in sidecars; `--max-cost` aborts a run mid-flight.
