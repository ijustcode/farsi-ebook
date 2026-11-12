# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`farsi2epub` converts Farsi (Persian) PDF books into RTL EPUB 3 ebooks using Claude vision models to transcribe page images. It is a CLI (`click`-based) installed as an editable package.

## Box placement: current state and plan

- **Current engine (experimental, never promoted):** two geometry sources only, `match` (embedded PDF text) then `scan` (image evidence through `scan.PlacementService`). Contracts, limits and history: `docs/bbox-consolidation.md`, `docs/geometry-best-attempt.md` (current coverage-first policy), `docs/page-geometry-coverage-status.md` (superseded strict-evidence experiment). The old `layout`/`scan_vlm` tiers, the QC-coordinate "model" fallback, `context_anchor_v1`/`legacy_v1` selection and the `locate_vlm*.json` caches are **removed from production**; they survive only as frozen evaluation snapshots under `tests/historical/`. Never restore them as live fallbacks.
- **Approved plan for the next engine:** `docs/bbox-old-fonts-plan.md` (v2, owner decisions in §10). Only its Step 0 is implemented (below). Do not start later steps without the owner.
- **Box counts are availability, not accuracy.** No audited held-out comparison has run. Do not call a box count accuracy or an instrument change a locator improvement.

## Commands

A virtual environment exists at `venv/`; activate it before any Python command (`python` may still resolve to pyenv; `./venv/bin/python` is safest):

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
farsi2epub transcribe <slug> [--pages ...] [--force] [--max-cost N] [--concurrency 4] [--model ...] [--res hi|std] [--qc auto|manual|skip|ask] [--yes] [--bbox-mode auto|offline] [--bbox-model ...] [--bbox-max-cost 5]
farsi2epub geometry <slug> [--pages ...] [--bbox-mode auto|offline] [--bbox-model ...] [--bbox-max-cost 5]   # (re)acquire page maps only; no transcription or QC
farsi2epub qc <slug> [--mode auto|manual] [--all] [--yes] [--force] [--pages ...]  # auto = LLM verifier pass (suggest-only), manual = review UI
farsi2epub review <slug> [--pages ...] [--wait-for-boxes] [--all] [--reset] [--background|--status|--stop] [--bbox-mode auto|offline] [--bbox-model ...] [--bbox-max-cost 5]  # local web UI; -b detaches
farsi2epub build <slug>      # assemble EPUB into books/<slug>/out/
```

`epubcheck` is run automatically after `build` if it's on PATH (optional). Hidden deprecated review flags: `--bbox-refine/--no-bbox-refine` (aliases for `--bbox-mode auto/offline`), `--bbox-refine-model` (alias for `--bbox-model`); `--bbox-refine-algorithm` is rejected with a migration error.

**Scoping and budget defaults are per-stage and do not compose — verified in `cli.py`, not inferred:**

- `analyze --pages` writes `page_range` into `book.yaml`, and `transcribe` and `geometry` fall back to it. Passing `--pages` explicitly at each stage is the safer habit.
- `transcribe` **also acquires page geometry** for every requested, transcribed page (even already-transcribed ones) before optional QC, with `--bbox-mode auto` by default. That is paid image-reader work under its own `--bbox-max-cost` (default $5), separate from `--max-cost`. Use `--bbox-mode offline` to avoid it.
- `transcribe --qc auto` does **not** QC every page it transcribed. It calls `qc.run_qc(...)` without `all_pages`, so QC risk-selects: forced (needs_review or any validator issue) + risky (`risk_score >= RISK_THRESHOLD`) + a `RANDOM_SAMPLE_FRAC` sample of clean pages. Full coverage is only reachable via `farsi2epub qc --mode auto --all`.
- `--max-cost` guards **transcription only**, as a running total of actual spend checked as pages complete — in-flight pages still finish, so it can overshoot by up to `--concurrency` pages. Auto-QC and geometry acquisition bill separately and are not counted against it.
- `--bbox-max-cost` is per process (one review session, one `geometry`/`transcribe` run). Reservations are made before each reader request and are a worst-case upper bound (`scan.reader_reservation`): each crop's visual tokens from its PNG pixel dimensions under the documented rule (`config.image_tokens`: ⌈w/28⌉·⌈h/28⌉ after the API's resize, capped at 4,784 per image), plus the prompt/schema bytes and a fixed 16,000-token allowance, plus the full `REGION_OUTPUT_TOKENS` output, ×1.10. A 4-crop batch reserves about $0.08–0.10 on `claude-sonnet-5`, mostly the 4,096-token output cap, so reservations still exceed typical actual spend (~10×) but four concurrent pages hold well under $0.50.
- `review --pages 61,128` scopes both the UI and acquisition to those transcribed pages and bypasses risk selection. Without `--pages`, review surfaces only the worst `ceil(total/5)` flagged pages unless `--all` (the rest are auto-accepted with `review_skipped: true`). Pages already accepted have `needs_review: false` and no pending suggestion, so they are not surfaced again without `--pages` or `--reset`.
- `review --wait-for-boxes` finishes placement for every surfaced page before opening the browser (a readiness guarantee, not an accuracy one). Stop an existing review before starting one with `--pages` or `--wait-for-boxes`. Review binds to `127.0.0.1`; use the printed URL.
- Review in `auto` mode falls back to `offline` when no API key is available.
- Auto-QC re-runs **skip** pages whose previous suggestion is still pending; `--force` replaces it.

Transcription, QC and `--bbox-mode auto` require `ANTHROPIC_API_KEY`, read from the environment or from `.env` at the project root (see `llm.load_env`).

### Tests

No pytest/linter is configured. Tests are standalone scripts run with `./venv/bin/python tests/<name>.py`. None needs an API key.

Current engine (synthetic fixtures; no `books/` needed):

```bash
./venv/bin/python tests/consolidated_bbox_regression.py      # placement contracts, caches, claims, budgets, CLI aliases
./venv/bin/python tests/page_geometry_regression.py          # correction-independent page maps, resumability, CLI scope
./venv/bin/python tests/geometry_best_attempt_regression.py  # estimated placements, migration, offline replay
./venv/bin/python tests/no_evidence_placement_regression.py  # Step 0: zero-reading pages -> no box, reason, banner
./venv/bin/python tests/review_gate_regression.py            # Step 0/T3: needs_review gate on no-oracle (scanned) pages
./venv/bin/python tests/review_pages_regression.py           # review --pages scope, --wait-for-boxes
./venv/bin/python tests/box_truth_regression.py              # review feedback capture: records, split, endpoints (uses node if present)
./venv/bin/python tests/synth_pages_regression.py            # synthetic page generator: determinism, truth transforms
./venv/bin/python tests/box_audit_regression.py              # box-audit packet: contract, save/resume, undo, audited rule, HTTP
./venv/bin/python tests/real_truth_eval_regression.py        # real-page truth scorer: metrics, freeze/held-out refusals, exclusions
./venv/bin/python tests/transcription_experiments_regression.py # Step 2 scorer: WER foldings, T2 flags, band stitching, spend guard
./venv/bin/python tests/bbox_metrics_regression.py           # v6 instrument (tests/bbox_metrics.py)
./venv/bin/python tests/bbox_benchmark_regression.py         # benchmark export contracts
./venv/bin/python tests/bbox_annotation_workflow_regression.py
./venv/bin/python tests/bbox_development_pilot_regression.py # scan66 pilot replay (tracked fixture)
./venv/bin/python tests/bbox_line_metrics_regression.py
```

Historical instrument and old locator (import the frozen modules in `tests/historical/`; several need processed books under `books/`):

```bash
./venv/bin/python tests/word_miss_smoke.py          # exact v5 reproduction of the 240 frozen boxes and metrics
./venv/bin/python tests/bbox_judge_regression.py    # v5 judge line alignment, saved-box p64 scores (0/0/3)
./venv/bin/python tests/locator_regression.py       # historical locator tiers; live check permanently disabled
./venv/bin/python tests/review_cache_regression.py  # historical review/strip-cache contracts
./venv/bin/python tests/headings_regression.py      # headings survive md -> EPUB; needs the `book2` workspace
```

`tests/bbox_eval.py` (historical contact sheet) can make paid calls unless `--no-refine`. `tests/bbox_capture.py`, `bbox_benchmark.py`, `bbox_annotation_workflow.py`, `bbox_detector_eval.py` are offline tooling described in `docs/bbox-consolidation.md`.

## Review feedback capture and synthetic benchmark (plan Step 1)

- **`box_truth.jsonl`** (project root, gitignored, append-only; `farsi2epub/feedback.py`): what the reviewer says about boxes. Review adds per-correction "on target" / "Wrong line" / "Fix box" (an explicit drawing mode: drag to draw, drag again per wrapped line, Shift-drag pans, Esc cancels) / "Undo mark", an optional header "Spot check" of 5 randomly sampled boxes per session, and a `text_edit` record per changed correction on Accept. Records carry the page image SHA-256, correction key (same id as `tests/bbox_benchmark.py`), the box that was shown with engine versions, and the human verdict/rectangles. Latest record wins per (image, correction, family); undo appends a `retract`. Each page is fixed to `dev` or `heldout` by a hash of its image SHA: **tune on `dev` only, gate on `heldout` only, and compute precision from spot-check records only** (fix/confirm records are biased). Overrides: `FARSI2EPUB_BOX_TRUTH_PATH`, `FARSI2EPUB_SPOT_SEED`. Feedback observes; it never changes engine output.
- `./venv/bin/python tests/box_truth_replay.py` replays the store against the current engine offline (it may write derived box caches under the book's `locate_evidence/`).
- **Synthetic benchmark** (`tests/synth_pages.py`; `synth_smoke_1a.py`, `synth_smoke_1b.py`): lays out Persian text with PyMuPDF in local fonts, degrades the image, and carries exact word/line truth through the same transform. Output goes to `out/synth/`; only the manifest is tracked. **Grade against each word's `ink_aabb`, not its layout box** (layout boxes miss ink in nastaliq and DecoType Naskh). Synthetic pages are easier than real photographed ones; real pages remain the gate. Hashes are tied to PyMuPDF 1.28.0 and the system font files.

## Real-page geometry truth: the box-audit packet (plan §6.2)

- `tests/box_audit.py` (+ `box_audit_draft.py`, `box_audit_ui.py`) drafts word rectangles for the five truth pages and serves a local audit tool: `./venv/bin/python tests/box_audit.py serve [--port 8770] [--open]` (127.0.0.1 only, stdlib, no browser unless `--open`). Other commands: `status`, `draft --slug S --page N [--force]`, `import-pilot`, `overlay`, `export`, `synth-eval`.
- Pages: hadpi-articles p1, p2 (text from the owner's `truth/NNNN.lines.txt`), ghessehaye-majid p23 and p109 (text from `text/NNNN.md`, words distributed over detected lines), bachehaye_ghali p66 (the audited pilot fixture, imported, labelled "previously audited"). **hadpi p3–p5 are held out**: the tool refuses to draft them; never tune on them.
- Files (private, under `books/<slug>/truth/`): `NNNN.geometry.draft.json` (untouched draft, never rewritten by the tool), `NNNN.geometry.audit.json` (owner's version, created from the draft on first open, saved atomically after every action), `NNNN.geometry.audit.undo.jsonl` (undo stack), `NNNN.page.png` (audit image rendered from the PDF at the embedded scan's native resolution when finer than `pages/hires/`).
- Record shape (`farsi2epub.box-audit/v1`): the `bbox_annotation_workflow` line/word records (`line_id`, `word_id`, `order`, `text`, normalised `rect`) plus a per-line frame (centre, angle) with line/word `box` in that frame, derived `quad` (TL, TR, BR, BL), word `state` draft/accepted/fixed, `uncertain` + reasons, line `text_flag`. `export` validates an audited page with `bbox_annotation_workflow.validate_lines`. Page ids match `tests/bbox_benchmark.py`.
- Moving a word across a line break (E/B) never carries its box: the giving line drops only the moved word's box, the receiving line is re-split inside its unchanged outline (draft, uncertain). A word box centred outside its line's outline is refused. R first shrinks an outline that a stray box inflated. `serve --root DIR` runs on a copy.
- A page is `audited` only when every line was explicitly accepted with as many boxes as truth words (or a flagged text problem); any later edit re-opens it.
- Draft quality is availability, not truth: synthetic typewriter pages are much easier than the real ones, and `uncertain` flags only order the review. Grade later steps against the audited files only.
- Test: `./venv/bin/python tests/box_audit_regression.py` (synthetic, no `books/`).

- **Hard typewriter presets** (`tw_photo_medium`, `tw_photo_hard`; pipeline v2 in `tests/synth_pages.py`): margin specks at line starts (plan §13 O1), word gaps tightened to or below the letter gap (O2), multi-line stamps and signatures with per-word coverage, per-line baseline slope/jitter/curvature, typewriter texture and typed underlines, on a flush-right `memo` layout with invented pool text (`text_source="pool"`, or `"file:<path>"` at run time). Every v2 switch is off by default; v1 presets and their hashes are unchanged. Truth adds `word_gaps` (per boundary: gap, tightened, `o2_trigger`), speck/underline/overlay records with `text: false`, `covered_by`, per-line `slope_deg`. **MuPDF mirrors `text-align` under `dir="rtl"`: v1's non-justified layouts are flush-left, which is wrong for Persian** (fixed only in `memo`). `./venv/bin/python tests/synth_hard_eval.py` compares presets using the current detector and the box-audit draft splitter (O1/O2 counts). `tw_photo_medium` is the closer match to hadpi; `tw_photo_hard` is harder than the real pages.

## Real-page geometry truth (plan §6.2)

- `tests/real_truth_eval.py freeze` records each audited `books/<slug>/truth/NNNN.geometry.audit.json` in a private `truth_freeze.json` (content SHA without the audit tool's `cursor`/`revision`, image SHA, counts, excluded lines). Lines with an owner text flag are excluded automatically (hadpi p1 line 0). `run` refuses a page whose audit no longer matches its freeze, and always refuses hadpi p3–p5 (held out). bachehaye_ghali p66 is `agent_audited` and is never pooled with owner-audited pages.
- `run` (about 6 minutes, offline, forces a dead API base URL) scores line finders (`find_lines(page) -> [Region]`) and word placers (`place(page, targets) -> [{status, segments}]`) by page and by provenance/class, with Wilson intervals (optimistic: pages, not words, are the real unit). Word "correct" is **stricter than plan §1**: ≥90% of the truth word covered, under 0.5 letter taken from any other word, ≤25% blank overreach, so over-wide boxes fail; the plan-literal and covering-only rates are printed beside it. O1 cascades and O2 boundary errors (plan §13) are counted separately.
- Baseline (2026-11-11, $0, evidence already on disk only): the live engine places 0 words correctly on all five pages. Hadpi p1–p2 and majid p23/p109 show no box at all (no reader evidence for those pages, so this says nothing about the engine with evidence); p66 is all `estimated` (boxes cover the target for 52/60 spans but are several times too wide: median IoU 0.15). `_scan_page_lines` clean line recall: hadpi 25/60 with 10 merges, majid 54/54, p66 31/32. Results: `out/real_truth/`.

## Transcription experiments (plan Step 2)

- `tests/transcription_experiments.py` (`plan`, `smoke`, `run`, `score`) runs transcription arms on hadpi p1–p2 (owner text truth) and majid p23/p109 (controls) with a disk cache, a call ledger and a hard spend cap; `score` re-scores from the cache at $0. Outputs contain page text and stay under `out/transcription_experiments/`. WER is reported strict (as printed), ortho, and lenient (also ignoring ZWNJ, punctuation and pure typewriter joins).
- Result (2026-11-12, $1.82): no whole-page or banded Sonnet 5 arm reaches usable word accuracy on hadpi (best arm 11.5–21% / 31–36% lenient WER on p1 / p2; current pipeline 17–34% / 33–40%), and **run-to-run spread is as large as the arm differences, so never read a single-run difference as a finding**. Clean pages are fine (0–2%). Two readings that differ in prompt and resolution make a useful page/line review trigger (flags 0.7–2% of words on clean pages, 22–34% on hadpi), not a word-level error locator. The model almost never emits unreadable markers. Haiku is unusable as a second reader. Per-line enlarged reading with a stronger model is untested and is the open follow-up.

## Placement engine (current code)

`farsi2epub/placement.py` only re-exports `PlacementService`, `PlacementResult`, `PrintedWord`, `place`. The work is in:

- `locate.py` — `Query` (full Markdown span, alternatives, kind), embedded-text `locate_queries` (the `match` source, gated per page on a usable Unicode text layer), and the projection line/word-gap detector `_scan_page_lines` (global Otsu, row projection, `_split_tall_runs`; no deskew).
- `scan.py` — `PlacementService`: correction-independent page maps (`ensure_page_map`), reader acquisition (`llm.read_regions` on line crops, then paired word-group crops, bounded hi-res retries, contextual recovery), four background workers, per-key file claims, `Budget` reservations, receipts, offline replay.
- `page_map.py` — `PrintedWord`/`PlacementResult` and the strict `place()` (supported by independently read word groups). `DERIVATION_VERSION` 7.
- `geometry_attempt.py` — coverage-first `best_attempt`: character-proportional `estimated` boxes over detected ink when strict placement fails (`ATTEMPT_VERSION` 1).

**Statuses** (live chips describe evidence support, never measured accuracy):

| Status | Meaning | UI |
|---|---|---|
| `located` | strict placement, target words independently supported | solid box, "located" |
| `buffered` | supported, includes up to 2 adjacent words per side | solid box, "buffered +N words" |
| `estimated` | best attempt from character proportions/anchors over real ink | amber dashed box, "approximate" |
| `unresolved` | no box; reason in label/tooltip | muted chip |
| `unresolved` + `no evidence: <cause>` | Step 0: page has detected lines but **zero** read lines | no box, muted "no evidence" chip, page banner |
| `pending` | session-only, evidence work queued | "locating…" |

**Step 0 rule (2026-11-11, `scan.no_evidence_reason`):** when a scan page map has `detected_lines > 0` and `read_lines == 0`, every image-path correction on that page is `unresolved` with reason `no evidence: <acquisition_reasons>` (`reader_credit_unavailable`, `reader_access_denied`, `budget_exhausted`, `api_failure`, `cancelled`), or `no evidence: no_reading_stored` when the map recorded no reason (built offline, or every reading was rejected as illegible/incomplete — rejected readings are not persisted). `PlacementService.results()` checks this **before** the box-cache lookup, so boxes cached before Step 0 are never replayed (the cache key binds the map revision, which a policy change does not alter; no version bump was needed and other books' caches stay valid). The page stays `pending` until `refine()` re-checks it, so `--bbox-mode auto` retries acquisition. The review page shows a banner with the cause and the resume command (`farsi2epub geometry <slug> --pages N --bbox-mode auto`), on first render and through `/boxes/<page>` polling. Pages with at least one read line keep `estimated` boxes; `match` pages are unaffected. Plan decision D1 (line box or nothing) replaces estimates entirely only with the new engine.

**Evidence layout** — `books/<slug>/locate_evidence/`:

- `readings/<key>.json` — successful, legible, complete reader observations only, keyed by crop PNG SHA-256 + model + prompt/schema/render versions; `.lock` files are per-key claims (a lock with no `.json` means the read was attempted but nothing valid was stored).
- `pages/<page_key>.json` — page map: lines, words, ink `regions`, `summary` (`status`, `detected_lines`, `read_lines`, `verified_words`, `acquisition_reasons`, …), `revision`. Key binds PDF SHA, page, model, render, prompt, detector and `DERIVATION_VERSION`.
- `boxes/<key>.json` — derived results with a box; key binds page key, map revision, `ATTEMPT_VERSION`, Markdown and the full query. Unresolved results are session-only.
- `attempts/<run_id>-<page>.json` — the last exception recorded for that page in that run (overwritten, so earlier errors are lost).
- `runs/<run_id>.json`, `last_run_cost.json` — budget receipts (`spent`, `reserved`, `uncertain`). `spent` counts every successful paid call, including calls whose readings were all rejected, so spend with nothing persisted is possible (this explains hadpi's $0.10; see `docs/bbox-old-fonts-plan.md` E4).

Explicit pre-inference HTTP rejections (400/401/403/404/413/422/429) settle at $0; transport and parse failures stay `uncertain` and are never recycled. Credit/access errors stop new acquisition for the rest of the session. Existing sidecar `bbox` values are ignored; QC no longer requests coordinates.

## Historical: v5 bbox accuracy loop (`tests/bbox_score.py`, `tests/historical/`)

This section describes the **frozen v5 instrument and the removed locator it measured**. It is kept because the v5 reproduction is still tested and its lessons still hold; the v6 instrument for the current engine is `tests/bbox_metrics.py` (see `docs/bbox-consolidation.md`). Keep v5 and v6 results separate.

**The VLM transcribes, Python grades** — the model is never asked "is this box right?", only to read what is printed; the verdict (`exact`/`partial`/`shifted`/`wrong_line`/`wrong_region`/`no_box`), `shift_words` and the golden metric are computed deterministically.

**The golden metric is word-miss distance** (`_word_miss_distances`). For each word that must be inside the box: 0 if the box covers it, else the **word distance to the nearest covered word**, in reading-order index space. English illustration — sentence `my father bought some milk`, target word `milk`, box covers `my father`: the nearest covered word is `father` at distance 3, so `milk` scores 3. The objective is **zero**.

- **Headline is `perfect_rate`** — fraction of corrections scoring exactly 0 — over the **full graded denominator**, with every degenerate case (`no_box`, `no_covered_words`, `wrong_region`, `phrase_absent`) counted as not-perfect, so turning a wrong box into no box can never score as a win.
- **Magnitude** (`word_miss_mean`/`median`/`p90`/`histogram`) is computed over **measured cases only**. Degenerates are excluded rather than given a synthetic penalty (a page-length-scaled penalty would destroy cross-book comparability). `word_miss_penalized_rate` reports their share and is gated to non-increase.
- **The metric is unbounded and heavy-tailed. Never read the mean alone.** In v5, `haaji-agha` has a 3.22 mean with median 0 and p90 1: two cases scoring 105 and 50 contribute 155 of its 161 total.
- **`geom_unconfident` is a judge failure, not a locator failure**: neutral for magnitude and `word_miss_penalized_rate`, but inside `perfect_rate`'s denominator; capped at 3% by `--gate`.
- **An over-wide box costs nothing, by design.** Nothing gates against widening correctly placed boxes; `mean_iou` is the only signal — check it by hand before accepting a change justified on word-miss alone.
- Distance is reading-order index space, not screen coordinates (RTL), so it crosses line boundaries. Intended.
- `acc@1`, `acc@0` and verdict buckets are still reported and gated but are not the objective (`acc@1` read 0.7957 where `perfect_rate` read 0.7021, and rated `boof-e-koor` 0.659 while 27% of its boxes covered nothing).

Frozen-sample baseline of record: **`out/score_word_miss_baseline_v5.json`** (`_SCORE_CACHE_VERSION` 5) — an offline regrade of the exact 240 boxes serialized in v4 (`box_sha1` `6d739ee16febdf344664dea7c37d7698a103615b`). Overall: `perfect_rate` 0.8426, mean 0.9174, p90 0.0, penalized 5.96%, `geom_unconfident` 1.28%, `acc@1` 0.8255, `acc@0` 0.3362, mean IoU 0.6697. By book — `Hossein_shenasi` 1.0 (n=1), `bachehaye_ghali` 0.8722, `haaji-agha` 0.8627, `boof-e-koor` 0.7500, `review-test` 0.6667. By tier — `scan_vlm` 0.9070 (n=172), `match` 0.8222 (n=45), `scan` 0.8000 (n=5), `layout` 0.5000 (n=2). These were measured on clean, upright book scans only.

Lessons that still apply:

- **Truth is per PAGE, never per box.** The answer key is a full-page reading frozen in `books/<slug>/locate_page_read.json`; it mentions no box, so no locator change can invalidate it. Pages with a clean Unicode text layer rebuild it from PyMuPDF words (`pdf_words`). Crop-keyed truth silently dropped half the sample once; `score` now refuses more than 10% excluded (`_MAX_EXCLUDED_FRACTION`).
- **Score a frozen sample** (`tests/data/bbox_sample240.json`, no `--sample`): stratifying on `tier`, an output of the locator, lets a change alter its own population.
- **Instrument changes are not locator progress.** v1→v2 line alignment, v2→v3 geometry routing (+24 cases on `boof-e-koor`), v4→v5 target identity: all rebases, proved on identical boxes (`word_miss_smoke.py`). `compare` refuses different `truth_sha1` or `score_cache_version`; `--allow-instrument-drift` is diagnostic only and incompatible with `--gate`.
- **A superseded claim:** the `match` tier once measured 0.444 perfect. That was the geometry-routing bug, not a weak text-layer path. In this corpus no book is scored on more than one tier, so a per-tier difference is a per-book difference until an experiment separates them.
- **Paired counts are the only evidence.** "3 fixed, 0 broken" is p=0.25. `compare` reports Wilcoxon signed-rank p (hand-rolled, no scipy; `None` below 10 non-zero deltas, advisory below 25) and McNemar's exact p; keep ≥150 scored cases. `compare --gate` requires `perfect_rate` to improve with Wilcoxon p<0.05 (underpowered fails), caps `acc@1` regression at 2 points, and forbids `word_miss_penalized_rate`, `no_box_rate` increases and `judge_unusable_rate` above 3%.
- **Line alignment:** `_align_line_sequences` aligns reading lines to ink lines monotonically (Needleman–Wunsch); a mid-page dropped line mis-grades everything after it. `locate._split_tall_runs` (still in the current detector) splits fused row runs taller than `1.6 * H0` at a genuine ink valley; `_assign_word_rects` latches a forward-only drift flag. `geom_confident` is reduced over the case's own words.
- **A case is "measured" only when `degenerate is None`**; `judge_unusable` is excluded, not penalised; `no_covered_words` is a real locator failure and is penalised.
- `context_anchor_v1` was made the old default on 2026-08-03 by owner decision after **failing** `compare --gate` on ghessehaye-majid (86 cases, perfect_rate 0.4767→0.4651). It is not live today.
- `cmd_golden`/`tests/data/bbox_golden.json` snapshots previously `exact` boxes and gates on IoU; it is **not** an answer key.

Acquisition/scoring commands (historical; `pages` acquisition reads PNGs via subagents, scoring is offline):

```bash
./venv/bin/python tests/bbox_score.py pages --cases tests/data/bbox_cases.json --sample 240 --seed 7 --stratify book,tier,kind --out-dir out/pagekey
./venv/bin/python tests/bbox_score.py ingest-pages --manifest out/pagekey/manifest.json --readings out/pagekey/readings/
./venv/bin/python tests/bbox_score.py score --cases tests/data/bbox_sample240.json --out out/score_NNN.json
./venv/bin/python tests/bbox_score.py compare --baseline A.json --candidate B.json --gate
```

## Workspace layout (the on-disk contract)

Everything revolves around `books/<slug>/` (gitignored), managed by `workspace.Workspace`:

- `source.pdf` — copied input
- `book.yaml` — metadata: title/author (LLM-proposed on first transcribe), `page_count`, `source_type`, optional `page_range`, optional `chapters: [{title, start_page}]` list that overrides heading-based chapter splitting at build time
- `pages/NNNN.png` — standard-res renders (long edge 1568 px); `pages/hires/` — 2576 px renders (default transcription input)
- `text/NNNN.md` — transcribed Markdown per page; `text/NNNN.json` — sidecar with model used, confidence, quality score, flags, `needs_review`, `starts_mid_paragraph`/`continues_next`, cost, `qc`, `reviewed`; `text/NNNN.orig.md` — pre-edit backup written once on the first edited Accept
- `locate_evidence/` — placement evidence (see above)
- `review/server.log` — detached review server log
- `out/` — final EPUB

A page counts as "done" when both its `.md` and `.json` exist (`Workspace.pages_done`); `transcribe` skips done pages unless `--force`.

## Pipeline architecture

Each module is one stage; they communicate only through the workspace files above, so stages can be rerun independently:

1. **`render.py`** — PyMuPDF rasterization plus `classify_pdf` (digital/scanned/mixed by embedded-text density) and `extract_embedded_text` (the validation oracle for digital PDFs).
2. **`transcribe.py` + `llm.py`** — per-page vision transcription via `client.messages.parse` with Pydantic `output_format` (`PageTranscription`). Pages are independent and run in a thread pool; cross-page paragraph joins are deferred to build time using the `continues_next`/`starts_mid_paragraph` sidecar fields. The Persian transcription rules (ی/ک codepoints, ZWNJ, verse blocks with ` --- ` hemistich separators, footnote syntax) live in `TRANSCRIBE_SYSTEM` in `llm.py` — the build step's parser depends on this output format. **Headings contract**: every string in the structured `headings` field must also appear in `text_md` as a `#`/`##`/`###` line; `validators.evaluate` enforces this (`missing_heading` issue), `transcribe.py` auto-promotes matching bold lines and retries once with a corrective hint on failure. After transcription, `transcribe` runs page geometry (see Commands).
3. **`validators.py`** — deterministic quality scoring because LLM self-confidence is poorly calibrated. Core trick: compare char-sorted word multisets (cosine similarity) between the transcription and PyMuPDF's embedded text, immune to the extractor's broken word ordering and ligatures. A faithful digital-page transcription scores ~0.99, so 0.97 is already the escalation cutoff. Failing pages escalate to a hi-res render on the strong model unless the first pass was already Sonnet + hi-res (the default, so default runs never escalate). **Review gate** (`needs_review`): confidence < `CONF_REVIEW` (0.70), the `illegible` flag, embedded similarity < 0.90, or a review-level validator issue. Scanned pages have no oracle, so for them the gate is confidence + `illegible` only. Measured 2026-11-11: no-oracle medians are 0.75 (`bachehaye_ghali`) and 0.90 (`ghessehaye-majid`, `jameeshenasi_khodemani`); raising the cutoff to 0.80 would flag 82 of 94 `bachehaye_ghali` pages, so it stays 0.70. **The sidecar's `needs_review` is overwritten to `false` by a review Accept** (with `reviewed: edited|accepted`); recompute `validators.needs_review` from the stored values to see the transcription gate's verdict.
4. **`qc.py`** — post-transcription QC. Auto mode runs an LLM verifier (`llm.qc_verify_page`) over risk-selected pages, writing findings + a suggested correction into the sidecar `qc` key (suggest-only — never overwrites `text_md`; the human decides per correction in the review UI). Findings carry no coordinates. Pages with a pending suggestion are skipped on re-runs unless `qc --force`. Maintains `qc_history.json` at the project root (gitignored): append-only issue events with feature tags, from which per-feature lift weights bias future risk selection and review ordering. Schemas/contracts live in `TODO.md`.
5. **`review.py`** — stdlib `http.server` + jinja2 single-page web app (no external assets) showing page image beside editable Markdown, validator/QC findings, and the QC suggestion broken into per-correction **hunks** (whitespace-preserving token diff via `_derive_hunks`; context-anchored so hunks apply independently in any order — do not replace with a plain `split()` diff, it fails round-trip on real data). Each hunk gets Approve/Edit/Reject/Undo; page data is embedded as `<script type="application/json">` payloads (`_json_for_script` escapes `<`). Findings without a usable fix become **edit-only** corrections (`_derive_finding_hunks`, id offset `100000`). Each correction's full-passage `Query` goes through the placement engine above; the render path never calls the API — it shows cached results, and the browser polls `GET /boxes/<page>` while the four service workers acquire/replay evidence, swapping boxes, chips, the geometry summary and the no-evidence banner in place. Wrapped matches render ordered per-line `segments` as one SVG overlay with ripped edges. Clicking a segment or its hunk/issue smooth-zooms to `segments[0]`. A single **Accept** persists the textarea to `text/NNNN.md`, backs up pre-edit text once to `text/NNNN.orig.md`, sets `needs_review: false`, and records per-correction outcomes; `review --reset` resets decisions while keeping text edits.
6. **`epub.py`** — parses page Markdown into a block model, stitches cross-page paragraphs, splits chapters (by `book.yaml` `chapters` page numbers, else by `# ` headings), renders XHTML, and assembles an RTL EPUB with embedded Vazirmatn fonts (`assets/fonts/`). `normalize.py` runs first: conservative-only normalization (Arabic→Persian letterforms, kashida removal, punctuation spacing) — it deliberately never touches digits, ZWNJ placement, or newlines.

## Model choices (measured, don't casually revert)

- Default transcription model is Sonnet (`MODEL_STRONG` in `config.py`), not Haiku: measured Haiku first-pass failure was >90%, making the Haiku→Sonnet escalation ladder both worse and more expensive. Haiku is still used for the cheap metadata proposal and available via `--model`.
- Default input resolution is **hi-res (2576px, Sonnet 5's vision maximum)**, not the 1568px standard render: a std-res Sonnet pass was measured making character-level errors hi-res avoided (bidi-reversed dotted abbreviations like ه.ق → ق.ه, word transpositions, lost diacritics) — errors the word-bag validators are structurally blind to. `transcribe --res std` is the documented economy path (~30% cheaper/page) and failing pages escalate to Sonnet + hi-res. Sidecars record `resolution` used per page. Note that large embedded photos are still downscaled to 2576 px (plan E19). **The API then downsizes a 1821×2576 portrait render again to about 1623×2296 under its 4,784-token image cap (plan E25, measured 2026-11-12), so "hi-res" reaches the model at 0.89× on every portrait page.**
- Sonnet is called with `thinking: {"type": "disabled"}` for transcription and region reading — perception tasks, where adaptive thinking adds cost without accuracy.
- Cost is estimated in `config.estimate_cost` and tracked per page in sidecars; `--max-cost` aborts a run mid-flight.

`AGENTS.md` is a copy of this file for other coding agents; when updating CLAUDE.md, mirror substantive changes there.
