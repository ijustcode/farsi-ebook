# farsi2epub TODO

Quality-improvement effort started 2026-07-07, driven by human review of book2 (digital) and jamee (scanned) output. Phase 1 is being implemented now; Phase 2 items carry full context so they can be built cold in a later session.

## Phase 1 — headings fix + QC workflow

Root cause of missed headings: the vision model correctly lists headings in the sidecar `headings` field (verified: اطلاعات تکمیلی on book2 p3, فهرست مطالب on jamee p4) but omits them from `text_md` or demotes them to `**bold**`. jamee has NO embedded text layer (0/161 pages — "selectable text" in Preview is macOS Live Text OCR), so the fix must not depend on PDF text extraction; book2's text layer is uniform 14pt (font size useless) but headings are whole-line bold spans.

- [x] 1.1a `llm.py` TRANSCRIBE_SYSTEM: 3-level heading ladder (`#` chapter/page titles, `##` section banners incl. boxed strips, `###` minor bold labels); invariant that every `headings` entry appears as a `#{1,3}` line in `text_md`; sharpened running-header vs real-heading rule
- [x] 1.1b `validators.py`: `headings` param on `evaluate()`; `missing_heading` issue (declared heading absent from text_md heading lines, word_bag-compared); `embedded_heading_missing` issue from `render.extract_heading_candidates()` (digital only: whole-line bold or >1.15× body-size spans, <60 chars); both issues trigger review
- [x] 1.1c `transcribe.py`: safe auto-promotion (body line exactly matching a declared heading → promoted to heading in place); retry-with-hint on remaining heading issues (one retry, same model, `extra_hint` param on `llm.transcribe_page`, keep better quality_score); record validator issues as history events
- [x] 1.1d `epub.py`: `### ` → h3 (parser before `##`/`#` checks, renderer, CSS); chapter splitting stays h1-only
- [x] 1.2a `qc.py` (new): auto QC orchestration per contracts below; risk-based page selection (+~5% random sample); cost estimate + confirmation; suggest-don't-apply
- [x] 1.2b `llm.py`: `qc_verify_page()` verifier call returning QCReport (Sonnet, thinking disabled)
- [x] 1.2c `review.py`: findings pills (validator issues + QC issues); suggestion panel with diff + "Apply suggestion" button; budget bypass; `/save`//`/accept` record history events (human_edit, suggestion_accepted/edited/rejected)
- [x] 1.2d `cli.py`: `qc` command (`--mode auto|manual`, `--all`, `--yes`); `--qc` flag on transcribe + post-run TTY prompt
- [x] 1.3 housekeeping: `.gitignore` += `qc_history.json`; CLAUDE.md documents qc command + headings contract
- [x] verify: heading validator flags existing book2 p3 + jamee p4 (confirmed); re-transcribed samples come out correct (book2 p3: `###/##` headings incl. اطلاعات تکمیلی; jamee p4: `# فهرست مطالب`); auto QC run on book2 flagged real missing headings on old pages 1/2/5/8 ($0.076, suggestions pending); suggestion accept round-trip verified incl. history events; EPUB build with h3 passes epubcheck clean

Note: pages of book2/jamee transcribed BEFORE this fix still carry old-prompt output. Book2 pages 1, 2, 5, 8 have pending QC suggestions (`farsi2epub qc book2 --mode manual` to review). A full re-transcribe (or QC sweep with `--all`) of both books is the user's call, cost-wise.

### Post-phase-1 fix: hi-res input is the default (2026-07-07)

- [x] User found a regression: re-transcribing book2 p3 at std-res reversed the dotted abbreviation ه.ق → ق.ه (plus word transpositions, lost diacritics). Root cause: the old escalation path had produced a hi-res transcription; the Sonnet-default rerun used the 1568px render and the hi-res escalation branch was dead code (only fired when starting model ≠ Sonnet). The word-bag validators are structurally blind to this class (`word_bag("(ه.ق)")` is empty — dots stripped, single-char words dropped).
- [x] Fix: default = Sonnet + hi-res (2576px = Sonnet 5's vision max, ~4784 image tokens vs ~1568 at std; measured ~6.8k vs ~3.2k input tokens/page; ~$0.026 vs ~$0.018 per page). Economy path = `transcribe --res std`, documented in CLI help; escalation now fires whenever the first pass wasn't Sonnet+hi-res, so economy mode retries failing pages at full fidelity. `analyze` prints both estimates. Sidecars record `resolution`.
- Follow-up candidates (fold into Phase 2 validator work): order-sensitive check for short parenthesized dotted abbreviations against the embedded text layer (which has the correct ه.ق); QC verifier prompt line for abbreviation letter order.

## Phase 2 — planned, NOT yet implemented

### 2.1 Footnote off-by-one (refs numbered 1 instead of 2; real ref 1 missing)

**Root cause (traced 2026-07-07, confirm on a real page before fixing):** `epub.py` `_ChapterRenderer._resolve_footnotes` assigns numbers by *encounter order* of in-text `[^n]` markers (fresh per-chapter counter) while definitions are looked up by original label per page (`_defs_by_page`). If the model misses the first in-text superscript marker, `[^2]` renders as note 1 (content still correctly paired to def 2) and def 1 becomes an orphan that `_render_endnotes` silently drops — exactly "everything off by one and real ref 1 missing".

Steps:
1. Diagnose: scan `books/*/text/*.md` for pages with `[^n]:` definitions lacking a matching in-text `[^n]` marker.
2. `validators.py`: per-page refs-vs-defs label-set check → issue `footnote_mismatch` → review; history event with `char_signals: ["footnote"]`.
3. `epub.py` `_ChapterRenderer`: never drop orphaned definitions — append them to the chapter endnotes in sequence and print a build warning.
4. `llm.py` TRANSCRIBE_SYSTEM: explicit emphasis + example for superscript footnote markers (¹/۱ etc. → `[^1]`).
5. QC verifier prompt: `llm.QC_SYSTEM` already instructs footnote-marker checking (issue type `footnote_marker`); confirm it catches a real case.

### 2.2 ه mangling in two-part (ZWNJ) words

**Why current validators are blind:** `validators.word_bag` treats ZWNJ as a word separator (`_JOINERS`) and char-sorts words, so dropped/misplaced ZWNJ or ه/ۀ/ه‌ی substitutions score identically against the embedded-text oracle. The embedded layer itself is unreliable here: PyMuPDF emits U+200A hair-space where ZWNJ should be (observed on book2 p3).

Steps:
1. `validators.py`: extract tokens with ه at a join boundary (ه+ZWNJ, ه directly joined to common suffixes ها/ای/اش/ام/اند); record `heh_boundary` char signal; anomalous joins → issue `heh_boundary_suspect`.
2. QC verifier: `llm.QC_SYSTEM` already instructs ه-boundary character verification (issue type `heh_boundary`), and `qc.char_signals_from_diff` already tags `heh_boundary` from human edits — the history will show where this matters most.
3. `normalize.py`: small whitelisted set of unambiguous repairs only (e.g. `ه ها` → `ه‌ها` plural pattern); keep the module's deliberate no-ZWNJ-repair stance for anything ambiguous (see its docstring).

## Shared contracts (all sessions/agents must follow these)

### Page sidecar `qc` key (written by auto QC into `books/<slug>/text/NNNN.json`)

```json
"qc": {
  "date": "2026-07-07",
  "verifier_model": "claude-sonnet-5",
  "verdict": "fail",
  "issues": [{"type": "missing_heading", "description": "…", "snippet": "…"}],
  "suggested_text_md": "full proposed corrected page markdown, null when verdict is pass",
  "suggestion_status": "pending",
  "cost_usd": 0.0123
}
```

`verdict`: `"pass" | "fail"`. `suggestion_status`: `"pending" | "accepted" | "edited" | "rejected"` (review UI updates it; null on pass).

### `qc_history.json` (project root, gitignored, append-only)

```json
{"events": [{
  "date": "2026-07-07T12:00:00",
  "book": "book2",
  "page": 3,
  "source_type": "digital",
  "model_used": "claude-sonnet-5",
  "detected_by": "validator",
  "issue_type": "missing_heading",
  "features": {
    "flags": ["table"],
    "confidence_bucket": "0.9-1.0",
    "char_signals": ["heading"]
  }
}]}
```

`detected_by`: `"validator" | "auto_qc" | "human_edit" | "suggestion_accepted" | "suggestion_edited" | "suggestion_rejected"`. Auto QC also records `issue_type: "pass"` events for clean pages — these are the denominators for the lift weighting.
`confidence_bucket`: one of `"<0.7" | "0.7-0.8" | "0.8-0.9" | "0.9-1.0"`.
`char_signals` vocabulary: `heading`, `zwnj`, `heh_boundary`, `digits`, `punctuation`, `footnote`, `verse`, `table`.

### `farsi2epub/qc.py` public API

```python
def run_qc(ws: Workspace, mode: str, all_pages: bool = False, assume_yes: bool = False) -> None
    # mode "auto": risk-select pages (or all), confirm cost, run llm.qc_verify_page per page,
    #   write sidecar "qc" key, record events, then launch review UI on flagged pages.
    # mode "manual": launch review UI with findings (review.run_review(ws, budget_all=all_pages)).
def record_event(book: str, page: int, detected_by: str, issue_type: str, *,
                 source_type: str | None = None, model_used: str | None = None,
                 flags: list[str] | None = None, confidence: float | None = None,
                 char_signals: list[str] | None = None) -> None
def char_signals_from_diff(old_text: str, new_text: str) -> list[str]
def compute_feature_weights(events: list[dict]) -> dict[str, float]   # per-feature lift, clamped [0.5, 3.0]
def risk_score(sidecar: dict, source_type: str, weights: dict[str, float]) -> float
    # (1 - quality_score) * product of matching feature lifts
```

### `farsi2epub/llm.py` additions (implemented)

```python
class QCIssue(BaseModel): type: str; description: str; snippet: str
class QCReport(BaseModel): verdict: str; issues: list[QCIssue]; suggested_text_md: Optional[str]
def qc_verify_page(client, png_bytes, text_md, model, page_no) -> tuple[QCReport, dict, float]
def transcribe_page(..., extra_hint: Optional[str] = None)   # appended to the user text
```

## Phase 3 — bbox locator accuracy (loop rebuilt 2026-07-27)

The measurement loop was rebuilt so that iterations are free and comparable:
truth is now a frozen PER-PAGE reading (`books/<slug>/locate_page_read.json`),
never a crop around the box under test. See the "bbox accuracy loop" section of
CLAUDE.md for the commands and the guardrails. Baseline to beat: **acc@1 0.3362**
(`out/score_000_baseline.json`, crop-era; re-score against page truth before
quoting it as the baseline for a new iteration).

### 3.1 Persian punctuation is not stripped by the production word fold  ✅ FIXED, but NOT an accuracy win

**Found 2026-07-27 while calibrating the answer key; NOT yet fixed, NOT yet
measured.** `locate._NONWORD_RE` is `[^؀-ۿ0-9a-zA-Z]`, which preserves the whole
Arabic block — and that block contains Persian punctuation. So:

```python
locate._fold_word("زد،")  == "زد،"   # comma survives
locate._fold_word("زد")   == "زد"    # -> the two words DO NOT MATCH
locate._fold_word("متن.") == "متن"   # ASCII punctuation IS stripped
```

Affected codepoints: ، U+060C, ؛ U+061B, ؟ U+061F, ٪ U+066A, ٫ U+066B, ۔ U+06D4,
﴾ ﴿. Consequence: every window comparison in `locate` — Tier A's word-multiset
match, `_best_window`/`_window_candidates` scoring, `_align_strip` — treats a
comma-terminated word as a different token from the bare word. Persian prose is
comma-dense, so a query whose first or last word abuts a comma is systematically
penalized. Measured side-effect on the acquisition gate: ~7 points of apparent
recall loss, which is how it was noticed (`tests/bbox_score.py::_recall_words`
strips them for measurement only, deliberately leaving production untouched).

Steps:
1. Re-score the current locator against page truth → this is iteration 3's baseline.
2. Strip Arabic-block punctuation in `locate._fold_word` (keep digits: Arabic-Indic
   digits share the block and ARE word content).
3. Re-score; `compare --gate` against the baseline from step 1, and check McNemar's
   p — with ~150 scored cases a real effect should clear it.
4. `tests/locator_regression.py`: add a case pinning `_fold_word("زد،") == "زد"`.

Risk to check in step 3: `normalize.py` and the `review` hunk diff also consume
folded words; confirm no caller depends on punctuation surviving the fold.

### 3.2 Tier B/C never verify what is printed where they point  ← Tier C MEASURED, Tier B is the open work

Tier A *searches for* the query text and scores acc@1 0.73; Tiers B (layout) and
C (scan) *interpolate* a position from character offsets and never check what is
actually printed there — 0.21–0.28. Block anchoring (iteration 2) removed the
accumulated drift it was aimed at, but drift was never the dominant defect: the
remaining `wrong_region` cases need >1.6 line-heights of error to explain. The
paradigm gap, not a tuning gap: give B/C a content check against the page reading
they already have access to.

**MEASURED 2026-07-28 — the content-check paradigm is confirmed for Tier C.**
Tier C already HAS a content check in production: `refine_scan_boxes` crops the
guessed line(s), has a VLM transcribe them (`llm.read_strips`) and re-aligns the
query onto the real word rects. Every score before this point ran `--no-refine`,
so scan was being measured with its content check switched OFF.

On the frozen sample, 235 graded, same answer key:

| | `--no-refine` | `--refine` |
|---|---|---|
| overall acc@1 | 0.5447 | **0.6723** |
| scan -> scan_vlm | 0.5748 (130) | **0.8430** (121) |
| plain scan left over | — | 0.1667 (9) |
| bachehaye_ghali | 0.5489 | **0.7744** |
| match / layout | unchanged | unchanged |

**32 fixed, 2 broken, McNemar exact p = 0.0000** on 34 discordant pairs.
Reports: `out/score_007_frozen_norefine.json` vs `out/score_008_frozen_refine.json`.
Refinement cost $2.03 cold (cache 492 -> 649 entries); re-scores are free.

Read it as: an unchecked geometric guess is nearly worthless (the 9 cases where
refinement failed score 0.1667), and checking what is printed does almost all
the work. That is the justification for doing the same to Tier B.

**STAGE 1 DONE 2026-07-28 — precedence flip, the cheap fix, and it was large.**
Tier B was accepted in `locate_queries` unconditionally, and since it has
`no_box_rate` 0.0 it always returned something — so on a glyph-cipher page Tier C
was structurally NEVER reached, and neither was refinement. Tier B is now taken
only when `a_usable` (the layer decodes); cipher pages fall through to Tier C,
with Tier B retained as the fallback when scan yields nothing, so `no_box_rate`
is unchanged at 0.0426.

| | before | after |
|---|---|---|
| overall acc@1 (refined) | 0.6723 | **0.7787** |
| haaji-agha | 0.4118 | **0.9020** |
| overall acc@1 (unrefined) | 0.5447 | 0.5745 |
| scan_vlm | 0.8430 (121) | 0.8706 (170) |

**25 fixed, 0 broken, McNemar exact p = 0.0000.** Refinement of the newly
reachable haaji-agha strips cost $0.45 once; re-scores are free.
Reports: `out/score_010_precedence_refine.json` vs `out/score_008_frozen_refine.json`.

The line-local-offset proposal that prompted this was assessed and REJECTED as
incoherent: `llm.py:101` forbids hard-wrapping ("each paragraph is a single line
of output"), so markdown lines have no correspondence to printed lines in prose,
and line identity is an OUTPUT of the cumulative count, not an input. Measured
on the layout tier: 41/47 cases already have `line_delta == 0`, and the alarming
`mean_abs_shift=14.06` is ONE case (`haaji-agha:p53:h1`, 507 words / 32 lines);
excluding it the mean is 3.35, median 2. Line selection was not the defect.

**Remaining Tier B work.** `refine_scan_boxes` refuses any box whose
`source` is not `"scan"` (see its `todo` filter), so layout boxes never get a
content check at all — and layout is the cipher books, where the PDF reports
REAL line rectangles and garbage characters, which is exactly the situation the
strip oracle was built for. layout sits at 0.4231 over 54 cases; haaji-agha,
almost all layout, is the worst book at 0.4118.

Steps:
1. Extend the refiner to accept `source == "layout"`: its strip geometry comes
   from `_scan_page_lines`, so a layout box must first be mapped to the detected
   ink line(s) it overlaps (the cipher layer's own line rects are real and can
   seed this) before the existing `_align_strip` path runs unchanged.
2. Keep the tier label distinct (`layout_vlm`) so the report can price it.
3. Re-score the frozen sample and gate; expect a large effect if the analogy to
   Tier C holds, and treat anything under McNemar p<0.05 as unproven.
4. Watch the failure mode Tier C shows: when refinement fails the box is WORSE
   than useless (0.1667). Decide explicitly whether a failed layout refinement
   should keep the layout box or return no box at all.

### 3.2b Sampling must not depend on the locator (fixed 2026-07-28)

`--sample N --stratify book,tier,kind` strata on `tier`, which is an OUTPUT of
the locator. Enabling Tier C refinement moved boxes from `scan` to `scan_vlm`,
which changed the sampled population: 14 cases uncached, per-book counts moved,
and the run reported 0.6923 against the frozen sample's 0.6723. Fixed by
freezing the 240 case ids into `tests/data/bbox_sample240.json` and scoring that
with no `--sample` flag. Case identity is the hunk, so the list is stable
regardless of what the locator does.
