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
