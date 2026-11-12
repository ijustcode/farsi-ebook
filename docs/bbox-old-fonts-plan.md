# Box placement across all Persian print — reviewed plan (v2)

- Status: **PLAN v2, FINAL — owner decisions recorded 2026-11-11 (§10).** Approved to start at
  Step 0. Nothing here is implemented yet. No accuracy claim is made.
- Date: 2026-11-10, finalized 2026-11-11. Repo HEAD: `bc37742`.
- v1 author: Claude Opus 5.5. v2 reviewer and editor: Claude Fable 5.2, acting as critic and
  planner only. Implementation goes to a different model (Opus or Sonnet), per project practice.
- Objective (owner): boxes that are placed with **high accuracy across all varieties of Persian
  text and fonts**. Classify-then-route and a single unified method are both acceptable.
- The plan also covers **transcription quality**, because any text-anchored placement inherits
  the transcription's errors.

House rules this plan respects:
- Python only in `venv/`; new dependencies are an owner decision.
- The model reads; Python grades. No model judges its own box.
- Truth is per page, frozen, and independent of any box.
- Availability is never reported as accuracy.
- Trunk-based commits to `main`.
- Every stage has a smoke test with pass and kill criteria, run before building.

---

## 0. Review verdict on v1

**The diagnosis is right. The remedy was aimed at the second-hardest problem and skipped the
hardest one.**

What v1 got right:
- The two root causes (no reading evidence; line detector fails on photographed pages) and the
  silent degradation that hid them. All re-checked, see §2.
- The reframing: this is *alignment of known text to an image*, not OCR.
- Removing paid per-crop reads and proportional guessing.

What v1 got wrong or missed, in order of severity:

1. **The transcription dependency was left as an open question. It is the main risk.** The hadpi
   transcription is poor (§2 E20–E23): the model flagged the page `illegible` with confidence 0.55.
   (Correction after Step 0: the review gate *did* flag it; see E20.) The text contains fluent, plausible legal phrasing that is not
   on the page, and nonsense strings survive even after human review. An aligner fed this text
   cannot be accurate. v2 adds a transcription track (§5) and makes the aligner tolerate local
   text errors by design (§4.4).
2. **Line breaks were hand-waved, and they are the hard part.** Forced alignment needs to know
   which text is on which printed line. The Markdown has paragraphs, not printed lines. v1's step 3
   ("recognizer text + trigram similarity") quietly reintroduces an OCR dependency. v2 gets line
   structure from a single whole-page *line-faithful reading* (§4.2), which doubles as an
   independent second transcription for error detection.
3. **Global deskew cannot fix these pages.** I measured the local line slope on hadpi: on page 2
   it goes from about 0.1° at the top to about 3.3° at the bottom (E18). That is a warped page,
   not a rotated one. No single angle straightens it. v1's arm (b) would fail for this reason.
   Lines must be followed locally (per-line baselines), not globally.
4. **v1 jumped to a neural recognizer when a simpler, font-independent method was available.**
   Once the text of a line is known, finding word boxes is a small optimization problem: choose
   which ink gaps are word boundaries so that segment widths match the words' expected widths.
   This needs only numpy, no model and no font knowledge beyond a per-book width table it learns
   itself. On a typewriter face, where every glyph has the same width, it should be close to
   exact. v2 makes this the lead arm (§4.3, arm G) and keeps CTC forced alignment as the second
   arm (arm C), needed mainly for scripts where gaps do not exist (nastaliq).
5. **CTC details v1 missed.**
   - CTC outputs are spikes, not extents: a character fires in one frame somewhere inside its
     glyph. Word boxes taken from spike spans are clipped. Fix: use spikes only to decide *which*
     ink gap separates two words, then snap box edges to ink.
   - kraken trains on text in *display* order after bidi reordering `[UNVERIFIED — check]`. Lines
     with digits or Latin runs must be reordered the same way before alignment, or the path
     desynchronizes.
   - kraken appears to ship a forced-alignment function already (`kraken.align.forced_align`)
     `[UNVERIFIED — check before writing any Viterbi code]`.
6. **The truth set was too small and all from one document.** Seven pages cannot support "all
   varieties of Persian fonts". v2 adds a **synthetic benchmark with exact truth** (§6.1): render
   known Persian text in many fonts, extract exact word boxes from the generated PDF, then degrade
   the image (warp, blur, stamps, uneven light). Thousands of graded words per font class, no
   annotation.
7. **The pass bar was too low and ignored calibration.** 80% of words on hard pages is not "high
   accuracy". What the reviewer needs is: when the UI says "located", it is right. v2 gates on
   the precision of confident boxes and on honest fallback to a line-level box (§7).
8. **Smaller points.**
   - Tesseract has no Persian data on this machine (E24), so v1's arm R4 needs an install.
     Demoted to optional.
   - v1's S0-b ($1 re-run of the engine slated for removal) is not needed to establish the
     diagnosis. Demoted to optional.
   - The transcription renders the page at 1821×2576 although the embedded photo is 2320×3428
     (E19). We discard about 22% of linear resolution before the model sees a degraded page.

---

## 1. What "works" means

A page yields a **word map**: every printed word has a rectangle (or per-line segments), and a
status.

| Status | Meaning | UI |
|---|---|---|
| `located` | Word-level box, confidence above the calibrated threshold | solid box |
| `line` | Correct line known, word position not trusted | full-line box, labeled "line only" |
| `unresolved` | No trustworthy evidence | no box, reason shown |

Targets (measured as in §6; "correct" = at least 90% of the truth word's area inside the box and
at most one adjacent word of excess):

| Measure | Clean print | Hard pages (photographed, typewriter, degraded) |
|---|---|---|
| Words `located` and correct, of all words | ≥ 95% | ≥ 85% |
| Precision of `located` boxes | ≥ 98% | ≥ 97% |
| Words on the wrong line (any status that shows a box) | ≤ 0.5% | ≤ 1.5% |
| Proportional guesses shown | 0 | 0 |

A correction's box is the union of its words' boxes, so word-map accuracy bounds correction
accuracy.

---

## 2. Evidence ledger

Run from the repo root. E1–E16 are from v1 and were not re-derived, except where noted. E17–E24
are new in this review.

| # | Claim | How to check | Status |
|---|---|---|---|
| E1 | hadpi has zero stored readings: `locate_evidence/readings/` holds only `.lock` files. | `ls books/hadpi-articles/locate_evidence/readings \| grep -c json` → 0 | verified (v1) |
| E2 | Acquisition failed on every page (`BadRequestError` p1, p4; `RuntimeError` p2, p3, p5). | `cat books/hadpi-articles/locate_evidence/attempts/*.json` | verified (v1) |
| E3 | Page maps: `reader_credit_unavailable` ×4, `budget_exhausted` ×1; every line `missing_reading`. | print `summary` of `locate_evidence/pages/*.json` | verified (v1) |
| E4 | Receipt shows $0.10236 spent with nothing persisted. | `locate_evidence/last_run_cost.json` | number verified; **cause unknown** |
| E5 | All 35 cached boxes are `estimated`; none `located` or `buffered`. | tally `locate_evidence/boxes/*.json` | verified (v1) |
| E6 | Estimates come from `geometry_attempt.best_attempt` → `PageProjection.span`: character-proportional over ink regions, with no anchors when there are no readings. | read `farsi2epub/geometry_attempt.py` | verified (v1) |
| E7 | The UI shows these as the same amber "approximate" class as partly anchored estimates. | `farsi2epub/review.py` ~291–299, ~1212 | verified (v1) |
| E8 | `locate._scan_page_lines`: global Otsu, global row projection, no deskew, no non-text filtering; word boxes from gaps ≥ `0.08 × line_h`. | read `farsi2epub/locate.py` 184–240 | re-verified |
| E9 | hadpi p2: 22 detected "lines", with heights up to 193 PDF units against ~12 for a text line. One block holds the last ~8 lines plus signatures and stamp. | §8 snippet | verified (overlay inspected) |
| E10 | hadpi p2 has about 32 printed text lines. | count on `pages/hires/0002.png` | approximate; fixed by truth in step 1 |
| E11 | Other pages: p1 19 lines (max height 136.7), p3 10 (188.0), p4 28 (111.3), p5 28 (145.3). | §8 snippet | verified (v1) |
| E12 | Sep 2026: $3.44 acquisition for 30 pages vs $0.72 transcription + $0.85 QC; 0 complete page maps. | `docs/page-geometry-coverage-*.md` | repo docs |
| E13 | "109/111" = 12 located + 11 buffered + 86 approximate. Availability, not accuracy. | `docs/geometry-best-attempt.md` | repo docs |
| E14 | Zero audited held-out cases; the promotion gate never ran. | `docs/bbox-consolidation.md` | repo docs |
| E15 | CLAUDE.md/AGENTS.md describe removed tiers and flags. | compare with `docs/bbox-consolidation.md` | verified (v1) |
| E16 | kraken `blla` was tried once on 6 clean pages (32/32 vs 31/32 lines on scan66). Never on warped pages; recognition never tried. | `docs/bbox-consolidation.md` | repo docs |
| **E17** | The top 4 lines of hadpi p2 are detected correctly; merging starts where the slope starts. | same overlay as E9 | verified |
| **E18** | **Line slope varies within a page.** Cross-correlating row-ink profiles of a left and a right column strip, in three vertical bands: p1 ≈ 2.2° / 2.6° / 3.3°; p2 ≈ 0.1° / 2.2° / 3.3°; p4 ≈ 1.1° / 1.7° / 2.3° (top / middle / bottom). | crude measurement, ±0.3°; sign not determined | verified as a rough measurement |
| **E19** | Embedded photos are 2276–2392 × 3200–3428 px. The hi-res render sent to the transcriber is 1821×2576. | `page.get_images()`; `pages/hires/0002.png` size | verified |
| **E20** | hadpi sidecars: model confidence 0.45–0.68, `flags: ["illegible"]` on p2, and `needs_review: false`, `escalated: false` on all five. **Corrected after Step 0:** the gate worked. `validators.needs_review` flags confidence < 0.70 or `illegible`, and all five pages were flagged and surfaced. The stored `false` is written by the review Accept (`review.py` ~2397); `escalated: false` is by design, because the first pass was already Sonnet at hi-res. v2's original reading of this as a gate failure was wrong. | `tests/review_gate_regression.py` | verified, corrected |
| **E21** | Human review changed 49 of 417 words on p1, 42 of 396 on p2, and 3–10 on p3–p5 (word-level diff of `.orig.md` vs `.md`). | `difflib` over the two files | verified; note the review may be incomplete |
| **E22** | The reviewed p2 text still contains strings that are not Persian legal prose and do not match the image ("و در رقاب او", "جزء شریانند", "در رقتی"), and fluent phrases not visibly on the page ("لازم‌الاجرا و معتبر است"). | read `text/0002.md` beside `pages/hires/0002.png` | verified by reading; exact error rate needs step 1 truth |
| **E23** | The Markdown is normalized, not diplomatic: it inserts ZWNJ ("می‌گردد" where the print has no break) and splits words the print fuses ("به وسیله" for printed "بوسیله"). | same comparison | verified |
| **E24** | Tesseract 5.5.2 is installed, with only `osd` data: no `fas`. kraken is not installed anywhere on PATH. PyMuPDF 1.28 has `Page.insert_htmlbox`. | `tesseract --list-langs`; `which kraken` | verified |

**Diagnosis (unchanged from v1, now with a fourth cause):**
- RC1 — no reading evidence on hadpi (E1–E6).
- RC2 — line detection fails on warped photographed pages (E8–E11, E17–E18).
- RC3 — the UI hid both (E7).
- **RC4 — the transcription itself is unreliable on these pages** (E21–E23). The pipeline did
  flag the pages for review (E20, corrected); what it lacks is any way to say *which words* are
  wrong, or to do better than one low-confidence read.

---

## 3. Lessons from past attempts

The full history is in v1's table (git history of this file) and `docs/bbox-*.md`. The pattern:

- Every attempt let the *image* discover word boundaries (ink gaps, crop reads), then tried to
  reconcile them with the text. Each fix added a reconciliation layer.
- The best measured result (scan_vlm, `perfect_rate` 0.8426) was on clean, upright book scans only.
- The corpus has no audited photographed, typewritten, lithographed or nastaliq page.
- Persian ink gaps do not mark word boundaries: the non-joining letters (ا د ذ ر ز ژ و) leave
  gaps inside words. A fixed gap threshold cannot work. Typewriter spacing makes it worse.

---

## 4. Architecture: one alignment core, a condition-driven front end

A single core handles all scanned classes. Differences between page types are handled by a front
end that measures the page and picks pre-processing, and by fallbacks that trigger on measured
confidence. No up-front font classifier is needed. Digital PDFs keep their own path.

```
PDF page
 ├─ usable Unicode text layer ──────────► existing "match" path (unchanged)
 ├─ glyph-cipher text layer ────────────► PDF glyph rects feed 4.3 directly (no pixels needed)
 └─ image only
      4.1 front end: light normalisation, non-text masking, local baselines → rectified lines
      4.2 line text: line-faithful page reading, checked against detected lines
      4.3 words: transcript-constrained gap selection (arm G); CTC alignment (arm C) as second opinion
      4.4 confidence, status, persistence
```

### 4.1 Front end (pixels → rectified text lines)

- **Measure first**, and store the measurements in the page map: stroke width, median component
  height, local slope field (as in E18), background gradient, fraction of ink in oversized
  components. These descriptors select the pre-processing, and they are the "classification".
- **Binarize** with an adaptive (Sauvola-type) threshold. Global Otsu fails under uneven light.
- **Mask non-text ink**: components far larger than a glyph (stamps, signatures), long thin
  horizontal strokes (underlines), isolated margin marks. Masked ink is excluded from line
  finding but kept for display.
  - Masking is a hard yes/no decision taken before any text is matched, so it must be
    **conservative**: mask only what is safe by size and isolation alone. A mark at line height,
    near the text block and about the size of a dot or a narrow letter is *not* masked here
    (the same rule would delete punctuation and letters such as «ا»). Those marks are left for
    arm G to explain or skip (§4.3, §13 O1).
- **Estimate the text block's edges** (added, §13): fit the right and left edges of the block from
  all lines together, robustly, allowing indented and short lines. Each line's expected start and
  end come from this fit, not from its first and last ink.
- **Find lines by following them locally**, not by one global projection. Candidate arms, to be
  measured in step 3:
  - (L1) strip-wise projection: cut the page into vertical strips, find row runs per strip, link
    runs across strips into baselines. Classical, numpy only.
  - (L2) kraken `blla` neural baselines.
  - (L3) Surya line detection.
- **Rectify** each line along its baseline into a straight strip, and keep the mapping back to
  page coordinates.

### 4.2 Line text (which text is on which printed line)

- **One whole-page, line-faithful reading per page**: the model returns the printed lines in
  order, one string per physical line, written exactly as printed (no normalization, no ZWNJ
  repair, no paragraph joining), with an explicit marker for anything it cannot read.
  - Cost: about one transcription call per page (~$0.03), against ~$0.11/page for crop reads (E12).
  - For new books, test folding it into the transcription call (step 2). For existing books it is
    a backfill call, cached forever by image hash + model + prompt version.
  - It is also an **independent second transcription**. Its diff against `text_md` is the
    transcription error detector that scanned pages currently lack (§5 T2).
- **Check it against the pixels.** Align the sequence of read lines to the sequence of detected
  lines by length: characters per read line against ink width per detected line. Short lines
  (headings, paragraph ends) are natural anchors. The alignment allows a detected line with no
  text (stamp, underline), and 2-to-1 cases (detector merged two lines). This is the same idea as
  length-based sentence alignment in machine translation.
  - A count or length mismatch that cannot be explained locally marks the affected lines `line`
    or `unresolved`. It also flags a possible skipped or repeated line in the transcription.
- **Map `text_md` to lines**: align the Markdown character stream (folded: ZWNJ, diacritics,
  punctuation and spaces removed, Arabic/Persian letterforms unified) to the line-reading stream.
  This is text-to-text alignment and is cheap and robust. Each Markdown word gets a line and a
  character range within that line. Where the two readings disagree, the words are marked
  `text-disputed`.

### 4.3 Words (where each word is inside its line)

**Arm G — transcript-constrained gap selection (lead; numpy only).**
- Inputs: a rectified line strip, its ink runs and gaps, and the line's character sequence with
  word boundaries.
- Model: each character has an expected width. Start from a uniform width, then learn a per-book
  per-character width table by iterating: align → re-estimate widths → align.
  - **Typewriter faces (corrected, §13):** do not assume "all equal". Arabic-script typewriters
    are described as using two or three width classes on a common pitch. Constrain the table to
    a small number of width classes that are multiples of one pitch, and measure the classes on
    hadpi p1–p2 truth before relying on them.
- Optimization: dynamic programming over the line. Choose a cut for each word boundary so that
  segment widths match expected widths, cuts fall in gaps, and wider gaps are preferred.
  - A word boundary may have no gap (fused in print, E23). The cut is then placed by expected
    width and the pair is marked as fused.
  - A gap inside a word is free after a non-joining letter and penalized elsewhere.
  - **Ink may belong to no word (added, §13 O1).** Any ink run may be skipped as "not text" at a
    cost that falls with its size and density and rises inside the fitted text block (§4.1).
    Without this, a surviving speck must be given to a word and can shift the rest of the line.
  - On typewritten and other tight-set pages **width leads and gaps only refine**: a cut is never
    moved into a gap that makes the neighbouring segments' widths implausible (§13 O2).
- Output: box edges snapped to ink. Confidence from the margin between the best and second-best
  segmentation, and from the width residual.

**Arm C — CTC forced alignment (second opinion; needs kraken/torch).**
- A Persian line recognizer scores the strip. The known text is forced through its outputs. The
  spikes decide which gap separates each pair of words; box edges are again snapped to ink.
- Needed where gaps are unreliable or absent: nastaliq, heavy kerning, touching words, very
  degraded ink.
- Per-book fine-tuning is available for free, because aligned lines plus their text are training
  data. Treat this as an escalation, not a default.

**Use of both:** where G and C agree on a boundary, confidence is high. Where they disagree, the
word falls back to `line`. Step 4 decides whether C is needed at all for the corpus we have.

**Arm R — render and warp (optional candidate, added §13).** Render the known line in a similar
face with the synthetic generator, align its column ink profile to the strip by one-dimensional
dynamic time warping, and read word boundaries off the warp. numpy only, no model. Published for
manuscripts (Hassner, Wolf, Dershowitz 2013); unproven on Persian. Tried in step 4 beside G; it
is dropped without replacement if it does not beat G on any class.

**Arm V — direct model grounding (control only).** Ask the model for the box on the whole page;
grade deterministically. Kept only to measure whether it beats G/C. It never grades itself.

### 4.4 Robustness to transcription errors (a design requirement, not a hope)

- A wrong, missing or extra word must damage only its neighbours, not the rest of the line or
  page. The gap DP and the text-to-text alignment both allow insertions and deletions at a cost.
- Lines are aligned independently once line assignment is done, so an error cannot propagate
  across lines.
- `text-disputed` words (the two readings disagree) are never `located`.
- This is tested directly: step 4 injects synthetic text errors and requires the untouched words
  to stay correct.

### 4.5 What stays, what goes

- Stays: the `match` path, `PlacementResult`/`PrintedWord` contracts, `/boxes/<page>` polling,
  cache and lock conventions, the v6 metric, the annotation workflow.
- Goes (after promotion, not before): crop-reader acquisition and its budget ledger,
  `geometry_attempt.py` proportional estimates, `page_map.supported_groups/windows`.
- Coverage: the `line` status replaces the amber guess. It keeps a box on almost every correction
  (the reviewer can still zoom to the right line) without pretending to word precision. This
  replaces the September coverage-first policy (owner decision D1, §10).

---

## 5. Transcription quality track

Placement depends on it, and hadpi shows the current pipeline is blind on hard scanned pages:
with no embedded text there is no oracle, so `validators.py` can only fall back on the model's
own confidence. That is enough to flag a whole page (E20) but not to locate or reduce the errors.

| ID | Change | Why | How it is tested |
|---|---|---|---|
| T1 | **Stop discarding resolution.** When the embedded image is larger than the 2576 px render, or glyphs are small, transcribe in 2–3 overlapping horizontal bands at native resolution and stitch on the overlap. | E19: hadpi loses ~22% linear resolution before the model reads it. | WER on truth pages, bands vs whole page. |
| T2 | **Two independent readings and a deterministic comparison.** The line-faithful reading (§4.2) is the second reading. Compare it with `text_md` using the same word-multiset similarity `validators.py` uses against embedded text. Disagreeing spans go to a third, zoomed read of just those lines; two-of-three wins, otherwise the span is flagged for the human. | Gives scanned pages the oracle they lack. Independent errors rarely coincide. | Fraction of true errors (from truth) that fall inside flagged spans; false flag rate. |
| T3 | **Review gate (revised after Step 0).** The existing gate already flags low confidence and `illegible`; it is now pinned by `tests/review_gate_regression.py`. Remaining work: add low two-reading agreement (T2) as a trigger; reject non-finite confidence values (`ghessehaye-majid` p18 has `confidence: Infinity`, which passes `< 0.70`); decide whether Accept should keep a visible mark when an `illegible` flag was never resolved. | Step 0 found the original premise false (E20). | Regression test exists; extend it with each addition. |
| T4 | **Diplomatic rule and an explicit unreadable marker.** The prompt forbids completing or paraphrasing text that cannot be read, and requires a marker (e.g. `⟦؟⟧`) instead. The marker is **rendered visibly in the EPUB** until a human resolves it (D3), and review lists every unresolved marker. | E22: fluent legal phrasing that is not on the page. A marked gap is cheap to fix; a confident invention is not. | Count of invented words on truth pages; marker rate. |
| T5 | **Image clean-up before reading** (flatten lighting, straighten warp), as an A/B test. Do not assume it helps: vision models are often robust to this and resampling can hurt. | Cheap to test once §4.1 exists. | WER with and without. |
| T6 | **Geometry as a transcription check.** After alignment, flag: read lines with no detected line (or the reverse), lines whose text length is far from their ink width, words whose width is far from expected. | Catches skipped or duplicated lines and locally garbled words, with a location. | Recall of truth errors among flags. |
| T7 | **Record both forms.** Keep the diplomatic line reading next to the normalized `text_md` in the workspace. | E23: the aligner needs "as printed"; the EPUB needs normalized text. | Contract test. |

Model choice is part of T1/T2: test whether the second reading should use a different model or
resolution from the first, so that their errors are less correlated.

---

## 6. Measurement

### 6.1 Synthetic benchmark (new; the main instrument for "all fonts")

- **Generate**: lay out real Persian text (from our own transcriptions) with PyMuPDF
  `insert_htmlbox` in a chosen font, size, leading and justification. Read exact word rectangles
  back from the generated PDF with `get_text("words")`. Rasterize.
- **Font classes**: modern naskh (Vazirmatn, Noto Naskh), older book faces, typewriter-style,
  nastaliq (Noto Nastaliq Urdu or similar), bold headings, mixed digits and Latin. Font sourcing
  and licenses are a research task `[UNVERIFIED which typewriter-style Persian fonts are freely usable]`.
- **Layout classes**: prose, verse in two hemistich columns, footnotes, headings, numbered
  articles, justified lines with kashida.
- **Degradations**, with truth boxes passed through the same geometric transform: smooth warp
  with slope varying 0–3.5° down the page (matching E18), blur, ink spread and dropout, uneven
  lighting, JPEG, overlaid stamps, signatures and underlines.
  - Added (§13): margin specks near line starts and ends, recorded in truth as non-text marks;
    word spacing reduced to the within-word gap for a set fraction of boundaries, with the true
    gap recorded per boundary. Without these the synthetic set cannot show O1 or O2.
- **Text-error injection**: substitute, delete and insert words in the text given to the aligner
  (not in the image), at 5% and 15%.
- Deterministic by seed. Stored as a manifest plus seeds, not as images.

### 6.2 Real truth (small, audited; guards against synthetic-only optimism)

- **Text truth**: the owner corrects the text of hadpi p1 and p2 fully against the image, as
  printed. This gives the WER instrument for §5. About 1 hour.
- **Geometry truth**: word rectangles for hadpi p1–p2, two `ghessehaye-majid` pages, and the
  existing audited `bachehaye_ghali` p66. Drafted with the annotation workflow, audited by the
  owner. Hadpi p3–p5 are held out for a final check.
- Owner commitment (D4): the full ~3 hours — text truth for hadpi p1–p2 and the audit of all five
  geometry pages.
- **Ongoing**: review-time feedback capture, §6.3.

### 6.3 Review-time feedback capture (owner priority, D4)

The reviewer already looks at every box and every correction. Each look is a label we currently
throw away. Capture it, in a form that helps **every later book**, not only the one being reviewed.

**What is captured** (each action is one click or one drag; none is mandatory):

| Action in review | What it records | Unbiased? |
|---|---|---|
| **Drag to fix** a box | the corrected rectangle(s) for that correction | No: only wrong boxes get fixed |
| **Confirm** a box ("✓ on target") | the shown box is right | No: reviewer chooses which to confirm |
| **Wrong line** quick mark | the box is on the wrong line (no redraw needed) | No |
| **Spot check**: the UI asks about a few randomly chosen boxes per session | right / wrong (+ drag) for a *random* sample | **Yes** — the only source usable for precision estimates |
| Text edit accepted (exists today) | printed text for that span, as corrected by a human | Partly |
| Unreadable marker resolved (D3) | the true text of a span the model could not read | No, but these are exactly the hard cases |

**Where it goes.** One append-only, project-level store next to `qc_history.json` (gitignored,
local only): `box_truth.jsonl`. Each record carries book slug, page, image SHA, Markdown SHA,
correction key, target text, the shown box with its status/confidence/engine version, the human
verdict or rectangle, the page descriptors from §4.1 (slope, stroke width, etc.), and a timestamp.
Records are keyed to the image, so they survive re-transcription and engine changes.

**How it benefits all books** (nothing learns silently; each use is a deterministic, versioned step):

1. **A growing real-page test set.** Every engine change is replayed against all stored records
   from all books. This is what finally fills the "zero audited held-out cases" gap (E14).
2. **Calibration.** Spot-check records set the confidence threshold for `located` so the ≥ 97–98%
   precision target is measured on real pages, per page type.
3. **Failure maps.** Wrong-box records grouped by page descriptors show which page conditions
   break which stage, and tell us where the next work should go.
4. **Transcription lessons.** Accepted text edits and resolved unreadable markers build a
   per-condition error profile (which letters, digits and fonts get misread). This feeds the
   prompt rules, the QC risk selection that `qc_history.json` already drives, and the T2 flag
   thresholds.
5. **Within-book help, immediately.** A fixed box is a hard anchor: the line is re-aligned around
   it, and the book's character-width table (§4.3) is re-estimated, so later boxes in the same
   book improve during the same review.
6. **Training data, if arm C is adopted.** Confirmed line/word boxes with corrected text are
   exactly the pairs needed to fine-tune a recognizer per typeface.

**Rules that keep the data trustworthy:**
- Pages are assigned to *development* or *held-out* by a fixed hash of the image SHA. Thresholds
  and rules are tuned on development records only; gates run on held-out records only.
- Precision and accuracy numbers use spot-check records only. Drag-to-fix and confirm records are
  biased toward failures and toward easy cases respectively; they are used for tests, anchors and
  failure maps, never for headline rates.
- A record is tied to the engine version that produced the shown box, so old failures are not
  counted against a new engine; they are replayed instead.
- Human rectangles are truth for that correction only. They do not replace audited page truth.

### 6.4 Metrics

- **Word map**: per-word correct / wrong line / missing, using `tests/bbox_metrics.grade` (v6),
  over *all* words on the page, not only corrections. One page gives ~300–400 graded words.
- **Calibration**: precision of `located`; share of words that are `located`.
- **Lines**: `tests/bbox_line_metrics.py` (recall, merges, splits, spurious).
- **Transcription**: WER and CER against text truth; invented-word count; flag recall and false
  flag rate.
- **Statistics**: pages are the sampling unit (bootstrap over pages). Paired comparisons against
  the current engine use the existing `compare --gate` discipline on a frozen population.

---

## 7. Steps, smoke tests, pass and kill criteria

Run in order. A KILL stops that track and returns to planning. Steps 2 and 3 can run in parallel
after step 1.

### Step 0 — Stop the bleeding (no new dependencies, $0)

- A page with zero readings shows no estimate. Chip: "no evidence (reader failed: …)". Page
  banner with a resume hint.
- T3 gate: investigate and pin with a regression test (done; no gate change was needed, see E20).
- Update CLAUDE.md and AGENTS.md to match the current engine (E15).
- Find the cause of E4 ($0.10 spent, nothing saved).
- **PASS:** regression tests for both fixes; hadpi review shows no amber boxes.
- Optional, not gating: re-run the current engine on hadpi (≤ $1) to see how it behaves with
  readings present.

### Step 1 — Instruments ($0 API; ~3 h owner time)

- Build the synthetic generator (§6.1) and the real truth sets (§6.2).
- Build feedback capture (§6.3) **now, against the current engine**: drag to fix, confirm, wrong
  line, spot check, and the `box_truth.jsonl` store with the development/held-out split. It does
  not depend on the new aligner, and every review done from here on then produces labels.
  - PASS: a drag, a confirm and a spot check each round-trip to a valid record; replaying the
    store against the current engine reproduces the recorded boxes; a regression test covers the
    record contract and the split.
- **Smoke test 1a (synthetic feasibility):** generate one page per font class; confirm correct
  Persian shaping by eye, and that `get_text("words")` rectangles enclose the rendered words.
  - PASS: ≥ 99% of words have a rectangle whose ink content matches (checked by ink overlap).
  - KILL: PyMuPDF cannot shape a class (likely candidate: nastaliq). Then use a HarfBuzz-based
    renderer for that class, or drop the class from the synthetic set and cover it with real pages.
- **Smoke test 1b (sim-to-real sanity):** run the *current* detector on synthetic warped pages.
  - PASS: it fails in the same way as on hadpi (merged multi-line blocks under realistic slope). If it
    does not, the degradations are unrealistic; fix them before trusting any synthetic number.

**Step 1 results (2026-11-11).** Built: `tests/synth_pages.py` (+ `synth_smoke_1a.py`,
`synth_smoke_1b.py`, `synth_pages_regression.py`, manifest), `farsi2epub/feedback.py` with the
review UI actions and `tests/box_truth_replay.py`, and the owner's text-truth draft packet in
`books/hadpi-articles/truth/` (private, gitignored).
- **Smoke 1a: PASS for modern naskh, typewriter (Courier New, genuinely fixed-pitch), bold and
  mixed (100%); FAIL for nastaliq (78–81%) and DecoType Naskh (90.9%).** In those faces ink
  overflows the layout box PyMuPDF reports. Shaping is correct. Consequence: **truth for grading
  is the per-word ink box (`ink_aabb`), which is exact for every class, not the layout box.**
  Steps 3–4 must use `ink_aabb`. Both local nastaliq faces are Urdu-style; a Persian nastaliq
  face is still a gap.
- **Smoke 1b: PARTIAL.** The earlier "> ~1°" onset was this plan's own unsupported guess; the
  slope measurements (E18) only bracket it between 0.1° and 2.2°. On synthetic pages the current
  detector is correct when clean and merges lines from about 2° of slope; with the full photo
  preset it merges paragraphs and the stamp/signature area much as on hadpi (16/28 and 13/29
  lines found). But synthetic merges hold at most 3 lines against ~8 on real pages.
  **Synthetic pages are easier than the real ones.** Before Step 3 relies on synthetic line
  numbers: enlarge stamp/signature overlays and add baseline jitter, and keep real pages as the gate.
- The embedded text corpus is small, so phrases repeat on a page; use a `books/` text source for
  aligner tests.
- Feedback capture: tests pass; drag-to-fix was checked in a real browser (coordinates correct
  while zoomed). Multi-line boxes, spot-check panel, Shift-pan and Escape are covered by automated
  tests only.
- Box-audit packet (geometry truth, §6.2): `tests/box_audit.py` drafted word rectangles for hadpi
  p1–p2 (from the owner's line text) and ghessehaye-majid p23 (prose) and p109 (dialogue + a verse
  couplet), and exposes the audited bachehaye_ghali p66 fixture in the same format; the owner audits
  every line in a local tool (`serve`). The draft is text-constrained (strip-tracked line finder,
  line-to-text DP, word-gap DP; uncertain lines/words flagged), **not** a locator candidate. On
  synthetic typewriter pages it places 96–99% of words (photo preset) and 99%+ (clean), with at most
  1 wrong word per page left unflagged; real hadpi pages are visibly harder (stamp lines,
  margin specks, underlines). Not audited yet: no real-page number exists until the owner finishes.

**Real geometry truth frozen and first baseline (2026-11-11).** Frozen: hadpi p1 (line 1
excluded: owner text flag, outline sits on an underline) and p2, majid p23 and p109 (owner-audited);
p66 (agent-audited, reported separately). Scorer: `tests/real_truth_eval.py`; "correct" is stricter
than §1 (over-wide boxes fail), with the §1-literal rate printed beside it. Offline, $0, evidence
on disk only:
- Live engine: 0/729 hadpi, 0/632 majid, 0/356 p66 words located and correct. Hadpi and majid show
  no box (no reader evidence for these pages; this is not a measure of the engine with evidence).
  p66 is all `estimated`: boxes cover the target for 52/60 sampled spans but are several times
  too wide (median IoU 0.15).
- The pre-Step-0 estimated path would put 480/729 hadpi words on the wrong line.
- Current line detector, clean one-to-one recall: hadpi 25/60 (10 merges holding 34 lines), majid
  54/54, p66 31/32. This is the reference row for step 3.
- Truth feeds back at 100% except hadpi p2 (358/362): two word pairs on lines 29–30 overlap across
  lines in the audit. Open truth questions: those lines, two over-wide last words (p1 line 22,
  p2 line 26), and line-angle jumps above 1° on some short lines.
Not measured: the current engine with reader evidence present on these pages (needs paid reads).

**Hard synthetic presets (2026-11-11).** §13's synthetic additions are built: `tw_photo_medium`
and `tw_photo_hard` add margin specks, tight word spacing, multi-line overlays, per-line baselines
and typewriter texture, on a flush-right memo layout with a large invented text pool. (v1's
non-justified layouts turned out to be flush-left, because MuPDF mirrors `text-align` for RTL;
fixed only in the new layout, so old hashes hold.) Typewriter, seeds 3–6, old photo → medium →
hard:
- current detector, largest merge: 7 → 10 → 10 lines (hadpi about 8). Most of this comes from the
  memo layout itself (layout alone: 11), not from the new degradations.
- draft splitter, words on the right ink: 97.7% → 85.3% → 67.7%; both edges within half a letter
  (the agent's stand-in, not the audit criterion): 91.9% → 67.0% → 42.3%. The owner's audit left
  62–77% of hadpi draft boxes unchanged; the two measures are different instruments.
- O1 cascade lines: 0 → 4 → 21. O2 boundary errors: 1.5% → 4.3% → 12%, almost all at gaps no wider
  than an in-word gap.
Medium is the closer match to hadpi; hard is harder than the real pages for both the detector and
the draft. Not built: kashida line filling, a real Persian typewriter face (Courier New is the
only fixed-pitch face installed), the width-class pitch grid, paper texture. Real pages remain
the gate.

### Step 2 — Transcription experiments (≤ $3 API)

- Pages: hadpi p1–p2 (text truth), plus two clean control pages from another scanned book.
- Arms: current pipeline; T1 bands; T4 diplomatic prompt; T1+T4; T2 two-reading agreement on top
  of the best single arm; folded vs separate line-faithful reading.
- **PASS:**
  - WER on hadpi falls by at least half relative to the current pipeline, **or** reaches ≤ 5%.
  - Invented words (words in the output with no counterpart on the page): ≤ 1 per page.
  - T2 flags contain ≥ 80% of the remaining true errors, with ≤ 10% of words flagged.
  - No WER regression on the clean controls.
  - Line-faithful reading: line count within ±1 of truth on every page; line breaks correct for
    ≥ 95% of lines.
- **KILL:** no arm brings hadpi WER under 10% and T2 flag recall is under 60%. Then the page is
  beyond current transcription, placement on it must stay at `line` level, and the owner is told
  so explicitly. Placement work continues for readable pages.

### Step 3 — Line finding (research venv for L2/L3; $0)

- Data: synthetic warped set (all font classes), hadpi p1–p2, majid ×2, p66.
- Arms: current detector; L1; L2; L3; each with and without masking.
- Also measured here (§13): text-block edge fit (error of each line's fitted start and end against
  audited line outlines), and how many truth words lose ink to masking (must be zero on real
  pages; report per page).
- **PASS (best arm):**
  - Hard real pages: clean one-to-one recall ≥ 95%, zero merges of two or more truth lines,
    ≤ 1 spurious line per page.
  - Synthetic: recall ≥ 97% for slopes up to 3.5°, per font class. Report nastaliq separately.
  - p66: no worse than the current 31/32.
- **KILL:** best arm < 85% on hard real pages. Line finding is then the research problem;
  replan around full page dewarping.
- Tie-break: L1 (no dependency) wins unless a neural arm is better by ≥ 3 points on any class.

### Step 4 — Word placement (≤ $1 API for arm V)

- Inputs: step 3 lines; line text from truth first (isolates the aligner), then from step 2's
  line-faithful reading (end to end).
- Arms: G; C (kraken Persian model, zero-shot); G+C agreement; V (control). Optional: R
  (render and warp); Tesseract `fas` + text alignment, only if the owner installs the language data.
- Reported separately for every arm (§13): **O1 cascades** (runs of consecutive words each placed
  on a neighbour's ink) and **O2 boundary errors** (right word, an edge off by a letter or more),
  on real pages and on the synthetic speck and tight-spacing sets. G is run with and without the
  "not text" state so its effect is measured, not assumed. Touching-word cases are included so
  the numbers can settle D5.
- **PASS (some arm, end to end):** the §1 targets, i.e.
  - clean (p66 + synthetic clean): ≥ 95% words `located` and correct;
  - hard (hadpi + synthetic degraded): ≥ 85%;
  - precision of `located` ≥ 98% clean, ≥ 97% hard;
  - wrong line ≤ 0.5% clean, ≤ 1.5% hard;
  - **error-injection robustness:** with 15% of words corrupted in the text, ≥ 90% of the
    *uncorrupted* words are still correct, and ≥ 80% of corrupted positions end up not `located`;
  - deterministic; ≤ 10 s per page on the owner's Mac.
- **KILL:** no arm reaches 70% on hard pages with *truth* line text. The alignment idea itself is
  then failing; replan.
- Per-class reporting: nastaliq and verse layouts are reported separately. If G fails a class
  that C passes, C becomes required for that class and the torch dependency question goes to the
  owner with numbers.

### Step 5 — Correction-level paired comparison ($0)

- Population, frozen before running: every hadpi correction, the 86 `ghessehaye-majid` cases
  (cached evidence, free replay), the 111 bachehaye fresh30 cases.
- **PASS:** covered-target rate improves with Wilcoxon p < 0.05; wrong-line rate not higher; a
  visual audit of every changed box finds no wrong-occurrence placement; held-out hadpi p3–p5
  meets the hard-page targets; and the new engine does not do worse than the current one on the
  held-out half of `box_truth.jsonl` collected so far.
- Fail: keep as experimental; do not promote; record the result.

### Step 6 — Integrate, then delete

- `--bbox-engine align|legacy` on `review`, `transcribe`, `geometry`. Default to `align` only
  after step 5 passes.
- After one release as default: delete the components listed in §4.5; move placement code out of
  `review.py`; update the agent docs with the measurement of record.

**Spend cap:** ≤ $5 API in total. Owner time ≈ 3 h (text truth, geometry audit).

---

## 8. Reproduction snippet (E9, E11)

```python
import fitz, sys; sys.path.insert(0, '.')
from farsi2epub import locate
d = fitz.open('books/hadpi-articles/source.pdf')
for pn in range(len(d)):
    p = d[pn]; L = locate._scan_page_lines(p)
    print(pn + 1, len(L), sorted(round(l.rect.height, 1) for l in L))
    for l in L:
        p.draw_rect(l.rect, color=(1, 0, 0), width=1)
        for w in l.words: p.draw_rect(w, color=(0, 0, 1), width=.4)
    p.get_pixmap(matrix=fitz.Matrix(2, 2)).save(f'out/hadpi_p{pn+1}_detector.png')
```

---

## 9. Risks

| Risk | Likelihood | Response |
|---|---|---|
| The line-faithful reading gets line breaks wrong on dense or warped pages | Medium | Checked against detected lines (§4.2); step 2 measures it; affected lines fall back to `line` |
| Both readings share the same error, so T2 misses it | Medium | Vary model or resolution between readings; T6 geometry check is independent of both |
| Arm G fails on nastaliq and heavily kerned faces | High for nastaliq | Arm C; reported per class; honest `line` fallback meanwhile |
| kraken's Persian model is poor on typewriter print | Medium | G does not need it; per-book fine-tuning as escalation |
| Stray marks shift a line's boxes (§13 O1) | High on photographed pages | Conservative masking (§4.1) + text-block edges + "not text" state in G (§4.3); measured in steps 3–4 |
| Word gap no wider than a letter gap (§13 O2) | High on typewriter print | Width-led cuts with width classes (§4.3); arm C or R if G's O2 rate stays high |
| Synthetic pages are easier than real ones | Medium | Smoke test 1b; real truth gates every step; held-out hadpi pages |
| torch is too heavy for the CLI | Medium | Only needed if C proves necessary; optional extra |
| The unreadable marker complicates the EPUB build | Low | Contract test in T4; marker stripped or rendered at build |
| More layers on top of old ones | Medium | Step 6 deletion is part of done |

---

## 10. Owner decisions (final, 2026-11-11)

| # | Question | Decision |
|---|---|---|
| D1 | What to show when a word-level box is not trustworthy | **Line box or nothing.** A full-line box labeled "line only" when the line is known; no box with a reason otherwise. Amber proportional guesses are removed. This replaces the September coverage-first policy. |
| D2 | One extra whole-page, line-by-line reading per scanned page (~$0.03/page) | **Approved, on all scanned pages**, not opt-in. |
| D3 | Explicit unreadable marker in transcriptions | **Approved, and shown in the EPUB** until a human resolves it. Never stripped silently. |
| D4 | Owner time for truth | **The full ~3 hours**: correct the text of hadpi p1–p2 and audit word boxes on all five geometry pages. **Review-time feedback capture (§6.3) is an owner priority** and is expanded beyond drag-to-fix, on the condition that what it records benefits all future books. |
| D5 | kraken/torch; Tesseract Persian data | **Deferred, not asked.** Comes back to the owner with step 4 numbers only if arm C proves necessary. Tesseract baseline stays optional. |

---

## 11. Answers to v1's open questions

1. *kraken for CTC?* Reasonable, but it is no longer the lead arm. Verify the model's character
   set and display-order convention before use.
2. *NFD and letterform mapping?* Fold both sides to one canonical stream (ZWNJ, diacritics and
   punctuation removed; ی/ي and ک/ك unified). Alignment works on that stream; boxes map back by
   index.
3. *±5° deskew?* Wrong tool. Slope varies within the page (E18). Follow lines locally.
4. *Thresholds with 7 pages?* Not defensible alone. The synthetic benchmark supplies the volume;
   real pages gate; pages are the bootstrap unit.
5. *Model grounding arm?* Keep as a deterministic-graded control. It informs, it does not ship
   unless it wins outright.
6. *Remove estimates?* Replaced by the `line` status (owner decision D1).
7. *Missed root cause?* Yes: transcription quality (RC4). Addressed in §5.

## 12. Items the implementer must verify first (cheap, read-only)

- `kraken.align.forced_align` exists and what it returns.
- kraken's training text order for RTL (display vs logical).
- The Persian model's character set against all characters in `books/*/text/*.md`.
- Freely usable typewriter-style and nastaliq Persian fonts.
- Whether `insert_htmlbox` shapes nastaliq correctly (smoke test 1a).
- Whether Surya supports Persian recognition (only its line detection is used here).

## 13. Owner's audit observations: what the plan already covered, and what changed (2026-11-11)

**Source.** While auditing word boxes on hadpi p1–p2, the owner reported two repeating error
patterns. The boxes came from the audit tool's draft splitter (`tests/box_audit_draft.py`, a
simple text-constrained gap method), **not** from arm G, which is not built. So these are not
measurements of the planned engine. They are evidence of what a gap-led method does on this print,
and a test of whether the plan's design would have avoided it. Neither pattern is specific to
Persian or to typewriting.

- **O1, stray ink at the line start.** A speck in the margin receives the first word's box; every
  later box on the line is then off by that many words.
- **O2, word gap no wider than a letter gap.** The cut lands inside a neighbouring word: a box
  takes the next word's first letters, or loses its own last letters.

Measured on the draft against the audited boxes: 15 of 61 hadpi lines have three or more
consecutive wrong boxes, against 1 of 54 on the printed book pages.

### 13.1 O1 against the plan's existing measures

| Existing measure | What it does for O1 | Where it stops |
|---|---|---|
| §4.1 masking of "isolated margin marks" | Removes marks that are clearly isolated and small | A yes/no decision before any text is matched. A mark at line height, near the block, the size of a dot or narrow letter cannot be masked safely |
| §4.3 arm G expected widths | A speck is too narrow to be a word, so a full cascade is less likely than in the draft | G must still assign every ink run to some word: the first box stretches to the speck, and several specks can still shift the line |
| §4.4 insertions and deletions | Lets a *word* be missing or extra | Nothing lets *ink* be extra |
| §4.2 line-to-text check by length | Allows a detected line with no text (stamp, underline) | Works on whole lines, not on marks inside a line |

Verdict: partly covered; the hard case (a speck that survives masking) was not. Changes made:
conservative-masking rule and text-block edge fit in §4.1; "not text" state in arm G (§4.3); the
edge fit and masking damage measured in step 3; G measured with and without the new state in
step 4.

### 13.2 O2 against the plan's existing measures

| Existing measure | What it does for O2 | Where it stops |
|---|---|---|
| §4.3 arm G: cuts placed by expected width, gaps preferred | This is the designed answer: the boundary comes from letter widths, not from white space | Only as good as the width table |
| §4.3 fused boundaries (E23) | Covers a word pair with no gap at all | Same dependence on widths |
| §4.3 "typewriter converges to all equal" | Was meant to make typewriter pages the easy case | Probably wrong: sources describe two or three width classes on Arabic-script typewriters. With wrong widths a width-placed cut lands part of a letter off, which is O2 in a milder form |
| §4.3 arm C (forced alignment), deferred by D5 | The published answer to this exact problem: boundaries from where characters are | Needs kraken/torch; its Persian model is unverified on typewriter print (§12) |

Verdict: covered by design, with one wrong assumption. Changes made: width classes on a common
pitch replace "all equal", and "width leads, gaps refine" is stated as a rule (§4.3); O2 errors
are counted separately and touching-word cases are included in step 4 so that D5 is decided on
numbers. D5 itself is unchanged.

### 13.3 What the literature adds (searched 2026-11-11; targeted, not a systematic review)

- Separating word gaps from within-word gaps is treated as a classification problem and is known
  to fail where the two populations overlap or words touch. Arabic script is the hard case,
  because every non-joining letter leaves a gap inside a word. No threshold fixes O2.
- With known text, the established approach is alignment rather than segmentation (HMM/Viterbi,
  later CTC). Reported word-alignment rates on historical handwriting run from about 69% to 93%
  across papers with different data and definitions. They are not comparable with each other or
  with the §1 targets.
- A learning-free variant renders the known text and warps it onto the line image. Added as
  optional arm R (§4.3).
- Fixed-pitch typewriting is segmented by fitting a character-cell grid, not by gaps; degraded
  typewritten pages still break and merge characters, so the grid is a prior, not a solution.
- Margin specks and scan noise are handled as a separate problem before segmentation.

### 13.4 Status

All of §13 is reasoning from the design plus a draft tool's errors. Arm G is not built; neither
its protection nor its weak points are measured. The owner's audited boxes decide it in step 4.
The synthetic additions (§6.1) are being built now as a step 3 prerequisite.

Sources: Stutzmann & Bluche, text-image alignment overview
(dh-abstracts.library.virginia.edu/works/2258); Moccia Code alignment
(ncbi.nlm.nih.gov/pmc/articles/PMC9864051); Rothfeder, Manmatha & Rath
(maroo.cs.umass.edu/pub/web/getpdf.php?id=463); Hassner, Wolf & Dershowitz, OCR-free transcript
alignment (openu.ac.il/home/hassner/projects/Ofta); Peer, Scius-Bertrand & Fischer, CTC alignment
(arxiv.org/abs/2508.07904); kraken forced alignment (kraken.re/7.0/user_guide/api.html);
Louloudis thesis on gap classification (di.uoa.gr/sites/default/files/documents/grad/phdbook/louloudis.pdf);
Retsinas et al., typewritten recognition
(primaresearch.org/www/assets/papers/HIP2015_Retsinas_TypewrittenDocumentRecognition.pdf);
Berg-Kirkpatrick, Durrett & Klein 2013 (nlp.cs.berkeley.edu/pubs/BergKirkpatrick-Durrett-Klein_2013_OCR_paper.pdf);
Arabic typewriter widths (blog.29lt.com/2014/01/01/29lt-makina-multilingual-typewriter-typeface,
patents.google.com/patent/US2940575A/en).
