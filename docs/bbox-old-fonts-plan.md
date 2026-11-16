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
| **E25** | The API's 4,784-token image cap downsizes the 1821×2576 hi-res render again, to about 1623×2296 (scale 0.891; `count_tokens` returned 4,759). The model therefore sees about 68% of hadpi's native linear resolution, and 0.89× of the render on any portrait page of any book. Native bands of at most 82×58 patches are not downsized (3,775 tokens measured). The `config.py` comment saying there is no API downscaling is wrong for portrait pages. | `tests/transcription_experiments.py` (`effective_px`), `messages.count_tokens` | verified 2026-11-12 |
| **E26** | Claude Opus 5.5 (effort medium; $4/$20 per MTok taken from the claude-api skill's table, not from a bill) costs about $0.085–0.09 per hadpi page in one batched call (about 13k input, 1.8k output tokens), for 3 bands or for about 30 line crops. Opus cannot run with thinking disabled. | `tests/transcription_experiments.py` (`runpl`, `run t1r_opus`); ledger | measured 2026-11-12 |
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

**Step 2 result (2026-11-12, $1.82 of the $3 cap; `tests/transcription_experiments.py`).**
No arm passes on hadpi. WER below is the lenient folding (ignores ZWNJ, punctuation and pure
typewriter joins); strict "as printed" WER is 7–24 points higher on hadpi.

| Arm (Sonnet 5 unless stated) | hadpi p1 | hadpi p2 | controls (majid p23 / p109) |
|---|---|---|---|
| Stored text (what the owner has) | 14.9% | 35.3% | – |
| Current pipeline, 3 runs | 17.2–33.5% | 32.9–39.5% | 0.9% / 0.0% |
| T1 native-resolution bands, 2 runs | 19.5–20.1% | 32.9–34.1% | 1.8% / 0.7% |
| T4 diplomatic prompt, whole page | 35.5% | 49.9% | 0.6% / 1.0% |
| Bands + revised prompt (added arm), 3 runs | 11.5–21.2% | 30.9–35.9% | 1.8% / 1.0% |
| Same with adaptive thinking, 1 run | 13.5% | 37.1% | – |
| Haiku 4.5 line reading | 92.6% | 94.7% | 36.3% / 21.6% |

- **Run-to-run spread is as large as the differences between arms** (current pipeline on p1: 60, 90
  and 117 errors in three runs). Bands plus the revised prompt beat the current pipeline on p1 in
  mean only, with overlapping ranges, and show no difference on p2.
- Invented words: 9–41 per page for the best arm (criterion: ≤ 1). The model almost never uses
  the unreadable marker, so markers cannot be relied on to find errors (T4, D3).
- T2: two readings that differ in prompt and resolution catch about 80% of errors but flag
  22–34% of hadpi words (0.7–2% on the clean controls). Two runs of the same arm share most
  errors. Usable as a page/line review trigger (T3), not as a word-level error locator. The
  criterion "≥ 80% recall with ≤ 10% flagged" is unreachable when WER exceeds 10%; restate it as
  flag rate ≤ WER + 5 points, and ≤ 3% on clean pages.
- Line-faithful reading: a separate whole-page call is unreliable on hard pages (24, 32, 42 and
  once 2 lines); read from bands it gives 29–32 lines on hadpi (77–100% of breaks right) and is
  exact on the controls. §4.2 should take line text from the banded reading.
- Remaining errors (best arm): 36% letter- or dot-level misreadings of a real word, 27% unrelated
  words substituted or inserted, 15% dropped words; about 40% of errors sit on 5 of ~30 lines
  per page (faint lines, stamp-covered lines).
- **PASS criteria fail. KILL is formally not triggered** (T2 recall ≈ 0.8, but only at a high flag
  rate).
- **Not tested, and the reason this is not yet a kill:** a stronger model, and reading one
  straightened, enlarged line at a time. The owner's text-truth draft was made that way (line
  crops, read by an Opus-class agent) and needed about 1 correction on p1 and about 7 on p2,
  against 40–133 errors for every whole-page arm here. That comparison is biased in the draft's
  favour (the truth was made by correcting that draft), but the gap is far larger than any bias
  of that kind explains. Follow-up arm: per-line crop reading, Sonnet 5 and a stronger model, on
  the remaining Step 2 budget. Until it is measured, "hadpi is beyond word-level transcription"
  holds only for whole-page reading with Sonnet 5.

**Step 2 follow-up (2026-11-12, +$1.04; Step 2 total $2.86 of $3).** The model is the dominant
lever, not the cropping.

| Arm | hadpi p1 | hadpi p2 | controls |
|---|---|---|---|
| Opus 5.5, straightened line crops, batched, 2 runs ($0.088/page) | 2.9–4.3% | 4.2–5.3% | 1.5% / 1.4% |
| Opus 5.5, the same 3 native bands as before, 1 run ($0.086–0.089/page) | 2.3% | 4.5% | – |
| Opus 5.5, one line crop per request ($0.257/page) | – | 9.5% | – |
| Sonnet 5, line crops, batched, 1 run ($0.043/page) | 22.7% | 46.3% | – |
| Sonnet 5, bands + revised prompt, 3 runs (from above) | 11.5–21.2% | 30.9–35.9% | 1.8% / 1.0% |

- Opus reading bands is as good as Opus reading straightened lines; Sonnet reading lines is worse
  than Sonnet reading bands. Per-line cropping adds nothing for reading. Spread for Opus is small
  (10 and 15 errors on p1; 14 and 18 on p2).
- Strict "as printed" WER on p2 stays 23–31% for Opus: the remainder is joins, ZWNJ and
  punctuation spacing, not misread words. T7 (keep both forms) still matters for the aligner.
- Against the criteria, for Opus: WER halves and is near or under 5% (**pass**); invented words
  1 per run on p1 (**pass**), 4–7 on p2 (**fail**); two Opus readings by different routes (bands
  vs line crops) flag about 80–87% of errors with 4–5% of words flagged (**pass under the restated
  criterion**); controls 5 and 4 errors against 3 and 0 for the current pipeline, **unresolved**
  because several of those look like errors in the stored control text.
- Opus uses the unreadable marker (17 of its 57 errors over four hadpi runs sit inside markers),
  correctly on stamp-covered words and over-cautiously on the company name.
- **Anchoring caveat, attached to every Opus number here:** the hadpi truth was made by
  correcting a draft that an Opus-class model read from similar crops. These numbers are weaker
  evidence than the Sonnet ones. They do not explain Sonnet being 5–10 times worse on identical
  crops. **An unbiased test needs owner text truth for held-out hadpi p3–p5, before adoption.**
- The earlier conclusion "beyond word-level transcription" is **withdrawn**: it held for Sonnet 5.
- Not run: the context variant (crop with neighbouring lines), cut for budget; repeats of Opus on
  bands; Opus on the whole page with the current prompt; effort levels other than medium.
- Truth spots for the owner's second look (page and line as in `truth/crops/` names): hadpi p2
  line 15 (a number: three Opus readings agree on a value different from the truth); words under
  the stamp on hadpi p1 line 30 and p2 lines 30–31; majid p23 lines 2, 3, 12, 15 and p109 lines
  9, 14 (letter-level differences where the stored text may be the one that is wrong).
- **Proposed for the owner (not adopted):** escalate hard no-oracle pages (review gate fires:
  confidence < 0.70 or `illegible`) to Opus 5.5 reading native bands with the revised prompt,
  about $0.09/page; keep Sonnet 5 for clean pages; use a second Opus reading by a different route
  as the review flag where wanted (+$0.09/page).

**Held-out reading test, stage 1 (2026-11-12, $1.10 of a $1.50 cap): readings frozen and taken,
drafts with the owner.** Configurations frozen on p1–p2 before any p3–p5 image was opened
(`out/transcription_experiments/heldout_freeze.json`, with a byte-exact copy of the script):
Opus 5.5 on native bands (2 runs per page), Sonnet 5 on the same bands (2 runs), the production
whole-page Sonnet call (1 run). Per-line crops were not run (no audited outlines for these pages).
All 15 readings succeeded first time; nothing was scored. Text height reaching the model on
p3–p5: 27–29 px glyphs on bands, about 18 px on the production whole page (0.64–0.67× native).
The owner's drafts (`books/hadpi-articles/truth/000{3,4,5}.lines.txt`, private) are built from
the readers' votes with **every non-unanimous word marked with all alternatives in sorted
order**: 31, 31 and 33 lines; 63, 88 and 87 markers; 41–46% of words inside markers. Built the
same way on p1–p2, such a draft left 1 and 2 words wrong outside markers, all "as printed"
differences (a join, a ZWNJ, punctuation spacing), with the caveat that p1–p2 truth was itself
made from an Opus-class draft. Next: the owner corrects the three drafts; then the cached
readings are scored once, with no further API calls.

**Held-out reading test, stage 2 (2026-11-14, $0): scored once; the development finding holds.**
The owner audited all three drafts in the text audit tool (`status: audited`; 364, 414 and 386
words; one word and one punctuation mark left unreadable on p4). `tests/heldout_score.py` scored
the 15 cached readings with the byte-exact frozen copy of the scorer (SHA checked against the
freeze); results in `out/transcription_experiments/heldout_results.json` and
`heldout_summary.txt`. Errors over truth words, by page p3 / p4 / p5:

| Reading (frozen) | Lenient, each run | Lenient, pooled | Strict (as printed), pooled |
|---|---|---|---|
| Opus 5.5, native bands (`t1r_opus`), 2 runs | 2.3, 2.0 / 3.6, 2.3 / 2.5, 1.9% | **54/2194 = 2.5%** | 379/2326 = 16.3% |
| Sonnet 5, same bands (`t1r`), 2 runs | 12.2, 10.2 / 16.8, 26.0 / 29.0, 19.9% | 422/2194 = 19.2% | 736/2326 = 31.6% |
| Production whole-page Sonnet 5 (`current`), 1 run | 8.2 / 20.6 / 10.1% | 145/1097 = 13.2% | 393/1163 = 33.8% |

- Opus stays inside the 2–5% lenient range seen on p1–p2 on every page and run; Sonnet stays at
  8–29%. Three pages are still few (pages, not words, are the unit), but this is the first
  measurement on pages nobody looked at before the configurations were frozen.
- Strict error for Opus is 7–25% per run and is almost entirely word joins and splits (strict
  minus ortho/lenient); p4 moved from 25.2% to 11.4% between two runs of the same configuration.
  This is the join behaviour D14 and D16 already cover: placement scores outer edges.
- Invented words: Opus 0–3 per page; Sonnet bands 6–37; production 7–39.
- The draft method held: outside the markers 1 word per page differed from the owner's truth
  under the strict reading (0, 0 and 1 under the lenient reading).
- Two Opus runs as a review flag caught 5/8, 12/14 and 6/9 of the first run's errors while
  flagging 1–4% of words; an Opus and a Sonnet run caught 5/8, 14/14 and 9/9 but flagged 12–28%.
- Remaining bias, smaller than on p1–p2: the truth is the owner's choice among Opus and Sonnet
  readings with every disagreement marked, so a word that all four readings got wrong the same
  way would only be caught by the owner's reading of the plain text.
- The proposal above (escalate hard no-oracle pages to Opus 5.5 on native bands) now has its
  unbiased test. **Approved by the owner on 2026-11-14 (D17) and integrated; see the block below.**
- A book built from the owner's truth for all five pages: `tests/truth_book.py` writes the
  separate workspace `books/hadpi-articles-truth/` (lines joined into paragraphs, no word
  changed, checked word for word) and `farsi2epub build hadpi-articles-truth` gives
  `out/asasnameh-truth.epub` (epubcheck clean). The model-transcribed `hadpi-articles` workspace
  is left as the experiments' baseline.

**Book-level re-read trigger (2026-11-14, D20, $0).** The page trigger rests on the first reader's
confidence, which missed held-out hadpi p3 in one run (0.75). Two candidates were measured on cached
readings only. (a) Disagreement between the first pass and a Sonnet band reading (T2), as a share of
first-pass words: hadpi p1–p2, 18 reading pairs, 17–41%; 60 printed pages of majid and bachehaye
(one band reading each), median 2.7%, 90th percentile 10%, two above 15% (an 8-word page, and majid
p3 at 16.8% where review changed 3 of 117 words). The margin to hadpi's lowest pair (17.3%) is thin
and the second reading costs about $0.03 on every scanned page. Not adopted. (b) Share of a book's
no-oracle pages whose first pass trips the gate: hadpi 4–5 of 5 in all 128 combinations of cached
first-pass confidences; eight other workspaces 0–29% (bachehaye_ghali 24 of 89, jafarian-factory-sale
5 of 17). Adopted with the line at one half and a minimum of 3 pages (`transcribe.hard_book_decision`,
`config.HARD_BOOK_SHARE`). Smoke test with criteria fixed first: `tests/hard_book_rule_smoke.py`, PASS.
Limits: one hard document; the half line sits in a wide gap and is not tuned; a single hard page in a
clean book is still missed; held-out p3–p5 entered only through their first-pass confidence. No live
call has been made.

**Production integration of the hard-page reader (2026-11-14, D17, $0).** `farsi2epub/hard_reader.py`
is the measured arm `t1r_opus` and nothing else: same bands, instruction, schema, output limit and
effort. `tests/hard_reader_regression.py` compares it with the experiment script, and on hadpi
p1–p2 the full production request (image bytes included) equals the cached experiment request.

- **When:** in `transcribe`, after the normal pass, for a non-blank page with no embedded-text
  oracle that trips the review gate (confidence < 0.70 or `illegible`). `--hard-pages opus`
  (default) | `all` (every page without an oracle) | `off`. About $0.09–0.10 a page, counted in
  `--max-cost`; not in `analyze`'s estimate (it prints a note for scanned sources).
- **What is stored:** the page's words are the reader's words in order. A marked guess keeps its
  word and is listed in the sidecar (`hard_reader.guessed_words`); an unreadable word becomes
  `[؟]` and sets `illegible`. The reader returns printed lines, so paragraph breaks and heading
  levels are carried over from the first pass by aligning the two word sequences
  (`lines_to_markdown`); when fewer than half the words pair up, each printed line becomes its own
  paragraph. The first pass is kept as `text/NNNN.first.md`. The page always stays `needs_review`
  and keeps the first pass's confidence. A first pass with verse, a table or footnotes is not
  re-read (that structure cannot be carried over yet); a failed call leaves the first pass in
  place with the reason in the sidecar.
- **Replay from cached responses through `transcribe._transcribe_one` ($0):** hadpi p1 90 → 8
  lenient errors of 349, p2 115 → 15 of 337 (development pages); run once on the held-out pages,
  p4 80 → 14 of 388 and p5 37 → 9 of 366. In every case the final page scores exactly as the
  cached Opus reading scored directly, and the first pass's block count is kept.
- **Limit found on the held-out pages, not tuned:** in the cached production run p3 reported
  confidence 0.75, so the gate did not fire and the page kept its 28 errors of 343 (the stored
  sidecar from the original run says 0.60: the model's confidence moves between runs). With
  `--hard-pages all` p3 goes 28 → 8. The trigger rests on the first reader's own confidence; a
  better trigger (for example two-reading disagreement, T2/T3) is open and must not be chosen on
  p3–p5. For a book known to be typewritten or badly photographed, use `--hard-pages all`.
- No live API call was made for the integration; the request is byte-identical to calls that
  succeeded in Step 2.

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

**Step 3 result (2026-11-12, $0).** Built `tests/line_finder.py` (arm L1: paper-field and
Sauvola binarisation; conservative size/isolation masking with in-line restoration; strip peaks
linked across the page with neighbour-slope prediction; robust quadratic centre curves; shear
rectification; consensus text-block edge fit), `tests/line_finder_eval.py`,
`tests/line_finder_regression.py`. L2 (kraken) and L3 (Surya) were not run: they need installs.

| Data set (truth lines) | Current detector | L1 + masking |
|---|---|---|
| hadpi p1–p2 (60) | 25 clean, 10 merges, 5 spurious | **60 clean, 0 merges, 20 spurious** |
| majid p23, p109 (54) | 54, 0, 0 | 54, 0, 0 |
| p66 (32, agent-audited) | 31, 0, 0 | 32, 0, 0 |
| synthetic, untouched seeds (1999) | 1019 (51.0%), 293 merges | 1975 (98.8%), 0 merges |

- Against the criteria: recall ≥ 95% on hard real pages **passes** (100%); zero merges **passes**;
  p66 **passes**; synthetic ≥ 97% per class **passes** on the untouched seeds (typewriter 98.4%,
  nastaliq 99.1%, mixed 97.1%; mixed is 94.9% on the development seeds, because a bidi-wrapped
  token's truth box spans two lines). Kill not triggered.
- **"≤ 1 spurious line per page" fails as written: 11 and 9 on hadpi.** One per page lies inside
  the text block (a mark past a line end; a typed underline). The rest lie below the body:
  signatures, a handwritten name, the stamp, the scanner logo. None is in the truth and no ignore
  zone covers them. **Open decision for the owner:** either count spurious lines only inside the
  fitted text block (L1 then passes with 1 per page), or have those regions annotated. Until
  decided, the criterion is recorded as failed. What matters downstream is whether §4.2's
  line-to-text alignment leaves these extra lines without text; step 4 must measure that.
- Masking: no truth word loses ink on any real page. It does not change recall on real pages
  (recall comes from following lines locally); on synthetic pages it removes the merges caused by
  stamps and signatures (23 → 0). On synthetic pages some text is lost where a stamp or signature
  touches a heading through its underline.
- Edge fit: within 0.02–0.13 letters on the book pages; on hadpi the median is 0.2 letters but
  13 line starts are more than a letter off, because marks just past the line end are absorbed
  into the line (§13 O1). The fit can also cut real text on ragged layouts. **Use it as a flag
  (flush / inside / protrudes), not as a position.** Step 4 must not take a line's ink extent or
  its fitted edge as the line's start; arm G's "not text" state is needed exactly there.
- Real pages were scored three times during development; one rule (margin marks) and the
  edge-fit method changed after seeing them (`out/line_finder/run_log.txt`). Recall on hadpi was
  already 60/60 on the first scoring, before any change motivated by real pages. Still, hadpi
  p1–p2 are no longer an untouched test for L1; held-out p3–p5 are.
- 46 free parameters, all relative to stroke width, glyph height or line spacing. Run time
  0.7–3.7 s per real page. Deterministic. Not covered: colour (real stamps are coloured; only
  grey levels are used), real two-column verse, real footnotes.

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

**Step 4 result, arm G (2026-11-12, $0; `tests/word_placer.py`, `word_placer_eval.py`,
`word_placer_regression.py`).** Built: arm G (exact DP aligning text pieces to ink runs in each
rectified line, with "not text" skip, split, fuse and word-deletion moves; a per-page
(letter, joining form) width table learned by align → re-estimate; confidence from the margin
between best and second-best; boxes snapped to the word's own ink), and §4.2 line-text assignment
(by length, with an unbroken-text variant). **Arm R was not built; arms C and V were not run.**
All rates use the strict scorer ("located and correct" over all graded words).

| Experiment | hadpi p1–p2 (729) | majid p23, p109 (632) | p66 (356, agent-audited) |
|---|---|---|---|
| A: truth lines + truth text, G | 40.3%, precision 87.2% | 64.1%, 98.3% | 71.3%, 100% |
| A, G without "not text" | 43.9%, 86.0% | 83.1%, 98.1% | 96.1%, 99.7% |
| A, every G box shown (no threshold) | 70%, 77% | 95.4%, 97.9% | – |
| B: L1 lines + truth text, G (end to end) | 26.3%, 83.8% | 63.0%, 97.3% | 69.7%, 98.4% |
| Draft splitter (always shows a box) | 70.8% | 97.2% | – |
| D: stored text, not line-broken, G | 2/729; 239 `line` boxes on the wrong line | 30/632; 213 on the wrong line | 48.0%, 93.4% |

- **Kill criterion triggered for G on hard pages** (no arm reaches 70% with truth line text: best
  43.9%). Showing every G box gives 70% at 77% precision, the same as the draft splitter. The
  boxes themselves are wrong, not only the threshold. Synthetic untouched seeds agree: hard 48%
  at 94.8%; nastaliq 25% at 79%.
- On printed pages G's boxes are good (95% correct at 98% precision unthresholded); recall is
  lost to a threshold chosen on synthetic pages, and to the "not text" state, which lowers
  margins on clean lines. Targets (≥ 95% at ≥ 98%) are missed, by calibration rather than by the
  aligner.
- Wrong line: 0 on every real page in A and B (target met). Run time ≤ 3.9 s per page.
- "Not text" state: fixes the constructed O1 case (5/5 lines) and lowers cascades and raises
  precision on hard pages (hadpi B: 15 cascades vs 19; precision 83.8% vs 81.4%), but costs recall
  on clean pages. It should be applied by page condition, not everywhere.
- Width model: the learned table is essential (uniform widths: hadpi ceiling 270/729 instead of
  512). Width classes never helped. Measured on the audited hadpi boxes: three classes at about
  1 : 1.5 : 2 on a common pitch fit almost as well as a free table, **but the per-word residual
  is about 12 px, 0.7 of a letter, which is larger than the scorer's half-letter tolerance.**
  Typewriter in-word and word gap distributions coincide. So geometry alone cannot resolve these
  boundaries: §13's "width leads" is necessary and not sufficient.
- §4.4 robustness **fails**: with 15% of words corrupted, 56% of intact words stay correct on
  clean synthetic pages (target 90%) and 7% on hadpi; a corrupted word lowers margins across its
  whole line. Corrupted positions do end up not located (82–94%), so it fails safe.
- Unbroken text distributed over lines (D) fails on justified pages, where every line has the
  same width. **Line structure must come from the line-faithful reading (§4.2), never from
  distributing Markdown.** The `line` status must require a trusted line assignment.
- Line-text assignment with truth lines of text: 2 of 33 non-text detected lines on hadpi
  received text; 2 of 60 text lines went to the wrong detected line, both flagged untrusted.
- Two-fold check: a `located` threshold chosen on hadpi p1 does not transfer to p2 and the
  reverse. The margin is not a calibrated confidence on typewriter photos; thresholds need
  spot-check records per page type (§6.3).
- 57 free parameters; two were searched, on synthetic development seeds only. Real pages were
  scored twice (`out/word_placer/run_log.txt`).
- **Open, in order of cost:** (1) arm R, $0, not built; (2) arm V on line crops (the plan's ≤ $1),
  now a real candidate rather than a control, since Step 2 showed a stronger model reads these
  lines at 2–5% error; (3) arm C (kraken/torch, D5), the published answer, an install and an
  unverified Persian model. D5 goes back to the owner with these numbers after (1) and (2).

**Step 4 continuation: arms V and R (2026-11-12, $0.67 of the $1 cap; `tests/word_placer_v.py`,
`tests/word_placer_r.py`).** Strict scorer; "within 1 letter" is a diagnostic added for every arm
(right word and line, each horizontal edge within one letter), not the headline.

| Experiment A (audited lines + truth text) | hadpi (729): correct, precision; within 1 letter | majid (632): correct, precision; within 1 letter |
|---|---|---|
| Draft splitter | 70.8%, 70.8%; 76.0% | 97.2%, 97.2%; 97.8% |
| G | 40.3%, 87.2%; 45.7% | 64.1%, 98.3%; 64.9% |
| **R** (render and warp) | 55.0%, 55.6%; 70.9% | **95.9%, 95.9%; 97.8%** |
| **R∧G** (located only where R and G agree) | 49.0%, 85.0%; 56.5% | **91.9%, 98.1%; 93.4%** |
| V, numbered candidate cuts, Opus 5.5 (p1 / p23 only) | 30/367, precision 30/351 | 53/336, precision 53/330 |
| V, same, Sonnet 5 | 18/367, precision 18/297 | 20/336, precision 20/101 |

End to end with L1 lines (B): R 44.2% at 55.3% on hadpi; 82.1% at 95.6% on majid (89 words
unresolved on p23 through untrusted line assignment); R∧G 79.6% at 97.7% on majid. p66 (A): R
97.8%, R∧G 89.0% at 99.4%. Synthetic untouched seeds (A, the page's own face excluded from R's
candidates): clean R 97.5%, R∧G 91.7% at 99.6%; hard R 61.2%; nastaliq R 84.4% (G: 25%).

- **Arm V is dropped as a word locator.** Given one straightened line and its word list, with
  numbered candidate cuts or a pixel ruler, neither Sonnet 5 nor Opus 5.5 localises words: 5–16%
  of V-located words are correct, on clean book print as well as on hadpi. The answers divide the
  line by proportion and drift. Repeats and one-line-per-request confirm it. A model that reads
  the line at 2–5% error still cannot point at its words. V∧G and a G+V hybrid add little and the
  hybrid lowers precision.
- **Arm R solves printed pages** and replaces G as the lead arm there, with R∧G agreement as the
  `located` rule: the first configuration near the clean-page targets (≥ 95% at ≥ 98%): 91.9% at
  98.1% with truth lines; 79.6% at 97.7% end to end, where the loss is line-text assignment on
  p23, not word placement.
- **hadpi is still far off**: R 55% at 56%; R∧G 49% at 85%. No installed face resembles the
  typewriter (R picked DecoType Naskh on p1 and Tahoma on p2). No arm meets the hard-page targets
  or the 70% kill line with usable precision.
- Text errors (C5): on majid R keeps 75.8% of intact words correct but places substituted words
  confidently (0 of 13 flagged); on hadpi 39.1%. §4.4 still fails.
- Not run: V under B and C5 (V had already failed under A); Opus above effort low; sparser marks.
- **Next, at $0, before D5:** a page-trained template aligner (arm T): learn each letter form's
  appearance from the page itself by forced alignment of the known text, starting from R's
  alignment. This is the recogniser-based alignment of the literature (§13.3) without an
  external model or install. If it fails, D5 (kraken/torch) goes to the owner.

**Step 4 continuation: arm T, arm S and agreement rules (2026-11-12, $0; `tests/word_placer_t.py`,
`tests/word_placer_s.py`).** T = page-trained template aligner: letter-form templates learned
from the page itself by forced alignment of the known text (no font, no external model). S = the
audit tool's draft word splitter run inside the same lines as the other arms (owner's suggestion).
Agreement, defined once beforehand: two arms agree when both box edges along the line are within
half a letter. Strict scorer; A = audited lines + truth text, B = L1 lines + truth text.

| Arm | hadpi A: correct, precision | hadpi B | majid A | majid B | p66 A |
|---|---|---|---|---|---|
| S | 68.3%, 68.3% | 53.9%, 66.8% | 98.3%, 98.3% | 83.5%, 97.2% | 94.9%, 94.9% |
| R∧G (previous best) | 49.0%, 85.0% | 35.7%, 83.3% | 91.9%, 98.1% | 79.6%, 97.7% | 89.0%, 99.4% |
| **T** | **74.3%, 80.1%** | 63.6%, 81.3% | 96.8%, 98.4% | 82.6%, 98.1% | 96.6%, 98.6% |
| **S∧T** | **63.5%, 95.3%** | 51.9%, 94.3% | 97.8%, 99.0% | 83.2%, 98.5% | 96.3%, 98.8% |
| Majority ≥ 3 of {S, R, G, T} | 65.0%, 89.3% | 53.6%, 89.9% | 97.8%, 98.6% | 84.0%, 98.0% | 96.9%, 98.9% |
| Majority ≥ 2 | 77.1%, 82.0% | 61.7%, 81.4% | 98.3%, 98.3% | 84.2%, 98.0% | 98.6%, 98.9% |

- **No arm meets the hard-page targets (≥ 85% at ≥ 97%).** The kill line (70% with truth lines and
  text) is no longer triggered: T 74.3%, majority ≥ 2 77.1%. S∧T is the first rule near usable
  precision on hadpi (95.3%; 97.4% on p1, 92.7% on p2) while keeping 106 more words than R∧G.
- **T places hadpi words horizontally within half a letter for 88.2% of words** (precision about
  95%); S∧T is about 99.6% precise horizontally at 66.4%; majority ≥ 3 about 98.3% at 71.6%. Of
  T's 135 strict failures under A, 79 have both horizontal edges within half a letter and fail
  only on vertical extent, 26 fall short of 90% area cover, 30 are horizontally wrong. The
  implementer attributes the vertical failures to the audited boxes being taller than the ink.
  **The planner's check does not support that as stated:** sampled in each line's own frame,
  ink spans the full height of the audited box for nearly all hadpi words, and an overlay shows
  the audited boxes close to ink-tight. The likelier cause is the arms' own vertical extent
  (the detached-ink gate dropping dots and marks). To be settled by giving every arm's box the
  full ink extent of its horizontal span within the line band, with no scorer change.
- **Gate G1 (printed pages):** T, S∧T and both majority rules beat R∧G on majid in recall at equal
  or better precision; on p66 and synthetic clean pages they gain 5–7 points of recall for about
  0.5 point of precision (p66: 4 wrong boxes against 2). R∧G is not replaced on printed pages
  without the owner accepting that trade.
- **Gate G2 (extra lines):** boxes shown on a non-text detected line: 0 for every arm on every
  page. One text line per page (hadpi p1, p2, majid p23) was assigned to a non-text line, fell
  below the trust margin, and its words are unresolved.
- End to end (B), 141 hadpi words and 89 majid p23 words are unresolved because their line-text
  assignment is not trusted, for every arm. Line-text assignment is now the largest single loss.
- Text errors (C5): T on hadpi keeps 61.5% of intact words (2 points below its own B result, so
  a wrong word no longer spreads along the line) and leaves 17 of 18 corrupted positions not
  located; p66 and synthetic clean meet both §4.4 targets; majid just misses (80.4%, 75%).
- Templates learned on one hadpi page work on the other (p2's on p1: 293 against 294 correct), so
  they capture the typewriter, not the page. Pool over a book when several pages exist.
- T's own margin is not a calibrated confidence (two-fold: no threshold transfers). **Agreement
  between independent arms is the confidence for `located`.** A page-condition selector is
  proposed, not validated beyond these books: the share of words where R and T agree (hadpi 59%,
  majid 97%, p66 97%).
- Glyph-cut-out R (Rc) helps R on hadpi (61.6% against 55.0%) but stays far below T.
- Synthetic untouched seeds, hard pages: T 80.7% at 93.9% (R 61.2%); T renders no font.
- Run time: T 1.1–2.5 s per page; all arms together on a hadpi page about 12–14 s, over the 10 s
  target.
- D5 (kraken): not asked yet. A recogniser would help the about 30 horizontally wrong boxes and
  the missing confidence, not the vertical issue or line-text assignment.

**Step 4, side-edge measure and template-scored line assignment (2026-11-12/13, $0;
`tests/word_placer_la.py`, additive `word_correct_h1` in `tests/real_truth_eval.py`).** From here
on every word-placement number uses the side-edge measure (D12): right line, each horizontal edge
within half a letter of the truth edge, under half a letter taken from a neighbour, not grossly
taller than the line. **This is an instrument change. Numbers below are not comparable with the
blocks above.** The implementer was stopped by a usage limit before writing its report; the
numbers were read by the planner from `out/word_placer/real_R13_old_h1.json` (code of 246c39c,
re-scored on identical boxes) and `real_R14_new_h1.json` (new line assignment and shared vertical
rule), and the regression suites were re-run.

Bridging table (instrument change only; same boxes, audited lines + truth text, hadpi 729 words):

| Arm | Old area rule | Side-edge measure |
|---|---|---|
| T | 74.3%, precision 80.1% | 88.2%, precision 95.0% |
| S∧T | 63.5%, 95.3% | 66.4%, 99.6% |
| Majority ≥ 3 | 65.0%, 89.3% | 71.6%, 98.3% |
| S | 68.3%, 68.3% | 70.4%, 70.4% |
| R∧G | 49.0%, 85.0% | 54.9%, 95.2% |

End to end (experiment B: L1 lines + truth line text), before and after the template-scored line
assignment, side-edge measure, "located and correct, precision":

| Arm | hadpi before | **hadpi after** | majid before | **majid after** | p66 (unchanged) |
|---|---|---|---|---|---|
| T | 74.9%, 95.6% | **91.5%, 96.4%** | 83.7%, 99.4% | **97.2%, 99.5%** | 95.8%, 99.1% |
| S∧T | 54.7%, 99.5% | **66.4%, 99.6%** | 84.2%, 99.6% | **97.9%, 99.7%** | 98.3%, 99.4% |
| Majority ≥ 3 | 58.3%, 97.7% | **70.2%, 98.7%** | 85.3%, 99.4% | **99.4%, 99.7%** | 98.3%, 99.4% |
| Majority ≥ 2 | 69.7%, 91.9% | **86.7%, 93.8%** | 85.4%, 99.4% | **99.5%, 99.5%** | 98.9%, 99.2% |
| S | 56.1%, 69.6% | 67.8%, 68.7% | 84.7%, 98.5% | 98.7%, 98.7% | 99.2%, 99.2% |

- Words unresolved because their line's text was not trusted: hadpi 141 → 10; majid 89 → 0.
  Wrong-line boxes: 0 on every real page for every arm (gate G2 holds on that count).
- **Printed pages meet the clean-page targets end to end with truth line text** (≥ 95% at ≥ 98%):
  S∧T 97.9% at 99.7% on majid, 98.3% at 99.4% on p66 (which has one known truth error, D13).
- **Hard pages:** T alone is 91.5% at 96.4% (target ≥ 85% at ≥ 97%: recall met, precision 0.6
  point short). S∧T meets precision (99.6%) at 66.4%. No single rule meets both yet.
- With 5% of words corrupted (C5): hadpi T 86.6% at 95.5%, S∧T 60.4% at 99.8%; majid T 93.0% at
  98.0%, S∧T 85.3% at 99.1%.
- **Caveats.** (1) All of this uses the owner's corrected line text; production text comes from a
  model. (2) hadpi p1–p2 were looked at throughout development, including for the new line
  assignment; these are development numbers, and only held-out p3–p5 can confirm them. (3) Run
  time on a hadpi page with all arms is 13.5 s, over the 10 s target. (4) The implementer's own
  account (what the vertical measurement showed, the cause of T's over-extension on p66 line 28,
  G2 text-line counts, synthetic untouched seeds) is on disk under `out/word_placer/` but has not
  been reported or reviewed.

**Step 4: a rule that meets the hard-page targets with truth text, and the first end-to-end run
on model-read text (2026-11-13, $0; `tests/word_placer_e2e.py`, rules in `tests/word_placer_s.py`).**
Side-edge measure only.

Rule **TA**, defined before scoring: arm T's box, shown as `located` only when at least one of
S, G, R agrees with it on both side edges (within half a letter). No tuned part.

| Rule, end to end with the owner's line text (B) | hadpi | majid | p66 |
|---|---|---|---|
| T alone | 91.5%, precision 96.4% | 97.2%, 99.5% | 95.8%, 99.1% |
| **TA** | **86.0%, 97.5%** | 98.7%, 99.7% | 98.6%, 99.2% |
| S∧T (printed-page rule, D13) | 66.4%, 99.6% | 97.9%, 99.7% | 98.3%, 99.4% |

- **TA meets both hard-page targets (≥ 85% at ≥ 97%) on hadpi with truth line text**, with
  audited lines as well (85.6% at 97.3%). Per page: p1 98.0% precision, p2 96.9%. Wrong-line boxes
  0. On printed pages TA is equal to or better than S∧T in recall at about the same precision
  (p66: 351 correct with 3 wrong, against 350 with 2). Development pages; not yet confirmed on
  held-out p3–p5.
- Why agreement works: T's `located` boxes next to a word T bypassed are wrong 17 of 38 times;
  all its other located boxes are wrong 54 of 3,264 times. Agreement rules drop the first kind
  (the p66 line 28 over-extension is this class). A direct guard against it (rule TQ) costs
  recall on clean pages and is not adopted.
- Vertical extent, settled: of the 79 earlier "vertical only" failures, 78 were audited boxes
  taller than the ink and 1 was an arm box too short. This matches the owner's account (D12).

**End to end with a model's reading as the line text** (cached Step 2 responses; no API call):

| Line text from | hadpi: words read / misread / no counterpart | hadpi T | hadpi S∧T | hadpi TA | majid TA |
|---|---|---|---|---|---|
| Owner's truth | 729 / 0 / 0 | 91.5%, 96.4% | 66.4%, 99.6% | 86.0%, 97.5% | 98.7%, 99.7% |
| Opus 5.5, native bands | 658 / 9 / 62 | 82.4%, 94.1% | 60.8%, 97.1% | 76.3%, 96.0% | 96.7%, 99.7% |
| Opus 5.5, line crops | 649 / 11 / 69 | 81.6%, 95.5% | 60.2%, 99.1% | 79.0%, 97.0% | 95.9%, 99.5% |
| Sonnet 5, native bands | 535 / 62 / 132 | 60.9%, 88.6% | 46.0%, 100% | 55.4%, 96.7% | 97.6%, 99.8% |

- **Printed pages hold with model-read text**: 95–98% at ≥ 99% for every reader.
- **hadpi with Opus text: 76–79% located and correct at 96–97% precision** (TA). No rule reaches
  85% at 97% with model text. The loss against truth text is mostly not misreading: of the 62–69
  truth words with no counterpart, most are words the model joined or split differently from the
  owner's truth with the same letters (on p2, 36 of 54), so they cannot be matched one to one.
  Their boxes were drawn but not scored ("extra": T 78, TA 44). **The recall figures for model
  text are therefore a lower bound**; scoring joined or split tokens against the union of their
  truth words is still to do.
- Boxes on misread words (wrong text pointing at the right ink): Opus 5–7 per two hadpi pages,
  nearly all in the right place; Sonnet 53, of which 30 in the right place.
- Wrong-line boxes: 0 for every reader, arm and page.
- **First breach of gate G2:** with Sonnet 5's reading, 2 boxes on hadpi p2 and 2 on majid p23
  were shown on a non-text detected line, because Sonnet produced more lines than are printed
  (33 for 31; 28 for 27) and the extra lines were paired with non-text detections. No breach with
  Opus readings or truth text. **Must be fixed before any reader that can invent lines is used:**
  a text line may be paired with a detected line only if the template score confirms it.
- Not done: readings other than the first repeat; model text plus further corruption; held-out
  p3–p5.

**Step 4: joined and split words scored; stricter line assignment tested and rejected
(2026-11-13, $0; committed 2dfdf5f).** Side-edge measure. Scoring completion only; no box moved.

Words a model joins or splits differently from the truth with identical letters are matched as
groups (never across a printed or model line) and a group is correct only if all its tokens are
located and the union of their boxes passes the side-edge test against the union of its words.

| Line text from | hadpi: read / misread / joined / split / not read | T | TA | S∧T |
|---|---|---|---|---|
| Owner's truth | 729 / 0 / 0 / 0 / 0 | 91.5%, 96.4% | 86.0%, 97.5% | 66.4%, 99.6% |
| Opus 5.5, bands | 658 / 9 / 8 / 31 / 23 | 87.4%, 94.0% | 79.1%, 95.1% | 61.7%, 96.4% |
| Opus 5.5, line crops | 649 / 11 / 4 / 42 / 23 | 87.5%, 95.5% | 82.6%, 96.3% | 60.4%, 98.2% |
| Sonnet 5, bands | 535 / 62 / 14 / 14 / 104 | 64.3%, 89.0% | 56.7%, 96.3% | 46.8%, 99.4% |

- This supersedes the lower-bound figures in the previous block. With Opus text only 23 of 729
  hadpi words are genuinely not read. majid with any reader: T 96%, S∧T 97%, TA 97.5%, precision
  99.0–99.8%. Wrong-line boxes: 0 for every reader.
- Agreement is rarer on joined or split tokens (TA confirms 89–93% of one-to-one tokens T shows,
  but 57–65% of joined or split ones), because S and T put the internal boundary of a split word
  in different places. That is why TA loses more than T with model text. A line-faithful reading
  that keeps joins as typed (T4/T7) would reduce it.
- **Gate G2, corrected:** the two "boxes on a non-text line" reported with Sonnet text were both on
  real printed text: the running head on majid p23 (inside the truth's own ignore zone; now
  counted separately as "ignored"), and a typed line under the signature on hadpi p2 that is not
  in the owner's truth. **Corrected by the owner (D15): the hadpi p2 case was the stamp and the CamScanner watermark,
  not typed text, so with Sonnet 5 text two boxes were shown on non-document marks: a real G2
  breach for that reader.** None occurred with Opus text or truth text.
- **A stricter line-text assignment (`--la T2`: the given text must explain a line's ink almost
  as well as an unconstrained letter sequence does) is rejected: it fails gate G1.** With truth
  text, p66 falls from 98.3% to 95.2% (S∧T) and from 95.8% to 93.3% (T), with 12 words left
  unresolved; with 5% corrupted text hadpi T falls from 86.6% to 81.3% and majid from 93.0% to
  89.9%. The committed assignment (`--la T`) stays. The code remains behind the switch.
- **Open risk, synthetic only:** on one synthetic page type an invented or dropped text line
  shifts the following text by one line (19–24 wrong-line boxes), where a truth line has no
  detected line of its own in a stamp and signature zone. It has not occurred on a real page.
  `tests/word_placer_g2.py` is the harness for it.
- **D14/D15 follow-up (2026-11-13, $0, development pages; `out/word_placer/e2e_20261113-173510.json`,
  run log `out/word_placer/run_log.txt`).** New scoring views in `tests/word_placer_passage.py`
  (passage level: T and one of S, G, R agree on the outer edges of each line segment of a
  correction span; span lengths drawn from the 538 stored review hunks, 75% one word),
  non-document zones in `tests/word_placer_zones.py`, guards `--la TM/TZ/TZ2` in
  `tests/word_placer_la.py`, shift study `tests/word_placer_shift.py`.
  - hadpi p1–p2, located and correct at precision, word level TA then passage level:
    truth text 86.0% at 97.5% → 89.2% at 97.8%; Opus line crops 82.6% at 96.3% → 89.2% at 98.3%;
    Opus bands (`t1r_opus`, the D16 reader) 79.1% at 95.1% → 82.9% at 95.3%; Sonnet bands 56.7%
    at 96.3% → 64.8% at 88.8%. So the passage view meets 85% at 97% with truth text and with
    one of the two Opus readings; the two Opus readings differ by as much as the rules do
    (two pages, one run each: not a finding). On longer seeded spans no reader meets it
    (78–80% at 91–96%). Printed pages are unchanged (97–99% at ≥ 99%).
  - Group boxes for joined or split words (rule TAG) add 5–7 points of word coverage but are a
    median 4.5–5.8 letters wider than the word and lower precision for every reader, printed
    pages included. **Not adopted** (proposed to the owner as such).
  - D15 / G2: the only breach is Sonnet text on hadpi p2 (2 boxes on the stamp zone); Opus and
    truth text have none. Guard TM blocks nothing on real pages. Guard TZ2 (lines below the
    text block that do not look like normal text lines are not trusted) removes the breach and
    leaves truth-text and Opus results identical (R19 = R15 on all 18 cells), but it changes
    Sonnet-text results on hadpi (413/729 at 96.3% → 462/729 at 94.7%), costs 14 of 3,228
    words on fresh synthetic pages where there was nothing to block, and its band was chosen
    after a synthetic look. **Owner decision pending; not part of the frozen configuration.**
  - One-line shifts (synthetic): detectable (17% of a shifted line's tokens pass TA against
    100% on correct lines) but a line-level cut removes 2–3 correct boxes per wrong one; no
    rule adopted. Run time about 11 s per hadpi page warm; cheap savings listed in
    `out/word_placer/profile_hadpi1.txt`.
  - **Word placement cannot be scored on held-out p3–p5 yet:** those pages now have owner text
    truth but no audited word boxes.
- **Held-out pages, spot-check instead of a box audit (2026-11-14, D19, $0;
  `tests/heldout_box_spotcheck.py`).** hadpi p3–p5 have no audited word boxes, so the frozen
  configuration was run once (code at 092e091; the default path gives the same development
  results as 2dfdf5f) and a seeded random sample of the boxes rule TA would show was drawn: 40
  per page for each of two line texts (owner truth, cached Opus reading run 1), 226 boxes after
  merging identical ones, shown to the owner without saying which text they came from. Coverage
  (boxes shown over words, no truth needed): truth text 334/365, 295/415, 366/387 = 85.3% pooled;
  Opus text 328/347, 351/398, 366/381 = 92.8% pooled. Coverage is availability, not accuracy:
  precision comes from the owner's marks (`report`), and their product estimates "located and
  correct". p4 with truth text shows far fewer boxes than
  with Opus text; not investigated, because nothing may be tuned on these pages.
  **Result (owner marked all 226 on 2026-11-14, `report` run once,
  `out/word_placer/heldout_spotcheck.json`):** precision 119/120 = 99.2% for each line text (95%
  interval 95–100%; no "cannot tell"). Both wrong boxes took letters from a neighbouring word
  (one letter, two letters). Estimated located and correct (coverage × precision): **truth text
  84.5%** (p3 91.5%, p4 71.1%, p5 92.2%), **Opus text 92.0%** of the reader's words (p3 92.2%,
  p4 88.2%, p5 96.1%), which is 88.8% of the truth word count. Against the hard-page targets
  (85% at 97%): precision met on the point estimate, though 120 boxes cannot exclude 95%;
  located-and-correct met with the reader's text, half a point short with truth text, all of it
  from p4's coverage. Marking rule the owner applied: a box is judged against the printed ink the
  label names, so a label with letters missing (a reading error, counted in the reading test) was
  marked right when the box sat on that ink; a full letter taken from a neighbouring word is
  wrong. That is slightly more lenient than the development scoring for a truncated label. Three
  pages of one document; coverage is exact, precision is a sample.

- **Configuration to freeze for the held-out pages (hadpi p3–p5):** L1 lines; line-text assignment
  `--la T`; shared vertical rule; arms T with S, G, R; `located` by TA on hard pages and S∧T on
  printed pages (D13); side-edge measure; code at commit 2dfdf5f. Nothing is to be tuned on p3–p5.

### Step 5 — Correction-level paired comparison ($0)

- Population, frozen before running: every hadpi correction, the 86 `ghessehaye-majid` cases
  (cached evidence, free replay), the 111 bachehaye fresh30 cases.
- **PASS:** covered-target rate improves with Wilcoxon p < 0.05; wrong-line rate not higher; a
  visual audit of every changed box finds no wrong-occurrence placement; held-out hadpi p3–p5
  meets the hard-page targets; and the new engine does not do worse than the current one on the
  held-out half of `box_truth.jsonl` collected so far.
- Fail: keep as experimental; do not promote; record the result.
- **Scoping (2026-11-14, $0): the current engine's cached line readings cannot serve as the new
  engine's line text.** They exist only where the old reader ran: none for the audited majid
  pages p23/p109, and 23 of 31 detected lines on bachehaye p66 (agent-audited; the one audited
  page that has any). On p66 the frozen configuration with those 23 lines as line text gives
  S∧T 201/356 = 56.5% located and correct at 201/231 = 87.0% precision, against 350/356 = 98.3%
  at 350/352 = 99.4% with truth text (97 words have no text at all; the readings follow the old
  detector's lines, which do not match L1's). One page, deterministic code. So Step 5 needs a
  line-by-line reading of each comparison page: hadpi 5 pages are cached; majid (86 cases on 35
  pages) and bachehaye fresh30 (111 cases on 28 pages) are not. At the measured cost per page
  (Sonnet bands `t1r` about $0.04–0.05, Opus bands about $0.09) that is about $3 or about $6 for
  63 pages; the plan's $5 cap has about $0.06 left. Owner decision needed before any call.
- **Execution plan (2026-11-14; owner approved the readings and raised the spend cap by up to
  $3.80; `tests/step5_compare.py`).** Fixed before any new-engine box existed:
  - *Population* (`freeze`, $0, run once): every correction the review UI derives on the pages of
    `ghessehaye-majid` and the bachehaye fresh30 workspace where the current engine has at least
    one cached line reading, so that it is compared with its evidence: **202 corrections on 58
    pages** (majid 93 on 31, bachehaye 109 on 27). This replaces the plan's "86 majid cases", a
    case list of the removed locator. hadpi has no pending correction and the current engine
    shows nothing there; hadpi enters through its audited-truth and held-out results. The current
    engine's result per correction is recorded at the freeze by offline replay: 12 located, 11
    buffered, 173 estimated, 6 no box.
  - *New engine*: one Sonnet 5 band reading per page (arm `t1r`, unchanged, one run) → L1 lines
    → `--la T` → word rule S∧T (D13). A correction's Markdown span is mapped to the reading's
    words by page-wide word alignment; its box is the union of its words' boxes when all are
    located, a line-only box when only the lines are known, else nothing. A correction without a
    span is searched for by its text; an insertion takes the word on each side.
  - *Judge*: these pages have no audited boxes, so the owner marks every shown box of both
    engines, shuffled, without being told the engine: right (whole target inside, at most one
    extra word a side: the plan's §1 "correct"), wide (inside, more excess), miss, wrong line,
    cannot tell. Covered = right + wide. Because every box is seen, this is also the visual audit
    the pass rule asks for.
  - *Test*: paired per correction. With a yes/no outcome the Wilcoxon signed-rank test reduces to
    the exact sign (McNemar) test, which is what `report` prints, for covered and for correct.
    Corrections cluster on pages, so the p-value is optimistic. **PASS** as the plan states:
    covered rate higher with p < 0.05, wrong-line rate not higher, and gate G1. "Correct" is
    reported beside it and is the number that matters for the reviewer, since a box over most of
    a line also counts as covered.
  - *Smoke test ($0, `smoke`), criteria fixed first:* on the audited majid pages p23/p109 with
    their cached `t1r` readings, 100 seeded Markdown spans per page (review-like lengths) through
    the same mapping, graded by the side-edge measure. Pass needs ≥ 85% shown, ≥ 98% of shown
    correct, ≤ 15% skipped for lack of a clean truth target. **Result: PASS**, 191/198 shown
    (96.5%), 191/191 correct, 2 skipped, 7 line-only. Development pages.
  - *Paid smoke*: two pages first ($0.028 a page, lines parsed), then the rest.
  - *Run (2026-11-14):* 58 readings, $1.65 in total (`ledger_step5.jsonl`), none cut off. `build`
    ran once. New engine on the 202 corrections: 145 located, 42 line-only, 15 no box (6 have
    neither text nor a position, 5 not found or ambiguous in the reading, 3 on two majid pages
    (p26, p30) where the research engine raised an error and so shows nothing, 1 unresolved
    word). Word-level availability over the 58 pages: 15,682 of 17,455 words located (89.8%;
    availability, not accuracy). Sample: 383 boxes to mark (196 current, 187 new).
  - **Result (owner marked all 383 on 2026-11-14, `report` run once, `out/step5/result.json`):
    Step 5 FAILS; the new engine is not promoted.** Of 202 corrections (rates over those not
    marked "cannot tell"):

    | Engine | Right | Wide | Miss | Wrong line | No box | Covered | Correct |
    |---|---|---|---|---|---|---|---|
    | Current | 170 | 11 | 2 | 6 | 6 | 92.8% | 87.2% |
    | New | 161 | 24 | 0 | 0 | 15 | 92.5% | 80.5% |

    Paired, covered: new only 7, current only 5 (194 pairs, exact p 0.77). Paired, correct: new
    only 14, current only 24 (p 0.14); on bachehaye alone 2 against 12 (p 0.013), on majid 12
    against 12. Wrong line: new 0, current 6 (3%). By box status: the new engine's `located`
    boxes were right 142 times, wide once, never a miss or a wrong line (2 cannot tell); its 42
    line-only boxes were right 19 times and wide 23; the current engine's 173 `estimated` boxes
    were right 150 times. Where the current engine was right and the new one was not: 18
    line-only boxes, 5 no box, 2 others.
    The owner's standard for "right" was more forgiving than the on-screen rule, and is the one
    to use from now on: about two extra words a side is fine, an extra sentence is fine for a
    target of two or more sentences, but a whole-line box for a 2–3 word target is not (the
    reviewer cannot zoom to the place). Under it the current engine's approximate boxes are
    mostly good enough on printed pages, which the availability counts could not show.
    Reading: on printed pages the current engine already serves the reviewer; the new engine's
    word boxes are more reliable where it shows one, and its losses are the line-only fallback
    and missing boxes. Post hoc, from the same marks and not a test: taking the new engine's
    `located` box when there is one and the current engine's box otherwise would have been right
    181 times against 170, with 5 wrong-line boxes against 6. Limits: two books, corrections
    cluster on pages (p-values optimistic), one reading per page, one marker. Hard pages are not
    covered here: on hadpi the current engine had no reader evidence at all, so how it does
    there with evidence is unmeasured, while the new engine met the held-out targets with the
    Opus reading (Step 4).

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

### Decisions of 2026-11-12 (owner, after Steps 2–4)

| # | Question | Decision |
|---|---|---|
| D6 | **Extra "lines" from the Step 3 line finder** (signatures, handwriting, stamp, scanner logo below the text; 11 and 9 on hadpi p1, p2) | **Count spurious lines only inside the fitted text block** (the Step 3 criterion "≤ 1 per page" is read that way; L1 passes with 1 per page). **Condition: box accuracy must not be put at risk by the extra lines.** Enforced as gate G2 below. |
| D7 | Stronger model for hard pages | **Approved in principle:** pages that trip the review gate (confidence < 0.70 or `illegible`) are re-read by the strongest reader on native-resolution bands; clean pages stay on the cheaper model. **Not limited to Anthropic models:** for the hardest pages use the best model from any vendor, unless it costs about 3× or more than the alternative for a comparable result. Requires a cross-vendor comparison first (Step 2b). |
| D8 | Held-out test on hadpi p3–p5 | **Approved**, owner time and API spend included. The owner corrects text for p3–p5; the draft must not favour one contender (see Step 2b). |
| D9 | Image resolution reaching the model (E25) | **Must be under our control.** No image we send may be resized by a vendor without our knowing: send sizes the vendor does not shrink, and use bands or tiles when more detail is needed. |
| D10 | Putting anything into the app | **Owner approval is asked for each integration**, before production code changes. |
| D11 | Page-trained aligner (arm T) | **Retained** as part of the word-placement engine (owner, 2026-11-12, after it became the best single arm on hadpi and did not regress printed pages). Whether it replaces R∧G on printed pages is a separate, open trade (p66: more words found, 4 wrong boxes instead of 2), to be decided by the owner from a side-by-side view. |
| D13 | Word placement on printed book pages | **Switch from R∧G to the agreement rule S∧T** (a box is shown as located only where the draft splitter and the page-trained aligner agree on both side edges within half a letter). Owner, 2026-11-12, after viewing every wrong box on bachehaye_ghali p66 under the side-edge measure: R∧G 319 shown, 1 wrong; S∧T 347 shown, 2 wrong, 31 more words placed. The one extra error is a box that includes a leading dialogue dash. The shared "error" (line 7, word 12) is the agent-audited box being too short, confirmed by the owner; p66's truth should be corrected there. The aligner alone (T) is **not** the chosen rule: it over-extends on line 28 word 5, which the agreement rule does not show. Gate G1 now takes S∧T as the printed-page reference. |
| D12 | What the audited word boxes mean | **Authoritative horizontally, loose vertically.** The owner kept boxes snug to the ink's width; height was not instructed and was drawn freely, so an audited box may be taller than the word's ink. Consequence: word correctness on owner-audited pages is judged on the right line and on horizontal extent; an arm must not fail for being shorter or taller than the audited box. The planner's earlier doubt about this (Step 4 continuation block) is withdrawn: the owner's statement settles it. **Extended the same day: the side-edge measure is the headline everywhere, not only on owner-audited pages.** Owner: box height has never been what was wrong with a box in review. The §1 targets (≥ 95% clean, ≥ 85% hard; precision ≥ 98% / ≥ 97%) are read under the side-edge rule from here on. **The height-sensitive area rule is dropped from reporting altogether (owner: not wanted even as a secondary number).** Results above this point in the plan were measured under the old rule and are kept as history, labelled as such; they are not comparable with later numbers. |
| D14 | Old spelling and word joins | **Transcribe as printed; do not modernise.** Older Persian print joins many words that modern usage writes apart; keeping the original joins is correct and wanted. Consequences: (1) the reader for hard pages must be instructed and measured on keeping joins as typed (with Opus 5.5 on hadpi p1–p2, 31–42 of 729 words were split where the print joins them); (2) word placement and its scoring must not lose a box because a model split or joined a word differently: agreement and correctness are to be judged on the outer edges of the group, and of the passage a correction covers, not on a boundary inside it. |
| D15 | Marks that are not the document's text (stamp, signatures, handwriting, scanner watermark such as CamScanner) | **Not part of the page's text and never to be boxed or transcribed.** The owner confirmed that what Sonnet 5 read below the signatures on hadpi p2 was the stamp and the CamScanner mark, not typed text (the implementer's reading of it as a typed line is withdrawn). Those regions are ignore zones for scoring; a box shown there counts as a gate G2 breach. Removing such marks from the image before transcription is tracked in `TODO.md`. |
| D16 | Reading instruction for hard pages (2026-11-13) | **Keep the current Opus reading (`t1r_opus`); adopt no new instruction.** Three new wordings were tried on hadpi p1–p2 ($1.41, two runs each, `runv2`/`scorev2` in `tests/transcription_experiments.py`): "keep joins as printed" cut splits (34 and 2 → 4 and 5) but raised joins of really separate words to 17–18, almost all after a non-joining letter; the "what the typist typed" wording was as unstable as the current one; the current instruction itself swings between runs (118 and 145 strict errors of 729; splits 34 and 2). Printed controls were not shown to be no worse (Opus 3 vs 2 errors of 296 on one page; Sonnet clearly worse). No more prompt variants on two pages. Split and joined words are absorbed downstream by outer-edge scoring (D14). **Writing rule for truth and audits:** write what the typist typed, not what the ink looks like: a gap inside one word (after د ر ز و ا) is the machine and is written closed; two words printed joined stay joined; two words printed apart stay apart. |
| D17 | Hard-page reader in production (2026-11-14) | **Approved.** After the held-out reading test (Opus 5.5 on native bands 2.5% lenient word error, production call 13.2%), `transcribe` re-reads pages with no embedded-text oracle that trip the review gate with `farsi2epub/hard_reader.py` (arm `t1r_opus` unchanged). `--hard-pages opus|all|off`. The page stays flagged for review. |
| D18 | Step 4 follow-up choices (2026-11-14) | **Zone guard `--la TZ2`: not adopted** (it removes the one Sonnet stamp breach but costs precision on that reader, and is identical for truth and Opus text). **Passage-level score: reported beside the word-level score** from now on. **Group boxes (TAG): not adopted** (too wide). The frozen configuration stays `--la T`, rule TA, side-edge measure. |
| D19 | Word placement on held-out hadpi p3–p5 (2026-11-14) | **Judged by a random spot-check of shown boxes marked by the owner, not by a full box audit.** Precision from the marks, coverage from counts, their product as the estimate of "located and correct". |
| D20 | Re-read trigger for hard pages (2026-11-14) | **Book-level rule: when at least half of a book's pages without embedded text (minimum 3) trip the review gate on the first pass, the remaining ones are re-read by the hard-page reader too.** Free on normal books. A second Sonnet reading on every scanned page as a disagreement trigger is not adopted (thin margin, about $0.03 a page). |
| D21 | Book-level line lowered (2026-11-15) | **One fifth instead of one half**, after the owner read hard-reader pages of bachehaye_ghali and found them much better than the first pass. Takes in bachehaye_ghali (27%) and jafarian-factory-sale (29%); other books 0–4%. A judgement fitted to two books, not a measurement; about 3× the page cost on such books. |
| D22 | Hard-page reader (2026-11-15) | **Gemini 4 Carbon is the default hard-page reader; Opus 5.5 and every earlier option stay selectable** (`--hard-reader`, `--hard-pages`). Basis: held-out hadpi p3–p5 lenient 2.1% against 2.5% at about a quarter of the price (D7). Known and accepted: worse as printed (ZWNJ, spacing); a spacing fix was not to be tested now. Supersedes D17's choice of model, not its mechanism; D16 (no prompt tuning) holds for both readers. |
| D23 | First reader (2026-11-15) | **Gemini 4 Carbon is the default first (whole-page) reader as well, wherever it shows promise; Sonnet and Opus stay selectable.** Basis: development pages only (hadpi p1–p2 3.3–4.6% against 17–40%; clean pages level; half the price). Review gate and book rule were set on Sonnet's confidences and are not re-set for Gemini. |
| D24 | Reader preference (2026-11-15) | **Gemini 4 Carbon is preferred for transcription going forward.** Opus 5.5 stays selectable as the other strong reader; Sonnet 5 stays selectable for pages we are really sure are easy. Condition attached to the approval of the line-finder work (`docs/line-finder-integration-plan.md`). |
| D25 | Bar for changes that affect output quality (2026-11-15) | **Higher from now on.** Output quality is good, so any change that can alter the transcription is benchmarked rigorously against ground truth before it lands, with criteria fixed first, and must be shown not to hurt current quality. Use the truth we have fully; make more only when really needed. Working rules: `CLAUDE.md`, "Standing owner rules for output quality". |
| D26 | Review gate with Gemini as first reader (2026-11-15) | **Left as it is for now; no new gate is built.** With Gemini first, no page trips the 0.70 confidence gate, so nothing is flagged or re-read automatically; the owner reviews by eye and uses `--hard-pages all` on books known to be hard. Reasons: (1) the free probe of the proposed replacement, disagreement between Gemini's whole-page and band readings (`tests/gate_disagreement_probe.py`, criteria written before the run, development pages only), was inconclusive: clean pages 0 of 336 and 0 of 296 words differ, hadpi 0.8–4.1% in six run pairs and 11.6–13.8% in the two pairs that use one unstable band run, so the hard pages sit 3 words above zero at the low end; and only 58 of 108 real errors lie at differing words (bar 60%). (2) The gate's main job was to trigger the re-read, and with Gemini first a re-read is not shown to help: on hadpi p1–p2 the whole-page reading has 11–16 errors a page and the band reading 5–18, while bands are worse as printed. (3) It would cost about $0.025 on every scanned page. (4) Under D25 a gate changes which text is kept, so it needs truth-based evidence that four pages cannot give. Noted, not tested: Gemini's own confidence orders the four pages (hadpi 0.72–0.85, majid 0.98–0.99; bachehaye_ghali 0.90–1.00), so a higher Gemini-specific cutoff is the cheapest candidate if this is reopened; a held-out whole-page Gemini read of hadpi p3–p5 (about $0.10) was offered for that and declined for now. Also decided the same day: the reader stays on fixed bands (`docs/line-finder-integration-plan.md` section 8). |
| D23 held-out test | Gemini as first reader on hadpi p3–p5 (2026-11-15, $0.099, owner-approved) | **Held-out test 2026-11-15 ($0.099 Google; `tests/heldout_first_pass.py`, criteria fixed in its docstring before any reading; frozen scorer, owner truth, hadpi p3–p5, two runs):** lenient 60/2194 = 2.7% (30/1097 in each run; per page 2.0–2.3%, 1.3%, 4.6–4.9%) against Sonnet whole page 145/1097 = 13.2%, Opus bands 54/2194 = 2.5%, Gemini bands 46/2194 = 2.1%. As printed it is level with Sonnet and far behind Opus bands: strict 787/2326 = 33.8% against 393/1163 = 33.8% and 16.3%. **Verdict by the pre-set criteria: INCONCLUSIVE**, because the strict condition (no worse than Sonnet) missed by one error in 2,326 (33.83% against 33.79%); the word-identity conditions passed with a wide margin. The default stays (D23 unchanged). **Confidence did not track error on these pages:** p5, the worst page (4.6–4.9%), reported 0.92; p4, the best (1.3%), 0.78; p3 0.85–0.90. So a higher Gemini confidence cutoff is not supported as a review gate (D26 note). No page was refused. Three pages of one document, run once; nothing may be tuned on them. |
| — | One `located` rule for all page types (TA) or two (TA for hard pages, S∧T for printed, D13) | **Open; under discussion with the owner.** D13 stands until decided. |
| D5 | kraken/torch (arm C) | Still open. Not asked until arm T (page-trained aligner), the draft-splitter arms and a cross-vendor pointing test are measured. |

**Standing gates from these decisions (apply to every later step and to integration):**

- **G1, no regression on printed book pages.** Every word-placement or transcription change is
  reported on the printed-book truth (majid p23, p109; p66 separately) beside the hard pages, and
  is rejected for printed pages if it is worse there than the best printed-page configuration so
  far (word placement: R∧G, 91.9% at 98.1% with truth lines, 79.6% at 97.7% end to end;
  transcription: Sonnet 5 whole page, 0–1% lenient WER). The engine may use different arms for
  different page conditions; a hard-page improvement never replaces a better printed-page arm.
- **G2, extra lines must not cost box accuracy.** Reported for every arm on every page: text
  lines assigned to a non-text detected line, and word boxes placed on a non-text detected line.
  Both must be zero for any box shown as `located` or `line`. Measured so far (Step 4, truth line
  text): 2 of 33 non-text lines on hadpi received text and both were flagged untrusted, so no box
  was shown on them; wrong-line boxes 0 of 729 (hadpi) and 0 of 632 (majid).

**Step 2b, cross-vendor reader comparison (new, from D7 and D8).** Same scorer, cache, ledger and
spend guard as Step 2. Contenders: the strongest vision model of each major vendor, with Opus 5.5
and Sonnet 5 as references, each given images at a size it does not downscale. Tune prompts and
banding on hadpi p1–p2 only, freeze, then run the frozen configurations once on held-out p3–p5.
The p3–p5 truth draft handed to the owner is the contenders' majority reading with **every
disagreement marked**, so that no single model's errors are favoured. Also re-test word pointing
(arm V) with any model documented to return bounding boxes: arm V was closed on Anthropic models
only. Needs the owner's API keys for the other vendors.

**Deferred by the owner on 2026-11-12 ("not for now"); tracked in `TODO.md` under "Deferred by the owner". Until it runs, D7 is applied with Anthropic models only (Opus 5.5 for hard pages), and the held-out test (D8) uses a draft built from Opus 5.5 and Sonnet 5 readings with every disagreement marked.**

**Step 2b phase 1 status (2026-11-12, $0.115, Anthropic only): blocked on keys.** `.env` holds
only the Anthropic key. The shell has a placeholder `OPENAI_API_KEY` (set in `~/.zshrc`) that
OpenAI rejects (401), and it would shadow a real key added to `.env`, because `llm.load_env` does
not override variables that are already set. There is no Google key. Built and tested offline in
`tests/transcription_experiments.py`: stdlib HTTPS adapters for OpenAI and Google behind the same
cache, ledger and spend guard; key redaction; skip-without-key; each vendor's image sizing rule;
commands `plan2b`, `smoke2b`, `run2b`, `score2b`. Neither adapter has been exercised live.
- Contenders, from each vendor's own documentation read 2026-11-12 (not from a bill):
  `gpt-6-astra` ($10/$50 per MTok), `gpt-6.1-sol` (price conflicting between two OpenAI pages),
  `gemini-4-carbon` ($2/$10 until 2026-11-30, then $4/$20), `gemini-3.8-flash` ($0.75/$3.75).
  Estimated cost per hadpi page: Astra about $0.32 (about 3.6× Opus, outside the owner's ~3× rule
  unless clearly better), Carbon about $0.056 now and about $0.11 later.
- **D9 by vendor:** Anthropic: long edge ≤ 2,576 px and ≤ 4,784 patches of 28 px (verified by
  token counts). OpenAI: with `detail: "original"`, `gpt-6-astra` keeps images up to 30,000
  patches of 32 px and rejects, rather than resizes, larger ones (documented, not yet checked
  live). **Google: Gemini 3/4 give each image a fixed token budget (ultra_high 2,240) and do not
  document the pixel size that budget preserves, so D9 cannot be verified for Google from
  documentation;** it needs a resolution probe (a synthetic ladder of text sizes) once a key
  exists.
- Word pointing: only Google documents box output (`box_2d`). OpenAI's vision guide says its
  models struggle with precise spatial localisation.
- G1 check: Opus 5.5 on native bands reads the printed controls at 5/336 and 2/296 lenient
  errors, against 3 and 0 for Sonnet 5 whole-page, which stays the printed-page configuration.
- Phase 2 (held-out p3–p5) is specified but not started; the disagreement-marked draft builder
  still has to be written and tested before any p3–p5 call.

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
