# Plan: line finder into production, then decide how to cut pages for the reader

Status: **Phases 1 and 2 approved by the owner 2026-11-15 (answers in section 7); nothing built yet. Phases 3 and 4 each need their own approval.** Planner: Fable 5.2. An implementer
(Opus or Sonnet) builds it after the owner approves. Parent plan: `docs/bbox-old-fonts-plan.md`
(Step 3 result, Step 6, decisions D9, D10, D15, D16, D22, gates G1/G2).

## 1. Goal and what is out of scope

**Goal.** Make the Step 3 line finder (`tests/line_finder.py`, arm L1) production code, and use it
to answer one question with measurements: should the hard-page reader get the page as fixed
overlapping bands (today), as bands cut between detected lines, or as single line crops. The
review-gate decision (whole-page reading against band reading) is postponed until that answer
exists (owner, 2026-11-15).

**Out of scope, each needs its own decision:**

- Replacing `locate._scan_page_lines` in the box-placement engine (`scan.py`). The detector name
  is part of every page-map key, so a swap discards the paid geometry evidence of every book, and
  Step 5 showed the current engine already serves printed pages.
- The word placer (Step 4 arms). Step 5 failed; nothing of it moves.
- The review gate itself, the review UI, removing stamps from images (TODO.md).
- Any change to the reading instruction (D16).

## 2. What is known and what is not

- L1 finds 60 of 60 truth lines on hadpi p1–p2 with no merges (current detector 25, 10 merges),
  54 of 54 on majid, 32 of 32 on p66, 98.8% on untouched synthetic seeds.
- It also finds about 10 extra "lines" a page on hadpi (signatures, stamp, scanner logo), and its
  line ends are not trustworthy as positions.
- It has been scored on five real pages, all looked at during development. **How it behaves on
  whole books is not known.** Phase 2 measures that before anything depends on it.
- Line crops against bands was measured once, with Opus on hadpi p1–p2: no visible difference.
  With Gemini it has not been measured.
- Gemini on fixed bands is already at 7–18 lenient errors of about 340 words a page on hadpi
  p1–p2 (38 of 1,372 pooled over two runs), with a spread between runs as large as that. **So
  this plan can show that a new cut does harm; it probably cannot show a gain in word accuracy on
  two pages.** A gain, if any, is more likely in structure (see 3.3).

## 3. Design

### 3.1 Phase 1: move the module ($0, no behaviour change)

- `tests/line_finder.py` becomes `farsi2epub/line_finder.py`. The algorithm, `PARAMS` and
  `VERSION` ("line_finder/L1-1") do not change by one character; only the module docstring's
  "evaluation code only" sentence does.
- `tests/line_finder.py` stays as a two-line re-export, so the eleven test scripts that import it
  keep working unchanged.
- No production module calls it yet. Dependencies are already there (numpy, PyMuPDF).
- `tests/transcription_experiments.py` records a SHA-256 of the file `tests/line_finder.py` as
  provenance. The implementer must point that at the new file and confirm it is not part of any
  cache key (`score` and `score2b` must print the same numbers before and after, at $0).

### 3.2 Phase 2: cut rows from detected lines ($0, not wired into `transcribe`)

New function in `farsi2epub/hard_reader.py`, beside `band_rows`:
`line_rows(gray, lines_per_band, overlap_lines=1) -> (rows, info)`.

Rules, chosen so that a line-finder mistake can never hide text from the reader:

1. **Bands still cover the whole image, top to bottom, with no gap.** The line finder only
   chooses where the cuts fall. Ink the finder missed or called "not text" is still inside a
   band, exactly as today.
2. **A cut falls in the blank gap between two consecutive text lines**, taken across the full
   width (lowest point of the upper line's band, highest point of the lower line's band). Where
   slope leaves no horizontal gap, the cut moves to the next gap that exists.
3. **Each band holds `lines_per_band` lines plus `overlap_lines` whole neighbouring lines above
   and below.** The existing stitcher removes the repeats, as it does now.
4. **No band may exceed the size the vendor leaves unresized** (the existing `band_rows` limit,
   D9). If a band would, it is split at the nearest gap.
5. **Fallback to today's fixed bands** when the finder raises, finds fewer than 3 text lines,
   or cannot satisfy rules 2–4. `info` records `cut: lines|fixed`, the reason, the finder's
   version, the number of lines and bands.
6. Only lines of kind "text" place cuts. Extra lines on stamps and signatures change nothing
   except that a cut is never placed through them when avoidable.

`lines_per_band` values to prepare: 1 (single lines), 3 (the owner's idea), and "as many as give
the same number of bands as the fixed cut" (isolates the effect of where the cuts fall, at equal
cost).

### 3.3 A by-product worth measuring

With line-cut bands we know how many printed lines each band holds. The reader returning a
different number of lines for a band is a free signal that something went wrong there. Phase 3
records that count per band. Whether it is useful as part of a review gate is a question for the
postponed decision, not a claim of this plan.

### 3.4 Phase 3: paid comparison (needs the owner's approval for the spend)

New arms in `tests/transcription_experiments.py`, same instruction, schema, model settings, cache,
ledger and spend guard as arm `gg_carbon`; only the cut differs:

| Arm | Cut | Estimated cost a page (planner's estimate, not measured) |
|---|---|---|
| `gg_carbon` (baseline, cached, $0) | fixed bands | $0.025 |
| `gg_lsame` | line-cut, same band count | about $0.025 |
| `gg_l3` | line-cut, 3 lines a band | about $0.05–0.07 |
| `gg_l1` | single lines, one call | about $0.12–0.18 |

Gemini charges a fixed amount per image, so cost grows with the number of bands. The `plan2b`
command prints the exact worst-case reservation before any call; that number, not this table, is
what the owner approves.

Pages: hadpi p1–p2 and majid p23, p109 (development pages only). Two runs per arm and page.
Estimated total: `gg_lsame` + `gg_l3` about $0.60–0.75; adding `gg_l1` about $1.00–1.40 more.
Recommendation: run `gg_lsame` and `gg_l3`; run `gg_l1` only if the owner wants single lines
tested despite the cost, because at that price it would have to be clearly better to be used.
Page images go to Google.

### 3.5 Phase 4: held-out check and production switch (separate approval, D10)

Only if an arm passes 4.3. The winning arm, frozen, runs once on held-out hadpi p3–p5 (two runs,
`tests/heldout_vendor.py`, frozen scorer). If it passes 4.4: `--hard-cut fixed|lines` on
`transcribe`, the default moved to `lines` for the Gemini reader, `fixed` kept selectable;
`tests/hard_reader_regression.py` holds the production request equal to the new arm; the sidecar
records `hard_reader.cut`. The Opus reader stays on fixed bands (arm `t1r_opus`, unmeasured on
the new cut, $1.15 left on that account). Then the review-gate decision is reopened.

## 4. Smoke tests. Criteria are fixed here, before any run

### 4.1 Phase 1 (move): PASS only if all hold

- For every page of the five truth pages and every synthetic seed in `out/synth/` manifests,
  `to_json(find_lines(page))` minus the `seconds` field is byte-identical before and after the
  move (hashes written to `out/line_finder/move_check.json`).
- `tests/line_finder_regression.py`, `word_placer_regression.py`, `real_truth_eval_regression.py`,
  `text_audit_regression.py`, `transcription_experiments_regression.py`,
  `hard_reader_regression.py` pass.
- `tests/line_finder_eval.py real` reproduces `out/line_finder/real.json` (60/60, 54/54, 32/32).
- `transcription_experiments.py score` and `score2b` print the same numbers as before, with no
  network call.

FAIL on any difference: revert the move.

### 4.2 Phase 2 (cuts): new `tests/line_rows_smoke.py`, offline, $0

On the five truth pages, with audited line outlines as truth, for each `lines_per_band` value:

| Check | PASS |
|---|---|
| Bands cover every image row, in order | 100% of pages |
| Truth lines whose ink is crossed by a cut and that are not whole inside a neighbouring band | 0 |
| Every truth line lies whole inside at least one band | 100% of lines |
| Bands over the vendor's no-resize size | 0 |
| Same input twice gives the same rows | yes |

For comparison the same table is printed for today's fixed bands (expected: some lines cut,
all whole in a neighbour).

Whole-book robustness, same script, on every non-blank page of `books/bachehaye_ghali/`,
`books/ghessehaye-majid/` and one other printed book on disk, plus hadpi p1–p2 (**never p3–p5**):

| Check | PASS |
|---|---|
| Exceptions | 0 |
| Covering and size invariants | 100% of pages |
| Pages that fall back to fixed bands | reported per book; **FAIL above 10% of non-blank prose pages in any book** |
| Line-finder time per page | reported; **FAIL above 10 s median** on the band source image |

If the fallback or time criterion fails, stop and report: the finder is not ready for whole
books and Phase 3 is not proposed.

### 4.3 Phase 3 (paid arms): an arm is "no worse" only if all hold, pooled over its two runs

- hadpi p1+p2 lenient errors ≤ 38 of 1,372 (the fixed-band baseline pooled over its two runs).
- majid p23+p109 lenient errors, pooled over two runs, ≤ baseline pooled + 2 (gate G1; the
  baseline's second run is taken from the cache or run at about $0.05).
- No truth line missing from the stitched reading in any run (a dropped or doubled line counts).
- Measured cost a page ≤ 2× the fixed-band cost, unless the arm is "better" as defined next.

"**Better**" is claimed only if hadpi lenient errors are at most half the baseline in **both**
runs separately. Anything between is reported as "no difference shown".

Outcome table:

| Result | Action |
|---|---|
| No arm is "no worse" | Keep fixed bands. The line finder stays in production code, unused by the reader. |
| An arm is "no worse", none "better" | Owner chooses: keep fixed bands (nothing gained in accuracy), or switch for the per-band line count (3.3). Planner's default: keep fixed bands unless 3.3 shows value. |
| An arm is "better" | Go to Phase 4 with that arm. |

### 4.4 Phase 4 (held-out): PASS only if

Pooled over two runs on hadpi p3–p5: lenient errors ≤ 55 of 2,194 (2.5%, the level of both
current readers) and no missing printed line. One test, three pages of one document, nothing
tuned on them; a fail means fixed bands stay the default.

## 5. Order of work and approvals

1. Owner approves this plan. That covers Phases 1 and 2 only: $0, no change to what `transcribe`
   produces.
2. Implementer does Phase 1, runs 4.1; then Phase 2, runs 4.2; reports the tables.
3. Owner sees the `plan2b` reservation and approves or declines the Phase 3 spend, and says
   whether single lines (`gg_l1`) are included.
4. Phase 3 result against 4.3. Owner decides on Phase 4.
5. After Phase 4, or after a decision to keep fixed bands: reopen the review-gate question.

Docs to update when a phase lands: `CLAUDE.md` and `AGENTS.md` (line finder section, tests list,
commands), the parent plan (a dated result block), `HANDOFF.md`. Commits only when the owner asks.

## 6. Risks

| Risk | Response |
|---|---|
| The finder fails on page types it has never seen (verse, tables, title pages, nastaliq) | Rule 1 and the fallback keep all ink in front of the reader; 4.2 counts fallbacks on whole books first |
| Two development pages cannot show a real gain | Stated in section 2; "better" needs both runs; held-out check before any switch |
| More bands cost more with Gemini, and its price doubles on 2026-12-01 | Cost criterion in 4.3; equal-band-count arm isolates the effect of cut placement |
| Readers end up on different cuts (Gemini line-cut, Opus fixed) | Recorded per page in the sidecar; Opus is moved only after its own measurement |
| Moving the file disturbs paid caches | 4.1 requires identical `score` output at $0 |

## 7. Owner's answers (2026-11-15)

1. **Phases 1 and 2 approved**, on the condition of D24: Gemini 4 Carbon is the preferred
   reader going forward; Opus stays as the other strong option and Sonnet for pages known to be
   easy. Nothing in this work may remove or break either.
2. **Box engine's line detector: left to the planner. Decision: it stays out of this work.** A
   swap would discard every book's paid geometry evidence, and Step 5 showed the current engine
   serves printed pages. It can be reopened with its own measured plan.
3. **Phase 3 arms: `gg_lsame` and `gg_l3` only.** Single line crops (`gg_l1`) are not run. The
   spend itself is still to be approved when the reservation is known.

D25 (higher bar for output-changing work) applies to Phases 3 and 4 as written: criteria in
section 4 are fixed and are not to be loosened after seeing results.

## 8. Result of Phases 1 and 2 (2026-11-15, $0; built by an Opus sub-agent, re-run by the orchestrator)

**Phase 1: PASS on every 4.1 criterion.** `farsi2epub/line_finder.py` differs from the committed
`tests/line_finder.py` in one docstring line. Output identical on 100 pages (5 truth pages, 20
manifest synthetic pages, 75 generated ones; the `final` synthetic seeds were not hashed).
Eight regressions pass; `line_finder_eval real` reproduces 60/60, 54/54, 32/32; `score` and
`score2b` outputs unchanged. The provenance hash is not part of any cache key. Left to fix:
`tests/word_placer_eval.py` `code_fingerprint()` still hashes the shim file.

**Phase 2: 44 PASS, 3 FAIL. By 4.2 the work stops here and Phase 3 is not proposed as written.**

| Book | Non-blank (prose) pages | Exceptions | Invariants | Fallback to fixed bands | Finder s/page, median / max |
|---|---|---|---|---|---|
| bachehaye_ghali | 89 (80) | 0 | 100% | 1 = 1.2% | 0.60 / 11.19 |
| ghessehaye-majid | 189 (186) | 0 | 100% | 0 | 0.60 / 1.30 |
| jameeshenasi_khodemani | 161 (138) | 0 | 100% | 13 prose = 9.4% | 0.58 / 6.24 |
| hadpi p1–p2 | 2 (2) | 0 | 100% | 1 (p1) = 50%, **FAIL** | 1.82 / 1.94 |

On the five truth pages no truth line is lost by any cut setting (147 of 147 whole inside a band).

**Cause of the failure (orchestrator's diagnosis, development pages).** Both hadpi photos are
tilted by about 3 degrees. A full-width line then spans 1.4–1.8 line spacings vertically, so
neighbouring lines overlap in height and no horizontal cut passes between them: p1 has 10 usable
gaps for 41 detected lines, p2 has 17 for 40 (p2 passed only because 17 happened to be enough).
So rule 2 (horizontal cuts between lines) does not work on tilted photographs, which are the
pages this work is for. It works on flat scans: 1 and 0 fallbacks on the two printed books.
jameeshenasi is just under the bar (9.4%), mostly short pages where one line plus neighbours
exceeds the size limit.

**$0 probe, not a result:** rotating the image by the finder's own measured angle before cutting
gives 30 gaps of 39 on p1 and 27 of 39 on p2, and line cuts succeed for all three settings on
both. Rotation resamples the pixels the reader sees, so under D25 it is a new arm that has to be
measured against truth, not a fix to assume.

**Clarification after the owner's question.** The line finder does not assume horizontal lines:
it follows each line along its own slope and curve, and its outlines run parallel to the text.
What is horizontal is the cut: a band sent to the reader is an upright rectangle of the photo,
and on a tilted page no upright rectangle's edge passes between two lines. A crop that follows
the line exists (the finder's per-line straightening), but it resamples the pixels, so it is a
new reader input to be measured (D25), like rotation.

**Owner's decision, 2026-11-15: keep fixed bands for now.** They are the reader's current code
path and nothing changes in it. Phases 3 and 4 are not run. Phase 1 stands; `line_rows` stays in
`hard_reader.py`, unused by `transcribe`. Line-following crops or straightening can be reopened
later with their own measured plan. The review-gate question is open again, on fixed bands.
