# Plan: choose the box-placement engine for review, integrate it, show it

Status: **evaluation done 2026-11-15: no candidate passed, the current engine stays the default,
Gemini boxes added as an option (section 6).** Owner's instruction: evaluate the options on the truth
already collected (including Gemini's bounding boxes), choose the best without asking him,
implement it, and show it in a review page. Approvals given up front: Google ≤ $3 for this task;
Anthropic ≤ $1.00; the winner may become `review`'s default if it beats the current engine under
the criteria below, the current engine stays selectable; commits at verified milestones (D10
approval); demo on `books/bachehaye_ghali-gemini/`. Rule D25 applies: every criterion in this
file was written before the run it judges. Parent: `docs/bbox-old-fonts-plan.md` (Steps 4–6).

## 1. Candidates

| Id | Engine | Cost per page |
|---|---|---|
| A0 | Current engine (`scan.PlacementService`: `match`, then line/word-group readings; `located` / `buffered` / `estimated`) | Anthropic reader, about $0.02–0.09 |
| A1 | New engine, frozen as in Step 5: one band reading as line text → line finder L1 → line assignment `--la T` → word arms; a correction's box is the union of its words when all are `located`, a line-only box when only the lines are known, else nothing. Two `located` rules are scored: S∧T (D13) and TA | one band reading (Gemini $0.025), CPU |
| A2 | Gemini `box_2d`: one call with the page image and the list of passages, asking for a box per passage | about $0.02–0.05 |
| H* | Combinations computed offline from the three outputs, no new calls: H0 = A1 `located`, else A0. H1 = A1 `located`, else A2. H2 = A1 `located`, else A2 only when A2's box lies on a line A1 assigned to that passage, else A1's line-only box. H3 = A2 alone where A1 `located` agrees with it on both side edges, else A1. | sum of parts |

Only one `located` rule goes into the app for all page types (this settles the open D13 question
for review boxes): the rule with the higher RIGHT count in E1, provided at least 97% of its
`located` boxes are RIGHT in each page class; if neither meets that, S∧T.

## 2. Truth and instruments

**E1, span test on audited pages (exact truth, every engine, $0 judge).** Pages: hadpi p1, p2
(typewritten photos, owner-audited), ghessehaye-majid p23, p109 (clean print, owner-audited),
bachehaye_ghali p66 (old print, agent-audited, always reported separately). Never hadpi p3–p5.
Targets: 100 seeded spans per page of review-like lengths, drawn by the generator of
`step5_compare.py smoke` with a seed fixed in the script; the same spans for every engine and run.
Line text for A1: the cached Gemini band readings (`gg_carbon`; run 0, and run 1 where it
exists); for p66 one new Gemini band reading.

**Judge (automatic, the owner's standard of 2026-11-14, computed from audited word boxes; height
ignored, D12):**

- RIGHT: every box segment is on a line that holds target words; on each such line the box
  covers the target's outer edges to within half a letter, and it runs past the target by at
  most two words on each side.
- WIDE: covers the target as above but with more excess than that.
- MISS: on a right line, but part of the target is outside.
- WRONG LINE: a segment on a line with no target word.
- NO BOX.

**V0, validating the judge against the owner ($0).** On the Step 5 pages there are no audited
boxes, but the new engine's `located` words there were right 142 of 143 times. Using those words
as stand-in truth, the judge labels the Step 5 boxes whose target words are all located, and its
labels are compared with the owner's 383 marks. **PASS: agreement on right / not right ≥ 90%,
and no more than 5% of the owner's "not right" boxes labelled RIGHT.** If V0 fails, the judge's
excess rule is the suspect: report the confusion table, and E1 is then read on "covered and not
wrong line" plus box width, not on RIGHT.

**E2, confirmation on real corrections (only if the winner contains A2).** A seeded sample of 20
of the 58 Step 5 pages; A2 boxes for their corrections; judged by the V0 stand-in truth where it
exists; compared on the same corrections with the owner's marks for the current engine.

## 3. Runs

- A2 is asked twice per page (two runs). A prompt is fixed on a synthetic page (`out/synth/`,
  exact truth) before any real page is sent: first to confirm the coordinate convention, then to
  choose between at most two wordings. **No wording is chosen on the audited pages.**
- A2 smoke test on the synthetic page, criteria fixed here: with the convention read correctly,
  ≥ 60% of 40 seeded spans RIGHT and ≤ 10% WRONG LINE in the better wording. Below that, A2 is
  recorded as not viable for passages and is run once on the audited pages only to document it.
- Spend guard in code: cache, ledger, hard caps. Google: synthetic smoke ≤ $0.15, E1 ≤ $0.60,
  E2 ≤ $0.60, demo ≤ $1.20. Anthropic: QC for the demo ≤ $0.85.

## 4. Decision rule (fixed before E1 runs)

For each candidate and page class (hard = hadpi p1+p2; clean = majid p23+p109; old print = p66):
RIGHT rate over all 200 (or 100) spans, a missing box counting as not right; for A2-based
candidates the lower of the two runs.

1. **Trust floor.** A candidate is eligible only if WRONG LINE + MISS ≤ 3% of spans in every
   class (the current engine showed 3% wrong-line boxes in Step 5).
2. **Winner** = the eligible candidate with the highest mean of the three class RIGHT rates.
   Candidates within 2 points of the best are tied; a tie goes to the one with fewer paid calls
   per page, then to the one that needs no Anthropic call.
3. **Default or option.** The winner becomes `review`'s default only if:
   - it is at least as good as A0 wherever A0 can be measured in E1 (p66), and
   - on real corrections it does not lose to A0: for H0 the Step 5 marks already give 181
     against 170 right (post hoc, so E1's p66 result must point the same way); for a winner
     containing A2, E2 must show at least as many right boxes as the owner marked for A0 on
     the same corrections.
   Otherwise it is added as an option and used for the demo, and A0 stays the default.
4. If no candidate is eligible, A0 stays, the best candidate is shown in the demo as an option,
   and the result is reported as such.

## 5. Integration and demo

- The winner's code moves from `tests/` into `farsi2epub/` unchanged (output identical on stored
  inputs, as for the line finder), behind `--bbox-engine`, with the status vocabulary the review
  page already has (`located`, `line only` as a line box, no box with a reason). No amber
  proportional guess is produced by the new path (D1); where the winner falls back to A0, A0's
  own statuses are shown as they are today.
- Demo: a seeded sample of about 20 prose pages of `books/bachehaye_ghali-gemini/`; corrections
  from the existing QC pass (`farsi2epub qc --mode auto --pages …`); A0's stored line readings
  for the same PDF are copied from the other bachehaye workspaces so that A0 needs no new call;
  `review --pages … --wait-for-boxes`, opened in the browser.
- Before the demo is shown: every regression passes; the new path's boxes on the demo pages are
  checked by the orchestrator on rendered page crops (a sample of at least 30 boxes, counted and
  reported, not presented as accuracy).

## 6. Results

Run 2026-11-15 by `tests/box_options_eval.py` (sub-agent built and ran it; the orchestrator re-ran
`score` and got the same table). Outputs: `out/box_options/` (`e1_result.json`, `score.log`,
`v0_result.json`, `a2_freeze.json`, ledgers). Spend: Google $0.564 (synthetic smoke $0.058, E1
$0.507), Anthropic $0 for the evaluation and $0.61 for the demo's QC pass.

**V0: FAIL.** 290 of the 383 Step 5 boxes could be judged. Agreement with the owner on right /
not right 248/285 = 87.0% (needed 90%); 6 of the 12 boxes the owner called not right were
labelled RIGHT (needed ≤ 5%). The judge is stricter than the owner on side edges (13 boxes
short by 1–6 letters that he marked right) and on excess (13), and looser on 5 boxes several
times wider than the target that take no more than two whole words. So, as fixed in section 2,
E1 is read on covered (RIGHT + WIDE) and not wrong line, with box width beside it.

**A2 synthetic smoke: PASS.** Order `[ymin, xmin, ymax, xmax]` confirmed (the swapped order
gives 0 of 40). Both wordings 40/40 RIGHT; W1 frozen by the tie rule.

**E1** (RIGHT / WIDE / MISS / WRONG LINE / NO BOX; lower run where there are two):

| Candidate | Hard (200) | Clean (200) | Old print p66 (100) |
|---|---|---|---|
| A0 current | no evidence on disk | no evidence on disk | 92/2/5/0/1, median width 2.9× target |
| A1 S∧T | 100/97/1/0/2 | 191/6/3/0/0 | 68/8/24/0/0 |
| A1 TA | 164/29/4/1/2 | 196/1/3/0/0 | 75/1/24/0/0 |
| A2 Gemini boxes | 123/0/77/0/0 | 173/0/27/0/0 | 88/0/12/0/0 |
| H1 S∧T (A1 located, else A2) | 74.5% right | 99.0% | 76% |

- A2: 0 wrong-line boxes in 1,000, never without a box, median width 1.01–1.08× the target.
  Every failure is a side edge off by more than half a letter (about ±1 letter on the
  typewritten pages, ±0.5 on print). Up to 9 points between runs on one page.
- A1 on p66: its 24 misses are words with attached punctuation (the Gemini reading splits the
  comma off, so the box stops a letter short). On hard pages S∧T locates half and gives a line
  box for the rest; TA locates most but 8 of 182 located boxes are not right.
- A0 on p66: covers 94 of 100, boxes about three times wider than the target.

**Decision rule, applied as written.** Located rule: neither S∧T nor TA has 97% of its located
boxes right in every class (p66), so S∧T. Trust floor (WRONG LINE + MISS ≤ 3% in every class):
no candidate passes (A2 38.5% on hard pages, every A1-based candidate 24% on p66). **So by rule
4 there is no winner: the current engine stays `review`'s default,** and the best candidate is
added as an option and shown in the demo. Reading the same table on RIGHT gives the same
outcome. E2 was not run (no winner contains A2).

**Option chosen for the app and the demo: A2, Gemini boxes (`--bbox-engine gemini`).** By the
mean of class rates H1 S∧T is ahead of A2 (83% against 79%), but H1 needs a band reading as well
as the box call (about $0.05 a page, above the owner's $0.03) and the whole word placer moved
into the app, and its lead comes from clean pages where both are already at 87–99%. A2 alone is
one call a page, has never put a box on a wrong line, and can be shown on every demo page; the
current engine has stored evidence for only 4 of the 20 demo pages. This choice is a judgement
of the orchestrator, not a result of the rule.

**What is not shown.** That Gemini boxes are better than the current engine for the reviewer:
on p66, the only page where both were measured, the current engine covered more targets (94
against 88–97 in two runs, wider boxes). The judge's half-letter edge rule is stricter than the
owner's marks (V0), so A2's 12–38% "miss" overstates what he would call wrong; only his marks on
real corrections can settle it.

**Integration and demo (2026-11-16).** `farsi2epub/gemini_boxes.py`, `review --bbox-engine
gemini`; the request is arm A2 exactly (cache keys of the 12 evaluation requests unchanged after
the move; `score` output byte-identical). Regressions pass. Demo: 20 pages of
`books/bachehaye_ghali-gemini/`, 71 of 74 corrections boxed (3 have no span in the page text),
20 calls, $0.149. The orchestrator looked at 31 of the boxes on page crops: all 31 on the
target words (two a little small: one full stop, one word box thin in height). A count on one
book, not an accuracy figure. Task spend: Google $0.71 of $3, Anthropic $0.61 of $1.00.
