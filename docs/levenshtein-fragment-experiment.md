# Levenshtein fragment alignment experiment

## Decision and baseline

The owner accepts commit `bc37742d81d70ab0e7e6b55e2d14bfe7605b214c` as the practical baseline: 109/111 correction rows have a placement (12 located, 11 buffered, 86 estimated, two unresolved). Work is isolated on `codex/levenshtein-fragment-alignment`. This experiment does not change the production default.

The objective is better correspondence between text and physical geometry, particularly among the 86 estimates. Availability is already 98.2%; the two unresolved rows lack an authoritative target snippet. Manufacturing boxes for those rows is not the objective. A distance score measures text agreement, not a calibrated probability of correct placement.

## Hypothesis and inputs

The detector returns ink rectangles without recognized characters. Levenshtein distance therefore needs an independently read crop associated with each physical group. Reuse successful cached crop readings and independent line readings, including readings that failed the strict exact-match geometry check. Anonymous ink alone cannot be text-matched. Failed, incomplete, or missing crop observations remain missing evidence.

Align normalized crop characters to a contiguous substring of the corresponding line reading. Preserve logical Persian order, normalize Arabic/Persian letter forms, digits and ZWNJ boundaries, and retain character-to-word offsets. Splits and fusions are handled by concatenating tokens; do not sort letters. Permit a crop to cover part of a word or several words. Context is the full independent line, not a correction hint supplied to the reader.

## Implementation stages

1. Freeze the exact 111 `(page, correction key)` identities, queries, Markdown/sidecar hashes, baseline boxes, page maps and compatible raw observation identities. Assert that offline recomputation reproduces all baseline boxes. Keep live book caches, decisions and text unchanged.
2. Recover crop-to-rectangle associations by replaying the existing crop generator with API access disabled into an isolated experiment directory. Bind every recovered observation to its image hash and physical rectangle. Reuse evidence only; do not spend on new readings in this experiment.
3. Implement a bounded, semiglobal Levenshtein substring matcher with substitutions, insertions and deletions. Require enough characters, a bounded normalized edit distance, and separation from a competing location. Enforce consistency with RTL physical order and supported word anchors. Repeated/contradictory readings abstain. Record distance, runner-up margin, character coverage, evidence IDs and rejection reasons.
4. For unsupported words, aggregate compatible fragment evidence. Require substantial character coverage before replacing a character-proportional estimate; retain actual detector/crop rectangles and a boundary buffer. Preserve every supported word and all 23 supported correction boxes. Keep all new placements `estimated`; unresolved and rejected experiments retain the baseline box. Do not alter target resolution or the measuring instrument in this experiment.
5. Add synthetic regressions for partial words, split/fused tokens, one-character recognition errors, repeated phrases, competing geometry, missing readings, supported anchors, Persian normalization, and abstention. Verify the existing placement/review contracts.
6. Compare on the identical 111 rows: availability, supported preservation, geometry changes, evidence-backed associations, abstentions, box area/line changes, runtime and cost. Include an exact-only ablation on identical recovered evidence to isolate the effect of permitting edits. These are diagnostic measurements, not an accuracy claim. Create a side-by-side image review for every changed box, including its original query and surrounding context, with a local export of human verdicts.

## Evaluation and proposal rule

The initial population is development data. Inspect all changes for full target-word coverage, wrong-line placement, clipped partial words and excess neighboring text. Ask the owner to classify changed placements as candidate better, baseline better, equivalent, or uncertain. Keep original versus suggested wording visible; score the printed target, not a correction that does not yet exist on the page. Do not use the crop under test to generate its own answer key.

Only propose promotion if review demonstrates a useful gain without unresolved regressions, availability falls by zero, and all supported boxes remain identical. Ambiguous or insufficient evidence means keep the baseline. The independent v6 held-out gate (at least 200 audited held-out word corrections across books) remains separate and unchanged; this 111-row pilot cannot pass it. Any future production integration must version derived cache keys while retaining raw readings and the `estimated` status until separately justified.

## Deliverables

- Opt-in matcher and offline experiment runner, with synthetic regression coverage.
- Frozen input manifest and machine-readable baseline/candidate comparison under `out/levenshtein_fragment_final/`.
- Local side-by-side review artifact and documented results/limitations.
- A recommendation grounded in the comparison, or an explicit decision to retain the baseline pending review.

## Replay commands

```bash
source venv/bin/activate
# Once, into a new directory; fails if inputs are already frozen:
python tests/fragment_alignment_experiment.py --collect --out out/levenshtein_fragment_final
# Subsequent comparisons use frozen inputs and require no book or API:
python tests/fragment_alignment_experiment.py --out out/levenshtein_fragment_final
python tests/fragment_alignment_regression.py
python tests/fragment_alignment_review.py --out out/levenshtein_fragment_final
```

The collector reconstructs both current and compatible earlier crop layouts, but imports no historical locator. The observation key and rendered image hash must match. One cached image can occur at multiple page coordinates; each physical occurrence is retained separately. The two initial extraction diagnostics remain under `out/levenshtein_fragment/` and `out/levenshtein_fragment_compatible/`; use the final capture for conclusions.

The matcher defaults were fixed before image inspection: at least four normalized crop characters, edit distance at most 25% of crop length, an alternative-location margin of at least 12 percentage points, and at least 75% character correspondence for each replaced word. Deletions from the line reading do not count as observed characters. These are experimental abstention thresholds, not empirically calibrated confidence probabilities.

## Completed pilot results

**Recommendation: retain the accepted baseline.** The implementation is opt-in and not imported by the production placement service. This development pilot finds usable approximate correspondences, but does not demonstrate better placement accuracy or calibrated confidence.

The final capture reproduces all 111 original correction boxes exactly before treatment, on the 28 pages containing corrections (pages 2 and 7 have no correction rows). It freezes the same queries, Markdown/sidecar hashes, source PDF, page maps, and compatible observations. Full source text and evidence-directory hashes are unchanged after collection. The corrected duplicate-image bookkeeping produces identical inputs on this corpus; its multiple-location behavior is separately covered by a synthetic regression.

| Measurement | Accepted baseline | Exact-only control | Levenshtein candidate |
|---|---:|---:|---:|
| Correction population | 111 | 111 | 111 |
| Boxes available | 109 | 109 | 109 |
| Located / buffered | 12 / 11 | 12 / 11 | 12 / 11 |
| Estimated | 86 | 86 | 86 |
| Unresolved | 2 | 2 | 2 |
| Correction boxes changed from baseline | — | 0 | 4 |
| Original supported boxes preserved exactly | 23 | 23 | 23 |

There are 1,168 recovered physical crop observations. The matcher accepts 827 correspondences: 723 exact, 97 at edit distance one, and seven at distance two. It rejects 308 short/unreadable strings, 16 distant matches, 14 ambiguous occurrences, and three geometry conflicts. Accepted correspondence does not imply an accurate word rectangle. Forty-two unsupported word geometries change out of 5,464 page-map records, but only four correction boxes meet the complete-target evidence requirement for replacement. Sixty-two estimates retain their baseline because target identity/insertion handling is outside this experiment; another 20 lack sufficient fragment coverage. Thus this is a narrow test of fragment-to-line geometry, not a new Markdown-to-page locator.

All four changes come from allowing nonzero edit distance: the exact-only control changes no correction boxes. They remain on the same detected lines. Total box area increases by 1.51%–21.34%; this is a width/area diagnostic, not a target-coverage score. The complete experiment spends **$0**; frozen comparison replay takes about 16 seconds on this machine, while recovering raw crop associations takes several minutes.

### Primary visual inspection, not an independent audit

All four baseline/candidate image pairs were inspected. The owner verdicts remain pending and unfilled.

| Page / correction | Area change | Initial observation |
|---|---:|---|
| 10 / h100000 | +8.36% | Baseline already appears to contain the requested phrase. Candidate extends into more neighboring text; no clear target-coverage gain. |
| 25 / h100001 | +14.91% | Candidate adds punctuation/boundary padding and some preceding text. No demonstrated gain in full lexical-word coverage. |
| 26 / h0 | +21.34% | Wider neighboring-text inclusion; no clear improvement to the requested phrase. |
| 26 / h2 | +1.51% | The first segment shifts right and appears to clip the left edge of «آدم». Treat as a possible regression, despite slightly greater total area. |

The side-by-side comparison keeps original text and context visible, offers an unmarked-image toggle, and allows owner verdicts/notes to be exported or discussed with Codex. The viewer was checked at desktop, 736px and 360px widths; navigation, box toggling and verdict state were exercised. Test verdicts were cleared. The viewer is not an audited word-rectangle answer key, and these observations are not a passing v6 result.

### Artifacts and checks

- `out/levenshtein_fragment_final/inputs.json`: frozen baseline, page maps, crop-to-image bindings and source hashes; SHA-256 `4d6d42f563932ebf65b0e88c7d6e30b5b95b7b31164b63936387cbb5ddd1b827`.
- `out/levenshtein_fragment_final/comparison.json`: all 111 paired outputs, exact-only ablation, implementation hash and fixed thresholds.
- `out/levenshtein_fragment_final/alignment_diagnostics.json`: per-crop character positions, distance, margin and rejection reasons.
- `tests/fragment_alignment_review.py` and `tests/data/fragment_review.html`: reproducible local review; verdict exports bind both the input and comparison hashes.
- Passing standalone regressions: `fragment_alignment_regression.py`, `geometry_best_attempt_regression.py`, `consolidated_bbox_regression.py`, and `review_pages_regression.py`. The new suite includes a brute-force distance oracle, normalization, partial/fused words, ambiguity, RTL conflicts, supported-anchor preservation, missing evidence, exact-only ablation and duplicate image identities.

No production cache versions, review defaults, historical instruments, text, or human decisions were changed. Do not merge this as a placement improvement based on the counts above. The next useful experiment would use fragment matches as constraints on boundary allocation, rather than assigning a whole crop to every partly matched word. It should first address the page-26 boundary issue and then repeat the same frozen comparison, with owner/independent review before any promotion.
