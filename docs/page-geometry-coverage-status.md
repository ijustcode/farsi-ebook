> Superseded for live placement policy by [the coverage-first update](geometry-best-attempt.md). The results below record the earlier strict-evidence experiment.

# Page geometry repair status — 2026-09-07

The implementation is partial. The 30-page acceptance target is **not met** and no placement accuracy improvement or promotion is claimed.

## Implemented

- `transcribe` automatically resumes page geometry for requested, successfully transcribed pages before optional QC, including already-transcribed pages. `--bbox-mode`, `--bbox-model`, and `--bbox-max-cost` use a separate default $5 service budget. Stage costs have separate labels.
- `geometry <slug> --pages ...` backfills without retranscription or QC. `PlacementService.ensure_page_map` is correction-independent and also serves review. Clean embedded word geometry avoids image acquisition.
- Version-5 derived maps retain compatible raw observations and verified version-4 words. Maps persist line readings, normalized words, stable identities, evidence references, missing geometry, revision, completeness and regional diagnostics. Query caches bind the map revision. Nonempty partial maps are no longer complete.
- Acquisition reads every detected line, coalesces nearby ink fragments and reads paired physical groups before individual retry regions, pads group crops to include detached marks within neighboring-line boundaries, and retains bounded high-resolution and adjacent-group retries; overlapping contextual reads recover a missing detected line only when independently read neighboring lines agree exactly. Independent crop text must support geometry; no character-width coordinates or historical fallback was introduced.
- Successful raw batches and incremental line maps survive later failures, budget exhaustion, and cancellation. Conflicting new line text cannot discard already verified identities.
- A conservative exact-target fallback uses unique flanking Markdown/page alignment blocks with split/fused normalization. Incomplete-map uniqueness alone does not certify an occurrence. Repeated finding-only snippets no longer inherit the first Markdown occurrence.
- Correction chips expose unresolved reasons; page summaries expose reading/geometry completeness; wait completion prints correction statuses and map states.

## Frozen review result

Workspace: `bachehaye-ghali-fresh30-20260906-123256`, PDF pages 1–30.

The UI population is **111 corrections**. The benchmark exporter yields 109 hunks because two issue-only rows are outside its population; its counts are not substituted for the UI denominator. Baseline text, sidecars, decisions, evidence, full queries and boxes were preserved under `out/page_geometry_repair/baseline/`. After-capture verifies Markdown and sidecar SHA-256 identities and the exact page/correction-key population.

| Measurement | Result |
|---|---:|
| Supported before | 22/111 (19.8%) |
| Supported after bounded acquisition | 23/111 (20.7%) |
| Located / buffered / unresolved | 12 / 11 / 88 |
| Changed boxes / lost supported boxes | 1 / 0 |
| Complete text maps | 0 |
| Partial / acquisition-blocked / blank or nontext maps at capture | 26 / 2 / 2 |
| Read / detected lines | 514 / 722 |
| Verified / recognized words | 634 / 5,255 |

The 5,255 recognized words exclude explicit missing-line placeholders; this is not directly comparable to the plan's 5,404 total map entries. Availability and map support counts are not independently audited accuracy.

Unresolved results: 58 absent or ambiguously aligned targets, 21 missing word geometry, six finding-only targets blocked by missing reading, and three excessive buffers. Every correction's query, reason, and box is in `out/page_geometry_repair/after.json`. All 30 maps were backfilled offline; transcription, QC and human decisions were preserved.

The unmarked page-13 image was inspected before the candidate acquisition attempt. The printed `«آقا» گفت:` is on the short line below the speech ending `اینقدر ظلم نکنه`. It still lacks verified word geometry. The sole new placement, page 13 h0 (`حرفته`), was checked against its previously recorded unmarked-image location and an exact box crop: the target is fully present, with one adjacent word (`بزن`) buffered. No wrong occurrence or wrong line was found in that new placement. The unchanged baseline is not independently audited. Audit notes and crop are in `out/page_geometry_repair/image_audit.json` and `audit-13-h0.png`.

## Spending and blocking condition

The resumable repair ledger is `out/page_geometry_repair/ledger.json`; acquisition receipts are linked by run ID. Total repair limit: $5. The initial exposure cap was $1. Confirmed additional spend: **$0.505434**; conservative uncertain reservations: **$0.392744**; total accounted exposure: **$0.898178**. Further requests could not fit their pre-dispatch reservations under the initial cap. Offline backfill spent $0.

The first network retry was rejected by automatic approval review. The user subsequently said “Proceed”; the authorized network retry succeeded. Page 13 improved from 25/31 to 31/31 read lines and from 1 to 167 verified words. Page 4 increased from 11 to 24 verified words. Neither completed. The initial $1 stage therefore cannot supply the plan's required completed-page cost measurement before using the remainder. **No remaining-stage acquisition was launched.** The remaining total authorization is $4.101822, subject to that checkpoint.

No full text page completed acquisition, so observed cost per completed page and projected full-book cost are **unavailable**. A partial-page cost is not a defensible full-map estimate.

## Remaining work

- Resolve the initial cost checkpoint: no text page completed within the initial reservation cap. Further acquisition requires a decision on proceeding without the specified completed-page cost measurement; keep all prior spend and uncertainty in the ledger.
- Extend the guarded contextual recovery to detector omissions, split/merged lines and partially readable regions. The implemented recovery requires agreeing neighboring readings and a single missing detected line; other unreadable regions remain barriers.
- Calibrate the physical group proposals and ordered alignment against real independent images; the added logic currently has synthetic regression coverage and only the bounded page-13/page-4 live evidence in this repair.
- Complete maps within the ledger cap, audit every new or changed placement from unmarked images, establish full-map cost if a page completes, and rerun the exact 111-correction comparison. Stop at the authorized cap and report any shortfall.
- Reach at least 100 supported placements with no accepted wrong-occurrence/wrong-line placements and no loss of independently correct prior coverage. The separate multi-book promotion gate remains outstanding.

The final review was recreated for pages 1–30 in offline, wait-for-boxes mode at `http://127.0.0.1:8766/`. A discovered detached-startup timing issue was also repaired: the server now publishes readiness before constructing the acquisition queue.

## Validation

Passed: page-geometry regression (query-free acquisition, concurrent deduplication, budget blocking, revision invalidation, offline replay, later failure preserving earlier geometry, old-workspace CLI scope, and damaged-context guards); consolidated placement; review page scope/wait; seven benchmark regressions; 29 v6 metric regressions; historical locator/cache/judge checks; and the exact v5 word-miss reproduction (unchanged serialized boxes and metrics). These checks do not substitute for the missing real-image acceptance audit.
