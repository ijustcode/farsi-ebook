# Consolidated box placement — experimental implementation

The working tree implements `match → scan → unresolved`. It has **not** passed the promotion gate. Independent page/word annotations, a frozen audited held-out set, and a valid paired accuracy comparison are still outstanding. No accuracy gain is claimed.

For the subsequent correction-independent page-map repair and its incomplete acceptance status, see [page-geometry-coverage-status.md](page-geometry-coverage-status.md).

## Runtime and contracts

`farsi2epub.placement` exposes `PlacementService`, `PlacementResult`, `PrintedWord`, and `place`. Review uses the same service. Queries contain the full original/suggested span, exact Markdown boundaries, and correction kind. There is no five-word cap or QC-coordinate input. Successful outputs contain `source: match|scan`, ordered normalized line segments, a union envelope, evidence references, status, and buffer counts. Unresolved/pending outputs have no accepted rectangle. Anchored insertions without printed target words use a distinct insertion-boundary marker.

The embedded path matches normalized words and surrounding content. It transforms unrotated PDF word coordinates into displayed-page coordinates. Ambiguous repeated occurrences defer to scan. Scan reads explicit line crops before matching corrections. It independently reads physical word groups and reconciles their concatenated text against the corresponding line. Equal counts or character-width estimates cannot certify geometry. Fused groups can cause a buffer of up to two adjacent words at each end; larger unsupported coverage is unresolved.

Successful observations are stored under `books/<slug>/locate_evidence/readings/`, with image, reader prompt/schema, model and renderer identity. Derived page maps and boxes bind the PDF, derivation version and, for query outputs, Markdown/full query. Atomic writes and file locks deduplicate raw acquisitions across processes. Invalid JSON or malformed geometry is a cache miss. Source replacement invalidates in-memory identity; a source changed during acquisition cannot receive that result.

The service owns four acquisition workers, per-page work deduplication, pre-dispatch cost reservations, bounded high-resolution retries, cancellation and offline replay. Cost receipts live in `locate_evidence/runs/` with a latest-run summary. Known credit/access failures stop new acquisitions for the session while preserving offline replay. Transport failures retain an uncertain reservation; explicit pre-inference HTTP rejections consume zero actual spend. Runtime limits are per service/review session, not shared with transcription or QC budgets.

`/boxes/<page>` polls pending queries even when none has a seed box. An edit can queue new work. Chips describe evidence support and expose status/source/evidence/buffer information; they are not independent accuracy scores. Existing sidecar `bbox` values are ignored without rewriting sidecars. New QC prompts, schemas, and persistence omit coordinates.

## CLI migration

```bash
source venv/bin/activate
farsi2epub review my-book --bbox-mode auto --bbox-max-cost 5
farsi2epub review my-book --pages 61,128 --bbox-mode auto --bbox-max-cost 1
farsi2epub review my-book --bbox-mode offline
farsi2epub review my-book --bbox-model claude-sonnet-5
```

`--bbox-refine-model` aliases `--bbox-model`; `--bbox-refine`/`--no-bbox-refine` alias auto/offline, with migration messages. Conflicting aliases are errors. Explicit `--bbox-refine-algorithm` requests fail with an offline-tooling migration message. Detached server arguments use the same new options. Old strip caches use incompatible reading prompts/schema and are not silently certified as the new engine's evidence.

`--pages` selects PDF page numbers for both the UI and background evidence work. It surfaces only those transcribed pages, bypasses risk selection, and does not mark other pages skipped. Stop an existing review process before changing its scope.

## Measurement artifacts and remaining work

`tests/bbox_benchmark.py` exports cases from book/page/correction identity without running a locator. The current export in `out/consolidated_bbox/benchmark/` contains 698 corrections on 227 pages from six books: 399 development cases and 299 held-out cases on 97 held-out pages. Initial image-only drafts now exist for development pages bachehaye_ghali 66 (final independently audited 356 words plus two nonlexical dialogue marks) and Hossein_shenasi 1 (92 initial tokens). The digital draft received an independent all-record audit; two title word boundaries were repaired and rechecked. Final approval remains pending on photograph lettering scope and a consistent nonlexical-marker convention. The owner confirmed «یه» in the scan-page last-line excerpt. All 32 scan-page lines are independently audited and imported. The scan-page pilot's ten full-passage target assignments were separately audited from the unmarked image, frozen word sequence and correction context before either box treatment was opened. **299 selected held-out cases is not 299 evaluable audited cases; the held-out audited count remains zero.** The split also needs a category audit for blur, touching lines, footnotes, poetry, repetition and long passages before freezing.

`tests/bbox_annotation_workflow.py` now exports a complete blinded queue of all 227 page identities, seals annotation drafts, requires a distinct auditor and exact draft/image hash bindings, tracks explicit ambiguity/discrepancy resolutions, and merges approved page geometry while keeping correction targets pending. Seven workflow regressions pass. Original drafts and audit corrections remain separately preserved under `out/consolidated_bbox/annotations/` and `annotation_workflow/`.

The annotation editor uses unmarked images and explicit reading-order word rectangles. Import/freeze validates identity, source PDF/Markdown/image hashes, reading order, insertion anchors, audit metadata, and held-out requirements. It refuses pending or malformed truth. Production never imports this tooling or its annotations.

`tests/bbox_metrics.py` is the separate v6 instrument: at least 90% of each target-word rectangle must be covered; partial coverage remains visible; excess sums fractional coverage of non-target words; overlapping segments count once. It reports full/tight/buffered success, unresolved, wrong-line, word-miss distributions, per-book results, insertion boundaries separately, and judge uncertainty. Comparison binds instrument/image/truth/target/population identities, uses paired page-cluster statistics, and gates coverage improvement plus non-increasing jointly-covered excess, unresolved and wrong-line rates. The promotion gate requires at least 200 audited held-out word corrections across books. Synthetic gate tests do not constitute a production pass.

### Scan-page development pilot

The bounded bachehaye_ghali page-66 pilot is replayable with no API key or source book:

```bash
source venv/bin/activate
./venv/bin/python tests/bbox_development_pilot.py \
    --out out/consolidated_bbox/scan66_development_pilot_report.json
./venv/bin/python tests/bbox_development_pilot_regression.py
```

The tracked fixture `tests/data/bbox_scan66_development_pilot.json` contains the audited 32-line/356-word page geometry, all ten independently audited target identities, and the exact ten serialized treatment boxes from each saved capture. It binds the original image, manifest, geometry audit, target draft, target audit and treatment files by SHA-256. `--verify-sources` additionally checks those ignored workspace artifacts when they are present. Production does not read the fixture or annotations.

Using `tests/bbox_metrics.grade` on the identical targets, the historical full-query diagnostic covers 8/10 targets fully, with no unresolved or wrong-line cases. Its failures are one adjacent-word-only box (`c_5ada9f4e1cb48bc34d4a`) and one clipped target boundary at 77.23% coverage (`c_636fda97354b75c88f3c`). The saved consolidated offline capture has 10/10 null boxes because it contained no compatible cached evidence for this page. That is an availability result, not a measured placement error. It does not support a production derivation change, so this pilot deliberately makes none. The replay costs $0 and neither reads nor changes runtime evidence caches. One development page supplies no statistical inference, detector selection or promotion evidence.

Historical v5 locator, reader, review and metric modules are frozen under `tests/historical/`, with original source hashes in `provenance.json`. Historical scripts explicitly import those snapshots. The 240 saved boxes and v5 metrics reproduce identically, including box SHA-1 `6d739ee16febdf344664dea7c37d7698a103615b` and truth SHA-1 `e38e0ed322edb770c33a40a0680db60c8b146b14`. They must remain separate from v6.

`tests/bbox_capture.py` captures full-query outputs offline against a manifest without opening annotations. Development captures recorded 23 located/376 unresolved for the consolidated engine without newly acquired scan observations, and 374 located/25 unresolved for the historical full-query control. These are **box availability counts, not accuracy scores**. The historical full-query control changes the query contract and is not an exact old-production snapshot. Do not use these captures to claim improvement or promotion. Re-capture after derivation changes.

A user can start a new review process after installing this code using `farsi2epub review ghessehaye-majid --all --bbox-mode auto --bbox-max-cost 5`. This acquires new supported evidence up to the per-process budget and may leave corrections unresolved. It keeps prior human decisions and text edits; it does not reset a book or reuse old derived fallback boxes. No review was launched on that book during this pilot.

## Detector experiment

`tests/bbox_detector_eval.py` ran projection and Kraken 6.0.3 baseline segmentation on the same six development pages. Kraken was installed only in an isolated research directory, not added to the project runtime. Its documented baseline segmenter supports `horizontal-rl`; model suitability still needs corpus evaluation ([Kraken documentation](https://kraken.re/6.0.0/advanced/segmentation.html)).

| Book/page | Projection lines | Kraken lines |
|---|---:|---:|
| ghessehaye-majid 128 | 24 | 27 |
| ghessehaye-majid 35 | 28 | 30 |
| bachehaye_ghali 91 | 30 | 31 |
| ghessehaye-majid 61 | 28 | 30 |
| Hossein_shenasi 1 | 8 | 14 |
| ghessehaye-majid 159 | 28 | 32 |

`tests/bbox_line_metrics.py` now scores independently reviewed development geometry: clean one-to-one line recall/precision, missed/spurious lines, split/merge errors, and per-word area coverage on clean matches. It rejects held-out tuning, image drift, and changed audit artifacts. Detector runs also cover the two annotation pilot pages in `projection_annotation_pilot.json` and `kraken_annotation_pilot.json`; scan66 now has an independently audited development line score: projection clean recall 31/32, Kraken 32/32; projection misses only the folio. Both detect all 31 body lines without splits/merges. Minimum clean-line word-area coverage is 0.9486 for projection and 0.9127 for Kraken. This one-page diagnostic is not a detector-selection or promotion result.

Reports: `out/consolidated_bbox/projection_dev.json` and `kraken_dev.json`. More detected lines is not evidence of better recall. No winner is selected until audited recall/split/merge errors and downstream placement scores are available. The current candidate retains projection provisionally.

## Limits and release status

- Full independent annotation and second-person audit have not been completed. Ambiguous excerpts still require owner decisions before truth can freeze.
- Unreadable lines are retained as unsupported barriers; they no longer veto unrelated readable passages. Word-gap candidates use narrower thresholds and independently read overlapping windows. Unique exact text correspondence certifies window geometry; missing target evidence, repeated-window ambiguity, and excess buffers remain unresolved. Majid page61 now produces a supported buffered box; the two page128 corrections remain unresolved. This is a live functionality check, not an audited accuracy improvement.
- Automatic deskewing, alternate boundary hypotheses, and independently ink-validated embedded geometry candidates are not implemented. The current geometry work covers original-page crop/scaling transforms and PDF rotation, not arbitrary dewarping.
- Real reader requests now succeed after the account was funded. Two page-91 development runs spent $0.055004 and $0.026854 ($0.081858 total confirmed). Recognition retries now expand crops within adjacent-line whitespace to recover detached ink, while leaving placement rectangles unchanged. Compatible older high-resolution observations replay offline. Three detected lines remain unreadable after combining saved evidence, so all three corrections remain unresolved. `out/consolidated_bbox/live_smoke.json` and `offline_smoke.json` record this outcome; it is not evidence of improved placement accuracy. Region failure diagnostics distinguish unreadability, clipping, missing responses and duplicate region identities. The $1 authorization retains $0.558224 of conservative uncertainty from earlier transport failures; $0.359918 remains available.
- No audited current-production/full-passage baseline or held-out candidate score exists. Consequently S3/S4/S6/S14 remain outstanding, and detector/boundary selection parts of S8/S9 are incomplete.

## Validation and rollback

Run the standalone consolidated, benchmark, metric, historical locator/cache, judge, and exact-v5 regression scripts. `tests/headings_regression.py` could not run here because its required `book2` workspace is absent; it is unrelated to placement. API tests require explicit live execution and separate spending records.

Keep this candidate experimental until the frozen held-out comparison passes and changed boxes are independently audited. If it fails, continue experiments or leave placements unresolved; do not restore `layout`, `scan_vlm`, or model-coordinate fallbacks. Existing historical modules are evaluation-only. Roll back a candidate derivation by reverting its code/version coherently and retaining raw observations; never relabel old boxes or modify the audited answer key to improve a score. User text/sidecars and historical reports are not migration targets.

## Plan tracking

The requested assignments remain the intended owners. Three agents were launched; usage limits interrupted their work twice, and the primary integrated their artifacts and completed additional code locally. This table records deliverable state, not a claim that all 15 agents ran.

| ID | Assigned agent / model / effort | State |
|---|---|---|
| S1 | failure_audit / gpt-6-astra / xhigh | Routing and disputed-grade diagnostics; independent numeric truth still pending |
| S2 | benchmark_tools / gpt-5.6-sol / high | Export, provenance, editor, import/freeze and seven regression tests implemented |
| S3 | page_annotations / gpt-6-astra / high | Two independent development page drafts produced; remaining page queue pending |
| S4 | truth_auditor / gpt-6-astra / xhigh | Digital-page independent audit completed with repaired title geometry; digital scope decision pending; scan geometry and all ten target identities audited |
| S5 | metric_engine / gpt-6-astra / xhigh | v6 instrument and 29 regression tests implemented; real truth validation pending |
| S6 | baseline_runner / gpt-5.6-sol / high | Offline capture tooling and development availability controls; historical full-query capture of all 698 cases complete; audited baselines pending |
| S7 | placement_core / gpt-6-astra / high | Experimental shared service, full queries and QC-coordinate removal implemented |
| S8 | line_geometry / gpt-6-astra / high | Page map/transforms and six-page detector experiment; selection/deskew/ink-validated hints pending |
| S9 | word_geometry / gpt-6-astra / xhigh | Independent line/group recognition and retry path implemented; live reader acquisition verified, page-91 placements unresolved; boundary alternatives pending |
| S10 | target_alignment / gpt-6-astra / xhigh | Full spans, overlap ambiguity, insertion gaps and buffers implemented; corpus calibration pending |
| S11 | evidence_runtime / gpt-5.6-sol / high | Raw/derived caches, claims, four-worker queue, reservations and receipts implemented |
| S12 | review_ui / gpt-5.6-sol / high | Two-source UI, statuses, pending polling and CLI aliases implemented; browser render checked |
| S13 | integration_tests / gpt-5.6-sol / high | Offline, historical and synthetic integration checks pass; live acquisition and free replay verified; real located scan boxes still unvalidated |
| S14 | release_auditor / gpt-6-astra / xhigh | Pending frozen held-out truth, paired evaluation and changed-box audit; no promotion |
| S15 | documentation / gpt-5.6-luna / medium | Architecture/migration/experiment/rollback draft recorded; final release record pending |

Local review binds to 127.0.0.1 only. Historical all-interface binding was removed.
