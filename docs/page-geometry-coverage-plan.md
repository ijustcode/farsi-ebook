# Build reusable page geometry and repair review coverage

## Findings and acceptance target

The fresh workspace is `bachehaye-ghali-fresh30-20260906-123256`, PDF pages 1–30.

- **22/111 corrections have boxes: 19.8% availability.**
- Saved maps contain 5,404 entries, but only 455 have verified geometry; 213/720 detected lines lack readings.
- The screenshot’s phrase, **«آقا» گفت:** on page 13, matches successfully. Placement fails because **آقا** lacks verified geometry.
- Fourteen rejected matching attempts contain unique exact text occurrences but fail context checks.
- Acquisition cost $3.44, versus $0.72 for transcription and $0.85 for QC. Completing maps by extending the current tiny-crop approach would be expensive.

Acceptance: **at least 100/111 supported placements**, image verification of every new or changed placement, and an explanation for every unresolved correction. Availability and independently checked placement quality remain separate measurements.

## Standard pipeline and page-map contract

- Make geometry acquisition automatic after transcription and before optional QC, for every requested successfully transcribed page in every slug—even pages without QC findings.
- Add `--bbox-mode auto|offline`, `--bbox-model`, and `--bbox-max-cost` to transcription; default to automatic acquisition with a separate **$5 per-run cap**. Print transcription, geometry, and QC spending separately.
- Provide a `geometry <slug>` command with page selection and the same geometry options for backfilling or resuming existing books without retranscription. Rerunning transcription also resumes missing geometry for already-transcribed requested pages.
- Introduce correction-independent `ensure_page_map(page)` and page-map progress/results contracts. Review and transcription use the same service.
- Persist ordered lines, stable word identities, original and normalized text, verified rectangles or explicit missing geometry, evidence references, and per-region failure reasons. Include detected/read/verified counts and distinguish complete, partial, blank/nontext, and acquisition-blocked pages.
- Preserve successful transcription when geometry is incomplete. Report the incomplete processing explicitly and make it resumable; do not label a nonempty map complete.
- Merge acquired evidence without discarding previously verified words. Bind derived placements to the map revision; preserve compatible raw readings and invalidate obsolete derived maps/boxes through version changes.

## Evidence acquisition and matching fixes

**Read and locate every readable line independently of correction queries.**

- Remove the current dependency that acquires word geometry only after a correction already matches.
- Expand line crops to include detached ink within neighboring-line boundaries. Reconcile missed, split, or merged lines using overlapping contextual reads; retain uncertainty locally instead of discarding an otherwise readable line.
- Replace isolated fragment-first requests with batched physical word-group crops. Generate candidate boundaries from ink components and gaps, including attached dots, quotation marks, and adjacent fragments.
- Reconcile independent group readings with the line’s ordered text using monotone alignment. Retry unmatched regions with merged adjacent groups, additional whitespace, and higher resolution, bounded to two retry rounds.
- Accept geometry only when the evidence establishes its word identity. Preserve split/fused-word handling and the existing two-word buffer limit; do not introduce character-width estimates or historical coordinate fallbacks.
- Preserve successful work after each batch, including when a later request fails or the budget stops acquisition. Record failed attempts separately from reusable evidence.

**Resolve correction identity against the reusable page reading.**

- Build an ordered Markdown-to-page-word alignment with Persian letter/digit normalization and split/fused-token handling. Keep original text offsets for correction spans.
- Use confident surrounding alignment to identify occurrences despite a transcription error or an unread neighboring line. Missing context should not automatically veto an otherwise supported match.
- Evaluate unique exact matches against the available page alignment; uniqueness in an incomplete map alone must not certify identity.
- Retain ambiguity rejection for repeated phrases. Finding-only snippets must not silently choose their first occurrence.
- Preserve full-passage targets, multiline segments, insertion-boundary markers, and punctuation attachment.

## Diagnostics, costs, and validation

- Show actionable unresolved reasons in correction rows and a page summary: missing reading, missing word geometry, ambiguous occurrence, excessive buffer, API failure, or budget limit.
- Make wait-for-boxes completion report located/buffered/unresolved totals and page-map completeness.
- Add regressions for the screenshot, clipped/detached ink, partial line readability, fragmented Persian words, repeated phrases, damaged context, finding-only corrections, multiline targets, and insertions.
- Test acquisition without any correction queries, incremental map merging, old-workspace backfill, cache invalidation, offline replay, concurrent deduplication, interruption, and budget exhaustion preserving completed work.
- Run the consolidated placement, review-scope, benchmark, and v6 metric regressions; preserve historical v5 reproducibility.
- Spend **at most $5 additional across all repair experiments and acquisition runs**, tracked in one resumable repair ledger. Use at most $1 initially on representative pages, including page 13; measure cost per completed page before using the remainder.
- Report the observed full-map cost and projected book cost. If the cap prevents completion or the coverage target, preserve results and report the shortfall; do not increase spending or declare success.

## Recreate the same review

1. Freeze the current 111 correction identities, text/QC hashes, and baseline boxes before implementation changes. Preserve human edits and decisions.
2. Establish target locations from unmarked page images before examining candidate placements.
3. Backfill pages 1–30 through the general geometry service, reusing compatible evidence. Do not rerun transcription or QC.
4. Compare the identical correction population. Require at least 100 supported placements, no loss of previously correct coverage, and no accepted wrong-occurrence or wrong-line placements in the image audit. Inspect excessive coverage separately.
5. Stop only this workspace’s review server and recreate its review with `--pages 1-30 --bbox-mode offline --wait-for-boxes`, opening the actual returned local URL.
6. Deliver before/after coverage, map completeness, image-audit findings, remaining failures, and additional spend.

This is a production-system repair validated on the same 30-page review. Passing this review’s acceptance criteria does not constitute the separate multi-book held-out promotion gate.
