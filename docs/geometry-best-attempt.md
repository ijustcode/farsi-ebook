# Coverage-first word geometry and review placement

Owner-requested update, 2026-09-07: prioritize geometry availability and best-attempt placement over withholding uncertain boxes.

The earlier 23/111 result measured strict evidence support. Ink detection produced fragments, sometimes several per Persian word or one shared across words; line recognition supplied text without reliable word boundaries. The strict locator required independently verified word groups, unambiguous target identity, and at most two buffered words at each edge. Failure of any check hid the box. Thus a low supported count did not mean the remaining page regions had no useful geometry.

Every detected ink region now persists in a correction-independent page map. Recognized words without verified geometry receive approximate regions through monotone allocation over actual ink, using verified word anchors where available. Missing recognition lines retain anonymous ink regions. A separate Markdown-to-page projection uses reading anchors and character proportions for best-attempt correction placement. Estimates include adjacent fragments to reduce clipping. Finding-only queries use ranked text matching; ambiguity is reported. No historical locator is imported, and no QC coordinates are used.

Supported boxes are preserved. Other usable results have status `estimated`, amber dashed overlays, and an “approximate” chip with an explanatory reason. Larger buffers no longer suppress estimates. Sources remain `match|scan`; approximate geometry never changes a word's independently supported flag. Empty targets or missing physical geometry can still be unresolved. Page inventory counts describe detected regions, **not a count of independently recognized linguistic words**.

Derived maps use version 7, migrate compatible observations offline, and bind geometry inventories into their revision. Box keys bind that revision, the Markdown/query, and the attempt version. Migration persists the new inventory before replay. No new model calls are needed for offline estimates.

## Frozen 30-page replay

Workspace: `bachehaye-ghali-fresh30-20260906-123256`, pages 1–30. Same 111 UI correction identities and unchanged Markdown/sidecar hashes. (The hunk-only benchmark has a different denominator.)

| Result | Before | After |
|---|---:|---:|
| Located | 12 | 12 |
| Buffered | 11 | 11 |
| Approximate | 0 | 86 |
| Unresolved | 88 | 2 |
| Box availability | 20.7% | 98.2% |

All 23 original supported boxes remain identical. All 10,915 detected ink regions have rectangles; 634 word associations retain independent support. The two unresolved records have empty snippets: page 16 describes a missing footnote, and page 22 says the purported issue was withdrawn. Neither has an authoritative target span. New API cost: $0. Capture and hash assertions: `out/geometry_best_attempt/capture.py`, `baseline.json`, `after.json`.

Representative page-13 estimate crops were visually inspected. This is not an independent audit of all 86 estimates. Text/ink correspondence can still drift, particularly across missing readings and unusual layouts. **98.2% is placement availability, not verified target coverage or accuracy.** The v6 held-out promotion gate remains outstanding and unchanged.

Validation: best-attempt geometry (including migration, ambiguity, punctuation, RTL insertion, and offline replay), page geometry, consolidated placement, review page scope, v6 metrics (29 tests), and benchmark contracts (7 tests) all pass.
