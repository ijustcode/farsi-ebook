# Disputed historical grades

Diagnostic artifacts are in `out/consolidated_bbox/audit/index.html` and `diagnostics.json`. They include unmarked source crops, saved candidates, historical inferred word geometry, source hashes, and detector witnesses. These are investigation artifacts, not independently audited truth.

Both `haaji-agha` p62 and `bachehaye_ghali` p64 have the same rendered page pixels as the stored high-resolution pages. Changed source pixels do not explain their disputed grades.

On `haaji-agha` p62, the old grader has 34 reading lines and 34 detected lines. A compensating extra and missing detection can therefore pass equal-count pairing while shifting local correspondence. The saved h100000 and h100001 boxes receive word-miss sums 105 and 50, respectively. The page crops show the queried passages on the printed page; exact independently annotated word rectangles are still required to assign repaired scores. Equal line counts must not certify correspondence.

On `bachehaye_ghali` p64 h2, the saved rectangle visibly includes one `یواش` and the neighboring `و`; the old geometry attributes the covered words to `کرد و` and reports word-miss 3 for the two-word target `یواش یواش`. The image does not support interpreting this as a tight, fully covered target either: the second `یواش` is outside the saved box. This is simultaneously a grading-geometry problem and an incomplete candidate, not evidence that every disputed box is correct.

The old scores remain unchanged as reproducibility controls. Any repaired instrument must regrade those exact serialized boxes, identify its independent full-passage targets, and record the instrument effect separately from placement changes. No repaired numeric scores are asserted in this audit.
