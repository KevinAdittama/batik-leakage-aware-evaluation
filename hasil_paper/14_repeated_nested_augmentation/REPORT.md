# Repeated Strictly Nested Fold-Local Augmentation

- Clean development originals: 201 (137 batik, 64 non-batik).
- Design: 5 repeats x 5 outer folds x 4 inner folds.
- Candidate space: 4 fixed feature groups x 3 fixed model families.
- Every training split contains 200 instances per class. Only descendants of originals in that training split are eligible.
- Inner validation and outer test folds contain originals only. The external collection is never loaded.

## Repeat-level estimates

- Macro-F1: 0.907305 +/- 0.012955.
- Balanced accuracy: 0.907573 +/- 0.009989.
- MCC: 0.814880 +/- 0.025965.

These values are a repeated nested sensitivity analysis and are not numerically interchangeable with the active single-loop five-fold estimate.
