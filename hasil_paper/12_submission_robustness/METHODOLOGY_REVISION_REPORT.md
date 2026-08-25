# Methodology revision report


## Valid analyses completed


- Perceptual-hash audit: 201 development and 60 external originals. Candidate rule: pHash Hamming <= 8 OR dHash Hamming <= 8. Found 37 development--development and 6 development--external candidate pairs. These are **not confirmed duplicates** until manually reviewed.

- Nested stratified CV used only development originals. The inner 4-fold loop selected among 3 fixed model families and 4 predefined feature groups; the outer 5-fold loop estimated performance. Outer macro-F1 = 0.925 +/- 0.014; balanced accuracy = 0.924; MCC = 0.853. Scaling for LR/SVM was fit inside each inner/outer training split.

- Learned metadata-only negative control (extension, dimensions, aspect ratio, file size) achieved development 5-fold macro-F1 = 0.947 +/- 0.046, and external macro-F1 = 0.348. This diagnoses acquisition predictiveness; it does not prove what pixel models learned.

- Format-by-class support was exported descriptively. No arbitrary matched-subset inferential score was produced because sparse/confounded cells would make that estimate unstable and researcher-dependent.


## Interpretation limits


- No source/object/session identifiers exist; therefore neither this script nor the existing pipeline performs group-aware CV. Stratified file-level folds can still share unobserved acquisition groups.

- Perceptual hashes are heuristic and can miss crops, strong edits, or semantically repeated source images. Every candidate needs visual/manual adjudication before exclusion or grouping.

- Nested CV addresses model/feature-family selection optimism, but it cannot repair acquisition--class confounding or missing scientific ground truth.

- This nested-CV diagnostic intentionally uses originals only. It is not numerically interchangeable with the active augmented-training CV; a nested fold-local augmentation experiment would be a separate, more expensive analysis.

- The external collection has already informed manuscript development; its results remain exploratory rather than a new prospective confirmation.


## Still requires new provenance/data


1. Expert label protocol, annotator IDs/agreement, source URLs, licenses, and source/object/session group IDs.

2. Manual adjudication of near-duplicate candidates and rerun with confirmed groups.

3. Group-aware nested CV after group IDs exist.

4. Acquisition-balanced development data and a newly frozen multi-source external benchmark.

5. Controlled RGB/grayscale and preprocessing/compression experiments if architectural explanations are claimed.

