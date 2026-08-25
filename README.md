# Leakage-Aware Evaluation Reveals Acquisition Bias and External Degradation in Binary Batik Recognition

Code and non-image artifacts for IJIES paper 20265049.

Feddy Setio Pribadi, Budi Sunarko, Anan Nugroho, Febry Putra Rochim, Hasan Firdaus Mohd Zaki, Kevin Muhammad Tegar Aditama, Fairizal Arifianto

This repository supports independent verification of every number reported in
the manuscript. It is an audit trail, not a batik recognition library.

## Layout

The workspace layout is preserved so the code runs unchanged.

| Path | Contents |
|---|---|
| `*.py` | the analysis pipeline and the table and figure builders |
| `tests/` | the test suite |
| `web_app/`, `app.py` | a small Streamlit demo of the trained classifier |
| `audit_config/` | audited decisions read by stage 01: exclusions and source groups |
| `hasil_paper/` | every text artifact produced by the pipeline, plus the fitted models |
| `RECONSTRUCTION.md` | how to rebuild the image corpus this code expects |
| `ARTIFACT_MANIFEST.csv` | SHA-256 and byte size of every file published here |

`hasil_paper/` holds the sample-level external predictions with true labels,
subtype and score; fold and augmentation-origin manifests; SHA-256 records;
perceptual-hash candidates with adjudication status; fixed seeds;
configurations; and bootstrap indices.

## What this repository does not contain

**No images.** Licence records for the source collections are incomplete, so
the images themselves are not redistributed. Every image is instead identified
by SHA-256, dimensions, subtype and provenance in the manifests under
`hasil_paper/01_audit/`, and `RECONSTRUCTION.md` explains how to rebuild an
identical corpus and verify it hash by hash. For the same reason the
image-rendering stages are omitted from `hasil_paper/`: previews, figures and
normalisation examples.

**No manuscript.** The Word manuscript and the submission correspondence are
not redistributed. The scripts that edit them are included so that every edit
made to the paper can be read and audited, but they cannot run here.

**No expert-audit package.** The blinded package prepared for independent label
validation is withheld because it contains the code map that would undo the
blinding.

## What runs from this repository

Two groups, and the difference matters if you intend to verify anything.

**Runs once the image corpus is rebuilt.** Stages 01 to 12 are the analysis
pipeline, driven by `run_all.py`. Stages 14 to 19, 24, 25 and 30 are the
sensitivity, adjudication and ablation analyses. These read images and
`hasil_paper/`, and reproduce every number in the manuscript.

```powershell
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.txt
.\env\Scripts\python.exe -B .\run_all.py
```

The pipeline is deterministic. `RANDOM_SEED` is 42; repeat seeds are
`RANDOM_SEED + repeat * 10000` and inner seeds `outer_seed + outer_fold * 100`.

**Published for reading, not running.** Stages 18, 22, 23, 26 to 29, 31 and 32
build the tables and figures that appear in the manuscript, and write them into
the Word file. Without that file they fail immediately rather than producing
anything. They are included because they are the evidence behind a claim the
manuscript makes: that every reported number is read from an audited CSV rather
than typed by hand. Read `26_refresh_manuscript_numbers.py` and
`27_refresh_manuscript_prose.py` to check that claim yourself.

## Stages omitted from this repository

The stage numbers are not contiguous. Nothing is hidden; these scripts handle
editorial and administrative work that has no bearing on any reported result:

- `20_citation_audit.py` - reference formatting audit
- `21_mendeley_field_audit.py` - reference manager field audit
- `33_build_combined_tracker.py` - internal reviewer response tracker
- `34_refresh_submission_letters.py` - cover letter and response letters
- `35_update_dosen_package.py` - internal package for the supervisor
- `36_revise_decision_sheet.py` - internal decision sheet
- `37_build_expert_audit_package.py` - builds the blinded label-audit package and its code map
- `38_build_normalisation_examples.py` - illustrations for the blinded package
- `39_build_decision_sheet_pdf.py` - internal decision sheet, PDF form
- `40_build_revision_update.py` - internal revision progress summary
- `41_build_public_repository.py` - assembles this repository
- `build_paket_dosen.py` - internal package for the supervisor

The corresponding tests are omitted with them. So is the blinded package
prepared for independent label validation, which contains the code map.

The test suite reflects the same split.

```powershell
.\env\Scripts\python.exe -B -m unittest discover -s tests
```

Run against the published artifacts alone, this reports **39 passed,
0 skipped, 15 errored** out of 54 started.
The errors are expected and are not defects: those tests read the image corpus
or the manuscript, neither of which is redistributed. They are reported rather
than hidden so that nothing appears verified when it was not. Rebuild the
corpus as described in `RECONSTRUCTION.md` and the full suite runs.

## Evaluation protocol

Folds are separated at the level of the **source photograph**, not the file.
Re-examining every image pair with full-resolution alignment found
8 pairs that were crops of a single photograph, and
8 of them straddled validation folds. Perceptual hashing had
missed these because it is not shift invariant. The development manifest is
therefore grouped from 201 files into 194 source
groups, and every split uses `StratifiedGroupKFold` over `group_id`.

## Headline results

The selection rule does not separate the two leading classical models. On the
source-group-aware protocol SVM (RBF) leads Random Forest by
0,0069 macro-F1, which is 0,33 of one
between-fold standard error. Across 25 nested outer folds the two are chosen
almost equally often. The manuscript therefore reports all three classical
models side by side and treats the instability itself as a finding.

A controlled ablation of the deep branch crosses colour, pretraining and
dimensionality over the same folds:

| Factor | Mean difference in macro-F1 |
|---|---:|
| ImageNet pretraining | +0.1065 |
| RGB minus grayscale | -0.0244 |
| 512-d minus 6-component PCA | +0.0104 |

Only pretraining matters.

## Licence

The Python code is released under the MIT Licence; see `LICENSE`.

Derived data in `audit_config/` and `hasil_paper/` is released under Creative
Commons Attribution 4.0 International; see `LICENSE-DATA.md`. Those files are
measurements and decisions produced by this project, not the source images.

## Citing

See `CITATION.cff`.

Built 25 Agustus 2026 from the project workspace.
