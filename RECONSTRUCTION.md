# Rebuilding the image corpus

The images are not redistributed here. This note explains how to assemble a
corpus this code will accept, and how to prove that what you assembled is
byte-for-byte the corpus the paper used.

## What the manifests give you

| Manifest | Rows | Purpose |
|---|---:|---|
| `hasil_paper/01_audit/development_manifest.csv` | 201 | the clean development set, one row per file |
| `hasil_paper/01_audit/external_manifest.csv` | 60 | the separate external collection |
| `hasil_paper/01_audit/excluded_from_analysis.csv` | 11 | files removed by audited decision, with the reason |

Every row carries `sha256`, `file_size_bytes`, `width`, `height`, `extension`,
`kelas`, `subjenis`, `source_set` and the relative `path` the pipeline expects.
Development rows also carry `group_id`, the source-photograph grouping that
every fold split respects.

## Procedure

1. Recreate the directory layout named in the `path` column, rooted at
   `dataset_batik/` for development files and `uji_eksternal/` for external
   files.
2. Place each image at its path.
3. Verify. Every file must match its recorded SHA-256:

```powershell
.\env\Scripts\python.exe -B .\01_audit_dataset.py
```

Stage 01 recomputes the hash of every file it reads and fails loudly on any
mismatch, missing file, or unreadable image. It also re-applies the exclusion
and source-group decisions in `audit_config/` and checks each one against the
recorded hash, so a substituted file cannot pass silently.

## What differs if you cannot obtain identical files

The grouping in `audit_config/source_groups.csv` was decided from full-
aligned residuals between specific files, identified by SHA-256. If your copy
of an image differs by even one byte, that decision no longer applies to it and
stage 01 will refuse it rather than guess. Reconstructing the study with
different images is possible, but it is a different study and the published
numbers will not reproduce.

## Provenance limits

The corpus was assembled from secondary collections and online sources, and
licence records are incomplete. This is stated as a limitation in the
manuscript rather than resolved. The 8 same-photograph
pairs found during the audit are a direct consequence: material gathered this
way can contain crops of one original without any metadata saying so.
