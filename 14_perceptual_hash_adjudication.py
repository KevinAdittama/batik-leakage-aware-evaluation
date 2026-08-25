"""Reproduce and adjudicate perceptual-hash candidates without touching images.

This audit imports the pHash/dHash implementation from the existing submission
robustness stage, verifies the persisted outputs, and creates review artifacts.
Visual decisions remain explicitly preliminary until a named human reviewer
confirms them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import textwrap

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "hasil_paper" / "01_audit"
ROBUSTNESS_DIR = ROOT / "hasil_paper" / "12_submission_robustness"
OUT = (
    ROOT
    / "IJIES_REVISI_FINAL"
    / "04_Tabel_Manifest_dan_Hasil"
    / "Audit_Numerik_dan_Eksperimen"
    / "outputs"
    / "perceptual_hash_adjudication"
)
STAGE12 = ROOT / "12_submission_robustness.py"

DEV_MANIFEST = AUDIT_DIR / "development_manifest.csv"
EXT_MANIFEST = AUDIT_DIR / "external_manifest.csv"
EXISTING_HASHES = ROBUSTNESS_DIR / "perceptual_hash_manifest.csv"
EXISTING_CROSS = ROBUSTNESS_DIR / "near_duplicate_candidates_development_external.csv"
EXISTING_WITHIN = ROBUSTNESS_DIR / "near_duplicate_candidates_development.csv"
APPROVED_EXCLUSIONS = AUDIT_DIR / "approved_exclusion_decisions.csv"

PHASH_THRESHOLD = 8
DHASH_THRESHOLD = 8

# Filled only after the generated contact sheet is visually inspected. Values
# are preliminary assistant review, not expert/human adjudication.
PRELIMINARY_ADJUDICATION: dict[tuple[str, str], tuple[str, str]] = {
    (
        "dataset_batik/non_batik/kotak/kotak1.jpg",
        "uji_eksternal/non_batik/kain_kotak (1).jpg",
    ): (
        "confirmed_near_duplicate",
        "Same red-and-white checkered cloth photograph; differences are consistent with re-encoding or resizing (pHash=0, dHash=0).",
    ),
    (
        "dataset_batik/batik/batik solo/solo3.jpg",
        "uji_eksternal/batik/batik_kawung (2).jpg",
    ): (
        "confirmed_near_duplicate",
        "Same brown-and-white kawung tile image; only rendering dimensions/compression differ (gray correlation approximately 0.999).",
    ),
    (
        "dataset_batik/non_batik/bunga_nonbatik/bunga (2).png",
        "uji_eksternal/batik/batik_betawi (3).jpg",
    ): (
        "not_duplicate_hash_collision",
        "Different motifs: pale multicolor floral repeat versus dark multicolor vertical geometric stripes.",
    ),
    (
        "dataset_batik/non_batik/bunga_nonbatik/bunga (2).png",
        "uji_eksternal/batik/batik_kawung.png",
    ): (
        "not_duplicate_hash_collision",
        "Different motifs: pale floral repeat versus red-and-white kawung geometry.",
    ),
    (
        "dataset_batik/batik/Yogyakarta_Kawung/0016.jpg",
        "uji_eksternal/batik/batik_kawung (2).jpg",
    ): (
        "not_duplicate_hash_collision",
        "Both depict kawung-like repeats, but the element shapes, colors, spacing, and source image are visibly different.",
    ),
    (
        "dataset_batik/non_batik/tenun/tenun (9).png",
        "uji_eksternal/non_batik/kain_tenun_ikat (2).jpg",
    ): (
        "not_duplicate_hash_collision",
        "Different textile objects and patterns: orange-brown geometric panel versus blue fringed woven cloth.",
    ),
    (
        "dataset_batik/non_batik/tenun/tenun (4).png",
        "uji_eksternal/batik/batik_betawi (3).jpg",
    ): (
        "not_duplicate_hash_collision",
        "Different images: blue floral textile panel versus multicolor vertical geometric stripes.",
    ),
    (
        "dataset_batik/non_batik/tenun/tenun (4).png",
        "uji_eksternal/batik/batik_kawung.png",
    ): (
        "not_duplicate_hash_collision",
        "Different images: blue floral textile panel versus red-and-white kawung geometry.",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_stage12():
    spec = importlib.util.spec_from_file_location("submission_robustness_stage12", STAGE12)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {STAGE12}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise AssertionError(f"Missing columns in {source}: {sorted(missing)}")


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(DEV_MANIFEST)
    ext = pd.read_csv(EXT_MANIFEST)
    required = {"source_set", "source_id", "path", "kelas", "subjenis", "sha256"}
    require_columns(dev, required, DEV_MANIFEST)
    require_columns(ext, required, EXT_MANIFEST)
    if dev.empty or ext.empty:
        raise AssertionError("Clean development/external manifests must not be empty")
    combined = pd.concat([dev, ext], ignore_index=True)
    if combined["source_id"].duplicated().any():
        raise AssertionError("source_id is not unique across development/external")
    if combined["sha256"].duplicated().any():
        raise AssertionError("Exact SHA-256 overlap remains in the clean manifests")
    for path in combined["path"]:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(ROOT / path)
    return dev, ext


def normalize_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "left_set", "left_path", "left_class", "left_subtype",
        "right_set", "right_path", "right_class", "right_subtype",
        "phash_hamming", "dhash_hamming", "trigger",
    ]
    return frame[columns].sort_values(
        ["phash_hamming", "dhash_hamming", "left_path", "right_path"]
    ).reset_index(drop=True)


def reproduce_hash_outputs(dev: pd.DataFrame, ext: pd.DataFrame):
    stage12 = load_stage12()
    combined = pd.concat([dev, ext], ignore_index=True)
    hashes = stage12.build_hash_table(combined)
    dev_hash = hashes.query("source_set == 'development'").reset_index(drop=True)
    ext_hash = hashes.query("source_set == 'external'").reset_index(drop=True)
    cross = stage12.candidate_pairs(dev_hash, ext_hash)
    within = stage12.candidate_pairs(dev_hash)

    existing_hashes = pd.read_csv(EXISTING_HASHES)
    hash_columns = [
        "source_set", "source_id", "path", "kelas", "subjenis", "phash_64", "dhash_64"
    ]
    pd.testing.assert_frame_equal(
        hashes[hash_columns].sort_values("source_id").reset_index(drop=True),
        existing_hashes[hash_columns].sort_values("source_id").reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        normalize_candidate(cross),
        normalize_candidate(pd.read_csv(EXISTING_CROSS)),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        normalize_candidate(within),
        normalize_candidate(pd.read_csv(EXISTING_WITHIN)),
        check_dtype=False,
    )
    return stage12, hashes, dev_hash, ext_hash, cross, within


def all_cross_distances(stage12, dev_hash: pd.DataFrame, ext_hash: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left in dev_hash.itertuples(index=False):
        for right in ext_hash.itertuples(index=False):
            rows.append({
                "left_source_id": left.source_id,
                "right_source_id": right.source_id,
                "left_path": left.path,
                "right_path": right.path,
                "phash_hamming": stage12._hex_distance(left.phash_64, right.phash_64),
                "dhash_hamming": stage12._hex_distance(left.dhash_64, right.dhash_64),
            })
    frame = pd.DataFrame(rows)
    if len(frame) != len(dev_hash) * len(ext_hash):
        raise AssertionError("Cross-distance matrix is incomplete")
    return frame


def threshold_summary(distances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in range(0, 17):
        p = distances["phash_hamming"] <= threshold
        d = distances["dhash_hamming"] <= threshold
        rows.append({
            "hamming_threshold": threshold,
            "phash_candidates": int(p.sum()),
            "dhash_candidates": int(d.sum()),
            "or_candidates": int((p | d).sum()),
            "and_candidates": int((p & d).sum()),
        })
    return pd.DataFrame(rows)


def image_similarity(left_path: str, right_path: str) -> dict[str, float | int]:
    left = cv2.imread(str(ROOT / left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(ROOT / right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise ValueError(f"Unreadable pair: {left_path}, {right_path}")
    left_resized = cv2.resize(left, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    right_resized = cv2.resize(right, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    mae = float(np.mean(np.abs(left_resized - right_resized)) / 255.0)
    left_gray = cv2.cvtColor(left_resized.astype(np.uint8), cv2.COLOR_BGR2GRAY).ravel()
    right_gray = cv2.cvtColor(right_resized.astype(np.uint8), cv2.COLOR_BGR2GRAY).ravel()
    corr = float(np.corrcoef(left_gray, right_gray)[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "left_width": int(left.shape[1]),
        "left_height": int(left.shape[0]),
        "right_width": int(right.shape[1]),
        "right_height": int(right.shape[0]),
        "normalized_rgb_mae": mae,
        "normalized_gray_correlation": corr,
    }


def enrich_cross_candidates(cross: pd.DataFrame) -> pd.DataFrame:
    dev = pd.read_csv(DEV_MANIFEST).set_index("path")
    ext = pd.read_csv(EXT_MANIFEST).set_index("path")
    rows = []
    for index, row in cross.reset_index(drop=True).iterrows():
        record = row.to_dict()
        record["candidate_id"] = f"X{index + 1:02d}"
        record["left_source_id"] = dev.loc[row["left_path"], "source_id"]
        record["left_sha256"] = dev.loc[row["left_path"], "sha256"]
        record["right_source_id"] = ext.loc[row["right_path"], "source_id"]
        record["right_sha256"] = ext.loc[row["right_path"], "sha256"]
        record.update(image_similarity(row["left_path"], row["right_path"]))
        decision = PRELIMINARY_ADJUDICATION.get((row["left_path"], row["right_path"]))
        if decision is None:
            record["preliminary_visual_status"] = "pending_visual_review"
            record["preliminary_visual_reason"] = "Contact sheet not yet adjudicated"
        else:
            record["preliminary_visual_status"], record["preliminary_visual_reason"] = decision
        record["recommended_action"] = (
            "exclude_development_source_after_confirmation"
            if record["preliminary_visual_status"] == "confirmed_near_duplicate"
            else "retain_both"
        )
        record["human_adjudication_status"] = "pending_named_human_review"
        record["human_reviewer_id"] = ""
        record["human_review_date"] = ""
        rows.append(record)
    columns_first = [
        "candidate_id", "left_path", "right_path", "left_class", "right_class",
        "phash_hamming", "dhash_hamming", "trigger",
        "normalized_rgb_mae", "normalized_gray_correlation",
        "preliminary_visual_status", "preliminary_visual_reason",
        "recommended_action",
        "human_adjudication_status", "human_reviewer_id", "human_review_date",
    ]
    frame = pd.DataFrame(rows)
    return frame[columns_first + [column for column in frame.columns if column not in columns_first]]


def fit_image(path: Path, box: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(box, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", box, "white")
        x = (box[0] - image.width) // 2
        y = (box[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def make_contact_sheet(candidates: pd.DataFrame, output: Path) -> None:
    font = ImageFont.load_default()
    width, image_h, label_h = 1500, 260, 86
    margin, gutter = 24, 18
    cell_w = (width - 2 * margin - gutter) // 2
    row_h = image_h + label_h
    canvas = Image.new("RGB", (width, margin * 2 + row_h * len(candidates)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in candidates.reset_index(drop=True).iterrows():
        y = margin + index * row_h
        left_img = fit_image(ROOT / row.left_path, (cell_w, image_h))
        right_img = fit_image(ROOT / row.right_path, (cell_w, image_h))
        canvas.paste(left_img, (margin, y))
        canvas.paste(right_img, (margin + cell_w + gutter, y))
        draw.rectangle((margin, y, margin + cell_w, y + image_h), outline="#333333", width=2)
        draw.rectangle(
            (margin + cell_w + gutter, y, margin + 2 * cell_w + gutter, y + image_h),
            outline="#333333", width=2,
        )
        summary = (
            f"{row.candidate_id} | pHash={row.phash_hamming}, dHash={row.dhash_hamming}, "
            f"MAE={row.normalized_rgb_mae:.3f}, corr={row.normalized_gray_correlation:.3f}"
        )
        left_label = "DEV: " + row.left_path
        right_label = "EXT: " + row.right_path
        for line_no, line in enumerate(textwrap.wrap(summary, width=105)[:2]):
            draw.text((margin, y + image_h + 4 + 13 * line_no), line, fill="black", font=font)
        for line_no, line in enumerate(textwrap.wrap(left_label, width=78)[:3]):
            draw.text((margin, y + image_h + 31 + 13 * line_no), line, fill="#1F4E79", font=font)
        for line_no, line in enumerate(textwrap.wrap(right_label, width=78)[:3]):
            draw.text(
                (margin + cell_w + gutter, y + image_h + 31 + 13 * line_no),
                line, fill="#7A3E00", font=font,
            )
        draw.line((margin, y + row_h - 1, width - margin, y + row_h - 1), fill="#BBBBBB", width=1)
    canvas.save(output, optimize=True)


def write_report(
    hashes: pd.DataFrame,
    cross: pd.DataFrame,
    within: pd.DataFrame,
    candidates: pd.DataFrame,
    thresholds: pd.DataFrame,
    approved: pd.DataFrame,
) -> None:
    exact_like = candidates.query("phash_hamming == 0 and dhash_hamming == 0")
    preliminary_counts = candidates["preliminary_visual_status"].value_counts().to_dict()
    proposed = candidates.query("recommended_action == 'exclude_development_source_after_confirmation'")
    development_n = int((hashes.source_set == "development").sum())
    external_n = int((hashes.source_set == "external").sum())
    text = f"""# Perceptual-hash audit and adjudication tracker

As of 15 August 2026, this audit reproduced the persisted pHash/dHash results
from the clean original-image manifests without reading `dataset_aug`.

## Verified scope

- Development originals: {int((hashes.source_set == 'development').sum())}
- External originals: {int((hashes.source_set == 'external').sum())}
- Cross-set comparisons: {development_n * external_n}
- Candidate rule: pHash Hamming <= {PHASH_THRESHOLD} OR dHash Hamming <= {DHASH_THRESHOLD}
- Development--external candidates: {len(cross)}
- Development--development candidates: {len(within)}
- Cross candidates with both pHash=0 and dHash=0: {len(exact_like)}

The current pHash manifest and candidate CSVs were reproduced exactly. The
threshold-sensitivity table reports candidate counts for Hamming cutoffs 0--16.

## Adjudication state

Preliminary visual statuses: {preliminary_counts}.

The two previously confirmed cross-set near duplicates were approved by the
user and applied as development-side manifest exclusions. The current clean
development manifest therefore contains {development_n} originals, and this
rerun contains {len(proposed)} additional proposed exclusions. The applied
decision records are preserved in `applied_development_exclusions.csv`
({len(approved)} rows); no physical source image was deleted.

The six remaining cross-set candidates are retained because visual review
classifies them as simple-hash collisions rather than duplicate source images.
Their named-review fields remain open for traceability, but they do not trigger
further exclusion in the current approved dataset decision.

## Interpretation rule

- `confirmed_near_duplicate`: visually the same source image or a re-encoded,
  resized, or lightly edited rendering of it.
- `not_duplicate_hash_collision`: visually different images whose simple
  structure produces a small Hamming distance in one hash.
- `uncertain_requires_human`: insufficient evidence for either conclusion.

pHash/dHash are screening heuristics. They can miss crops or strong edits and
must not be described as complete proof of independence.
"""
    (OUT / "PERCEPTUAL_HASH_AUDIT_REPORT.md").write_text(text, encoding="utf-8")


def write_output_manifest() -> None:
    manifest_path = OUT / "output_manifest_sha256.csv"
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path != manifest_path:
            rows.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        if old.is_file():
            old.unlink()

    dev, ext = load_manifests()
    stage12, hashes, dev_hash, ext_hash, cross, within = reproduce_hash_outputs(dev, ext)
    distances = all_cross_distances(stage12, dev_hash, ext_hash)
    thresholds = threshold_summary(distances)
    candidates = enrich_cross_candidates(cross)
    approved = pd.read_csv(APPROVED_EXCLUSIONS)
    if len(approved) != 2 or not approved["status"].eq("approved_exclude").all():
        raise AssertionError("Expected exactly two approved development exclusions")

    hashes.to_csv(OUT / "perceptual_hash_manifest_verified.csv", index=False)
    candidates.to_csv(OUT / "near_duplicate_candidates_cross_adjudication.csv", index=False)
    candidates.query(
        "recommended_action == 'exclude_development_source_after_confirmation'"
    ).to_csv(OUT / "proposed_development_exclusions.csv", index=False)
    approved.to_csv(OUT / "applied_development_exclusions.csv", index=False)
    within.to_csv(OUT / "near_duplicate_candidates_development_pending_review.csv", index=False)
    thresholds.to_csv(OUT / "cross_threshold_sensitivity.csv", index=False)
    make_contact_sheet(candidates, OUT / "cross_candidate_contact_sheet.png")
    write_report(hashes, cross, within, candidates, thresholds, approved)
    write_output_manifest()

    print(f"Verified hashes: {len(hashes)}")
    print(f"Cross candidates: {len(candidates)}")
    print(f"Within-development candidates pending review: {len(within)}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
