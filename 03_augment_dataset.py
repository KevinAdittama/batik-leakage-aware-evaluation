"""Buat final train seimbang dan pool augmentasi fold-safe."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline_common import (
    read_image_color,
    reset_directory,
    resolve_project_path,
    safe_stem,
    write_image,
)
from pipeline_config import (
    AUDIT_DIR,
    AUGMENTATION_DIR,
    AUGMENTED_DIR,
    CV_POOL_PER_KELAS,
    RANDOM_SEED,
    TARGET_PER_KELAS,
)


SMALL_ROTATION_DEGREES = 15
MAX_ZOOM = 0.10
MAX_BRIGHTNESS_CONTRAST = 0.15


def apply_transform(image: np.ndarray, index: int, rng: np.random.Generator):
    height, width = image.shape[:2]
    kind = index % 9
    if kind == 0:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "rot90"
    if kind == 1:
        return cv2.rotate(image, cv2.ROTATE_180), "rot180"
    if kind == 2:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "rot270"
    if kind == 3:
        angle = float(rng.uniform(-SMALL_ROTATION_DEGREES, SMALL_ROTATION_DEGREES))
        if abs(angle) < 2:
            angle = 2.0 if angle >= 0 else -2.0
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        output = cv2.warpAffine(
            image, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        return output, f"rot{angle:+.1f}".replace(".", "p")
    if kind == 4:
        return cv2.flip(image, 1), "flip_h"
    if kind == 5:
        return cv2.flip(image, 0), "flip_v"
    if kind == 6:
        zoom = float(rng.uniform(0.02, MAX_ZOOM))
        crop_x, crop_y = max(1, int(width * zoom / 2)), max(1, int(height * zoom / 2))
        crop = image[crop_y:height - crop_y, crop_x:width - crop_x]
        return cv2.resize(crop, (width, height)), f"zoom{round(zoom * 100):02.0f}"
    if kind == 7:
        factor = float(rng.uniform(0.85, 1.15))
        if abs(factor - 1.0) < 0.03:
            factor = 1.03 if factor >= 1.0 else 0.97
        output = cv2.convertScaleAbs(image, alpha=1.0, beta=(factor - 1) * 255)
        return output, f"brightness{factor:.2f}".replace(".", "p")
    factor = float(rng.uniform(0.85, 1.15))
    if abs(factor - 1.0) < 0.03:
        factor = 1.03 if factor >= 1.0 else 0.97
    return cv2.convertScaleAbs(image, alpha=factor), f"contrast{factor:.2f}".replace(".", "p")


def output_name(record: dict, transform: str, sequence: int, suffix: str) -> str:
    return (
        f"{safe_stem(Path(record['path']).stem)}__{record['source_id']}__"
        f"{transform}_{sequence:04d}{suffix.lower()}"
    )


def original_row(record: dict, partition: str, sequence: int):
    source = resolve_project_path(record["path"])
    destination = (
        AUGMENTED_DIR / partition / record["kelas"]
        / output_name(record, "orig", sequence, source.suffix)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "generated_path": destination.relative_to(AUGMENTED_DIR.parent).as_posix(),
        "source_id": record["source_id"],
        "group_id": record.get("group_id", record["source_id"]),
        "source_path": record["path"],
        "kelas": record["kelas"],
        "label": record["label"],
        "subjenis": record["subjenis"],
        "partition": partition,
        "is_augmented": False,
        "transform": "original",
    }


def build_partition(records: list[dict], partition: str, target: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = [
        original_row(record, partition, sequence)
        for sequence, record in enumerate(sorted(records, key=lambda x: x["path"]), 1)
    ]
    for class_name in ("batik", "non_batik"):
        sources = [record for record in records if record["kelas"] == class_name]
        if len(sources) > target:
            raise ValueError(f"Target {target} lebih kecil dari data asli {class_name}.")
        order = rng.permutation(len(sources))
        for index in range(target - len(sources)):
            source = sources[int(order[index % len(order)])]
            image = read_image_color(resolve_project_path(source["path"]))
            if image is None:
                raise OSError(f"Citra gagal dibaca: {source['path']}")
            augmented, transform = apply_transform(image, index, rng)
            destination = (
                AUGMENTED_DIR / partition / class_name
                / output_name(source, transform, index + 1, ".jpg")
            )
            write_image(destination, augmented)
            rows.append(
                {
                    "generated_path": destination.relative_to(AUGMENTED_DIR.parent).as_posix(),
                    "source_id": source["source_id"],
                    "group_id": source.get("group_id", source["source_id"]),
                    "source_path": source["path"],
                    "kelas": class_name,
                    "label": source["label"],
                    "subjenis": source["subjenis"],
                    "partition": partition,
                    "is_augmented": True,
                    "transform": transform,
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["kelas", "source_id", "is_augmented", "generated_path"]
    )
    counts = frame.groupby("kelas").size()
    if not (counts == target).all():
        raise AssertionError(f"Partisi {partition} tidak seimbang: {counts.to_dict()}")
    return frame


def main() -> None:
    manifest_path = AUDIT_DIR / "development_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Jalankan 01_audit_dataset.py dahulu.")
    reset_directory(AUGMENTATION_DIR)
    reset_directory(AUGMENTED_DIR)
    records = pd.read_csv(manifest_path).to_dict("records")
    original_counts = pd.DataFrame(records).groupby("kelas").size()

    final_train = build_partition(records, "final_train", TARGET_PER_KELAS, RANDOM_SEED)
    cv_pool = build_partition(records, "cv_pool", CV_POOL_PER_KELAS, RANDOM_SEED + 1000)
    final_train.to_csv(AUGMENTATION_DIR / "final_train_manifest.csv", index=False)
    cv_pool.to_csv(AUGMENTATION_DIR / "cv_pool_manifest.csv", index=False)

    summary_rows = []
    for class_name in ("batik", "non_batik"):
        summary_rows.append(
            {
                "kelas": class_name,
                "asli": int(original_counts[class_name]),
                "final_train_total": TARGET_PER_KELAS,
                "final_train_augmented": TARGET_PER_KELAS - int(original_counts[class_name]),
                "cv_pool_total": CV_POOL_PER_KELAS,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(AUGMENTATION_DIR / "augmentation_summary.csv", index=False)
    (AUGMENTATION_DIR / "augmentation_report.md").write_text(
        "# Ringkasan Augmentasi\n\n"
        + "\n".join(
            f"- {row['kelas']}: {row['asli']} asli → {row['final_train_total']} "
            f"training instances ({row['final_train_augmented']} hasil augmentasi)."
            for row in summary_rows
        )
        + "\n\nUji eksternal tidak diaugmentasi.\n",
        encoding="utf-8",
    )
    print("=" * 72)
    print("TAHAP 03 — AUGMENTASI SEIMBANG")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nDataset augmentasi: {AUGMENTED_DIR}")
    print(f"Manifest dan laporan: {AUGMENTATION_DIR}")


if __name__ == "__main__":
    main()
