"""Ekstrak enam fitur konsisten untuk seluruh himpunan penelitian."""

import numpy as np
import pandas as pd

from pipeline_common import extract_six, read_image_color, reset_directory, resolve_project_path
from pipeline_config import AUDIT_DIR, AUGMENTATION_DIR, FEATURE_DIR, MODEL_FEATURES


META_COLUMNS = [
    "path", "source_id", "group_id", "source_path", "kelas", "label", "subjenis",
    "partition", "is_augmented", "transform",
]


def records_original(path, partition: str):
    frame = pd.read_csv(path)
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "path": row["path"],
                "source_id": row["source_id"],
                "group_id": row.get("group_id", row["source_id"]),
                "source_path": row["path"],
                "kelas": row["kelas"],
                "label": row["label"],
                "subjenis": row["subjenis"],
                "partition": partition,
                "is_augmented": False,
                "transform": "original",
            }
        )
    return rows


def records_generated(path):
    frame = pd.read_csv(path)
    frame["path"] = frame["generated_path"]
    return frame.to_dict("records")


def extract_records(records: list[dict], name: str) -> pd.DataFrame:
    rows = []
    print(f"\n[{name}] {len(records)} citra")
    for index, record in enumerate(records, 1):
        image = read_image_color(resolve_project_path(record["path"]))
        if image is None:
            raise OSError(f"Citra gagal dibaca: {record['path']}")
        features, _ = extract_six(image)
        rows.append(
            {
                **{column: record.get(column, "") for column in META_COLUMNS},
                **features,
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"  {index}/{len(records)} selesai")
    frame = pd.DataFrame(rows, columns=META_COLUMNS + MODEL_FEATURES)
    values = frame[MODEL_FEATURES].to_numpy(dtype=float)
    if frame.empty or not np.isfinite(values).all():
        raise RuntimeError(f"Fitur {name} kosong atau non-finite.")
    return frame


def main() -> None:
    required = [
        AUDIT_DIR / "development_manifest.csv",
        AUDIT_DIR / "external_manifest.csv",
        AUGMENTATION_DIR / "final_train_manifest.csv",
        AUGMENTATION_DIR / "cv_pool_manifest.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Manifest belum tersedia: {missing}")
    reset_directory(FEATURE_DIR)
    groups = {
        "development_original": records_original(required[0], "development"),
        "final_train": records_generated(required[2]),
        "cv_pool": records_generated(required[3]),
        "external": records_original(required[1], "external"),
    }
    output_names = {
        "development_original": "development_original_features.csv",
        "final_train": "final_train_features.csv",
        "cv_pool": "cv_pool_features.csv",
        "external": "external_features.csv",
    }
    summary = []
    for name, records in groups.items():
        frame = extract_records(records, name)
        frame.to_csv(FEATURE_DIR / output_names[name], index=False)
        counts = frame.groupby("kelas").size().to_dict()
        summary.append({"dataset": name, "total": len(frame), **counts})
        print(f"  [OK] {name}: {counts}")
    pd.DataFrame(summary).to_csv(FEATURE_DIR / "feature_dataset_summary.csv", index=False)
    print(f"\nSeluruh fitur disimpan di: {FEATURE_DIR}")


if __name__ == "__main__":
    main()
