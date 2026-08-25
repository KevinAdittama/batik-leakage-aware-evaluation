"""Visual QA transformasi augmentasi per kelas."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline_common import read_image_color, reset_directory, resolve_project_path
from pipeline_config import AUGMENTATION_DIR, PREVIEW_AUGMENTED_DIR, RANDOM_SEED


def make_class_preview(frame: pd.DataFrame, class_name: str):
    augmented = frame.query("kelas == @class_name and is_augmented == True")
    if augmented.empty:
        raise RuntimeError(f"Tidak ada augmentasi untuk {class_name}.")
    chosen = augmented.drop_duplicates("transform").sample(
        n=min(12, augmented["transform"].nunique()), random_state=RANDOM_SEED
    )
    figure, axes = plt.subplots(3, 4, figsize=(13, 10))
    axes = axes.ravel()
    for axis in axes:
        axis.axis("off")
    rows = []
    for axis, row in zip(axes, chosen.to_dict("records")):
        image = read_image_color(resolve_project_path(row["generated_path"]))
        if image is None:
            raise OSError(f"Gagal membaca augmentasi: {row['generated_path']}")
        axis.imshow(image[:, :, ::-1])
        axis.set_title(f"{row['transform']}\n{row['subjenis']}", fontsize=9)
        axis.axis("off")
        rows.append(row)
    figure.suptitle(f"Preview Augmentasi — {class_name}", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output = PREVIEW_AUGMENTED_DIR / f"preview_augmentasi_{class_name}.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return rows


def main() -> None:
    manifest_path = AUGMENTATION_DIR / "cv_pool_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Jalankan 03_augment_dataset.py dahulu.")
    reset_directory(PREVIEW_AUGMENTED_DIR)
    frame = pd.read_csv(manifest_path)
    selected = []
    for class_name in ("batik", "non_batik"):
        selected.extend(make_class_preview(frame, class_name))
    frame.groupby(["kelas", "transform"]).size().rename("jumlah").reset_index().to_csv(
        PREVIEW_AUGMENTED_DIR / "transform_counts.csv", index=False
    )
    pd.DataFrame(selected).to_csv(
        PREVIEW_AUGMENTED_DIR / "preview_index.csv", index=False
    )
    print(f"Preview augmentasi: {PREVIEW_AUGMENTED_DIR}")


if __name__ == "__main__":
    main()
