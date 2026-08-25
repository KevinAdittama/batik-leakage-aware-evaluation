"""Preview visual preprocessing dan domain fitur pada citra asli."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline_common import diagnostic_views, read_image_color, reset_directory, resolve_project_path
from pipeline_config import AUDIT_DIR, PREVIEW_ORIGINAL_DIR, PREVIEW_PER_CLASS, RANDOM_SEED


def choose_samples(frame: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for class_name, group in frame.groupby("kelas", sort=False):
        group = group.sample(frac=1.0, random_state=RANDOM_SEED)
        diverse = group.drop_duplicates("subjenis").head(PREVIEW_PER_CLASS)
        if len(diverse) < PREVIEW_PER_CLASS:
            remaining = group.loc[~group.index.isin(diverse.index)]
            diverse = pd.concat(
                [diverse, remaining.head(PREVIEW_PER_CLASS - len(diverse))]
            )
        selected.append(diverse)
    return pd.concat(selected).sort_values(["kelas", "subjenis"]).reset_index(drop=True)


def save_panel(row: dict, sequence: int) -> str:
    image = read_image_color(resolve_project_path(row["path"]))
    if image is None:
        raise OSError(f"Gagal membaca preview: {row['path']}")
    views, features = diagnostic_views(image)
    figure, axes = plt.subplots(3, 3, figsize=(14, 13))
    for axis, (title, view) in zip(axes.ravel(), views.items()):
        cmap = None if view.ndim == 3 else "gray"
        axis.imshow(view, cmap=cmap)
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.axis("off")
    feature_text = " | ".join(
        f"{name}={value:.3f}" for name, value in features.items()
    )
    figure.suptitle(
        f"{row['kelas']} / {row['subjenis']}\n{feature_text}", fontsize=11
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"{sequence:02d}_{row['kelas']}_{row['source_id']}.png"
    figure.savefig(PREVIEW_ORIGINAL_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return filename


def main() -> None:
    manifest_path = AUDIT_DIR / "development_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Jalankan 01_audit_dataset.py dahulu.")
    reset_directory(PREVIEW_ORIGINAL_DIR)
    manifest = pd.read_csv(manifest_path)
    samples = choose_samples(manifest)
    output_rows = []
    for sequence, row in enumerate(samples.to_dict("records"), 1):
        filename = save_panel(row, sequence)
        output_rows.append({**row, "preview_file": filename})
        print(f"  [{sequence}/{len(samples)}] {row['kelas']}/{row['subjenis']}")
    pd.DataFrame(output_rows).to_csv(
        PREVIEW_ORIGINAL_DIR / "preview_index.csv", index=False
    )
    print(f"Preview original: {PREVIEW_ORIGINAL_DIR}")


if __name__ == "__main__":
    main()
