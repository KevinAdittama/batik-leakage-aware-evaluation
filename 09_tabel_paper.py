"""Susun tabel dan grafik siap pakai untuk naskah paper."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import reset_directory
from pipeline_config import (
    AUDIT_DIR,
    AUGMENTATION_DIR,
    CV_DIR,
    EXTERNAL_RESULT_DIR,
    TABLE_DIR,
)


def mean_std(row, metric):
    return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    cv_path = CV_DIR / "cv_summary_primary.csv"
    external_path = EXTERNAL_RESULT_DIR / "external_model_summary.csv"
    selected_path = EXTERNAL_RESULT_DIR / "selected_model_result.json"
    if not cv_path.exists() or not external_path.exists() or not selected_path.exists():
        raise FileNotFoundError("Jalankan tahap 07 dan 08 dahulu.")
    reset_directory(TABLE_DIR)
    cv = pd.read_csv(cv_path)
    external = pd.read_csv(external_path)
    merged = cv.merge(external, on="model", suffixes=("_cv", "_external"))
    merged = merged.sort_values("f1_macro_mean", ascending=False)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))

    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "CV Accuracy": mean_std(row, "accuracy"),
                "CV Balanced Accuracy": mean_std(row, "balanced_accuracy"),
                "CV Macro-F1": mean_std(row, "f1_macro"),
                "CV MCC": mean_std(row, "mcc"),
                "CV Recall Batik": mean_std(row, "recall_batik"),
                "CV Recall Non-Batik": mean_std(row, "recall_non_batik"),
                "External Accuracy": f"{row['accuracy']:.3f}",
                "External Balanced Accuracy": f"{row['balanced_accuracy']:.3f}",
                "External Macro-F1": f"{row['f1_macro']:.3f}",
                "External MCC": f"{row['mcc']:.3f}",
                "External Recall Batik": f"{row['recall_batik']:.3f}",
                "External Recall Non-Batik": f"{row['recall_non_batik']:.3f}",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "table_model_comparison.csv", index=False)
    (TABLE_DIR / "table_model_comparison.md").write_text(
        "# Tabel Perbandingan Model\n\n" + markdown_table(table) + "\n",
        encoding="utf-8",
    )

    positions = np.arange(len(merged))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.bar(
        positions - width / 2, merged["f1_macro_mean"], width,
        yerr=merged["f1_macro_std"], capsize=4, label="5-fold CV",
        color="#35618f",
    )
    axis.bar(
        positions + width / 2, merged["f1_macro"], width,
        label="Uji eksternal", color="#d9822b",
    )
    axis.set_xticks(positions, merged["model"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Perbandingan Model: CV vs Uji Eksternal")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(TABLE_DIR / "figure_model_f1.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    development = pd.read_csv(AUDIT_DIR / "development_manifest.csv")
    external_manifest = pd.read_csv(AUDIT_DIR / "external_manifest.csv")
    augmentation = pd.read_csv(AUGMENTATION_DIR / "augmentation_summary.csv")
    summary_lines = [
        "# Ringkasan Hasil untuk Paper",
        "",
        "## Data",
        "",
        f"- Development asli: {len(development)} citra.",
        f"- Uji eksternal independen: {len(external_manifest)} citra.",
        "- Final train: 400 training instances seimbang (200 per kelas).",
        "- Augmentasi hanya digunakan pada train; validation dan eksternal tetap asli.",
        "",
        "## Model formal",
        "",
        f"- Dipilih dari CV: **{selected['model']}**.",
        f"- CV macro-F1: **{selected['cv_f1_macro_mean']:.3f} ± "
        f"{selected['cv_f1_macro_std']:.3f}**.",
        f"- External macro-F1: **{selected['external_f1_macro']:.3f}**.",
        f"- External recall batik: **{selected['external_recall_batik']:.3f}**.",
        f"- External recall non-batik: **{selected['external_recall_non_batik']:.3f}**.",
        "",
        "Model formal dipilih hanya dari CV. Uji eksternal tidak digunakan untuk seleksi.",
        "",
        "## Tabel model",
        "",
        markdown_table(table),
        "",
    ]
    (TABLE_DIR / "paper_results_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    augmentation.to_csv(TABLE_DIR / "table_augmentation.csv", index=False)
    print("=" * 72)
    print("TAHAP 09 — TABEL DAN GRAFIK PAPER")
    print("=" * 72)
    print(table.to_string(index=False))
    print(f"\nHasil siap paper: {TABLE_DIR}")


if __name__ == "__main__":
    main()
