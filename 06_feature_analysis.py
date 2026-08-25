"""Analisis deskriptif dan separasi enam fitur pada citra asli."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline_common import reset_directory
from pipeline_config import FEATURE_ANALYSIS_DIR, FEATURE_DIR, MODEL_FEATURES


def cohen_d(first: np.ndarray, second: np.ndarray) -> float:
    n1, n2 = len(first), len(second)
    pooled = np.sqrt(
        ((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1))
        / max(n1 + n2 - 2, 1)
    )
    return float((first.mean() - second.mean()) / pooled) if pooled > 0 else 0.0


def main() -> None:
    source_path = FEATURE_DIR / "development_original_features.csv"
    if not source_path.exists():
        raise FileNotFoundError("Jalankan 05_extract_features.py dahulu.")
    reset_directory(FEATURE_ANALYSIS_DIR)
    frame = pd.read_csv(source_path)

    descriptive = frame.groupby("kelas")[MODEL_FEATURES].agg(["mean", "std", "median"])
    descriptive.to_csv(FEATURE_ANALYSIS_DIR / "descriptive_statistics.csv")
    frame[MODEL_FEATURES].corr().to_csv(FEATURE_ANALYSIS_DIR / "feature_correlation.csv")

    separation_rows = []
    for feature in MODEL_FEATURES:
        values = frame[feature].to_numpy(float)
        labels = frame["label"].to_numpy(int)
        auc = roc_auc_score(labels, values)
        batik = frame.loc[frame["kelas"] == "batik", feature].to_numpy(float)
        non = frame.loc[frame["kelas"] == "non_batik", feature].to_numpy(float)
        separation_rows.append(
            {
                "feature": feature,
                "mean_batik": batik.mean(),
                "mean_non_batik": non.mean(),
                "cohen_d_batik_minus_non": cohen_d(batik, non),
                "univariate_auc": auc,
                "separation_auc": max(auc, 1 - auc),
            }
        )
    separation = pd.DataFrame(separation_rows).sort_values(
        "separation_auc", ascending=False
    )
    separation.to_csv(FEATURE_ANALYSIS_DIR / "feature_separation.csv", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, feature in zip(axes.ravel(), MODEL_FEATURES):
        batik = frame.loc[frame["kelas"] == "batik", feature]
        non = frame.loc[frame["kelas"] == "non_batik", feature]
        axis.boxplot([non, batik], tick_labels=["Non-Batik", "Batik"], showfliers=False)
        axis.set_title(feature.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Distribusi Enam Fitur pada Citra Asli", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(
        FEATURE_ANALYSIS_DIR / "feature_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    ordered = separation.sort_values("separation_auc")
    axis.barh(ordered["feature"], ordered["separation_auc"], color="#35618f")
    axis.axvline(0.5, color="#aa3333", linestyle="--", label="Acak (0,5)")
    axis.set_xlim(0.45, 1.0)
    axis.set_xlabel("Daya separasi univariat (AUC arah terbaik)")
    axis.set_title("Daya Separasi Fitur")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FEATURE_ANALYSIS_DIR / "feature_separation.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    lines = ["# Analisis Fitur", "", "Analisis dihitung hanya pada 212 citra asli.", ""]
    for row in separation.to_dict("records"):
        lines.append(
            f"- `{row['feature']}`: separation AUC={row['separation_auc']:.3f}, "
            f"Cohen's d={row['cohen_d_batik_minus_non']:.3f}."
        )
    (FEATURE_ANALYSIS_DIR / "feature_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("=" * 72)
    print("TAHAP 06 — ANALISIS FITUR ASLI")
    print("=" * 72)
    print(separation.to_string(index=False))
    print(f"\nHasil: {FEATURE_ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
