"""Final-fit tiga model pada 400 instances dan uji eksternal asli."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pipeline_common import (
    extract_six,
    read_image_color,
    reset_directory,
    resolve_project_path,
    safe_stem,
)
from pipeline_config import (
    CV_DIR,
    EXTERNAL_RESULT_DIR,
    FEATURE_DIR,
    LABEL_TO_CLASS,
    MODEL_FEATURES,
)
from pipeline_models import (
    batik_score,
    build_models,
    metric_values,
    model_slug,
    per_class_metrics,
)


FEATURE_LABELS = {
    "motif_complexity_score": "Kompleksitas motif",
    "small_contour_count": "Jumlah kontur kecil",
    "glcm_entropy": "GLCM entropy",
    "glcm_homogeneity": "GLCM homogeneity",
    "lbp_entropy": "LBP entropy",
    "fft_peak_ratio": "FFT peak ratio",
}


def save_step_visualization(model_name, row, predicted_label, score_batik, output_dir):
    """Panel enam langkah untuk menjelaskan prediksi satu citra eksternal."""
    image = read_image_color(resolve_project_path(row["path"]))
    if image is None:
        raise OSError(f"Gagal membaca citra eksternal: {row['path']}")
    _, visualization = extract_six(image, want_viz=True)
    predicted_class = LABEL_TO_CLASS[int(predicted_label)]
    correct = predicted_class == row["kelas"]
    confidence = score_batik if predicted_label == 1 else 1 - score_batik
    verdict = "BATIK" if predicted_label == 1 else "NON-BATIK"
    actual = "BATIK" if row["label"] == 1 else "NON-BATIK"
    color = "#2e8b57" if predicted_label == 1 else "#c0392b"

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 1])

    axis = figure.add_subplot(grid[0, 0])
    axis.imshow(image[:, :, ::-1])
    axis.set_title("1. Gambar Asli", fontweight="bold")
    axis.axis("off")

    axis = figure.add_subplot(grid[0, 1])
    axis.imshow(visualization["motif_edge"], cmap="gray")
    axis.set_title("2. Motif Edge\n(kompleksitas dan kontur kecil)", fontweight="bold")
    axis.axis("off")

    axis = figure.add_subplot(grid[0, 2])
    axis.imshow(visualization["lbp"], cmap="inferno")
    axis.set_title("3. Tekstur LBP\n(pola mikro permukaan)", fontweight="bold")
    axis.axis("off")

    axis = figure.add_subplot(grid[1, 0])
    axis.imshow(visualization["fft"], cmap="viridis")
    axis.set_title("4. Spektrum FFT\n(periodisitas/pengulangan pola)", fontweight="bold")
    axis.axis("off")

    axis = figure.add_subplot(grid[1, 1])
    axis.axis("off")
    axis.set_title("5. Nilai Enam Fitur", fontweight="bold")
    feature_lines = []
    for feature in MODEL_FEATURES:
        value = float(row[feature])
        formatted = f"{value:.1f}" if abs(value) >= 100 else f"{value:.4f}"
        feature_lines.append(f"{FEATURE_LABELS[feature]:<24}: {formatted}")
    axis.text(
        0.02, 0.90, "\n".join(feature_lines), va="top", fontsize=10.5,
        family="monospace",
    )

    axis = figure.add_subplot(grid[1, 2])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("6. Keputusan Model", fontweight="bold")
    axis.add_patch(plt.Rectangle((0.05, 0.43), 0.90, 0.42, color=color, alpha=0.14))
    axis.text(0.5, 0.75, "VONIS", ha="center", fontsize=11, color="#555555")
    axis.text(0.5, 0.61, verdict, ha="center", fontsize=27, fontweight="bold", color=color)
    axis.text(0.5, 0.49, f"Keyakinan: {confidence * 100:.1f}%", ha="center", fontsize=11)
    axis.add_patch(plt.Rectangle((0.10, 0.22), 0.80, 0.07, color="#dddddd"))
    axis.add_patch(plt.Rectangle((0.10, 0.22), 0.80 * score_batik, 0.07, color=color))
    axis.text(0.10, 0.15, "0 (non-batik)", fontsize=8)
    axis.text(0.90, 0.15, "1 (batik)", fontsize=8, ha="right")
    axis.text(0.5, 0.31, f"skor batik = {score_batik:.3f}", ha="center", fontsize=9)
    status = "BENAR" if correct else "SALAH"
    axis.text(0.5, 0.05, f"Aktual: {actual} | {status}", ha="center", fontsize=10)

    figure.suptitle(
        f"Uji Eksternal — {model_name} → {verdict} ({confidence * 100:.1f}%)",
        fontsize=15, fontweight="bold", color=color,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    filename = (
        f"{row['kelas']}__{safe_stem(row['subjenis'])}__{row['source_id']}.png"
    )
    figure.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "path": row["path"],
        "source_id": row["source_id"],
        "kelas_aktual": row["kelas"],
        "kelas_prediksi": predicted_class,
        "score_batik": score_batik,
        "confidence": confidence,
        "correct": correct,
        "visualization_file": filename,
    }


def main() -> None:
    train_path = FEATURE_DIR / "final_train_features.csv"
    external_path = FEATURE_DIR / "external_features.csv"
    best_path = CV_DIR / "best_cv_model.json"
    if not train_path.exists() or not external_path.exists() or not best_path.exists():
        raise FileNotFoundError("Jalankan tahap 05 dan 07 dahulu.")
    reset_directory(EXTERNAL_RESULT_DIR)
    train = pd.read_csv(train_path)
    external = pd.read_csv(external_path)
    best_cv = json.loads(best_path.read_text(encoding="utf-8"))
    x_train = train[MODEL_FEATURES].to_numpy(float)
    y_train = train["label"].to_numpy(int)
    x_external = external[MODEL_FEATURES].to_numpy(float)
    y_external = external["label"].to_numpy(int)
    rows = []
    selected_model = None
    selected_prediction = None
    selected_score = None

    for model_name, estimator in build_models().items():
        model = clone(estimator)
        model.fit(x_train, y_train)
        prediction = model.predict(x_external).astype(int)
        score = batik_score(model, x_external)
        metrics = metric_values(y_external, prediction)
        slug = model_slug(model_name)
        joblib.dump(model, EXTERNAL_RESULT_DIR / f"model_{slug}.joblib")
        output = external.copy()
        output["model"] = model_name
        output["predicted_label"] = prediction
        output["predicted_class"] = [LABEL_TO_CLASS[value] for value in prediction]
        output["score_batik"] = score
        output["correct"] = output["label"] == output["predicted_label"]
        output.to_csv(EXTERNAL_RESULT_DIR / f"predictions_{slug}.csv", index=False)
        per_class_metrics(y_external, prediction).to_csv(
            EXTERNAL_RESULT_DIR / f"metrics_per_class_{slug}.csv", index=False
        )
        rows.append({"model": model_name, **metrics})
        if model_name == best_cv["model"]:
            selected_model = model
            selected_prediction = prediction
            selected_score = score

        matrix = confusion_matrix(y_external, prediction, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(6, 5.3))
        ConfusionMatrixDisplay(matrix, display_labels=["Non-Batik", "Batik"]).plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d"
        )
        axis.set_title(f"Uji Eksternal — {model_name}")
        figure.tight_layout()
        figure.savefig(
            EXTERNAL_RESULT_DIR / f"confusion_matrix_{slug}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(figure)

    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(EXTERNAL_RESULT_DIR / "external_model_summary.csv", index=False)
    formal = summary.loc[summary["model"] == best_cv["model"]].iloc[0]
    result = {
        **best_cv,
        "external_accuracy": float(formal["accuracy"]),
        "external_balanced_accuracy": float(formal["balanced_accuracy"]),
        "external_f1_macro": float(formal["f1_macro"]),
        "external_mcc": float(formal["mcc"]),
        "external_recall_batik": float(formal["recall_batik"]),
        "external_recall_non_batik": float(formal["recall_non_batik"]),
        "note": "Formal model is selected by CV only; external results never change selection.",
    }
    (EXTERNAL_RESULT_DIR / "selected_model_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if selected_model is None:
        raise RuntimeError("Model formal dari CV tidak ditemukan pada daftar model.")
    visualization_dir = EXTERNAL_RESULT_DIR / "visualisasi_step_by_step"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    visualization_rows = []
    for index, row in enumerate(external.to_dict("records")):
        visualization_rows.append(
            save_step_visualization(
                best_cv["model"], row,
                int(selected_prediction[index]), float(selected_score[index]),
                visualization_dir,
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(external):
            print(f"  visualisasi eksternal: {index + 1}/{len(external)}")
    pd.DataFrame(visualization_rows).to_csv(
        visualization_dir / "visualization_index.csv", index=False
    )
    print("=" * 72)
    print("TAHAP 08 — FINAL TRAIN DAN UJI EKSTERNAL")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nModel formal dari CV: {best_cv['model']}")
    print(f"Visualisasi step-by-step: {visualization_dir}")
    print(f"Hasil: {EXTERNAL_RESULT_DIR}")


if __name__ == "__main__":
    main()
