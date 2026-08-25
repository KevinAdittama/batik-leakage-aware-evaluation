"""Analisis kesalahan eksternal model formal berdasarkan subjenis."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from pipeline_common import reset_directory
from pipeline_config import ERROR_DIR, EXTERNAL_RESULT_DIR


def main() -> None:
    result_path = EXTERNAL_RESULT_DIR / "selected_model_result.json"
    if not result_path.exists():
        raise FileNotFoundError("Jalankan 08_train_external.py dahulu.")
    selected = json.loads(result_path.read_text(encoding="utf-8"))
    predictions_path = EXTERNAL_RESULT_DIR / f"predictions_{selected['model_slug']}.csv"
    predictions = pd.read_csv(predictions_path)
    required = {
        "source_id", "kelas", "label", "subjenis", "predicted_label",
        "score_batik", "correct",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Kolom prediksi wajib hilang: {missing}")
    if predictions["source_id"].duplicated().any():
        raise AssertionError("source_id prediksi eksternal harus unik.")
    if (
        not predictions["label"].isin([0, 1]).all()
        or not predictions["predicted_label"].isin([0, 1]).all()
    ):
        raise AssertionError("Label eksternal harus biner 0/1.")
    if (
        predictions["score_batik"].isna().any()
        or not predictions["score_batik"].between(0, 1).all()
    ):
        raise AssertionError("score_batik harus lengkap dan berada pada [0, 1].")
    predictions["correct"] = predictions["correct"].astype(bool)
    recomputed_correct = predictions["label"].eq(predictions["predicted_label"])
    if not predictions["correct"].equals(recomputed_correct):
        raise AssertionError("Kolom correct tidak cocok dengan label dan prediksi.")
    predictions["error_type"] = "benar"
    predictions.loc[
        (predictions["label"] == 1) & (~predictions["correct"]), "error_type"
    ] = "batik_menjadi_non_batik"
    predictions.loc[
        (predictions["label"] == 0) & (~predictions["correct"]), "error_type"
    ] = "non_batik_menjadi_batik"

    summary = (
        predictions.groupby(["kelas", "subjenis"])
        .agg(n=("correct", "size"), benar=("correct", "sum"))
        .reset_index()
    )
    summary["salah"] = summary["n"] - summary["benar"]
    summary["error_rate"] = summary["salah"] / summary["n"]
    summary["recall_subjenis"] = summary["benar"] / summary["n"]
    summary = summary.sort_values(["error_rate", "n"], ascending=[False, False])
    wrong = predictions.loc[~predictions["correct"]].copy()
    matrix = confusion_matrix(
        predictions["label"], predictions["predicted_label"], labels=[0, 1]
    )
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    if int(summary["n"].sum()) != len(predictions):
        raise AssertionError("Total denominator subjenis tidak cocok dengan total prediksi.")
    if int(summary["salah"].sum()) != false_negative + false_positive:
        raise AssertionError("Total error subjenis tidak cocok dengan confusion matrix.")
    batik_errors = int(summary.loc[summary["kelas"].eq("batik"), "salah"].sum())
    non_batik_errors = int(
        summary.loc[summary["kelas"].eq("non_batik"), "salah"].sum()
    )
    if batik_errors != false_negative or non_batik_errors != false_positive:
        raise AssertionError("Arah error subjenis tidak cocok dengan confusion matrix.")
    reset_directory(ERROR_DIR)
    summary.to_csv(ERROR_DIR / "error_by_subtype.csv", index=False)
    wrong.to_csv(ERROR_DIR / "misclassified_images.csv", index=False)

    affected = summary.loc[summary["salah"] > 0].sort_values("error_rate")
    if not affected.empty:
        labels = affected["kelas"] + "/" + affected["subjenis"]
        figure, axis = plt.subplots(figsize=(9, max(4.5, len(affected) * 0.45)))
        bars = axis.barh(labels, affected["error_rate"], color="#b24c3d")
        axis.bar_label(
            bars,
            labels=[
                f"{int(salah)}/{int(total)}"
                for salah, total in zip(affected["salah"], affected["n"])
            ],
            padding=3,
        )
        axis.set_xlim(0, 1.05)
        axis.set_xlabel("Error rate")
        axis.set_title(f"Kesalahan per Subjenis — {selected['model']}")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        figure.savefig(ERROR_DIR / "error_rate_by_subtype.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    if false_negative != int(
        (wrong["error_type"] == "batik_menjadi_non_batik").sum()
    ):
        raise AssertionError("Jumlah false negative tidak konsisten.")
    if false_positive != int(
        (wrong["error_type"] == "non_batik_menjadi_batik").sum()
    ):
        raise AssertionError("Jumlah false positive tidak konsisten.")
    lines = [
        "# Analisis Error Eksternal",
        "",
        f"- Model formal: **{selected['model']}**.",
        f"- Total uji: **{len(predictions)}**.",
        f"- Salah prediksi: **{len(wrong)}**.",
        f"- Batik → non-batik: **{false_negative}**.",
        f"- Non-batik → batik: **{false_positive}**.",
        "",
        "## Subjenis yang mengalami kesalahan",
        "",
    ]
    if affected.empty:
        lines.append("Tidak ada kesalahan.")
    else:
        for row in affected.sort_values("error_rate", ascending=False).to_dict("records"):
            lines.append(
                f"- `{row['kelas']}/{row['subjenis']}`: {row['salah']}/{row['n']} "
                f"salah ({row['error_rate']:.1%})."
            )
    lines.extend(
        [
            "",
            "Subjenis hanya dipakai untuk analisis; label selalu berasal dari folder kelas.",
            "",
        ]
    )
    (ERROR_DIR / "error_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("=" * 72)
    print(f"TAHAP 10 — ANALISIS ERROR ({selected['model']})")
    print("=" * 72)
    print(f"Total={len(predictions)} | salah={len(wrong)} | FN batik={false_negative} | FP={false_positive}")
    if not affected.empty:
        print(affected.sort_values("error_rate", ascending=False).to_string(index=False))
    print(f"\nHasil: {ERROR_DIR}")


if __name__ == "__main__":
    main()
