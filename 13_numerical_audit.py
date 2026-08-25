"""Audit numerik kritis untuk revisi IJIES tanpa melatih ulang model.

Tahap ini membaca artefak prediksi/manifes yang sudah ada, merekonsiliasi
Figure 8a, confusion matrix, metrik eksternal, bootstrap berpasangan, serta
komposisi fold. Semua keluaran baru ditulis ke paket audit utama agar
hasil eksperimen lama tidak ditimpa.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from pipeline_config import N_SPLITS, PROJECT_DIR, RANDOM_SEED, TARGET_PER_KELAS


OUTPUT_DIR = (
    PROJECT_DIR
    / "IJIES_REVISI_FINAL"
    / "04_Tabel_Manifest_dan_Hasil"
    / "Audit_Numerik_dan_Eksperimen"
    / "outputs"
)
PAPER_REVISION_DIR = (
    PROJECT_DIR / "paper" / "revisi_hasan" / "latex_ijies_q3_submission_revision"
)
WORD_REVISION_DIR = (
    PROJECT_DIR / "paper" / "revisi_hasan" / "word_ijies_q3_submission_revision"
)
PRIMARY_MANUSCRIPT = (
    WORD_REVISION_DIR
    / "siap"
    / "SIAP_SUBMIT_IJIES"
    / "siap"
    / "Leakage-Aware Evaluation Reveals Acquisition Bias and External Degradation in Binary Batik Recognition.docx"
)
N_BOOTSTRAP = 10_000
DISPLAY_DECIMALS = 3
FLOAT_TOLERANCE = 1e-12

PREDICTION_FILES = {
    "Logistic Regression": PROJECT_DIR
    / "hasil_paper/08_uji_eksternal/predictions_logistic_regression.csv",
    "SVM-RBF": PROJECT_DIR
    / "hasil_paper/08_uji_eksternal/predictions_svm_rbf.csv",
    "Random Forest": PROJECT_DIR
    / "hasil_paper/08_uji_eksternal/predictions_random_forest.csv",
    "ResNet18": PROJECT_DIR
    / "hasil_paper/11_deep_learning_baseline/external_predictions_resnet18.csv",
    "MobileNetV2": PROJECT_DIR
    / "hasil_paper/11_deep_learning_baseline/external_predictions_mobilenet_v2.csv",
}


SELECTED_MODEL_FILE = PROJECT_DIR / "hasil_paper/08_uji_eksternal/selected_model_result.json"


def formal_model() -> tuple[str, str]:
    """Model formal menurut aturan seleksi yang berlaku, bukan menurut asumsi.

    Sebelumnya skrip ini menetapkan Random Forest secara harfiah. Setelah fold
    menjadi group-aware, pemenang aturan seleksi berpindah, dan penetapan
    harfiah itu berhenti menjadi penjaga dan berubah menjadi asumsi yang salah.
    Aturan seleksinya sendiri tidak berubah: macro-F1 CV tertinggi pada enam
    fitur gabungan.

    Perlu diingat saat membaca keluaran audit ini: margin antara dua model
    teratas jauh lebih kecil daripada simpangan antar-fold, sehingga identitas
    pemenang tidak stabil. Tahap 25 mengukur ketidakstabilan itu.
    """
    selected = json.loads(SELECTED_MODEL_FILE.read_text(encoding="utf-8"))
    name = selected["model"].replace("SVM (RBF)", "SVM-RBF")
    if name not in PREDICTION_FILES:
        raise AssertionError(f"Model formal tidak dikenal: {selected['model']!r}")
    return name, str(selected["model_slug"])


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom wajib hilang dari {source}: {missing}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Kolom boolean memuat nilai selain true/false.")
    return normalized.eq("true")


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = recall_score(
        y_true, y_pred, labels=[1, 0], average=None, zero_division=0
    )
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "recall_batik": recalls[0],
        "recall_non_batik": recalls[1],
    }


def metrics_from_counts(tp, fn, tn, fp) -> dict[str, np.ndarray]:
    recall_batik = tp / (tp + fn)
    recall_non_batik = tn / (tn + fp)
    precision_batik = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) != 0,
    )
    precision_non_batik = np.divide(
        tn,
        tn + fn,
        out=np.zeros_like(tn, dtype=float),
        where=(tn + fn) != 0,
    )
    f1_batik = np.divide(
        2 * precision_batik * recall_batik,
        precision_batik + recall_batik,
        out=np.zeros_like(precision_batik),
        where=(precision_batik + recall_batik) != 0,
    )
    f1_non_batik = np.divide(
        2 * precision_non_batik * recall_non_batik,
        precision_non_batik + recall_non_batik,
        out=np.zeros_like(precision_non_batik),
        where=(precision_non_batik + recall_non_batik) != 0,
    )
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return {
        "balanced_accuracy": (recall_batik + recall_non_batik) / 2,
        "macro_f1": (f1_batik + f1_non_batik) / 2,
        "mcc": mcc,
        "recall_batik": recall_batik,
        "recall_non_batik": recall_non_batik,
    }


def bootstrap_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, sampled: np.ndarray
) -> dict[str, np.ndarray]:
    truth = y_true[sampled]
    pred = y_pred[sampled]
    tp = np.sum((truth == 1) & (pred == 1), axis=1)
    fn = np.sum((truth == 1) & (pred == 0), axis=1)
    tn = np.sum((truth == 0) & (pred == 0), axis=1)
    fp = np.sum((truth == 0) & (pred == 1), axis=1)
    return metrics_from_counts(tp, fn, tn, fp)


def load_external_predictions() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest_path = PROJECT_DIR / "hasil_paper/01_audit/external_manifest.csv"
    manifest = pd.read_csv(manifest_path).sort_values("source_id").reset_index(drop=True)
    require_columns(
        manifest,
        {"source_id", "path", "kelas", "label", "subjenis", "sha256"},
        manifest_path,
    )
    if manifest["source_id"].duplicated().any():
        raise AssertionError("External manifest memiliki source_id duplikat.")
    if len(manifest) != 60:
        raise AssertionError(f"External manifest seharusnya 60 baris, ditemukan {len(manifest)}.")

    predictions: dict[str, pd.DataFrame] = {}
    required = {
        "source_id",
        "path",
        "kelas",
        "label",
        "subjenis",
        "predicted_label",
        "score_batik",
    }
    for model, path in PREDICTION_FILES.items():
        frame = pd.read_csv(path).sort_values("source_id").reset_index(drop=True)
        require_columns(frame, required, path)
        if frame["source_id"].duplicated().any():
            raise AssertionError(f"Prediksi {model} memiliki source_id duplikat.")
        if frame["source_id"].tolist() != manifest["source_id"].tolist():
            raise AssertionError(f"Cakupan/source order prediksi tidak cocok: {model}")
        for column in ["path", "kelas", "label", "subjenis"]:
            if not frame[column].astype(str).equals(manifest[column].astype(str)):
                raise AssertionError(f"Kolom {column} tidak cocok dengan manifest: {model}")
        if not frame["predicted_label"].isin([0, 1]).all():
            raise AssertionError(f"Predicted label di luar 0/1: {model}")
        if frame["score_batik"].isna().any() or not frame["score_batik"].between(0, 1).all():
            raise AssertionError(f"Score batik invalid: {model}")
        threshold_prediction = frame["score_batik"].ge(0.5).astype(int)
        if not threshold_prediction.equals(frame["predicted_label"].astype(int)):
            raise AssertionError(f"Prediksi tidak konsisten dengan threshold 0.5: {model}")
        predictions[model] = frame
    return manifest, predictions


def external_audit(
    manifest: pd.DataFrame, predictions: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    matrix_rows = []
    for model, frame in predictions.items():
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = frame["predicted_label"].to_numpy(dtype=int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        for true_label, true_name in [(0, "non_batik"), (1, "batik")]:
            for predicted_label, predicted_name in [(0, "non_batik"), (1, "batik")]:
                matrix_rows.append(
                    {
                        "model": model,
                        "true_label": true_label,
                        "true_class": true_name,
                        "predicted_label": predicted_label,
                        "predicted_class": predicted_name,
                        "n": int(matrix[true_label, predicted_label]),
                    }
                )
        metrics = point_metrics(y_true, y_pred)
        metric_rows.append({"model": model, "n": len(frame), **metrics})

    metrics_frame = pd.DataFrame(metric_rows)
    matrices = pd.DataFrame(matrix_rows)

    selected_path = PROJECT_DIR / "hasil_paper/08_uji_eksternal/selected_model_result.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    formal_name, _ = formal_model()
    rf_metrics = metrics_frame.loc[metrics_frame["model"] == formal_name].iloc[0]
    stored_metric_map = {
        "external_balanced_accuracy": "balanced_accuracy",
        "external_f1_macro": "macro_f1",
        "external_mcc": "mcc",
        "external_recall_batik": "recall_batik",
        "external_recall_non_batik": "recall_non_batik",
    }
    for stored_name, recomputed_name in stored_metric_map.items():
        if not np.isclose(
            float(selected[stored_name]),
            float(rf_metrics[recomputed_name]),
            atol=FLOAT_TOLERANCE,
            rtol=0,
        ):
            raise AssertionError(
                f"selected_model_result tidak cocok untuk {stored_name}."
            )

    rf = predictions[formal_name].copy()
    rf["correct"] = rf["label"].eq(rf["predicted_label"])
    rf["error_type"] = "correct"
    rf.loc[(rf["label"] == 1) & ~rf["correct"], "error_type"] = "false_negative"
    rf.loc[(rf["label"] == 0) & ~rf["correct"], "error_type"] = "false_positive"
    authoritative = manifest[
        ["source_id", "path", "sha256", "kelas", "label", "subjenis"]
    ].copy()
    authoritative["predicted_label"] = rf["predicted_label"].to_numpy(dtype=int)
    authoritative["predicted_class"] = np.where(
        authoritative["predicted_label"].eq(1), "batik", "non_batik"
    )
    authoritative["score_batik"] = rf["score_batik"].to_numpy(dtype=float)
    authoritative["correct"] = rf["correct"].to_numpy(dtype=bool)
    authoritative["error_type"] = rf["error_type"].to_numpy()

    subtype = (
        authoritative.groupby(["kelas", "label", "subjenis"], as_index=False)
        .agg(n=("correct", "size"), correct=("correct", "sum"))
        .assign(
            errors=lambda data: data["n"] - data["correct"],
            error_rate=lambda data: data["errors"] / data["n"],
            false_negative=lambda data: np.where(
                data["label"].eq(1), data["errors"], 0
            ),
            false_positive=lambda data: np.where(
                data["label"].eq(0), data["errors"], 0
            ),
        )
        .sort_values(["error_rate", "kelas", "subjenis"], ascending=[False, True, True])
    )
    rf_matrix = confusion_matrix(
        authoritative["label"], authoritative["predicted_label"], labels=[0, 1]
    )
    tn, fp, fn, tp = (int(value) for value in rf_matrix.ravel())
    if int(subtype["n"].sum()) != len(authoritative):
        raise AssertionError("Denominator subtype tidak menjumlah ke ukuran external.")
    if int(subtype["errors"].sum()) != fp + fn:
        raise AssertionError("Total error subtype tidak cocok dengan confusion matrix.")
    if int(subtype["false_negative"].sum()) != fn:
        raise AssertionError("Total false negative subtype tidak cocok.")
    if int(subtype["false_positive"].sum()) != fp:
        raise AssertionError("Total false positive subtype tidak cocok.")
    return metrics_frame, matrices, authoritative, subtype


def bootstrap_audit(
    predictions: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = predictions[formal_model()[0]]
    source_ids = reference["source_id"].astype(str).to_numpy()
    y_true = reference["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    sampled = np.concatenate(
        [
            rng.choice(indices, size=(N_BOOTSTRAP, len(indices)), replace=True)
            for indices in class_indices
        ],
        axis=1,
    )
    np.savez_compressed(
        OUTPUT_DIR / "external_bootstrap_indices.npz",
        sampled_indices=sampled,
        source_ids=source_ids,
        seed=np.array([RANDOM_SEED], dtype=int),
        stratified_labels=np.array([0, 1], dtype=int),
    )

    rows = []
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    for model, frame in predictions.items():
        y_pred = frame["predicted_label"].to_numpy(dtype=int)
        point = point_metrics(y_true, y_pred)
        arrays = bootstrap_arrays(y_true, y_pred, sampled)
        bootstrap[model] = arrays
        for metric, array in arrays.items():
            low, high = np.percentile(array, [2.5, 97.5])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": RANDOM_SEED,
                    "stratified_by_class": True,
                }
            )
    intervals = pd.DataFrame(rows)

    comparison_rows = []
    formal_name, _ = formal_model()
    for first, second in [("ResNet18", formal_name), ("MobileNetV2", formal_name)]:
        differences = bootstrap[first]["macro_f1"] - bootstrap[second]["macro_f1"]
        point_difference = point_metrics(
            y_true, predictions[first]["predicted_label"].to_numpy(dtype=int)
        )["macro_f1"] - point_metrics(
            y_true, predictions[second]["predicted_label"].to_numpy(dtype=int)
        )["macro_f1"]
        low, high = np.percentile(differences, [2.5, 97.5])
        comparison_rows.append(
            {
                "comparison": f"{first} minus {second}",
                "metric": "macro_f1",
                "estimate_difference": point_difference,
                "ci95_low": low,
                "ci95_high": high,
                "paired_on_external_files": True,
                "n_bootstrap": N_BOOTSTRAP,
                "seed": RANDOM_SEED,
            }
        )
    paired = pd.DataFrame(comparison_rows)
    return intervals, paired


def reconcile_existing_bootstrap(
    intervals: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    support = PAPER_REVISION_DIR / "analysis_support"
    existing_intervals = pd.read_csv(support / "external_bootstrap_ci.csv")
    existing_paired = pd.read_csv(support / "paired_bootstrap_differences.csv")
    checks = []
    merged = intervals.merge(
        existing_intervals,
        on=["model", "metric"],
        suffixes=("_audit", "_existing"),
        validate="one_to_one",
    )
    for column in ["estimate", "ci95_low", "ci95_high"]:
        difference = np.abs(merged[f"{column}_audit"] - merged[f"{column}_existing"])
        checks.append(
            {
                "artifact": "external_bootstrap_ci.csv",
                "field": column,
                "max_abs_difference": float(difference.max()),
                "pass": bool((difference <= 5e-12).all()),
            }
        )
    paired_merged = paired.merge(
        existing_paired,
        on=["comparison", "metric"],
        suffixes=("_audit", "_existing"),
        validate="one_to_one",
    )
    for column in ["estimate_difference", "ci95_low", "ci95_high"]:
        difference = np.abs(
            paired_merged[f"{column}_audit"] - paired_merged[f"{column}_existing"]
        )
        checks.append(
            {
                "artifact": "paired_bootstrap_differences.csv",
                "field": column,
                "max_abs_difference": float(difference.max()),
                "pass": bool((difference <= 5e-12).all()),
            }
        )
    result = pd.DataFrame(checks)
    if not result["pass"].all():
        raise AssertionError("Bootstrap audit tidak cocok dengan analysis_support.")
    return result


def load_cv_module():
    path = PROJECT_DIR / "07_eval_5fold.py"
    spec = importlib.util.spec_from_file_location("eval_5fold_audit_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat memuat {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(
        PROJECT_DIR / "hasil_paper/05_fitur/development_original_features.csv"
    ).reset_index(drop=True)
    pool = pd.read_csv(PROJECT_DIR / "hasil_paper/05_fitur/cv_pool_features.csv")
    pool["is_augmented"] = bool_series(pool["is_augmented"])
    cv_module = load_cv_module()
    with contextlib.redirect_stdout(io.StringIO()):
        folds = cv_module.prepare_folds(original, pool)

    summary_rows = []
    source_rows = []
    assignment_rows = []
    validation_fold_by_id = {}
    for fold, validation_index, training in folds:
        validation = original.iloc[validation_index].copy()
        for source_id in validation["source_id"]:
            if source_id in validation_fold_by_id:
                raise AssertionError("Satu source_id masuk lebih dari satu validation fold.")
            validation_fold_by_id[source_id] = fold
        train_source_ids = set(original["source_id"]) - set(validation["source_id"])
        if set(training["source_id"]) - train_source_ids:
            raise AssertionError("Training fold memuat source validation/tidak dikenal.")
        training = training.copy()
        training["is_augmented"] = bool_series(training["is_augmented"])
        candidates = pool.loc[pool["source_id"].isin(train_source_ids)].copy()

        for class_name, class_training in training.groupby("kelas", sort=False):
            originals = class_training.loc[~class_training["is_augmented"]]
            derivatives = class_training.loc[class_training["is_augmented"]]
            class_ids = sorted(originals["source_id"].astype(str).unique())
            if len(originals) != len(class_ids):
                raise AssertionError("Original source muncul lebih dari sekali di training fold.")
            selected_counts = derivatives.groupby("source_id").size().reindex(class_ids, fill_value=0)
            available_counts = (
                candidates.loc[
                    candidates["kelas"].eq(class_name) & candidates["is_augmented"]
                ]
                .groupby("source_id")
                .size()
                .reindex(class_ids, fill_value=0)
            )
            if (selected_counts > available_counts).any():
                raise AssertionError("Turunan terpilih melebihi pool yang tersedia.")
            distribution = selected_counts.value_counts().sort_index().to_dict()
            validation_original = int(validation["kelas"].eq(class_name).sum())
            summary_rows.append(
                {
                    "fold": fold,
                    "kelas": class_name,
                    "validation_original": validation_original,
                    "train_original": len(originals),
                    "train_derivative": len(derivatives),
                    "train_total": len(class_training),
                    "derivatives_per_original_min": int(selected_counts.min()),
                    "derivatives_per_original_max": int(selected_counts.max()),
                    "derivatives_per_original_mean": len(derivatives) / len(originals),
                    "derivative_count_distribution": json.dumps(
                        {str(int(key)): int(value) for key, value in distribution.items()},
                        sort_keys=True,
                    ),
                }
            )
            source_lookup = originals.set_index("source_id")
            for source_id in class_ids:
                row = source_lookup.loc[source_id]
                source_rows.append(
                    {
                        "fold": fold,
                        "kelas": class_name,
                        "source_id": source_id,
                        "source_path": row["source_path"],
                        "original_instances": 1,
                        "selected_derivatives": int(selected_counts[source_id]),
                        "available_derivatives": int(available_counts[source_id]),
                        "total_training_instances": 1 + int(selected_counts[source_id]),
                    }
                )
        counts = training.groupby("kelas").size().to_dict()
        if counts != {"batik": TARGET_PER_KELAS, "non_batik": TARGET_PER_KELAS}:
            raise AssertionError(f"Fold {fold} tidak seimbang: {counts}")

    if len(validation_fold_by_id) != len(original):
        raise AssertionError("Tidak semua original mendapat tepat satu validation fold.")
    manifest = pd.read_csv(PROJECT_DIR / "hasil_paper/01_audit/development_manifest.csv")
    for row in manifest.to_dict("records"):
        assignment_rows.append(
            {
                "source_id": row["source_id"],
                "path": row["path"],
                "sha256": row["sha256"],
                "kelas": row["kelas"],
                "label": row["label"],
                "subjenis": row["subjenis"],
                "validation_fold": validation_fold_by_id[row["source_id"]],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["fold", "kelas"])
    sources = pd.DataFrame(source_rows).sort_values(["fold", "kelas", "source_id"])
    assignments = pd.DataFrame(assignment_rows).sort_values("source_id")
    actual_validation_totals = summary.groupby("fold")[
        "validation_original"
    ].sum().tolist()
    if sum(actual_validation_totals) != len(original) or (
        max(actual_validation_totals) - min(actual_validation_totals) > 1
    ):
        raise AssertionError(
            f"Ukuran validation fold tidak membagi original secara seimbang: {actual_validation_totals}"
        )
    if not summary["train_total"].eq(TARGET_PER_KELAS).all():
        raise AssertionError("Train total per fold/class bukan 200.")
    return summary, sources, assignments


def model_selection_audit() -> pd.DataFrame:
    single_loop = pd.read_csv(
        PROJECT_DIR / "hasil_paper/07_cross_validation/cv_summary_primary.csv"
    ).sort_values("f1_macro_mean", ascending=False)
    nested = pd.read_csv(
        PROJECT_DIR / "hasil_paper/12_submission_robustness/nested_cv_outer_fold_metrics.csv"
    )
    single_winner = single_loop.iloc[0]
    nested_counts = nested["selected_model"].value_counts()
    nested_winner = nested_counts.index[0]
    stored = json.loads(SELECTED_MODEL_FILE.read_text(encoding="utf-8"))
    if single_winner["model"] != stored["model"]:
        raise AssertionError(
            "Pemenang single-loop pada tabel CV tidak sama dengan yang tersimpan: "
            f"tabel={single_winner['model']!r}, tersimpan={stored['model']!r}"
        )
    if int(nested_counts.sum()) != len(nested):
        raise AssertionError("Frekuensi nested selection tidak mencakup semua outer fold.")
    return pd.DataFrame(
        [
            {
                "protocol": "single-loop augmented handcrafted analysis",
                "required_term": "single-loop augmented-analysis winner",
                "model": single_winner["model"],
                "selection_evidence": f"mean macro-F1={single_winner['f1_macro_mean']:.15g}",
            },
            {
                "protocol": "originals-only nested cross-validation",
                "required_term": "most frequently selected model in originals-only nested cross-validation",
                "model": nested_winner,
                "selection_evidence": f"selected in {int(nested_counts.iloc[0])}/{len(nested)} outer folds",
            },
        ]
    )


def table6_source(
    metrics: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    comparison = pd.read_csv(
        PROJECT_DIR
        / "hasil_paper/11_deep_learning_baseline/model_comparison_classical_vs_deep.csv"
    )
    external_lookup = metrics.set_index("model")
    interval_lookup = intervals.loc[intervals["metric"].eq("macro_f1")].set_index("model")
    name_map = {"SVM (RBF)": "SVM-RBF"}
    rows = []
    for row in comparison.to_dict("records"):
        numerical_name = name_map.get(row["model"], row["model"])
        metric = external_lookup.loc[numerical_name]
        interval = interval_lookup.loc[numerical_name]
        values = {
            "model_group": row["model_group"],
            "model": row["model"],
            "cv_f1_macro_mean": row["cv_f1_macro_mean"],
            "cv_f1_macro_std": row["cv_f1_macro_std"],
            "external_f1_macro": metric["macro_f1"],
            "external_f1_ci95_low": interval["ci95_low"],
            "external_f1_ci95_high": interval["ci95_high"],
            "external_mcc": metric["mcc"],
            "external_recall_batik": metric["recall_batik"],
            "external_recall_non_batik": metric["recall_non_batik"],
        }
        for key, value in list(values.items()):
            if key not in {"model_group", "model"}:
                values[f"{key}_display"] = f"{float(value):.{DISPLAY_DECIMALS}f}"
        rows.append(values)
    return pd.DataFrame(rows)


def make_figure8a(subtype: pd.DataFrame) -> None:
    affected = subtype.loc[subtype["errors"] > 0].copy().sort_values("error_rate")
    affected["display_label"] = affected.apply(
        lambda row: (
            f"{'B' if row['kelas'] == 'batik' else 'NB'}: "
            f"{str(row['subjenis']).replace('_', ' ')}"
        ),
        axis=1,
    )
    colors = ["#348B5E" if value == "batik" else "#3E75AF" for value in affected["kelas"]]
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    bars = axis.barh(affected["display_label"], affected["error_rate"], color=colors)
    axis.bar_label(
        bars,
        labels=[f"{int(errors)}/{int(total)}" for errors, total in zip(affected["errors"], affected["n"])],
        padding=3,
        fontsize=8,
    )
    axis.set_xlim(0, 0.86)
    axis.set_xlabel(f"External error rate ({formal_model()[0]})")
    axis.set_title("Audited descriptive subtype errors")
    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    fn = int(subtype["false_negative"].sum())
    fp = int(subtype["false_positive"].sum())
    axis.text(
        0.0,
        -0.12,
        f"Subtype numerators: {fn} FN + {fp} FP = {fn + fp} total errors",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.tight_layout()
    slug = formal_model()[1]
    figure.savefig(OUTPUT_DIR / f"figure8a_{slug}_subtype_errors_audited.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"figure8a_{slug}_subtype_errors_audited.pdf", bbox_inches="tight")
    plt.close(figure)


def artifact_inventory() -> pd.DataFrame:
    paths: list[tuple[str, str, Path, str]] = []
    for path in sorted(PROJECT_DIR.glob("*.py")):
        paths.append(("code", "pipeline_or_analysis_script", path, "inspect"))
    critical = [
        ("manifest", "development_clean_manifest", PROJECT_DIR / "hasil_paper/01_audit/development_manifest.csv"),
        ("manifest", "external_clean_manifest", PROJECT_DIR / "hasil_paper/01_audit/external_manifest.csv"),
        ("manifest", "cv_augmentation_origin_manifest", PROJECT_DIR / "hasil_paper/03_augmentasi/cv_pool_manifest.csv"),
        ("manifest", "final_train_origin_manifest", PROJECT_DIR / "hasil_paper/03_augmentasi/final_train_manifest.csv"),
        ("prediction", "formal_model_external_predictions", PREDICTION_FILES[formal_model()[0]]),
        ("prediction", "resnet18_external_predictions", PREDICTION_FILES["ResNet18"]),
        ("prediction", "all_classical_oof_predictions", PROJECT_DIR / "hasil_paper/07_cross_validation/cv_oof_predictions.csv"),
        ("metric", "classical_external_summary", PROJECT_DIR / "hasil_paper/08_uji_eksternal/external_model_summary.csv"),
        ("metric", "deep_external_summary", PROJECT_DIR / "hasil_paper/11_deep_learning_baseline/dl_external_summary.csv"),
        ("metric", "nested_outer_fold_metrics", PROJECT_DIR / "hasil_paper/12_submission_robustness/nested_cv_outer_fold_metrics.csv"),
        ("table", "pipeline_table_source", PROJECT_DIR / "hasil_paper/09_tabel_paper/table_model_comparison.csv"),
        ("table", "paper_external_bootstrap", PAPER_REVISION_DIR / "analysis_support/external_bootstrap_ci.csv"),
        ("table", "paper_paired_bootstrap", PAPER_REVISION_DIR / "analysis_support/paired_bootstrap_differences.csv"),
        ("figure", "figure7_asset", PAPER_REVISION_DIR / "figures/external_confusion_matrices_q3.png"),
        ("figure", "figure8_asset", PAPER_REVISION_DIR / "figures/error_analysis_combined_q3.png"),
        ("generator", "paper_figure_generator", PAPER_REVISION_DIR / "make_q3_figures.py"),
        ("generator", "bootstrap_generator", PAPER_REVISION_DIR / "analysis_support/build_revision_statistics.py"),
        ("generator", "alternate_analysis_support_generator", PAPER_REVISION_DIR / "analysis_support/generate_analysis_support.py"),
        ("generator", "word_table_builder", WORD_REVISION_DIR / "build_word_package.py"),
        ("manuscript", "primary_manuscript", PRIMARY_MANUSCRIPT),
    ]
    for category, role, path in critical:
        paths.append((category, role, path, "critical"))
    rows = []
    for category, role, path, priority in paths:
        exists = path.exists()
        rows.append(
            {
                "category": category,
                "role": role,
                "relative_path": path.relative_to(PROJECT_DIR).as_posix(),
                "priority": priority,
                "exists": exists,
                "bytes": path.stat().st_size if exists else None,
                "modified_time": path.stat().st_mtime if exists else None,
                "sha256": file_sha256(path) if exists else None,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("relative_path").sort_values(
        ["category", "relative_path"]
    )


def dataset_inventory() -> pd.DataFrame:
    rows = []
    manifest_specs = [
        ("development_clean", PROJECT_DIR / "hasil_paper/01_audit/development_manifest.csv"),
        ("external_clean", PROJECT_DIR / "hasil_paper/01_audit/external_manifest.csv"),
        ("cv_pool", PROJECT_DIR / "hasil_paper/03_augmentasi/cv_pool_manifest.csv"),
        ("final_train", PROJECT_DIR / "hasil_paper/03_augmentasi/final_train_manifest.csv"),
    ]
    for dataset, path in manifest_specs:
        frame = pd.read_csv(path)
        path_column = "generated_path" if "generated_path" in frame.columns else "path"
        for (class_name, subtype), group in frame.groupby(["kelas", "subjenis"], dropna=False):
            rows.append(
                {
                    "dataset": dataset,
                    "kelas": class_name,
                    "subjenis": subtype,
                    "records": len(group),
                    "unique_source_ids": group["source_id"].nunique(),
                    "augmented_records": int(bool_series(group["is_augmented"]).sum())
                    if "is_augmented" in group.columns
                    else 0,
                    "path_column": path_column,
                    "manifest": path.relative_to(PROJECT_DIR).as_posix(),
                }
            )
    return pd.DataFrame(rows)


def provenance_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact_or_claim": "Figure 7 external confusion matrices",
                "generator": "paper/revisi_hasan/latex_ijies_q3_submission_revision/make_q3_figures.py::external_confusion_matrices",
                "primary_inputs": "RF and ResNet18 sample-level external prediction CSVs",
                "status": "generated from predictions; labels/totals audited here",
            },
            {
                "artifact_or_claim": f"Figure 8a {formal_model()[0]} subtype errors",
                "generator": "10_analisis_error.py -> make_q3_figures.py::subtype_errors",
                "primary_inputs": f"selected_model_result.json -> predictions_{formal_model()[1]}.csv",
                "status": "current values reconcile; upstream scripts lacked cross-artifact assertions",
            },
            {
                "artifact_or_claim": f"External confusion_matrix_{formal_model()[1]}.png",
                "generator": "08_train_external.py",
                "primary_inputs": "final_train_features.csv and external_features.csv",
                "status": "sample-level predictions reproduce the current persisted matrix",
            },
            {
                "artifact_or_claim": "Table 6 external model comparison",
                "generator": "hard-coded in sections/04_results.tex and build_word_package.py",
                "primary_inputs": "model comparison CSV plus bootstrap CSVs (manual transfer)",
                "status": "not single-source generated; replacement-ready audit table created",
            },
            {
                "artifact_or_claim": "External bootstrap intervals and paired differences",
                "generator": "analysis_support/build_revision_statistics.py",
                "primary_inputs": "five aligned sample-level external prediction CSVs",
                "status": "reproduced with shared stratified indices; indices now stored",
            },
            {
                "artifact_or_claim": "Five-fold training composition",
                "generator": "07_eval_5fold.py::prepare_folds/balanced_fold",
                "primary_inputs": "development_original_features.csv and cv_pool_features.csv",
                "status": "previously not persisted; audit fold/source manifests created",
            },
            {
                "artifact_or_claim": "Model-selection terminology",
                "generator": "07_eval_5fold.py and 12_submission_robustness.py",
                "primary_inputs": "cv_summary_primary.csv and nested_cv_outer_fold_metrics.csv",
                "status": "protocol-specific wording recorded",
            },
        ]
    )


def manuscript_media_audit() -> pd.DataFrame:
    rows = []
    if not PRIMARY_MANUSCRIPT.exists():
        return pd.DataFrame(rows)
    with zipfile.ZipFile(PRIMARY_MANUSCRIPT) as archive:
        for name in sorted(
            value for value in archive.namelist() if value.startswith("word/media/")
        ):
            payload = archive.read(name)
            rows.append(
                {
                    "docx_relative_path": PRIMARY_MANUSCRIPT.relative_to(PROJECT_DIR).as_posix(),
                    "media_entry": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def write_output_hash_manifest() -> None:
    rows = []
    manifest_path = OUTPUT_DIR / "output_manifest_sha256.csv"
    for path in sorted(OUTPUT_DIR.iterdir()):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def write_report(
    metrics: pd.DataFrame,
    subtype: pd.DataFrame,
    paired: pd.DataFrame,
    fold_summary: pd.DataFrame,
    selection: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    formal_name = formal_model()[0]
    resnet_rf = paired.loc[
        paired["comparison"].eq(f"ResNet18 minus {formal_name}")
    ].iloc[0]
    metric_index = metrics.set_index("model")
    rf_macro_f1 = float(metric_index.loc[formal_name, "macro_f1"])
    resnet_macro_f1 = float(metric_index.loc["ResNet18", "macro_f1"])
    tn = int(subtype.loc[subtype["label"].eq(0), "correct"].sum())
    fp = int(subtype.loc[subtype["label"].eq(0), "errors"].sum())
    fn = int(subtype.loc[subtype["label"].eq(1), "errors"].sum())
    tp = int(subtype.loc[subtype["label"].eq(1), "correct"].sum())
    validation_sizes = (
        fold_summary.groupby("fold")["validation_original"].sum().astype(int).tolist()
    )
    fold_lines = []
    for row in fold_summary.to_dict("records"):
        fold_lines.append(
            "| {fold} | {kelas} | {validation_original} | {train_original} | "
            "{train_derivative} | {train_total} | {derivatives_per_original_min}-{derivatives_per_original_max} | "
            "{derivative_count_distribution} |".format(**row)
        )
    lines = [
        "# IJIES critical numerical audit",
        "",
        "## Overall assessment: rerun audited; manuscript still needs synchronized revision",
        "",
        "The post-exclusion sample-level predictions are internally consistent. Table 6 remains manually duplicated in manuscript builders, so the audited replacement source must be used when the manuscript is revised. This audit does not claim that the wider reviewer-requested experiment plan is complete.",
        "",
        "## Figure 8a and external confusion matrix",
        "",
        f"- Formal model for this run: **{formal_name}**, chosen by the CV selection rule.",
        f"- Authoritative {formal_name} prediction rows: {int(subtype['n'].sum())}.",
        f"- Subtype errors: {int(subtype['errors'].sum())} total = {int(subtype['false_negative'].sum())} false negatives + {int(subtype['false_positive'].sum())} false positives.",
        f"- Recomputed {formal_name} confusion matrix (true rows 0/1, predicted columns 0/1): `[[{tn}, {fp}], [{fn}, {tp}]]`.",
        "- Assertions require subtype denominators and error directions to reproduce the confusion matrix exactly.",
        "- The archived 203-image baseline produced 22 RF errors. After the approved two-image exclusion and full rerun, the current 201-image development analysis produces the values above; Figure 8a and its prose must be rebuilt from this audited source.",
        "",
        "## Macro-F1 precision and paired bootstrap",
        "",
        f"- {formal_name} macro-F1: `{rf_macro_f1:.15f}`; ResNet18 macro-F1: `{resnet_macro_f1:.15f}`.",
        f"- Full-precision paired point difference: `{resnet_rf['estimate_difference']:.15f}`.",
        f"- Paired stratified bootstrap 95% CI: `{resnet_rf['ci95_low']:.15f}` to `{resnet_rf['ci95_high']:.15f}`.",
        f"- Display convention: {DISPLAY_DECIMALS} decimals for table/prose values, calculated from unrounded predictions. Therefore the paired difference displays as `{resnet_rf['estimate_difference']:.3f}`; it is not obtained by subtracting the already-rounded table cells.",
        "- The bootstrap audit matches the existing analysis-support CSV values and now stores the exact shared bootstrap indices.",
        "",
        "## Fold composition",
        "",
        "| Fold | Class | Validation originals | Train originals | Derivatives | Total | Derivatives/original min-max | Count distribution |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
        *fold_lines,
        "",
        f"The {'/'.join(map(str, validation_sizes))} values are total original validation-fold sizes, not training-original counts. Every training fold contains all eligible training originals plus sampled derivatives up to 200 instances per class. The class-level derivative burden is intentionally unequal because the clean classes contain different numbers of originals; this remains a methodological sensitivity issue, not evidence that the augmentation plan is final.",
        "",
        "## Model selection terminology",
        "",
        *[
            f"- `{row['required_term']}`: {row['model']} ({row['selection_evidence']})."
            for row in selection.to_dict("records")
        ],
        "",
        "## Inventory and provenance risks",
        "",
        f"- Critical/code artifacts inventoried: {len(inventory)} files with SHA-256.",
        "- `.git` is present as an empty directory and the workspace is not a functioning Git repository; the pre-change snapshot is therefore the available local baseline.",
        "- Table 6 is hard-coded independently in LaTeX and Word builders. It must be generated from `table6_external_model_metrics_audited.csv` (or a common structured source) before finalization.",
        "- Two analysis-support scripts overlap in purpose; `build_revision_statistics.py` produced the currently cited bootstrap column format, while `generate_analysis_support.py` is an alternate path. Consolidation is required.",
        "",
        "## Experiments still not complete",
        "",
        "The approved two-image cross-set pHash exclusion has been applied and rerun. This numerical reconciliation still does not complete expert relabeling, the 37-pair within-development group review, a prospectively frozen external set, acquisition interventions, controlled handcrafted/deep ablations, recent classifiers, or repeated strictly nested augmentation sensitivity analyses.",
        "",
    ]
    (OUTPUT_DIR / "NUMERICAL_AUDIT_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, predictions = load_external_predictions()
    metrics, matrices, authoritative, subtype = external_audit(manifest, predictions)
    intervals, paired = bootstrap_audit(predictions)
    bootstrap_reconciliation = reconcile_existing_bootstrap(intervals, paired)
    fold_summary, fold_sources, fold_assignments = fold_audit()
    selection = model_selection_audit()
    table6 = table6_source(metrics, intervals)
    inventory = artifact_inventory()
    datasets = dataset_inventory()
    provenance = provenance_map()
    manuscript_media = manuscript_media_audit()

    authoritative.to_csv(
        OUTPUT_DIR / f"authoritative_external_predictions_{formal_model()[1]}.csv",
        index=False,
        float_format="%.15g",
    )
    subtype.to_csv(
        OUTPUT_DIR / f"{formal_model()[1]}_subtype_error_counts_audited.csv",
        index=False,
        float_format="%.15g",
    )
    matrices.to_csv(OUTPUT_DIR / "external_confusion_matrices_audited.csv", index=False)
    metrics.to_csv(
        OUTPUT_DIR / "external_metrics_full_precision.csv",
        index=False,
        float_format="%.15g",
    )
    intervals.to_csv(
        OUTPUT_DIR / "external_bootstrap_ci_audited.csv",
        index=False,
        float_format="%.15g",
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_macro_f1_bootstrap_audited.csv",
        index=False,
        float_format="%.15g",
    )
    bootstrap_reconciliation.to_csv(
        OUTPUT_DIR / "existing_bootstrap_reconciliation.csv", index=False
    )
    fold_summary.to_csv(OUTPUT_DIR / "fold_composition_summary.csv", index=False)
    fold_sources.to_csv(OUTPUT_DIR / "fold_source_derivative_counts.csv", index=False)
    fold_assignments.to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "model_selection_protocols.csv", index=False)
    table6.to_csv(
        OUTPUT_DIR / "table6_external_model_metrics_audited.csv",
        index=False,
        float_format="%.15g",
    )
    inventory.to_csv(OUTPUT_DIR / "artifact_inventory.csv", index=False)
    datasets.to_csv(OUTPUT_DIR / "dataset_inventory.csv", index=False)
    provenance.to_csv(OUTPUT_DIR / "provenance_map.csv", index=False)
    manuscript_media.to_csv(OUTPUT_DIR / "manuscript_media_inventory.csv", index=False)
    make_figure8a(subtype)
    write_report(metrics, subtype, paired, fold_summary, selection, inventory)

    formal_name = formal_model()[0]
    rf_rows = matrices.loc[matrices["model"].eq(formal_name)]
    rf_matrix = (
        rf_rows.pivot(index="true_label", columns="predicted_label", values="n")
        .sort_index()
        .sort_index(axis=1)
        .astype(int)
        .to_numpy()
        .tolist()
    )

    summary = {
        "status": "passed",
        "formal_model": formal_name,
        "formal_model_confusion_matrix": rf_matrix,
        "formal_model_errors": int(subtype["errors"].sum()),
        "formal_model_false_negative": int(subtype["false_negative"].sum()),
        "formal_model_false_positive": int(subtype["false_positive"].sum()),
        "resnet18_minus_formal_model_macro_f1": float(
            paired.loc[
                paired["comparison"].eq(f"ResNet18 minus {formal_name}"),
                "estimate_difference",
            ].iloc[0]
        ),
        "display_decimals": DISPLAY_DECIMALS,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "output_dir": OUTPUT_DIR.relative_to(PROJECT_DIR).as_posix(),
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_output_hash_manifest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
