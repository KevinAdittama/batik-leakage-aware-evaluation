"""5-fold CV fold-safe untuk tiga model dan empat kelompok fitur."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from pipeline_common import reset_directory
from pipeline_config import (
    CV_DIR,
    FEATURE_DIR,
    FEATURE_GROUPS,
    LABEL_TO_CLASS,
    N_SPLITS,
    RANDOM_SEED,
    TARGET_PER_KELAS,
)
from pipeline_models import batik_score, build_models, metric_values, model_slug


METRICS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
    "f1_macro", "mcc", "recall_non_batik", "recall_batik",
]


def balanced_fold(pool: pd.DataFrame, seed: int):
    pieces = []
    for class_name in ("batik", "non_batik"):
        group = pool.loc[pool["kelas"] == class_name]
        original = group.loc[~group["is_augmented"].astype(bool)]
        augmented = group.loc[group["is_augmented"].astype(bool)]
        needed = TARGET_PER_KELAS - len(original)
        if needed < 0:
            raise RuntimeError("Target lebih kecil dari jumlah citra asli fold.")
        if needed > len(augmented):
            raise RuntimeError(f"CV pool {class_name} tidak cukup untuk train fold.")
        selected = augmented.sample(n=needed, random_state=seed)
        pieces.append(pd.concat([original, selected], ignore_index=True))
    output = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed)
    counts = output.groupby("kelas").size().to_dict()
    if counts != {"batik": TARGET_PER_KELAS, "non_batik": TARGET_PER_KELAS}:
        raise AssertionError(f"Train fold tidak seimbang: {counts}")
    return output.reset_index(drop=True)


def prepare_folds(original: pd.DataFrame, pool: pd.DataFrame):
    # Grain pemisahan adalah group_id, yaitu identitas foto sumber, bukan
    # identitas berkas. Dua potongan dari satu foto berbagi group_id sehingga
    # tidak dapat jatuh di sisi latih dan sisi uji sekaligus (tahap 24).
    splitter = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    y = original["label"].to_numpy(int)
    groups = original["group_id"].to_numpy()
    definitions = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(original, y, groups), 1
    ):
        train_groups = set(original.iloc[train_index]["group_id"])
        validation_groups = set(original.iloc[validation_index]["group_id"])
        if train_groups & validation_groups:
            raise AssertionError("Group ID bocor antar-fold.")
        train_sources = set(original.iloc[train_index]["source_id"])
        validation_sources = set(original.iloc[validation_index]["source_id"])
        if train_sources & validation_sources:
            raise AssertionError("Source ID bocor antar-fold.")
        candidates = pool.loc[pool["source_id"].isin(train_sources)].copy()
        if set(candidates["source_id"]) & validation_sources:
            raise AssertionError("Turunan validation masuk train fold.")
        training = balanced_fold(candidates, RANDOM_SEED + fold)
        definitions.append((fold, validation_index, training))
        print(
            f"  fold {fold}: train=400 (200+200), "
            f"validation asli={len(validation_index)}"
        )
    return definitions


def main() -> None:
    original_path = FEATURE_DIR / "development_original_features.csv"
    pool_path = FEATURE_DIR / "cv_pool_features.csv"
    if not original_path.exists() or not pool_path.exists():
        raise FileNotFoundError("Jalankan 05_extract_features.py dahulu.")
    reset_directory(CV_DIR)
    original = pd.read_csv(original_path).reset_index(drop=True)
    pool = pd.read_csv(pool_path)
    folds = prepare_folds(original, pool)
    models = build_models()
    metric_rows, prediction_rows = [], []

    for feature_set, features in FEATURE_GROUPS.items():
        all_x = original[features].to_numpy(float)
        all_y = original["label"].to_numpy(int)
        for model_name, estimator in models.items():
            oof_pred = np.full(len(original), -1, dtype=int)
            oof_score = np.full(len(original), np.nan)
            for fold, validation_index, training in folds:
                model = clone(estimator)
                model.fit(training[features].to_numpy(float), training["label"].to_numpy(int))
                prediction = model.predict(all_x[validation_index]).astype(int)
                score = batik_score(model, all_x[validation_index])
                oof_pred[validation_index] = prediction
                oof_score[validation_index] = score
                metric_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_features": len(features),
                        "train_batik": TARGET_PER_KELAS,
                        "train_non_batik": TARGET_PER_KELAS,
                        "validation_original": len(validation_index),
                        **metric_values(all_y[validation_index], prediction),
                    }
                )
            prediction_frame = original[["path", "source_id", "kelas", "label", "subjenis"]].copy()
            prediction_frame["feature_set"] = feature_set
            prediction_frame["model"] = model_name
            prediction_frame["predicted_label"] = oof_pred
            prediction_frame["predicted_class"] = [LABEL_TO_CLASS[value] for value in oof_pred]
            prediction_frame["score_batik"] = oof_score
            prediction_frame["correct"] = prediction_frame["label"] == oof_pred
            prediction_rows.append(prediction_frame)

    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(CV_DIR / "cv_fold_metrics.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(CV_DIR / "cv_oof_predictions.csv", index=False)

    summary_rows = []
    for (feature_set, model_name), group in fold_metrics.groupby(
        ["feature_set", "model"], sort=False
    ):
        row = {"feature_set": feature_set, "model": model_name}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["feature_set", "f1_macro_mean"], ascending=[True, False]
    )
    summary.to_csv(CV_DIR / "cv_summary_all_feature_sets.csv", index=False)
    primary = summary.loc[summary["feature_set"] == "Gabungan 6 Fitur"].sort_values(
        "f1_macro_mean", ascending=False
    )
    primary.to_csv(CV_DIR / "cv_summary_primary.csv", index=False)
    best = primary.iloc[0]
    (CV_DIR / "best_cv_model.json").write_text(
        json.dumps(
            {
                "model": best["model"],
                "model_slug": model_slug(best["model"]),
                "feature_set": "Gabungan 6 Fitur",
                "cv_f1_macro_mean": float(best["f1_macro_mean"]),
                "cv_f1_macro_std": float(best["f1_macro_std"]),
                "selection_rule": "Highest mean macro-F1 among three models using all six features",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pivot = summary.pivot(index="feature_set", columns="model", values="f1_macro_mean")
    axis = pivot.plot(kind="bar", figsize=(11, 6), color=["#35618f", "#d9822b", "#4f8f59"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro-F1 CV")
    axis.set_xlabel("")
    axis.set_title("Ablasi Domain Fitur dan Perbandingan Model")
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(CV_DIR / "feature_ablation_cv.png", dpi=300, bbox_inches="tight")
    plt.close()

    for model_name in models:
        subset = predictions.query(
            "feature_set == 'Gabungan 6 Fitur' and model == @model_name"
        )
        matrix = confusion_matrix(subset["label"], subset["predicted_label"], labels=[0, 1])
        figure, axis = plt.subplots(figsize=(6, 5.3))
        ConfusionMatrixDisplay(matrix, display_labels=["Non-Batik", "Batik"]).plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d"
        )
        axis.set_title(f"OOF Confusion Matrix — {model_name}")
        figure.tight_layout()
        figure.savefig(
            CV_DIR / f"oof_confusion_matrix_{model_slug(model_name)}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(figure)

    print("=" * 72)
    print("TAHAP 07 — 5-FOLD CV DAN ABLASI FITUR")
    print("=" * 72)
    display_columns = [
        "model", "accuracy_mean", "balanced_accuracy_mean", "f1_macro_mean",
        "mcc_mean", "recall_batik_mean", "recall_non_batik_mean",
    ]
    print(primary[display_columns].to_string(index=False))
    print(f"\nModel terbaik CV: {best['model']} | F1={best['f1_macro_mean']:.3f}")
    print(f"Hasil: {CV_DIR}")


if __name__ == "__main__":
    main()
