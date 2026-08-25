"""Baseline deep learning: ResNet18 dan MobileNetV2 transfer learning.

Protokol:
- Backbone ImageNet dibekukan sebagai feature extractor.
- Classifier head linear/logistic dilatih ulang pada training fold saja.
- Training fold memakai sumber asli train + augmentasi turunan train saja.
- Validation fold dan uji eksternal selalu memakai citra asli.
- Fold mengikuti StratifiedKFold seed yang sama dengan 07_eval_5fold.py.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from pipeline_common import reset_directory, resolve_project_path
from pipeline_config import (
    AUDIT_DIR,
    AUGMENTATION_DIR,
    CV_DIR,
    DEEP_LEARNING_DIR,
    EXTERNAL_RESULT_DIR,
    LABEL_TO_CLASS,
    N_SPLITS,
    PROJECT_DIR,
    RANDOM_SEED,
    TARGET_PER_KELAS,
)
from pipeline_models import metric_values, model_slug, per_class_metrics


ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 24
NUM_WORKERS = 0
MODEL_SPECS = {
    "ResNet18": {
        "slug": "resnet18",
        "weights_name": "IMAGENET1K_V1",
        "description": "ResNet18 ImageNet frozen feature extractor + logistic head",
    },
    "MobileNetV2": {
        "slug": "mobilenet_v2",
        "weights_name": "IMAGENET1K_V2",
        "description": "MobileNetV2 ImageNet frozen feature extractor + logistic head",
    },
}
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mcc",
    "recall_non_batik",
    "recall_batik",
]


class Tee:
    """Simpan log ke file sekaligus tampilkan ke terminal."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="")

    def write(self, value: str) -> None:
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        self.handle.write(value)
        self.handle.flush()

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return str(path), tensor


def set_reproducible(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def balanced_fold(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    pieces = []
    pool = pool.copy()
    pool["is_augmented"] = bool_series(pool["is_augmented"])
    for class_name in ("batik", "non_batik"):
        group = pool.loc[pool["kelas"] == class_name]
        original = group.loc[~group["is_augmented"]]
        augmented = group.loc[group["is_augmented"]]
        needed = TARGET_PER_KELAS - len(original)
        if needed < 0:
            raise RuntimeError("Target lebih kecil dari jumlah citra asli fold.")
        if needed > len(augmented):
            raise RuntimeError(f"CV pool {class_name} tidak cukup untuk train fold.")
        selected = augmented.sample(n=needed, random_state=seed)
        pieces.append(pd.concat([original, selected], ignore_index=True))
    output = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed)
    counts = output.groupby("kelas").size().to_dict()
    expected = {"batik": TARGET_PER_KELAS, "non_batik": TARGET_PER_KELAS}
    if counts != expected:
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
        if set(original.iloc[train_index]["group_id"]) & set(
            original.iloc[validation_index]["group_id"]
        ):
            raise AssertionError("Group ID bocor antar-fold.")
        train_sources = set(original.iloc[train_index]["source_id"])
        validation_sources = set(original.iloc[validation_index]["source_id"])
        if train_sources & validation_sources:
            raise AssertionError("Source ID bocor antar-fold.")
        candidates = pool.loc[pool["source_id"].isin(train_sources)].copy()
        candidates["pool_row_id"] = candidates.index
        if set(candidates["source_id"]) & validation_sources:
            raise AssertionError("Turunan validation masuk train fold.")
        training = balanced_fold(candidates, RANDOM_SEED + fold)
        definitions.append((fold, validation_index, training))
        print(
            f"  fold {fold}: train=400 (200+200), "
            f"validation asli={len(validation_index)}"
        )
    return definitions


def build_feature_extractor(model_name: str):
    if model_name == "ResNet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        transform = weights.transforms()
        backbone = models.resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        embedding_dim = 512
    elif model_name == "MobileNetV2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        transform = weights.transforms()
        backbone = models.mobilenet_v2(weights=weights)
        backbone.classifier = torch.nn.Identity()
        embedding_dim = 1280
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone, transform, embedding_dim, weights


def resolve_paths(frame: pd.DataFrame, column: str) -> list[Path]:
    paths = [resolve_project_path(value) for value in frame[column].tolist()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Ada {len(missing)} citra tidak ditemukan. Contoh: {missing[0]}")
    return paths


@torch.inference_mode()
def extract_embeddings(
    model: torch.nn.Module,
    transform,
    frame: pd.DataFrame,
    path_column: str,
    device: torch.device,
    label: str,
) -> np.ndarray:
    paths = resolve_paths(frame, path_column)
    unique_paths = list(dict.fromkeys(paths))
    dataset = ImagePathDataset(unique_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    model.to(device)
    feature_map: dict[str, np.ndarray] = {}
    print(f"  ekstraksi embedding {label}: {len(unique_paths)} file unik")
    for batch_paths, batch in loader:
        batch = batch.to(device)
        features = model(batch).detach().cpu().numpy()
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        for path, vector in zip(batch_paths, features):
            feature_map[str(Path(path))] = vector.astype(np.float32)
    return np.vstack([feature_map[str(path)] for path in paths])


def build_head(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=seed,
                    class_weight=None,
                ),
            ),
        ]
    )


def batik_probability(model: Pipeline, values: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(values)
    class_index = list(classifier.classes_).index(1)
    return probabilities[:, class_index]


def save_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5.3))
    ConfusionMatrixDisplay(matrix, display_labels=["Non-Batik", "Batik"]).plot(
        ax=axis, cmap="Blues", colorbar=False, values_format="d"
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_prediction_frame(
    frame: pd.DataFrame,
    model_name: str,
    predicted: np.ndarray,
    scores: np.ndarray,
    split: str,
) -> pd.DataFrame:
    columns = ["path", "source_id", "kelas", "label", "subjenis"]
    available = [column for column in columns if column in frame.columns]
    output = frame[available].copy()
    output["split"] = split
    output["model"] = model_name
    output["predicted_label"] = predicted.astype(int)
    output["predicted_class"] = [LABEL_TO_CLASS[value] for value in predicted.astype(int)]
    output["score_batik"] = scores
    output["correct"] = output["label"].to_numpy(int) == predicted.astype(int)
    return output


def write_markdown_table(frame: pd.DataFrame, output_path: Path) -> None:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    data = comparison.copy()
    data["label"] = data["model"]
    colors = ["#35618f" if kind == "Classical ML" else "#d9822b" for kind in data["model_group"]]
    x = np.arange(len(data))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(x - width / 2, data["cv_f1_macro_mean"], width, label="CV macro-F1", color=colors)
    axis.bar(
        x + width / 2,
        data["external_f1_macro"],
        width,
        label="External macro-F1",
        color=["#7fa6c9" if kind == "Classical ML" else "#efb36b" for kind in data["model_group"]],
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro-F1")
    axis.set_xticks(x)
    axis.set_xticklabels(data["label"], rotation=20, ha="right")
    axis.set_title("Classical ML vs Frozen Deep Feature Baselines")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_comparison_table(dl_cv: pd.DataFrame, dl_external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    classical_cv_path = CV_DIR / "cv_summary_primary.csv"
    classical_external_path = EXTERNAL_RESULT_DIR / "external_model_summary.csv"
    if classical_cv_path.exists() and classical_external_path.exists():
        classical_cv = pd.read_csv(classical_cv_path)
        classical_external = pd.read_csv(classical_external_path)
        for _, cv_row in classical_cv.iterrows():
            match = classical_external.loc[classical_external["model"] == cv_row["model"]]
            if match.empty:
                continue
            ext = match.iloc[0]
            rows.append(
                {
                    "model_group": "Classical ML",
                    "model": cv_row["model"],
                    "cv_f1_macro_mean": cv_row["f1_macro_mean"],
                    "cv_f1_macro_std": cv_row["f1_macro_std"],
                    "external_f1_macro": ext["f1_macro"],
                    "external_balanced_accuracy": ext["balanced_accuracy"],
                    "external_mcc": ext["mcc"],
                    "external_recall_batik": ext["recall_batik"],
                    "external_recall_non_batik": ext["recall_non_batik"],
                }
            )
    for _, cv_row in dl_cv.iterrows():
        match = dl_external.loc[dl_external["model"] == cv_row["model"]]
        if match.empty:
            continue
        ext = match.iloc[0]
        rows.append(
            {
                "model_group": "Frozen DL features",
                "model": cv_row["model"],
                "cv_f1_macro_mean": cv_row["f1_macro_mean"],
                "cv_f1_macro_std": cv_row["f1_macro_std"],
                "external_f1_macro": ext["f1_macro"],
                "external_balanced_accuracy": ext["balanced_accuracy"],
                "external_mcc": ext["mcc"],
                "external_recall_batik": ext["recall_batik"],
                "external_recall_non_batik": ext["recall_non_batik"],
            }
        )
    return pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)


def write_method_note(output_dir: Path, best_model: dict) -> None:
    note = f"""# Deep-learning baseline note

Baseline deep learning dievaluasi sebagai frozen ImageNet feature extractor,
bukan sebagai model end-to-end yang di-fine-tune penuh. Dua backbone digunakan:
ResNet18 dan MobileNetV2. Untuk setiap fold, backbone dibekukan dan hanya
classifier head linear/logistic dilatih pada training fold yang sudah seimbang.

Aturan anti-kebocoran:

- Split fold mengikuti `StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})`.
- Augmentasi hanya berasal dari source ID training fold.
- Validation fold memakai citra development asli, bukan augmentasi.
- Uji eksternal memakai citra asli dan tidak dipakai untuk model selection.
- StandardScaler pada embedding CNN berada di dalam Pipeline classifier dan
  di-fit hanya pada data training fold.

Model DL terbaik berdasarkan CV macro-F1: **{best_model['model']}**
({best_model['cv_f1_macro_mean']:.3f} ± {best_model['cv_f1_macro_std']:.3f}).

Catatan interpretasi untuk paper: baseline ini menjawab komentar reviewer/dosen
bahwa pembanding deep learning perlu tersedia, tetapi fokus utama paper tetap
pada enam fitur interpretable.
"""
    (output_dir / "paper_method_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    start_time = time.time()
    set_reproducible()
    reset_directory(DEEP_LEARNING_DIR)
    log_path = DEEP_LEARNING_DIR / f"dl_run_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    tee = Tee(log_path)
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        print("=" * 78)
        print("TAHAP 11 — BASELINE DEEP LEARNING")
        print("=" * 78)
        print(f"Output: {DEEP_LEARNING_DIR}")
        print(f"Log   : {log_path}")

        manifest_paths = {
            "development_original": AUDIT_DIR / "development_manifest.csv",
            "cv_pool": AUGMENTATION_DIR / "cv_pool_manifest.csv",
            "final_train": AUGMENTATION_DIR / "final_train_manifest.csv",
            "external": AUDIT_DIR / "external_manifest.csv",
        }
        for name, path in manifest_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Manifest {name} tidak ditemukan: {path}")

        original = pd.read_csv(manifest_paths["development_original"]).reset_index(drop=True)
        cv_pool = pd.read_csv(manifest_paths["cv_pool"])
        final_train = pd.read_csv(manifest_paths["final_train"])
        external = pd.read_csv(manifest_paths["external"]).reset_index(drop=True)

        print("\nJumlah data:")
        print("  development asli:", original.groupby("kelas").size().to_dict())
        print("  cv_pool:", cv_pool.groupby(["kelas", "is_augmented"]).size().to_dict())
        print("  final_train:", final_train.groupby(["kelas", "is_augmented"]).size().to_dict())
        print("  external:", external.groupby("kelas").size().to_dict())

        folds = prepare_folds(original, cv_pool)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {device}")
        print(f"Torch threads: {torch.get_num_threads()}")

        all_fold_metrics = []
        all_oof_predictions = []
        external_summary_rows = []
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "random_seed": RANDOM_SEED,
            "n_splits": N_SPLITS,
            "target_per_class": TARGET_PER_KELAS,
            "batch_size": BATCH_SIZE,
            "method": "Frozen ImageNet feature extractor + StandardScaler + LogisticRegression head",
            "project_dir": str(PROJECT_DIR),
        }

        for model_name, spec in MODEL_SPECS.items():
            print("\n" + "-" * 78)
            print(f"Model: {model_name}")
            print("-" * 78)
            backbone, transform, embedding_dim, weights = build_feature_extractor(model_name)
            environment[f"{spec['slug']}_weights"] = str(weights)
            environment[f"{spec['slug']}_embedding_dim"] = embedding_dim

            original_x = extract_embeddings(
                backbone, transform, original, "path", device, f"{model_name} development asli"
            )
            pool_x = extract_embeddings(
                backbone, transform, cv_pool, "generated_path", device, f"{model_name} cv_pool"
            )
            final_train_x = extract_embeddings(
                backbone, transform, final_train, "generated_path", device, f"{model_name} final_train"
            )
            external_x = extract_embeddings(
                backbone, transform, external, "path", device, f"{model_name} external"
            )

            oof_pred = np.full(len(original), -1, dtype=int)
            oof_score = np.full(len(original), np.nan, dtype=float)

            for fold, validation_index, training in folds:
                train_positions = training["pool_row_id"].to_numpy(int)
                train_x = pool_x[train_positions]
                train_y = training["label"].to_numpy(int)
                val_x = original_x[validation_index]
                val_y = original.iloc[validation_index]["label"].to_numpy(int)

                head = build_head(RANDOM_SEED + fold)
                head.fit(train_x, train_y)
                prediction = head.predict(val_x).astype(int)
                score = batik_probability(head, val_x)
                oof_pred[validation_index] = prediction
                oof_score[validation_index] = score
                metrics = metric_values(val_y, prediction)
                all_fold_metrics.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "embedding_dim": embedding_dim,
                        "train_batik": TARGET_PER_KELAS,
                        "train_non_batik": TARGET_PER_KELAS,
                        "validation_original": len(validation_index),
                        **metrics,
                    }
                )
                print(
                    f"  fold {fold}: f1_macro={metrics['f1_macro']:.3f}, "
                    f"balanced_acc={metrics['balanced_accuracy']:.3f}, "
                    f"mcc={metrics['mcc']:.3f}"
                )

            oof_frame = make_prediction_frame(
                original, model_name, oof_pred, oof_score, "cv_oof_original"
            )
            all_oof_predictions.append(oof_frame)
            save_confusion_matrix(
                original["label"].to_numpy(int),
                oof_pred,
                f"OOF Confusion Matrix — {model_name}",
                DEEP_LEARNING_DIR / f"oof_confusion_matrix_{spec['slug']}.png",
            )

            final_head = build_head(RANDOM_SEED)
            final_head.fit(final_train_x, final_train["label"].to_numpy(int))
            final_model_path = DEEP_LEARNING_DIR / f"classifier_head_{spec['slug']}.joblib"
            joblib.dump(final_head, final_model_path)
            external_pred = final_head.predict(external_x).astype(int)
            external_score = batik_probability(final_head, external_x)
            external_metrics = metric_values(external["label"].to_numpy(int), external_pred)
            external_summary_rows.append(
                {
                    "model": model_name,
                    "embedding_dim": embedding_dim,
                    "train_batik": int((final_train["kelas"] == "batik").sum()),
                    "train_non_batik": int((final_train["kelas"] == "non_batik").sum()),
                    "external_n": len(external),
                    **external_metrics,
                }
            )
            external_predictions = make_prediction_frame(
                external, model_name, external_pred, external_score, "external_original"
            )
            external_predictions.to_csv(
                DEEP_LEARNING_DIR / f"external_predictions_{spec['slug']}.csv",
                index=False,
            )
            per_class_metrics(external["label"].to_numpy(int), external_pred).to_csv(
                DEEP_LEARNING_DIR / f"external_metrics_per_class_{spec['slug']}.csv",
                index=False,
            )
            save_confusion_matrix(
                external["label"].to_numpy(int),
                external_pred,
                f"External Confusion Matrix — {model_name}",
                DEEP_LEARNING_DIR / f"external_confusion_matrix_{spec['slug']}.png",
            )
            print(
                f"  external: f1_macro={external_metrics['f1_macro']:.3f}, "
                f"balanced_acc={external_metrics['balanced_accuracy']:.3f}, "
                f"mcc={external_metrics['mcc']:.3f}"
            )

        fold_metrics = pd.DataFrame(all_fold_metrics)
        fold_metrics.to_csv(DEEP_LEARNING_DIR / "dl_cv_fold_metrics.csv", index=False)
        oof_predictions = pd.concat(all_oof_predictions, ignore_index=True)
        oof_predictions.to_csv(DEEP_LEARNING_DIR / "dl_cv_oof_predictions.csv", index=False)

        summary_rows = []
        for model_name, group in fold_metrics.groupby("model", sort=False):
            row = {"model": model_name}
            for metric in METRICS:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
            summary_rows.append(row)
        dl_cv_summary = pd.DataFrame(summary_rows).sort_values(
            "f1_macro_mean", ascending=False
        )
        dl_cv_summary.to_csv(DEEP_LEARNING_DIR / "dl_cv_summary.csv", index=False)
        write_markdown_table(dl_cv_summary, DEEP_LEARNING_DIR / "dl_cv_summary.md")

        dl_external = pd.DataFrame(external_summary_rows).sort_values(
            "f1_macro", ascending=False
        )
        dl_external.to_csv(DEEP_LEARNING_DIR / "dl_external_summary.csv", index=False)
        write_markdown_table(dl_external, DEEP_LEARNING_DIR / "dl_external_summary.md")

        best = dl_cv_summary.iloc[0].to_dict()
        best_payload = {
            "model": best["model"],
            "model_slug": MODEL_SPECS[best["model"]]["slug"],
            "cv_f1_macro_mean": float(best["f1_macro_mean"]),
            "cv_f1_macro_std": float(best["f1_macro_std"]),
            "selection_rule": "Highest mean macro-F1 among frozen deep feature baselines",
            "note": "Baseline only; formal handcrafted model selection remains defined in stage 07.",
        }
        (DEEP_LEARNING_DIR / "best_dl_baseline.json").write_text(
            json.dumps(best_payload, indent=2), encoding="utf-8"
        )
        write_method_note(DEEP_LEARNING_DIR, best_payload)

        comparison = build_comparison_table(dl_cv_summary, dl_external)
        comparison.to_csv(DEEP_LEARNING_DIR / "model_comparison_classical_vs_deep.csv", index=False)
        write_markdown_table(
            comparison, DEEP_LEARNING_DIR / "model_comparison_classical_vs_deep.md"
        )
        if not comparison.empty:
            plot_comparison(
                comparison, DEEP_LEARNING_DIR / "model_comparison_f1_bar.png"
            )

        environment["elapsed_seconds"] = round(time.time() - start_time, 2)
        (DEEP_LEARNING_DIR / "dl_environment.json").write_text(
            json.dumps(environment, indent=2, default=str), encoding="utf-8"
        )

        print("\n" + "=" * 78)
        print("RINGKASAN BASELINE DEEP LEARNING")
        print("=" * 78)
        print(dl_cv_summary[["model", "f1_macro_mean", "f1_macro_std", "mcc_mean"]].to_string(index=False))
        print("\nExternal:")
        print(dl_external[["model", "f1_macro", "balanced_accuracy", "mcc", "recall_batik", "recall_non_batik"]].to_string(index=False))
        print(f"\nModel DL terbaik CV: {best_payload['model']}")
        print(f"Hasil: {DEEP_LEARNING_DIR}")
        print(f"Log: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
