"""R2.5 acquisition-intervention experiment for the IJIES revision.

This stage does not modify source images or the active 01--12 outputs. It tests
two prespecified canonical re-encoding pipelines on the existing final training
instances and external originals:

1. 512 x 512 lossless PNG, compression level 9.
2. 512 x 512 JPEG, quality 95, 4:4:4 sampling.

For Random Forest and frozen ResNet18, each intervention is evaluated in two
modes:

- fixed_original_model: keep the original fitted classifier and intervene only
  on external inputs;
- refit_on_same_variant: refit the unchanged classifier family on the same
  intervention applied to the fixed 200/class final-training manifest, then
  evaluate the corresponding external intervention.

All variants are reported. The external collection is never used to select a
variant, model, threshold, or hyperparameter. Results remain exploratory because
the external collection was previously inspected and historical JPEG artifacts
cannot be undone by later re-encoding.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from pipeline_common import extract_six, file_sha256, read_image_color, reset_directory, resolve_project_path
from pipeline_config import AUDIT_DIR, AUGMENTATION_DIR, DEEP_LEARNING_DIR, EXTERNAL_RESULT_DIR, LABEL_TO_CLASS, MODEL_FEATURES, RANDOM_SEED, RESULTS_DIR
from pipeline_models import batik_score, build_models, metric_values


OUT = (
    Path(__file__).resolve().parent
    / "IJIES_REVISI_FINAL"
    / "04_Tabel_Manifest_dan_Hasil"
    / "Audit_Numerik_dan_Eksperimen"
    / "R2_5_acquisition_intervention"
)
FINAL_TRAIN_PATH = AUGMENTATION_DIR / "final_train_manifest.csv"
EXTERNAL_MANIFEST_PATH = AUDIT_DIR / "external_manifest.csv"
BOOTSTRAP_REPLICATES = 10_000
CANONICAL_SIZE = 512

VARIANTS = {
    "canonical_png_512": {
        "codec": "png",
        "size": CANONICAL_SIZE,
        "png_compression": 9,
    },
    "canonical_jpeg_q95_444_512": {
        "codec": "jpeg",
        "size": CANONICAL_SIZE,
        "jpeg_quality": 95,
        "jpeg_sampling": "4:4:4",
    },
}

SUBSETS = (
    "all_external",
    "jpeg_original_format",
    "coarsened_acquisition_match",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resize_interpolation(image: np.ndarray, size: int) -> int:
    height, width = image.shape[:2]
    return cv2.INTER_AREA if height >= size and width >= size else cv2.INTER_CUBIC


def encode_intervention(image: np.ndarray, variant: str) -> tuple[np.ndarray, bytes]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown intervention variant: {variant}")
    config = VARIANTS[variant]
    size = int(config["size"])
    resized = cv2.resize(
        image,
        (size, size),
        interpolation=resize_interpolation(image, size),
    )
    if config["codec"] == "png":
        extension = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(config["png_compression"])]
    elif config["codec"] == "jpeg":
        extension = ".jpg"
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(config["jpeg_quality"]),
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ]
    else:
        raise ValueError(f"Unsupported codec: {config['codec']}")
    ok, encoded = cv2.imencode(extension, resized, params)
    if not ok:
        raise OSError(f"OpenCV failed to encode {variant}")
    payload = encoded.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise OSError(f"OpenCV failed to decode the generated {variant} bytes")
    if decoded.shape[:2] != (size, size):
        raise AssertionError(f"Unexpected decoded shape for {variant}: {decoded.shape}")
    return decoded, payload


def transform_path(path_value: str, variant: str) -> tuple[np.ndarray, bytes, Path]:
    path = resolve_project_path(path_value)
    image = read_image_color(path)
    if image is None:
        raise OSError(f"Unreadable image: {path_value}")
    transformed, payload = encode_intervention(image, variant)
    return transformed, payload, path


def feature_table(
    frame: pd.DataFrame,
    path_column: str,
    split: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict] = []
    transform_rows: list[dict] = []
    total = len(frame)
    for position, row in enumerate(frame.itertuples(index=False), 1):
        path_value = str(getattr(row, path_column))
        transformed, payload, source_path = transform_path(path_value, variant)
        features, _ = extract_six(transformed, want_viz=False)
        base = {
            "split": split,
            "row_id": f"{split}:{position:04d}",
            "path": path_value,
            "source_id": str(row.source_id),
            "kelas": str(row.kelas),
            "label": int(row.label),
            "subjenis": str(row.subjenis),
            "variant": variant,
        }
        if hasattr(row, "is_augmented"):
            base["is_augmented"] = bool(row.is_augmented)
        if hasattr(row, "transform"):
            base["augmentation_transform"] = str(row.transform)
        feature_rows.append({**base, **features})
        transform_rows.append(
            {
                **base,
                "original_sha256": file_sha256(source_path),
                "transformed_sha256": sha256_bytes(payload),
                "transformed_bytes": len(payload),
                "transformed_width": int(transformed.shape[1]),
                "transformed_height": int(transformed.shape[0]),
                "codec": VARIANTS[variant]["codec"],
                "parameters_json": json.dumps(VARIANTS[variant], sort_keys=True),
            }
        )
        if position % 50 == 0 or position == total:
            print(f"  {variant} {split}: {position}/{total}")
    features = pd.DataFrame(feature_rows)
    transforms = pd.DataFrame(transform_rows)
    if features[MODEL_FEATURES].isna().any().any():
        raise AssertionError(f"NaN feature produced for {variant} {split}")
    if not np.isfinite(features[MODEL_FEATURES].to_numpy(float)).all():
        raise AssertionError(f"Non-finite feature produced for {variant} {split}")
    if transforms["transformed_sha256"].str.len().ne(64).any():
        raise AssertionError("Invalid transformed SHA-256")
    return features, transforms


def coarsened_match_manifest(external: pd.DataFrame) -> pd.DataFrame:
    frame = external.copy()
    frame["pixel_count"] = frame["width"].astype(int) * frame["height"].astype(int)
    frame["aspect_ratio"] = frame["width"].astype(float) / frame["height"].clip(lower=1).astype(float)
    frame["pixel_bin"] = pd.cut(
        frame["pixel_count"],
        [-1, 100_000, 500_000, np.inf],
        labels=["small", "medium", "large"],
    ).astype(str)
    frame["aspect_bin"] = pd.cut(
        frame["aspect_ratio"],
        [-np.inf, 0.8, 1.25, np.inf],
        labels=["portrait", "near_square", "landscape"],
    ).astype(str)
    frame["stratum"] = (
        frame["extension"].astype(str)
        + "|"
        + frame["pixel_bin"]
        + "|"
        + frame["aspect_bin"]
    )
    frame["selected_coarsened_match"] = False
    frame["available_in_stratum_class"] = 0
    frame["selected_per_class_in_stratum"] = 0
    jpeg = frame.loc[frame["extension"].eq(".jpg")]
    for stratum, group in jpeg.groupby("stratum", sort=True):
        class_counts = group.groupby("kelas").size()
        per_class = int(min(class_counts.get("batik", 0), class_counts.get("non_batik", 0)))
        for class_name in ("batik", "non_batik"):
            class_index = group.loc[group["kelas"].eq(class_name)].sort_values("source_id").index
            frame.loc[class_index, "available_in_stratum_class"] = len(class_index)
            frame.loc[class_index, "selected_per_class_in_stratum"] = per_class
            if per_class:
                frame.loc[class_index[:per_class], "selected_coarsened_match"] = True
    selected = frame.loc[frame["selected_coarsened_match"]]
    counts = selected.groupby("kelas").size().to_dict()
    if counts != {"batik": 14, "non_batik": 14}:
        raise AssertionError(f"Unexpected coarsened-match class counts: {counts}")
    selected_strata = selected.groupby(["stratum", "kelas"]).size().unstack(fill_value=0)
    if not selected_strata["batik"].eq(selected_strata["non_batik"]).all():
        raise AssertionError("Coarsened match is not class-balanced within strata")
    return frame


def subset_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all_external":
        return np.ones(len(frame), dtype=bool)
    if subset == "jpeg_original_format":
        return frame["extension"].eq(".jpg").to_numpy(bool)
    if subset == "coarsened_acquisition_match":
        return frame["selected_coarsened_match"].to_numpy(bool)
    raise KeyError(subset)


def make_prediction_frame(
    external: pd.DataFrame,
    model: str,
    variant: str,
    training_mode: str,
    predicted: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    output = external[
        [
            "source_id",
            "path",
            "kelas",
            "label",
            "subjenis",
            "extension",
            "width",
            "height",
            "pixel_bin",
            "aspect_bin",
            "stratum",
            "selected_coarsened_match",
        ]
    ].copy()
    output["model"] = model
    output["variant"] = variant
    output["training_mode"] = training_mode
    output["condition"] = f"{training_mode}__{variant}"
    output["predicted_label"] = np.asarray(predicted, dtype=int)
    output["predicted_class"] = [LABEL_TO_CLASS[value] for value in output["predicted_label"]]
    output["score_batik"] = np.asarray(scores, dtype=float)
    output["correct"] = output["label"].to_numpy(int) == output["predicted_label"].to_numpy(int)
    if output["source_id"].duplicated().any():
        raise AssertionError("Duplicate external source_id in predictions")
    if not output["score_batik"].between(0, 1).all():
        raise AssertionError("Scores outside [0, 1]")
    return output


def align_existing_predictions(path: Path, external: pd.DataFrame, model: str) -> pd.DataFrame:
    existing = pd.read_csv(path)
    required = {"source_id", "label", "predicted_label", "score_batik"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(f"Missing existing prediction columns in {path}: {sorted(missing)}")
    existing = existing.drop_duplicates("source_id").set_index("source_id")
    expected_ids = external["source_id"].astype(str).tolist()
    if set(existing.index.astype(str)) != set(expected_ids):
        raise AssertionError(f"Existing predictions do not match external manifest: {path}")
    aligned = existing.loc[expected_ids]
    if not np.array_equal(aligned["label"].to_numpy(int), external["label"].to_numpy(int)):
        raise AssertionError(f"Existing labels do not match external manifest: {path}")
    return make_prediction_frame(
        external,
        model,
        "original",
        "baseline_original",
        aligned["predicted_label"].to_numpy(int),
        aligned["score_batik"].to_numpy(float),
    )


def random_forest_conditions(
    train_features: dict[str, pd.DataFrame],
    external_features: dict[str, pd.DataFrame],
    external: pd.DataFrame,
) -> list[pd.DataFrame]:
    predictions = [
        align_existing_predictions(
            EXTERNAL_RESULT_DIR / "predictions_random_forest.csv",
            external,
            "Random Forest",
        )
    ]
    original_model = joblib.load(EXTERNAL_RESULT_DIR / "model_random_forest.joblib")
    estimator = build_models()["Random Forest"]
    for variant in VARIANTS:
        ext_x = external_features[variant][MODEL_FEATURES].to_numpy(float)
        fixed_pred = original_model.predict(ext_x).astype(int)
        fixed_score = batik_score(original_model, ext_x)
        predictions.append(
            make_prediction_frame(
                external,
                "Random Forest",
                variant,
                "fixed_original_model",
                fixed_pred,
                fixed_score,
            )
        )
        train = train_features[variant]
        refit = clone(estimator).fit(
            train[MODEL_FEATURES].to_numpy(float),
            train["label"].to_numpy(int),
        )
        refit_pred = refit.predict(ext_x).astype(int)
        refit_score = batik_score(refit, ext_x)
        predictions.append(
            make_prediction_frame(
                external,
                "Random Forest",
                variant,
                "refit_on_same_variant",
                refit_pred,
                refit_score,
            )
        )
        joblib.dump(refit, OUT / f"random_forest_refit_{variant}.joblib")
    return predictions


def load_deep_module():
    path = Path(__file__).resolve().parent / "11_eval_deep_learning.py"
    spec = importlib.util.spec_from_file_location("deep_baseline_stage", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def intervention_embeddings(
    backbone,
    transform,
    frame: pd.DataFrame,
    path_column: str,
    variant: str,
    device,
    label: str,
) -> np.ndarray:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class InterventionDataset(Dataset):
        def __init__(self, records: pd.DataFrame):
            self.records = records.reset_index(drop=True)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records.iloc[index]
            image, _, _ = transform_path(str(row[path_column]), variant)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb))
            return index, tensor

    dataset = InterventionDataset(frame)
    loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    values = np.empty((len(frame), 512), dtype=np.float32)
    backbone.to(device)
    backbone.eval()
    print(f"  ResNet18 embedding {label}: {len(frame)}")
    with torch.inference_mode():
        for batch_index, (positions, tensors) in enumerate(loader, 1):
            embeddings = backbone(tensors.to(device)).detach().cpu().numpy()
            if embeddings.ndim > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            values[np.asarray(positions, dtype=int)] = embeddings.astype(np.float32)
            if batch_index % 5 == 0 or batch_index == len(loader):
                print(f"    batch {batch_index}/{len(loader)}")
    return values


def resnet_conditions(
    final_train: pd.DataFrame,
    external: pd.DataFrame,
) -> list[pd.DataFrame]:
    import torch

    deep = load_deep_module()
    deep.set_reproducible(RANDOM_SEED)
    backbone, transform, embedding_dim, _ = deep.build_feature_extractor("ResNet18")
    if embedding_dim != 512:
        raise AssertionError(f"Unexpected ResNet18 embedding dimension: {embedding_dim}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = [
        align_existing_predictions(
            DEEP_LEARNING_DIR / "external_predictions_resnet18.csv",
            external,
            "ResNet18",
        )
    ]
    original_head = joblib.load(DEEP_LEARNING_DIR / "classifier_head_resnet18.joblib")
    for variant in VARIANTS:
        train_x = intervention_embeddings(
            backbone,
            transform,
            final_train,
            "generated_path",
            variant,
            device,
            f"{variant} final_train",
        )
        ext_x = intervention_embeddings(
            backbone,
            transform,
            external,
            "path",
            variant,
            device,
            f"{variant} external",
        )
        fixed_pred = original_head.predict(ext_x).astype(int)
        fixed_score = deep.batik_probability(original_head, ext_x)
        predictions.append(
            make_prediction_frame(
                external,
                "ResNet18",
                variant,
                "fixed_original_model",
                fixed_pred,
                fixed_score,
            )
        )
        refit = deep.build_head(RANDOM_SEED)
        refit.fit(train_x, final_train["label"].to_numpy(int))
        refit_pred = refit.predict(ext_x).astype(int)
        refit_score = deep.batik_probability(refit, ext_x)
        predictions.append(
            make_prediction_frame(
                external,
                "ResNet18",
                variant,
                "refit_on_same_variant",
                refit_pred,
                refit_score,
            )
        )
        joblib.dump(refit, OUT / f"resnet18_head_refit_{variant}.joblib")
    return predictions


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "variant", "training_mode", "condition"]
    for values, group in predictions.groupby(keys, sort=False):
        for subset in SUBSETS:
            mask = subset_mask(group, subset)
            data = group.loc[mask]
            counts = data.groupby("kelas").size().to_dict()
            rows.append(
                {
                    **dict(zip(keys, values)),
                    "subset": subset,
                    "n": len(data),
                    "n_batik": int(counts.get("batik", 0)),
                    "n_non_batik": int(counts.get("non_batik", 0)),
                    **metric_values(
                        data["label"].to_numpy(int),
                        data["predicted_label"].to_numpy(int),
                    ),
                }
            )
    return pd.DataFrame(rows)


def stratified_bootstrap_indices(
    labels: np.ndarray,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    classes = np.unique(labels)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError(f"Expected binary labels [0, 1], got {classes.tolist()}")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in classes]
    output = np.empty((n_replicates, len(labels)), dtype=np.int32)
    for replicate in range(n_replicates):
        pieces = [rng.choice(index, size=len(index), replace=True) for index in class_indices]
        output[replicate] = np.concatenate(pieces)
    return output


def bootstrap_macro_f1(
    labels: np.ndarray,
    predicted: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Vectorized binary macro-F1 for a shared bootstrap-index matrix."""
    labels = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(predicted, dtype=np.int8)
    sampled_labels = labels[indices]
    sampled_predicted = predicted[indices]
    class_f1 = []
    for class_value in (0, 1):
        true_class = sampled_labels == class_value
        predicted_class = sampled_predicted == class_value
        true_positive = np.sum(true_class & predicted_class, axis=1)
        false_positive = np.sum(~true_class & predicted_class, axis=1)
        false_negative = np.sum(true_class & ~predicted_class, axis=1)
        denominator = 2 * true_positive + false_positive + false_negative
        values = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator != 0,
        )
        class_f1.append(values)
    return (class_f1[0] + class_f1[1]) / 2.0


def paired_differences(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    saved_indices: dict[str, np.ndarray] = {}
    for subset_index, subset in enumerate(SUBSETS):
        template = predictions.loc[
            predictions["condition"].eq("baseline_original__original")
            & predictions["model"].eq("Random Forest")
        ].sort_values("source_id")
        template = template.loc[subset_mask(template, subset)]
        indices = stratified_bootstrap_indices(
            template["label"].to_numpy(int),
            BOOTSTRAP_REPLICATES,
            RANDOM_SEED + subset_index,
        )
        saved_indices[subset] = indices
        for model, model_frame in predictions.groupby("model", sort=False):
            baseline = model_frame.loc[
                model_frame["condition"].eq("baseline_original__original")
            ].sort_values("source_id")
            baseline = baseline.loc[subset_mask(baseline, subset)].reset_index(drop=True)
            for condition, condition_frame in model_frame.groupby("condition", sort=False):
                if condition == "baseline_original__original":
                    continue
                condition_frame = condition_frame.sort_values("source_id")
                condition_frame = condition_frame.loc[
                    subset_mask(condition_frame, subset)
                ].reset_index(drop=True)
                if not baseline["source_id"].equals(condition_frame["source_id"]):
                    raise AssertionError("Prediction frames are not source-aligned")
                y = baseline["label"].to_numpy(int)
                base_pred = baseline["predicted_label"].to_numpy(int)
                cond_pred = condition_frame["predicted_label"].to_numpy(int)
                base_f1 = f1_score(y, base_pred, average="macro", zero_division=0)
                cond_f1 = f1_score(y, cond_pred, average="macro", zero_division=0)
                replicate_differences = bootstrap_macro_f1(
                    y, cond_pred, indices
                ) - bootstrap_macro_f1(y, base_pred, indices)
                base_correct = base_pred == y
                cond_correct = cond_pred == y
                first = condition_frame.iloc[0]
                rows.append(
                    {
                        "model": model,
                        "variant": first["variant"],
                        "training_mode": first["training_mode"],
                        "condition": condition,
                        "subset": subset,
                        "n": len(y),
                        "baseline_macro_f1": base_f1,
                        "condition_macro_f1": cond_f1,
                        "macro_f1_difference": cond_f1 - base_f1,
                        "ci_lower": float(np.quantile(replicate_differences, 0.025)),
                        "ci_upper": float(np.quantile(replicate_differences, 0.975)),
                        "prediction_flips": int(np.sum(base_pred != cond_pred)),
                        "baseline_wrong_condition_right": int(np.sum(~base_correct & cond_correct)),
                        "baseline_right_condition_wrong": int(np.sum(base_correct & ~cond_correct)),
                    }
                )
    return pd.DataFrame(rows), saved_indices


def validate_inputs(final_train: pd.DataFrame, external: pd.DataFrame) -> None:
    if len(final_train) != 400:
        raise AssertionError(f"Expected 400 final-training instances, found {len(final_train)}")
    train_counts = final_train.groupby("kelas").size().to_dict()
    if train_counts != {"batik": 200, "non_batik": 200}:
        raise AssertionError(f"Unexpected final-training counts: {train_counts}")
    if final_train["generated_path"].duplicated().any():
        raise AssertionError("Duplicate generated_path in final-training manifest")
    if len(external) != 60 or external.groupby("kelas").size().to_dict() != {"batik": 30, "non_batik": 30}:
        raise AssertionError("External manifest must contain 30 images per class")
    if external["source_id"].duplicated().any():
        raise AssertionError("Duplicate external source_id")
    for column in ("width", "height", "extension", "sha256"):
        if external[column].isna().any():
            raise AssertionError(f"Missing external manifest value: {column}")


def dataframe_to_markdown(frame: pd.DataFrame, float_digits: int = 3) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
        else:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(row.astype(str).tolist()) + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    elapsed: float,
    execution_mode: str,
) -> None:
    all_metrics = metrics.loc[metrics["subset"].eq("all_external")].copy()
    display_columns = [
        "model",
        "training_mode",
        "variant",
        "n",
        "f1_macro",
        "balanced_accuracy",
        "mcc",
        "recall_non_batik",
        "recall_batik",
    ]
    table = dataframe_to_markdown(all_metrics[display_columns])
    diff_all = differences.loc[differences["subset"].eq("all_external")].copy()
    diff_columns = [
        "model",
        "training_mode",
        "variant",
        "macro_f1_difference",
        "ci_lower",
        "ci_upper",
        "prediction_flips",
    ]
    diff_table = dataframe_to_markdown(diff_all[diff_columns])
    completion_text = (
        f"The full experiment completed successfully in {elapsed:.1f} seconds."
        if execution_mode == "full"
        else "Finalization completed successfully from the previously saved model predictions."
    )
    report = f"""# R2.5 Acquisition-Intervention Report

## Status

{completion_text} Results are
exploratory and must not be described as a prospective confirmatory test.

## Prespecified question

Do Random Forest or the frozen ResNet18 pipeline change their external
predictions after image format, resolution, and the new encoding operation are
made common across both supplied classes?

## Design

- Source images were never modified.
- Two variants were prespecified before running the experiment:
  `canonical_png_512` and `canonical_jpeg_q95_444_512`.
- Every image was decoded, resized to 512 x 512 pixels, encoded with the stated
  fixed codec parameters, decoded again, and then passed to the unchanged active
  feature or frozen-backbone definition.
- `fixed_original_model` changes only the external input while retaining the
  originally fitted classifier.
- `refit_on_same_variant` uses the unchanged fixed model family and seed, refits
  on the existing 400-row balanced final-training manifest after the same
  intervention, and evaluates the corresponding external variant.
- The external collection never selected a variant, threshold, model family, or
  hyperparameter. All conditions are reported.
- Three analysis populations were prespecified: all 60 external images; the 54
  original-format JPEG images; and a deterministic 28-image coarsened match
  balanced within JPEG/pixel-count/aspect-ratio strata.

## External metrics, all 60 images

{table}

## Paired macro-F1 differences versus each model's original baseline

{diff_table}

## Interpretation limits

1. Later re-encoding cannot remove historical JPEG artifacts, interpolation,
   sharpening, or collection style already present in decoded pixels.
2. Resizing to a common square is itself an intervention and can alter motif
   geometry or aspect ratio.
3. Fixed-model sensitivity shows prediction instability under a controlled
   input change; it does not identify which visual feature caused a flip.
4. Same-variant refitting estimates a normalized pipeline, not the isolated
   causal effect of one acquisition attribute.
5. The 28-image matched subset is small, discards non-overlapping acquisition
   strata, and is reported only as an exploratory sensitivity analysis.
6. The external collection had already been inspected during manuscript
   development. No condition is confirmatory, and no external result may replace
   development-only model selection.

## Reproducibility outputs

- `intervention_config.json`
- `intervention_transform_manifest.csv`
- `external_intervention_predictions.csv`
- `external_intervention_metrics.csv`
- `paired_macro_f1_differences.csv`
- `external_matched_subset_manifest.csv`
- `paired_bootstrap_indices.npz`
- per-variant transformed six-feature tables and refitted classifier files
"""
    (OUT / "R2_5_ACQUISITION_INTERVENTION_REPORT.md").write_text(report, encoding="utf-8")


def finalize_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    elapsed: float,
    execution_mode: str,
) -> None:
    differences, bootstrap_indices = paired_differences(predictions)
    differences.to_csv(OUT / "paired_macro_f1_differences.csv", index=False)
    np.savez_compressed(OUT / "paired_bootstrap_indices.npz", **bootstrap_indices)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "variants": VARIANTS,
        "subsets": list(SUBSETS),
        "selection_rule": "No intervention condition is selected; all prespecified conditions are reported.",
        "source_images_modified": False,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "execution_mode": execution_mode,
    }
    (OUT / "intervention_config.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    write_report(metrics, differences, elapsed, execution_mode)
    (OUT / "run.log").write_text(
        "16_acquisition_intervention.py completed successfully\n"
        f"execution_mode={execution_mode}\n"
        f"elapsed_seconds={elapsed:.3f}\n"
        f"prediction_rows={len(predictions)}\n"
        f"metric_rows={len(metrics)}\n"
        f"paired_difference_rows={len(differences)}\n"
        "status=success\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 78)
    print("R2.5 ACQUISITION INTERVENTION COMPLETE")
    print("=" * 78)
    print(metrics.loc[metrics["subset"].eq("all_external"), [
        "model", "training_mode", "variant", "n", "f1_macro",
        "balanced_accuracy", "mcc", "recall_non_batik", "recall_batik",
    ]].to_string(index=False))
    print(f"\nOutput: {OUT}")


def main() -> None:
    start = time.time()
    if "--resume-bootstrap" in sys.argv:
        prediction_path = OUT / "external_intervention_predictions.csv"
        metric_path = OUT / "external_intervention_metrics.csv"
        if not prediction_path.exists() or not metric_path.exists():
            raise FileNotFoundError(
                "Resume requires existing prediction and metric CSV files."
            )
        predictions = pd.read_csv(prediction_path)
        metrics = pd.read_csv(metric_path)
        finalize_outputs(predictions, metrics, time.time() - start, "resume_bootstrap")
        return
    reset_directory(OUT)
    final_train = pd.read_csv(FINAL_TRAIN_PATH)
    external = pd.read_csv(EXTERNAL_MANIFEST_PATH).reset_index(drop=True)
    validate_inputs(final_train, external)
    matched = coarsened_match_manifest(external)
    external = external.merge(
        matched[
            [
                "source_id",
                "pixel_bin",
                "aspect_bin",
                "stratum",
                "selected_coarsened_match",
            ]
        ],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    matched.to_csv(OUT / "external_matched_subset_manifest.csv", index=False)

    train_features: dict[str, pd.DataFrame] = {}
    external_features: dict[str, pd.DataFrame] = {}
    transform_manifests = []
    for variant in VARIANTS:
        print(f"\nExtracting six features for {variant}")
        train_table, train_manifest = feature_table(
            final_train,
            "generated_path",
            "final_train",
            variant,
        )
        external_table, external_manifest = feature_table(
            external,
            "path",
            "external",
            variant,
        )
        train_table.to_csv(OUT / f"features_final_train_{variant}.csv", index=False)
        external_table.to_csv(OUT / f"features_external_{variant}.csv", index=False)
        train_features[variant] = train_table
        external_features[variant] = external_table
        transform_manifests.extend([train_manifest, external_manifest])
    pd.concat(transform_manifests, ignore_index=True).to_csv(
        OUT / "intervention_transform_manifest.csv",
        index=False,
    )

    print("\nEvaluating Random Forest conditions")
    prediction_frames = random_forest_conditions(train_features, external_features, external)
    print("\nEvaluating ResNet18 conditions")
    prediction_frames.extend(resnet_conditions(final_train, external))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_rows = 2 * (1 + 2 * len(VARIANTS)) * len(external)
    if len(predictions) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} prediction rows, found {len(predictions)}")
    predictions.to_csv(OUT / "external_intervention_predictions.csv", index=False)

    metrics = metric_table(predictions)
    metrics.to_csv(OUT / "external_intervention_metrics.csv", index=False)
    elapsed = time.time() - start
    finalize_outputs(predictions, metrics, elapsed, "full")


if __name__ == "__main__":
    main()
