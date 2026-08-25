"""Repeated strictly nested CV with fold-local augmentation.

This submission diagnostic is intentionally separate from the active 01--10
pipeline. All splits are formed from the 201 clean development originals.
Augmented features are generated deterministically per original, then filtered
by source_id so that only descendants of the current inner/outer training split
can enter model fitting. Inner validation and outer test folds remain originals.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from pipeline_common import extract_six, read_image_color, reset_directory, resolve_project_path
from pipeline_config import (
    FEATURE_DIR,
    FEATURE_GROUPS,
    MODEL_FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_PER_KELAS,
)
from pipeline_models import build_models, metric_values


OUT = RESULTS_DIR / "14_repeated_nested_augmentation"
N_REPEATS = 5
OUTER_SPLITS = 5
INNER_SPLITS = 4
DERIVATIVES_PER_ORIGINAL = 9

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


def load_apply_transform():
    script = Path(__file__).resolve().parent / "03_augment_dataset.py"
    spec = importlib.util.spec_from_file_location("augmentation_stage", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import augmentation implementation: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_transform


def validate_originals(original: pd.DataFrame) -> None:
    required = {"path", "source_id", "group_id", "kelas", "label", "subjenis", *MODEL_FEATURES}
    missing = sorted(required - set(original.columns))
    if missing:
        raise ValueError(f"Original feature table is missing columns: {missing}")
    if original.source_id.duplicated().any():
        raise AssertionError("Development source_id must be unique at original-file grain")
    counts = original.groupby("kelas").size().to_dict()
    if counts != {"batik": 137, "non_batik": 64}:
        raise AssertionError(f"Expected post-exclusion class counts 137/64, found {counts}")
    if len(original) != 201:
        raise AssertionError(f"Expected 201 clean originals, found {len(original)}")


def build_feature_pool(original: pd.DataFrame) -> pd.DataFrame:
    apply_transform = load_apply_transform()
    rows: list[dict] = []
    ordered = original.sort_values("source_id").reset_index(drop=True)
    for source_index, row in enumerate(ordered.itertuples(index=False), 1):
        base = {
            "path": row.path,
            "source_id": row.source_id,
            "kelas": row.kelas,
            "label": int(row.label),
            "subjenis": row.subjenis,
        }
        rows.append({
            **base,
            "is_augmented": False,
            "transform": "original",
            "transform_index": -1,
            **{name: float(getattr(row, name)) for name in MODEL_FEATURES},
        })
        image = read_image_color(resolve_project_path(row.path))
        if image is None:
            raise OSError(f"Unreadable development original: {row.path}")
        rng = np.random.default_rng(RANDOM_SEED + 100_000 + source_index)
        for transform_index in range(DERIVATIVES_PER_ORIGINAL):
            augmented, transform = apply_transform(image, transform_index, rng)
            features, _ = extract_six(augmented)
            rows.append({
                **base,
                "is_augmented": True,
                "transform": transform,
                "transform_index": transform_index,
                **{name: float(features[name]) for name in MODEL_FEATURES},
            })
        if source_index % 25 == 0 or source_index == len(ordered):
            print(f"  augmented feature pool: {source_index}/{len(ordered)} originals")
    pool = pd.DataFrame(rows)
    expected = len(original) * (DERIVATIVES_PER_ORIGINAL + 1)
    if len(pool) != expected:
        raise AssertionError(f"Feature pool row count mismatch: {len(pool)} != {expected}")
    derivative_counts = pool.query("is_augmented").groupby("source_id").size()
    if not (derivative_counts == DERIVATIVES_PER_ORIGINAL).all():
        raise AssertionError("Every original must have the same derivative budget")
    return pool


def balanced_training(
    pool: pd.DataFrame,
    train_sources: set[str],
    seed: int,
) -> pd.DataFrame:
    candidates = pool.loc[pool.source_id.isin(train_sources)].copy()
    if set(candidates.source_id) != train_sources:
        missing = sorted(train_sources - set(candidates.source_id))
        raise AssertionError(f"Training sources absent from feature pool: {missing[:5]}")
    pieces = []
    for class_index, class_name in enumerate(("batik", "non_batik")):
        group = candidates.loc[candidates.kelas == class_name]
        originals = group.loc[~group.is_augmented]
        derivatives = group.loc[group.is_augmented]
        needed = TARGET_PER_KELAS - len(originals)
        if needed < 0:
            raise AssertionError("Fold contains more originals than the training target")
        if needed > len(derivatives):
            raise AssertionError(
                f"Insufficient {class_name} derivatives: need {needed}, have {len(derivatives)}"
            )
        selected = derivatives.sample(
            n=needed,
            replace=False,
            random_state=seed + class_index,
        )
        pieces.append(pd.concat([originals, selected], ignore_index=True))
    training = pd.concat(pieces, ignore_index=True).sample(
        frac=1,
        random_state=seed + 99,
    ).reset_index(drop=True)
    counts = training.groupby("kelas").size().to_dict()
    if counts != {"batik": TARGET_PER_KELAS, "non_batik": TARGET_PER_KELAS}:
        raise AssertionError(f"Fold-local training is not balanced: {counts}")
    if not set(training.source_id).issubset(train_sources):
        raise AssertionError("A derivative from outside the training source set was selected")
    return training


def manifest_rows(
    training: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    inner_fold: int | str,
    phase: str,
) -> pd.DataFrame:
    columns = [
        "source_id", "path", "kelas", "label", "subjenis",
        "is_augmented", "transform", "transform_index",
    ]
    output = training[columns].copy()
    output.insert(0, "phase", phase)
    output.insert(0, "inner_fold", inner_fold)
    output.insert(0, "outer_fold", outer_fold)
    output.insert(0, "repeat", repeat)
    return output


def run_repeated_nested(
    original: pd.DataFrame,
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = build_models()
    candidate_specs = [
        (feature_name, features, model_name, estimator)
        for feature_name, features in FEATURE_GROUPS.items()
        for model_name, estimator in models.items()
    ]
    y = original.label.to_numpy(int)
    groups = original.group_id.to_numpy()
    outer_rows: list[dict] = []
    inner_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    origin_rows: list[pd.DataFrame] = []
    assignment_rows: list[dict] = []

    for repeat in range(1, N_REPEATS + 1):
        outer_seed = RANDOM_SEED + repeat * 10_000
        outer = StratifiedGroupKFold(
            OUTER_SPLITS,
            shuffle=True,
            random_state=outer_seed,
        )
        repeat_prediction_count = 0
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
            outer.split(original, y, groups), 1
        ):
            outer_train = original.iloc[outer_train_idx].reset_index(drop=True)
            outer_test = original.iloc[outer_test_idx].copy()
            train_sources = set(outer_train.source_id)
            test_sources = set(outer_test.source_id)
            if train_sources & test_sources:
                raise AssertionError("Outer train/test source leakage")
            for source_id in outer_test.source_id:
                assignment_rows.append({
                    "repeat": repeat,
                    "source_id": source_id,
                    "outer_fold": outer_fold,
                })

            inner_seed = outer_seed + outer_fold * 100
            inner = StratifiedGroupKFold(
                INNER_SPLITS,
                shuffle=True,
                random_state=inner_seed,
            )
            prepared_inner = []
            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
                inner.split(outer_train, outer_train.label, outer_train.group_id), 1
            ):
                inner_train_sources = set(outer_train.iloc[inner_train_idx].source_id)
                inner_valid = outer_train.iloc[inner_valid_idx].copy()
                inner_valid_sources = set(inner_valid.source_id)
                if inner_train_sources & inner_valid_sources:
                    raise AssertionError("Inner train/validation source leakage")
                training_seed = inner_seed + inner_fold
                inner_training = balanced_training(
                    pool,
                    inner_train_sources,
                    training_seed,
                )
                if set(inner_training.source_id) & inner_valid_sources:
                    raise AssertionError("Validation descendants entered inner training")
                origin_rows.append(manifest_rows(
                    inner_training,
                    repeat,
                    outer_fold,
                    inner_fold,
                    "inner_selection",
                ))
                prepared_inner.append((inner_fold, inner_training, inner_valid))

            scored_candidates = []
            for candidate_order, (feature_name, features, model_name, estimator) in enumerate(
                candidate_specs
            ):
                scores = []
                for _, inner_training, inner_valid in prepared_inner:
                    model = clone(estimator)
                    model.fit(
                        inner_training[features].to_numpy(float),
                        inner_training.label.to_numpy(int),
                    )
                    prediction = model.predict(inner_valid[features].to_numpy(float))
                    scores.append(f1_score(
                        inner_valid.label.to_numpy(int),
                        prediction,
                        average="macro",
                        zero_division=0,
                    ))
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores, ddof=1))
                scored_candidates.append((
                    mean_score,
                    candidate_order,
                    feature_name,
                    features,
                    model_name,
                    estimator,
                ))
                inner_rows.append({
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "feature_set": feature_name,
                    "model": model_name,
                    "inner_macro_f1_mean": mean_score,
                    "inner_macro_f1_std": std_score,
                    "inner_fold_scores": json.dumps([round(v, 12) for v in scores]),
                })

            (
                best_score,
                _,
                selected_feature_set,
                selected_features,
                selected_model,
                selected_estimator,
            ) = sorted(scored_candidates, key=lambda item: (-item[0], item[1]))[0]

            outer_training_seed = outer_seed + outer_fold * 100 + 90
            outer_training = balanced_training(pool, train_sources, outer_training_seed)
            if set(outer_training.source_id) & test_sources:
                raise AssertionError("Outer test descendants entered final outer training")
            origin_rows.append(manifest_rows(
                outer_training,
                repeat,
                outer_fold,
                "all",
                "outer_fit",
            ))
            model = clone(selected_estimator)
            model.fit(
                outer_training[selected_features].to_numpy(float),
                outer_training.label.to_numpy(int),
            )
            prediction = model.predict(outer_test[selected_features].to_numpy(float)).astype(int)
            metrics = metric_values(outer_test.label.to_numpy(int), prediction)
            outer_rows.append({
                "repeat": repeat,
                "outer_fold": outer_fold,
                "outer_seed": outer_seed,
                "selected_feature_set": selected_feature_set,
                "selected_model": selected_model,
                "inner_macro_f1": best_score,
                "outer_train_original_n": len(outer_train),
                "outer_train_instance_n": len(outer_training),
                "outer_test_original_n": len(outer_test),
                **metrics,
            })
            predicted = outer_test[["path", "source_id", "kelas", "label", "subjenis"]].copy()
            predicted.insert(0, "outer_fold", outer_fold)
            predicted.insert(0, "repeat", repeat)
            predicted["selected_feature_set"] = selected_feature_set
            predicted["selected_model"] = selected_model
            predicted["predicted_label"] = prediction
            predicted["correct"] = predicted.label.to_numpy(int) == prediction
            prediction_rows.append(predicted)
            repeat_prediction_count += len(predicted)
            print(
                f"  repeat {repeat}/{N_REPEATS}, outer {outer_fold}/{OUTER_SPLITS}: "
                f"{selected_model} | {selected_feature_set} | F1={metrics['f1_macro']:.3f}"
            )
        if repeat_prediction_count != len(original):
            raise AssertionError(
                f"Repeat {repeat} produced {repeat_prediction_count} predictions, expected {len(original)}"
            )

    outer_metrics = pd.DataFrame(outer_rows)
    inner_candidates = pd.DataFrame(inner_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    origin_manifest = pd.concat(origin_rows, ignore_index=True)
    assignments = pd.DataFrame(assignment_rows)
    return outer_metrics, inner_candidates, predictions, origin_manifest, assignments


def summarize(
    predictions: pd.DataFrame,
    outer_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    for repeat, group in predictions.groupby("repeat", sort=True):
        if len(group) != 201 or group.source_id.nunique() != 201:
            raise AssertionError(f"Repeat {repeat} is not complete at original-file grain")
        repeat_rows.append({
            "repeat": int(repeat),
            "n_original_predictions": len(group),
            **metric_values(group.label.to_numpy(int), group.predicted_label.to_numpy(int)),
        })
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary_rows = []
    for metric in METRICS:
        summary_rows.append({
            "metric": metric,
            "repeat_mean": repeat_metrics[metric].mean(),
            "repeat_std": repeat_metrics[metric].std(ddof=1),
            "repeat_min": repeat_metrics[metric].min(),
            "repeat_max": repeat_metrics[metric].max(),
            "outer_fold_mean": outer_metrics[metric].mean(),
            "outer_fold_std": outer_metrics[metric].std(ddof=1),
        })
    summary = pd.DataFrame(summary_rows)
    selection = (
        outer_metrics.groupby(["selected_feature_set", "selected_model"])
        .size()
        .rename("outer_fold_selections")
        .reset_index()
        .sort_values("outer_fold_selections", ascending=False, ignore_index=True)
    )
    return repeat_metrics, summary, selection


def main() -> None:
    original_path = FEATURE_DIR / "development_original_features.csv"
    if not original_path.exists():
        raise FileNotFoundError("Run 05_extract_features.py before this diagnostic")
    original = pd.read_csv(original_path).reset_index(drop=True)
    validate_originals(original)
    reset_directory(OUT)

    print("=" * 76)
    print("STAGE 14 - REPEATED STRICTLY NESTED FOLD-LOCAL AUGMENTATION")
    print("=" * 76)
    pool = build_feature_pool(original)
    pool.to_csv(OUT / "augmentation_feature_pool.csv", index=False)

    outer, inner, predictions, origins, assignments = run_repeated_nested(original, pool)
    repeat_metrics, summary, selection = summarize(predictions, outer)
    outer.to_csv(OUT / "outer_fold_metrics.csv", index=False)
    inner.to_csv(OUT / "inner_candidate_metrics.csv", index=False)
    predictions.to_csv(OUT / "outer_oof_predictions.csv", index=False)
    origins.to_csv(OUT / "training_origin_manifest.csv", index=False)
    assignments.to_csv(OUT / "outer_fold_assignments.csv", index=False)
    repeat_metrics.to_csv(OUT / "repeat_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    selection.to_csv(OUT / "selection_frequency.csv", index=False)

    composition = (
        origins.groupby(["repeat", "outer_fold", "inner_fold", "phase", "kelas", "is_augmented"])
        .size()
        .rename("instances")
        .reset_index()
    )
    composition.to_csv(OUT / "training_composition.csv", index=False)
    method = {
        "analysis_role": "submission sensitivity diagnostic; not external model selection",
        "clean_development_originals": len(original),
        "class_counts": original.groupby("kelas").size().to_dict(),
        "repeats": N_REPEATS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "candidate_space": f"{len(FEATURE_GROUPS)} feature groups x {len(build_models())} model families",
        "training_target_per_class": TARGET_PER_KELAS,
        "derivatives_per_original_in_pool": DERIVATIVES_PER_ORIGINAL,
        "split_grain": "development source photo group_id (tahap 24)",
        "augmentation_rule": "only descendants of current inner/outer training sources",
        "validation_rule": "inner validation and outer test use originals only",
        "scaling_rule": "StandardScaler remains inside LR/SVM sklearn Pipeline",
        "external_data_used": False,
        "base_random_seed": RANDOM_SEED,
    }
    (OUT / "methodology.json").write_text(json.dumps(method, indent=2), encoding="utf-8")

    f1 = summary.loc[summary.metric == "f1_macro"].iloc[0]
    bal = summary.loc[summary.metric == "balanced_accuracy"].iloc[0]
    mcc = summary.loc[summary.metric == "mcc"].iloc[0]
    report = (
        "# Repeated Strictly Nested Fold-Local Augmentation\n\n"
        f"- Clean development originals: {len(original)} "
        f"({(original.kelas == 'batik').sum()} batik, "
        f"{(original.kelas == 'non_batik').sum()} non-batik).\n"
        f"- Design: {N_REPEATS} repeats x {OUTER_SPLITS} outer folds x "
        f"{INNER_SPLITS} inner folds.\n"
        f"- Candidate space: {len(FEATURE_GROUPS)} fixed feature groups x "
        f"{len(build_models())} fixed model families.\n"
        "- Every training split contains 200 instances per class. Only descendants "
        "of originals in that training split are eligible.\n"
        "- Inner validation and outer test folds contain originals only. The external "
        "collection is never loaded.\n\n"
        "## Repeat-level estimates\n\n"
        f"- Macro-F1: {f1.repeat_mean:.6f} +/- {f1.repeat_std:.6f}.\n"
        f"- Balanced accuracy: {bal.repeat_mean:.6f} +/- {bal.repeat_std:.6f}.\n"
        f"- MCC: {mcc.repeat_mean:.6f} +/- {mcc.repeat_std:.6f}.\n\n"
        "These values are a repeated nested sensitivity analysis and are not "
        "numerically interchangeable with the active single-loop five-fold estimate.\n"
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("\nRepeat metrics:")
    print(repeat_metrics.to_string(index=False))
    print("\nSelection frequency:")
    print(selection.to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
