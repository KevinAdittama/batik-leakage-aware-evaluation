"""Independent structural and numerical audit for stage 14 outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline_config import FEATURE_DIR, RANDOM_SEED, RESULTS_DIR, TARGET_PER_KELAS
from pipeline_models import metric_values


OUT = RESULTS_DIR / "14_repeated_nested_augmentation"
N_REPEATS = 5
OUTER_SPLITS = 5
INNER_SPLITS = 4


def main() -> None:
    original = pd.read_csv(FEATURE_DIR / "development_original_features.csv").reset_index(drop=True)
    pool = pd.read_csv(OUT / "augmentation_feature_pool.csv")
    outer_metrics = pd.read_csv(OUT / "outer_fold_metrics.csv")
    inner_candidates = pd.read_csv(OUT / "inner_candidate_metrics.csv")
    predictions = pd.read_csv(OUT / "outer_oof_predictions.csv")
    origins = pd.read_csv(OUT / "training_origin_manifest.csv")
    assignments = pd.read_csv(OUT / "outer_fold_assignments.csv")
    repeat_metrics = pd.read_csv(OUT / "repeat_metrics.csv")

    assertions: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not bool(condition):
            raise AssertionError(message)
        assertions.append(message)

    check(len(original) == 201, "201 clean development originals")
    check(original.source_id.nunique() == 201, "source_id unique at original-file grain")
    check(
        original.groupby("kelas").size().to_dict() == {"batik": 137, "non_batik": 64},
        "post-exclusion class counts are 137 batik and 64 non-batik",
    )
    check(len(pool) == 2010, "feature pool contains one original and nine derivatives per source")
    pool_counts = pool.groupby(["source_id", "is_augmented"]).size().unstack(fill_value=0)
    check(
        (pool_counts[False] == 1).all() and (pool_counts[True] == 9).all(),
        "every source has exactly one original row and nine derivative rows",
    )
    check(len(outer_metrics) == 25, "25 outer-fold metric rows")
    check(len(inner_candidates) == 300, "12 candidates evaluated in each of 25 outer folds")
    check(len(predictions) == 1005, "201 original predictions in each of five repeats")
    check(len(origins) == 50000, "125 training splits x 400 instances in origin manifest")
    check(len(assignments) == 1005, "outer-fold assignment covers every source in every repeat")

    original_sources = set(original.source_id)
    check(set(pool.source_id) == original_sources, "feature pool contains development sources only")
    check(set(predictions.source_id) == original_sources, "predictions contain development sources only")
    check(set(origins.source_id).issubset(original_sources), "training origins contain development sources only")

    y = original.label.to_numpy(int)
    validation_rows: list[dict] = []
    for repeat in range(1, N_REPEATS + 1):
        outer_seed = RANDOM_SEED + repeat * 10_000
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=outer_seed)
        repeat_predictions = predictions.loc[predictions.repeat == repeat]
        check(
            len(repeat_predictions) == 201 and repeat_predictions.source_id.nunique() == 201,
            f"repeat {repeat} predicts every original exactly once",
        )
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
            outer.split(original, y), 1
        ):
            outer_train = original.iloc[outer_train_idx].reset_index(drop=True)
            outer_test = original.iloc[outer_test_idx]
            train_sources = set(outer_train.source_id)
            test_sources = set(outer_test.source_id)
            observed_test = set(
                repeat_predictions.loc[
                    repeat_predictions.outer_fold == outer_fold, "source_id"
                ]
            )
            check(observed_test == test_sources, f"repeat {repeat} outer {outer_fold} test assignment reproducible")
            outer_origin = origins.loc[
                (origins.repeat == repeat)
                & (origins.outer_fold == outer_fold)
                & (origins.phase == "outer_fit")
            ]
            check(len(outer_origin) == 400, f"repeat {repeat} outer {outer_fold} has 400 fit instances")
            check(not (set(outer_origin.source_id) & test_sources), f"repeat {repeat} outer {outer_fold} excludes test descendants")
            check(set(outer_origin.source_id) == train_sources, f"repeat {repeat} outer {outer_fold} retains all training originals")
            check(
                outer_origin.groupby("kelas").size().to_dict()
                == {"batik": TARGET_PER_KELAS, "non_batik": TARGET_PER_KELAS},
                f"repeat {repeat} outer {outer_fold} balanced at 200 instances per class",
            )
            for row in outer_test.itertuples(index=False):
                validation_rows.append({
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "inner_fold": "all",
                    "phase": "outer_test",
                    "source_id": row.source_id,
                    "kelas": row.kelas,
                    "label": int(row.label),
                })

            inner_seed = outer_seed + outer_fold * 100
            inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=inner_seed)
            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
                inner.split(outer_train, outer_train.label), 1
            ):
                inner_train_sources = set(outer_train.iloc[inner_train_idx].source_id)
                inner_valid = outer_train.iloc[inner_valid_idx]
                inner_valid_sources = set(inner_valid.source_id)
                inner_origin = origins.loc[
                    (origins.repeat == repeat)
                    & (origins.outer_fold == outer_fold)
                    & (origins.phase == "inner_selection")
                    & (origins.inner_fold.astype(str) == str(inner_fold))
                ]
                check(len(inner_origin) == 400, f"repeat {repeat} outer {outer_fold} inner {inner_fold} has 400 training instances")
                check(not (set(inner_origin.source_id) & inner_valid_sources), f"repeat {repeat} outer {outer_fold} inner {inner_fold} excludes validation descendants")
                check(set(inner_origin.source_id) == inner_train_sources, f"repeat {repeat} outer {outer_fold} inner {inner_fold} retains all training originals")
                check(
                    inner_origin.groupby("kelas").size().to_dict()
                    == {"batik": TARGET_PER_KELAS, "non_batik": TARGET_PER_KELAS},
                    f"repeat {repeat} outer {outer_fold} inner {inner_fold} balanced at 200 instances per class",
                )
                for row in inner_valid.itertuples(index=False):
                    validation_rows.append({
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "phase": "inner_validation",
                        "source_id": row.source_id,
                        "kelas": row.kelas,
                        "label": int(row.label),
                    })

    recomputed = []
    for repeat, group in predictions.groupby("repeat", sort=True):
        recomputed.append({
            "repeat": int(repeat),
            **metric_values(group.label.to_numpy(int), group.predicted_label.to_numpy(int)),
        })
    recomputed = pd.DataFrame(recomputed)
    for metric in [
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
        "f1_macro", "mcc", "recall_non_batik", "recall_batik",
    ]:
        check(
            np.allclose(recomputed[metric], repeat_metrics[metric], rtol=0, atol=1e-12),
            f"repeat-level {metric} independently recomputed",
        )

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "validation_source_manifest.csv", index=False)
    audit = {
        "status": "pass",
        "assertion_count": len(assertions),
        "assertions": assertions,
        "observed_rows": {
            "feature_pool": len(pool),
            "outer_fold_metrics": len(outer_metrics),
            "inner_candidate_metrics": len(inner_candidates),
            "outer_oof_predictions": len(predictions),
            "training_origin_manifest": len(origins),
            "validation_source_manifest": len(validation),
        },
        "repeat_macro_f1_mean": float(recomputed.f1_macro.mean()),
        "repeat_macro_f1_std": float(recomputed.f1_macro.std(ddof=1)),
    }
    (OUT / "validation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
