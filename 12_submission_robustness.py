"""Robustness checks for submission; does not modify the active 01--11 pipeline.

Outputs are diagnostic. Perceptual-hash pairs are candidates for manual review,
not confirmed duplicates. No source-aware claim is made because source/object IDs
are unavailable.
"""

from __future__ import annotations

import json
import platform
import sys
from itertools import combinations, product
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import PIL
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from pipeline_common import read_image_color, reset_directory, resolve_project_path
from pipeline_config import (
    AUDIT_DIR,
    FEATURE_DIR,
    FEATURE_GROUPS,
    MODEL_FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
)
from pipeline_models import metric_values


OUT = RESULTS_DIR / "12_submission_robustness"
PHASH_THRESHOLD = 8
DHASH_THRESHOLD = 8
OUTER_SPLITS = 5
INNER_SPLITS = 4


def _bits_to_hex(bits: np.ndarray) -> str:
    return "".join(f"{value:02x}" for value in np.packbits(bits.astype(np.uint8)))


def _hex_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def perceptual_hashes(path: Path) -> tuple[str, str]:
    image = read_image_color(path)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    # Exclude the DC coefficient when selecting the robust median threshold.
    phash = low > np.median(low.ravel()[1:])
    dhash_img = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = dhash_img[:, 1:] > dhash_img[:, :-1]
    return _bits_to_hex(phash.ravel()), _bits_to_hex(dhash.ravel())


def build_hash_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        phash, dhash = perceptual_hashes(resolve_project_path(row.path))
        rows.append({
            "source_set": row.source_set,
            "source_id": row.source_id,
            "path": row.path,
            "kelas": row.kelas,
            "subjenis": row.subjenis,
            "phash_64": phash,
            "dhash_64": dhash,
        })
    return pd.DataFrame(rows)


def candidate_pairs(first: pd.DataFrame, second: pd.DataFrame | None = None) -> pd.DataFrame:
    pairs = combinations(range(len(first)), 2) if second is None else product(range(len(first)), range(len(second)))
    other = first if second is None else second
    rows = []
    for left_i, right_i in pairs:
        left, right = first.iloc[left_i], other.iloc[right_i]
        pdist = _hex_distance(left.phash_64, right.phash_64)
        ddist = _hex_distance(left.dhash_64, right.dhash_64)
        if pdist <= PHASH_THRESHOLD or ddist <= DHASH_THRESHOLD:
            rows.append({
                "left_set": left.source_set, "left_path": left.path,
                "left_class": left.kelas, "left_subtype": left.subjenis,
                "right_set": right.source_set, "right_path": right.path,
                "right_class": right.kelas, "right_subtype": right.subjenis,
                "phash_hamming": pdist, "dhash_hamming": ddist,
                "trigger": "both" if pdist <= PHASH_THRESHOLD and ddist <= DHASH_THRESHOLD
                else ("phash" if pdist <= PHASH_THRESHOLD else "dhash"),
                "manual_status": "unreviewed_candidate",
            })
    columns = ["left_set", "left_path", "left_class", "left_subtype", "right_set",
               "right_path", "right_class", "right_subtype", "phash_hamming",
               "dhash_hamming", "trigger", "manual_status"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["phash_hamming", "dhash_hamming"], ignore_index=True
    ) if rows else pd.DataFrame(columns=columns)


def nested_cv(original: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = []
    for group_name, features in FEATURE_GROUPS.items():
        candidates.extend([
            (group_name, features, "Logistic Regression", Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
            ])),
            (group_name, features, "SVM (RBF)", Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ])),
            (group_name, features, "Random Forest", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
            )),
        ])

    y = original.label.to_numpy(int)
    groups = original.group_id.to_numpy()
    outer = StratifiedGroupKFold(OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_rows, selection_rows = [], []
    for outer_fold, (train_idx, test_idx) in enumerate(outer.split(original, y, groups), 1):
        train = original.iloc[train_idx].reset_index(drop=True)
        test = original.iloc[test_idx]
        if set(train.group_id) & set(test.group_id):
            raise AssertionError("Group ID bocor antara outer train dan test")
        inner = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED + outer_fold)
        candidate_scores = []
        for group_name, features, model_name, estimator in candidates:
            scores = []
            for inner_train_idx, inner_valid_idx in inner.split(train, train.label, train.group_id):
                model = clone(estimator)
                model.fit(train.iloc[inner_train_idx][features], train.iloc[inner_train_idx].label)
                pred = model.predict(train.iloc[inner_valid_idx][features])
                scores.append(f1_score(train.iloc[inner_valid_idx].label, pred, average="macro"))
            candidate_scores.append((float(np.mean(scores)), group_name, features, model_name, estimator))
            selection_rows.append({"outer_fold": outer_fold, "feature_set": group_name,
                                   "model": model_name, "inner_macro_f1_mean": np.mean(scores),
                                   "inner_macro_f1_std": np.std(scores, ddof=1)})
        best_score, group_name, features, model_name, estimator = max(
            candidate_scores, key=lambda item: (item[0], item[1], item[3])
        )
        model = clone(estimator).fit(train[features], train.label)
        prediction = model.predict(test[features])
        fold_rows.append({"outer_fold": outer_fold, "selected_feature_set": group_name,
                          "selected_model": model_name, "inner_macro_f1": best_score,
                          "outer_train_n": len(train), "outer_test_n": len(test),
                          **metric_values(test.label, prediction)})
    return pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def metadata_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    frame = manifest.copy()
    frame["log_file_size"] = np.log1p(frame.file_size_bytes.astype(float))
    frame["log_width"] = np.log1p(frame.width.astype(float))
    frame["log_height"] = np.log1p(frame.height.astype(float))
    frame["log_pixels"] = np.log1p(frame.width.astype(float) * frame.height.astype(float))
    frame["aspect_ratio"] = frame.width.astype(float) / frame.height.astype(float).clip(lower=1)
    return frame


def metadata_negative_control(dev: pd.DataFrame, ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["log_file_size", "log_width", "log_height", "log_pixels", "aspect_ratio"]
    categorical = ["extension"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    estimator = Pipeline([("metadata", prep), ("model", LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED))])
    dev_m, ext_m = metadata_frame(dev), metadata_frame(ext)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_SEED)
    oof = np.full(len(dev_m), -1, int)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(dev_m, dev_m.label, dev_m.group_id), 1
    ):
        model = clone(estimator).fit(dev_m.iloc[train_idx], dev_m.iloc[train_idx].label)
        pred = model.predict(dev_m.iloc[valid_idx])
        oof[valid_idx] = pred
        rows.append({"evaluation": "development_5fold", "fold": fold,
                     "train_n": len(train_idx), "test_n": len(valid_idx),
                     **metric_values(dev_m.iloc[valid_idx].label, pred)})
    full = clone(estimator).fit(dev_m, dev_m.label)
    ext_pred = full.predict(ext_m)
    rows.append({"evaluation": "external_after_fit_all_development", "fold": "all",
                 "train_n": len(dev_m), "test_n": len(ext_m),
                 **metric_values(ext_m.label, ext_pred)})
    predictions = pd.concat([
        dev_m[["source_set", "path", "kelas", "label"]].assign(
            evaluation="development_oof", predicted_label=oof),
        ext_m[["source_set", "path", "kelas", "label"]].assign(
            evaluation="external", predicted_label=ext_pred),
    ], ignore_index=True)
    predictions["correct"] = predictions.label == predictions.predicted_label
    return pd.DataFrame(rows), predictions


def main() -> None:
    reset_directory(OUT)
    dev = pd.read_csv(AUDIT_DIR / "development_manifest.csv")
    ext = pd.read_csv(AUDIT_DIR / "external_manifest.csv")
    original = pd.read_csv(FEATURE_DIR / "development_original_features.csv")

    hashes = build_hash_table(pd.concat([dev, ext], ignore_index=True))
    hashes.to_csv(OUT / "perceptual_hash_manifest.csv", index=False)
    dev_hash = hashes.query("source_set == 'development'").reset_index(drop=True)
    ext_hash = hashes.query("source_set == 'external'").reset_index(drop=True)
    within = candidate_pairs(dev_hash)
    cross = candidate_pairs(dev_hash, ext_hash)
    within.to_csv(OUT / "near_duplicate_candidates_development.csv", index=False)
    cross.to_csv(OUT / "near_duplicate_candidates_development_external.csv", index=False)

    nested_folds, nested_inner = nested_cv(original)
    nested_folds.to_csv(OUT / "nested_cv_outer_fold_metrics.csv", index=False)
    nested_inner.to_csv(OUT / "nested_cv_inner_candidates.csv", index=False)
    nested_summary = nested_folds.select_dtypes(include=np.number).drop(columns=["outer_fold"]).agg(["mean", "std"]).T
    nested_summary.to_csv(OUT / "nested_cv_summary.csv")

    metadata_metrics, metadata_predictions = metadata_negative_control(dev, ext)
    metadata_metrics.to_csv(OUT / "metadata_negative_control_metrics.csv", index=False)
    metadata_predictions.to_csv(OUT / "metadata_negative_control_predictions.csv", index=False)
    support = pd.concat([dev, ext]).groupby(["source_set", "kelas", "extension"], dropna=False).size().rename("n").reset_index()
    support.to_csv(OUT / "format_class_support.csv", index=False)

    environment = {
        "python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "constants": {"random_seed": RANDOM_SEED, "outer_splits": OUTER_SPLITS,
                      "inner_splits": INNER_SPLITS, "phash_threshold": PHASH_THRESHOLD,
                      "dhash_threshold": DHASH_THRESHOLD, "model_features": MODEL_FEATURES},
        "candidate_rule": "pHash Hamming <= 8 OR dHash Hamming <= 8; candidates require manual review",
        "nested_cv_search_space": "3 fixed model families x 4 predefined feature groups; originals only",
    }
    (OUT / "software_and_parameters.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    dev_meta = metadata_metrics.query("evaluation == 'development_5fold'")
    ext_meta = metadata_metrics.query("evaluation == 'external_after_fit_all_development'").iloc[0]
    report = f"""# Methodology revision report\n\n"
## Valid analyses completed\n\n"
- Perceptual-hash audit: {len(dev_hash)} development and {len(ext_hash)} external originals. Candidate rule: pHash Hamming <= {PHASH_THRESHOLD} OR dHash Hamming <= {DHASH_THRESHOLD}. Found {len(within)} development--development and {len(cross)} development--external candidate pairs. These are **not confirmed duplicates** until manually reviewed.\n"
- Nested stratified CV used only development originals. The inner 4-fold loop selected among 3 fixed model families and 4 predefined feature groups; the outer 5-fold loop estimated performance. Outer macro-F1 = {nested_folds.f1_macro.mean():.3f} +/- {nested_folds.f1_macro.std(ddof=1):.3f}; balanced accuracy = {nested_folds.balanced_accuracy.mean():.3f}; MCC = {nested_folds.mcc.mean():.3f}. Scaling for LR/SVM was fit inside each inner/outer training split.\n"
- Learned metadata-only negative control (extension, dimensions, aspect ratio, file size) achieved development 5-fold macro-F1 = {dev_meta.f1_macro.mean():.3f} +/- {dev_meta.f1_macro.std(ddof=1):.3f}, and external macro-F1 = {ext_meta.f1_macro:.3f}. This diagnoses acquisition predictiveness; it does not prove what pixel models learned.\n"
- Format-by-class support was exported descriptively. No arbitrary matched-subset inferential score was produced because sparse/confounded cells would make that estimate unstable and researcher-dependent.\n\n"
## Interpretation limits\n\n"
- No source/object/session identifiers exist; therefore neither this script nor the existing pipeline performs group-aware CV. Stratified file-level folds can still share unobserved acquisition groups.\n"
- Perceptual hashes are heuristic and can miss crops, strong edits, or semantically repeated source images. Every candidate needs visual/manual adjudication before exclusion or grouping.\n"
- Nested CV addresses model/feature-family selection optimism, but it cannot repair acquisition--class confounding or missing scientific ground truth.\n"
- This nested-CV diagnostic intentionally uses originals only. It is not numerically interchangeable with the active augmented-training CV; a nested fold-local augmentation experiment would be a separate, more expensive analysis.\n"
- The external collection has already informed manuscript development; its results remain exploratory rather than a new prospective confirmation.\n\n"
## Still requires new provenance/data\n\n"
1. Expert label protocol, annotator IDs/agreement, source URLs, licenses, and source/object/session group IDs.\n"
2. Manual adjudication of near-duplicate candidates and rerun with confirmed groups.\n"
3. Group-aware nested CV after group IDs exist.\n"
4. Acquisition-balanced development data and a newly frozen multi-source external benchmark.\n"
5. Controlled RGB/grayscale and preprocessing/compression experiments if architectural explanations are claimed.\n"
""".replace('"\n', '\n')
    (OUT / "METHODOLOGY_REVISION_REPORT.md").write_text(report, encoding="utf-8")
    run_log = (
        "12_submission_robustness.py completed successfully\n"
        f"development originals={len(dev)}; external originals={len(ext)}\n"
        f"near-duplicate candidates dev-dev={len(within)}; dev-external={len(cross)}\n"
        f"nested outer CV macro-F1={nested_folds.f1_macro.mean():.6f} "
        f"+/- {nested_folds.f1_macro.std(ddof=1):.6f}\n"
        f"metadata-only development CV macro-F1={dev_meta.f1_macro.mean():.6f} "
        f"+/- {dev_meta.f1_macro.std(ddof=1):.6f}\n"
        f"metadata-only external macro-F1={ext_meta.f1_macro:.6f}\n"
        "status=success\n"
    )
    (OUT / "run.log").write_text(run_log, encoding="utf-8")
    print(f"Output: {OUT}")
    print(f"Near-duplicate candidates: dev-dev={len(within)}, dev-external={len(cross)}")
    print(f"Nested outer CV macro-F1: {nested_folds.f1_macro.mean():.3f} +/- {nested_folds.f1_macro.std(ddof=1):.3f}")
    print(f"Metadata-only CV macro-F1: {dev_meta.f1_macro.mean():.3f} +/- {dev_meta.f1_macro.std(ddof=1):.3f}")
    print(f"Metadata-only external macro-F1: {ext_meta.f1_macro:.3f}")


if __name__ == "__main__":
    main()
