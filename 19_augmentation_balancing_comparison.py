"""R2.7 - Perbandingan strategi penyeimbangan kelas pada fold nested yang sama.

Reviewer 2 butir 7 meminta augmentasi fold-local dibandingkan dengan class
weighting dan balanced-original sampling, pada protokol nested yang sama dan
diulang pada beberapa seed. Tahap 14 sudah menyediakan lengan augmentasi.
Skrip ini menambahkan dua lengan pembanding dan menyatukan ketiganya.

Tiga lengan yang dibandingkan:

  augmented_balanced
      Lengan tahap 14. Setiap split pelatihan berisi 200 instance per kelas,
      terdiri atas original pada split tersebut ditambah derivative yang hanya
      berasal dari original yang sama.

  class_weighted_originals
      Hanya original pada split pelatihan, tanpa augmentasi, dengan
      class_weight="balanced" pada seluruh keluarga model.

  balanced_original_sampling
      Hanya original pada split pelatihan, kelas mayoritas diturunkan secara
      acak sampai sama dengan kelas minoritas, tanpa bobot kelas.

Kontrol keadilan perbandingan:

  - Pembagian outer/inner identik untuk ketiga lengan karena seed dan urutan
    StratifiedKFold sama persis dengan tahap 14.
  - Seleksi nested dijalankan terpisah untuk setiap lengan pada ruang kandidat
    yang sama, yaitu 4 kelompok fitur x 3 keluarga model.
  - Inner validation dan outer test selalu berisi original tanpa augmentasi.
  - Perbandingan dilaporkan berpasangan pada tingkat (repeat, outer fold).
  - Koleksi eksternal tidak pernah dimuat.

Jalankan dari folder proyek:
    .\\env\\Scripts\\python.exe -B .\\19_augmentation_balancing_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline_config import (
    FEATURE_DIR,
    FEATURE_GROUPS,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_PER_KELAS,
)
from pipeline_models import metric_values

POOL_PATH = RESULTS_DIR / "14_repeated_nested_augmentation" / "augmentation_feature_pool.csv"
OUT = RESULTS_DIR / "19_balancing_comparison"
PARTIAL = OUT / "_partial"

N_REPEATS = 5
OUTER_SPLITS = 5
INNER_SPLITS = 4
DERIVATIVES_PER_ORIGINAL = 9

BASELINE_ARM = "augmented_balanced"
ARMS = (BASELINE_ARM, "class_weighted_originals", "balanced_original_sampling")


def build_models(class_weight: str | None):
    """Ruang kandidat yang sama; hanya class_weight yang dibedakan antar lengan."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_SEED,
                class_weight=class_weight,
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                random_state=RANDOM_SEED,
                class_weight=class_weight,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight=class_weight,
        ),
    }


def augmented_balanced(pool, train_sources, seed):
    """Lengan tahap 14: 200 instance per kelas dari original + derivative."""
    candidates = pool.loc[pool.source_id.isin(train_sources)]
    pieces = []
    for class_index, class_name in enumerate(("batik", "non_batik")):
        group = candidates.loc[candidates.kelas == class_name]
        originals = group.loc[~group.is_augmented]
        derivatives = group.loc[group.is_augmented]
        needed = TARGET_PER_KELAS - len(originals)
        if needed < 0:
            raise AssertionError("Fold memuat original melebihi target pelatihan")
        if needed > len(derivatives):
            raise AssertionError(f"Derivative {class_name} tidak cukup")
        selected = derivatives.sample(n=needed, replace=False, random_state=seed + class_index)
        pieces.append(pd.concat([originals, selected], ignore_index=True))
    training = pd.concat(pieces, ignore_index=True)
    return training.sample(frac=1, random_state=seed + 99).reset_index(drop=True)


def originals_only(pool, train_sources, seed):
    """Seluruh original pada split pelatihan, tanpa augmentasi."""
    training = pool.loc[pool.source_id.isin(train_sources) & ~pool.is_augmented]
    return training.sample(frac=1, random_state=seed + 99).reset_index(drop=True)


def balanced_original_sampling(pool, train_sources, seed):
    """Original saja; kelas mayoritas diturunkan sampai setara kelas minoritas."""
    candidates = pool.loc[pool.source_id.isin(train_sources) & ~pool.is_augmented]
    counts = candidates.groupby("kelas").size()
    minority = int(counts.min())
    pieces = []
    for class_index, class_name in enumerate(("batik", "non_batik")):
        group = candidates.loc[candidates.kelas == class_name]
        if len(group) > minority:
            group = group.sample(n=minority, replace=False, random_state=seed + class_index)
        pieces.append(group)
    training = pd.concat(pieces, ignore_index=True)
    return training.sample(frac=1, random_state=seed + 99).reset_index(drop=True)


ARM_SPEC = {
    "augmented_balanced": {
        "builder": augmented_balanced,
        "class_weight": None,
        "uses_augmentation": True,
    },
    "class_weighted_originals": {
        "builder": originals_only,
        "class_weight": "balanced",
        "uses_augmentation": False,
    },
    "balanced_original_sampling": {
        "builder": balanced_original_sampling,
        "class_weight": None,
        "uses_augmentation": False,
    },
}


def check_training(training, train_sources, valid_sources, arm):
    if not set(training.source_id).issubset(train_sources):
        raise AssertionError(f"[{arm}] sumber di luar split pelatihan ikut terpakai")
    if set(training.source_id) & valid_sources:
        raise AssertionError(f"[{arm}] turunan fold validasi bocor ke pelatihan")
    if training.label.nunique() < 2:
        raise AssertionError(f"[{arm}] split pelatihan hanya memuat satu kelas")


def run_arm(arm, original, pool, repeats=None):
    spec = ARM_SPEC[arm]
    selected_repeats = tuple(repeats) if repeats else tuple(range(1, N_REPEATS + 1))
    build_training = spec["builder"]
    models = build_models(spec["class_weight"])
    candidate_specs = [
        (feature_name, features, model_name, estimator)
        for feature_name, features in FEATURE_GROUPS.items()
        for model_name, estimator in models.items()
    ]
    y = original.label.to_numpy(int)
    groups = original.group_id.to_numpy()

    outer_rows, composition_rows, prediction_frames = [], [], []

    for repeat in selected_repeats:
        outer_seed = RANDOM_SEED + repeat * 10_000
        outer = StratifiedGroupKFold(OUTER_SPLITS, shuffle=True, random_state=outer_seed)

        for outer_fold, (train_idx, test_idx) in enumerate(
            outer.split(original, y, groups), 1
        ):
            outer_train = original.iloc[train_idx].reset_index(drop=True)
            outer_test = original.iloc[test_idx].copy()
            train_sources = set(outer_train.source_id)
            test_sources = set(outer_test.source_id)
            if train_sources & test_sources:
                raise AssertionError("Kebocoran sumber outer train/test")

            inner_seed = outer_seed + outer_fold * 100
            inner = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=inner_seed)

            prepared = []
            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
                inner.split(outer_train, outer_train.label, outer_train.group_id), 1
            ):
                inner_train_sources = set(outer_train.iloc[inner_train_idx].source_id)
                inner_valid = outer_train.iloc[inner_valid_idx].copy()
                inner_valid_sources = set(inner_valid.source_id)
                if inner_train_sources & inner_valid_sources:
                    raise AssertionError("Kebocoran sumber inner train/validation")
                training = build_training(pool, inner_train_sources, inner_seed + inner_fold)
                check_training(training, inner_train_sources, inner_valid_sources, arm)
                prepared.append((training, inner_valid))

            scored = []
            for order, (feature_name, features, model_name, estimator) in enumerate(
                candidate_specs
            ):
                scores = []
                for training, inner_valid in prepared:
                    model = clone(estimator)
                    model.fit(
                        training[features].to_numpy(float),
                        training.label.to_numpy(int),
                    )
                    prediction = model.predict(inner_valid[features].to_numpy(float))
                    scores.append(f1_score(
                        inner_valid.label.to_numpy(int),
                        prediction,
                        average="macro",
                        zero_division=0,
                    ))
                scored.append((float(np.mean(scores)), order, feature_name,
                               features, model_name, estimator))

            best_score, _, feature_set, features, model_name, estimator = sorted(
                scored, key=lambda item: (-item[0], item[1])
            )[0]

            outer_training = build_training(
                pool, train_sources, outer_seed + outer_fold * 100 + 90
            )
            check_training(outer_training, train_sources, test_sources, arm)

            model = clone(estimator)
            model.fit(
                outer_training[features].to_numpy(float),
                outer_training.label.to_numpy(int),
            )
            prediction = model.predict(
                outer_test[features].to_numpy(float)
            ).astype(int)
            metrics = metric_values(outer_test.label.to_numpy(int), prediction)

            counts = outer_training.groupby("kelas").size().to_dict()
            outer_rows.append({
                "arm": arm,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "selected_feature_set": feature_set,
                "selected_model": model_name,
                "inner_macro_f1": best_score,
                "train_instances": len(outer_training),
                "train_batik": counts.get("batik", 0),
                "train_non_batik": counts.get("non_batik", 0),
                "train_originals": int((~outer_training.is_augmented).sum()),
                "train_derivatives": int(outer_training.is_augmented.sum()),
                "test_originals": len(outer_test),
                **metrics,
            })
            composition_rows.append({
                "arm": arm,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "class_weight": str(spec["class_weight"]),
                "uses_augmentation": spec["uses_augmentation"],
                **{f"n_{k}": v for k, v in counts.items()},
            })

            predicted = outer_test[["source_id", "kelas", "label", "subjenis"]].copy()
            predicted.insert(0, "outer_fold", outer_fold)
            predicted.insert(0, "repeat", repeat)
            predicted.insert(0, "arm", arm)
            predicted["predicted_label"] = prediction
            predicted["correct"] = predicted.label.to_numpy(int) == prediction
            prediction_frames.append(predicted)

        print(f"  [{arm}] repeat {repeat}/{N_REPEATS} selesai")

    return (
        pd.DataFrame(outer_rows),
        pd.DataFrame(composition_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def repeat_level(predictions, n_original):
    """Metrik tingkat repeat dari seluruh prediksi out-of-fold.

    Menggunakan konvensi yang sama dengan tahap 14: seluruh prediksi satu repeat
    digabung lebih dahulu, lalu metrik dihitung sekali. Konvensi ini tidak sama
    dengan merata-ratakan metrik per outer fold, sehingga harus dipertahankan
    agar angka kedua tahap dapat dibandingkan langsung.
    """
    rows = []
    for (arm, repeat), group in predictions.groupby(["arm", "repeat"], sort=True):
        if len(group) != n_original or group.source_id.nunique() != n_original:
            raise AssertionError(f"{arm} repeat {repeat} tidak lengkap")
        rows.append({
            "arm": arm,
            "repeat": int(repeat),
            "n_original_predictions": len(group),
            **metric_values(
                group.label.to_numpy(int),
                group.predicted_label.to_numpy(int),
            ),
        })
    return pd.DataFrame(rows)


def arm_summary(repeats):
    rows = []
    for arm, group in repeats.groupby("arm", sort=False):
        rows.append({
            "arm": arm,
            "macro_f1_mean": group.f1_macro.mean(),
            "macro_f1_std": group.f1_macro.std(ddof=1),
            "balanced_accuracy_mean": group.balanced_accuracy.mean(),
            "balanced_accuracy_std": group.balanced_accuracy.std(ddof=1),
            "mcc_mean": group.mcc.mean(),
            "recall_batik_mean": group.recall_batik.mean(),
            "recall_non_batik_mean": group.recall_non_batik.mean(),
            "n_repeats": len(group),
        })
    frame = pd.DataFrame(rows)
    frame["arm"] = pd.Categorical(frame.arm, categories=ARMS, ordered=True)
    return frame.sort_values("arm").reset_index(drop=True)


def reproduces_stage14(repeats) -> str:
    """Bandingkan lengan augmentasi dengan hasil tahap 14 yang sudah tersimpan.

    Lengan augmentasi di sini seharusnya mereproduksi tahap 14 secara persis
    karena seed, pembagian fold, ruang kandidat, dan aturan augmentasi sama.
    """
    reference = RESULTS_DIR / "14_repeated_nested_augmentation" / "repeat_metrics.csv"
    if not reference.exists():
        return "tahap 14 tidak tersedia untuk pembandingan"

    stage14 = pd.read_csv(reference).set_index("repeat").sort_index()
    mine = (
        repeats[repeats.arm == BASELINE_ARM]
        .drop(columns="arm")
        .set_index("repeat")
        .sort_index()
    )
    shared = [c for c in ("f1_macro", "balanced_accuracy", "mcc") if c in stage14.columns]
    deviations = {
        column: float(np.abs(mine[column] - stage14[column]).max())
        for column in shared
    }
    worst = max(deviations.values())
    if worst > 1e-9:
        raise AssertionError(
            f"Lengan augmentasi tidak mereproduksi tahap 14; deviasi {deviations}"
        )
    return f"lengan augmentasi mereproduksi tahap 14 (deviasi maks {worst:.2e})"


def paired_differences(outer_metrics):
    """Selisih berpasangan terhadap lengan augmentasi pada fold yang sama."""
    key = ["repeat", "outer_fold"]
    base = outer_metrics[outer_metrics.arm == BASELINE_ARM].set_index(key)
    rows = []
    for arm in ARMS:
        if arm == BASELINE_ARM:
            continue
        other = outer_metrics[outer_metrics.arm == arm].set_index(key)
        aligned = other.join(base, rsuffix="_base", how="inner")
        if len(aligned) != len(base):
            raise AssertionError(f"Fold {arm} tidak berpasangan penuh dengan baseline")
        for metric in ("f1_macro", "balanced_accuracy", "mcc"):
            diff = aligned[metric].to_numpy(float) - aligned[f"{metric}_base"].to_numpy(float)
            rows.append({
                "comparison": f"{arm} minus {BASELINE_ARM}",
                "metric": metric,
                "mean_difference": float(diff.mean()),
                "std_difference": float(diff.std(ddof=1)),
                "n_paired_folds": int(len(diff)),
                "folds_better": int((diff > 0).sum()),
                "folds_worse": int((diff < 0).sum()),
                "folds_tied": int((diff == 0).sum()),
                "min_difference": float(diff.min()),
                "max_difference": float(diff.max()),
            })
    return pd.DataFrame(rows)


def selection_frequency(outer_metrics):
    rows = []
    for arm, group in outer_metrics.groupby("arm", sort=False):
        for model, count in group.selected_model.value_counts().items():
            rows.append({
                "arm": arm,
                "kind": "model",
                "choice": model,
                "count": int(count),
                "of": len(group),
            })
        for feature_set, count in group.selected_feature_set.value_counts().items():
            rows.append({
                "arm": arm,
                "kind": "feature_set",
                "choice": feature_set,
                "count": int(count),
                "of": len(group),
            })
    return pd.DataFrame(rows)


def validate(outer_metrics, predictions, original):
    checks = []

    expected_folds = N_REPEATS * OUTER_SPLITS
    for arm in ARMS:
        subset = outer_metrics[outer_metrics.arm == arm]
        assert len(subset) == expected_folds, f"{arm}: {len(subset)} fold"
    checks.append(f"setiap lengan menghasilkan {expected_folds} outer fold")

    for arm in ARMS:
        subset = predictions[predictions.arm == arm]
        assert len(subset) == N_REPEATS * len(original), f"{arm}: prediksi tidak lengkap"
        for repeat, group in subset.groupby("repeat"):
            assert group.source_id.nunique() == len(original), (
                f"{arm} repeat {repeat}: original tidak diprediksi tepat sekali"
            )
    checks.append("setiap original diprediksi tepat sekali per repeat di semua lengan")

    key = ["repeat", "outer_fold"]
    reference = set(map(tuple, outer_metrics[outer_metrics.arm == BASELINE_ARM][key].values))
    for arm in ARMS:
        current = set(map(tuple, outer_metrics[outer_metrics.arm == arm][key].values))
        assert current == reference, f"{arm}: struktur fold berbeda dari baseline"
    checks.append("struktur fold identik pada ketiga lengan")

    for repeat in range(1, N_REPEATS + 1):
        assignments = {}
        for arm in ARMS:
            subset = predictions[(predictions.arm == arm) & (predictions.repeat == repeat)]
            assignments[arm] = dict(zip(subset.source_id, subset.outer_fold))
        base = assignments[BASELINE_ARM]
        for arm in ARMS:
            assert assignments[arm] == base, (
                f"{arm} repeat {repeat}: penempatan fold berbeda dari baseline"
            )
    checks.append("setiap original jatuh di outer fold yang sama pada ketiga lengan")

    augmented = outer_metrics[outer_metrics.arm == BASELINE_ARM]
    assert (augmented.train_batik == TARGET_PER_KELAS).all()
    assert (augmented.train_non_batik == TARGET_PER_KELAS).all()
    checks.append(f"lengan augmentasi selalu {TARGET_PER_KELAS} instance per kelas")

    for arm in ("class_weighted_originals", "balanced_original_sampling"):
        subset = outer_metrics[outer_metrics.arm == arm]
        assert (subset.train_derivatives == 0).all(), f"{arm} memakai derivative"
    checks.append("lengan pembanding tidak memakai satu pun derivative")

    balanced = outer_metrics[outer_metrics.arm == "balanced_original_sampling"]
    assert (balanced.train_batik == balanced.train_non_batik).all()
    checks.append("balanced-original sampling menghasilkan jumlah kelas yang sama")

    weighted = outer_metrics[outer_metrics.arm == "class_weighted_originals"]
    assert (weighted.train_batik > weighted.train_non_batik).all()
    checks.append("class weighting mempertahankan distribusi original yang timpang")

    return checks


REPORT = """# R2.7 - Perbandingan Strategi Penyeimbangan Kelas

Diagnostik sensitivitas untuk Reviewer 2 butir 7. Ketiga lengan memakai
pembagian outer/inner yang identik, ruang kandidat yang sama, dan hanya
original tanpa augmentasi pada inner validation serta outer test. Koleksi
eksternal tidak pernah dimuat.

- Desain: {repeats} repeat x {outer} outer fold x {inner} inner fold.
- Original development bersih: {n_original} ({n_batik} batik, {n_non_batik} non-batik).
- Ruang kandidat: {n_features} kelompok fitur x {n_models} keluarga model.
- Seed dasar: {seed}.

## Estimasi tingkat repeat

{summary}

## Selisih berpasangan terhadap lengan augmentasi

{paired}

## Frekuensi seleksi model

{selection}

## Catatan interpretasi

Perbandingan ini menilai jadwal penyeimbangan, bukan mengubah definisi fitur
atau keluarga model. Nilai di sini merupakan diagnostik sensitivitas internal
dan tidak dapat dipertukarkan dengan estimasi single-loop lima fold yang aktif.
Seluruh selisih dilaporkan berpasangan pada tingkat fold agar variasi antarfold
tidak disalahartikan sebagai efek strategi.
"""


def load_inputs():
    original_path = FEATURE_DIR / "development_original_features.csv"
    if not original_path.exists():
        raise FileNotFoundError("Jalankan 05_extract_features.py lebih dahulu")
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            "augmentation_feature_pool.csv tidak ada. Jalankan "
            "14_repeated_nested_augmentation.py lebih dahulu."
        )

    original = pd.read_csv(original_path).reset_index(drop=True)
    pool = pd.read_csv(POOL_PATH)

    expected_pool = len(original) * (DERIVATIVES_PER_ORIGINAL + 1)
    if len(pool) != expected_pool:
        raise AssertionError(f"Feature pool {len(pool)} baris, diharapkan {expected_pool}")
    if set(pool.loc[~pool.is_augmented, "source_id"]) != set(original.source_id):
        raise AssertionError("Original pada pool tidak sama dengan manifest development")
    return original, pool


def run_single_arm(arm: str, repeats=None) -> None:
    """Jalankan satu lengan (opsional sebagian repeat) dan simpan hasil sementara.

    Dipakai bila lingkungan eksekusi membatasi durasi satu perintah. Hasil
    numeriknya identik dengan menjalankan seluruh lengan sekaligus karena setiap
    repeat memakai seed dan pembagian fold yang ditentukan oleh nomor repeat,
    bukan oleh urutan eksekusi.
    """
    original, pool = load_inputs()
    PARTIAL.mkdir(parents=True, exist_ok=True)
    selected = tuple(repeats) if repeats else tuple(range(1, N_REPEATS + 1))
    tag = "all" if len(selected) == N_REPEATS else "r" + "-".join(str(r) for r in selected)
    print(f"Lengan: {arm} | repeat: {', '.join(str(r) for r in selected)}")
    outer_metrics, composition, predictions = run_arm(arm, original, pool, selected)
    outer_metrics.to_csv(PARTIAL / f"{arm}__{tag}__outer_metrics.csv", index=False)
    composition.to_csv(PARTIAL / f"{arm}__{tag}__composition.csv", index=False)
    predictions.to_csv(PARTIAL / f"{arm}__{tag}__predictions.csv", index=False)
    print(f"Tersimpan sementara: {PARTIAL}")


def collect_frames(original, pool, use_partial: bool):
    if not use_partial:
        metrics_frames, composition_frames, prediction_frames = [], [], []
        for arm in ARMS:
            print(f"\nLengan: {arm}")
            outer_metrics, composition, predictions = run_arm(arm, original, pool)
            metrics_frames.append(outer_metrics)
            composition_frames.append(composition)
            prediction_frames.append(predictions)
        return metrics_frames, composition_frames, prediction_frames

    metrics_frames, composition_frames, prediction_frames = [], [], []
    for arm in ARMS:
        collected = {}
        for kind, target in (
            ("outer_metrics", metrics_frames),
            ("composition", composition_frames),
            ("predictions", prediction_frames),
        ):
            parts = sorted(PARTIAL.glob(f"{arm}__*__{kind}.csv"))
            if not parts:
                raise FileNotFoundError(
                    f"Hasil sementara lengan '{arm}' ({kind}) belum ada. "
                    f"Jalankan lebih dahulu: --arm {arm}"
                )
            frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
            collected[kind] = frame
            target.append(frame)

        repeats_found = sorted(collected["outer_metrics"].repeat.unique())
        if repeats_found != list(range(1, N_REPEATS + 1)):
            raise AssertionError(
                f"Lengan '{arm}' baru memiliki repeat {repeats_found}; "
                f"diperlukan 1..{N_REPEATS}"
            )
    print("Menggabungkan hasil sementara ketiga lengan.")
    return metrics_frames, composition_frames, prediction_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=ARMS,
        help="Jalankan satu lengan saja dan simpan hasil sementara.",
    )
    parser.add_argument(
        "--repeats",
        help="Daftar repeat yang dijalankan, misalnya 1,2,3. Default seluruhnya.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Gabungkan hasil sementara ketiga lengan menjadi keluaran final.",
    )
    args = parser.parse_args()

    repeats = None
    if args.repeats:
        repeats = [int(value) for value in args.repeats.split(",") if value.strip()]
        invalid = [r for r in repeats if not 1 <= r <= N_REPEATS]
        if invalid:
            parser.error(f"Nomor repeat di luar 1..{N_REPEATS}: {invalid}")

    if args.arm:
        run_single_arm(args.arm, repeats)
        return

    original, pool = load_inputs()

    print("=" * 76)
    print("TAHAP 19 - PERBANDINGAN STRATEGI PENYEIMBANGAN KELAS (R2.7)")
    print("=" * 76)

    metrics_frames, composition_frames, prediction_frames = collect_frames(
        original, pool, use_partial=args.combine
    )

    outer_metrics = pd.concat(metrics_frames, ignore_index=True)
    composition = pd.concat(composition_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    checks = validate(outer_metrics, predictions, original)

    OUT.mkdir(parents=True, exist_ok=True)

    repeats = repeat_level(predictions, len(original))
    checks.append(reproduces_stage14(repeats))
    summary = arm_summary(repeats)
    paired = paired_differences(outer_metrics)
    selection = selection_frequency(outer_metrics)

    outer_metrics.to_csv(OUT / "outer_fold_metrics.csv", index=False)
    composition.to_csv(OUT / "training_composition.csv", index=False)
    predictions.to_csv(OUT / "outer_oof_predictions.csv", index=False)
    repeats.to_csv(OUT / "repeat_metrics.csv", index=False)
    summary.to_csv(OUT / "arm_summary.csv", index=False)
    paired.to_csv(OUT / "paired_differences.csv", index=False)
    selection.to_csv(OUT / "selection_frequency.csv", index=False)

    (OUT / "methodology.json").write_text(json.dumps({
        "analysis_role": "submission sensitivity diagnostic for R2.7",
        "arms": list(ARMS),
        "baseline_arm": BASELINE_ARM,
        "repeats": N_REPEATS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "candidate_space": f"{len(FEATURE_GROUPS)} feature groups x 3 model families",
        "split_grain": "development source photo group_id (tahap 24)",
        "validation_rule": "inner validation and outer test use originals only",
        "augmentation_rule": "only descendants of current training sources",
        "external_data_used": False,
        "base_random_seed": RANDOM_SEED,
        "assertions_passed": checks,
    }, indent=2), encoding="utf-8")

    def block(frame):
        return frame.to_string(index=False)

    (OUT / "REPORT.md").write_text(REPORT.format(
        repeats=N_REPEATS,
        outer=OUTER_SPLITS,
        inner=INNER_SPLITS,
        n_original=len(original),
        n_batik=int((original.kelas == "batik").sum()),
        n_non_batik=int((original.kelas == "non_batik").sum()),
        n_features=len(FEATURE_GROUPS),
        n_models=3,
        seed=RANDOM_SEED,
        summary=block(summary.round(6)),
        paired=block(paired.round(6)),
        selection=block(selection),
    ), encoding="utf-8")

    print("\n" + "=" * 76)
    print(summary.round(6).to_string(index=False))
    print("-" * 76)
    print(paired.round(6).to_string(index=False))
    print("-" * 76)
    for item in checks:
        print(f"  [OK] {item}")
    print("-" * 76)
    print("Keluaran:", OUT)


if __name__ == "__main__":
    main()
