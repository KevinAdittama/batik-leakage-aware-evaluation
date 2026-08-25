"""R2.4 - Ablasi terkontrol pada backbone ResNet18.

Permintaan reviewer
-------------------
    The handcrafted branch is grayscale and low-dimensional, while the deep
    branch uses RGB, ImageNet pretraining, center cropping, and much higher
    dimensional embeddings. Controlled ablations and the same nested selection
    process are required.

Perbandingan handcrafted versus deep pada manuskrip mengubah beberapa hal
sekaligus, sehingga selisih kinerjanya tidak dapat diatribusikan ke satu sebab.
Skrip ini memisahkan tiga di antaranya.

Faktor yang divariasikan
------------------------
    Warna         RGB versus grayscale yang direplikasi ke tiga kanal.
    Pretraining   Bobot ImageNet versus inisialisasi acak.
    Dimensi       Embedding 512-d penuh versus PCA 6 komponen.

2 x 2 x 2 menghasilkan delapan kondisi. PCA 6 komponen dipilih agar setara
dengan enam fitur handcrafted, sehingga keluhan dimensionalitas terjawab pada
jumlah dimensi yang sama persis.

Center cropping sengaja tidak divariasikan. Crop adalah bagian dari transform
bawaan bobot ImageNet, sehingga melepasnya mengubah pula resize dan normalisasi,
dan efeknya tidak dapat dipisahkan. Keputusan itu dinyatakan terbuka di REPORT.md
dan pada manuskrip, bukan disembunyikan.

Protokol
--------
Seluruh kondisi memakai protokol nested yang identik dengan tahap 19: 5 repeat x
5 outer fold x 4 inner fold, StratifiedGroupKFold pada `group_id`, seed sama
persis, sehingga pembagian foldnya benar-benar sama. Head classifier dipilih di
inner loop dari tiga keluarga yang sama dengan cabang handcrafted. Kelompok fitur
tidak ikut diseleksi karena embedding tidak punya kelompok.

Rambu yang dipegang
-------------------
1. Inner validation dan outer test hanya berisi original tanpa augmentasi.
2. Augmentasi fold-local: hanya turunan dari original pada split pelatihan.
3. PCA berada di dalam Pipeline sehingga hanya di-fit pada data latih.
4. Koleksi eksternal tidak pernah dimuat.
5. Seed tetap RANDOM_SEED = 42.
6. Ekstraksi embedding kondisi baseline harus identik dengan tahap 11. Tanpa
   pengaman itu, selisih antarkondisi bisa berasal dari perbedaan harness.

Catatan preprocessing: di sini ketiga head memakai StandardScaler, sedangkan pada
tahap 19 Random Forest tidak. Penyeragaman itu perlu agar faktor dimensionalitas
bersih, dan karena konstan di seluruh kondisi ia tidak memengaruhi perbandingan
antarkondisi. Konsekuensinya, angka di sini tidak dapat dibandingkan langsung
dengan tabel handcrafted.

Jalankan dari folder proyek:
    .\\env\\Scripts\\python.exe -B .\\30_controlled_ablation.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import models, transforms

from pipeline_common import read_image_color, resolve_project_path
from pipeline_config import FEATURE_DIR, RANDOM_SEED, RESULTS_DIR, TARGET_PER_KELAS

OUT = RESULTS_DIR / "30_controlled_ablation"
EMBEDDING_DIR = OUT / "embeddings"
STAGE11 = RESULTS_DIR / "11_deep_learning_baseline"
# Pool yang sama dengan tahap 19: sembilan derivative per original, dibangkitkan
# on-the-fly. Cv_pool tahap 03 terlalu kecil untuk protokol nested dan akan
# kehabisan derivative di inner loop.
POOL_PATH = RESULTS_DIR / "14_repeated_nested_augmentation" / "augmentation_feature_pool.csv"
DERIVATIVES_PER_ORIGINAL = 9

N_REPEATS = 5
OUTER_SPLITS = 5
INNER_SPLITS = 4
BATCH_SIZE = 24
PCA_COMPONENTS = 6
EMBEDDING_DIM = 512
BASELINE = ("rgb", "pretrained", "full512")
ANCHOR_TOLERANCE = 1e-5


def set_reproducible(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)


def build_backbone(color: str, pretraining: str):
    """ResNet18 dengan bobot ImageNet atau acak, plus transform yang sesuai.

    Transform ImageNet tetap dipakai pada varian acak agar hanya bobotnya yang
    berbeda. Kalau normalisasinya ikut diubah, faktor pretraining bercampur
    dengan faktor preprocessing.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    if color == "gray":
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3), transform]
        )
    set_reproducible()
    backbone = models.resnet18(weights=weights if pretraining == "pretrained" else None)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone, transform


def load_apply_transform():
    """Pakai fungsi augmentasi yang sama dengan tahap 03, lewat tahap 14."""
    spec = importlib.util.spec_from_file_location(
        "augment", Path(__file__).resolve().parent / "03_augment_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_transform


def to_tensor(image_bgr, transform):
    from PIL import Image

    rgb = image_bgr[:, :, ::-1]
    return transform(Image.fromarray(rgb))


@torch.inference_mode()
def extract_pool(color: str, pretraining: str, original: pd.DataFrame) -> tuple:
    """Embedding untuk seluruh pool, dengan derivative dibangkitkan ulang.

    Derivative tahap 14 tidak disimpan sebagai berkas; ia dibangkitkan dari
    original memakai RNG per sumber. Urutan dan seed di sini direplikasi persis
    agar baris embedding sejajar dengan baris pool.
    """
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    cache = EMBEDDING_DIR / f"resnet18_{color}_{pretraining}.npz"
    ordered = original.sort_values("source_id").reset_index(drop=True)
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        keys = [tuple(item) for item in stored["keys"]]
        return stored["features"], keys

    apply_transform = load_apply_transform()
    backbone, transform = build_backbone(color, pretraining)
    backbone.to(torch.device("cpu"))

    features, keys = [], []
    for source_index, row in enumerate(ordered.itertuples(index=False), 1):
        image = read_image_color(resolve_project_path(row.path))
        if image is None:
            raise OSError(f"Citra development tidak terbaca: {row.path}")
        batch = [to_tensor(image, transform)]
        keys.append((row.source_id, -1))
        rng = np.random.default_rng(RANDOM_SEED + 100_000 + source_index)
        for transform_index in range(DERIVATIVES_PER_ORIGINAL):
            augmented, _ = apply_transform(image, transform_index, rng)
            batch.append(to_tensor(augmented, transform))
            keys.append((row.source_id, transform_index))
        stacked = torch.stack(batch)
        features.append(backbone(stacked).detach().cpu().numpy().astype(np.float32))
        if source_index % 50 == 0 or source_index == len(ordered):
            print(f"    {color}/{pretraining}: {source_index}/{len(ordered)} original")

    matrix = np.vstack(features)
    np.savez_compressed(
        cache, keys=np.array(keys, dtype=object), features=matrix
    )
    print(f"  embedding {color}/{pretraining}: {matrix.shape} -> {cache.name}")
    return matrix, keys


@torch.inference_mode()
def extract_originals(color: str, pretraining: str, paths: list[str]) -> np.ndarray:
    """Embedding original saja, dipakai untuk pengaman terhadap tahap 11."""
    from PIL import Image

    backbone, transform = build_backbone(color, pretraining)
    backbone.to(torch.device("cpu"))
    batch, out = [], []
    for index, path in enumerate(paths, 1):
        with Image.open(resolve_project_path(path)) as image:
            batch.append(transform(image.convert("RGB")))
        if len(batch) == BATCH_SIZE or index == len(paths):
            out.append(backbone(torch.stack(batch)).detach().cpu().numpy().astype(np.float32))
            batch = []
    return np.vstack(out)


def anchor_against_stage11(paths: list[str], _unused=None) -> dict:
    """Buktikan ekstraksi baseline identik dengan tahap 11.

    Perbedaan sekecil apa pun di sini akan merambat ke seluruh kondisi, sehingga
    pemeriksaan ini dijalankan sebelum ablasinya, bukan sesudah.
    """
    spec = importlib.util.spec_from_file_location(
        "stage11", Path(__file__).resolve().parent / "11_eval_deep_learning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    backbone, transform, dim, _ = module.build_feature_extractor("ResNet18")
    frame = pd.DataFrame({"path": paths})
    reference = module.extract_embeddings(
        backbone, transform, frame, "path", torch.device("cpu"), "anchor"
    )
    here = extract_originals("rgb", "pretrained", paths)
    deviation = float(np.abs(reference - here).max())
    return {
        "embedding_dim_stage11": int(dim),
        "embedding_dim_here": int(here.shape[1]),
        "n_originals_compared": len(paths),
        "max_abs_deviation": deviation,
        "within_tolerance": deviation <= ANCHOR_TOLERANCE,
        "tolerance": ANCHOR_TOLERANCE,
    }


def build_heads(dimensionality: str) -> dict:
    """Tiga keluarga head yang sama, dengan PCA di dalam Pipeline bila diminta."""
    def wrap(name: str, estimator):
        steps = [("scaler", StandardScaler())]
        if dimensionality == "pca6":
            steps.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)))
        steps.append(("model", estimator))
        return name, Pipeline(steps)

    return dict([
        wrap("Logistic Regression", LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED)),
        wrap("SVM (RBF)", SVC(kernel="rbf", random_state=RANDOM_SEED)),
        wrap("Random Forest", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)),
    ])


def metric_values(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "recall_non_batik": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_batik": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def augmented_balanced(pool: pd.DataFrame, train_sources: set, seed: int) -> pd.DataFrame:
    """Sama persis dengan lengan augmentasi tahap 19: 200 instance per kelas."""
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


def run_condition(condition: tuple, original, pool, features, row_of) -> tuple:
    color, pretraining, dimensionality = condition
    name = f"{color}+{pretraining}+{dimensionality}"
    heads = build_heads(dimensionality)
    y = original.label.to_numpy(int)
    groups = original.group_id.to_numpy()

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        if "transform_index" in frame.columns:
            keys = zip(frame.source_id, frame.transform_index)
        else:
            keys = ((source_id, -1) for source_id in frame.source_id)
        return features[[row_of[key] for key in keys]]

    outer_rows, prediction_frames = [], []
    for repeat in range(1, N_REPEATS + 1):
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
                if inner_train_sources & set(inner_valid.source_id):
                    raise AssertionError("Kebocoran sumber inner train/validation")
                training = augmented_balanced(pool, inner_train_sources, inner_seed + inner_fold)
                if not set(training.source_id).issubset(inner_train_sources):
                    raise AssertionError("Sumber di luar split pelatihan ikut terpakai")
                prepared.append((training, inner_valid))

            scored = []
            for order, (model_name, estimator) in enumerate(heads.items()):
                scores = []
                for training, inner_valid in prepared:
                    model = clone(estimator)
                    model.fit(matrix(training), training.label.to_numpy(int))
                    prediction = model.predict(matrix(inner_valid))
                    scores.append(f1_score(
                        inner_valid.label.to_numpy(int), prediction,
                        average="macro", zero_division=0,
                    ))
                scored.append((float(np.mean(scores)), order, model_name, estimator))

            best_score, _, model_name, estimator = sorted(
                scored, key=lambda item: (-item[0], item[1])
            )[0]

            outer_training = augmented_balanced(
                pool, train_sources, outer_seed + outer_fold * 100 + 90
            )
            model = clone(estimator)
            model.fit(matrix(outer_training), outer_training.label.to_numpy(int))
            prediction = model.predict(matrix(outer_test)).astype(int)
            metrics = metric_values(outer_test.label.to_numpy(int), prediction)

            outer_rows.append({
                "condition": name, "color": color, "pretraining": pretraining,
                "dimensionality": dimensionality, "repeat": repeat,
                "outer_fold": outer_fold, "selected_model": model_name,
                "inner_macro_f1": best_score,
                "train_instances": len(outer_training),
                "test_originals": len(outer_test), **metrics,
            })
            predicted = outer_test[["source_id", "group_id", "kelas", "label", "subjenis"]].copy()
            predicted.insert(0, "outer_fold", outer_fold)
            predicted.insert(0, "repeat", repeat)
            predicted.insert(0, "condition", name)
            predicted["predicted_label"] = prediction
            predicted["correct"] = predicted.label.to_numpy(int) == prediction
            prediction_frames.append(predicted)
        print(f"  [{name}] repeat {repeat}/{N_REPEATS} selesai")

    return pd.DataFrame(outer_rows), pd.concat(prediction_frames, ignore_index=True)


def repeat_level(predictions: pd.DataFrame, n_original: int) -> pd.DataFrame:
    rows = []
    for (condition, repeat), block in predictions.groupby(["condition", "repeat"]):
        if len(block) != n_original:
            raise AssertionError(f"{condition} repeat {repeat}: prediksi tidak lengkap")
        rows.append({
            "condition": condition, "repeat": repeat,
            **metric_values(block.label.to_numpy(int), block.predicted_label.to_numpy(int)),
        })
    return pd.DataFrame(rows)


def condition_summary(repeats: pd.DataFrame) -> pd.DataFrame:
    grouped = repeats.groupby("condition")
    return pd.DataFrame({
        "macro_f1_mean": grouped.f1_macro.mean(),
        "macro_f1_std": grouped.f1_macro.std(ddof=1),
        "balanced_accuracy_mean": grouped.balanced_accuracy.mean(),
        "mcc_mean": grouped.mcc.mean(),
        "recall_batik_mean": grouped.recall_batik.mean(),
        "recall_non_batik_mean": grouped.recall_non_batik.mean(),
        "n_repeats": grouped.size(),
    }).reset_index()


def paired_differences(outer: pd.DataFrame) -> pd.DataFrame:
    """Selisih berpasangan pada tingkat fold untuk tiap faktor.

    Rata-rata saja menyembunyikan bahwa fold yang sama bisa bergerak berlawanan
    arah, jadi jumlah fold yang membaik dan memburuk ikut dilaporkan.
    """
    key = ["repeat", "outer_fold"]
    factors = {
        "color": ("rgb", "gray"),
        "pretraining": ("pretrained", "random"),
        "dimensionality": ("full512", "pca6"),
    }
    rows = []
    for factor, (level_a, level_b) in factors.items():
        others = [name for name in factors if name != factor]
        for combination in outer[others].drop_duplicates().to_dict("records"):
            mask = np.ones(len(outer), dtype=bool)
            for name, value in combination.items():
                mask &= outer[name].to_numpy() == value
            subset = outer[mask]
            side_a = subset[subset[factor] == level_a].set_index(key).sort_index()
            side_b = subset[subset[factor] == level_b].set_index(key).sort_index()
            if not side_a.index.equals(side_b.index):
                raise AssertionError(f"Fold tidak sejajar untuk faktor {factor}")
            delta = side_a.f1_macro - side_b.f1_macro
            rows.append({
                "factor": factor,
                "contrast": f"{level_a} minus {level_b}",
                "holding": ", ".join(f"{k}={v}" for k, v in combination.items()),
                "mean_difference": float(delta.mean()),
                "std_difference": float(delta.std(ddof=1)),
                "n_paired_folds": int(len(delta)),
                "folds_better": int((delta > 0).sum()),
                "folds_worse": int((delta < 0).sum()),
                "folds_tied": int((delta == 0).sum()),
                "min_difference": float(delta.min()),
                "max_difference": float(delta.max()),
            })
    return pd.DataFrame(rows)


def factor_effect(paired: pd.DataFrame) -> pd.DataFrame:
    """Efek rata-rata tiap faktor di seluruh kombinasi faktor lainnya."""
    grouped = paired.groupby(["factor", "contrast"])
    return pd.DataFrame({
        "mean_difference": grouped.mean_difference.mean(),
        "min_across_settings": grouped.mean_difference.min(),
        "max_across_settings": grouped.mean_difference.max(),
        "settings": grouped.size(),
    }).reset_index()


def selection_frequency(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, block in outer.groupby("condition"):
        counts = block.selected_model.value_counts()
        for model, count in counts.items():
            rows.append({
                "condition": condition, "choice": model,
                "count": int(count), "of": int(len(block)),
            })
    return pd.DataFrame(rows)


def load_inputs():
    original = pd.read_csv(FEATURE_DIR / "development_original_features.csv")
    missing = {"path", "source_id", "group_id", "kelas", "label"} - set(original.columns)
    if missing:
        raise AssertionError(f"Kolom hilang pada original: {sorted(missing)}")
    if original.source_id.duplicated().any():
        raise AssertionError("source_id development tidak unik")

    if not POOL_PATH.exists():
        raise FileNotFoundError(
            "augmentation_feature_pool.csv tidak ada. Jalankan "
            "14_repeated_nested_augmentation.py lebih dahulu."
        )
    pool = pd.read_csv(POOL_PATH)
    expected = len(original) * (DERIVATIVES_PER_ORIGINAL + 1)
    if len(pool) != expected:
        raise AssertionError(f"Pool {len(pool)} baris, diharapkan {expected}")
    if set(pool.loc[~pool.is_augmented, "source_id"]) != set(original.source_id):
        raise AssertionError("Original pada pool tidak sama dengan manifest development")
    return original, pool


def write_report(summary, paired, effects, selection, anchor, checks) -> None:
    baseline_name = "+".join(BASELINE)
    ordered = summary.sort_values("macro_f1_mean", ascending=False)
    lines = [
        "# Tahap 30 - Ablasi terkontrol ResNet18 (R2.4)",
        "",
        "Tiga faktor divariasikan satu per satu pada fold nested yang identik: warna,",
        "pretraining, dan dimensionalitas. Center cropping sengaja tidak divariasikan;",
        "alasannya ada di bagian batas di bawah.",
        "",
        "## Pengaman ekstraksi",
        "",
        f"- Deviasi maksimum embedding baseline terhadap tahap 11: `{anchor['max_abs_deviation']:.3e}`",
        f"- Toleransi: `{anchor['tolerance']:.0e}`",
        f"- Lolos: **{anchor['within_tolerance']}**",
        "",
        "Tanpa pengaman ini, selisih antarkondisi bisa berasal dari perbedaan harness",
        "dan bukan dari faktor yang divariasikan.",
        "",
        "## Kinerja per kondisi",
        "",
        "| Kondisi | Macro-F1 | Bal. accuracy | MCC | Recall B | Recall NB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in ordered.to_dict("records"):
        marker = " (baseline)" if record["condition"] == baseline_name else ""
        lines.append(
            f"| {record['condition']}{marker} | "
            f"{record['macro_f1_mean']:.3f} +/- {record['macro_f1_std']:.3f} | "
            f"{record['balanced_accuracy_mean']:.3f} | {record['mcc_mean']:.3f} | "
            f"{record['recall_batik_mean']:.3f} | {record['recall_non_batik_mean']:.3f} |"
        )

    lines += [
        "",
        "## Efek tiap faktor",
        "",
        "Selisih macro-F1 dihitung berpasangan pada tingkat fold, lalu dirata-ratakan",
        "di seluruh kombinasi faktor lainnya. Rentang menunjukkan apakah efeknya",
        "konsisten atau bergantung pada setelan lain.",
        "",
        "| Faktor | Kontras | Rata-rata | Terendah | Tertinggi |",
        "|---|---|---:|---:|---:|",
    ]
    for record in effects.to_dict("records"):
        lines.append(
            f"| {record['factor']} | {record['contrast']} | "
            f"{record['mean_difference']:+.4f} | {record['min_across_settings']:+.4f} | "
            f"{record['max_across_settings']:+.4f} |"
        )

    lines += [
        "",
        "## Batas ablasi ini",
        "",
        "Center cropping tidak divariasikan. Crop merupakan bagian dari transform",
        "bawaan bobot ImageNet, sehingga melepasnya turut mengubah resize dan",
        "normalisasi; efeknya tidak dapat dipisahkan dari faktor lain dan hasilnya",
        "akan sulit ditafsirkan. Reviewer menyebut crop secara eksplisit, jadi",
        "pembatasan ini dinyatakan terbuka, bukan diabaikan.",
        "",
        "Ablasi ini juga memakai satu backbone saja. Hasilnya berlaku untuk ResNet18",
        "pada koleksi ini dan tidak digeneralisasi ke arsitektur lain.",
        "",
        "Ketiga head memakai StandardScaler, berbeda dari tahap 19 yang tidak",
        "menskalakan Random Forest. Penyeragaman itu konstan di seluruh kondisi",
        "sehingga tidak memengaruhi perbandingan antarkondisi, tetapi angka di sini",
        "tidak dapat dibandingkan langsung dengan tabel handcrafted.",
        "",
        "## Assertion yang lolos",
        "",
    ]
    lines += [f"- {check}" for check in checks]
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-anchor", action="store_true",
                        help="Lewati pemeriksaan terhadap tahap 11 (untuk debug saja).")
    args = parser.parse_args()

    original, pool = load_inputs()

    print("=" * 76)
    print("TAHAP 30 - ABLASI TERKONTROL RESNET18 (R2.4)")
    print("=" * 76)
    print(f"baris pool: {len(pool)} | original: {len(original)} | "
          f"grup: {original.group_id.nunique()}")

    print("ekstraksi embedding empat varian backbone")
    embeddings, row_of = {}, None
    for color, pretraining in product(("rgb", "gray"), ("pretrained", "random")):
        matrix, keys = extract_pool(color, pretraining, original)
        embeddings[(color, pretraining)] = matrix
        current = {key: index for index, key in enumerate(keys)}
        if row_of is None:
            row_of = current
        elif row_of != current:
            raise AssertionError("Urutan baris embedding berbeda antar varian")

    pool_keys = set(zip(pool.source_id, pool.transform_index))
    if pool_keys != set(row_of):
        raise AssertionError("Kunci embedding tidak sejajar dengan baris pool")

    checks = []
    anchor = {"max_abs_deviation": float("nan"), "within_tolerance": None,
              "tolerance": ANCHOR_TOLERANCE}
    if not args.skip_anchor:
        print("pengaman: bandingkan embedding baseline dengan tahap 11")
        anchor = anchor_against_stage11(original.path.tolist(), None)
        if not anchor["within_tolerance"]:
            raise AssertionError(
                "Embedding baseline tidak identik dengan tahap 11 "
                f"(deviasi {anchor['max_abs_deviation']:.3e})"
            )
        checks.append(
            f"ekstraksi baseline identik dengan tahap 11 "
            f"(deviasi maks {anchor['max_abs_deviation']:.2e})"
        )

    outer_frames, prediction_frames = [], []
    for condition in product(("rgb", "gray"), ("pretrained", "random"), ("full512", "pca6")):
        color, pretraining, _ = condition
        outer, predictions = run_condition(
            condition, original, pool, embeddings[(color, pretraining)], row_of
        )
        outer_frames.append(outer)
        prediction_frames.append(predictions)

    outer = pd.concat(outer_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    expected_folds = N_REPEATS * OUTER_SPLITS
    for condition, block in outer.groupby("condition"):
        if len(block) != expected_folds:
            raise AssertionError(f"{condition}: {len(block)} outer fold, bukan {expected_folds}")
    checks.append(f"setiap kondisi menghasilkan {expected_folds} outer fold")

    reference = (
        predictions[predictions.condition == "+".join(BASELINE)]
        .set_index(["repeat", "source_id"]).sort_index()
    )
    for condition, block in predictions.groupby("condition"):
        aligned = block.set_index(["repeat", "source_id"]).sort_index()
        if not aligned.index.equals(reference.index):
            raise AssertionError(f"{condition}: pembagian fold berbeda dari baseline")
        if not (aligned.outer_fold == reference.outer_fold).all():
            raise AssertionError(f"{condition}: original jatuh di outer fold berbeda")
    checks.append("seluruh kondisi memakai pembagian fold yang identik")

    repeats = repeat_level(predictions, len(original))
    summary = condition_summary(repeats)
    paired = paired_differences(outer)
    effects = factor_effect(paired)
    selection = selection_frequency(outer)
    checks.append("setiap original diprediksi tepat sekali per repeat di semua kondisi")

    OUT.mkdir(parents=True, exist_ok=True)
    outer.to_csv(OUT / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "outer_oof_predictions.csv", index=False)
    repeats.to_csv(OUT / "repeat_metrics.csv", index=False)
    summary.to_csv(OUT / "condition_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_differences.csv", index=False)
    effects.to_csv(OUT / "factor_effects.csv", index=False)
    selection.to_csv(OUT / "selection_frequency.csv", index=False)
    (OUT / "anchor_check.json").write_text(json.dumps(anchor, indent=2), encoding="utf-8")

    (OUT / "methodology.json").write_text(json.dumps({
        "analysis_role": "controlled ablation requested by Reviewer 2 item 4",
        "backbone": "ResNet18",
        "factors": {
            "color": ["rgb", "gray"],
            "pretraining": ["pretrained", "random"],
            "dimensionality": ["full512", f"pca{PCA_COMPONENTS}"],
        },
        "n_conditions": 8,
        "not_varied": "center cropping; part of the ImageNet weight transform",
        "repeats": N_REPEATS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "split_grain": "development source photo group_id (tahap 24)",
        "candidate_space": "3 classifier families; feature groups not applicable",
        "validation_rule": "inner validation and outer test use originals only",
        "augmentation_rule": "only descendants of current training sources",
        "pca_inside_pipeline": True,
        "external_data_used": False,
        "base_random_seed": RANDOM_SEED,
        "anchor": anchor,
        "assertions_passed": checks,
    }, indent=2), encoding="utf-8")

    write_report(summary, paired, effects, selection, anchor, checks)

    print("-" * 76)
    print(summary.sort_values("macro_f1_mean", ascending=False).to_string(index=False))
    print("-" * 76)
    print(effects.to_string(index=False))
    print("-" * 76)
    for check in checks:
        print(f"  [OK] {check}")
    print("Keluaran:", OUT)


if __name__ == "__main__":
    main()
