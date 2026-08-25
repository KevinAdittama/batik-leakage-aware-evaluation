"""Stabilitas seleksi model formal pada protokol file-level versus group-aware.

Latar belakang
--------------
Aturan seleksi formal artikel ini adalah macro-F1 CV tertinggi pada enam fitur
gabungan. Aturan itu tidak berubah. Yang berubah adalah pemenangnya.

Setelah fold dipisahkan pada grain foto sumber (tahap 24), Random Forest dan
SVM-RBF berpindah tempat pada seleksi single-loop, dan pada seleksi nested
berulang preferensinya runtuh dari 20 banding 5 menjadi hampir seimbang. Itu
bukan pergeseran angka kecil; itu aturan seleksi yang kehilangan daya pisah.

Skrip ini mengukurnya, bukan menyembunyikannya. Pertanyaan yang dijawab: apakah
selisih antara dua model teratas cukup besar dibandingkan derau antar-fold untuk
membenarkan penobatan salah satu sebagai model formal?

Metode
------
Tiga sudut pandang, seluruhnya dari keluaran yang sudah ada, tidak ada model
yang dilatih ulang di sini.

  1. Margin single-loop.
        Selisih macro-F1 antara peringkat 1 dan 2, dibandingkan dengan
        simpangan baku antar-fold dan galat baku rata-ratanya.

  2. Frekuensi seleksi nested.
        Berapa dari 25 outer fold pada tahap 14 memilih tiap model, dan hal yang
        sama pada tiga lengan tahap 19.

  3. Perbandingan protokol.
        Angka file-level yang diarsipkan versus angka group-aware sekarang,
        untuk metrik dan frekuensi seleksi yang sama.

Keluaran eksternal sengaja ikut dilaporkan untuk seluruh model, bukan hanya
model formal, karena justru itu inti temuannya: menobatkan satu pemenang dari
margin yang lebih kecil daripada deraunya sendiri tidak dapat dipertanggung
jawabkan.

Skrip ini tidak melatih model, tidak mengubah fold, dan tidak menyentuh
manuskrip.

Jalankan dari folder proyek:
    .\\env\\Scripts\\python.exe -B .\\25_model_selection_stability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import PROJECT_DIR, RANDOM_SEED, RESULTS_DIR

OUT = RESULTS_DIR / "25_model_selection_stability"
ARCHIVE = (
    PROJECT_DIR
    / "IJIES_REVISI_FINAL"
    / "99_Arsip_Pendukung"
    / "Hasil_Sebelum_Group_Aware_20260819"
    / "hasil_paper"
)

PRIMARY_FEATURE_SET = "Gabungan 6 Fitur"


def single_loop_margin(base: Path) -> tuple[pd.DataFrame, dict]:
    """Peringkat single-loop beserta margin puncak relatif terhadap derau."""
    summary = (
        pd.read_csv(base / "07_cross_validation/cv_summary_primary.csv")
        .sort_values("f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )
    folds = pd.read_csv(base / "07_cross_validation/cv_fold_metrics.csv")
    folds = folds[folds.feature_set == PRIMARY_FEATURE_SET]
    n_folds = int(folds.groupby("model").size().max())

    top, runner_up = summary.iloc[0], summary.iloc[1]
    margin = float(top.f1_macro_mean - runner_up.f1_macro_mean)
    pooled_std = float(summary.f1_macro_std.mean())
    standard_error = pooled_std / np.sqrt(n_folds)

    return summary, {
        "winner": str(top.model),
        "runner_up": str(runner_up.model),
        "winner_macro_f1": float(top.f1_macro_mean),
        "runner_up_macro_f1": float(runner_up.f1_macro_mean),
        "margin": margin,
        "pooled_fold_std": pooled_std,
        "standard_error_of_mean": float(standard_error),
        "margin_in_standard_errors": float(margin / standard_error) if standard_error else float("nan"),
        "n_folds": n_folds,
    }


def nested_selection(base: Path) -> pd.DataFrame:
    """Frekuensi model terpilih pada seluruh protokol nested yang tersedia."""
    rows = []

    stage12 = base / "12_submission_robustness/nested_cv_outer_fold_metrics.csv"
    if stage12.exists():
        counts = pd.read_csv(stage12).selected_model.value_counts()
        for model, count in counts.items():
            rows.append({"protocol": "nested single-repeat (tahap 12)",
                         "arm": "-", "model": model,
                         "selections": int(count), "of": int(counts.sum())})

    stage14 = base / "14_repeated_nested_augmentation/outer_fold_metrics.csv"
    if stage14.exists():
        counts = pd.read_csv(stage14).selected_model.value_counts()
        for model, count in counts.items():
            rows.append({"protocol": "nested 5 repeat (tahap 14)",
                         "arm": "fold-local augmentation", "model": model,
                         "selections": int(count), "of": int(counts.sum())})

    stage19 = base / "19_balancing_comparison/outer_fold_metrics.csv"
    if stage19.exists():
        frame = pd.read_csv(stage19)
        for arm, block in frame.groupby("arm"):
            counts = block.selected_model.value_counts()
            for model, count in counts.items():
                rows.append({"protocol": "nested 5 repeat (tahap 19)",
                             "arm": arm, "model": model,
                             "selections": int(count), "of": int(counts.sum())})

    return pd.DataFrame(rows)


def external_by_model(base: Path) -> pd.DataFrame:
    classical = pd.read_csv(base / "08_uji_eksternal/external_model_summary.csv")
    classical["family"] = "handcrafted six features"
    deep = pd.read_csv(base / "11_deep_learning_baseline/dl_external_summary.csv")
    deep["family"] = "frozen deep benchmark"
    shared = ["model", "family", "balanced_accuracy", "f1_macro", "mcc",
              "recall_batik", "recall_non_batik"]
    return pd.concat(
        [classical[[c for c in shared if c in classical.columns]],
         deep[[c for c in shared if c in deep.columns]]],
        ignore_index=True,
    )


def protocol_comparison() -> pd.DataFrame:
    """File-level yang diarsipkan versus group-aware sekarang."""
    if not ARCHIVE.exists():
        return pd.DataFrame()
    rows = []
    for label, base in (("file-level", ARCHIVE), ("source-group-aware", RESULTS_DIR)):
        summary = pd.read_csv(base / "07_cross_validation/cv_summary_primary.csv")
        for record in summary.to_dict("records"):
            rows.append({"protocol": label, "source": "single-loop CV (tahap 07)",
                         "model": record["model"],
                         "macro_f1_mean": record["f1_macro_mean"],
                         "macro_f1_std": record["f1_macro_std"]})
        nested = pd.read_csv(base / "14_repeated_nested_augmentation/summary.csv").set_index("metric")
        rows.append({"protocol": label, "source": "nested 5 repeat (tahap 14)",
                     "model": "protocol mean",
                     "macro_f1_mean": float(nested.loc["f1_macro", "repeat_mean"]),
                     "macro_f1_std": float(nested.loc["f1_macro", "repeat_std"])})
    return pd.DataFrame(rows)


def write_report(margin_now: dict, margin_before: dict | None, selection: pd.DataFrame,
                 selection_before: pd.DataFrame, external: pd.DataFrame,
                 comparison: pd.DataFrame, checks: list[str]) -> None:
    lines = [
        "# Tahap 25 - Stabilitas seleksi model formal",
        "",
        "Aturan seleksi tidak berubah: macro-F1 CV tertinggi pada enam fitur",
        "gabungan. Yang diukur di sini adalah apakah aturan itu masih memisahkan",
        "model secara bermakna setelah fold dipisahkan pada grain foto sumber.",
        "",
        "## Margin single-loop",
        "",
        "| Protokol | Pemenang | Peringkat 2 | Margin | SD antar-fold | Galat baku | Margin dalam galat baku |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for label, data in (("file-level", margin_before), ("source-group-aware", margin_now)):
        if data is None:
            continue
        lines.append(
            f"| {label} | {data['winner']} ({data['winner_macro_f1']:.4f}) | "
            f"{data['runner_up']} ({data['runner_up_macro_f1']:.4f}) | "
            f"{data['margin']:.4f} | {data['pooled_fold_std']:.4f} | "
            f"{data['standard_error_of_mean']:.4f} | {data['margin_in_standard_errors']:.2f} |"
        )
    lines += [
        "",
        "Margin sebesar sepersekian galat baku tidak dapat membedakan dua model.",
        "Penobatan pemenang formal karena itu tidak dapat dipertanggungjawabkan",
        "sebagai klaim keunggulan.",
        "",
        "## Frekuensi seleksi nested",
        "",
        "| Protokol | Lengan | Model | file-level | source-group-aware |",
        "|---|---|---|---:|---:|",
    ]
    before_index = {
        (r["protocol"], r["arm"], r["model"]): r["selections"]
        for r in selection_before.to_dict("records")
    } if not selection_before.empty else {}
    for record in selection.to_dict("records"):
        key = (record["protocol"], record["arm"], record["model"])
        before = before_index.get(key, 0)
        lines.append(
            f"| {record['protocol']} | {record['arm']} | {record['model']} | "
            f"{before}/{record['of']} | {record['selections']}/{record['of']} |"
        )

    lines += [
        "",
        "## Hasil eksternal seluruh model",
        "",
        "Dilaporkan untuk semua model, bukan hanya model formal. Hasil eksternal",
        "tidak bergantung pada struktur fold, sehingga identik pada kedua protokol.",
        "",
        "| Model | Keluarga | Bal. accuracy | Macro-F1 | MCC | Recall batik | Recall non-batik |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in external.sort_values("f1_macro", ascending=False).to_dict("records"):
        lines.append(
            f"| {record['model']} | {record['family']} | {record['balanced_accuracy']:.3f} | "
            f"{record['f1_macro']:.3f} | {record['mcc']:.3f} | "
            f"{record['recall_batik']:.3f} | {record['recall_non_batik']:.3f} |"
        )

    lines += [
        "",
        "## Batas pembacaan",
        "",
        "Dua hal berubah bersamaan antara kedua protokol: tujuh citra kini terkunci",
        "dalam grup yang sama, dan `StratifiedGroupKFold` mempartisi secara berbeda",
        "dari `StratifiedKFold`. Selisih di sini karena itu adalah sensitivitas",
        "terhadap protokol, bukan estimasi kausal atas besarnya kebocoran.",
        "",
        "## Assertion yang lolos",
        "",
    ]
    lines += [f"- {check}" for check in checks]
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary_now, margin_now = single_loop_margin(RESULTS_DIR)
    margin_before = None
    selection_before = pd.DataFrame()
    if ARCHIVE.exists():
        _, margin_before = single_loop_margin(ARCHIVE)
        selection_before = nested_selection(ARCHIVE)

    selection = nested_selection(RESULTS_DIR)
    external = external_by_model(RESULTS_DIR)
    comparison = protocol_comparison()

    checks = []
    stored = json.loads(
        (RESULTS_DIR / "08_uji_eksternal/selected_model_result.json").read_text(encoding="utf-8")
    )
    if stored["model"] != margin_now["winner"]:
        raise AssertionError(
            f"Model tersimpan {stored['model']!r} bukan pemenang tabel CV "
            f"{margin_now['winner']!r}"
        )
    checks.append("model formal tersimpan sama dengan pemenang tabel CV")

    for protocol, block in selection.groupby(["protocol", "arm"]):
        totals = block["of"].unique()
        if len(totals) != 1 or int(block.selections.sum()) != int(totals[0]):
            raise AssertionError(f"Frekuensi seleksi tidak menjumlah penuh: {protocol}")
    checks.append("frekuensi seleksi menjumlah ke seluruh outer fold pada tiap protokol")

    if margin_now["margin"] >= margin_now["pooled_fold_std"]:
        checks.append("PERINGATAN: margin puncak melampaui simpangan antar-fold")
    else:
        checks.append("margin puncak lebih kecil daripada simpangan antar-fold")

    OUT.mkdir(parents=True, exist_ok=True)
    summary_now.to_csv(OUT / "single_loop_ranking.csv", index=False)
    selection.to_csv(OUT / "nested_selection_frequency.csv", index=False)
    external.to_csv(OUT / "external_by_model.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(OUT / "protocol_comparison.csv", index=False)
    pd.DataFrame([
        {"protocol": "file-level", **(margin_before or {})},
        {"protocol": "source-group-aware", **margin_now},
    ]).to_csv(OUT / "selection_margin.csv", index=False)

    (OUT / "methodology.json").write_text(json.dumps({
        "analysis_role": "selection-stability diagnostic; no model is retrained here",
        "selection_rule": "highest mean CV macro-F1 on the combined six features",
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "inputs": [
            "hasil_paper/07_cross_validation/cv_summary_primary.csv",
            "hasil_paper/07_cross_validation/cv_fold_metrics.csv",
            "hasil_paper/12_submission_robustness/nested_cv_outer_fold_metrics.csv",
            "hasil_paper/14_repeated_nested_augmentation/outer_fold_metrics.csv",
            "hasil_paper/19_balancing_comparison/outer_fold_metrics.csv",
            "hasil_paper/08_uji_eksternal/external_model_summary.csv",
            "hasil_paper/11_deep_learning_baseline/dl_external_summary.csv",
        ],
        "archive_compared": ARCHIVE.relative_to(PROJECT_DIR).as_posix() if ARCHIVE.exists() else None,
        "random_seed": RANDOM_SEED,
        "caveat": "protocol sensitivity, not a causal estimate of leakage magnitude",
        "assertions_passed": checks,
    }, indent=2), encoding="utf-8")

    write_report(margin_now, margin_before, selection, selection_before,
                 external, comparison, checks)

    print("=" * 76)
    print("TAHAP 25 - STABILITAS SELEKSI MODEL FORMAL")
    print("=" * 76)
    for label, data in (("file-level", margin_before), ("source-group-aware", margin_now)):
        if data is None:
            continue
        print(f"{label:>20s}: {data['winner']} {data['winner_macro_f1']:.4f} vs "
              f"{data['runner_up']} {data['runner_up_macro_f1']:.4f} | "
              f"margin {data['margin']:.4f} = {data['margin_in_standard_errors']:.2f} galat baku")
    print("-" * 76)
    pivot = selection.pivot_table(index=["protocol", "arm"], columns="model",
                                  values="selections", fill_value=0)
    print(pivot.to_string())
    print("-" * 76)
    for check in checks:
        print(f"  [OK] {check}")
    print("Keluaran:", OUT)


if __name__ == "__main__":
    main()
