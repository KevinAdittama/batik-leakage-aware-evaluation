"""Regresi untuk ablasi terkontrol ResNet18 (tahap 30, butir R2.4).

Metrik dihitung ulang dari prediksi mentah, bukan dibaca dari ringkasan yang
ditulis skripnya sendiri. Kalau skrip salah meringkas, tes ini menangkapnya.

Yang paling penting diuji di sini bukan angkanya, melainkan keadilan
perbandingannya: kedelapan kondisi harus memakai pembagian fold yang benar-benar
sama, dan kondisi baseline harus mereproduksi ekstraksi tahap 11. Tanpa keduanya,
selisih antarkondisi tidak dapat diatribusikan ke faktor yang divariasikan.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
STAGE30 = ROOT / "hasil_paper" / "30_controlled_ablation"

N_REPEATS = 5
OUTER_SPLITS = 5
N_ORIGINAL = 201
N_CONDITIONS = 8
TOLERANCE = 1e-9

COLORS = ("rgb", "gray")
PRETRAINING = ("pretrained", "random")
DIMENSIONALITY = ("full512", "pca6")
BASELINE = "rgb+pretrained+full512"


class ControlledAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            STAGE30 / "outer_fold_metrics.csv",
            STAGE30 / "outer_oof_predictions.csv",
            STAGE30 / "repeat_metrics.csv",
            STAGE30 / "condition_metrics.csv",
            STAGE30 / "paired_differences.csv",
            STAGE30 / "factor_effects.csv",
            STAGE30 / "methodology.json",
            STAGE30 / "anchor_check.json",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                f"Jalankan 30_controlled_ablation.py lebih dahulu ({missing[0].name} belum ada)"
            )
        cls.outer = pd.read_csv(STAGE30 / "outer_fold_metrics.csv")
        cls.predictions = pd.read_csv(STAGE30 / "outer_oof_predictions.csv")
        cls.repeats = pd.read_csv(STAGE30 / "repeat_metrics.csv")
        cls.summary = pd.read_csv(STAGE30 / "condition_metrics.csv")
        cls.paired = pd.read_csv(STAGE30 / "paired_differences.csv")
        cls.effects = pd.read_csv(STAGE30 / "factor_effects.csv")
        cls.methodology = json.loads((STAGE30 / "methodology.json").read_text())
        cls.anchor = json.loads((STAGE30 / "anchor_check.json").read_text())

    def test_all_eight_conditions_are_present(self):
        expected = {
            f"{color}+{pretraining}+{dimensionality}"
            for color in COLORS
            for pretraining in PRETRAINING
            for dimensionality in DIMENSIONALITY
        }
        self.assertEqual(len(expected), N_CONDITIONS)
        self.assertEqual(set(self.summary.condition), expected)
        self.assertEqual(set(self.outer.condition), expected)

    def test_every_condition_has_all_outer_folds(self):
        for condition, block in self.outer.groupby("condition"):
            with self.subTest(condition=condition):
                self.assertEqual(len(block), N_REPEATS * OUTER_SPLITS)

    def test_baseline_extraction_matches_stage11(self):
        """Pengaman utama: harness ini menghasilkan embedding tahap 11."""
        self.assertTrue(self.anchor["within_tolerance"], self.anchor)
        self.assertLessEqual(
            float(self.anchor["max_abs_deviation"]), float(self.anchor["tolerance"])
        )
        self.assertEqual(int(self.anchor["embedding_dim_stage11"]), 512)
        self.assertEqual(int(self.anchor["embedding_dim_here"]), 512)

    def test_all_conditions_share_identical_folds(self):
        """Kalau foldnya berbeda, seluruh perbandingan runtuh."""
        key = ["repeat", "source_id"]
        reference = (
            self.predictions[self.predictions.condition == BASELINE]
            .set_index(key).sort_index()
        )
        self.assertEqual(len(reference), N_REPEATS * N_ORIGINAL)
        for condition, block in self.predictions.groupby("condition"):
            with self.subTest(condition=condition):
                aligned = block.set_index(key).sort_index()
                self.assertTrue(aligned.index.equals(reference.index))
                self.assertTrue((aligned.outer_fold == reference.outer_fold).all())

    def test_each_original_predicted_once_per_repeat(self):
        for (condition, repeat), block in self.predictions.groupby(["condition", "repeat"]):
            with self.subTest(condition=condition, repeat=repeat):
                self.assertEqual(len(block), N_ORIGINAL)
                self.assertEqual(block.source_id.nunique(), N_ORIGINAL)

    def test_repeat_metrics_recomputed_from_predictions(self):
        indexed = self.repeats.set_index(["condition", "repeat"])
        for (condition, repeat), block in self.predictions.groupby(["condition", "repeat"]):
            y_true = block.label.to_numpy(int)
            y_pred = block.predicted_label.to_numpy(int)
            record = indexed.loc[(condition, repeat)]
            with self.subTest(condition=condition, repeat=repeat):
                self.assertAlmostEqual(
                    record.f1_macro,
                    f1_score(y_true, y_pred, average="macro", zero_division=0),
                    delta=TOLERANCE,
                )
                self.assertAlmostEqual(
                    record.balanced_accuracy,
                    balanced_accuracy_score(y_true, y_pred), delta=TOLERANCE,
                )
                self.assertAlmostEqual(
                    record.mcc, matthews_corrcoef(y_true, y_pred), delta=TOLERANCE
                )

    def test_condition_summary_matches_repeat_metrics(self):
        for record in self.summary.to_dict("records"):
            block = self.repeats[self.repeats.condition == record["condition"]]
            with self.subTest(condition=record["condition"]):
                self.assertEqual(int(record["n_repeats"]), len(block))
                self.assertAlmostEqual(
                    record["macro_f1_mean"], block.f1_macro.mean(), delta=TOLERANCE
                )
                self.assertAlmostEqual(
                    record["macro_f1_std"], block.f1_macro.std(ddof=1), delta=TOLERANCE
                )

    def test_paired_differences_recomputed_from_outer_folds(self):
        key = ["repeat", "outer_fold"]
        for record in self.paired.to_dict("records"):
            level_a, level_b = record["contrast"].split(" minus ")
            holding = dict(
                item.split("=") for item in record["holding"].split(", ")
            )
            mask = np.ones(len(self.outer), dtype=bool)
            for column, value in holding.items():
                mask &= self.outer[column].to_numpy() == value
            subset = self.outer[mask]
            side_a = subset[subset[record["factor"]] == level_a].set_index(key).sort_index()
            side_b = subset[subset[record["factor"]] == level_b].set_index(key).sort_index()
            delta = side_a.f1_macro - side_b.f1_macro
            with self.subTest(factor=record["factor"], holding=record["holding"]):
                self.assertEqual(int(record["n_paired_folds"]), len(delta))
                self.assertAlmostEqual(
                    record["mean_difference"], float(delta.mean()), delta=TOLERANCE
                )
                self.assertEqual(int(record["folds_better"]), int((delta > 0).sum()))
                self.assertEqual(int(record["folds_worse"]), int((delta < 0).sum()))

    def test_paired_contrasts_cover_every_factor(self):
        self.assertEqual(
            set(self.paired.factor), {"color", "pretraining", "dimensionality"}
        )
        for factor, block in self.paired.groupby("factor"):
            with self.subTest(factor=factor):
                # Dua faktor sisanya punya dua level masing-masing.
                self.assertEqual(len(block), 4)

    def test_selected_models_come_from_the_declared_space(self):
        allowed = {"Logistic Regression", "SVM (RBF)", "Random Forest"}
        self.assertTrue(set(self.outer.selected_model).issubset(allowed))

    def test_methodology_records_the_protocol_and_its_limit(self):
        self.assertEqual(self.methodology["backbone"], "ResNet18")
        self.assertEqual(self.methodology["n_conditions"], N_CONDITIONS)
        self.assertFalse(self.methodology["external_data_used"])
        self.assertTrue(self.methodology["pca_inside_pipeline"])
        self.assertIn("center cropping", self.methodology["not_varied"])
        self.assertEqual(
            self.methodology["split_grain"],
            "development source photo group_id (tahap 24)",
        )


if __name__ == "__main__":
    unittest.main()
