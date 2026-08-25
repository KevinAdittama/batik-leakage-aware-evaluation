"""R2.7 - Regresi untuk perbandingan strategi penyeimbangan kelas (tahap 19).

Tes ini menghitung ulang metrik dari prediksi mentah, memeriksa bahwa ketiga
lengan benar-benar berbagi pembagian fold yang sama, dan memastikan lengan
augmentasi mereproduksi tahap 14. Tanpa pemeriksaan terakhir itu, selisih
antarlengan bisa saja berasal dari perbedaan harness, bukan dari strategi
penyeimbangan.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
STAGE19 = ROOT / "hasil_paper" / "19_balancing_comparison"
STAGE14 = ROOT / "hasil_paper" / "14_repeated_nested_augmentation"

ARMS = (
    "augmented_balanced",
    "class_weighted_originals",
    "balanced_original_sampling",
)
BASELINE = "augmented_balanced"
N_REPEATS = 5
OUTER_SPLITS = 5
N_ORIGINAL = 201
TOLERANCE = 1e-9


class BalancingComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            STAGE19 / "outer_fold_metrics.csv",
            STAGE19 / "outer_oof_predictions.csv",
            STAGE19 / "repeat_metrics.csv",
            STAGE19 / "arm_summary.csv",
            STAGE19 / "paired_differences.csv",
            STAGE19 / "methodology.json",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise unittest.SkipTest(
                f"Jalankan 19_augmentation_balancing_comparison.py lebih dahulu "
                f"({missing[0].name} belum ada)"
            )
        cls.outer = pd.read_csv(STAGE19 / "outer_fold_metrics.csv")
        cls.predictions = pd.read_csv(STAGE19 / "outer_oof_predictions.csv")
        cls.repeats = pd.read_csv(STAGE19 / "repeat_metrics.csv")
        cls.summary = pd.read_csv(STAGE19 / "arm_summary.csv")
        cls.paired = pd.read_csv(STAGE19 / "paired_differences.csv")
        cls.methodology = json.loads((STAGE19 / "methodology.json").read_text())

    def test_all_arms_present_and_complete(self):
        self.assertEqual(sorted(self.outer.arm.unique()), sorted(ARMS))
        for arm in ARMS:
            with self.subTest(arm=arm):
                subset = self.outer[self.outer.arm == arm]
                self.assertEqual(len(subset), N_REPEATS * OUTER_SPLITS)
                predictions = self.predictions[self.predictions.arm == arm]
                self.assertEqual(len(predictions), N_REPEATS * N_ORIGINAL)

    def test_fold_assignment_identical_across_arms(self):
        """Perbandingan hanya sah bila setiap citra jatuh di fold yang sama."""
        base = self.predictions[self.predictions.arm == BASELINE]
        reference = base.set_index(["repeat", "source_id"]).outer_fold.sort_index()
        for arm in ARMS:
            with self.subTest(arm=arm):
                current = (
                    self.predictions[self.predictions.arm == arm]
                    .set_index(["repeat", "source_id"])
                    .outer_fold.sort_index()
                )
                pd.testing.assert_series_equal(current, reference, check_names=False)

    def test_repeat_metrics_recomputed_from_predictions(self):
        for _, row in self.repeats.iterrows():
            group = self.predictions[
                (self.predictions.arm == row["arm"])
                & (self.predictions.repeat == row["repeat"])
            ]
            y_true = group.label.to_numpy(int)
            y_pred = group.predicted_label.to_numpy(int)
            with self.subTest(arm=row["arm"], repeat=int(row["repeat"])):
                self.assertAlmostEqual(
                    row["f1_macro"], f1_score(y_true, y_pred, average="macro"), delta=TOLERANCE
                )
                self.assertAlmostEqual(
                    row["balanced_accuracy"],
                    balanced_accuracy_score(y_true, y_pred),
                    delta=TOLERANCE,
                )
                self.assertAlmostEqual(
                    row["mcc"], matthews_corrcoef(y_true, y_pred), delta=TOLERANCE
                )

    def test_augmented_arm_reproduces_stage14(self):
        reference_path = STAGE14 / "repeat_metrics.csv"
        if not reference_path.exists():
            self.skipTest("Hasil tahap 14 tidak tersedia")
        stage14 = pd.read_csv(reference_path).set_index("repeat").sort_index()
        mine = (
            self.repeats[self.repeats.arm == BASELINE]
            .set_index("repeat")
            .sort_index()
        )
        for column in ("f1_macro", "balanced_accuracy", "mcc"):
            with self.subTest(metrik=column):
                deviation = float(np.abs(mine[column] - stage14[column]).max())
                self.assertLess(
                    deviation,
                    TOLERANCE,
                    f"Lengan augmentasi menyimpang dari tahap 14 pada {column}",
                )

    def test_paired_differences_recomputed(self):
        key = ["repeat", "outer_fold"]
        base = self.outer[self.outer.arm == BASELINE].set_index(key).sort_index()
        for _, row in self.paired.iterrows():
            arm = row["comparison"].split(" minus ")[0]
            metric = row["metric"]
            other = self.outer[self.outer.arm == arm].set_index(key).sort_index()
            diff = other[metric] - base[metric]
            with self.subTest(comparison=row["comparison"], metric=metric):
                self.assertEqual(int(row["n_paired_folds"]), len(diff))
                self.assertAlmostEqual(row["mean_difference"], diff.mean(), delta=TOLERANCE)
                self.assertEqual(int(row["folds_better"]), int((diff > 0).sum()))
                self.assertEqual(int(row["folds_worse"]), int((diff < 0).sum()))
                self.assertEqual(int(row["folds_tied"]), int((diff == 0).sum()))
                self.assertEqual(
                    int(row["folds_better"]) + int(row["folds_worse"]) + int(row["folds_tied"]),
                    len(diff),
                )

    def test_training_composition_matches_arm_definition(self):
        augmented = self.outer[self.outer.arm == BASELINE]
        self.assertTrue((augmented.train_batik == 200).all())
        self.assertTrue((augmented.train_non_batik == 200).all())
        self.assertTrue((augmented.train_derivatives > 0).all())

        weighted = self.outer[self.outer.arm == "class_weighted_originals"]
        self.assertTrue((weighted.train_derivatives == 0).all())
        self.assertTrue((weighted.train_batik > weighted.train_non_batik).all())

        balanced = self.outer[self.outer.arm == "balanced_original_sampling"]
        self.assertTrue((balanced.train_derivatives == 0).all())
        self.assertTrue((balanced.train_batik == balanced.train_non_batik).all())

    def test_no_external_data_used(self):
        self.assertFalse(self.methodology["external_data_used"])
        self.assertEqual(self.methodology["baseline_arm"], BASELINE)
        self.assertEqual(sorted(self.methodology["arms"]), sorted(ARMS))

    def test_summary_matches_repeat_level(self):
        for _, row in self.summary.iterrows():
            group = self.repeats[self.repeats.arm == row["arm"]]
            with self.subTest(arm=row["arm"]):
                self.assertEqual(int(row["n_repeats"]), len(group))
                self.assertAlmostEqual(
                    row["macro_f1_mean"], group.f1_macro.mean(), delta=TOLERANCE
                )
                self.assertAlmostEqual(
                    row["macro_f1_std"], group.f1_macro.std(ddof=1), delta=TOLERANCE
                )


if __name__ == "__main__":
    unittest.main()
