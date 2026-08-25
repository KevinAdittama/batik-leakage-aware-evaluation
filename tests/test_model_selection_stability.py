"""Regresi untuk diagnostik stabilitas seleksi model (tahap 25).

Tahap 25 tidak melatih apa pun; ia membaca keluaran tahap lain. Karena itu tes
ini menghitung ulang setiap angkanya dari sumber aslinya. Kalau tahap 25 hanya
menyalin, tes ini akan menangkapnya; kalau ia salah menyalin, tes ini gagal.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE25 = ROOT / "hasil_paper" / "25_model_selection_stability"
STAGE07 = ROOT / "hasil_paper" / "07_cross_validation"
STAGE08 = ROOT / "hasil_paper" / "08_uji_eksternal"
STAGE14 = ROOT / "hasil_paper" / "14_repeated_nested_augmentation"

PRIMARY_FEATURE_SET = "Gabungan 6 Fitur"
TOLERANCE = 1e-9


class ModelSelectionStabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            STAGE25 / "single_loop_ranking.csv",
            STAGE25 / "nested_selection_frequency.csv",
            STAGE25 / "external_by_model.csv",
            STAGE25 / "selection_margin.csv",
            STAGE25 / "methodology.json",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                f"Jalankan 25_model_selection_stability.py lebih dahulu "
                f"({missing[0].name} belum ada)"
            )
        cls.ranking = pd.read_csv(STAGE25 / "single_loop_ranking.csv")
        cls.selection = pd.read_csv(STAGE25 / "nested_selection_frequency.csv")
        cls.external = pd.read_csv(STAGE25 / "external_by_model.csv")
        cls.margin = pd.read_csv(STAGE25 / "selection_margin.csv")
        cls.methodology = json.loads((STAGE25 / "methodology.json").read_text())

    def current_margin(self):
        row = self.margin[self.margin.protocol == "source-group-aware"]
        self.assertEqual(len(row), 1)
        return row.iloc[0]

    def test_ranking_matches_stage07(self):
        source = pd.read_csv(STAGE07 / "cv_summary_primary.csv")
        self.assertEqual(len(self.ranking), len(source))
        merged = self.ranking.merge(source, on="model", suffixes=("_25", "_07"))
        self.assertEqual(len(merged), len(source))
        for record in merged.to_dict("records"):
            self.assertAlmostEqual(
                record["f1_macro_mean_25"], record["f1_macro_mean_07"], delta=TOLERANCE
            )

    def test_ranking_is_sorted_descending(self):
        values = self.ranking.f1_macro_mean.tolist()
        self.assertEqual(values, sorted(values, reverse=True))

    def test_margin_recomputed_from_ranking(self):
        margin = self.current_margin()
        top, runner_up = self.ranking.iloc[0], self.ranking.iloc[1]
        self.assertEqual(margin.winner, top.model)
        self.assertEqual(margin.runner_up, runner_up.model)
        self.assertAlmostEqual(
            float(margin.margin),
            float(top.f1_macro_mean - runner_up.f1_macro_mean),
            delta=TOLERANCE,
        )

    def test_standard_error_recomputed_from_fold_metrics(self):
        margin = self.current_margin()
        folds = pd.read_csv(STAGE07 / "cv_fold_metrics.csv")
        folds = folds[folds.feature_set == PRIMARY_FEATURE_SET]
        n_folds = int(folds.groupby("model").size().max())
        self.assertEqual(int(margin.n_folds), n_folds)

        pooled = float(self.ranking.f1_macro_std.mean())
        self.assertAlmostEqual(float(margin.pooled_fold_std), pooled, delta=TOLERANCE)
        self.assertAlmostEqual(
            float(margin.standard_error_of_mean),
            pooled / np.sqrt(n_folds),
            delta=TOLERANCE,
        )

    def test_winner_matches_stored_selection(self):
        stored = json.loads(
            (STAGE08 / "selected_model_result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.current_margin().winner, stored["model"])

    def test_nested_selection_counts_match_source(self):
        source = pd.read_csv(STAGE14 / "outer_fold_metrics.csv")
        expected = source.selected_model.value_counts().to_dict()
        block = self.selection[
            self.selection.protocol == "nested 5 repeat (tahap 14)"
        ]
        recorded = dict(zip(block.model, block.selections))
        self.assertEqual(recorded, {k: int(v) for k, v in expected.items()})
        self.assertTrue((block["of"] == len(source)).all())

    def test_selection_counts_sum_to_all_outer_folds(self):
        for (protocol, arm), block in self.selection.groupby(["protocol", "arm"]):
            with self.subTest(protocol=protocol, arm=arm):
                self.assertEqual(block["of"].nunique(), 1)
                self.assertEqual(int(block.selections.sum()), int(block["of"].iloc[0]))

    def test_external_table_reports_every_model(self):
        classical = pd.read_csv(STAGE08 / "external_model_summary.csv")
        deep = pd.read_csv(
            ROOT / "hasil_paper/11_deep_learning_baseline/dl_external_summary.csv"
        )
        self.assertEqual(
            set(self.external.model), set(classical.model) | set(deep.model)
        )
        merged = self.external.merge(classical, on="model", suffixes=("_25", "_08"))
        for record in merged.to_dict("records"):
            self.assertAlmostEqual(
                record["f1_macro_25"], record["f1_macro_08"], delta=TOLERANCE
            )

    def test_margin_is_smaller_than_fold_noise(self):
        """Inti temuannya: margin puncak tenggelam dalam derau antar-fold."""
        margin = self.current_margin()
        self.assertLess(float(margin.margin), float(margin.pooled_fold_std))
        self.assertLess(abs(float(margin.margin_in_standard_errors)), 2.0)

    def test_methodology_records_the_caveat(self):
        self.assertIn("caveat", self.methodology)
        self.assertEqual(
            self.methodology["selection_rule"],
            "highest mean CV macro-F1 on the combined six features",
        )
        self.assertTrue(self.methodology["assertions_passed"])


if __name__ == "__main__":
    unittest.main()
