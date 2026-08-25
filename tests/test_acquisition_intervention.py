"""Unit tests for the R2.5 acquisition-intervention stage."""

import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_module():
    path = Path(__file__).resolve().parents[1] / "16_acquisition_intervention.py"
    spec = importlib.util.spec_from_file_location("acquisition_intervention", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AcquisitionInterventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_module()

    def test_intervention_encoding_is_deterministic_and_canonical(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(123, 217, 3), dtype=np.uint8)
        for variant in self.stage.VARIANTS:
            first_image, first_payload = self.stage.encode_intervention(image, variant)
            second_image, second_payload = self.stage.encode_intervention(image, variant)
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(first_image.shape, (512, 512, 3))
            self.assertTrue(np.array_equal(first_image, second_image))
            self.assertEqual(
                self.stage.sha256_bytes(first_payload),
                self.stage.sha256_bytes(second_payload),
            )

    def test_coarsened_external_match_is_balanced_within_strata(self):
        external = pd.read_csv(self.stage.EXTERNAL_MANIFEST_PATH)
        matched = self.stage.coarsened_match_manifest(external)
        selected = matched.loc[matched["selected_coarsened_match"]]
        self.assertEqual(selected.groupby("kelas").size().to_dict(), {"batik": 14, "non_batik": 14})
        table = selected.groupby(["stratum", "kelas"]).size().unstack(fill_value=0)
        self.assertTrue(table["batik"].eq(table["non_batik"]).all())
        self.assertTrue(selected["extension"].eq(".jpg").all())

    def test_stratified_bootstrap_preserves_class_counts(self):
        labels = np.array([0, 0, 0, 1, 1], dtype=int)
        indices = self.stage.stratified_bootstrap_indices(labels, 25, 42)
        self.assertEqual(indices.shape, (25, 5))
        for sample in indices:
            sampled = labels[sample]
            self.assertEqual(int(np.sum(sampled == 0)), 3)
            self.assertEqual(int(np.sum(sampled == 1)), 2)

    def test_vectorized_bootstrap_macro_f1_matches_sklearn(self):
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        predicted = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        indices = self.stage.stratified_bootstrap_indices(labels, 40, 123)
        actual = self.stage.bootstrap_macro_f1(labels, predicted, indices)
        expected = np.array(
            [
                f1_score(
                    labels[sample],
                    predicted[sample],
                    average="macro",
                    zero_division=0,
                )
                for sample in indices
            ]
        )
        self.assertTrue(np.allclose(actual, expected, rtol=0, atol=1e-15))


if __name__ == "__main__":
    unittest.main()
