"""Integration checks for saved R2.5 acquisition-intervention evidence."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef


def load_module():
    path = Path(__file__).resolve().parents[1] / "16_acquisition_intervention.py"
    spec = importlib.util.spec_from_file_location("acquisition_intervention_outputs", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AcquisitionInterventionOutputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_module()
        cls.predictions = pd.read_csv(cls.stage.OUT / "external_intervention_predictions.csv")
        cls.metrics = pd.read_csv(cls.stage.OUT / "external_intervention_metrics.csv")
        cls.differences = pd.read_csv(cls.stage.OUT / "paired_macro_f1_differences.csv")
        cls.transforms = pd.read_csv(cls.stage.OUT / "intervention_transform_manifest.csv")

    def test_expected_output_dimensions(self):
        self.assertEqual(len(self.predictions), 600)
        self.assertEqual(len(self.metrics), 30)
        self.assertEqual(len(self.differences), 24)
        self.assertEqual(len(self.transforms), 920)
        self.assertEqual(self.predictions["model"].nunique(), 2)
        self.assertEqual(self.predictions["condition"].nunique(), 5)

    def test_all_external_metrics_recompute_from_predictions(self):
        for _, saved in self.metrics.loc[self.metrics["subset"].eq("all_external")].iterrows():
            group = self.predictions.loc[
                self.predictions["model"].eq(saved["model"])
                & self.predictions["condition"].eq(saved["condition"])
            ]
            y_true = group["label"].to_numpy(int)
            y_pred = group["predicted_label"].to_numpy(int)
            self.assertEqual(len(group), 60)
            self.assertAlmostEqual(f1_score(y_true, y_pred, average="macro"), saved["f1_macro"])
            self.assertAlmostEqual(balanced_accuracy_score(y_true, y_pred), saved["balanced_accuracy"])
            self.assertAlmostEqual(matthews_corrcoef(y_true, y_pred), saved["mcc"])

    def test_transform_manifest_hashes_and_codec_fields(self):
        for source_path, records in self.transforms.groupby("path", sort=False):
            self.assertEqual(records["original_sha256"].nunique(), 1)
            self.assertEqual(
                self.stage.file_sha256(Path(self.stage.__file__).resolve().parent / source_path),
                records["original_sha256"].iloc[0],
            )
        self.assertTrue(self.transforms["transformed_width"].eq(512).all())
        self.assertTrue(self.transforms["transformed_height"].eq(512).all())
        codec_by_variant = self.transforms.groupby("variant")["codec"].unique().to_dict()
        self.assertEqual(codec_by_variant["canonical_png_512"].tolist(), ["png"])
        self.assertEqual(codec_by_variant["canonical_jpeg_q95_444_512"].tolist(), ["jpeg"])

    def test_paired_point_differences_and_bootstrap_shapes(self):
        self.assertTrue(
            np.allclose(
                self.differences["condition_macro_f1"] - self.differences["baseline_macro_f1"],
                self.differences["macro_f1_difference"],
                rtol=0,
                atol=1e-15,
            )
        )
        with np.load(self.stage.OUT / "paired_bootstrap_indices.npz") as arrays:
            self.assertEqual(arrays["all_external"].shape, (10000, 60))
            self.assertEqual(arrays["jpeg_original_format"].shape, (10000, 54))
            self.assertEqual(arrays["coarsened_acquisition_match"].shape, (10000, 28))


if __name__ == "__main__":
    unittest.main()
