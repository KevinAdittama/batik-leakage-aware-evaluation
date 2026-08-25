"""Smoke test inferensi web terhadap artefak penelitian yang sebenarnya."""

import json
from pathlib import Path
import unittest

import numpy as np

from pipeline_config import MODEL_FEATURES, PROJECT_DIR
from web_app.inference import decode_uploaded_image, load_model_bundle, predict_image


class WebInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle()
        candidates = sorted((PROJECT_DIR / "uji_eksternal" / "batik").glob("*"))
        cls.sample = next(path for path in candidates if path.is_file())

    def test_selected_model_matches_cv_result(self):
        self.assertEqual(self.bundle.model_name, self.bundle.cv_result["model"])
        selected = json.loads(
            (PROJECT_DIR / "hasil_paper/08_uji_eksternal/selected_model_result.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(self.bundle.model_slug, selected["model_slug"])

    def test_real_image_prediction_is_complete(self):
        image = decode_uploaded_image(Path(self.sample).read_bytes())
        result = predict_image(image, self.bundle)
        self.assertIn(result.predicted_label, (0, 1))
        self.assertGreaterEqual(result.score_batik, 0.0)
        self.assertLessEqual(result.score_batik, 1.0)
        self.assertEqual(list(result.features), MODEL_FEATURES)
        self.assertTrue(np.isfinite(list(result.features.values())).all())
        self.assertEqual(
            set(result.visualizations), {"original", "motif_edge", "lbp", "fft"}
        )

    def test_invalid_upload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tidak dapat dikenali"):
            decode_uploaded_image(b"bukan citra")


if __name__ == "__main__":
    unittest.main()

