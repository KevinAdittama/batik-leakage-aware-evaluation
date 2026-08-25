"""Uji alur antarmuka Streamlit tanpa membuka browser eksternal."""

import json
from pathlib import Path
import unittest

from pipeline_config import PROJECT_DIR

from streamlit.testing.v1 import AppTest


class StreamlitAppTest(unittest.TestCase):
    def test_upload_renders_complete_result(self):
        sample = next((Path("uji_eksternal") / "batik").glob("*.jpg"))
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.file_uploader), 1)

        app.file_uploader[0].upload(
            sample.name, sample.read_bytes(), "image/jpeg"
        ).run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual([tab.label for tab in app.tabs], [
            "Proses citra", "Enam fitur", "Tentang model"
        ])
        self.assertEqual(len(app.get("imgs")), 4)
        self.assertEqual(len(app.dataframe), 1)
        metric_values = {metric.label: metric.value for metric in app.metric}
        selected = json.loads(
            (PROJECT_DIR / "hasil_paper/08_uji_eksternal/selected_model_result.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(metric_values["Model"], selected["model"])
        cv = json.loads(
            (PROJECT_DIR / "hasil_paper/07_cross_validation/best_cv_model.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            metric_values["CV macro-F1"], f"{cv['cv_f1_macro_mean']:.3f}"
        )


if __name__ == "__main__":
    unittest.main()
