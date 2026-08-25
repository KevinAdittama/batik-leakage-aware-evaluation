"""Regression tests for the IJIES critical numerical audit."""

import importlib.util
from pathlib import Path
import unittest

import pandas as pd
from sklearn.metrics import f1_score


def load_audit_module():
    path = Path(__file__).resolve().parents[1] / "13_numerical_audit.py"
    spec = importlib.util.spec_from_file_location("numerical_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumericalAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_audit_module()
        cls.manifest, cls.predictions = cls.audit.load_external_predictions()

    def matrix_for(self, matrices, model):
        rows = matrices.loc[matrices["model"].eq(model)]
        return (
            rows.pivot(index="true_label", columns="predicted_label", values="n")
            .sort_index()
            .sort_index(axis=1)
            .to_numpy()
            .tolist()
        )

    def test_formal_model_subtypes_reproduce_confusion_matrix(self):
        """Analisis subjenis harus mereproduksi confusion matrix model formal.

        Angkanya tidak dipatok harfiah lagi. Model formal dapat berpindah karena
        margin seleksi lebih kecil daripada derau antar-fold (tahap 25), jadi
        yang diuji adalah konsistensi internalnya, bukan identitas pemenangnya.
        """
        _, matrices, authoritative, subtype = self.audit.external_audit(
            self.manifest, self.predictions
        )
        formal_name, _ = self.audit.formal_model()
        matrix = self.matrix_for(matrices, formal_name)
        (tn, fp), (fn, tp) = matrix

        self.assertEqual(len(authoritative), 60)
        self.assertEqual(int(subtype["n"].sum()), 60)
        self.assertEqual(int(subtype["errors"].sum()), fp + fn)
        self.assertEqual(int(subtype["false_negative"].sum()), fn)
        self.assertEqual(int(subtype["false_positive"].sum()), fp)

    def test_random_forest_external_matrix_is_unchanged(self):
        """Prediksi eksternal per model tidak bergantung pada struktur fold.

        Random Forest tidak lagi menjadi model formal, tetapi matriks
        eksternalnya harus tetap sama persis seperti sebelum fold menjadi
        group-aware. Kalau ini bergeser, berarti ada yang bocor dari sisi
        development ke evaluasi eksternal.
        """
        _, matrices, _, _ = self.audit.external_audit(self.manifest, self.predictions)
        self.assertEqual(self.matrix_for(matrices, "Random Forest"), [[21, 9], [14, 16]])

    def test_full_precision_paired_macro_f1_difference(self):
        rf = self.predictions["Random Forest"]
        resnet = self.predictions["ResNet18"]
        rf_f1 = f1_score(rf["label"], rf["predicted_label"], average="macro")
        resnet_f1 = f1_score(
            resnet["label"], resnet["predicted_label"], average="macro"
        )
        self.assertAlmostEqual(rf_f1, 0.613986013986014, places=15)
        self.assertAlmostEqual(resnet_f1, 0.8331479421579533, places=15)
        self.assertAlmostEqual(resnet_f1 - rf_f1, 0.21916192817193925, places=15)

    def test_fold_composition_is_persistable_and_leakage_safe(self):
        summary, sources, assignments = self.audit.fold_audit()
        validation_sizes = (
            summary.groupby("fold")["validation_original"].sum().tolist()
        )
        self.assertEqual(validation_sizes, [41, 40, 40, 40, 40])
        self.assertTrue(summary["train_total"].eq(200).all())
        self.assertEqual(assignments["source_id"].nunique(), len(assignments))
        self.assertLessEqual(int(sources["selected_derivatives"].max()), 4)
        self.assertTrue(
            sources["selected_derivatives"].le(
                sources["available_derivatives"]
            ).all()
        )

    def test_protocol_specific_model_selection_terms(self):
        """Istilah seleksi harus konsisten dengan artefak, bukan dengan ingatan.

        Sebelumnya tes ini memaku Random Forest sebagai pemenang single-loop.
        Pematokan itu berubah menjadi asumsi yang salah begitu fold dipisahkan
        pada grain foto sumber, jadi yang diuji sekarang adalah konsistensi
        antara tabel CV, berkas seleksi tersimpan, dan hitungan nested.
        """
        import json

        selection = self.audit.model_selection_audit().set_index("protocol")
        root = Path(__file__).resolve().parents[1]
        stored = json.loads(
            (root / "hasil_paper/08_uji_eksternal/selected_model_result.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            selection.loc["single-loop augmented handcrafted analysis", "model"],
            stored["model"],
        )

        nested = pd.read_csv(
            root / "hasil_paper/12_submission_robustness/nested_cv_outer_fold_metrics.csv"
        )
        self.assertEqual(
            selection.loc["originals-only nested cross-validation", "model"],
            nested["selected_model"].value_counts().index[0],
        )


if __name__ == "__main__":
    unittest.main()
