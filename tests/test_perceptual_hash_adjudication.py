from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "IJIES_REVISI_FINAL"
    / "04_Tabel_Manifest_dan_Hasil"
    / "Audit_Numerik_dan_Eksperimen"
    / "outputs"
    / "perceptual_hash_adjudication"
)


class PerceptualHashAdjudicationTest(unittest.TestCase):
    def test_cross_candidate_decisions_are_complete(self):
        frame = pd.read_csv(OUT / "near_duplicate_candidates_cross_adjudication.csv")
        counts = frame["preliminary_visual_status"].value_counts().to_dict()
        self.assertEqual(len(frame), 6)
        self.assertIsNone(counts.get("confirmed_near_duplicate"))
        self.assertEqual(counts.get("not_duplicate_hash_collision"), 6)
        self.assertFalse((frame["preliminary_visual_status"] == "pending_visual_review").any())
        self.assertTrue((frame["human_adjudication_status"] == "pending_named_human_review").all())

    def test_approved_exclusions_are_applied_and_no_new_exclusion_is_proposed(self):
        proposed = pd.read_csv(OUT / "proposed_development_exclusions.csv")
        applied = pd.read_csv(OUT / "applied_development_exclusions.csv")
        self.assertTrue(proposed.empty)
        self.assertEqual(
            set(applied["source_id"]), {"b742ba0efad4", "693f3e1e007d"}
        )
        self.assertEqual(set(applied["status"]), {"approved_exclude"})
        self.assertTrue(applied["matched_manifest"].astype(bool).all())

    def test_threshold_eight_reproduces_stage12_candidates(self):
        frame = pd.read_csv(OUT / "cross_threshold_sensitivity.csv")
        row = frame.loc[frame["hamming_threshold"] == 8].iloc[0]
        self.assertEqual(int(row["phash_candidates"]), 0)
        self.assertEqual(int(row["dhash_candidates"]), 6)
        self.assertEqual(int(row["or_candidates"]), 6)
        self.assertEqual(int(row["and_candidates"]), 0)

    def test_output_manifest_hashes_match(self):
        with (OUT / "output_manifest_sha256.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 6)
        for row in rows:
            path = OUT / row["file"]
            self.assertTrue(path.is_file())
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            self.assertEqual(digest, row["sha256"])


if __name__ == "__main__":
    unittest.main()
