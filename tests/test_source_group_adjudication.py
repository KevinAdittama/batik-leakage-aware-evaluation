"""Regresi untuk adjudikasi grup sumber (tahap 24).

Tes ini tidak mempercayai keluaran skrip begitu saja. Residual setiap pasangan
terkonfirmasi dihitung ulang dari berkas citra dengan implementasi terpisah,
pengelompokan dihitung ulang dari daftar pasangan, dan ambang yang tercatat di
methodology.json dibandingkan dengan yang benar-benar dipenuhi datanya.

Tanpa perhitungan ulang ini, tes hanya akan mengonfirmasi bahwa skrip menulis
apa yang baru saja dihitungnya sendiri.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STAGE24 = ROOT / "hasil_paper" / "24_source_groups"
AUDIT = ROOT / "hasil_paper" / "01_audit"

CANVAS = 256
N_DEVELOPMENT = 201
N_EXTERNAL = 60
RESIDUAL_TOLERANCE = 0.01


def load_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        with Image.open(path) as pil_image:
            image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.resize(image, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)


def residual_at(left: Path, right: Path, dx: int, dy: int) -> float:
    """Hitung residual wilayah tumpang tindih secara mandiri dari dx, dy."""
    a, b = load_color(left), load_color(right)
    ax, ay = max(0, -dx), max(0, -dy)
    bx, by = max(0, dx), max(0, dy)
    width, height = CANVAS - abs(dx), CANVAS - abs(dy)
    patch_a = a[ay:ay + height, ax:ax + width].astype(np.float32)
    patch_b = b[by:by + height, bx:bx + width].astype(np.float32)
    return float(np.abs(patch_a - patch_b).mean())


def components(paths: list[str], edges: list[tuple[str, str]]) -> dict[str, frozenset]:
    """Komponen terhubung, implementasi sederhana yang berdiri sendiri."""
    neighbours: dict[str, set] = {path: set() for path in paths}
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)

    seen: set[str] = set()
    membership: dict[str, frozenset] = {}
    for path in paths:
        if path in seen:
            continue
        stack, group = [path], set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(neighbours[node] - group)
        seen |= group
        frozen = frozenset(group)
        for member in group:
            membership[member] = frozen
    return membership


class SourceGroupAdjudicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            STAGE24 / "pair_screening.csv",
            STAGE24 / "pair_verification.csv",
            STAGE24 / "confirmed_pairs.csv",
            STAGE24 / "source_groups.csv",
            STAGE24 / "threshold_sensitivity.csv",
            STAGE24 / "methodology.json",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                f"Jalankan 24_source_group_adjudication.py lebih dahulu "
                f"({missing[0].name} belum ada)"
            )
        cls.screening = pd.read_csv(STAGE24 / "pair_screening.csv")
        cls.verification = pd.read_csv(STAGE24 / "pair_verification.csv")
        cls.confirmed = pd.read_csv(STAGE24 / "confirmed_pairs.csv")
        cls.groups = pd.read_csv(STAGE24 / "source_groups.csv")
        cls.sensitivity = pd.read_csv(STAGE24 / "threshold_sensitivity.csv")
        cls.methodology = json.loads((STAGE24 / "methodology.json").read_text())

    def test_population_matches_audit_manifests(self):
        development = pd.read_csv(AUDIT / "development_manifest.csv")
        external = pd.read_csv(AUDIT / "external_manifest.csv")
        self.assertEqual(len(development), N_DEVELOPMENT)
        self.assertEqual(len(external), N_EXTERNAL)
        self.assertEqual(len(self.groups), N_DEVELOPMENT + N_EXTERNAL)
        self.assertEqual(
            set(self.groups.path),
            set(development.path) | set(external.path),
        )

    def test_every_pair_was_screened(self):
        total = len(self.groups)
        self.assertEqual(len(self.screening), total * (total - 1) // 2)
        self.assertFalse(self.screening.duplicated(["left", "right"]).any())

    def test_verification_covers_exactly_the_screen_survivors(self):
        threshold = self.methodology["screen_min_patch_corr"]
        survivors = self.screening[self.screening.patch_corr >= threshold]
        self.assertEqual(len(self.verification), len(survivors))
        self.assertEqual(
            set(zip(self.verification.left, self.verification.right)),
            set(zip(survivors.left, survivors.right)),
        )

    def test_confirmed_pairs_satisfy_declared_thresholds(self):
        align_min = self.methodology["align_score_min"]
        residual_max = self.methodology["residual_max_rgb"]
        self.assertGreater(len(self.confirmed), 0)
        self.assertTrue((self.confirmed.align_score >= align_min).all())
        self.assertTrue((self.confirmed.residual_rgb < residual_max).all())

        rejected = self.verification[
            ~self.verification.set_index(["left", "right"]).index.isin(
                self.confirmed.set_index(["left", "right"]).index
            )
        ]
        violating = rejected[
            (rejected.align_score >= align_min) & (rejected.residual_rgb < residual_max)
        ]
        self.assertTrue(
            violating.empty,
            f"Ada pasangan yang memenuhi ambang tapi tidak dikonfirmasi:\n{violating}",
        )

    def test_residuals_recomputed_from_raw_images(self):
        for record in self.confirmed.to_dict("records"):
            with self.subTest(left=record["left"], right=record["right"]):
                recomputed = residual_at(
                    ROOT / record["left"], ROOT / record["right"],
                    int(record["dx"]), int(record["dy"]),
                )
                self.assertAlmostEqual(
                    recomputed, record["residual_rgb"], delta=RESIDUAL_TOLERANCE,
                    msg="Residual tersimpan tidak dapat dihitung ulang dari citra",
                )

    def test_groups_are_connected_components_of_confirmed_pairs(self):
        paths = self.groups.path.tolist()
        edges = list(zip(self.confirmed.left, self.confirmed.right))
        expected = components(paths, edges)
        recorded: dict[int, set] = {}
        for path, group_id in zip(self.groups.path, self.groups.group_id):
            recorded.setdefault(group_id, set()).add(path)
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(expected[path], frozenset(recorded[
                    self.groups.loc[self.groups.path == path, "group_id"].iloc[0]
                ]))

    def test_group_sizes_column_is_consistent(self):
        counts = self.groups.group_id.value_counts()
        for record in self.groups.to_dict("records"):
            self.assertEqual(record["group_size"], counts[record["group_id"]])

    def test_no_group_mixes_labels_or_sets(self):
        for group_id, block in self.groups.groupby("group_id"):
            with self.subTest(group_id=group_id):
                self.assertEqual(block.label.nunique(), 1)
                self.assertEqual(block["set"].nunique(), 1)

    def test_confirmed_pairs_are_within_one_group(self):
        group_of = dict(zip(self.groups.path, self.groups.group_id))
        for record in self.confirmed.to_dict("records"):
            with self.subTest(left=record["left"]):
                self.assertEqual(group_of[record["left"]], group_of[record["right"]])

    def test_threshold_plateau_covers_the_chosen_point(self):
        """Ambang yang dipilih harus berada di dataran datar, bukan di tebing."""
        chosen = self.sensitivity[
            (self.sensitivity.align_score_min == self.methodology["align_score_min"])
            & (self.sensitivity.residual_max == self.methodology["residual_max_rgb"])
        ]
        self.assertEqual(len(chosen), 1)
        value = int(chosen.images_in_multi_groups.iloc[0])
        neighbourhood = self.sensitivity[
            self.sensitivity.align_score_min.between(0.93, 0.95)
            & self.sensitivity.residual_max.between(15.0, 25.0)
        ]
        self.assertTrue(
            (neighbourhood.images_in_multi_groups == value).all(),
            f"Jawaban berubah di sekitar ambang terpilih:\n{neighbourhood}",
        )

    def test_fold_leakage_recorded_for_every_confirmed_pair(self):
        path = STAGE24 / "fold_leakage_check.csv"
        if not path.exists():
            self.skipTest("fold_assignments.csv tidak tersedia")
        leakage = pd.read_csv(path)
        self.assertEqual(len(leakage), len(self.confirmed))
        self.assertTrue(leakage.fold_left.notna().all())
        self.assertTrue(leakage.fold_right.notna().all())
        self.assertEqual(
            leakage.different_fold.tolist(),
            (leakage.fold_left != leakage.fold_right).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
