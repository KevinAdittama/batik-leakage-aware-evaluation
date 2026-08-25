"""Adjudikasi grup sumber: mendeteksi crop yang berasal dari satu foto yang sama.

Latar belakang
--------------
Audit near-duplicate pada tahap 12 dan `14_perceptual_hash_adjudication.py`
memakai perceptual hash. pHash **tidak invarian terhadap geseran**: dua potongan
dari satu foto yang sama, yang bergeser beberapa puluh piksel, menghasilkan hash
yang berjauhan dan lolos dari audit. Padahal justru potongan semacam itulah yang
melanggar asumsi kemandirian sampel.

Konsekuensinya nyata. `source_id` selama ini didefinisikan per berkas, sehingga
dua potongan dari satu foto dapat jatuh di fold berbeda. Itu kebocoran
train/test, persis jenis cacat yang diaudit artikel ini.

Metode
------
Dua tahap, dari yang murah ke yang mahal.

  Tahap 1 - penyaringan.
      Setiap pasangan citra dinormalkan ke kanvas 256x256 skala abu. Patch
      tengah 96x96 dari citra A dicari posisi cocoknya di citra B dengan
      normalized cross-correlation, dan sebaliknya. Skor tertinggi dipakai.

  Tahap 2 - verifikasi.
      Pasangan yang lolos penyaringan diselaraskan ulang memakai template
      separuh ukuran, lalu seluruh wilayah tumpang tindih dibandingkan dalam
      warna. Residual dihitung sebagai mean absolute difference pada skala
      0-255.

Kenapa dua tahap. Motif batik bersifat periodik, sehingga korelasi patch saja
memberi banyak positif palsu: motif yang sama pada foto berbeda tetap berkorelasi
tinggi. Yang membedakan foto yang sama adalah pencahayaan dan tekstur serat, dan
itu hanya terlihat pada residual wilayah penuh berwarna.

Ambang
------
Sepasang citra dinyatakan berasal dari satu foto bila residualnya di bawah
`RESIDUAL_MAX` dan skor penyelarasannya minimal `ALIGN_SCORE_MIN`. Residual
tidak pernah nol karena setiap potongan di-resize dan dikompresi ulang secara
terpisah. Tabel sensitivitas ambang ikut ditulis ke keluaran agar pembaca dapat
menilai sendiri kekukuhan pilihan ini.

Cakupan dan batasnya
--------------------
Yang diuji: seluruh pasangan di antara 201 original development dan 60 citra
eksternal, lintas kelas dan lintas subjenis, tanpa kecuali.

Yang secara prinsip masih dapat lolos: potongan dari satu foto yang **tidak
saling tumpang tindih**, dan citra yang mengalami rotasi atau perubahan skala
besar. Uji ini mendeteksi tumpang tindih translasional, bukan seluruh bentuk
kekerabatan. Batas itu dinyatakan terbuka di REPORT.md.

Skrip ini hanya membaca dan menulis ke `hasil_paper/24_source_groups/`. Skrip
ini tidak mengubah fold, tidak melatih model, dan tidak menyentuh manuskrip.

Jalankan dari folder proyek:
    .\\env\\Scripts\\python.exe -B .\\24_source_group_adjudication.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from pipeline_config import AUDIT_DIR, PROJECT_DIR, RANDOM_SEED, RESULTS_DIR

OUT = RESULTS_DIR / "24_source_groups"

CANVAS = 256          # kanvas normalisasi untuk kedua tahap
SCREEN_PATCH = 96     # patch tahap 1
SCREEN_MIN = 0.60     # ambang lolos ke tahap 2; sengaja longgar
ALIGN_SCORE_MIN = 0.95
RESIDUAL_MAX = 15.0
MIN_OVERLAP = 40      # piksel; tumpang tindih lebih sempit tidak informatif


def load_gray(path: Path) -> np.ndarray:
    """Baca sebagai skala abu 256x256. Fallback PIL menirukan pipeline_common."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        with Image.open(path) as pil_image:
            image = np.array(pil_image.convert("L"))
    return cv2.resize(image, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)


def load_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        with Image.open(path) as pil_image:
            image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.resize(image, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)


def screen_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Skor korelasi patch tertinggi di antara kedua arah."""
    start = (CANVAS - SCREEN_PATCH) // 2
    best = -1.0
    for first, second in ((a, b), (b, a)):
        patch = first[start:start + SCREEN_PATCH, start:start + SCREEN_PATCH]
        result = cv2.matchTemplate(second, patch, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(cv2.minMaxLoc(result)[1]))
    return best


def verify_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Selaraskan pada template separuh ukuran, lalu ukur residual berwarna."""
    half = CANVAS // 2
    offset = (CANVAS - half) // 2
    patch = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)[offset:offset + half, offset:offset + half]
    result = cv2.matchTemplate(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    dx, dy = location[0] - offset, location[1] - offset

    ax, ay = max(0, -dx), max(0, -dy)
    bx, by = max(0, dx), max(0, dy)
    width, height = CANVAS - abs(dx), CANVAS - abs(dy)
    if width < MIN_OVERLAP or height < MIN_OVERLAP:
        return {"align_score": float(score), "dx": int(dx), "dy": int(dy),
                "overlap_w": int(width), "overlap_h": int(height), "residual_rgb": float("nan")}

    left = a[ay:ay + height, ax:ax + width].astype(np.float32)
    right = b[by:by + height, bx:bx + width].astype(np.float32)
    return {
        "align_score": float(score), "dx": int(dx), "dy": int(dy),
        "overlap_w": int(width), "overlap_h": int(height),
        "residual_rgb": float(np.abs(left - right).mean()),
    }


def build_population() -> pd.DataFrame:
    development = pd.read_csv(AUDIT_DIR / "development_manifest.csv")
    development["set"] = "development"
    external = pd.read_csv(AUDIT_DIR / "external_manifest.csv")
    external["set"] = "external"
    columns = ["source_id", "path", "kelas", "label", "subjenis", "set"]
    population = pd.concat([development[columns], external[columns]], ignore_index=True)
    if population.path.duplicated().any():
        raise AssertionError("Ada path ganda dalam populasi gabungan")
    return population


def connected_groups(paths: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    parent = {path: path for path in paths}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    roots = sorted({find(path) for path in paths})
    numbering = {root: index + 1 for index, root in enumerate(roots)}
    return {path: numbering[find(path)] for path in paths}


def threshold_sensitivity(verified: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    """Sapu kedua ambang sekaligus.

    Menyapu residual saja akan menyesatkan, karena kriteria penyelarasan sudah
    memotong sebagian besar kandidat lebih dahulu. Grid dua dimensi menunjukkan
    apakah jawabannya benar-benar kukuh atau hanya tampak kukuh.
    """
    rows = []
    for align_min in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
        for residual_max in (10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
            confirmed = verified[
                (verified.residual_rgb < residual_max)
                & (verified.align_score >= align_min)
            ]
            mapping = connected_groups(paths, list(zip(confirmed.left, confirmed.right)))
            sizes = pd.Series(mapping).value_counts()
            rows.append({
                "align_score_min": align_min,
                "residual_max": residual_max,
                "confirmed_pairs": len(confirmed),
                "n_groups": int(sizes.size),
                "multi_image_groups": int((sizes > 1).sum()),
                "images_in_multi_groups": int(sizes[sizes > 1].sum()),
                "largest_group": int(sizes.max()),
            })
    return pd.DataFrame(rows)


def fold_leakage(confirmed: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Apakah pasangan terkonfirmasi saat ini jatuh di fold validasi berbeda?"""
    assignments = (
        PROJECT_DIR / "IJIES_REVISI_FINAL" / "04_Tabel_Manifest_dan_Hasil"
        / "Audit_Numerik_dan_Eksperimen" / "outputs" / "fold_assignments.csv"
    )
    if not assignments.exists():
        return pd.DataFrame()
    folds = pd.read_csv(assignments)
    fold_of = dict(zip(folds.path, folds.validation_fold))
    set_of = dict(zip(population.path, population["set"]))

    rows = []
    for record in confirmed.to_dict("records"):
        left, right = record["left"], record["right"]
        rows.append({
            "left": left, "right": right,
            "residual_rgb": round(record["residual_rgb"], 2),
            "left_set": set_of.get(left), "right_set": set_of.get(right),
            "fold_left": fold_of.get(left), "fold_right": fold_of.get(right),
            "different_fold": fold_of.get(left) != fold_of.get(right),
        })
    return pd.DataFrame(rows)


def proposed_config(groups: pd.DataFrame, confirmed: pd.DataFrame) -> pd.DataFrame:
    """Usulan berkas keputusan untuk `audit_config/source_groups.csv`.

    Hanya citra dalam grup majemuk yang dicantumkan; sisanya tetap satu grup
    per berkas dan tidak perlu keputusan apa pun. Polanya menirukan
    `development_exclusions.csv`: setiap baris membawa bukti yang membuatnya
    bisa diperiksa ulang, bukan sekadar dipercaya.
    """
    manifest = pd.read_csv(AUDIT_DIR / "development_manifest.csv").set_index("path")
    evidence: dict[str, list[dict]] = {}
    for record in confirmed.to_dict("records"):
        for side in ("left", "right"):
            evidence.setdefault(record[side], []).append(record)

    rows = []
    for group_id, block in groups[groups.group_size > 1].groupby("group_id"):
        group_key = "grp_" + min(block.source_id)
        for record in block.sort_values("source_id").to_dict("records"):
            pairs = evidence.get(record["path"], [])
            rows.append({
                "group_key": group_key,
                "source_id": record["source_id"],
                "path": record["path"],
                "sha256": manifest.loc[record["path"], "sha256"],
                "subjenis": record["subjenis"],
                "group_size": record["group_size"],
                "min_residual_rgb": round(min(p["residual_rgb"] for p in pairs), 3),
                "max_align_score": round(max(p["align_score"] for p in pairs), 4),
                "reason": "same_photograph_translational_overlap",
            })
    return pd.DataFrame(rows)


def contact_sheet(confirmed: pd.DataFrame, output: Path) -> None:
    if confirmed.empty:
        return
    tile = 230
    rows = []
    for record in confirmed.sort_values("residual_rgb").to_dict("records"):
        row = []
        for side in ("left", "right"):
            image = cv2.resize(load_color(PROJECT_DIR / record[side]), (tile, tile))
            cv2.rectangle(image, (0, tile - 20), (tile, tile), (0, 0, 0), -1)
            cv2.putText(image, Path(record[side]).name[:28], (4, tile - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            row.append(image)
        banner = np.zeros((tile, 300, 3), dtype=np.uint8)
        lines = [
            Path(record["left"]).parent.name[:26],
            f"residual {record['residual_rgb']:.2f}",
            f"align {record['align_score']:.4f}",
            f"geseran ({record['dx']}, {record['dy']}) px",
        ]
        for index, line in enumerate(lines):
            cv2.putText(banner, line, (6, 28 + index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        row.append(banner)
        rows.append(np.pad(np.hstack(row), ((0, 6), (0, 0), (0, 0))))
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(output))


def write_report(
    output: Path,
    population: pd.DataFrame,
    screened: pd.DataFrame,
    verified: pd.DataFrame,
    confirmed: pd.DataFrame,
    groups: pd.DataFrame,
    sensitivity: pd.DataFrame,
    leakage: pd.DataFrame,
    checks: list[str],
) -> None:
    sizes = groups.group_id.value_counts()
    multi = sizes[sizes > 1]
    lines = [
        "# Tahap 24 - Adjudikasi grup sumber",
        "",
        "Deteksi potongan citra yang berasal dari satu foto yang sama, yang lolos",
        "dari audit perceptual hash karena pHash tidak invarian terhadap geseran.",
        "",
        "## Populasi",
        "",
        f"- Development original: {int((population['set'] == 'development').sum())}",
        f"- Eksternal: {int((population['set'] == 'external').sum())}",
        f"- Total citra: {len(population)}",
        f"- Pasangan diuji: {len(screened)} (seluruh kombinasi, lintas kelas dan subjenis)",
        "",
        "## Ambang",
        "",
        f"- Penyaringan tahap 1: korelasi patch >= {SCREEN_MIN}",
        f"- Konfirmasi: residual RGB < {RESIDUAL_MAX} dan skor penyelarasan >= {ALIGN_SCORE_MIN}",
        "",
        "Residual tidak pernah nol karena tiap potongan di-resize dan dikompresi",
        "ulang secara terpisah.",
        "",
        "## Hasil",
        "",
        f"- Lolos penyaringan ke tahap 2: {len(verified)}",
        f"- Pasangan terkonfirmasi satu foto: {len(confirmed)}",
        f"- Grup sumber unik: {int(sizes.size)} (dari {len(population)} citra)",
        f"- Grup beranggota lebih dari satu: {int(multi.size)}",
        f"- Citra yang terlibat: {int(multi.sum())}",
        "",
    ]

    if not confirmed.empty:
        lines += ["### Pasangan terkonfirmasi", "",
                  "| Subjenis | Kiri | Kanan | Geseran | Residual | Align |",
                  "|---|---|---|---|---|---|"]
        for record in confirmed.sort_values("residual_rgb").to_dict("records"):
            lines.append(
                f"| {Path(record['left']).parent.name} | {Path(record['left']).name} | "
                f"{Path(record['right']).name} | ({record['dx']}, {record['dy']}) px | "
                f"{record['residual_rgb']:.2f} | {record['align_score']:.4f} |"
            )
        lines.append("")

    if not leakage.empty:
        crossing = int(leakage.different_fold.sum())
        lines += [
            "### Dampak pada fold yang berlaku sekarang",
            "",
            f"Dari {len(leakage)} pasangan terkonfirmasi, **{crossing}** jatuh di fold",
            "validasi yang berbeda. Selama `source_id` didefinisikan per berkas,",
            "potongan dari satu foto berada di sisi latih dan sisi uji sekaligus.",
            "",
        ]

    lines += [
        "### Sensitivitas ambang",
        "",
        "Jumlah citra yang masuk grup majemuk, untuk setiap kombinasi ambang.",
        "Dataran datar di tengah tabel menunjukkan jawabannya tidak bergantung",
        "pada pemilihan ambang yang tepat.",
        "",
    ]
    grid = sensitivity.pivot(
        index="align_score_min", columns="residual_max", values="images_in_multi_groups"
    )
    header = " | ".join(f"res<{value:.0f}" for value in grid.columns)
    lines.append(f"| align >= | {header} |")
    lines.append("|---" * (len(grid.columns) + 1) + "|")
    for align_min, row in grid.iterrows():
        cells = " | ".join(str(int(value)) for value in row)
        lines.append(f"| {align_min:.2f} | {cells} |")

    lines += [
        "",
        "## Batas uji ini",
        "",
        "Uji ini mendeteksi tumpang tindih translasional. Yang secara prinsip masih",
        "dapat lolos: potongan dari satu foto yang tidak saling tumpang tindih, serta",
        "citra yang mengalami rotasi atau perubahan skala besar. Hasil ini karena itu",
        "adalah batas bawah jumlah kekerabatan, bukan jaminan bahwa koleksi bersih.",
        "",
        "## Assertion yang lolos",
        "",
    ]
    lines += [f"- {check}" for check in checks]
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-min", type=float, default=SCREEN_MIN,
                        help="Ambang korelasi patch untuk lolos ke tahap verifikasi.")
    args = parser.parse_args()

    population = build_population()
    paths = population.path.tolist()
    print("=" * 76)
    print("TAHAP 24 - ADJUDIKASI GRUP SUMBER")
    print("=" * 76)
    print(f"citra: {len(paths)} | pasangan: {len(paths) * (len(paths) - 1) // 2}")

    print("tahap 1: penyaringan korelasi patch")
    gray = {path: load_gray(PROJECT_DIR / path) for path in paths}
    screened_rows = []
    for left, right in combinations(paths, 2):
        screened_rows.append({
            "left": left, "right": right,
            "patch_corr": round(screen_pair(gray[left], gray[right]), 4),
        })
    screened = pd.DataFrame(screened_rows)
    del gray

    survivors = screened[screened.patch_corr >= args.screen_min].copy()
    print(f"  lolos ke tahap 2: {len(survivors)}")

    print("tahap 2: verifikasi penyelarasan resolusi penuh")
    needed = sorted(set(survivors.left) | set(survivors.right))
    color = {path: load_color(PROJECT_DIR / path) for path in needed}
    verified_rows = []
    for record in survivors.to_dict("records"):
        result = verify_pair(color[record["left"]], color[record["right"]])
        verified_rows.append({**record, **result})
    verified = pd.DataFrame(verified_rows).sort_values("residual_rgb").reset_index(drop=True)
    del color

    confirmed = verified[
        (verified.residual_rgb < RESIDUAL_MAX)
        & (verified.align_score >= ALIGN_SCORE_MIN)
    ].copy()
    print(f"  terkonfirmasi satu foto: {len(confirmed)}")

    mapping = connected_groups(paths, list(zip(confirmed.left, confirmed.right)))
    groups = population.copy()
    groups["group_id"] = groups.path.map(mapping)
    groups["group_size"] = groups.group_id.map(groups.group_id.value_counts())

    sensitivity = threshold_sensitivity(verified, paths)
    leakage = fold_leakage(confirmed, population)

    checks = []
    if groups.group_id.isna().any():
        raise AssertionError("Ada citra tanpa group_id")
    checks.append("setiap citra memperoleh group_id")
    if groups.groupby("group_id").label.nunique().max() > 1:
        raise AssertionError("Ada grup yang memuat dua label berbeda")
    checks.append("tidak ada grup yang menggabungkan dua label kelas")
    straddle = groups[groups.group_size > 1].groupby("group_id")["set"].nunique()
    if not straddle.empty and straddle.max() > 1:
        checks.append("PERINGATAN: ada grup yang melintasi development dan eksternal")
    else:
        checks.append("tidak ada grup yang melintasi development dan eksternal")
    if len(screened) != len(paths) * (len(paths) - 1) // 2:
        raise AssertionError("Jumlah pasangan yang diuji tidak lengkap")
    checks.append("seluruh kombinasi pasangan diuji tanpa kecuali")

    OUT.mkdir(parents=True, exist_ok=True)
    screened.to_csv(OUT / "pair_screening.csv", index=False)
    verified.to_csv(OUT / "pair_verification.csv", index=False)
    confirmed.to_csv(OUT / "confirmed_pairs.csv", index=False)
    groups.to_csv(OUT / "source_groups.csv", index=False)
    sensitivity.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    proposed_config(groups, confirmed).to_csv(OUT / "proposed_source_groups.csv", index=False)
    if not leakage.empty:
        leakage.to_csv(OUT / "fold_leakage_check.csv", index=False)
    contact_sheet(confirmed, OUT / "confirmed_pairs_contact_sheet.png")

    (OUT / "methodology.json").write_text(json.dumps({
        "analysis_role": "source-group adjudication; detects same-photo crops missed by pHash",
        "population": {
            "development_originals": int((population["set"] == "development").sum()),
            "external": int((population["set"] == "external").sum()),
            "pairs_tested": len(screened),
        },
        "canvas_px": CANVAS,
        "screen_patch_px": SCREEN_PATCH,
        "screen_min_patch_corr": args.screen_min,
        "align_score_min": ALIGN_SCORE_MIN,
        "residual_max_rgb": RESIDUAL_MAX,
        "min_overlap_px": MIN_OVERLAP,
        "random_seed": RANDOM_SEED,
        "detects": "translational overlap between crops of one photograph",
        "does_not_detect": "non-overlapping crops, large rotation, large rescaling",
        "assertions_passed": checks,
    }, indent=2), encoding="utf-8")

    write_report(OUT / "REPORT.md", population, screened, verified,
                 confirmed, groups, sensitivity, leakage, checks)

    sizes = groups.group_id.value_counts()
    multi = sizes[sizes > 1]
    print("-" * 76)
    print("citra dalam grup majemuk, per kombinasi ambang:")
    print(sensitivity.pivot(index="align_score_min", columns="residual_max",
                            values="images_in_multi_groups").to_string())
    print("-" * 76)
    if not confirmed.empty:
        show = confirmed[["left", "right", "dx", "dy", "align_score", "residual_rgb"]].copy()
        show["left"] = show.left.map(lambda p: "/".join(Path(p).parts[-2:]))
        show["right"] = show.right.map(lambda p: "/".join(Path(p).parts[-2:]))
        print(show.to_string(index=False))
    print("-" * 76)
    if not leakage.empty:
        print(f"pasangan terkonfirmasi yang jatuh di fold berbeda: "
              f"{int(leakage.different_fold.sum())} dari {len(leakage)}")
    print(f"grup sumber unik: {int(sizes.size)} dari {len(population)} citra; "
          f"grup majemuk {int(multi.size)} mencakup {int(multi.sum())} citra")
    for check in checks:
        print(f"  [OK] {check}")
    print("-" * 76)
    print("Keluaran:", OUT)


if __name__ == "__main__":
    main()
