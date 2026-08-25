"""Audit, deduplikasi manifest, dan proteksi independensi uji eksternal."""

import pandas as pd

from pipeline_common import (
    dataset_records,
    file_sha256,
    read_image_color,
    reset_directory,
    resolve_project_path,
)
from pipeline_config import (
    AUDIT_DIR,
    DATASET_DIR,
    DEVELOPMENT_EXCLUSIONS_FILE,
    EXTERNAL_DIR,
    SOURCE_GROUPS_FILE,
)


def enrich(records: list[dict], source_set: str) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records, 1):
        path = resolve_project_path(record["path"])
        image = read_image_color(path)
        rows.append(
            {
                **record,
                "source_set": source_set,
                "sha256": file_sha256(path),
                "file_size_bytes": path.stat().st_size,
                "readable": image is not None,
                "width": int(image.shape[1]) if image is not None else 0,
                "height": int(image.shape[0]) if image is not None else 0,
                "extension": path.suffix.lower(),
            }
        )
        if index % 50 == 0 or index == len(records):
            print(f"  {source_set}: {index}/{len(records)} diaudit")
    return pd.DataFrame(rows)


def duplicate_report(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sha256, group in combined.groupby("sha256"):
        if len(group) < 2:
            continue
        sets, classes = set(group["source_set"]), set(group["kelas"])
        for row in group.to_dict("records"):
            rows.append(
                {
                    "sha256": sha256,
                    "path": row["path"],
                    "source_set": row["source_set"],
                    "kelas": row["kelas"],
                    "cross_development_external": sets == {"development", "external"},
                    "cross_class": len(classes) > 1,
                    "group_size": len(group),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "sha256", "path", "source_set", "kelas",
            "cross_development_external", "cross_class", "group_size",
        ],
    )


def clean_manifests(development: pd.DataFrame, external: pd.DataFrame):
    """Pertahankan eksternal; keluarkan salinannya dari development via manifest."""
    exclusions = []
    external_sorted = external.sort_values("path")
    external_duplicate = external_sorted.duplicated("sha256", keep="first")
    for row in external_sorted.loc[external_duplicate].to_dict("records"):
        exclusions.append({**row, "reason": "duplicate_within_external"})
    external_clean = external_sorted.loc[~external_duplicate].copy()

    external_hashes = set(external_clean["sha256"])
    development_sorted = development.sort_values("path")
    keep = []
    seen_hashes = set()
    for row in development_sorted.to_dict("records"):
        if row["sha256"] in external_hashes:
            exclusions.append({**row, "reason": "overlap_with_external"})
        elif row["sha256"] in seen_hashes:
            exclusions.append({**row, "reason": "duplicate_within_development"})
        else:
            keep.append(row)
            seen_hashes.add(row["sha256"])
    return pd.DataFrame(keep), external_clean, pd.DataFrame(exclusions)


def apply_approved_development_exclusions(
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply explicitly approved exclusions after exact-hash deduplication.

    Every decision must match source_id, relative path, and SHA-256. This makes
    the judgment auditable while leaving the physical source image untouched.
    """
    columns = [
        "source_id", "path", "sha256", "external_source_id", "external_path",
        "external_sha256", "reason", "status", "decision_date",
        "approval_record", "evidence_file",
    ]
    if not DEVELOPMENT_EXCLUSIONS_FILE.is_file():
        return development.copy(), pd.DataFrame(columns=development.columns), pd.DataFrame(columns=columns)

    decisions = pd.read_csv(DEVELOPMENT_EXCLUSIONS_FILE, dtype=str).fillna("")
    missing = set(columns) - set(decisions.columns)
    if missing:
        raise RuntimeError(f"Kolom konfigurasi eksklusi tidak lengkap: {sorted(missing)}")
    if decisions["source_id"].duplicated().any():
        raise RuntimeError("source_id duplikat pada konfigurasi eksklusi development")
    unsupported = set(decisions["status"]) - {"approved_exclude", "retain"}
    if unsupported:
        raise RuntimeError(f"Status keputusan eksklusi tidak dikenal: {sorted(unsupported)}")

    approved = decisions.query("status == 'approved_exclude'").copy()
    manifest_lookup = development.set_index("source_id", drop=False)
    exclusion_rows = []
    decision_audit_rows = []
    for decision in approved.to_dict("records"):
        source_id = decision["source_id"]
        if source_id not in manifest_lookup.index:
            raise RuntimeError(f"Eksklusi tidak cocok dengan manifest bersih: {source_id}")
        record = manifest_lookup.loc[source_id]
        if isinstance(record, pd.DataFrame):
            raise RuntimeError(f"source_id tidak unik dalam manifest: {source_id}")
        for field in ("path", "sha256"):
            if str(record[field]) != str(decision[field]):
                raise RuntimeError(
                    f"Konfigurasi eksklusi {source_id} tidak cocok pada {field}: "
                    f"manifest={record[field]!r}, config={decision[field]!r}"
                )
        exclusion_rows.append(
            {
                **record.to_dict(),
                "reason": decision["reason"],
                "exclusion_source": DEVELOPMENT_EXCLUSIONS_FILE.relative_to(
                    DEVELOPMENT_EXCLUSIONS_FILE.parent.parent
                ).as_posix(),
                "approval_status": decision["status"],
                "decision_date": decision["decision_date"],
                "approval_record": decision["approval_record"],
                "external_source_id": decision["external_source_id"],
                "external_path": decision["external_path"],
                "external_sha256": decision["external_sha256"],
                "evidence_file": decision["evidence_file"],
            }
        )
        decision_audit_rows.append({**decision, "matched_manifest": True})

    excluded_ids = set(approved["source_id"])
    cleaned = development.loc[~development["source_id"].isin(excluded_ids)].copy()
    excluded = pd.DataFrame(exclusion_rows)
    decision_audit = pd.DataFrame(decision_audit_rows)
    if set(cleaned["source_id"]) & excluded_ids:
        raise AssertionError("Eksklusi yang disetujui masih terdapat pada manifest development")
    if len(development) - len(cleaned) != len(approved):
        raise AssertionError("Jumlah eksklusi development tidak sesuai konfigurasi")
    return cleaned, excluded, decision_audit


def apply_source_groups(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tetapkan `group_id`, yaitu grain pemisahan fold yang sebenarnya.

    Secara default satu berkas adalah satu grup. Berkas yang terbukti berasal
    dari satu foto yang sama digabung menjadi satu grup lewat berkas keputusan
    `audit_config/source_groups.csv`, yang dihasilkan tahap 24.

    Alasannya: `source_id` adalah identitas berkas, bukan identitas foto. Dua
    potongan dari satu foto punya `source_id` berbeda dan karenanya dapat jatuh
    di sisi latih dan sisi uji sekaligus. `source_id` tetap dipertahankan untuk
    melacak keturunan augmentasi.

    Seperti pada eksklusi yang disetujui, setiap baris keputusan harus cocok
    pada source_id, path, dan SHA-256. Keputusan yang tidak cocok dengan
    manifest adalah kesalahan, bukan sesuatu yang boleh diabaikan diam-diam.
    """
    manifest = manifest.copy()
    manifest["group_id"] = manifest["source_id"]
    if not SOURCE_GROUPS_FILE.is_file():
        return manifest, pd.DataFrame()

    decisions = pd.read_csv(SOURCE_GROUPS_FILE, dtype=str).fillna("")
    required = {"group_key", "source_id", "path", "sha256"}
    missing = required - set(decisions.columns)
    if missing:
        raise RuntimeError(f"Kolom wajib hilang pada {SOURCE_GROUPS_FILE.name}: {sorted(missing)}")
    if decisions["source_id"].duplicated().any():
        raise RuntimeError("source_id duplikat pada konfigurasi grup sumber")

    lookup = manifest.set_index("source_id", drop=False)
    audit_rows = []
    for decision in decisions.to_dict("records"):
        source_id = decision["source_id"]
        if source_id not in lookup.index:
            raise RuntimeError(f"Grup sumber tidak cocok dengan manifest bersih: {source_id}")
        record = lookup.loc[source_id]
        for field in ("path", "sha256"):
            if str(record[field]) != decision[field]:
                raise RuntimeError(
                    f"Konfigurasi grup {source_id} tidak cocok pada {field}: "
                    f"manifest={record[field]!r}, config={decision[field]!r}"
                )
        manifest.loc[manifest["source_id"] == source_id, "group_id"] = decision["group_key"]
        audit_rows.append({**decision, "matched_manifest": True})

    sizes = manifest.groupby("group_id").size()
    for group_id, block in manifest[manifest["group_id"].isin(sizes[sizes > 1].index)].groupby("group_id"):
        if block["kelas"].nunique() > 1:
            raise AssertionError(f"Grup {group_id} menggabungkan dua kelas berbeda")

    return manifest, pd.DataFrame(audit_rows)


def main() -> None:
    reset_directory(AUDIT_DIR)
    print("=" * 72)
    print("TAHAP 01 — AUDIT DAN MANIFEST BERSIH")
    print("=" * 72)
    development_all = enrich(dataset_records(DATASET_DIR), "development")
    external_all = enrich(dataset_records(EXTERNAL_DIR), "external")
    combined = pd.concat([development_all, external_all], ignore_index=True)
    duplicates = duplicate_report(combined)
    development, external, exact_exclusions = clean_manifests(development_all, external_all)
    development, approved_exclusions, decision_audit = apply_approved_development_exclusions(
        development
    )
    exclusions = pd.concat([exact_exclusions, approved_exclusions], ignore_index=True, sort=False)
    development, group_audit = apply_source_groups(development)
    # Koleksi eksternal tidak pernah dipakai untuk membentuk fold, dan tahap 24
    # memastikan tidak ada grup yang melintasi development dan eksternal. Karena
    # itu setiap berkas eksternal berdiri sendiri sebagai satu grup.
    external = external.copy()
    external["group_id"] = external["source_id"]
    if not group_audit.empty:
        group_audit.to_csv(AUDIT_DIR / "approved_source_group_decisions.csv", index=False)

    development_all.to_csv(AUDIT_DIR / "development_manifest_all.csv", index=False)
    external_all.to_csv(AUDIT_DIR / "external_manifest_all.csv", index=False)
    development.to_csv(AUDIT_DIR / "development_manifest.csv", index=False)
    external.to_csv(AUDIT_DIR / "external_manifest.csv", index=False)
    duplicates.to_csv(AUDIT_DIR / "duplicate_report.csv", index=False)
    exclusions.to_csv(AUDIT_DIR / "excluded_from_analysis.csv", index=False)
    decision_audit.to_csv(AUDIT_DIR / "approved_exclusion_decisions.csv", index=False)
    counts = (
        pd.concat([development, external])
        .groupby(["source_set", "kelas", "subjenis"])
        .size().rename("jumlah").reset_index()
    )
    counts.to_csv(AUDIT_DIR / "counts_by_subtype.csv", index=False)

    raw_dev = development_all.groupby("kelas").size().to_dict()
    clean_dev = development.groupby("kelas").size().to_dict()
    ext_counts = external.groupby("kelas").size().to_dict()
    reason_counts = exclusions.groupby("reason").size().to_dict() if not exclusions.empty else {}
    report = "\n".join(
        [
            "# Audit Dataset dan Deduplikasi",
            "",
            f"- Development mentah: **{len(development_all)}** citra ({raw_dev}).",
            f"- Development bersih: **{len(development)}** citra ({clean_dev}).",
            f"- Uji eksternal bersih: **{len(external)}** citra ({ext_counts}).",
            f"- Eksklusi berbasis manifest: **{reason_counts}**.",
            f"- Keputusan eksklusi manual yang cocok dengan manifest: **{len(decision_audit)}**.",
            f"- Grain pemisahan fold (`group_id`) development: "
            f"**{development['group_id'].nunique()}** grup dari {len(development)} berkas.",
            f"- Berkas yang digabung ke grup foto bersama: **{len(group_audit)}**.",
            "",
            "Salinan citra yang terdapat pada uji eksternal dikeluarkan dari development "
            "tanpa menghapus berkas fisik. Ini menjaga independensi uji eksternal.",
            "Near-duplicate yang disetujui juga dikeluarkan hanya melalui manifest, "
            "dengan source ID, SHA-256, pasangan eksternal, bukti, dan rekaman keputusan.",
            "",
        ]
    )
    (AUDIT_DIR / "audit_report.md").write_text(report, encoding="utf-8")

    print(f"Development mentah : {raw_dev}")
    print(f"Development bersih : {clean_dev}")
    print(f"Uji eksternal      : {ext_counts}")
    print(f"Eksklusi manifest  : {reason_counts}")
    print("\nJumlah bersih per subjenis:")
    print(development.groupby(["kelas", "subjenis"]).size().to_string())
    print(f"\nHasil audit: {AUDIT_DIR}")

    if not combined["readable"].all():
        raise RuntimeError("Ada citra tidak terbaca; lihat manifest audit.")
    if not duplicates.empty and duplicates["cross_class"].any():
        raise RuntimeError("Duplikasi lintas kelas ditemukan dan harus ditinjau manual.")
    if set(development["source_id"]) & set(decision_audit.get("source_id", [])):
        raise RuntimeError("Eksklusi yang disetujui masih masuk manifest development bersih.")


if __name__ == "__main__":
    main()
