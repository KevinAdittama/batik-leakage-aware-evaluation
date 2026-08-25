# Indeks Hasil Paper

## Pipeline utama, dijalankan `run_all.py`

- `01_audit/`: manifest bersih, hitungan data, laporan duplikasi dan eksklusi,
  serta keputusan pengelompokan foto sumber yang disetujui.
- `02_preview_original/`: panel edge, Otsu, garis, kurva, LBP, dan FFT.
- `03_augmentasi/`: manifest final train/CV pool serta ringkasan augmentasi.
- `04_preview_augmented/`: visual QA hasil transformasi.
- `05_fitur/`: CSV enam fitur untuk development, train, CV pool, dan eksternal.
- `06_analisis_fitur/`: statistik, korelasi, distribusi, dan separasi fitur.
- `07_cross_validation/`: metrik fold, OOF predictions, ablasi, confusion matrix.
- `08_uji_eksternal/`: model final, prediksi, metrik per kelas, confusion matrix,
  serta panel penjelasan dalam `visualisasi_step_by_step/`.
- `09_tabel_paper/`: tabel Markdown/CSV dan grafik utama paper.
- `10_analisis_error/`: daftar kesalahan dan ringkasan per subjenis.
- `11_deep_learning_baseline/`: pembanding ResNet18 dan MobileNetV2 beku.
- `12_submission_robustness/`: pemeriksaan ketahanan menjelang submission.

## Analisis tambahan hasil revisi

- `14_repeated_nested_augmentation/`: nested berulang lima repeat, lima outer,
  empat inner, dengan augmentasi lokal per fold.
- `19_balancing_comparison/`: tiga lengan penyeimbangan kelas.
- `24_source_groups/`: adjudikasi potongan dari satu foto. Penyaringan pasangan,
  verifikasi resolusi penuh, pasangan terkonfirmasi, dan pemeriksaan kebocoran
  antar-fold.
- `25_model_selection_stability/`: margin seleksi dibandingkan derau antar-fold,
  peringkat single-loop, frekuensi seleksi nested, dan hasil eksternal per model.
- `29_figures/`: gambar yang dipakai manuskrip.
- `30_controlled_ablation/`: ablasi terkontrol cabang deep, delapan kondisi
  menyilangkan warna, pretraining, dan dimensionalitas.

## Paket kiriman

- `37_paket_audit_ahli/`: paket berkode untuk validasi label independen.
  **Memuat peta kode rahasia; tidak boleh diunggah ke mana pun.**
- `38_contoh_normalisasi/`: contoh visual normalisasi paket audit.

## Mulai dari mana

Baca `09_tabel_paper/paper_results_summary.md` lebih dahulu, lalu
`01_audit/audit_report.md` dan `10_analisis_error/error_analysis.md` untuk
konteks.

Untuk memahami mengapa pemisahan fold dilakukan pada tingkat grup foto sumber
dan bukan tingkat berkas, baca `24_source_groups/` beserta
`25_model_selection_stability/selection_margin.csv`.

## Catatan bagi pembaca repositori publik

Repositori publik hanya memuat artefak berbentuk teks dan model terlatih.
Folder yang isinya citra tidak ikut diterbitkan, yaitu `02_preview_original/`,
`04_preview_augmented/`, `29_figures/`, `37_paket_audit_ahli/`, dan
`38_contoh_normalisasi/`. Alasannya ada di `RECONSTRUCTION.md`.
