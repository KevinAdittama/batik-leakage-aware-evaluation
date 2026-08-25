# Tahap 25 - Stabilitas seleksi model formal

Aturan seleksi tidak berubah: macro-F1 CV tertinggi pada enam fitur
gabungan. Yang diukur di sini adalah apakah aturan itu masih memisahkan
model secara bermakna setelah fold dipisahkan pada grain foto sumber.

## Margin single-loop

| Protokol | Pemenang | Peringkat 2 | Margin | SD antar-fold | Galat baku | Margin dalam galat baku |
|---|---|---|---:|---:|---:|---:|
| file-level | Random Forest (0.9070) | SVM (RBF) (0.8982) | 0.0088 | 0.0325 | 0.0145 | 0.61 |
| source-group-aware | SVM (RBF) (0.9108) | Random Forest (0.9039) | 0.0069 | 0.0461 | 0.0206 | 0.33 |

Margin sebesar sepersekian galat baku tidak dapat membedakan dua model.
Penobatan pemenang formal karena itu tidak dapat dipertanggungjawabkan
sebagai klaim keunggulan.

## Frekuensi seleksi nested

| Protokol | Lengan | Model | file-level | source-group-aware |
|---|---|---|---:|---:|
| nested single-repeat (tahap 12) | - | SVM (RBF) | 3/5 | 5/5 |
| nested 5 repeat (tahap 14) | fold-local augmentation | Random Forest | 5/25 | 13/25 |
| nested 5 repeat (tahap 14) | fold-local augmentation | SVM (RBF) | 20/25 | 12/25 |
| nested 5 repeat (tahap 19) | augmented_balanced | Random Forest | 5/25 | 13/25 |
| nested 5 repeat (tahap 19) | augmented_balanced | SVM (RBF) | 20/25 | 12/25 |
| nested 5 repeat (tahap 19) | balanced_original_sampling | SVM (RBF) | 23/25 | 21/25 |
| nested 5 repeat (tahap 19) | balanced_original_sampling | Random Forest | 2/25 | 4/25 |
| nested 5 repeat (tahap 19) | class_weighted_originals | SVM (RBF) | 21/25 | 19/25 |
| nested 5 repeat (tahap 19) | class_weighted_originals | Random Forest | 4/25 | 6/25 |

## Hasil eksternal seluruh model

Dilaporkan untuk semua model, bukan hanya model formal. Hasil eksternal
tidak bergantung pada struktur fold, sehingga identik pada kedua protokol.

| Model | Keluarga | Bal. accuracy | Macro-F1 | MCC | Recall batik | Recall non-batik |
|---|---|---:|---:|---:|---:|---:|
| MobileNetV2 | frozen deep benchmark | 0.867 | 0.866 | 0.740 | 0.800 | 0.933 |
| ResNet18 | frozen deep benchmark | 0.833 | 0.833 | 0.668 | 0.867 | 0.800 |
| SVM (RBF) | handcrafted six features | 0.667 | 0.661 | 0.346 | 0.533 | 0.800 |
| Random Forest | handcrafted six features | 0.617 | 0.614 | 0.237 | 0.533 | 0.700 |
| Logistic Regression | handcrafted six features | 0.500 | 0.495 | 0.000 | 0.600 | 0.400 |

## Batas pembacaan

Dua hal berubah bersamaan antara kedua protokol: tujuh citra kini terkunci
dalam grup yang sama, dan `StratifiedGroupKFold` mempartisi secara berbeda
dari `StratifiedKFold`. Selisih di sini karena itu adalah sensitivitas
terhadap protokol, bukan estimasi kausal atas besarnya kebocoran.

## Assertion yang lolos

- model formal tersimpan sama dengan pemenang tabel CV
- frekuensi seleksi menjumlah ke seluruh outer fold pada tiap protokol
- margin puncak lebih kecil daripada simpangan antar-fold
