# R2.7 - Perbandingan Strategi Penyeimbangan Kelas

Diagnostik sensitivitas untuk Reviewer 2 butir 7. Ketiga lengan memakai
pembagian outer/inner yang identik, ruang kandidat yang sama, dan hanya
original tanpa augmentasi pada inner validation serta outer test. Koleksi
eksternal tidak pernah dimuat.

- Desain: 5 repeat x 5 outer fold x 4 inner fold.
- Original development bersih: 201 (137 batik, 64 non-batik).
- Ruang kandidat: 4 kelompok fitur x 3 keluarga model.
- Seed dasar: 42.

## Estimasi tingkat repeat

                       arm  macro_f1_mean  macro_f1_std  balanced_accuracy_mean  balanced_accuracy_std  mcc_mean  recall_batik_mean  recall_non_batik_mean  n_repeats
        augmented_balanced       0.907305      0.012955                0.907573               0.009989  0.814880           0.940146               0.875000          5
  class_weighted_originals       0.898896      0.010963                0.907664               0.013688  0.799553           0.915328               0.900000          5
balanced_original_sampling       0.895993      0.012459                0.906307               0.011878  0.794383           0.909489               0.903125          5

## Selisih berpasangan terhadap lengan augmentasi

                                         comparison            metric  mean_difference  std_difference  n_paired_folds  folds_better  folds_worse  folds_tied  min_difference  max_difference
  class_weighted_originals minus augmented_balanced          f1_macro        -0.008500        0.031953              25             8           11           6       -0.069887        0.062442
  class_weighted_originals minus augmented_balanced balanced_accuracy         0.000364        0.034693              25             9           10           6       -0.076923        0.076923
  class_weighted_originals minus augmented_balanced               mcc        -0.015585        0.059526              25             8           11           6       -0.115500        0.111406
balanced_original_sampling minus augmented_balanced          f1_macro        -0.011156        0.044502              25            10           12           3       -0.108395        0.062442
balanced_original_sampling minus augmented_balanced balanced_accuracy        -0.000879        0.044477              25            11           11           3       -0.094017        0.076923
balanced_original_sampling minus augmented_balanced               mcc        -0.021674        0.084805              25            10           12           3       -0.217089        0.111406

## Frekuensi seleksi model

                       arm        kind           choice  count  of
        augmented_balanced       model    Random Forest     13  25
        augmented_balanced       model        SVM (RBF)     12  25
        augmented_balanced feature_set Gabungan 6 Fitur     25  25
  class_weighted_originals       model        SVM (RBF)     19  25
  class_weighted_originals       model    Random Forest      6  25
  class_weighted_originals feature_set Gabungan 6 Fitur     25  25
balanced_original_sampling       model        SVM (RBF)     21  25
balanced_original_sampling       model    Random Forest      4  25
balanced_original_sampling feature_set Gabungan 6 Fitur     25  25

## Catatan interpretasi

Perbandingan ini menilai jadwal penyeimbangan, bukan mengubah definisi fitur
atau keluarga model. Nilai di sini merupakan diagnostik sensitivitas internal
dan tidak dapat dipertukarkan dengan estimasi single-loop lima fold yang aktif.
Seluruh selisih dilaporkan berpasangan pada tingkat fold agar variasi antarfold
tidak disalahartikan sebagai efek strategi.
