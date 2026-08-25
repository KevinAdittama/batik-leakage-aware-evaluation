# Ringkasan Hasil untuk Paper

## Data

- Development asli: 201 citra.
- Uji eksternal independen: 60 citra.
- Final train: 400 training instances seimbang (200 per kelas).
- Augmentasi hanya digunakan pada train; validation dan eksternal tetap asli.

## Model formal

- Dipilih dari CV: **SVM (RBF)**.
- CV macro-F1: **0.911 ± 0.043**.
- External macro-F1: **0.661**.
- External recall batik: **0.533**.
- External recall non-batik: **0.800**.

Model formal dipilih hanya dari CV. Uji eksternal tidak digunakan untuk seleksi.

## Tabel model

| Model | CV Accuracy | CV Balanced Accuracy | CV Macro-F1 | CV MCC | CV Recall Batik | CV Recall Non-Batik | External Accuracy | External Balanced Accuracy | External Macro-F1 | External MCC | External Recall Batik | External Recall Non-Batik |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM (RBF) | 0.921 ± 0.040 | 0.917 ± 0.028 | 0.911 ± 0.043 | 0.830 ± 0.080 | 0.928 ± 0.076 | 0.906 ± 0.064 | 0.667 | 0.667 | 0.661 | 0.346 | 0.533 | 0.800 |
| Random Forest | 0.915 ± 0.029 | 0.909 ± 0.026 | 0.904 ± 0.031 | 0.814 ± 0.062 | 0.928 ± 0.056 | 0.891 ± 0.068 | 0.617 | 0.617 | 0.614 | 0.237 | 0.533 | 0.700 |
| Logistic Regression | 0.841 ± 0.060 | 0.834 ± 0.062 | 0.823 ± 0.064 | 0.653 ± 0.128 | 0.854 ± 0.068 | 0.814 ± 0.086 | 0.500 | 0.500 | 0.495 | 0.000 | 0.600 | 0.400 |
