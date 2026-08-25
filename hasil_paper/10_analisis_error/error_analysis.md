# Analisis Error Eksternal

- Model formal: **SVM (RBF)**.
- Total uji: **60**.
- Salah prediksi: **20**.
- Batik → non-batik: **14**.
- Non-batik → batik: **6**.

## Subjenis yang mengalami kesalahan

- `non_batik/tenun_ikat`: 4/4 salah (100.0%).
- `batik/megamendung`: 4/5 salah (80.0%).
- `batik/betawi`: 3/5 salah (60.0%).
- `batik/kawung`: 3/5 salah (60.0%).
- `batik/sogan`: 2/5 salah (40.0%).
- `non_batik/bunga`: 1/4 salah (25.0%).
- `batik/kalimantan_dayak`: 1/5 salah (20.0%).
- `batik/parang`: 1/5 salah (20.0%).
- `non_batik/kotak`: 1/7 salah (14.3%).

Subjenis hanya dipakai untuk analisis; label selalu berasal dari folder kelas.
