# Tahap 24 - Adjudikasi grup sumber

Deteksi potongan citra yang berasal dari satu foto yang sama, yang lolos
dari audit perceptual hash karena pHash tidak invarian terhadap geseran.

## Populasi

- Development original: 201
- Eksternal: 60
- Total citra: 261
- Pasangan diuji: 33930 (seluruh kombinasi, lintas kelas dan subjenis)

## Ambang

- Penyaringan tahap 1: korelasi patch >= 0.6
- Konfirmasi: residual RGB < 15.0 dan skor penyelarasan >= 0.95

Residual tidak pernah nol karena tiap potongan di-resize dan dikompresi
ulang secara terpisah.

## Hasil

- Lolos penyaringan ke tahap 2: 125
- Pasangan terkonfirmasi satu foto: 8
- Grup sumber unik: 254 (dari 261 citra)
- Grup beranggota lebih dari satu: 6
- Citra yang terlibat: 13

### Pasangan terkonfirmasi

| Subjenis | Kiri | Kanan | Geseran | Residual | Align |
|---|---|---|---|---|---|
| Yogyakarta_Parang | 0012.jpg | 0017.jpg | (-32, 0) px | 4.72 | 0.9982 |
| batik semarangan | semarangan3.jpg | semarangan5.jpg | (0, 0) px | 5.57 | 0.9929 |
| Solo_Parang | 0032.jpg | 0040.jpg | (0, -36) px | 5.95 | 0.9742 |
| Yogyakarta_Kawung | 0010.jpg | 0030.jpg | (-24, 0) px | 6.24 | 0.9958 |
| Solo_Parang | 0028.jpg | 0034.jpg | (-51, 0) px | 7.45 | 0.9869 |
| Solo_Parang | 0017.jpg | 0021.jpg | (21, 0) px | 13.29 | 0.9690 |
| Yogyakarta_Parang | 0012.jpg | 009.jpg | (-48, -21) px | 14.45 | 0.9654 |
| Yogyakarta_Parang | 0017.jpg | 009.jpg | (-36, 0) px | 14.78 | 0.9697 |

### Dampak pada fold yang berlaku sekarang

Dari 8 pasangan terkonfirmasi, **8** jatuh di fold
validasi yang berbeda. Selama `source_id` didefinisikan per berkas,
potongan dari satu foto berada di sisi latih dan sisi uji sekaligus.

### Sensitivitas ambang

Jumlah citra yang masuk grup majemuk, untuk setiap kombinasi ambang.
Dataran datar di tengah tabel menunjukkan jawabannya tidak bergantung
pada pemilihan ambang yang tepat.

| align >= | res<10 | res<15 | res<20 | res<25 | res<30 | res<40 |
|---|---|---|---|---|---|---|
| 0.80 | 10 | 13 | 13 | 15 | 17 | 25 |
| 0.85 | 10 | 13 | 13 | 15 | 17 | 25 |
| 0.90 | 10 | 13 | 13 | 15 | 15 | 17 |
| 0.93 | 10 | 13 | 13 | 13 | 13 | 13 |
| 0.95 | 10 | 13 | 13 | 13 | 13 | 13 |
| 0.97 | 10 | 10 | 10 | 10 | 10 | 10 |
| 0.99 | 6 | 6 | 6 | 6 | 6 | 6 |

## Batas uji ini

Uji ini mendeteksi tumpang tindih translasional. Yang secara prinsip masih
dapat lolos: potongan dari satu foto yang tidak saling tumpang tindih, serta
citra yang mengalami rotasi atau perubahan skala besar. Hasil ini karena itu
adalah batas bawah jumlah kekerabatan, bukan jaminan bahwa koleksi bersih.

## Assertion yang lolos

- setiap citra memperoleh group_id
- tidak ada grup yang menggabungkan dua label kelas
- tidak ada grup yang melintasi development dan eksternal
- seluruh kombinasi pasangan diuji tanpa kecuali
