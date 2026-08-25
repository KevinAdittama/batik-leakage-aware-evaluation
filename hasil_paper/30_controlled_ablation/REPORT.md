# Tahap 30 - Ablasi terkontrol ResNet18 (R2.4)

Tiga faktor divariasikan satu per satu pada fold nested yang identik: warna,
pretraining, dan dimensionalitas. Center cropping sengaja tidak divariasikan;
alasannya ada di bagian batas di bawah.

## Pengaman ekstraksi

- Deviasi maksimum embedding baseline terhadap tahap 11: `0.000e+00`
- Toleransi: `1e-05`
- Lolos: **True**

Tanpa pengaman ini, selisih antarkondisi bisa berasal dari perbedaan harness
dan bukan dari faktor yang divariasikan.

## Kinerja per kondisi

| Kondisi | Macro-F1 | Bal. accuracy | MCC | Recall B | Recall NB |
|---|---:|---:|---:|---:|---:|
| gray+pretrained+full512 | 0.940 +/- 0.016 | 0.942 | 0.879 | 0.956 | 0.928 |
| rgb+pretrained+full512 (baseline) | 0.937 +/- 0.009 | 0.935 | 0.873 | 0.964 | 0.906 |
| rgb+pretrained+pca6 | 0.911 +/- 0.026 | 0.913 | 0.822 | 0.939 | 0.887 |
| gray+pretrained+pca6 | 0.910 +/- 0.015 | 0.910 | 0.821 | 0.945 | 0.875 |
| gray+random+pca6 | 0.857 +/- 0.017 | 0.855 | 0.714 | 0.915 | 0.794 |
| gray+random+full512 | 0.826 +/- 0.026 | 0.828 | 0.652 | 0.883 | 0.772 |
| rgb+random+full512 | 0.806 +/- 0.029 | 0.804 | 0.613 | 0.883 | 0.725 |
| rgb+random+pca6 | 0.785 +/- 0.019 | 0.791 | 0.572 | 0.842 | 0.741 |

## Efek tiap faktor

Selisih macro-F1 dihitung berpasangan pada tingkat fold, lalu dirata-ratakan
di seluruh kombinasi faktor lainnya. Rentang menunjukkan apakah efeknya
konsisten atau bergantung pada setelan lain.

| Faktor | Kontras | Rata-rata | Terendah | Tertinggi |
|---|---|---:|---:|---:|
| color | rgb minus gray | -0.0244 | -0.0723 | -0.0004 |
| dimensionality | full512 minus pca6 | +0.0104 | -0.0317 | +0.0287 |
| pretraining | pretrained minus random | +0.1065 | +0.0542 | +0.1310 |

## Batas ablasi ini

Center cropping tidak divariasikan. Crop merupakan bagian dari transform
bawaan bobot ImageNet, sehingga melepasnya turut mengubah resize dan
normalisasi; efeknya tidak dapat dipisahkan dari faktor lain dan hasilnya
akan sulit ditafsirkan. Reviewer menyebut crop secara eksplisit, jadi
pembatasan ini dinyatakan terbuka, bukan diabaikan.

Ablasi ini juga memakai satu backbone saja. Hasilnya berlaku untuk ResNet18
pada koleksi ini dan tidak digeneralisasi ke arsitektur lain.

Ketiga head memakai StandardScaler, berbeda dari tahap 19 yang tidak
menskalakan Random Forest. Penyeragaman itu konstan di seluruh kondisi
sehingga tidak memengaruhi perbandingan antarkondisi, tetapi angka di sini
tidak dapat dibandingkan langsung dengan tabel handcrafted.

## Assertion yang lolos

- ekstraksi baseline identik dengan tahap 11 (deviasi maks 0.00e+00)
- setiap kondisi menghasilkan 25 outer fold
- seluruh kondisi memakai pembagian fold yang identik
- setiap original diprediksi tepat sekali per repeat di semua kondisi
