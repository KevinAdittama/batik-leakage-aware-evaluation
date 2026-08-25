# Deep-learning baseline note

Baseline deep learning dievaluasi sebagai frozen ImageNet feature extractor,
bukan sebagai model end-to-end yang di-fine-tune penuh. Dua backbone digunakan:
ResNet18 dan MobileNetV2. Untuk setiap fold, backbone dibekukan dan hanya
classifier head linear/logistic dilatih pada training fold yang sudah seimbang.

Aturan anti-kebocoran:

- Split fold mengikuti `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Augmentasi hanya berasal dari source ID training fold.
- Validation fold memakai citra development asli, bukan augmentasi.
- Uji eksternal memakai citra asli dan tidak dipakai untuk model selection.
- StandardScaler pada embedding CNN berada di dalam Pipeline classifier dan
  di-fit hanya pada data training fold.

Model DL terbaik berdasarkan CV macro-F1: **MobileNetV2**
(0.960 ± 0.038).

Catatan interpretasi untuk paper: baseline ini menjawab komentar reviewer/dosen
bahwa pembanding deep learning perlu tersedia, tetapi fokus utama paper tetap
pada enam fitur interpretable.
