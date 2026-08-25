# Audit Dataset dan Deduplikasi

- Development mentah: **212** citra ({'batik': 139, 'non_batik': 73}).
- Development bersih: **201** citra ({'batik': 137, 'non_batik': 64}).
- Uji eksternal bersih: **60** citra ({'batik': 30, 'non_batik': 30}).
- Eksklusi berbasis manifest: **{'duplicate_within_development': 1, 'overlap_with_external': 8, 'perceptual_near_duplicate_with_external': 2}**.
- Keputusan eksklusi manual yang cocok dengan manifest: **2**.
- Grain pemisahan fold (`group_id`) development: **194** grup dari 201 berkas.
- Berkas yang digabung ke grup foto bersama: **13**.

Salinan citra yang terdapat pada uji eksternal dikeluarkan dari development tanpa menghapus berkas fisik. Ini menjaga independensi uji eksternal.
Near-duplicate yang disetujui juga dikeluarkan hanya melalui manifest, dengan source ID, SHA-256, pasangan eksternal, bukti, dan rekaman keputusan.
