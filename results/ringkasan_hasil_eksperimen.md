# Ringkasan Hasil Eksperimen — Deteksi Deepfake Hibrida
## EfficientNet-B0 + DCT 192-dim vs EfficientNet-B0 (Baseline)

> **Dataset:** Subset Twitter TrueFake (Dell'Anna et al., 2025) — 10.000 REAL + 21.250 FAKE = 31.250 total  
> **Split:** 72% train · 18% val · 10% test (stratified, seed=42)  
> **Backbone:** EfficientNet-B0 pretrained ImageNet  
> **DCT:** statistik koefisien blok 8×8 pada kanal luminansi Y (mean + var + skew = 192-dim)

---

## 1. Identifikasi Model

| ID | Nama | Arsitektur | Checkpoint | Epoch Terbaik |
|---|---|---|---|---|
| **E-1** | Hybrid | EfficientNet-B0 + DCT 192-dim | `best_efficient_dct.pth` | 8 |
| **E-2** | Baseline | EfficientNet-B0 saja | `best_efficient_no_dct.pth` | 9 |

---

## 2. Performa Training & Validasi (Best Checkpoint)

| Metrik | E-1 (Hybrid) | E-2 (Baseline) |
|---|---|---|
| Best Val AUC | **0.9866** (ep.8) | 0.9865 (ep.9) |
| Best Val Accuracy | 0.9460 | **0.9470** |
| Best Val Macro F1 | 0.9382 | **0.9396** |
| Best Val F1 FAKE | 0.9163 | **0.9172** |
| Best Val F1 REAL | 0.9601 | **0.9608** |

---

## 3. Evaluasi Test Set — Best Checkpoint (n=3.125 sampel)

*Data ini adalah angka utama untuk dilaporkan di Skripsi*

| Metrik | E-1 (Hybrid) | E-2 (Baseline) | Delta (E1−E2) | Unggul |
|---|---|---|---|---|
| **AUC** | 0.9895 | 0.9901 | −0.0006 | E-2 |
| **Accuracy** | 0.9229 | 0.9555 | −0.0326 | E-2 |
| **Macro F1** | 0.9157 | 0.9492 | −0.0335 | E-2 |
| **F1 FAKE** | 0.9403 | 0.9671 | −0.0268 | E-2 |
| **F1 REAL** | 0.8911 | 0.9312 | −0.0401 | E-2 |
| **Prec FAKE** | **0.9927** | 0.9715 | +0.0212 | **E-1** |
| **Rec FAKE** | 0.8932 | 0.9628 | −0.0696 | E-2 |
| **Prec REAL** | 0.8129 | 0.9225 | −0.1096 | E-2 |
| **Rec REAL** | **0.9860** | 0.9400 | +0.0460 | **E-1** |

---

## 4. Uji Signifikansi Statistik — DeLong's Test (Clean Val Set)

| Parameter | Nilai |
|---|---|
| AUC E-1 (clean) | 0.994530 |
| AUC E-2 (clean) | 0.991491 |
| AUC Diff (E1−E2) | +0.003039 |
| Z-statistic | **7.688** |
| P-value | **< 0.0001** |
| Kesimpulan | **SIGNIFIKAN** secara statistik (p < 0.05) |

> Meskipun delta AUC kecil (0.003), perbedaan ini signifikan secara statistik karena ukuran sampel besar.

---

## 5. Evaluasi Robustness (Validation Subset, n=2.000)

*Penurunan performa (Delta AUC) dari kondisi Clean ke kondisi Terdegradasi. Nilai Delta yang **lebih kecil** menunjukkan ketahanan yang **lebih baik** (lebih robust).*

### 5a. Kompresi JPEG (Compression Artifacts)

| Kondisi | AUC E-1 | ΔE-1 | AUC E-2 | ΔE-2 | Lebih Robust |
|---|---|---|---|---|---|
| Clean | 0.9881 | 0.0000 | 0.9898 | 0.0000 | — |
| JPEG-70 | 0.9833 | 0.0048 | 0.9858 | 0.0040 | **E-2** |
| JPEG-50 | 0.9852 | 0.0029 | 0.9866 | 0.0032 | **E-1** |
| JPEG-30 | 0.9770 | 0.0111 | 0.9772 | 0.0126 | **E-1** |

✅ **E-1 lebih robust pada kompresi tinggi (JPEG-30 & 50)**, sejalan dengan teori bahwa DCT menyimpan sinyal frekuensi yang resisten terhadap kompresi agresif.

### 5b. Gaussian Noise (Noise Aditif)

| Kondisi | AUC E-1 | ΔE-1 | AUC E-2 | ΔE-2 | Lebih Robust |
|---|---|---|---|---|---|
| Noise-σ10 | 0.9343 | 0.0538 | 0.9297 | 0.0601 | **E-1** |
| Noise-σ25 | 0.6105 | 0.3776 | 0.5917 | 0.3982 | **E-1** |
| Noise-σ50 | 0.5301 | 0.4580 | 0.5115 | 0.4783 | **E-1** |

✅ **E-1 secara konsisten lebih robust terhadap Noise.** Fitur frekuensi global rupanya mampu menahan degradasi spasial acak lebih baik dibanding pure CNN.

### 5c. Gaussian Blur (High-Frequency Loss) & Downscale

| Kondisi | AUC E-1 | ΔE-1 | AUC E-2 | ΔE-2 | Lebih Robust |
|---|---|---|---|---|---|
| Blur-3×3 | 0.9879 | 0.0002 | 0.9889 | 0.0010 | **E-1** |
| Blur-5×5 | 0.9869 | 0.0012 | 0.9875 | 0.0023 | **E-1** |
| Blur-7×7 | 0.9839 | 0.0042 | 0.9843 | 0.0055 | **E-1** |
| DS-0.5 | 0.9869 | 0.0012 | 0.9877 | 0.0021 | **E-1** |
| DS-0.25 | 0.9839 | 0.0042 | 0.9841 | 0.0057 | **E-1** |

✅ **E-1 secara konsisten lebih robust terhadap kehilangan resolusi/ketajaman.**

### 5d. Occlusion (Kehilangan Informasi Spasial Kontigu)

| Kondisi | AUC E-1 | ΔE-1 | AUC E-2 | ΔE-2 | Lebih Robust |
|---|---|---|---|---|---|
| Occ-30% | 0.9683 | 0.0198 | 0.9753 | 0.0145 | **E-2** |
| Occ-50% | 0.9388 | 0.0493 | 0.9529 | 0.0369 | **E-2** |
| Occ-70% | 0.8463 | 0.1418 | 0.8610 | 0.1289 | **E-2** |

❌ **E-1 (Hybrid) signifikan lebih LEMAH menghadapi Occlusion.** 
*Penjelasan:* Kotak hitam besar di tengah wajah menghancurkan nilai kontigu dari gambar, menciptakan sinyal frekuensi tajam buatan (sharp edges dari blok hitam) yang merusak pola DCT global secara fatal. Sebaliknya, CNN murni (E-2) masih bisa mengekstraksi ciri-ciri deepfake dari sisa area wajah yang tidak tertutup.

---

## 6. Efisiensi Komputasi

| Metrik | E-1 (Hybrid) | E-2 (Baseline) | Selisih |
|---|---|---|---|
| Total Parameter | 4.010.494 | 4.010.110 | +384 (0.01%) |
| Ukuran Checkpoint | 46.35 MB | 46.35 MB | +0.0015 MB |
| Waktu Inferensi | 550.4 ± 8.5 ms | 552.9 ± 7.1 ms | Setara |

---

## 7. Temuan Utama (Untuk BAB IV & V Skripsi)

### Menjawab RQ1 — Akurasi & Kinerja Klasifikasi
> **Penambahan DCT tidak menaikkan performa AUC secara keseluruhan (test set), namun spesifik meningkatkan Presisi deteksi FAKE.** E-1 lebih menekan *False Positives*, menjadikannya ideal jika prioritas sistem adalah menghindari salah blok pada wajah manusia asli (REAL).

### Menjawab RQ2 — Robustness
Hasil eksperimen mengoreksi asumsi awal. Integrasi spasial-frekuensi memberikan ketahanan terarah:
1. **Model Hibrida (E-1) LEBIH KUAT menghadapi degradasi menyeluruh (global):** JPEG Compression, Downscaling, Gaussian Blur, dan Gaussian Noise. Analisis DCT secara efektif membantu model mendeteksi anomali saat kualitas piksel secara merata turun.
2. **Model Hibrida (E-1) LEBIH LEMAH menghadapi degradasi spasial lokal yang drastis (Occlusion).** Kerusakan besar pada satu area tertentu merusak perhitungan frekuensi global DCT. CNN murni (E-2) lebih handal di situasi wajah tertutup benda.

*Ini memperkuat teori bahwa tidak ada peluru perak dalam augmentasi, dan fusi spasial-frekuensi (DCT) unggul untuk mengatasi artefak media sosial (kompresi & penurunan kualitas).*
