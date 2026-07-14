# Final Project Computer Vision: Artifice vs Nature

Laporan skripsi, kode, dataset, serta semua artefak dari proses eksperimen.

---

# 📋 Konten

## Laporan Akhir
- [README_LAPORAN.md](README_LAPORAN.md)

## Dataset
- [data/](../data/)

## Eksperimen & Pelatihan
- [src/training/](./src/training/)
- [results/training_experiments/](./results/training_experiments/)

## Task-Spesifik Analisis
- [src/analysis/](./src/analysis/)
- [results/bab4_tambahan/](./results/bab4_tambahan/)
- [results/test_set_eval/](./results/test_set_eval/)

## Model & Checkpoint
- `models/best_*.pth` & `models/latest_*.pth`
- `models/checkpoints/best_*.pth` & `models/checkpoints/latest_*.pth`

## Logging
- [logs/](./logs/)

## Visualisasi
- [notebooks/](./notebooks/)



---

# Ringkasan Hasil Eksperimen — Deteksi Deepfake Hibrida
## EfficientNet-B0 + DCT 192-dim vs EfficientNet-B0 (Baseline)

> **Dataset:** Subset Twitter TrueFake — 10.000 REAL + 21.250 FAKE = 31.250 total | **Split:** 72/18/10% (stratified, seed=42)

---

## Model

| ID | Arsitektur | Checkpoint | Epoch Terbaik |
|---|---|---|---|
| **E-1** | EfficientNet-B0 + DCT 192-dim | `best_efficient_dct.pth` | 8 |
| **E-2** | EfficientNet-B0 saja | `best_efficient_no_dct.pth` | 9 |

---

## Test Set — Best Checkpoint *(Angka Utama Skripsi)*

| Metrik | E-1 Hybrid | E-2 Baseline | Unggul |
|---|---|---|---|
| **AUC** | 0.9895 | 0.9901 | E-2 |
| **Accuracy** | 0.9229 | 0.9555 | E-2 |
| **Macro F1** | 0.9157 | 0.9492 | E-2 |
| **F1 FAKE** | 0.9403 | 0.9671 | E-2 |
| **Prec FAKE** | **0.9927** | 0.9715 | **E-1** ✅ |
| **Rec REAL** | **0.9860** | 0.9400 | **E-1** ✅ |

---

## Robustness — Delta AUC

| Kondisi | ΔE-1 | ΔE-2 | Lebih Robust |
|---|---|---|---|
| JPEG-30 | −0.0053 | −0.0094 | **E-1** ✅ |
| JPEG-50 | −0.0010 | −0.0026 | **E-1** ✅ |
| Noise-σ25 | −0.1504 | −0.0826 | **E-2** ❌ |
| Blur-7×7 | −0.0018 | −0.0038 | **E-1** ✅ |
| DS-0.25 | −0.0030 | −0.0080 | **E-1** ✅ |
| Occlusion | *belum dijalankan* | — | ⏳ |

---

## DeLong's Test & Efisiensi

| | Nilai |
|---|---|
| AUC Diff (E1−E2 clean) | +0.00304 |
| Z-statistic | 7.688 |
| **p-value** | **< 0.0001** (signifikan) |
| Overhead parameter DCT | +384 param (0.01%) |
| Overhead waktu inferensi | ≈ −2.5ms (E-1 lebih cepat) |

---
