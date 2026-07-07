# TASK: Analisis Tambahan BAB IV (Robustness, ROC Curve, DeLong's Test, Efisiensi Komputasi)

## KONTEKS PROYEK

Proyek deteksi deepfake hibrida (EfficientNet-B0 + DCT 192-dim) vs baseline (EfficientNet-B0 saja).
Checkpoint terbaik SUDAH ADA dan TIDAK BOLEH ditimpa:
- `models/checkpoints/best_efficient_dct.pth` (E-1, model hibrida, epoch 8, Val AUC=0.99412)
- `models/checkpoints/best_efficient_no_dct.pth` (E-2, model baseline, epoch 8, Val AUC=0.99037)

TIDAK ADA training ulang. Semua task di bawah HANYA melakukan forward pass (inference) menggunakan kedua checkpoint di atas pada validation split yang SAMA dengan saat training.

## PRINSIP UMUM (WAJIB DIIKUTI)

1. Buat script BARU di `src/analysis/`, JANGAN modifikasi `train.py`, `model.py`, `dataset.py`, `config.py`, `validate.py` yang sudah ada. Boleh `import` dari modul tersebut.
2. Validation split HARUS identik dengan training: gunakan `train_test_split` dengan `random_state=CFG["seed"]` (42), `test_size` sesuai `CFG["val_ratio"]` (0.2), `stratify=labels_all`, persis seperti di `train.py`. Ini WAJIB agar sampel validasi yang dievaluasi sama dengan yang dipakai untuk menentukan checkpoint terbaik.
3. Gunakan transform evaluasi DETERMINISTIK (BUKAN `light_transform`/`medium_transform`/`heavy_transform` yang stokastik), didefinisikan sebagai:
   ```python
   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   eval_transform = A.Compose([
       A.Resize(224, 224),
       A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ToTensorV2(),
   ])
   ```
   Transform ini dipakai untuk SEMUA task (1-4) agar hasil antar-task konsisten dan dapat dibandingkan.
4. CATATAN PENTING untuk laporan: karena `eval_transform` di atas berbeda dari `light_transform` (yang stokastik) yang dipakai saat training, nilai "AUC bersih" (clean) pada Task 1 dan Task 2 KEMUNGKINAN sedikit berbeda dari Val AUC checkpoint (0.99412 / 0.99037). Ini WAJAR dan HARUS dilaporkan apa adanya di hasil, JANGAN disesuaikan/dipaksa sama.
5. Semua output gambar: grayscale/hitam-putih (`color="black"`, gunakan `linestyle` dan `marker` berbeda untuk membedakan E-1 vs E-2), font `Times New Roman` (fallback `serif`), simpan PNG `dpi=150`, background putih.
6. Semua output tabel: simpan sebagai CSV di `results/bab4_tambahan/` DAN tampilkan juga sebagai print table di terminal/log, agar mudah disalin ke dokumen skripsi.
7. Device tetap CPU (`FORCE_CPU=1`), `torch.no_grad()` untuk semua inference.
8. Buat folder output: `results/bab4_tambahan/` untuk semua file yang dihasilkan.

## HELPER BERSAMA (buat sekali, dipakai semua task)

Buat `src/analysis/common.py` berisi:

```python
def load_model(checkpoint_path, dct_dim, device):
    """Load backbone + head dari checkpoint, return (backbone, head) dalam eval mode."""
    # gunakan build_backbone() dan build_head() dari src/model.py
    # load state_dict dari checkpoint_path
    # backbone.eval(), head.eval()
    # return backbone, head

def get_val_subset():
    """Replikasi exact stratified split dari train.py, return list of (img_path, dct_path, label) untuk val_indices saja."""
    # gunakan MixedDataset(...) seperti di train.py untuk membangun full_dataset.samples
    # gunakan train_test_split dengan parameter identik train.py
    # return [full_dataset.samples[i] for i in val_indices]
```

---

## TASK 1: PENGUJIAN ROBUSTNESS (Tabel 3.8)

### Tujuan
Mengevaluasi E-1 dan E-2 pada subset validasi yang DIDEGRADASI sesuai Tabel 3.8 skripsi:
- Kompresi JPEG: quality = 30, 50, 70
- Gaussian Noise: sigma = 10, 25, 50
- Gaussian Blur: kernel = 3x3, 5x5, 7x7
- Downscale lalu upscale: faktor = 0.5, 0.25
- Kondisi "Clean" (tanpa degradasi) sebagai baseline pembanding

### Implementasi degradasi (pakai OpenCV, BUKAN Albumentations, untuk kepastian parameter)

```python
import cv2
import numpy as np

def apply_jpeg(img_rgb_uint8, quality):
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def apply_gaussian_noise(img_rgb_uint8, sigma):
    noise = np.random.normal(0, sigma, img_rgb_uint8.shape).astype(np.float32)
    noisy = img_rgb_uint8.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_gaussian_blur(img_rgb_uint8, ksize):
    return cv2.GaussianBlur(img_rgb_uint8, (ksize, ksize), 0)

def apply_downscale_upscale(img_rgb_uint8, factor):
    h, w = img_rgb_uint8.shape[:2]
    small = cv2.resize(img_rgb_uint8, (max(1,int(w*factor)), max(1,int(h*factor))),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
```

Gunakan `np.random.seed(42)` SEKALI di awal script (sebelum loop kondisi), agar noise Gaussian reproducible.

### CATATAN KRITIS: Fitur DCT untuk E-1 HARUS dihitung ulang dari gambar yang SUDAH didegradasi

JANGAN gunakan file `.npy` DCT yang sudah ada (itu dihitung dari gambar asli/bersih). Untuk E-1, setiap kondisi degradasi (termasuk "Clean") harus menghitung ulang fitur DCT 192-dim dari array gambar HASIL DEGRADASI menggunakan fungsi adaptasi dari `src/precompute_dct.py`:

```python
from scipy.fft import dctn
from scipy.stats import skew

def block_view_8x8(y_channel):
    # SALIN PERSIS dari src/precompute_dct.py

def compute_dct_feature_192_from_array(img_rgb_uint8):
    """Versi compute_dct_feature_192 yang menerima numpy array RGB, bukan path."""
    ycbcr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    blocks = block_view_8x8(y)
    dct_blocks = dctn(blocks, axes=(-2, -1), norm="ortho")
    coeffs = dct_blocks.reshape(dct_blocks.shape[0], 64)
    means = coeffs.mean(axis=0)
    variances = coeffs.var(axis=0)
    skews = skew(coeffs, axis=0, bias=False)
    features = np.concatenate([means, variances, np.nan_to_num(skews, nan=0.0)], axis=0)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
```

Setelah dihitung, normalisasi DCT per-sampel SAMA seperti `dataset.py`:
```python
dct_mean = dct.mean()
dct_std = max(dct.std(), 1e-6)
dct = (dct - dct_mean) / dct_std
dct = np.clip(np.nan_to_num(dct, nan=0.0, posinf=1e6, neginf=-1e6), -1e4, 1e4)
```

### Urutan pipeline per sampel per kondisi

1. Load image asli (`PIL.Image.open(img_path).convert("RGB")`) → numpy array uint8
2. Terapkan fungsi degradasi (atau lewati jika kondisi "Clean")
3. Untuk E-1: hitung fitur DCT 192-dim dari array hasil degradasi (langkah di atas), lalu normalisasi
4. Untuk kedua model: terapkan `eval_transform` (Resize 224x224 + Normalize ImageNet + ToTensorV2) pada array hasil degradasi → tensor citra
5. Forward pass: E-1 pakai `concat(backbone(img), dct)`, E-2 pakai `backbone(img)` saja (head 1280-dim)
6. Simpan probabilitas softmax kelas FAKE dan label asli

### Output yang diharapkan

**File CSV**: `results/bab4_tambahan/tabel_4_11_robustness.csv` dengan kolom:
`kondisi, parameter, model, accuracy, auc, delta_auc`

Di mana `delta_auc = auc_clean_model_tersebut - auc_kondisi_ini` (hitung AUC clean dulu untuk masing-masing model sebagai acuan delta).

Total baris = 2 model x (1 clean + 11 kondisi degradasi) = 24 baris. (11 = 3 JPEG + 3 noise + 3 blur + 2 downscale)

**Gambar**: `results/bab4_tambahan/gambar_4_4_robustness_delta_auc.png`
Bar chart berkelompok (grouped bar chart): sumbu-x = 11 kondisi degradasi (urut: JPEG30,50,70, Noise10,25,50, Blur3,5,7, Downscale0.5,0.25), sumbu-y = delta_auc. Dua grup bar per kondisi (E-1 hitam solid, E-2 hitam dengan pola garis/hatch berbeda untuk distingsi B&W). Beri garis horizontal y=0. Judul: "Penurunan AUC (Delta AUC) terhadap Kondisi Degradasi: E-1 vs E-2".

---

## TASK 2: KURVA ROC

### Tujuan
Membandingkan ROC curve E-1 dan E-2 pada kondisi "Clean" (tanpa degradasi), menggunakan `eval_transform` deterministik.

### Implementasi
- Gunakan probabilitas dan label dari kondisi "Clean" pada Task 1 (BISA reuse hasil Task 1, JANGAN compute ulang).
- Hitung ROC curve dengan `sklearn.metrics.roc_curve(y_true, y_prob)` untuk masing-masing model.
- Plot kedua kurva dalam satu figure: E-1 garis solid hitam, E-2 garis dashed hitam, garis diagonal referensi (random classifier) dotted abu-abu.
- Cantumkan AUC masing-masing model di legend, format: `E-1 (Hibrida, AUC=0.XXXX)` dan `E-2 (Baseline, AUC=0.XXXX)`.

### Output
`results/bab4_tambahan/gambar_4_5_roc_curve.png`
Label sumbu: "False Positive Rate" (x) dan "True Positive Rate" (y). Judul: "Kurva ROC Model Hibrida (E-1) dan Model Baseline (E-2) pada Data Validasi Bersih".

---

## TASK 3: UJI SIGNIFIKANSI STATISTIK (DeLong's Test)

### Tujuan
Menguji apakah selisih AUC antara E-1 dan E-2 (pada kondisi "Clean") signifikan secara statistik. Gunakan DeLong's test untuk dua ROC AUC yang berkorelasi (Sun & Xu, 2014), karena kedua model dievaluasi pada SAMPEL VALIDASI YANG SAMA.

### Implementasi (fast DeLong, gunakan implementasi berikut, JANGAN cari library eksternal baru)

```python
import numpy as np
from scipy import stats

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j] == Z[j + 1]:
            j += 1
        T[i:j+1] = 0.5 * (i + j) + 1
        i = j + 1
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(y_true, prob_e1, prob_e2):
    """y_true: 0/1 array. prob_e1, prob_e2: probabilitas kelas FAKE (label=1) dari E-1 dan E-2."""
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    preds = np.vstack([prob_e1[order], prob_e2[order]])
    aucs, delongcov = fastDeLong(preds, label_1_count)

    auc_diff = aucs[0] - aucs[1]
    var = delongcov[0,0] + delongcov[1,1] - 2*delongcov[0,1]
    z = auc_diff / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_e1": aucs[0], "auc_e2": aucs[1], "auc_diff": auc_diff,
            "z_statistic": z, "p_value": p_value}
```

### Output
`results/bab4_tambahan/tabel_4_12_delong_test.csv` dengan kolom:
`auc_e1, auc_e2, auc_diff, z_statistic, p_value`

Tidak perlu gambar untuk task ini.

---

## TASK 4: EFISIENSI KOMPUTASI

### Tujuan
Membandingkan E-1 vs E-2 dari segi jumlah parameter, ukuran checkpoint, dan waktu inferensi.

### Yang dihitung
1. **Jumlah parameter**: `sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())` untuk masing-masing model. Laporkan juga jumlah parameter HEAD saja (untuk highlight selisih akibat dimensi DCT: 1472 vs 1280 input).
2. **Ukuran file checkpoint**: `os.path.getsize()` dalam MB untuk `best_efficient_dct.pth` dan `best_efficient_no_dct.pth`.
3. **Waktu inferensi**: ukur rata-rata waktu forward pass untuk 1 batch (`batch_size=32`) dari val_subset, ULANGI 20 KALI (warmup 5 kali pertama dibuang), ambil mean dan std dalam milidetik. Gunakan `time.perf_counter()`.

### Output
`results/bab4_tambahan/tabel_4_13_efisiensi.csv` dengan kolom:
`model, total_parameters, head_parameters, checkpoint_size_mb, inference_time_ms_mean, inference_time_ms_std`

Tidak perlu gambar untuk task ini (kecuali ada waktu lebih, boleh tambahan bar chart waktu inferensi sebagai `gambar_4_6_efisiensi.png`, OPSIONAL, jangan dipaksakan jika menambah kompleksitas).

---

## VERIFIKASI (WAJIB dijalankan setelah semua task selesai)

1. Cek `auc_clean` Task 1 (kondisi Clean, masing-masing model) SAMA dengan `auc_e1`/`auc_e2` dari Task 3 (selisih harus 0, karena seharusnya reuse data yang sama).
2. Cek jumlah baris CSV Task 1 = 24 (2 model x 12 kondisi termasuk Clean).
3. Cek `delta_auc` untuk kondisi "Clean" = 0.0000 untuk kedua model (karena delta dihitung relatif terhadap clean).
4. Cek checkpoint `best_efficient_dct.pth` dan `best_efficient_no_dct.pth` TIDAK BERUBAH (bandingkan ukuran file/timestamp sebelum dan sesudah menjalankan semua task) untuk memastikan tidak ada proses training yang tidak sengaja menimpa checkpoint.
5. Print semua tabel CSV ke terminal dalam format yang mudah disalin (markdown table atau aligned text).

## RINGKASAN OUTPUT FILE

| File | Untuk Skripsi |
|---|---|
| `results/bab4_tambahan/tabel_4_11_robustness.csv` | Tabel 4.11 |
| `results/bab4_tambahan/gambar_4_4_robustness_delta_auc.png` | Gambar 4.4 |
| `results/bab4_tambahan/gambar_4_5_roc_curve.png` | Gambar 4.5 |
| `results/bab4_tambahan/tabel_4_12_delong_test.csv` | Tabel 4.12 |
| `results/bab4_tambahan/tabel_4_13_efisiensi.csv` | Tabel 4.13 |
| `results/bab4_tambahan/gambar_4_6_efisiensi.png` (opsional) | Gambar 4.6 (opsional) |

Setelah semua file dihasilkan, kirimkan isi seluruh CSV (sebagai teks/tabel) dan kelima/keenam file gambar PNG untuk dimasukkan ke BAB IV.
