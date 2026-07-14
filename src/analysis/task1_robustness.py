"""Task 1: Pengujian Robustness — evaluasi E-1 dan E-2 pada kondisi degradasi.

Output:
  results/bab4_tambahan/tabel_4_11_robustness.csv  (24 baris)
  results/bab4_tambahan/gambar_4_4_robustness_delta_auc.png

Jalankan dari root project:
  FORCE_CPU=1 python src/analysis/task1_robustness.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.fft import dctn
from scipy.stats import skew
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan src/ ke sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import resolve_device, apply_cpu_safety_overrides
from analysis.common import load_model, get_val_subset, get_dct_dim, CKPT_E1, CKPT_E2

# ─── Konstanta ─────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "bab4_tambahan"

EVAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Batas maksimum sampel untuk robustness eval.
# Cukup representatif untuk AUC yang stabil, drastis memangkas waktu CPU.
# Set ke None untuk evaluasi penuh (bisa memakan 10+ jam di CPU).
MAX_EVAL_SAMPLES = 2000
CONDITIONS = [
    ("Clean",      None,   "Clean"),
    ("JPEG",       30,     "JPEG-30"),
    ("JPEG",       50,     "JPEG-50"),
    ("JPEG",       70,     "JPEG-70"),
    ("Noise",      10,     "Noise-σ10"),
    ("Noise",      25,     "Noise-σ25"),
    ("Noise",      50,     "Noise-σ50"),
    ("Blur",       3,      "Blur-3×3"),
    ("Blur",       5,      "Blur-5×5"),
    ("Blur",       7,      "Blur-7×7"),
    ("Downscale",  0.5,    "DS-0.5"),
    ("Downscale",  0.25,   "DS-0.25"),
    # Occlusion: kotak hitam di tengah, sisi = ratio × dimensi gambar
    ("Occlusion",  0.30,   "Occ-30%"),
    ("Occlusion",  0.50,   "Occ-50%"),
    ("Occlusion",  0.70,   "Occ-70%"),
]

# ─── Fungsi Degradasi (OpenCV) ─────────────────────────────────────────────────
def apply_jpeg(img_rgb_uint8, quality):
    ok, enc = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
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
    small = cv2.resize(
        img_rgb_uint8,
        (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_occlusion(img_rgb_uint8, ratio):
    """Kotak hitam di tengah gambar (deterministic).

    Args:
        ratio: Fraksi sisi gambar yang ditutupi (0.30 = 30% sisi, 9% area;
               0.50 = 50% sisi, 25% area; 0.70 = 70% sisi, 49% area).
    """
    h, w = img_rgb_uint8.shape[:2]
    size_h = int(h * ratio)
    size_w = int(w * ratio)
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - size_h // 2)
    x1 = max(0, cx - size_w // 2)
    y2 = min(h, y1 + size_h)
    x2 = min(w, x1 + size_w)
    out = img_rgb_uint8.copy()
    out[y1:y2, x1:x2] = 0
    return out


def degrade(img_rgb_uint8, cond_type, param):
    """Terapkan degradasi sesuai tipe dan parameter."""
    if cond_type == "Clean" or cond_type is None:
        return img_rgb_uint8
    elif cond_type == "JPEG":
        return apply_jpeg(img_rgb_uint8, param)
    elif cond_type == "Noise":
        return apply_gaussian_noise(img_rgb_uint8, param)
    elif cond_type == "Blur":
        return apply_gaussian_blur(img_rgb_uint8, param)
    elif cond_type == "Downscale":
        return apply_downscale_upscale(img_rgb_uint8, param)
    elif cond_type == "Occlusion":
        return apply_occlusion(img_rgb_uint8, param)
    else:
        raise ValueError(f"Kondisi tidak dikenal: {cond_type}")


# ─── Fungsi DCT (disalin persis dari precompute_dct.py) ───────────────────────
def block_view_8x8(y_channel):
    """Create non-overlapping 8x8 blocks from luminance channel."""
    height, width = y_channel.shape
    if height < 8 or width < 8:
        pad_h = max(8 - height, 0)
        pad_w = max(8 - width, 0)
        y_channel = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode="edge")
        height, width = y_channel.shape
    height8 = (height // 8) * 8
    width8 = (width // 8) * 8
    y_channel = y_channel[:height8, :width8]
    blocks = y_channel.reshape(height8 // 8, 8, width8 // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
    return blocks


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


def normalize_dct(dct: np.ndarray) -> np.ndarray:
    """Normalisasi DCT per-sampel, identik dengan dataset.py."""
    dct_mean = dct.mean()
    dct_std = max(dct.std(), 1e-6)
    dct = (dct - dct_mean) / dct_std
    dct = np.clip(np.nan_to_num(dct, nan=0.0, posinf=1e6, neginf=-1e6), -1e4, 1e4)
    return dct


# ─── Inference satu kondisi ────────────────────────────────────────────────────
@torch.no_grad()
def run_inference_one_condition(
    val_samples,
    backbone_e1, head_e1,
    backbone_e2, head_e2,
    device,
    cond_type, param,
):
    """Forward pass semua sampel untuk satu kondisi degradasi.

    Returns:
        (probs_e1, probs_e2, labels) — numpy arrays, probs = P(FAKE).
    """
    probs_e1 = []
    probs_e2 = []
    labels = []

    for img_path, _dct_path, label in val_samples:
        # 1. Load gambar asli
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_arr = np.array(img_pil, dtype=np.uint8)

        # 2. Terapkan degradasi
        img_deg = degrade(img_arr, cond_type, param)

        # 3. Hitung DCT dari gambar HASIL DEGRADASI (E-1)
        dct_feat = compute_dct_feature_192_from_array(img_deg)
        dct_feat = normalize_dct(dct_feat)
        dct_tensor = torch.tensor(dct_feat, dtype=torch.float32).unsqueeze(0).to(device)

        # 4. Terapkan eval_transform → tensor citra
        aug = EVAL_TRANSFORM(image=img_deg)
        img_tensor = aug["image"].unsqueeze(0).to(device)

        # 5. Forward E-1 (backbone + dct)
        feat_e1 = backbone_e1(img_tensor)
        combined_e1 = torch.cat([feat_e1, dct_tensor], dim=1)
        logit_e1 = head_e1(combined_e1)
        prob_e1 = F.softmax(logit_e1, dim=1)[0, 1].item()  # P(FAKE)

        # 6. Forward E-2 (backbone only)
        feat_e2 = backbone_e2(img_tensor)
        logit_e2 = head_e2(feat_e2)
        prob_e2 = F.softmax(logit_e2, dim=1)[0, 1].item()  # P(FAKE)

        probs_e1.append(prob_e1)
        probs_e2.append(prob_e2)
        labels.append(label)

    return np.array(probs_e1), np.array(probs_e2), np.array(labels)


# ─── Plot bar chart ────────────────────────────────────────────────────────────
def plot_delta_auc(rows, out_path):
    """Grouped bar chart: delta AUC per kondisi degradasi."""
    # Filter hanya kondisi non-Clean
    non_clean = [r for r in rows if r["kondisi"] != "Clean"]
    labels_x = [r["label"] for r in non_clean if r["model"] == "E-1"]
    delta_e1 = [r["delta_auc"] for r in non_clean if r["model"] == "E-1"]
    delta_e2 = [r["delta_auc"] for r in non_clean if r["model"] == "E-2"]

    x = np.arange(len(labels_x))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_e1 = ax.bar(
        x - width / 2, delta_e1, width,
        label="E-1 (Hibrida)",
        color="black",
        hatch="",
    )
    bars_e2 = ax.bar(
        x + width / 2, delta_e2, width,
        label="E-2 (Baseline)",
        color="white",
        edgecolor="black",
        hatch="///",
    )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=45, ha="right",
                       fontfamily="serif", fontsize=9)
    ax.set_ylabel("Delta AUC (AUC Clean − AUC Kondisi)", fontfamily="serif", fontsize=10)
    ax.set_xlabel("Kondisi Degradasi", fontfamily="serif", fontsize=10)
    ax.set_title(
        "Penurunan AUC (Delta AUC) terhadap Kondisi Degradasi: E-1 vs E-2",
        fontfamily="serif", fontsize=11,
    )
    ax.legend(prop={"family": "serif", "size": 9})
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_yticklabels():
        label.set_fontfamily("serif")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[Task1] Gambar disimpan: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def run(return_clean_probs=False):
    """Jalankan Task 1. Jika return_clean_probs=True, return (probs_e1_clean, probs_e2_clean, labels)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device, _ = resolve_device()
    apply_cpu_safety_overrides(device)

    dct_dim = get_dct_dim()
    print(f"[Task1] DCT dim = {dct_dim}, device = {device}")

    # Load checkpoint ukuran (untuk verifikasi tidak berubah)
    ckpt_e1_size_before = os.path.getsize(str(CKPT_E1))
    ckpt_e2_size_before = os.path.getsize(str(CKPT_E2))

    # Load kedua model
    print("[Task1] Loading E-1 (best_efficient_dct.pth)...")
    backbone_e1, head_e1 = load_model(CKPT_E1, dct_dim, device)
    print("[Task1] Loading E-2 (best_efficient_no_dct.pth)...")
    backbone_e2, head_e2 = load_model(CKPT_E2, 0, device)

    # Dapatkan val subset
    print("[Task1] Membangun val subset...")
    val_samples = get_val_subset()
    print(f"[Task1] Val subset (full): {len(val_samples)} sampel")

    # Batasi jumlah sampel jika MAX_EVAL_SAMPLES diset
    if MAX_EVAL_SAMPLES is not None and len(val_samples) > MAX_EVAL_SAMPLES:
        # Stratified sampling: pertahankan rasio REAL:FAKE
        real_samples = [s for s in val_samples if s[2] == 0]
        fake_samples = [s for s in val_samples if s[2] == 1]
        n_total = len(val_samples)
        n_real_target = int(MAX_EVAL_SAMPLES * len(real_samples) / n_total)
        n_fake_target = MAX_EVAL_SAMPLES - n_real_target
        rng = np.random.default_rng(42)
        real_sel = [real_samples[i] for i in rng.choice(len(real_samples), n_real_target, replace=False)]
        fake_sel = [fake_samples[i] for i in rng.choice(len(fake_samples), n_fake_target, replace=False)]
        val_samples = real_sel + fake_sel
        rng.shuffle(val_samples)  # type: ignore[arg-type]
        val_samples = list(val_samples)
        print(
            f"[Task1] Sampel dibatasi ke {len(val_samples)} "
            f"(REAL={n_real_target}, FAKE={n_fake_target}) "
            f"— ubah MAX_EVAL_SAMPLES=None untuk evaluasi penuh"
        )
    else:
        print(f"[Task1] Menggunakan seluruh val subset: {len(val_samples)} sampel")

    # Seed untuk reproducibility noise Gaussian
    np.random.seed(42)

    rows = []
    clean_probs_e1 = None
    clean_probs_e2 = None
    clean_labels = None

    for cond_type, param, label in CONDITIONS:
        print(f"[Task1] Kondisi: {label} ...", end=" ", flush=True)
        probs_e1, probs_e2, lbls = run_inference_one_condition(
            val_samples,
            backbone_e1, head_e1,
            backbone_e2, head_e2,
            device,
            cond_type, param,
        )

        preds_e1 = (probs_e1 >= 0.5).astype(int)
        preds_e2 = (probs_e2 >= 0.5).astype(int)
        acc_e1 = accuracy_score(lbls, preds_e1)
        acc_e2 = accuracy_score(lbls, preds_e2)
        auc_e1 = roc_auc_score(lbls, probs_e1)
        auc_e2 = roc_auc_score(lbls, probs_e2)
        print(f"AUC E-1={auc_e1:.5f}  AUC E-2={auc_e2:.5f}")

        if cond_type == "Clean":
            clean_probs_e1 = probs_e1
            clean_probs_e2 = probs_e2
            clean_labels = lbls
            auc_clean_e1 = auc_e1
            auc_clean_e2 = auc_e2

        rows.append({
            "kondisi": cond_type if cond_type else "Clean",
            "parameter": str(param) if param is not None else "-",
            "label": label,
            "model": "E-1",
            "accuracy": acc_e1,
            "auc": auc_e1,
            "delta_auc": None,  # dihitung setelah clean diketahui
        })
        rows.append({
            "kondisi": cond_type if cond_type else "Clean",
            "parameter": str(param) if param is not None else "-",
            "label": label,
            "model": "E-2",
            "accuracy": acc_e2,
            "auc": auc_e2,
            "delta_auc": None,
        })

    # Hitung delta_auc
    for r in rows:
        if r["model"] == "E-1":
            r["delta_auc"] = round(auc_clean_e1 - r["auc"], 6)
        else:
            r["delta_auc"] = round(auc_clean_e2 - r["auc"], 6)

    # Simpan CSV
    csv_path = OUT_DIR / "tabel_4_11_robustness.csv"
    header = "kondisi,parameter,model,accuracy,auc,delta_auc"
    csv_lines = [header]
    for r in rows:
        csv_lines.append(
            f"{r['kondisi']},{r['parameter']},{r['model']},"
            f"{r['accuracy']:.6f},{r['auc']:.6f},{r['delta_auc']:.6f}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"[Task1] CSV disimpan: {csv_path}")

    # Print tabel ke terminal
    print("\n" + "=" * 80)
    print("TABEL 4.11 — ROBUSTNESS")
    print("=" * 80)
    print(f"{'Kondisi':<12} {'Param':<8} {'Model':<6} {'Accuracy':>10} {'AUC':>10} {'Delta AUC':>12}")
    print("-" * 64)
    for r in rows:
        print(
            f"{r['kondisi']:<12} {r['parameter']:<8} {r['model']:<6} "
            f"{r['accuracy']:>10.6f} {r['auc']:>10.6f} {r['delta_auc']:>12.6f}"
        )
    print("=" * 80)

    # Plot
    img_path = OUT_DIR / "gambar_4_4_robustness_delta_auc.png"
    plot_delta_auc(rows, img_path)

    # Verifikasi checkpoint tidak berubah
    ckpt_e1_size_after = os.path.getsize(str(CKPT_E1))
    ckpt_e2_size_after = os.path.getsize(str(CKPT_E2))
    assert ckpt_e1_size_before == ckpt_e1_size_after, "CHECKPOINT E-1 BERUBAH!"
    assert ckpt_e2_size_before == ckpt_e2_size_after, "CHECKPOINT E-2 BERUBAH!"
    print("[Task1] Verifikasi checkpoint: OK (tidak berubah)")

    if return_clean_probs:
        return clean_probs_e1, clean_probs_e2, clean_labels, rows

    return rows


if __name__ == "__main__":
    os.environ.setdefault("FORCE_CPU", "1")
    run()
