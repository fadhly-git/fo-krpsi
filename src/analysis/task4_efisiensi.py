"""Task 4: Efisiensi Komputasi — parameter count, ukuran checkpoint, waktu inferensi.

Output:
  results/bab4_tambahan/tabel_4_13_efisiensi.csv
  results/bab4_tambahan/gambar_4_6_efisiensi.png  (opsional)

Jalankan dari root project:
  FORCE_CPU=1 python src/analysis/task4_efisiensi.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan src/ ke sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CFG, resolve_device, apply_cpu_safety_overrides
from analysis.common import load_model, get_val_subset, get_dct_dim, CKPT_E1, CKPT_E2, CKPT_E3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "bab4_tambahan"

EVAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

BATCH_SIZE = 32
N_REPEATS = 20
N_WARMUP = 5


def count_parameters(backbone, head, fusion=None):
    """Hitung total parameter dan parameter head/fusion saja."""
    total = sum(p.numel() for p in backbone.parameters()) + \
            sum(p.numel() for p in head.parameters())
    head_only = sum(p.numel() for p in head.parameters())
    if fusion is not None:
        total += sum(p.numel() for p in fusion.parameters())
        head_only += sum(p.numel() for p in fusion.parameters())
    return total, head_only


def build_batch(val_samples, batch_size, dct_dim, device):
    """Bangun satu batch tensor (img_tensor, dct_tensor) dari val_samples[:batch_size]."""
    imgs = []
    dcts = []
    samples = val_samples[:batch_size]
    for img_path, _dct_path, _label in samples:
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_arr = np.array(img_pil, dtype=np.uint8)
        aug = EVAL_TRANSFORM(image=img_arr)
        imgs.append(aug["image"])

        if dct_dim > 0 and _dct_path is not None:
            dct = np.load(str(_dct_path)).astype(np.float32)
            dct_mean = dct.mean()
            dct_std = max(dct.std(), 1e-6)
            dct = (dct - dct_mean) / dct_std
            dct = np.clip(np.nan_to_num(dct, nan=0.0, posinf=1e6, neginf=-1e6), -1e4, 1e4)
        else:
            dct = np.zeros(dct_dim, dtype=np.float32)
        dcts.append(torch.tensor(dct, dtype=torch.float32))

    img_batch = torch.stack(imgs).to(device)
    dct_batch = torch.stack(dcts).to(device)
    return img_batch, dct_batch


@torch.no_grad()
def measure_inference_time(backbone, head, fusion, img_batch, dct_batch, dct_dim,
                            n_repeats=N_REPEATS, n_warmup=N_WARMUP):
    """Ukur rata-rata waktu forward pass 1 batch dalam ms.

    Warmup n_warmup kali pertama dibuang.
    """
    times_ms = []
    for i in range(n_warmup + n_repeats):
        t0 = time.perf_counter()
        feat = backbone(img_batch)
        if fusion is not None:
            combined = fusion(feat, dct_batch)
            _ = head(combined)
        elif dct_dim > 0:
            combined = torch.cat([feat, dct_batch], dim=1)
            _ = head(combined)
        else:
            _ = head(feat)
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def plot_inference_time(results, out_path):
    """Bar chart waktu inferensi E-1 vs E-2."""
    models = [r["model"] for r in results]
    means = [r["inference_time_ms_mean"] for r in results]
    stds = [r["inference_time_ms_std"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    colors = ["black", "white", "gray"]
    hatches = ["", "///", "..."]
    edges = ["black", "black", "black"]

    for i, (m, mean, std) in enumerate(zip(models, means, stds)):
        ax.bar(x[i], mean, yerr=std, capsize=4,
               color=colors[i], hatch=hatches[i],
               edgecolor=edges[i], width=0.4,
               error_kw={"elinewidth": 1.0, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontfamily="serif", fontsize=11)
    ax.set_ylabel("Waktu Inferensi (ms/batch)", fontfamily="serif", fontsize=10)
    ax.set_title(
        f"Perbandingan Waktu Inferensi\n(batch_size={BATCH_SIZE}, CPU, {N_REPEATS} pengulangan)",
        fontfamily="serif", fontsize=10,
    )
    for label in ax.get_yticklabels():
        label.set_fontfamily("serif")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[Task4] Gambar efisiensi disimpan: {out_path}")


def run():
    """Jalankan Task 4."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device, _ = resolve_device()
    apply_cpu_safety_overrides(device)

    dct_dim = get_dct_dim()
    print(f"[Task4] DCT dim = {dct_dim}, device = {device}")

    # Load models
    print("[Task4] Loading E-1...")
    backbone_e1, head_e1, _ = load_model(CKPT_E1, dct_dim, device)
    print("[Task4] Loading E-2...")
    backbone_e2, head_e2, _ = load_model(CKPT_E2, 0, device)
    print("[Task4] Loading E-3...")
    backbone_e3, head_e3, fusion_e3 = load_model(CKPT_E3, dct_dim, device, cross_attn=True)

    # Hitung parameter
    total_e1, head_e1_params = count_parameters(backbone_e1, head_e1)
    total_e2, head_e2_params = count_parameters(backbone_e2, head_e2)
    total_e3, head_e3_params = count_parameters(backbone_e3, head_e3, fusion_e3)

    # Ukuran checkpoint
    size_e1_mb = os.path.getsize(str(CKPT_E1)) / (1024 * 1024)
    size_e2_mb = os.path.getsize(str(CKPT_E2)) / (1024 * 1024)
    size_e3_mb = os.path.getsize(str(CKPT_E3)) / (1024 * 1024)

    # Bangun batch dari val subset
    print("[Task4] Membangun batch untuk timing...")
    val_samples = get_val_subset()
    img_batch_e1, dct_batch_e1 = build_batch(val_samples, BATCH_SIZE, dct_dim, device)
    img_batch_e2, dct_batch_e2 = build_batch(val_samples, BATCH_SIZE, 0, device)
    img_batch_e3, dct_batch_e3 = build_batch(val_samples, BATCH_SIZE, dct_dim, device)

    # Ukur waktu inferensi
    print(f"[Task4] Mengukur waktu inferensi E-1 ({N_WARMUP} warmup + {N_REPEATS} repeats)...")
    mean_e1, std_e1 = measure_inference_time(
        backbone_e1, head_e1, None, img_batch_e1, dct_batch_e1, dct_dim
    )
    print(f"[Task4] E-1: {mean_e1:.2f} ± {std_e1:.2f} ms")

    print(f"[Task4] Mengukur waktu inferensi E-2 ({N_WARMUP} warmup + {N_REPEATS} repeats)...")
    mean_e2, std_e2 = measure_inference_time(
        backbone_e2, head_e2, None, img_batch_e2, dct_batch_e2, 0
    )
    print(f"[Task4] E-2: {mean_e2:.2f} ± {std_e2:.2f} ms")

    print(f"[Task4] Mengukur waktu inferensi E-3 ({N_WARMUP} warmup + {N_REPEATS} repeats)...")
    mean_e3, std_e3 = measure_inference_time(
        backbone_e3, head_e3, fusion_e3, img_batch_e3, dct_batch_e3, dct_dim
    )
    print(f"[Task4] E-3: {mean_e3:.2f} ± {std_e3:.2f} ms")

    results = [
        {
            "model": "E-1",
            "total_parameters": total_e1,
            "head_parameters": head_e1_params,
            "checkpoint_size_mb": size_e1_mb,
            "inference_time_ms_mean": mean_e1,
            "inference_time_ms_std": std_e1,
        },
        {
            "model": "E-2",
            "total_parameters": total_e2,
            "head_parameters": head_e2_params,
            "checkpoint_size_mb": size_e2_mb,
            "inference_time_ms_mean": mean_e2,
            "inference_time_ms_std": std_e2,
        },
        {
            "model": "E-3",
            "total_parameters": total_e3,
            "head_parameters": head_e3_params,
            "checkpoint_size_mb": size_e3_mb,
            "inference_time_ms_mean": mean_e3,
            "inference_time_ms_std": std_e3,
        },
    ]

    # Simpan CSV
    csv_path = OUT_DIR / "tabel_4_13_efisiensi.csv"
    header = "model,total_parameters,head_parameters,checkpoint_size_mb,inference_time_ms_mean,inference_time_ms_std"
    csv_lines = [header]
    for r in results:
        csv_lines.append(
            f"{r['model']},{r['total_parameters']},{r['head_parameters']},"
            f"{r['checkpoint_size_mb']:.4f},{r['inference_time_ms_mean']:.4f},"
            f"{r['inference_time_ms_std']:.4f}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"[Task4] CSV disimpan: {csv_path}")

    # Print ke terminal
    print("\n" + "=" * 80)
    print("TABEL 4.13 — EFISIENSI KOMPUTASI")
    print("=" * 80)
    print(
        f"{'Model':<6} {'Total Params':>14} {'Head Params':>12} "
        f"{'Ckpt Size(MB)':>14} {'Time Mean(ms)':>14} {'Time Std(ms)':>13}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']:<6} {r['total_parameters']:>14,} {r['head_parameters']:>12,} "
            f"{r['checkpoint_size_mb']:>14.4f} {r['inference_time_ms_mean']:>14.4f} "
            f"{r['inference_time_ms_std']:>13.4f}"
        )
    print("=" * 80)

    # Plot opsional
    img_path = OUT_DIR / "gambar_4_6_efisiensi.png"
    plot_inference_time(results, img_path)

    return results


if __name__ == "__main__":
    os.environ.setdefault("FORCE_CPU", "1")
    run()
