"""Task 2: Kurva ROC — membandingkan E-1 dan E-2 pada kondisi Clean.

Output:
  results/bab4_tambahan/gambar_4_5_roc_curve.png

Jalankan dari root project:
  FORCE_CPU=1 python src/analysis/task2_roc.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Tambahkan src/ ke sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import resolve_device, apply_cpu_safety_overrides

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "bab4_tambahan"


def plot_roc(probs_e1, probs_e2, probs_e3, labels, out_path):
    """Plot ROC curve E-1, E-2, E-3 dan simpan PNG."""
    fpr_e1, tpr_e1, _ = roc_curve(labels, probs_e1)
    fpr_e2, tpr_e2, _ = roc_curve(labels, probs_e2)
    fpr_e3, tpr_e3, _ = roc_curve(labels, probs_e3)
    auc_e1 = roc_auc_score(labels, probs_e1)
    auc_e2 = roc_auc_score(labels, probs_e2)
    auc_e3 = roc_auc_score(labels, probs_e3)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        fpr_e1, tpr_e1,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"E-1 (Hibrida, AUC={auc_e1:.4f})",
    )
    ax.plot(
        fpr_e2, tpr_e2,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"E-2 (Baseline, AUC={auc_e2:.4f})",
    )
    ax.plot(
        fpr_e3, tpr_e3,
        color="gray",
        linestyle="-.",
        linewidth=1.5,
        label=f"E-3 (Cross-Attention, AUC={auc_e3:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label="Random Classifier",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontfamily="serif", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontfamily="serif", fontsize=11)
    ax.set_title(
        "Kurva ROC Model Hibrida (E-1), Baseline (E-2),\ndan Cross-Attention (E-3) pada Data Validasi Bersih",
        fontfamily="serif", fontsize=11,
    )
    ax.legend(prop={"family": "serif", "size": 10}, loc="lower right")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[Task2] ROC curve disimpan: {out_path}")
    return auc_e1, auc_e2, auc_e3


def run(probs_e1=None, probs_e2=None, probs_e3=None, labels=None):
    """Jalankan Task 2.

    Jika probs tidak di-pass, akan di-compute ulang via Task 1.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if probs_e1 is None or probs_e2 is None or probs_e3 is None or labels is None:
        print("[Task2] Probs clean tidak tersedia, compute ulang via Task 1...")
        from analysis.task1_robustness import run as run_task1
        probs_e1, probs_e2, probs_e3, labels, _ = run_task1(return_clean_probs=True)

    out_path = OUT_DIR / "gambar_4_5_roc_curve.png"
    auc_e1, auc_e2, auc_e3 = plot_roc(probs_e1, probs_e2, probs_e3, labels, out_path)

    print(f"[Task2] AUC E-1 (clean) = {auc_e1:.6f}")
    print(f"[Task2] AUC E-2 (clean) = {auc_e2:.6f}")
    print(f"[Task2] AUC E-3 (clean) = {auc_e3:.6f}")
    return auc_e1, auc_e2, auc_e3


if __name__ == "__main__":
    os.environ.setdefault("FORCE_CPU", "1")
    run()
