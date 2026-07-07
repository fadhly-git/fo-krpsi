"""Task 3: Uji Signifikansi Statistik (DeLong's Test).

Output:
  results/bab4_tambahan/tabel_4_12_delong_test.csv

Implementasi fast DeLong (Sun & Xu, 2014) — tanpa library eksternal tambahan.

Jalankan dari root project:
  FORCE_CPU=1 python src/analysis/task3_delong.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Tambahkan src/ ke sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "bab4_tambahan"


# ─── Implementasi fast DeLong ──────────────────────────────────────────────────
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
        T[i:j + 1] = 0.5 * (i + j) + 1
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
    """DeLong's test untuk dua ROC AUC berkorelasi (sampel sama).

    Args:
        y_true: array 0/1 label asli.
        prob_e1: probabilitas P(FAKE) dari E-1.
        prob_e2: probabilitas P(FAKE) dari E-2.

    Returns:
        dict dengan kunci: auc_e1, auc_e2, auc_diff, z_statistic, p_value.
    """
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    preds = np.vstack([prob_e1[order], prob_e2[order]])
    aucs, delongcov = fastDeLong(preds, label_1_count)

    auc_diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = auc_diff / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_e1": float(aucs[0]),
        "auc_e2": float(aucs[1]),
        "auc_diff": float(auc_diff),
        "z_statistic": float(z),
        "p_value": float(p_value),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────
def run(probs_e1=None, probs_e2=None, labels=None):
    """Jalankan Task 3.

    Jika probs tidak di-pass, akan di-compute ulang via Task 1.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if probs_e1 is None or probs_e2 is None or labels is None:
        print("[Task3] Probs clean tidak tersedia, compute ulang via Task 1...")
        from analysis.task1_robustness import run as run_task1
        probs_e1, probs_e2, labels, _ = run_task1(return_clean_probs=True)

    result = delong_roc_test(
        y_true=labels.astype(np.float64),
        prob_e1=probs_e1.astype(np.float64),
        prob_e2=probs_e2.astype(np.float64),
    )

    # Simpan CSV
    csv_path = OUT_DIR / "tabel_4_12_delong_test.csv"
    header = "auc_e1,auc_e2,auc_diff,z_statistic,p_value"
    csv_line = (
        f"{result['auc_e1']:.6f},{result['auc_e2']:.6f},"
        f"{result['auc_diff']:.6f},{result['z_statistic']:.6f},"
        f"{result['p_value']:.8f}"
    )
    csv_path.write_text(header + "\n" + csv_line + "\n", encoding="utf-8")
    print(f"[Task3] CSV disimpan: {csv_path}")

    # Print ke terminal
    print("\n" + "=" * 60)
    print("TABEL 4.12 — DELONG'S TEST")
    print("=" * 60)
    print(f"  AUC E-1        : {result['auc_e1']:.6f}")
    print(f"  AUC E-2        : {result['auc_e2']:.6f}")
    print(f"  AUC Diff (E1−E2): {result['auc_diff']:.6f}")
    print(f"  Z-statistic    : {result['z_statistic']:.6f}")
    print(f"  P-value        : {result['p_value']:.8f}")
    sig = "SIGNIFIKAN (p < 0.05)" if result["p_value"] < 0.05 else "Tidak signifikan (p ≥ 0.05)"
    print(f"  Kesimpulan     : {sig}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    os.environ.setdefault("FORCE_CPU", "1")
    run()
