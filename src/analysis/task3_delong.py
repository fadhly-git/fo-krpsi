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
def run(probs_e1=None, probs_e2=None, probs_e3=None, labels=None):
    """Jalankan Task 3.

    Jika probs tidak di-pass, akan di-compute ulang via Task 1.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if probs_e1 is None or probs_e2 is None or probs_e3 is None or labels is None:
        print("[Task3] Probs clean tidak tersedia, compute ulang via Task 1...")
        from analysis.task1_robustness import run as run_task1
        probs_e1, probs_e2, probs_e3, labels, _ = run_task1(return_clean_probs=True)

    labels = labels.astype(np.float64)
    p_e1 = probs_e1.astype(np.float64)
    p_e2 = probs_e2.astype(np.float64)
    p_e3 = probs_e3.astype(np.float64)

    res_1_vs_2 = delong_roc_test(labels, p_e1, p_e2)
    res_3_vs_1 = delong_roc_test(labels, p_e3, p_e1)
    res_3_vs_2 = delong_roc_test(labels, p_e3, p_e2)

    # Simpan CSV
    csv_path = OUT_DIR / "tabel_4_12_delong_test.csv"
    header = "comparison,auc_model_a,auc_model_b,auc_diff,z_statistic,p_value"
    
    lines = [header]
    for comp, res in [("E-1 vs E-2", res_1_vs_2), ("E-3 vs E-1", res_3_vs_1), ("E-3 vs E-2", res_3_vs_2)]:
        lines.append(
            f"{comp},{res['auc_e1']:.6f},{res['auc_e2']:.6f},"
            f"{res['auc_diff']:.6f},{res['z_statistic']:.6f},"
            f"{res['p_value']:.8f}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Task3] CSV disimpan: {csv_path}")

    # Print ke terminal
    print("\n" + "=" * 80)
    print("TABEL 4.12 — DELONG'S TEST")
    print("=" * 80)
    print(f"{'Perbandingan':<15} {'AUC A':<10} {'AUC B':<10} {'Diff(A-B)':<10} {'Z-stat':<10} {'P-value':<12} {'Kesimpulan'}")
    print("-" * 80)
    
    for comp, res in [("E-1 vs E-2", res_1_vs_2), ("E-3 vs E-1", res_3_vs_1), ("E-3 vs E-2", res_3_vs_2)]:
        sig = "SIGNIFIKAN" if res["p_value"] < 0.05 else "TIDAK SIGNIFIKAN"
        print(
            f"{comp:<15} {res['auc_e1']:<10.6f} {res['auc_e2']:<10.6f} "
            f"{res['auc_diff']:<10.6f} {res['z_statistic']:<10.6f} "
            f"{res['p_value']:<12.8f} {sig}"
        )
    print("=" * 80)

    # For compatibility with existing verifications, return the dict of E-1 vs E-2 as default, or return a combined dict
    # Let's return the E-1 vs E-2 for backward compatibility, and attach E-3 metrics to it
    res_1_vs_2["auc_e3"] = res_3_vs_1["auc_e1"]  # AUC model A in E-3 vs E-1 is E-3
    return res_1_vs_2


if __name__ == "__main__":
    os.environ.setdefault("FORCE_CPU", "1")
    run()
