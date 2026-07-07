"""Runner utama: jalankan semua task analisis BAB IV secara berurutan.

Keuntungan: probs clean dari Task 1 di-reuse oleh Task 2 dan Task 3
tanpa komputasi ulang (satu kali forward pass).

Jalankan dari root project:
  FORCE_CPU=1 python src/analysis/run_all.py
"""

import os
import sys
from pathlib import Path

# Tambahkan src/ ke sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

os.environ.setdefault("FORCE_CPU", "1")

from config import resolve_device
from analysis.common import CKPT_E1, CKPT_E2
from analysis.task1_robustness import run as run_task1
from analysis.task2_roc import run as run_task2
from analysis.task3_delong import run as run_task3
from analysis.task4_efisiensi import run as run_task4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "bab4_tambahan"


def main():
    print("\n" + "=" * 80)
    print("ANALISIS TAMBAHAN BAB IV — RUN ALL")
    print("=" * 80 + "\n")

    # Catat ukuran checkpoint sebelum apapun
    ckpt_e1_size_before = os.path.getsize(str(CKPT_E1))
    ckpt_e2_size_before = os.path.getsize(str(CKPT_E2))

    # ── TASK 1: Robustness ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 1: Pengujian Robustness")
    print("─" * 60)
    probs_e1_clean, probs_e2_clean, labels_clean, rows_task1 = run_task1(
        return_clean_probs=True
    )

    # ── TASK 2: ROC Curve ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 2: Kurva ROC")
    print("─" * 60)
    auc_e1_roc, auc_e2_roc = run_task2(
        probs_e1=probs_e1_clean,
        probs_e2=probs_e2_clean,
        labels=labels_clean,
    )

    # ── TASK 3: DeLong's Test ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 3: Uji Signifikansi Statistik (DeLong's Test)")
    print("─" * 60)
    delong_result = run_task3(
        probs_e1=probs_e1_clean,
        probs_e2=probs_e2_clean,
        labels=labels_clean,
    )

    # ── TASK 4: Efisiensi Komputasi ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 4: Efisiensi Komputasi")
    print("─" * 60)
    efisiensi_results = run_task4()

    # ── VERIFIKASI AKHIR ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERIFIKASI AKHIR")
    print("=" * 80)

    errors = []

    # 1. AUC clean Task1 == AUC dari Task3
    auc_clean_e1_task1 = next(
        r["auc"] for r in rows_task1 if r["model"] == "E-1" and r["kondisi"] == "Clean"
    )
    auc_clean_e2_task1 = next(
        r["auc"] for r in rows_task1 if r["model"] == "E-2" and r["kondisi"] == "Clean"
    )
    diff_e1 = abs(auc_clean_e1_task1 - delong_result["auc_e1"])
    diff_e2 = abs(auc_clean_e2_task1 - delong_result["auc_e2"])

    if diff_e1 < 1e-9:
        print(f"[V1] ✓ AUC clean E-1 konsisten: Task1={auc_clean_e1_task1:.6f} == Task3={delong_result['auc_e1']:.6f}")
    else:
        msg = f"[V1] ✗ AUC clean E-1 TIDAK KONSISTEN: Task1={auc_clean_e1_task1:.6f} vs Task3={delong_result['auc_e1']:.6f} (diff={diff_e1:.2e})"
        print(msg)
        errors.append(msg)

    if diff_e2 < 1e-9:
        print(f"[V1] ✓ AUC clean E-2 konsisten: Task1={auc_clean_e2_task1:.6f} == Task3={delong_result['auc_e2']:.6f}")
    else:
        msg = f"[V1] ✗ AUC clean E-2 TIDAK KONSISTEN: Task1={auc_clean_e2_task1:.6f} vs Task3={delong_result['auc_e2']:.6f} (diff={diff_e2:.2e})"
        print(msg)
        errors.append(msg)

    # 2. Jumlah baris CSV Task1 == 24
    csv_path = OUT_DIR / "tabel_4_11_robustness.csv"
    n_rows = len(csv_path.read_text(encoding="utf-8").strip().split("\n")) - 1  # minus header
    if n_rows == 24:
        print(f"[V2] ✓ Jumlah baris CSV Task1 = {n_rows} (expected 24)")
    else:
        msg = f"[V2] ✗ Jumlah baris CSV Task1 = {n_rows} (expected 24)"
        print(msg)
        errors.append(msg)

    # 3. delta_auc kondisi Clean == 0.0 untuk kedua model
    delta_clean_e1 = next(
        r["delta_auc"] for r in rows_task1 if r["model"] == "E-1" and r["kondisi"] == "Clean"
    )
    delta_clean_e2 = next(
        r["delta_auc"] for r in rows_task1 if r["model"] == "E-2" and r["kondisi"] == "Clean"
    )
    if abs(delta_clean_e1) < 1e-9 and abs(delta_clean_e2) < 1e-9:
        print(f"[V3] ✓ Delta AUC kondisi Clean = 0.0 untuk E-1 dan E-2")
    else:
        msg = f"[V3] ✗ Delta AUC Clean bukan nol: E-1={delta_clean_e1}, E-2={delta_clean_e2}"
        print(msg)
        errors.append(msg)

    # 4. Checkpoint tidak berubah
    ckpt_e1_size_after = os.path.getsize(str(CKPT_E1))
    ckpt_e2_size_after = os.path.getsize(str(CKPT_E2))
    if ckpt_e1_size_before == ckpt_e1_size_after:
        print(f"[V4] ✓ Checkpoint E-1 tidak berubah ({ckpt_e1_size_after:,} bytes)")
    else:
        msg = f"[V4] ✗ CHECKPOINT E-1 BERUBAH! before={ckpt_e1_size_before}, after={ckpt_e1_size_after}"
        print(msg)
        errors.append(msg)
    if ckpt_e2_size_before == ckpt_e2_size_after:
        print(f"[V4] ✓ Checkpoint E-2 tidak berubah ({ckpt_e2_size_after:,} bytes)")
    else:
        msg = f"[V4] ✗ CHECKPOINT E-2 BERUBAH! before={ckpt_e2_size_before}, after={ckpt_e2_size_after}"
        print(msg)
        errors.append(msg)

    # Ringkasan output file
    print("\n" + "=" * 80)
    print("RINGKASAN FILE OUTPUT")
    print("=" * 80)
    output_files = [
        OUT_DIR / "tabel_4_11_robustness.csv",
        OUT_DIR / "gambar_4_4_robustness_delta_auc.png",
        OUT_DIR / "gambar_4_5_roc_curve.png",
        OUT_DIR / "tabel_4_12_delong_test.csv",
        OUT_DIR / "tabel_4_13_efisiensi.csv",
        OUT_DIR / "gambar_4_6_efisiensi.png",
    ]
    for f in output_files:
        status = "✓" if f.exists() else "✗ TIDAK ADA"
        size_str = f"({f.stat().st_size:,} bytes)" if f.exists() else ""
        print(f"  {status}  {f.name}  {size_str}")

    if errors:
        print(f"\n[VERIFIKASI] {len(errors)} error ditemukan:")
        for e in errors:
            print(f"  {e}")
        return 1
    else:
        print("\n[VERIFIKASI] Semua cek lolos ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
