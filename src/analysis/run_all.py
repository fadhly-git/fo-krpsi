"""Runner utama: jalankan semua task analisis BAB IV secara berurutan.

Keuntungan: probs clean dari Task 1 di-reuse oleh Task 2 dan Task 3
tanpa komputasi ulang (satu kali forward pass).

Jalankan dari root project:
  FORCE_CPU=1 python src/analysis/run_all.py               # default: best
  FORCE_CPU=1 python src/analysis/run_all.py --mode latest  # latest saja
  FORCE_CPU=1 python src/analysis/run_all.py --mode both    # best + latest
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Tambahkan src/ ke sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

os.environ.setdefault("FORCE_CPU", "1")

from config import resolve_device
import analysis.common as _common
from analysis.common import (
    CKPT_E1      as CKPT_E1_BEST,
    CKPT_E2      as CKPT_E2_BEST,
    CKPT_E3      as CKPT_E3_BEST,
    REAL_CKPT_DIR,
)
import analysis.task1_robustness as _task1_mod
import analysis.task2_roc        as _task2_mod
import analysis.task3_delong     as _task3_mod
import analysis.task4_efisiensi  as _task4_mod
from analysis.task1_robustness import run as run_task1
from analysis.task2_roc        import run as run_task2
from analysis.task3_delong     import run as run_task3
from analysis.task4_efisiensi  import run as run_task4

# Latest checkpoint paths
CKPT_E1_LATEST = REAL_CKPT_DIR / "last_checkpoint.pth"
CKPT_E2_LATEST = REAL_CKPT_DIR / "latest_no_dct.pth"
CKPT_E3_LATEST = REAL_CKPT_DIR / "last_checkpoint_crossattn.pth"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_OUT    = PROJECT_ROOT / "results" / "bab4_tambahan"


def _patch_ckpts(ckpt_e1: Path, ckpt_e2: Path, ckpt_e3: Path, out_dir: Path):
    """Patch checkpoint constants and OUT_DIR in all task modules in-place."""
    for mod in (_common, _task1_mod, _task2_mod, _task3_mod, _task4_mod):
        if hasattr(mod, "CKPT_E1"):  mod.CKPT_E1  = ckpt_e1
        if hasattr(mod, "CKPT_E2"):  mod.CKPT_E2  = ckpt_e2
        if hasattr(mod, "CKPT_E3"):  mod.CKPT_E3  = ckpt_e3
        if hasattr(mod, "OUT_DIR"):   mod.OUT_DIR   = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)


def _run_all_tasks(mode: str) -> int:
    """Jalankan Task 1-5 untuk satu mode checkpoint (best atau latest)."""
    if mode == "latest":
        ckpt_e1, ckpt_e2, ckpt_e3 = CKPT_E1_LATEST, CKPT_E2_LATEST, CKPT_E3_LATEST
    else:
        ckpt_e1, ckpt_e2, ckpt_e3 = CKPT_E1_BEST, CKPT_E2_BEST, CKPT_E3_BEST

    out_dir = _BASE_OUT / mode
    _patch_ckpts(ckpt_e1, ckpt_e2, ckpt_e3, out_dir)

    # Cek ketersediaan checkpoint
    missing = [str(p) for p in [ckpt_e1, ckpt_e2, ckpt_e3] if not p.exists()]
    if missing:
        print(f"[{mode.upper()}] ⚠ Checkpoint tidak ditemukan, mode dilewati:")
        for m in missing:
            print(f"    {m}")
        return 1

    print("\n" + "=" * 80)
    print(f"ANALISIS TAMBAHAN BAB IV — RUN ALL [{mode.upper()} CHECKPOINT]")
    print("=" * 80 + "\n")
    print(f"  E-1 : {ckpt_e1.name}")
    print(f"  E-2 : {ckpt_e2.name}")
    print(f"  E-3 : {ckpt_e3.name}")
    print(f"  OUT : {out_dir}\n")

    # Catat ukuran checkpoint sebelum apapun
    ckpt_e1_size_before = os.path.getsize(str(ckpt_e1))
    ckpt_e2_size_before = os.path.getsize(str(ckpt_e2))
    ckpt_e3_size_before = os.path.getsize(str(ckpt_e3))

    # ── TASK 1: Robustness ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 1: Pengujian Robustness")
    print("─" * 60)
    probs_e1_clean, probs_e2_clean, probs_e3_clean, labels_clean, rows_task1 = run_task1(
        return_clean_probs=True
    )

    # ── TASK 2: ROC Curve ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 2: Kurva ROC")
    print("─" * 60)
    auc_e1_roc, auc_e2_roc, auc_e3_roc = run_task2(
        probs_e1=probs_e1_clean,
        probs_e2=probs_e2_clean,
        probs_e3=probs_e3_clean,
        labels=labels_clean,
    )

    # ── TASK 3: DeLong's Test ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 3: Uji Signifikansi Statistik (DeLong's Test)")
    print("─" * 60)
    delong_result = run_task3(
        probs_e1=probs_e1_clean,
        probs_e2=probs_e2_clean,
        probs_e3=probs_e3_clean,
        labels=labels_clean,
    )

    # ── TASK 4: Efisiensi Komputasi ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TASK 4: Efisiensi Komputasi")
    print("─" * 60)
    efisiensi_results = run_task4()

    # ── TASK 5: Evaluasi Test Set (Best + Latest) ──────────────────────────
    print("\n" + "─" * 60)
    print("TASK 5: Evaluasi Test Set — Best & Latest Checkpoint")
    print("─" * 60)
    _scripts_dir = PROJECT_ROOT / "scripts"
    _eval_script = _scripts_dir / "evaluate_test_set.py"
    if _eval_script.exists():
        print("[Task5] Menjalankan evaluate_test_set.py --both ...")
        ret_eval = subprocess.call(
            [sys.executable, str(_eval_script), "--both"],
            cwd=str(PROJECT_ROOT),
        )
        if ret_eval == 0:
            print("[Task5] ✓ Evaluasi test set selesai.")
        else:
            print(f"[Task5] ✗ evaluate_test_set.py --both keluar dengan kode {ret_eval}")
    else:
        print(f"[Task5] ⚠ Script tidak ditemukan: {_eval_script} — dilewati.")

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
    auc_clean_e3_task1 = next(
        r["auc"] for r in rows_task1 if r["model"] == "E-3" and r["kondisi"] == "Clean"
    )
    diff_e1 = abs(auc_clean_e1_task1 - delong_result["auc_e1"])
    diff_e2 = abs(auc_clean_e2_task1 - delong_result["auc_e2"])
    diff_e3 = abs(auc_clean_e3_task1 - delong_result["auc_e3"])

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
        
    if diff_e3 < 1e-9:
        print(f"[V1] ✓ AUC clean E-3 konsisten: Task1={auc_clean_e3_task1:.6f} == Task3={delong_result['auc_e3']:.6f}")
    else:
        msg = f"[V1] ✗ AUC clean E-3 TIDAK KONSISTEN: Task1={auc_clean_e3_task1:.6f} vs Task3={delong_result['auc_e3']:.6f} (diff={diff_e3:.2e})"
        print(msg)
        errors.append(msg)

    # 2. Jumlah baris CSV Task1 == 45 (15 kondisi x 3 model)
    csv_path = out_dir / "tabel_4_11_robustness.csv"
    n_rows = len(csv_path.read_text(encoding="utf-8").strip().split("\n")) - 1  # minus header
    if n_rows == 45:
        print(f"[V2] ✓ Jumlah baris CSV Task1 = {n_rows} (expected 45)")
    else:
        msg = f"[V2] ✗ Jumlah baris CSV Task1 = {n_rows} (expected 45)"
        print(msg)
        errors.append(msg)

    # 3. delta_auc kondisi Clean == 0.0 untuk ketiga model
    delta_clean_e1 = next(
        r["delta_auc"] for r in rows_task1 if r["model"] == "E-1" and r["kondisi"] == "Clean"
    )
    delta_clean_e2 = next(
        r["delta_auc"] for r in rows_task1 if r["model"] == "E-2" and r["kondisi"] == "Clean"
    )
    delta_clean_e3 = next(
        r["delta_auc"] for r in rows_task1 if r["model"] == "E-3" and r["kondisi"] == "Clean"
    )
    if abs(delta_clean_e1) < 1e-9 and abs(delta_clean_e2) < 1e-9 and abs(delta_clean_e3) < 1e-9:
        print(f"[V3] ✓ Delta AUC kondisi Clean = 0.0 untuk E-1, E-2, E-3")
    else:
        msg = f"[V3] ✗ Delta AUC Clean bukan nol: E-1={delta_clean_e1}, E-2={delta_clean_e2}, E-3={delta_clean_e3}"
        print(msg)
        errors.append(msg)

    # 4. Checkpoint tidak berubah
    ckpt_e1_size_after = os.path.getsize(str(ckpt_e1))
    ckpt_e2_size_after = os.path.getsize(str(ckpt_e2))
    ckpt_e3_size_after = os.path.getsize(str(ckpt_e3))
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
    if ckpt_e3_size_before == ckpt_e3_size_after:
        print(f"[V4] ✓ Checkpoint E-3 tidak berubah ({ckpt_e3_size_after:,} bytes)")
    else:
        msg = f"[V4] ✗ CHECKPOINT E-3 BERUBAH! before={ckpt_e3_size_before}, after={ckpt_e3_size_after}"
        print(msg)
        errors.append(msg)

    # Ringkasan output file
    print("\n" + "=" * 80)
    print(f"RINGKASAN FILE OUTPUT [{mode.upper()}]")
    print("=" * 80)
    output_files = [
        out_dir / "tabel_4_11_robustness.csv",
        out_dir / "gambar_4_4_robustness_delta_auc.png",
        out_dir / "gambar_4_5_roc_curve.png",
        out_dir / "tabel_4_12_delong_test.csv",
        out_dir / "tabel_4_13_efisiensi.csv",
        out_dir / "gambar_4_6_efisiensi.png",
        PROJECT_ROOT / "results" / "test_set_eval" / "test_eval_summary.csv",
        PROJECT_ROOT / "results" / "test_set_eval" / "test_eval_summary_best.csv",
        PROJECT_ROOT / "results" / "test_set_eval" / "test_eval_summary_latest.csv",
    ]
    for f in output_files:
        status = "✓" if f.exists() else "✗ TIDAK ADA"
        size_str = f"({f.stat().st_size:,} bytes)" if f.exists() else ""
        print(f"  {status}  {f.name}  {size_str}")

    # V5. CSV test_eval_summary.csv (gabungan) harus punya 6 baris data (3 model × 2 ckpt)
    test_eval_csv = PROJECT_ROOT / "results" / "test_set_eval" / "test_eval_summary.csv"
    if test_eval_csv.exists():
        n_test_rows = len(test_eval_csv.read_text(encoding="utf-8").strip().split("\n")) - 1
        if n_test_rows == 6:
            print(f"[V5] ✓ test_eval_summary.csv: {n_test_rows} baris (expected 6: 3 model × 2 ckpt)")
        else:
            msg = f"[V5] ✗ test_eval_summary.csv: {n_test_rows} baris (expected 6)"
            print(msg)
            errors.append(msg)
    else:
        print("[V5] ⚠ test_eval_summary.csv belum ada — jalankan evaluate_test_set.py --both")

    if errors:
        print(f"\n[VERIFIKASI] {len(errors)} error ditemukan:")
        for e in errors:
            print(f"  {e}")
        return 1
    else:
        print("\n[VERIFIKASI] Semua cek lolos ✓")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all BAB IV analysis tasks."
    )
    parser.add_argument(
        "--mode", choices=["best", "latest", "both"], default="best",
        help="Checkpoint mode: best (default), latest, atau both (jalankan keduanya)."
    )
    args = parser.parse_args()

    if args.mode == "both":
        rc_best   = _run_all_tasks("best")
        rc_latest = _run_all_tasks("latest")
        return max(rc_best, rc_latest)
    else:
        return _run_all_tasks(args.mode)


if __name__ == "__main__":
    sys.exit(main())
