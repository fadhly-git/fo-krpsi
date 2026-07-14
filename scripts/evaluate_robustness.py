"""Robustness evaluation untuk hybrid EfficientNet+DCT deepfake detector.

Menjawab RQ2: "Seberapa tinggi ketahanan arsitektur hibrida terhadap
degradasi visual seperti kompresi, noise, dan occlusion?"

Evaluasi dilakukan pada test set yang dipisahkan saat training (test_indices.json),
sehingga model tidak pernah melihat data ini sebelumnya.

Skenario degradasi:
  - Kompresi JPEG : quality 70, 50, 30
  - Gaussian Noise: variance 25, 50, 100
  - Occlusion     : kotak hitam di tengah, 30%, 50%, 70% area gambar

Usage:
    cd final-skripsi
    python scripts/evaluate_robustness.py              # eval model DCT (hybrid)
    python scripts/evaluate_robustness.py --use_dct 0  # eval model baseline
    python scripts/evaluate_robustness.py --ckpt models/checkpoints/best_efficient_dct.pth
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Add src/ ke path agar bisa import modul training
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CFG, CHECKPOINT_DIR, DATA_ROOT, DCT_ROOT, LOG_DIR, PROJECT_ROOT  # noqa: E402
from dataset import FaceOnlyDataset, detect_dct_dim  # noqa: E402
from model import build_backbone, build_head  # noqa: E402


TEST_INDICES_PATH = PROJECT_ROOT / "data/processed/test_indices.json"
IMG_SIZE = CFG["img_size"]
_NORM = [
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]


def _occlude_center(ratio: float):
    """Albumentations Lambda: tempel kotak hitam di tengah gambar (deterministic)."""
    size = int(IMG_SIZE * ratio)

    def _fn(image, **kwargs):
        h, w = image.shape[:2]
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - size // 2)
        x1 = max(0, cx - size // 2)
        y2 = min(h, y1 + size)
        x2 = min(w, x1 + size)
        out = image.copy()
        out[y1:y2, x1:x2] = 0
        return out

    return A.Lambda(image=_fn, p=1.0)


def build_degradation_scenarios() -> dict:
    """Kembalikan dict nama -> albumentations Compose transform."""
    resize = A.Resize(IMG_SIZE, IMG_SIZE)
    return {
        # === Tanpa degradasi ===
        "clean": A.Compose([resize] + _NORM),
        # === Kompresi JPEG ===
        "compression_q70": A.Compose([resize, A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0)] + _NORM),
        "compression_q50": A.Compose([resize, A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0)] + _NORM),
        "compression_q30": A.Compose([resize, A.ImageCompression(quality_lower=30, quality_upper=30, p=1.0)] + _NORM),
        # === Gaussian Noise ===
        "noise_var25":  A.Compose([resize, A.GaussNoise(var_limit=(25, 25), p=1.0)] + _NORM),
        "noise_var50":  A.Compose([resize, A.GaussNoise(var_limit=(50, 50), p=1.0)] + _NORM),
        "noise_var100": A.Compose([resize, A.GaussNoise(var_limit=(100, 100), p=1.0)] + _NORM),
        # === Occlusion (center patch, deterministic) ===
        "occlusion_30pct": A.Compose([resize, _occlude_center(0.30)] + _NORM),
        "occlusion_50pct": A.Compose([resize, _occlude_center(0.50)] + _NORM),
        "occlusion_70pct": A.Compose([resize, _occlude_center(0.70)] + _NORM),
    }


@torch.no_grad()
def evaluate_on_loader(loader, backbone, head, device: torch.device, use_dct: bool, dct_dim: int) -> dict:
    """Jalankan inferensi dan hitung semua metrik."""
    backbone.eval()
    head.eval()

    all_preds, all_probs, all_targets = [], [], []

    for imgs, dcts, labels in loader:
        imgs = imgs.to(device)

        if use_dct and dct_dim > 0:
            dcts = dcts.to(device)
            if dcts.dim() > 2:
                dcts = dcts.view(dcts.size(0), -1)
            feats = backbone(imgs)
            logits = head(torch.cat([feats, dcts], dim=1))
        else:
            feats = backbone(imgs)
            logits = head(feats)

        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_targets.extend(labels.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    f1_per = f1_score(all_targets, all_preds, labels=[0, 1], average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return {
        "acc": float(acc),
        "auc": float(auc),
        "f1_real": float(f1_per[0]),
        "f1_fake": float(f1_per[1]),
        "macro_f1": float(np.mean(f1_per)),
        "cm": cm.tolist(),
        "n_samples": len(all_targets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Robustness evaluation — deepfake detector")
    parser.add_argument("--use_dct", type=int, default=1,
                        help="1=hybrid (DCT+image), 0=baseline image-only")
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path ke checkpoint (.pth). Default: best checkpoint sesuai use_dct.")
    args = parser.parse_args()

    use_dct = bool(args.use_dct)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "hybrid_dct" if use_dct else "baseline_no_dct"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = LOG_DIR / f"robustness_{mode_str}_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"=== Robustness Evaluation | mode={mode_str} | device={device} ===")

    # --- Cek test_indices ---
    if not TEST_INDICES_PATH.exists():
        log(f"ERROR: {TEST_INDICES_PATH} tidak ditemukan.")
        log("Jalankan training terlebih dahulu (train.py menyimpan test_indices.json secara otomatis).")
        log_file.close()
        return 1

    with open(TEST_INDICES_PATH) as f:
        test_indices = json.load(f)
    log(f"Test indices: {len(test_indices)} sampel dari {TEST_INDICES_PATH}")

    # --- DCT dim ---
    if use_dct:
        dct_dim = detect_dct_dim(DCT_ROOT)
        if dct_dim is None:
            log("WARNING: DCT tidak ditemukan — fallback ke dct_dim=0 (image-only mode)")
            dct_dim = 0
        else:
            log(f"DCT dim = {dct_dim}")
    else:
        dct_dim = 0

    # --- Build model ---
    backbone, feature_dim = build_backbone(log_fn=log)
    backbone = backbone.to(device)
    head = build_head(feature_dim, dct_dim).to(device)

    # --- Load checkpoint ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_name = "best_efficient_dct.pth" if use_dct else "best_efficient_no_dct.pth"
        ckpt_path = CHECKPOINT_DIR / ckpt_name

    if not ckpt_path.exists():
        log(f"ERROR: Checkpoint tidak ditemukan di {ckpt_path}")
        log_file.close()
        return 1

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone_sd = ckpt.get("efficientnet_state_dict") or ckpt.get("resnet_state_dict")
    head_sd = ckpt.get("head_state_dict")
    if backbone_sd is None or head_sd is None:
        log("ERROR: Checkpoint tidak memiliki key yang dibutuhkan.")
        log_file.close()
        return 1

    backbone.load_state_dict(backbone_sd)
    head.load_state_dict(head_sd)
    best_auc_info = ckpt.get("best_auc", float("nan"))
    best_macro_f1_info = ckpt.get("best_macro_f1", float("nan"))
    log(
        f"Checkpoint loaded: {ckpt_path.name} | "
        f"epoch={ckpt.get('epoch', '?')} | "
        f"val_auc={best_auc_info:.4f} | val_macro_f1={best_macro_f1_info:.4f}"
    )

    # --- Build dataset (transform diganti per skenario) ---
    scenarios = build_degradation_scenarios()
    log("Membangun dataset index...")
    full_dataset = FaceOnlyDataset(
        DATA_ROOT,
        dct_root=DCT_ROOT,
        transform=scenarios["clean"],
        dct_dim=dct_dim,
        use_dct=use_dct,
        log_fn=log,
    )
    test_subset = Subset(full_dataset, test_indices)
    log(f"Test subset: {len(test_subset)} sampel\n")

    # --- Evaluasi per skenario ---
    results = {}
    log("--- Mulai evaluasi robustness ---\n")

    for scenario_name, transform in scenarios.items():
        full_dataset.transform = transform  # swap transform tanpa rebuild dataset
        loader = DataLoader(
            test_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        try:
            m = evaluate_on_loader(loader, backbone, head, device, use_dct, dct_dim)
            results[scenario_name] = m
            log(
                f"[{scenario_name:<20}] "
                f"AUC={m['auc']:.4f} | MacroF1={m['macro_f1']:.4f} | "
                f"F1_fake={m['f1_fake']:.4f} | F1_real={m['f1_real']:.4f} | "
                f"Acc={m['acc']:.4f} | n={m['n_samples']}"
            )
        except Exception as exc:
            log(f"[{scenario_name}] GAGAL: {exc}")
            results[scenario_name] = {"error": str(exc)}

    # --- Summary table ---
    log("\n" + "=" * 95)
    log(f"ROBUSTNESS SUMMARY — {mode_str}")
    log("=" * 95)
    log(f"{'Scenario':<22} {'AUC':>8} {'Delta AUC':>10} {'Macro F1':>10} {'F1 Fake':>9} {'F1 Real':>9} {'Acc':>8}")
    log("-" * 95)
    clean_auc = results.get("clean", {}).get("auc", float("nan"))
    for name, m in results.items():
        if "error" in m:
            log(f"{name:<22} ERROR: {m['error']}")
        else:
            delta = (m["auc"] - clean_auc) if name != "clean" and np.isfinite(clean_auc) else float("nan")
            delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "   —   "
            log(
                f"{name:<22} {m['auc']:>8.4f} {delta_str:>10} "
                f"{m['macro_f1']:>10.4f} {m['f1_fake']:>9.4f} {m['f1_real']:>9.4f} {m['acc']:>8.4f}"
            )

    # --- Simpan JSON ---
    results_path = LOG_DIR / f"robustness_results_{mode_str}_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log(f"\nHasil lengkap: {results_path}")
    log(f"Log: {log_path}")
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
