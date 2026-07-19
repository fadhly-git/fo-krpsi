"""Evaluasi final pada TEST SET untuk menjawab RQ1 secara resmi.

Menjawab RQ1: "Seberapa tinggi akurasi dan kinerja klasifikasi (AUC, F1-score)
sistem saat fitur spasial EfficientNet diintegrasikan dengan fitur frekuensi DCT,
dibandingkan dengan model spasial tunggal?"

Script ini mengevaluasi:
  - E-1: Model Hibrida (EfficientNet-B0 + DCT 192-dim)   → best_efficient_dct.pth
  - E-2: Model Baseline (EfficientNet-B0 saja)            → best_efficient_no_dct.pth
  - E-3: Model Cross-Attention                           → best_efficient_crossattn.pth

pada TEST SET yang dipisahkan saat training (test_indices.json, ~3.125 sampel),
sehingga model tidak pernah melihat data ini selama training maupun pemilihan checkpoint.

Usage (jalankan dari root project):
    FORCE_CPU=1 python scripts/evaluate_test_set.py
    FORCE_CPU=1 python scripts/evaluate_test_set.py --batch_size 16
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Tambahkan src/ ke path ────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    CFG,
    DATA_ROOT,
    DCT_ROOT,
    LOG_DIR,
    PROJECT_ROOT,
    apply_cpu_safety_overrides,
    resolve_device,
)
from dataset import FaceOnlyDataset, detect_dct_dim
from model import build_backbone, build_head

# ── Konstanta ─────────────────────────────────────────────────────────────────
REAL_CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"

# Checkpoint default (best = dipilih berdasarkan val AUC tertinggi saat training)
CKPT_BEST_E1   = REAL_CKPT_DIR / "best_efficient_dct.pth"
CKPT_BEST_E2   = REAL_CKPT_DIR / "best_efficient_no_dct.pth"
CKPT_BEST_E3   = REAL_CKPT_DIR / "best_efficient_crossattn.pth"

# Checkpoint latest (epoch terakhir pelatihan)
CKPT_LATEST_E1 = REAL_CKPT_DIR / "last_checkpoint.pth"
CKPT_LATEST_E2 = REAL_CKPT_DIR / "latest_no_dct.pth"
CKPT_LATEST_E3 = REAL_CKPT_DIR / "latest_crossattn.pth"

TEST_INDICES_PATH = PROJECT_ROOT / "data" / "processed" / "test_indices.json"

OUT_DIR = PROJECT_ROOT / "results" / "test_set_eval"

EVAL_TRANSFORM = A.Compose([
    A.Resize(CFG["img_size"], CFG["img_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ── Helper: load model ────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, dct_dim: int, device: torch.device, log, cross_attn: bool = False):
    backbone, feature_dim = build_backbone()
    if cross_attn:
        from model import build_head_cross_attention
        fusion, head = build_head_cross_attention(feature_dim, dct_dim)
    else:
        head = build_head(feature_dim, dct_dim)
        fusion = None

    ckpt = torch.load(str(ckpt_path), map_location=device)
    backbone_sd = ckpt.get("efficientnet_state_dict") or ckpt.get("resnet_state_dict")
    head_sd = ckpt.get("head_state_dict")

    if backbone_sd is None:
        raise KeyError(f"Checkpoint {ckpt_path.name} tidak memiliki 'efficientnet_state_dict'")
    if head_sd is None:
        raise KeyError(f"Checkpoint {ckpt_path.name} tidak memiliki 'head_state_dict'")

    backbone.load_state_dict(backbone_sd)
    head.load_state_dict(head_sd)

    if cross_attn:
        fusion_sd = ckpt.get("fusion_state_dict")
        if fusion_sd is None:
            raise KeyError(f"Checkpoint {ckpt_path.name} tidak memiliki 'fusion_state_dict'")
        fusion.load_state_dict(fusion_sd)
        fusion = fusion.to(device).eval()

    backbone = backbone.to(device).eval()
    head = head.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    val_auc = ckpt.get("best_auc", float("nan"))
    val_acc = ckpt.get("best_acc", float("nan"))
    val_f1  = ckpt.get("best_macro_f1", float("nan"))
    log(
        f"  Loaded {ckpt_path.name} | "
        f"epoch={epoch} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}"
    )
    return backbone, head, fusion


# ── Helper: DCT dari array (digunakan agar konsisten dengan task1_robustness) ─
try:
    from scipy.fft import dctn
    from scipy.stats import skew as scipy_skew

    def _block_view_8x8(y_channel):
        height, width = y_channel.shape
        if height < 8 or width < 8:
            y_channel = np.pad(
                y_channel,
                ((0, max(8 - height, 0)), (0, max(8 - width, 0))),
                mode="edge",
            )
            height, width = y_channel.shape
        h8 = (height // 8) * 8
        w8 = (width // 8) * 8
        y_channel = y_channel[:h8, :w8]
        return y_channel.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)

    def compute_dct_192(img_rgb_uint8: np.ndarray) -> np.ndarray:
        import cv2
        ycbcr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2YCrCb)
        y = ycbcr[:, :, 0].astype(np.float32)
        blocks = _block_view_8x8(y)
        dct_blocks = dctn(blocks, axes=(-2, -1), norm="ortho")
        coeffs = dct_blocks.reshape(dct_blocks.shape[0], 64)
        means = coeffs.mean(axis=0)
        variances = coeffs.var(axis=0)
        skews = scipy_skew(coeffs, axis=0, bias=False)
        feat = np.concatenate([means, variances, np.nan_to_num(skews, nan=0.0)])
        return np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def normalize_dct(dct: np.ndarray) -> np.ndarray:
        dct_mean = dct.mean()
        dct_std = max(dct.std(), 1e-6)
        dct = (dct - dct_mean) / dct_std
        return np.clip(np.nan_to_num(dct, nan=0.0, posinf=1e6, neginf=-1e6), -1e4, 1e4)

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ── Inference satu model pada test_samples ────────────────────────────────────
@torch.no_grad()
def run_inference(test_samples, backbone, head, fusion, device, dct_dim: int, log):
    """Forward pass seluruh test set, return (probs_fake, preds, labels)."""
    all_probs, all_preds, all_labels = [], [], []
    n = len(test_samples)

    for i, (img_path, dct_path, label) in enumerate(test_samples):
        if (i + 1) % 500 == 0:
            log(f"  Progress: {i+1}/{n}")

        # Load gambar
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_arr = np.array(img_pil, dtype=np.uint8)

        # Eval transform → tensor citra
        aug = EVAL_TRANSFORM(image=img_arr)
        img_tensor = aug["image"].unsqueeze(0).to(device)

        # Fitur DCT (E-1 saja)
        if dct_dim > 0:
            if _SCIPY_AVAILABLE:
                # Hitung DCT langsung dari array (lebih akurat, identik dengan training)
                dct_feat = normalize_dct(compute_dct_192(img_arr))
                dct_tensor = torch.tensor(dct_feat, dtype=torch.float32).unsqueeze(0).to(device)
            elif dct_path is not None and Path(str(dct_path)).exists():
                # Fallback: load file .npy precomputed
                dct_np = np.load(str(dct_path)).astype(np.float32)
                dct_mean = dct_np.mean()
                dct_std = max(dct_np.std(), 1e-6)
                dct_np = np.clip(
                    np.nan_to_num((dct_np - dct_mean) / dct_std, nan=0.0, posinf=1e6, neginf=-1e6),
                    -1e4, 1e4,
                )
                dct_tensor = torch.tensor(dct_np, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                dct_tensor = torch.zeros(1, dct_dim, dtype=torch.float32, device=device)

            feat = backbone(img_tensor)
            if fusion is not None:
                logit = head(fusion(feat, dct_tensor))
            else:
                logit = head(torch.cat([feat, dct_tensor], dim=1))
        else:
            feat = backbone(img_tensor)
            logit = head(feat)

        prob_fake = F.softmax(logit, dim=1)[0, 1].item()
        pred = int(logit.argmax(dim=1).item())

        all_probs.append(prob_fake)
        all_preds.append(pred)
        all_labels.append(label)

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


# ── Hitung semua metrik ───────────────────────────────────────────────────────
def compute_metrics(probs, preds, labels):
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    prec = precision_score(labels, preds, labels=[0, 1], average=None, zero_division=0)
    rec  = recall_score (labels, preds, labels=[0, 1], average=None, zero_division=0)
    f1   = f1_score     (labels, preds, labels=[0, 1], average=None, zero_division=0)
    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    macro_f1 = float(np.mean(f1))
    return {
        "acc": float(acc),
        "auc": float(auc),
        "prec_real": float(prec[0]),
        "prec_fake": float(prec[1]),
        "rec_real":  float(rec[0]),
        "rec_fake":  float(rec[1]),
        "f1_real":   float(f1[0]),
        "f1_fake":   float(f1[1]),
        "macro_f1":  macro_f1,
        "cm": cm.tolist(),
        "n_samples": int(len(labels)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Test-set final evaluation — deepfake detector")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default 1, per-sample inference untuk DCT realtime)")
    parser.add_argument("--use_latest", action="store_true",
                        help="Gunakan checkpoint epoch terakhir (last_checkpoint / latest_no_dct) "
                             "alih-alih checkpoint terbaik (best_efficient_dct / best_efficient_no_dct)")
    parser.add_argument("--ckpt_e1", type=str, default=None,
                        help="Override path checkpoint E-1 (hybrid). Jika diisi, --use_latest diabaikan untuk E-1.")
    parser.add_argument("--ckpt_e2", type=str, default=None,
                        help="Override path checkpoint E-2 (baseline). Jika diisi, --use_latest diabaikan untuk E-2.")
    parser.add_argument("--ckpt_e3", type=str, default=None,
                        help="Override path checkpoint E-3 (cross attention). Jika diisi, --use_latest diabaikan untuk E-3.")
    args = parser.parse_args()

    # Resolusi path checkpoint
    if args.ckpt_e1:
        ckpt_e1 = Path(args.ckpt_e1)
    elif args.use_latest:
        ckpt_e1 = CKPT_LATEST_E1
    else:
        ckpt_e1 = CKPT_BEST_E1

    if args.ckpt_e2:
        ckpt_e2 = Path(args.ckpt_e2)
    elif args.use_latest:
        ckpt_e2 = CKPT_LATEST_E2
    else:
        ckpt_e2 = CKPT_BEST_E2

    if args.ckpt_e3:
        ckpt_e3 = Path(args.ckpt_e3)
    elif args.use_latest:
        ckpt_e3 = CKPT_LATEST_E3
    else:
        ckpt_e3 = CKPT_BEST_E3

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"test_eval_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    ckpt_mode = "latest" if args.use_latest else "best"
    log("=" * 70)
    log("EVALUASI FINAL — TEST SET")
    log(f"Checkpoint mode : {ckpt_mode}")
    log(f"E-1 checkpoint  : {ckpt_e1.name}")
    log(f"E-2 checkpoint  : {ckpt_e2.name}")
    log(f"E-3 checkpoint  : {ckpt_e3.name}")
    log("=" * 70)

    # ── Device ────────────────────────────────────────────────────────────────
    device, force_cpu = resolve_device()
    apply_cpu_safety_overrides(device)
    log(f"Device: {device} | scipy available: {_SCIPY_AVAILABLE}")

    # ── Cek test_indices.json ─────────────────────────────────────────────────
    if not TEST_INDICES_PATH.exists():
        log(f"ERROR: {TEST_INDICES_PATH} tidak ditemukan.")
        log("Pastikan training sudah selesai — train.py menyimpan test_indices.json otomatis.")
        log_file.close()
        return 1

    with open(TEST_INDICES_PATH, encoding="utf-8") as f:
        test_indices = json.load(f)
    log(f"Test indices dimuat: {len(test_indices)} sampel dari {TEST_INDICES_PATH}")

    # ── Deteksi DCT dim ───────────────────────────────────────────────────────
    dct_dim = detect_dct_dim(DCT_ROOT)
    if dct_dim is None:
        log("WARNING: Precomputed DCT tidak ditemukan. Menggunakan dct_dim=192 (DCT dihitung realtime via scipy).")
        dct_dim = 192 if _SCIPY_AVAILABLE else 0
    log(f"DCT dim = {dct_dim}")

    # ── Build full dataset untuk ambil sample list ────────────────────────────
    log("Membangun dataset index...")
    full_dataset = FaceOnlyDataset(
        DATA_ROOT,
        dct_root=DCT_ROOT,
        transform=EVAL_TRANSFORM,
        dct_dim=dct_dim,
        use_dct=True,
        log_fn=log,
    )

    # Ambil test samples berdasarkan test_indices
    test_samples = [full_dataset.samples[i] for i in test_indices]
    labels_all = [s[2] for s in test_samples]
    n_real = labels_all.count(0)
    n_fake = labels_all.count(1)
    log(f"Test subset: {len(test_samples)} sampel | REAL={n_real} | FAKE={n_fake}")

    # ── Cek checkpoint ────────────────────────────────────────────────────────
    for ckpt_path in [ckpt_e1, ckpt_e2, ckpt_e3]:
        if not ckpt_path.exists():
            log(f"ERROR: Checkpoint tidak ditemukan: {ckpt_path}")
            log_file.close()
            return 1

    # ── Load dan evaluasi E-1 (Hybrid) ───────────────────────────────────────
    log("\n" + "─" * 60)
    log("E-1: Model Hibrida (EfficientNet-B0 + DCT)")
    log("─" * 60)
    backbone_e1, head_e1, _ = load_model(ckpt_e1, dct_dim, device, log)
    log(f"Menjalankan inference pada {len(test_samples)} sampel...")
    probs_e1, preds_e1, labels = run_inference(test_samples, backbone_e1, head_e1, None, device, dct_dim, log)
    m_e1 = compute_metrics(probs_e1, preds_e1, labels)

    # ── Load dan evaluasi E-2 (Baseline) ─────────────────────────────────────
    log("\n" + "─" * 60)
    log("E-2: Model Baseline (EfficientNet-B0 saja)")
    log("─" * 60)
    backbone_e2, head_e2, _ = load_model(ckpt_e2, 0, device, log)
    log(f"Menjalankan inference pada {len(test_samples)} sampel...")
    probs_e2, preds_e2, _ = run_inference(test_samples, backbone_e2, head_e2, None, device, 0, log)
    m_e2 = compute_metrics(probs_e2, preds_e2, labels)

    # ── Load dan evaluasi E-3 (Cross-Attention) ──────────────────────────────
    log("\n" + "─" * 60)
    log("E-3: Model Cross-Attention")
    log("─" * 60)
    backbone_e3, head_e3, fusion_e3 = load_model(ckpt_e3, dct_dim, device, log, cross_attn=True)
    log(f"Menjalankan inference pada {len(test_samples)} sampel...")
    probs_e3, preds_e3, _ = run_inference(test_samples, backbone_e3, head_e3, fusion_e3, device, dct_dim, log)
    m_e3 = compute_metrics(probs_e3, preds_e3, labels)

    # ── Print tabel hasil ─────────────────────────────────────────────────────
    log("\n" + "=" * 90)
    log("HASIL EVALUASI TEST SET — PERBANDINGAN E-1, E-2, dan E-3")
    log("=" * 90)
    log(f"{'Metrik':<15} | {'E-1 (Hybrid)':<15} | {'E-2 (Baseline)':<15} | {'E-3 (Cross)':<15} | {'Δ(E1-E2)':<10} | {'Δ(E3-E2)':<10}")
    log("-" * 90)

    metrics_rows = [
        ("Accuracy",    m_e1["acc"],       m_e2["acc"],       m_e3["acc"]),
        ("AUC",         m_e1["auc"],       m_e2["auc"],       m_e3["auc"]),
        ("Macro F1",    m_e1["macro_f1"],  m_e2["macro_f1"],  m_e3["macro_f1"]),
        ("F1 (FAKE)",   m_e1["f1_fake"],   m_e2["f1_fake"],   m_e3["f1_fake"]),
        ("F1 (REAL)",   m_e1["f1_real"],   m_e2["f1_real"],   m_e3["f1_real"]),
        ("Prec (FAKE)", m_e1["prec_fake"], m_e2["prec_fake"], m_e3["prec_fake"]),
        ("Rec  (FAKE)", m_e1["rec_fake"],  m_e2["rec_fake"],  m_e3["rec_fake"]),
        ("Prec (REAL)", m_e1["prec_real"], m_e2["prec_real"], m_e3["prec_real"]),
        ("Rec  (REAL)", m_e1["rec_real"],  m_e2["rec_real"],  m_e3["rec_real"]),
    ]
    for name, v1, v2, v3 in metrics_rows:
        delta12 = v1 - v2
        delta32 = v3 - v2
        log(f"{name:<15} | {v1:<15.4f} | {v2:<15.4f} | {v3:<15.4f} | {delta12:+.4f}    | {delta32:+.4f}")

    log("─" * 90)
    log(f"Confusion Matrix E-1 (REAL=0, FAKE=1):")
    cm1 = m_e1["cm"]
    log(f"  TN={cm1[0][0]:>5}  FP={cm1[0][1]:>5}")
    log(f"  FN={cm1[1][0]:>5}  TP={cm1[1][1]:>5}")
    log(f"Confusion Matrix E-2:")
    cm2 = m_e2["cm"]
    log(f"  TN={cm2[0][0]:>5}  FP={cm2[0][1]:>5}")
    log(f"  FN={cm2[1][0]:>5}  TP={cm2[1][1]:>5}")
    log(f"Confusion Matrix E-3:")
    cm3 = m_e3["cm"]
    log(f"  TN={cm3[0][0]:>5}  FP={cm3[0][1]:>5}")
    log(f"  FN={cm3[1][0]:>5}  TP={cm3[1][1]:>5}")
    log("=" * 90)

    # ── Simpan JSON ───────────────────────────────────────────────────────────
    results = {
        "timestamp": timestamp,
        "n_test_samples": len(test_samples),
        "n_real": n_real,
        "n_fake": n_fake,
        "dct_dim": dct_dim,
        "E1_hybrid": m_e1,
        "E2_baseline": m_e2,
        "E3_crossattn": m_e3,
    }
    json_path = OUT_DIR / f"test_eval_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Simpan CSV ────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "test_eval_summary.csv"
    csv_header = "model,n_samples,accuracy,auc,macro_f1,f1_fake,f1_real,prec_fake,prec_real,rec_fake,rec_real"
    e1_row = (
        f"E-1 (Hybrid),{m_e1['n_samples']},"
        f"{m_e1['acc']:.6f},{m_e1['auc']:.6f},{m_e1['macro_f1']:.6f},"
        f"{m_e1['f1_fake']:.6f},{m_e1['f1_real']:.6f},"
        f"{m_e1['prec_fake']:.6f},{m_e1['prec_real']:.6f},"
        f"{m_e1['rec_fake']:.6f},{m_e1['rec_real']:.6f}"
    )
    e2_row = (
        f"E-2 (Baseline),{m_e2['n_samples']},"
        f"{m_e2['acc']:.6f},{m_e2['auc']:.6f},{m_e2['macro_f1']:.6f},"
        f"{m_e2['f1_fake']:.6f},{m_e2['f1_real']:.6f},"
        f"{m_e2['prec_fake']:.6f},{m_e2['prec_real']:.6f},"
        f"{m_e2['rec_fake']:.6f},{m_e2['rec_real']:.6f}"
    )
    e3_row = (
        f"E-3 (Cross-Attention),{m_e3['n_samples']},"
        f"{m_e3['acc']:.6f},{m_e3['auc']:.6f},{m_e3['macro_f1']:.6f},"
        f"{m_e3['f1_fake']:.6f},{m_e3['f1_real']:.6f},"
        f"{m_e3['prec_fake']:.6f},{m_e3['prec_real']:.6f},"
        f"{m_e3['rec_fake']:.6f},{m_e3['rec_real']:.6f}"
    )
    csv_path.write_text("\n".join([csv_header, e1_row, e2_row, e3_row]) + "\n", encoding="utf-8")

    log(f"\nOutput:")
    log(f"  JSON lengkap : {json_path}")
    log(f"  CSV ringkasan: {csv_path}")
    log(f"  Log file     : {log_path}")
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
