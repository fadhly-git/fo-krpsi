"""Helper bersama untuk analisis tambahan BAB IV.

Menyediakan:
- load_model(): load backbone + head dari checkpoint, return dalam eval mode.
- get_val_subset(): replikasi exact 3-way stratified split dari train.py.
- REAL_CKPT_DIR: path ke direktori checkpoint aktual ('models/checkpoints/').
- CKPT_E1, CKPT_E2: path ke checkpoint terbaik masing-masing model.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Tambahkan src/ ke sys.path agar bisa import modul project
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Path checkpoint aktual — ada di 'models/checkpoints/'
_PROJECT_ROOT_COMMON = Path(__file__).resolve().parents[2]
REAL_CKPT_DIR = _PROJECT_ROOT_COMMON / "models" / "checkpoints"
CKPT_E1 = REAL_CKPT_DIR / "best_efficient_dct.pth"
CKPT_E2 = REAL_CKPT_DIR / "best_efficient_no_dct.pth"
CKPT_E3 = REAL_CKPT_DIR / "best_efficient_crossattn.pth"

from config import (
    CFG,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DCT_ROOT,
    apply_cpu_safety_overrides,
    resolve_device,
)
from dataset import FaceOnlyDataset, detect_dct_dim
from model import build_backbone, build_head


def load_model(checkpoint_path: Path, dct_dim: int, device: torch.device, cross_attn: bool = False):
    """Load backbone + head (serta fusion jika cross_attn=True) dari checkpoint.

    Args:
        checkpoint_path: Path ke file .pth checkpoint.
        dct_dim: Dimensi DCT feature (192 untuk E-1/E-3, 0 untuk E-2).
        device: torch.device target.
        cross_attn: Jika True, build fusion module dan muat state E-3.

    Returns:
        Tuple (backbone, head, fusion), semuanya dalam eval mode di device.
        Jika cross_attn=False, fusion bernilai None.
    """
    backbone, feature_dim = build_backbone()
    
    if cross_attn:
        from model import build_head_cross_attention
        fusion, head = build_head_cross_attention(feature_dim, dct_dim)
    else:
        head = build_head(feature_dim, dct_dim)
        fusion = None

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    backbone_state = ckpt.get("efficientnet_state_dict", ckpt.get("resnet_state_dict"))
    head_state = ckpt.get("head_state_dict")

    if backbone_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} tidak memiliki 'efficientnet_state_dict'")
    if head_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} tidak memiliki 'head_state_dict'")

    backbone.load_state_dict(backbone_state)
    head.load_state_dict(head_state)

    if cross_attn:
        fusion_state = ckpt.get("fusion_state_dict")
        if fusion_state is None:
            raise KeyError(f"Checkpoint {checkpoint_path} tidak memiliki 'fusion_state_dict'")
        fusion.load_state_dict(fusion_state)
        fusion = fusion.to(device)
        fusion.eval()

    backbone = backbone.to(device)
    head = head.to(device)
    backbone.eval()
    head.eval()
    return backbone, head, fusion


def get_val_subset():
    """Replikasi EXACT 3-way stratified split dari train.py.

    train.py melakukan dua tahap split:
      1. train_val vs test  — test_size = round(n_total * test_ratio=0.1)
      2. train   vs val     — test_size = round(len(train_val) * val_ratio=0.2)
    Keduanya dengan random_state=CFG["seed"]=42 dan stratify=labels.

    Fungsi ini mereplikasi KEDUA tahap tersebut menggunakan FaceOnlyDataset
    (sumber data tunggal, identik dengan yang dipakai train.py) sehingga
    val_indices yang dihasilkan PERSIS sama seperti saat training berlangsung.

    Returns:
        List of (img_path, dct_path_or_none, label) untuk val_indices saja.
    """
    dct_dim = detect_dct_dim(DCT_ROOT) or 0

    # Bangun dataset IDENTIK dengan train.py (FaceOnlyDataset, satu sumber data)
    full_dataset = FaceOnlyDataset(
        DATA_ROOT,
        dct_root=DCT_ROOT,
        transform=None,  # transform tidak dipakai untuk membangun sample list
        max_fake=CFG.get("max_subset_images"),
        max_real=CFG.get("max_subset_beauty_images"),
        dct_dim=dct_dim,
        use_dct=True,
    )

    n_total = len(full_dataset)
    labels_all = [label for (_, _, label) in full_dataset.samples]
    indices = np.arange(n_total)

    # ── Tahap 1: pisahkan test set (identik baris 180-186 train.py) ──────────
    test_ratio = float(CFG.get("test_ratio", 0.1))
    test_size = max(1, int(round(n_total * test_ratio)))

    train_val_indices, _ = train_test_split(
        indices,
        test_size=test_size,
        random_state=CFG["seed"],
        stratify=labels_all,
        shuffle=True,
    )

    # ── Tahap 2: pisahkan val dari train_val (identik baris 188-199 train.py) ─
    train_val_labels = [labels_all[i] for i in train_val_indices]
    val_ratio = float(CFG.get("val_ratio", 0.2))
    val_ratio = min(max(val_ratio, 0.01), 0.9)
    val_size = max(1, int(round(len(train_val_indices) * val_ratio)))
    val_size = min(val_size, len(train_val_indices) - 1)

    _, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=CFG["seed"],
        stratify=train_val_labels,
        shuffle=True,
    )

    return [full_dataset.samples[i] for i in val_indices]


def get_dct_dim():
    """Deteksi dct_dim dari data yang tersedia."""
    return detect_dct_dim(DCT_ROOT) or 192
