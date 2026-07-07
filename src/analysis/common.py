"""Helper bersama untuk analisis tambahan BAB IV.

Menyediakan:
- load_model(): load backbone + head dari checkpoint, return dalam eval mode.
- get_val_subset(): replikasi exact stratified split dari train.py.
- REAL_CKPT_DIR: path ke direktori checkpoint aktual ('models fix/checkpoints/').
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

# Path checkpoint aktual — ada di 'models fix/' bukan 'models/' (spasi dalam nama folder)
_PROJECT_ROOT_COMMON = Path(__file__).resolve().parents[2]
REAL_CKPT_DIR = _PROJECT_ROOT_COMMON / "models fix" / "checkpoints"
CKPT_E1 = REAL_CKPT_DIR / "best_efficient_dct.pth"
CKPT_E2 = REAL_CKPT_DIR / "best_efficient_no_dct.pth"

from config import (
    CFG,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DATA_ROOT_BEAUTY,
    DCT_ROOT,
    DCT_ROOT_BEAUTY,
    apply_cpu_safety_overrides,
    resolve_device,
)
from dataset import MixedDataset, detect_dct_dim
from model import build_backbone, build_head


def load_model(checkpoint_path: Path, dct_dim: int, device: torch.device):
    """Load backbone + head dari checkpoint, return (backbone, head) dalam eval mode.

    Args:
        checkpoint_path: Path ke file .pth checkpoint.
        dct_dim: Dimensi DCT feature (192 untuk E-1, 0 untuk E-2).
        device: torch.device target.

    Returns:
        Tuple (backbone, head), keduanya dalam eval mode di device.
    """
    backbone, feature_dim = build_backbone()
    head = build_head(feature_dim, dct_dim)

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    backbone_state = ckpt.get("efficientnet_state_dict", ckpt.get("resnet_state_dict"))
    head_state = ckpt.get("head_state_dict")

    if backbone_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} tidak memiliki 'efficientnet_state_dict'")
    if head_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} tidak memiliki 'head_state_dict'")

    backbone.load_state_dict(backbone_state)
    head.load_state_dict(head_state)

    backbone = backbone.to(device)
    head = head.to(device)
    backbone.eval()
    head.eval()
    return backbone, head


def get_val_subset():
    """Replikasi exact stratified split dari train.py.

    Returns:
        List of (img_path_str, dct_path_or_none, label) untuk val_indices saja.
        dct_path_or_none adalah Path atau None (sesuai MixedDataset.samples).
    """
    # Deteksi dct_dim (192) — digunakan hanya untuk membangun sample list
    dct_dim = detect_dct_dim(DCT_ROOT) or detect_dct_dim(DCT_ROOT_BEAUTY) or 0

    full_dataset = MixedDataset(
        DATA_ROOT,
        DATA_ROOT_BEAUTY,
        DCT_ROOT,
        DCT_ROOT_BEAUTY,
        transform=None,  # transform tidak dipakai untuk membangun samples list
        max_root1=CFG.get("max_subset_images"),
        max_root2=CFG.get("max_subset_beauty_images"),
        dct_dim=dct_dim,
        use_dct=True,  # gunakan True agar samples mencakup dct_path
    )

    n_total = len(full_dataset)
    labels_all = [label for (_, _, label) in full_dataset.samples]

    # REPLIKASI PERSIS dari train.py baris 176-191
    val_ratio = float(CFG.get("val_ratio", 0.2))
    val_ratio = min(max(val_ratio, 0.01), 0.9)
    val_size = max(1, int(round(n_total * val_ratio)))
    val_size = min(val_size, n_total - 1)

    indices = np.arange(n_total)
    _, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=CFG["seed"],
        stratify=labels_all,
        shuffle=True,
    )

    return [full_dataset.samples[i] for i in val_indices]


def get_dct_dim():
    """Deteksi dct_dim dari data yang tersedia."""
    return detect_dct_dim(DCT_ROOT) or detect_dct_dim(DCT_ROOT_BEAUTY) or 192
