"""E-3: Training pipeline dengan Cross-Attention Fusion (EfficientNet-B0 + DCT).

Skenario E-3 mengganti mekanisme fusi dari konkatenasi langsung (E-1) menjadi
cross-attention: Q dari fitur spasial EfficientNet-B0 (1280-dim),
K/V dari 3 token DCT @64-dim (mean/variance/skewness dari pre-computed DCT 192-dim).

Semua hyperparameter, split, seed, dan augmentasi identik dengan train.py
karena script ini hanya memaksa FUSION_MODE=cross_attention lalu memanggil
pipeline yang sama via train_main_with_crossattn().

Refs fusi:
  Chen et al. (CrossViT, arXiv:2103.14899)
  Qiao et al. (HCMA, arXiv:2504.17223)
  Sen & Mukherjee (CSAF, arXiv:2601.03382)
  Lv et al. (SFMFNet TSCA, arXiv:2508.20449)
  Khan et al. (CAMME, arXiv:2505.18035)

Usage:
    cd x:/
    python src/train_crossattn.py

Checkpoint yang dihasilkan:
  models/checkpoints/best_efficient_crossattn.pth   ← AUC validasi terbaik
  models/checkpoints/last_checkpoint_crossattn.pth  ← checkpoint terakhir tiap epoch
"""

import os
import sys

# Override env vars sebelum import apapun dari pipeline
os.environ["USE_DCT"] = "1"            # DCT harus aktif untuk E-3
os.environ["FUSION_MODE"] = "cross_attention"

# Import setelah env override (pola identik dengan train_no_dct.py)
from train_crossattn_main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
