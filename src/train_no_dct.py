"""Baseline training: EfficientNet-B0 image-only (tanpa DCT).

Digunakan untuk menjawab RQ1: membandingkan model hybrid (EfficientNet+DCT)
dengan model spasial tunggal pada kondisi yang identik.

Semua hyperparameter, split, seed, dan augmentasi identik dengan train.py
karena script ini hanya memaksa USE_DCT=0 lalu memanggil pipeline yang sama.

Usage:
    cd final-skripsi/src
    python train_no_dct.py
"""

import os
import sys

os.environ["USE_DCT"] = "0"

from train import main  # noqa: E402 (import setelah env override)

if __name__ == "__main__":
    sys.exit(main())
