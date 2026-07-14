"""Konfigurasi global training pipeline (CFG, path constants, dan device setup)."""

from pathlib import Path
import os

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_ROOT = PROJECT_ROOT / "data/raw/true-fake/Twitter"
DATA_ROOT_BEAUTY = PROJECT_ROOT / "data/raw/true-fake/TELEGRAM_DUMMY_DO_NOT_USE"

DCT_ROOT = PROJECT_ROOT / "data/processed/true-fake/Twitter/dct_features"
DCT_ROOT_BEAUTY = PROJECT_ROOT / "data/processed/true-fake/TELEGRAM_DUMMY_DO_NOT_USE/dct_features"

LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "models/checkpoints"

LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# valid image extensions (lowercase)
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

CFG = {
	# Ref: Tan & Le, ICML 2019 (EfficientNet-B0 native resolution 224)
	"img_size": 224,
	"use_dct": True,
	"batch_size": 32,
	"epochs": 45,
	"lr_backbone": 3e-5,
	"lr_head": 5e-4,
	"val_ratio": 0.2,
	"num_workers": 4,
	"seed": 42,
	"target_per_class": None,
	"max_subset_images": None,
	"max_subset_beauty_images": None,
	"debug_every_n_epochs": 5,
	"debug_massive": False,
	"debug_hist_bins": 64,
	"validate_train_max_batches": 100,
	# Ref: Yang et al., 2025 (smoother augmentation transition)
	"aug_medium_epoch": 10,
	"aug_heavy_epoch": 20,
	# Ref: Prechelt, 1998 (early stopping)
	"early_stopping_patience": 15,
	"min_epoch_before_stop": 25,
	# 3-way split: test set held out for final reporting (Batasan 1)
	"test_ratio": 0.1,
	# LR reset values when augmentation phase transitions (avoids LR=0 at heavy phase start)
	"lr_phase_reset_backbone": 1e-5,
	"lr_phase_reset_head": 2e-4,
}


def resolve_device():
	"""Resolve runtime device with optional FORCE_CPU override."""
	force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
	if force_cpu:
		device = torch.device("cpu")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return device, force_cpu


def apply_runtime_overrides(log_fn=None):
	"""Apply optional env-based runtime overrides to CFG in-place."""
	if CFG["num_workers"] < 0:
		CFG["num_workers"] = 0

	use_dct_env = os.environ.get("USE_DCT", None)
	if use_dct_env is not None:
		v = use_dct_env.strip().lower()
		if v in {"1", "true", "yes", "y", "on"}:
			CFG["use_dct"] = True
		elif v in {"0", "false", "no", "n", "off"}:
			CFG["use_dct"] = False
		elif log_fn is not None:
			log_fn(f"WARN: invalid USE_DCT='{use_dct_env}' — expected 0/1, true/false")

	max_subset_env = os.environ.get("MAX_SUBSET", None)
	max_subset_beauty_env = os.environ.get("MAX_SUBSET_BEAUTY", None)

	if max_subset_env is not None:
		try:
			CFG["max_subset_images"] = int(max_subset_env)
		except Exception:
			if log_fn is not None:
				log_fn(f"WARN: invalid MAX_SUBSET='{max_subset_env}' — ignoring")

	if max_subset_beauty_env is not None:
		try:
			CFG["max_subset_beauty_images"] = int(max_subset_beauty_env)
		except Exception:
			if log_fn is not None:
				log_fn(f"WARN: invalid MAX_SUBSET_BEAUTY='{max_subset_beauty_env}' — ignoring")


def apply_cpu_safety_overrides(device):
	"""Apply safer defaults for CPU runs to reduce OOM risk."""
	if device.type == "cpu":
		try:
			torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
		except Exception:
			pass
		CFG["num_workers"] = 0
