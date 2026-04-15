"""Dataset campuran image + fitur DCT serta utilitas deteksi dimensi DCT."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from config import VALID_EXT
from transforms import light_transform


class MixedDataset(Dataset):
	"""
	Build a unified list of samples from two ImageFolder roots.
	Each sample: (img_path, dct_path_or_none, label)
	"""

	def __init__(
		self,
		img_root1: Path,
		img_root2: Path,
		dct_root1: Path,
		dct_root2: Path,
		transform=None,
		max_root1: int = None,
		max_root2: int = None,
		dct_dim: int = 0,
		use_dct: bool = True,
		log_fn=None,
	):
		self.transform = transform if transform is not None else light_transform
		self.samples = []
		self.dct_dim = int(dct_dim)
		self.use_dct = bool(use_dct)
		self.log_fn = log_fn

		skipped_nonimage = 0
		skipped_missing_dct = 0

		def process_root(img_root: Path, dct_root: Path, max_items: int = None):
			nonlocal skipped_nonimage, skipped_missing_dct
			if not img_root.exists():
				return

			ds = datasets.ImageFolder(str(img_root))
			added = 0
			for img_path, label in ds.samples:
				ext = Path(img_path).suffix.lower()
				if ext not in VALID_EXT:
					skipped_nonimage += 1
					continue

				rel = Path(img_path).relative_to(img_root)
				dct_p = Path(dct_root) / rel.with_suffix(".npy")
				if self.use_dct and (not dct_p.exists()):
					skipped_missing_dct += 1
					dct_p = None
				elif not self.use_dct:
					dct_p = None

				self.samples.append((img_path, dct_p, label))
				added += 1

				if max_items is not None and added >= max_items:
					break

		process_root(img_root1, dct_root1, max_items=max_root1)
		process_root(img_root2, dct_root2, max_items=max_root2)

		if self.log_fn is not None:
			self.log_fn(
				f"Built MixedDataset: total valid samples={len(self.samples)} "
				f"(use_dct={self.use_dct}, skipped_nonimage={skipped_nonimage}, skipped_missing_dct={skipped_missing_dct})"
			)

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		img_path, dct_path, label = self.samples[idx]

		img = Image.open(img_path).convert("RGB")
		img = np.array(img)
		if self.transform:
			img = self.transform(image=img)["image"]

		if (not self.use_dct) or dct_path is None:
			dct = np.zeros((self.dct_dim,), dtype=np.float32)
		else:
			dct = np.load(dct_path).astype(np.float32)
			dct_mean = dct.mean() if dct.size else 0.0
			dct_std = dct.std() if dct.size else 1.0
			dct_std = max(dct_std, 1e-6)
			dct = (dct - dct_mean) / dct_std

		dct = np.nan_to_num(dct, nan=0.0, posinf=1e6, neginf=-1e6)
		dct = np.clip(dct, -1e4, 1e4)
		dct = torch.tensor(dct, dtype=torch.float32)

		return img, dct, label


def detect_dct_dim(dct_root: Path):
	"""Detect DCT feature dimension from the first .npy file under dct_root."""
	for root, _, files in os.walk(dct_root):
		for filename in files:
			if filename.endswith(".npy"):
				arr = np.load(Path(root) / filename)
				return int(np.prod(arr.shape))
	return None
