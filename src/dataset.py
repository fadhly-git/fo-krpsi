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

# ─── Generators: aturan path untuk data faces ───────────────────────────────
# StyleGAN* → semua file di setiap subdir adalah wajah
_STYLEGAN_GENERATORS = {"StyleGAN", "StyleGAN2", "StyleGAN3"}
# SD-based → wajah ada di subfolder 'faces/'
_SD_GENERATORS = {
	"FLUX.1",
	"StableDiffusion1.5",
	"StableDiffusion2",
	"StableDiffusion3",
	"StableDiffusionXL",
}


class FaceOnlyDataset(Dataset):
	"""
	Dataset khusus untuk direktori Twitter dengan struktur:
	  Twitter/
	    Fake/<Generator>/faces/<img>       (SD-based)
	    Fake/<Generator>/<sub>/<img>        (StyleGAN*)
	    Real/FFHQ/<img>                    (real)

	Label: 0 = real, 1 = fake  (mengikuti konvensi MixedDataset).
	"""

	def __init__(
		self,
		twitter_root: Path,
		dct_root: Path = None,
		transform=None,
		max_fake: int = None,
		max_real: int = None,
		dct_dim: int = 0,
		use_dct: bool = False,
		log_fn=None,
	):
		self.transform = transform if transform is not None else light_transform
		self.dct_dim = int(dct_dim)
		self.use_dct = bool(use_dct)
		self.log_fn = log_fn
		self.samples = []  # (img_path, dct_path_or_none, label)

		skipped = 0
		fake_dir = Path(twitter_root) / "Fake"
		real_dir = Path(twitter_root) / "Real"

		# ── FAKE ──────────────────────────────────────────────────────────────
		fake_paths: list[Path] = []
		if fake_dir.exists():
			for generator in sorted(fake_dir.iterdir()):
				if not generator.is_dir():
					continue
				name = generator.name
				if name in _STYLEGAN_GENERATORS:
					# semua file di seluruh subdir adalah wajah
					for f in sorted(generator.rglob("*")):
						if f.is_file() and f.suffix.lower() in VALID_EXT:
							fake_paths.append(f)
							if max_fake is not None and len(fake_paths) >= max_fake:
								break
					if max_fake is not None and len(fake_paths) >= max_fake:
						break
				elif name in _SD_GENERATORS:
					faces_dir = generator / "faces"
					if not faces_dir.exists():
						if log_fn:
							log_fn(f"[FaceOnlyDataset] WARN: faces/ not found for {name}, skipping.")
						continue
					for f in sorted(faces_dir.rglob("*")):
						if f.is_file() and f.suffix.lower() in VALID_EXT:
							fake_paths.append(f)
							if max_fake is not None and len(fake_paths) >= max_fake:
								break
					if max_fake is not None and len(fake_paths) >= max_fake:
						break
				else:
					if log_fn:
						log_fn(f"[FaceOnlyDataset] WARN: Unknown generator '{name}', skipping.")

		# ── REAL (hanya FFHQ) ─────────────────────────────────────────────────
		real_paths: list[Path] = []
		ffhq_dir = real_dir / "FFHQ"
		if ffhq_dir.exists():
			for f in sorted(ffhq_dir.rglob("*")):
				if f.is_file() and f.suffix.lower() in VALID_EXT:
					real_paths.append(f)
					if max_real is not None and len(real_paths) >= max_real:
						break
		else:
			if log_fn:
				log_fn(f"[FaceOnlyDataset] WARN: FFHQ dir not found at {ffhq_dir}")

		# ── Build samples list ────────────────────────────────────────────────
		# label 0 = real, label 1 = fake  (sama seperti MixedDataset / ImageFolder)
		for p in real_paths:
			dct_p = self._resolve_dct(p, twitter_root, dct_root)
			self.samples.append((str(p), dct_p, 0))
		for p in fake_paths:
			dct_p = self._resolve_dct(p, twitter_root, dct_root)
			self.samples.append((str(p), dct_p, 1))

		if log_fn:
			log_fn(
				f"[FaceOnlyDataset] Built: real={len(real_paths)} fake={len(fake_paths)} "
				f"total={len(self.samples)} skipped={skipped} use_dct={self.use_dct}"
			)

	def _resolve_dct(self, img_path: Path, img_root: Path, dct_root) -> "Path | None":
		if not self.use_dct or dct_root is None:
			return None
		try:
			rel = img_path.relative_to(img_root)
			dct_p = Path(dct_root) / rel.with_suffix(".npy")
			return dct_p if dct_p.exists() else None
		except ValueError:
			return None

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
			dct_std = max(dct.std() if dct.size else 1.0, 1e-6)
			dct = (dct - dct_mean) / dct_std

		dct = np.nan_to_num(dct, nan=0.0, posinf=1e6, neginf=-1e6)
		dct = np.clip(dct, -1e4, 1e4)
		dct = torch.tensor(dct, dtype=torch.float32)

		return img, dct, label
