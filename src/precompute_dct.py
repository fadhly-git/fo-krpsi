"""Script standalone untuk precompute fitur DCT 192-dim dari dataset gambar."""

import argparse
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.stats import skew
from tqdm import tqdm


VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def block_view_8x8(y_channel):
	"""Create non-overlapping 8x8 blocks from luminance channel."""
	height, width = y_channel.shape
	if height < 8 or width < 8:
		pad_h = max(8 - height, 0)
		pad_w = max(8 - width, 0)
		y_channel = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode="edge")
		height, width = y_channel.shape

	height8 = (height // 8) * 8
	width8 = (width // 8) * 8
	y_channel = y_channel[:height8, :width8]

	blocks = y_channel.reshape(height8 // 8, 8, width8 // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
	return blocks


def compute_dct_feature_192(img_path):
	"""Compute 192-D DCT statistics feature (mean/var/skew over 8x8 frequencies)."""
	img = Image.open(img_path).convert("YCbCr")
	y = np.asarray(img, dtype=np.float32)[:, :, 0]

	blocks = block_view_8x8(y)
	dct_blocks = dctn(blocks, axes=(-2, -1), norm="ortho")
	coeffs = dct_blocks.reshape(dct_blocks.shape[0], 64)

	means = coeffs.mean(axis=0)
	variances = coeffs.var(axis=0)
	skews = skew(coeffs, axis=0, bias=False)

	features = np.concatenate([means, variances, np.nan_to_num(skews, nan=0.0)], axis=0)
	features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
	if features.shape[0] != 192:
		raise RuntimeError(f"Unexpected feature dim {features.shape[0]} for {img_path}")
	return features


def process_one_image(img_path_str, img_root_str, dct_root_str, overwrite=False):
	"""Worker process: compute and save DCT feature for one image."""
	img_path = Path(img_path_str)
	img_root = Path(img_root_str)
	dct_root = Path(dct_root_str)

	rel = img_path.relative_to(img_root)
	out_path = dct_root / rel.with_suffix(".npy")
	out_path.parent.mkdir(parents=True, exist_ok=True)

	if out_path.exists() and not overwrite:
		return "skipped", str(img_path), str(out_path), "exists"

	try:
		feat = compute_dct_feature_192(img_path)
		np.save(out_path, feat)
		return "ok", str(img_path), str(out_path), ""
	except Exception as exc:
		return "error", str(img_path), str(out_path), str(exc)


def collect_image_paths(img_root):
	"""Collect all valid image paths recursively."""
	paths = []
	for path in img_root.rglob("*"):
		if path.is_file() and path.suffix.lower() in VALID_EXT:
			paths.append(path)
	return sorted(paths)


def verify_features(dct_root):
	"""Verify all .npy files under dct_root have dim 192."""
	npy_files = sorted(dct_root.rglob("*.npy"))
	if not npy_files:
		print(f"No .npy files found under {dct_root}")
		return 1

	bad = []
	for npy_path in tqdm(npy_files, desc="Verify DCT", ncols=120):
		try:
			arr = np.load(npy_path)
			if int(np.prod(arr.shape)) != 192:
				bad.append((str(npy_path), int(np.prod(arr.shape))))
		except Exception as exc:
			bad.append((str(npy_path), f"ERR:{exc}"))

	if bad:
		print(f"Verification failed: {len(bad)} invalid files")
		for item in bad[:20]:
			print(" -", item)
		return 2

	print(f"Verification OK: {len(npy_files)} files with dim=192")
	return 0


def main():
	parser = argparse.ArgumentParser(description="Precompute 192-D DCT features from image folders")
	parser.add_argument("--img_root", type=str, required=True, help="Root folder of source images")
	parser.add_argument("--dct_root", type=str, required=True, help="Output root for .npy DCT features")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
	parser.add_argument("--verify", action="store_true", help="Verify all .npy dim=192 without recomputing")
	args = parser.parse_args()

	img_root = Path(args.img_root)
	dct_root = Path(args.dct_root)
	dct_root.mkdir(parents=True, exist_ok=True)

	if args.verify:
		return verify_features(dct_root)

	if not img_root.exists():
		print(f"Image root not found: {img_root}")
		return 1

	image_paths = collect_image_paths(img_root)
	if not image_paths:
		print(f"No valid images found under: {img_root}")
		return 1

	ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	error_log_path = dct_root / f"precompute_dct_errors_{ts}.txt"

	ok_count = 0
	skipped_count = 0
	err_count = 0

	with open(error_log_path, "w", encoding="utf-8") as err_file:
		with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
			futures = [
				executor.submit(
					process_one_image,
					str(path),
					str(img_root),
					str(dct_root),
					args.overwrite,
				)
				for path in image_paths
			]

			for future in tqdm(as_completed(futures), total=len(futures), desc="Precompute DCT", ncols=120):
				status, img_path, out_path, message = future.result()
				if status == "ok":
					ok_count += 1
				elif status == "skipped":
					skipped_count += 1
					err_file.write(f"SKIP\t{img_path}\t{out_path}\t{message}\n")
				else:
					err_count += 1
					err_file.write(f"ERR\t{img_path}\t{out_path}\t{message}\n")

	print("=" * 72)
	print("DCT PRECOMPUTE COMPLETE")
	print(f"Total images: {len(image_paths)}")
	print(f"Saved: {ok_count}")
	print(f"Skipped: {skipped_count}")
	print(f"Errors: {err_count}")
	print(f"Error log: {error_log_path}")
	print("=" * 72)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
