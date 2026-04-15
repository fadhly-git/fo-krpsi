"""Entry point inferensi batch folder untuk checkpoint hasil training baru."""

import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn
from scipy.stats import skew
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import build_backbone, build_head


device = torch.device("cpu")
torch.set_num_threads(4)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "models/checkpoints/last_checkpoint.pth"
INPUT_FOLDER = PROJECT_ROOT / "data/test"
OUTPUT_FOLDER = PROJECT_ROOT / "results/labeled_predictions"
MAX_IMAGES = None

transform = A.Compose([
	A.Resize(224, 224),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2(),
])


def extract_dct_features(img_pil):
	"""Extract 192-D DCT stats (mean/var/skew) from Y channel 8x8 blocks."""
	y = np.asarray(img_pil.convert("YCbCr"), dtype=np.float32)[:, :, 0]
	h, w = y.shape
	if h < 8 or w < 8:
		y = np.pad(y, ((0, max(8 - h, 0)), (0, max(8 - w, 0))), mode="edge")
		h, w = y.shape
	h8 = (h // 8) * 8
	w8 = (w // 8) * 8
	y = y[:h8, :w8]

	blocks = y.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
	dct_blocks = dctn(blocks, axes=(-2, -1), norm="ortho").reshape(-1, 64)

	means = dct_blocks.mean(axis=0)
	variances = dct_blocks.var(axis=0)
	skews = np.nan_to_num(skew(dct_blocks, axis=0, bias=False), nan=0.0)
	feat = np.concatenate([means, variances, skews], axis=0).astype(np.float32)

	feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
	feat_std = max(float(feat.std()), 1e-6)
	feat = (feat - float(feat.mean())) / feat_std
	feat = np.clip(feat, -1e4, 1e4)
	return torch.from_numpy(feat).unsqueeze(0)


def add_label_to_image(img_pil, label, prob_fake, prob_real):
	"""Add prediction label overlay to image."""
	img_labeled = img_pil.copy()
	draw = ImageDraw.Draw(img_labeled)

	color = (255, 50, 50) if "FAKE" in label else (50, 200, 50)
	text = f"{label}\nFake: {prob_fake:.3f} | Real: {prob_real:.3f}"

	try:
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
	except Exception:
		font = ImageFont.load_default()

	bbox = draw.textbbox((0, 0), text, font=font)
	text_width = bbox[2] - bbox[0]
	text_height = bbox[3] - bbox[1]
	x, y = 15, 15
	draw.rectangle([x - 10, y - 10, x + text_width + 10, y + text_height + 10], fill=(255, 255, 255, 230))
	draw.text((x, y), text, fill=color, font=font)
	return img_labeled


def main():
	print(f"Device: {device}")
	print(f"PyTorch version: {torch.__version__}")

	if not CHECKPOINT_PATH.exists():
		print(f"Checkpoint not found: {CHECKPOINT_PATH}")
		return 1

	ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
	dct_dim = ckpt.get("dct_dim", 192)

	backbone, feature_dim = build_backbone(log_fn=print)
	head = build_head(feature_dim, dct_dim)

	if "efficientnet_state_dict" in ckpt:
		backbone.load_state_dict(ckpt["efficientnet_state_dict"])
	elif "resnet_state_dict" in ckpt:
		backbone.load_state_dict(ckpt["resnet_state_dict"])
	elif "backbone_state_dict" in ckpt:
		backbone.load_state_dict(ckpt["backbone_state_dict"])

	if "head_state_dict" in ckpt:
		head.load_state_dict(ckpt["head_state_dict"])

	backbone.to(device).eval()
	head.to(device).eval()

	OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
	if not INPUT_FOLDER.exists():
		print(f"Input folder not found: {INPUT_FOLDER}")
		return 1

	valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
	image_paths = sorted([p for p in INPUT_FOLDER.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts])
	if MAX_IMAGES is not None:
		image_paths = image_paths[:MAX_IMAGES]
	if not image_paths:
		print("No input images found.")
		return 1

	print(f"Processing {len(image_paths)} images...")
	for image_path in tqdm(image_paths, desc="Inference", ncols=120):
		try:
			img_pil = Image.open(image_path).convert("RGB")
			img_np = np.array(img_pil)
			image_tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
			dct_tensor = extract_dct_features(img_pil).to(device)

			with torch.no_grad():
				logits = head(torch.cat([backbone(image_tensor), dct_tensor], dim=1))
				probs = torch.softmax(logits, dim=1)
				prob_fake = float(probs[0, 0].item())
				prob_real = float(probs[0, 1].item())

			label = "FAKE (AI)" if prob_fake > prob_real else "REAL (Asli)"
			labeled = add_label_to_image(img_pil, label, prob_fake, prob_real)

			out_path = OUTPUT_FOLDER / image_path.name
			labeled.save(out_path)
		except Exception as exc:
			print(f"Failed {image_path}: {exc}")

	print(f"Done. Output: {OUTPUT_FOLDER}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
