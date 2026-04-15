"""Definisi transformasi augmentasi light, medium, dan heavy untuk training."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG


light_transform = A.Compose([
	A.RandomResizedCrop(CFG["img_size"], CFG["img_size"], scale=(0.9, 1.0)),
	A.HorizontalFlip(p=0.5),
	A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2(),
])

# Ref: Yang et al., 2025 (medium phase to reduce augmentation shock)
medium_transform = A.Compose([
	A.RandomResizedCrop(CFG["img_size"], CFG["img_size"], scale=(0.85, 1.0)),
	A.HorizontalFlip(p=0.5),
	A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
	A.GaussNoise(var_limit=(20, 80), p=0.3),
	A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2(),
])

heavy_transform = A.Compose([
	A.RandomResizedCrop(CFG["img_size"], CFG["img_size"], scale=(0.8, 1.0)),
	A.OneOf([
		A.CoarseDropout(max_holes=1, max_height=min(180, CFG["img_size"]), max_width=min(180, CFG["img_size"]), p=0.7),
		A.CoarseDropout(max_holes=1, max_height=min(100, CFG["img_size"]), max_width=CFG["img_size"], p=0.3),
		A.GridDropout(ratio=0.4, p=0.3),
	], p=0.8),
	A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
	A.GaussNoise(var_limit=(50, 200), p=0.6),
	A.ImageCompression(quality_lower=30, quality_upper=95, p=0.5),
	A.HorizontalFlip(p=0.5),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2(),
])
