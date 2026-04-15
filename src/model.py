"""Factory model untuk backbone EfficientNet dan head klasifikasi."""

import torch.nn as nn
from torchvision import models


def build_backbone(log_fn=None):
	"""Build pretrained EfficientNet backbone and return (model, feature_dim)."""
	feature_dim = None

	# Ref: Tan & Le, ICML 2019 (prefer EfficientNet-B0 for 224 inputs)
	try:
		try:
			weights = models.EfficientNet_B0_Weights.DEFAULT
			backbone = models.efficientnet_b0(weights=weights)
			if log_fn is not None:
				log_fn("Using EfficientNet-B0 pretrained weights")
		except Exception:
			try:
				backbone = models.efficientnet_b0(pretrained=True)
				if log_fn is not None:
					log_fn("Using legacy EfficientNet-B0 pretrained=True")
			except Exception:
				raise

		cls = getattr(backbone, "classifier", None)
		if cls is None:
			feature_dim = 1280
		elif isinstance(cls, nn.Sequential):
			linear = None
			for module in reversed(cls):
				if isinstance(module, nn.Linear):
					linear = module
					break
			feature_dim = linear.in_features if linear is not None else 1280
		elif isinstance(cls, nn.Linear):
			feature_dim = cls.in_features
		else:
			feature_dim = getattr(cls, "in_features", 1280)

		backbone.classifier = nn.Identity()
		return backbone, int(feature_dim)
	except Exception as exc:
		if log_fn is not None:
			log_fn(f"Failed to instantiate EfficientNet-B0: {exc}. Falling back to ResNet50.")

		try:
			weights = models.ResNet50_Weights.DEFAULT
			backbone = models.resnet50(weights=weights)
		except Exception:
			backbone = models.resnet50(pretrained=True)
		backbone.fc = nn.Identity()
		return backbone, 2048


def build_head(feature_dim, dct_dim):
	"""Build classification head from concatenated (image features + DCT features)."""
	# Ref: Srivastava et al., JMLR 2014 (dropout regularization)
	return nn.Sequential(
		nn.Dropout(p=0.3),
		nn.Linear(int(feature_dim) + int(dct_dim), 2),
	)
