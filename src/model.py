"""Factory model untuk backbone EfficientNet dan head klasifikasi."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# E-3: Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
	"""Single-head cross-attention fusion: Q dari fitur spasial, K/V dari token DCT.

	Fitur DCT (dct_dim) dipecah menjadi n_dct_tokens token seimbang, masing-masing
	berukuran token_dim = dct_dim // n_dct_tokens (misalnya 192 // 3 = 64).
	Sesuai struktur statistik per-segmen Bab III:
	  token 0 → mean     (indeks 0..63)
	  token 1 → variance (indeks 64..127)
	  token 2 → skewness (indeks 128..191)

	Forward:
	  spatial_feat [B, feature_dim]  → Q ∈ R^{B×1×d}
	  dct_feat     [B, dct_dim]      → K,V ∈ R^{B×n_tokens×d}
	  Attention(Q,K,V) = softmax(QK^T / sqrt(d)) · V
	  output = spatial_feat + proj_out(attention_output)  [B, feature_dim]

	Refs:
	  - Chen et al. (CrossViT, ICCV 2021, arXiv:2103.14899):
	      CLS token sebagai single-token Query; K/V dari cabang lain.
	  - Qiao et al. (HCMA, arXiv:2504.17223 §3.3.2):
	      Q = S'·W_Q (spasial); K = F'·W_K, V = F'·W_V (frekuensi);
	      residual connection ke S.
	  - Sen & Mukherjee (CSAF, arXiv:2601.03382 §III-E):
	      H = CSAF(S,F) + S; residual ke fitur spasial.
	  - Lv et al. (SFMFNet TSCA, arXiv:2508.20449 §3.3):
	      token selection lalu cross-attention + residual learnable α.
	  - Khan et al. (CAMME, arXiv:2505.18035 §4.3):
	      tiap statistik modalitas sebagai token independen.
	"""

	def __init__(self, feature_dim: int = 1280, dct_dim: int = 192,
				 attn_dim: int = 64, n_dct_tokens: int = 3):
		super().__init__()
		if dct_dim % n_dct_tokens != 0:
			raise ValueError(
				f"dct_dim ({dct_dim}) harus habis dibagi n_dct_tokens ({n_dct_tokens})"
			)
		self.n_dct_tokens = n_dct_tokens
		self.token_dim = dct_dim // n_dct_tokens  # e.g. 64
		self.attn_dim = attn_dim                   # d = 64
		self.scale = attn_dim ** -0.5

		# Q: proyeksikan fitur spasial → R^d
		# Ref: Qiao et al. §3.3.2: Q = S'·W_Q
		self.proj_q = nn.Linear(feature_dim, attn_dim)

		# K, V: satu Linear independen per token DCT
		# Ref: CAMME §4.2 — tiap statistik modalitas punya proyeksi sendiri
		self.proj_k = nn.ModuleList(
			[nn.Linear(self.token_dim, attn_dim) for _ in range(n_dct_tokens)]
		)
		self.proj_v = nn.ModuleList(
			[nn.Linear(self.token_dim, attn_dim) for _ in range(n_dct_tokens)]
		)

		# Proyeksikan output attention kembali ke feature_dim (untuk residual)
		# Ref: CSAF §III-E: output dimensi harus kompatibel dengan spatial_feat
		self.proj_out = nn.Linear(attn_dim, feature_dim)

	def forward(self, spatial_feat: torch.Tensor, dct_feat: torch.Tensor) -> torch.Tensor:
		"""Forward pass cross-attention fusion.

		Args:
			spatial_feat: Tensor [B, feature_dim]  — output backbone EfficientNet
			dct_feat:     Tensor [B, dct_dim]       — fitur DCT pre-computed

		Returns:
			Tensor [B, feature_dim]  — spatial_feat + attention residual
		"""
		# --- Query (dari fitur spasial) ---
		q = self.proj_q(spatial_feat)  # [B, attn_dim]
		q = q.unsqueeze(1)             # [B, 1, attn_dim] — 1 query token

		# --- Key & Value (dari token DCT) ---
		# Pecah fitur DCT menjadi n_dct_tokens segmen: [B, token_dim] × n
		tokens = dct_feat.split(self.token_dim, dim=-1)  # tuple of [B, token_dim]

		k = torch.stack(
			[self.proj_k[i](tokens[i]) for i in range(self.n_dct_tokens)], dim=1
		)  # [B, n_dct_tokens, attn_dim]
		v = torch.stack(
			[self.proj_v[i](tokens[i]) for i in range(self.n_dct_tokens)], dim=1
		)  # [B, n_dct_tokens, attn_dim]

		# --- Scaled dot-product attention (single-head) ---
		# Ref: Vaswani et al. (2017); konsensus kelima paper
		scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
		# scores: [B, 1, n_dct_tokens]
		attn_weights = F.softmax(scores, dim=-1)

		# attn_out: [B, 1, attn_dim] → squeeze → [B, attn_dim]
		attn_out = torch.matmul(attn_weights, v).squeeze(1)

		# --- Residual connection ke fitur spasial ---
		# Ref: CSAF: H = CSAF(S,F) + S; Qiao et al. §3.3.2 eq.15
		out = self.proj_out(attn_out)  # [B, feature_dim]
		return out + spatial_feat      # residual: [B, feature_dim]


def build_head_cross_attention(
	feature_dim: int = 1280,
	dct_dim: int = 192,
	attn_dim: int = 64,
	n_dct_tokens: int = 3,
):
	"""Buat CrossAttentionFusion + classification head untuk E-3.

	Head (Dropout p=0.3 + Linear(feature_dim, 2)) identik dengan E-2 —
	input head adalah feature_dim (bukan feature_dim+dct_dim) karena
	cross-attention sudah mengintegrasikan DCT via residual dan menghasilkan
	vektor feature_dim-dim. Ini memastikan perbandingan fusi fair dengan E-1/E-2.

	Returns:
		fusion (CrossAttentionFusion): modul fusi cross-attention terpisah
		head (nn.Sequential): Dropout + Linear ke 2 kelas
	"""
	fusion = CrossAttentionFusion(
		feature_dim=feature_dim,
		dct_dim=dct_dim,
		attn_dim=attn_dim,
		n_dct_tokens=n_dct_tokens,
	)
	head = nn.Sequential(
		nn.Dropout(p=0.3),
		nn.Linear(feature_dim, 2),
	)
	return fusion, head
