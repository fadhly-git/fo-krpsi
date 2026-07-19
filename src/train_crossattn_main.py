"""Main function training E-3: Cross-Attention Fusion.

Modifikasi dari train.py dengan 4 perubahan minimal:
1. Nama checkpoint: best_efficient_crossattn.pth / last_checkpoint_crossattn.pth
2. Inisialisasi model: build_head_cross_attention() → (fusion, head)
3. Forward pass: fusion(backbone(imgs), dcts) → head(fused)
4. Gradient clip mencakup fusion.parameters()

Semua hyperparameter, split, augmentasi, scheduler, dan early stopping
identik dengan train.py — tidak ada yang diubah untuk menjaga komparabilitas.
"""

import datetime
import json
import gc
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
	CFG,
	CHECKPOINT_DIR,
	DATA_ROOT,
	DCT_ROOT,
	LOG_DIR,
	PROJECT_ROOT,
	apply_cpu_safety_overrides,
	apply_runtime_overrides,
	resolve_device,
)
from dataset import FaceOnlyDataset, detect_dct_dim
from model import build_backbone, build_head_cross_attention
from transforms import heavy_transform, light_transform, medium_transform
from validate import save_param_debug, validate


log_path = LOG_DIR / f"train_crossattn_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_path, "a", encoding="utf-8")


def nowstr():
	return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def log(msg):
	line = f"{nowstr()} {msg}"
	try:
		tqdm.write(line)
	except Exception:
		print(line)
	log_file.write(line + "\n")
	log_file.flush()


def log_exception(exc):
	log("EXCEPTION: " + str(exc))
	tb = traceback.format_exc()
	log(tb)


def state_has_nonfinite(state_dict):
	"""Check whether state_dict tensors contain NaN/Inf values."""
	for name, value in state_dict.items():
		try:
			arr = value.detach().cpu().numpy()
		except Exception:
			continue
		if not np.isfinite(arr).all():
			return True, name
	return False, None


def get_phase_mix_probs(epoch):
	"""Return (mixup_prob, cutmix_prob) based on 3-phase augmentation schedule."""
	if epoch < CFG["aug_medium_epoch"]:
		return 0.1, 0.05
	if epoch < CFG["aug_heavy_epoch"]:
		return 0.3, 0.15
	return 0.5, 0.3


def get_aug_phase(epoch):
	"""Return augmentation phase label for the given epoch."""
	if epoch < CFG["aug_medium_epoch"]:
		return "light"
	if epoch < CFG["aug_heavy_epoch"]:
		return "medium"
	return "heavy"


def main():
	print(f"Logging to: {log_path}")

	device, force_cpu = resolve_device()
	apply_cpu_safety_overrides(device)
	apply_runtime_overrides(log_fn=log)
	log(f"DEVICE: {device} (torch {torch.__version__}) FORCE_CPU={force_cpu}")
	log(f"FUSION_MODE: {CFG.get('fusion_mode', 'concat')}")

	# E-3 selalu membutuhkan DCT
	use_dct = True
	CFG["use_dct"] = True
	log("DCT mode: enabled (required for cross-attention fusion)")

	# Checkpoint names: TIDAK menimpa E-1 (best_efficient_dct.pth) atau E-2
	last_ckpt_name = "last_checkpoint_crossattn.pth"
	best_ckpt_name = "best_efficient_crossattn.pth"
	log(f"Checkpoint names: best={best_ckpt_name}, last={last_ckpt_name}")

	dct_dim = detect_dct_dim(DCT_ROOT)
	if dct_dim is None:
		log("No precomputed DCT found — proceeding with dct_dim=0 (zeros will be used)")
		dct_dim = 0
	else:
		log(f"Detected DCT dim (early) = {dct_dim}")

	log("Building dataset index (no file loads yet)...")
	full_dataset = FaceOnlyDataset(
		DATA_ROOT,
		dct_root=DCT_ROOT,
		transform=light_transform,
		max_fake=CFG.get("max_subset_images"),
		max_real=CFG.get("max_subset_beauty_images"),
		dct_dim=dct_dim,
		use_dct=use_dct,
		log_fn=log,
	)
	n_total = len(full_dataset)
	log(f"Total samples (valid) found: {n_total}")

	if n_total == 0:
		log("ERROR: No valid samples found. Check raw image folders and DCT precompute output.")
		log(f"Checked IMG roots: {DATA_ROOT}")
		log(f"Checked DCT roots: {DCT_ROOT}")
		return 1

	labels_all = [label for (_, _, label) in full_dataset.samples]
	count_real = labels_all.count(0)
	count_fake = labels_all.count(1)
	log(f"Class counts (valid samples): REAL={count_real}, FAKE={count_fake}")
	log(f"Using DCT dim = {dct_dim}")

	# Ref: King & Zeng 2001; Aurelio et al. 2019 (weighted cross-entropy)
	class_weights = torch.tensor(
		[
			n_total / (2.0 * max(count_real, 1)),
			n_total / (2.0 * max(count_fake, 1)),
		],
		dtype=torch.float32,
	)
	log(f"Class weights (CrossEntropy): REAL={class_weights[0].item():.6f}, FAKE={class_weights[1].item():.6f}")

	# E-3: backbone tetap sama, head = CrossAttentionFusion + Linear(1280, 2)
	backbone, feature_dim = build_backbone(log_fn=log)
	backbone = backbone.to(device)

	fusion, head = build_head_cross_attention(
		feature_dim=feature_dim,
		dct_dim=dct_dim if dct_dim > 0 else 192,  # fallback jika DCT belum ada
		attn_dim=64,
		n_dct_tokens=3,
	)
	fusion = fusion.to(device)
	head = head.to(device)

	log(
		f"Model E-3 — backbone feature_dim={feature_dim}, "
		f"dct_dim={dct_dim}, attn_dim=64, n_dct_tokens=3"
	)
	n_fusion = sum(p.numel() for p in fusion.parameters())
	n_head = sum(p.numel() for p in head.parameters())
	n_backbone = sum(p.numel() for p in backbone.parameters())
	log(f"Parameter count — backbone: {n_backbone:,}, fusion: {n_fusion:,}, head: {n_head:,}")
	log(f"Parameter count — CrossAttentionFusion+head total: {n_fusion + n_head:,}")

	for name, param in backbone.named_parameters():
		if not param.requires_grad:
			log(f"[WARN] parameter frozen: {name}")
	for param in backbone.parameters():
		param.requires_grad = True

	# Optimizer: 3 param groups (backbone, fusion, head)
	optimizer = optim.AdamW(
		[
			{"params": backbone.parameters(), "lr": CFG["lr_backbone"]},
			{"params": fusion.parameters(), "lr": CFG["lr_head"]},
			{"params": head.parameters(), "lr": CFG["lr_head"]},
		],
		weight_decay=1e-2,
	)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-7)
	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

	# Ref: Kohavi, IJCAI 1995 (stratified split); 3-way: train / val / test
	indices = np.arange(n_total)
	test_ratio = float(CFG.get("test_ratio", 0.1))
	test_size = max(1, int(round(n_total * test_ratio)))

	train_val_indices, test_indices_arr = train_test_split(
		indices,
		test_size=test_size,
		random_state=CFG["seed"],
		stratify=labels_all,
		shuffle=True,
	)
	train_val_labels = [labels_all[i] for i in train_val_indices]
	val_ratio = float(CFG.get("val_ratio", 0.2))
	val_ratio = min(max(val_ratio, 0.01), 0.9)
	val_size = max(1, int(round(len(train_val_indices) * val_ratio)))
	val_size = min(val_size, len(train_val_indices) - 1)

	train_indices, val_indices = train_test_split(
		train_val_indices,
		test_size=val_size,
		random_state=CFG["seed"],
		stratify=train_val_labels,
		shuffle=True,
	)
	train_indices = train_indices.tolist()
	val_indices = val_indices.tolist()
	test_indices_list = test_indices_arr.tolist()

	# Persist test indices (reuse split yang sama dengan E-1/E-2 via seed)
	test_indices_path = PROJECT_ROOT / "data/processed/test_indices.json"
	test_indices_path.parent.mkdir(parents=True, exist_ok=True)
	with open(test_indices_path, "w", encoding="utf-8") as _f:
		json.dump(test_indices_list, _f)

	log(f"Split sizes — train: {len(train_indices)}  val: {len(val_indices)}  test: {len(test_indices_list)}")
	train_labels = [full_dataset.samples[idx][2] for idx in train_indices]
	val_labels = [full_dataset.samples[idx][2] for idx in val_indices]
	test_labels = [full_dataset.samples[idx][2] for idx in test_indices_list]
	log(f"Stratified dist train — REAL={train_labels.count(0)}, FAKE={train_labels.count(1)}")
	log(f"Stratified dist val   — REAL={val_labels.count(0)}, FAKE={val_labels.count(1)}")
	log(f"Stratified dist test  — REAL={test_labels.count(0)}, FAKE={test_labels.count(1)}")

	train_subset = Subset(full_dataset, train_indices)
	val_subset = Subset(full_dataset, val_indices)

	pin_memory = device.type == "cuda"
	train_loader = DataLoader(
		train_subset,
		batch_size=CFG["batch_size"],
		shuffle=True,
		num_workers=CFG["num_workers"],
		pin_memory=pin_memory,
	)
	val_loader = DataLoader(
		val_subset,
		batch_size=CFG["batch_size"],
		shuffle=False,
		num_workers=CFG["num_workers"],
		pin_memory=pin_memory,
	)

	def forward_pass(imgs, dcts):
		"""Forward pass E-3: backbone → fusion(spatial, dct) → head."""
		feats = backbone(imgs)
		if dct_dim > 0:
			fused = fusion(feats, dcts)
		else:
			# Fallback: dct zeros, attention trivial tapi tidak error
			fused = feats
		return head(fused), feats

	try:
		imgs, dcts, _ = next(iter(train_loader))
		if torch.is_tensor(imgs):
			log(
				"DATA DEBUG — image batch mean="
				f"{imgs.mean().item():.4f} std={imgs.std().item():.4f} "
				f"min={imgs.min().item():.4f} max={imgs.max().item():.4f}"
			)
		if torch.is_tensor(dcts) and dct_dim > 0:
			dmean = dcts.view(dcts.size(0), -1).mean(1).mean().item()
			dstd = dcts.view(dcts.size(0), -1).std(1).mean().item()
			log(f"DATA DEBUG — dct batch mean(mean)={dmean:.4f} mean(std)={dstd:.4f}")
	except Exception as exc:
		log("DATA DEBUG failed:")
		log_exception(exc)

	best_acc = 0.0
	best_auc = -1.0
	best_macro_f1 = float("nan")
	phase_best_auc = -1.0
	epochs_no_improve = 0
	start_epoch = 1
	current_aug_phase = None

	last_ckpt_path = CHECKPOINT_DIR / last_ckpt_name
	if last_ckpt_path.exists():
		log("Last checkpoint found. Loading for resume...")
		try:
			ckpt = torch.load(last_ckpt_path, map_location=device)
			ckpt_backbone = ckpt.get("efficientnet_state_dict", ckpt.get("resnet_state_dict"))
			ckpt_fusion = ckpt.get("fusion_state_dict")
			ckpt_head = ckpt.get("head_state_dict")
			ckpt_opt = ckpt.get("optimizer_state_dict")

			corrupted = False
			corrupt_name = None
			for sd in [ckpt_backbone, ckpt_fusion, ckpt_head]:
				if sd is not None and not corrupted:
					corrupted, corrupt_name = state_has_nonfinite(sd)

			if corrupted:
				log(f"Checkpoint contains non-finite parameter: {corrupt_name}. Starting fresh.")
			else:
				if ckpt_backbone is None:
					raise KeyError("Checkpoint missing backbone state_dict")
				backbone.load_state_dict(ckpt_backbone)
				if ckpt_fusion is not None:
					fusion.load_state_dict(ckpt_fusion)
				if ckpt_head is not None:
					head.load_state_dict(ckpt_head)
				if ckpt_opt is not None:
					try:
						optimizer.load_state_dict(ckpt_opt)
						for g in optimizer.param_groups:
							g["lr"] = CFG["lr_head"]
						optimizer.param_groups[0]["lr"] = CFG["lr_backbone"]
					except Exception:
						log("Warning: unable to load optimizer state. Starting optimizer fresh.")

				start_epoch = ckpt.get("epoch", 0) + 1
				best_acc = ckpt.get("best_acc", 0.0)
				best_auc = ckpt.get("best_auc", -1.0)
				phase_best_auc = ckpt.get("phase_best_auc", -1.0) or -1.0
				epochs_no_improve = ckpt.get("epochs_no_improve", 0)
				current_aug_phase = ckpt.get("current_aug_phase", None)
				ckpt_scheduler = ckpt.get("scheduler_state_dict")
				if ckpt_scheduler is not None:
					try:
						scheduler.load_state_dict(ckpt_scheduler)
						log("Scheduler state restored from checkpoint.")
					except Exception:
						log("Warning: unable to load scheduler state.")
				log(
					f"Resumed from epoch {ckpt.get('epoch', 0)} "
					f"(best_auc={best_auc:.4f}, aug_phase={current_aug_phase})"
				)
		except Exception as exc:
			log("Failed to load last checkpoint — training will start fresh.")
			log_exception(exc)
	else:
		log("No resume checkpoint found. Starting fresh training.")

	try:
		for epoch in range(start_epoch, CFG["epochs"] + 1):
			new_aug_phase = get_aug_phase(epoch)
			if epoch >= CFG["aug_heavy_epoch"] and full_dataset.transform != heavy_transform:
				log("=" * 80)
				log(f"EPOCH {epoch}: Switching to HEAVY augmentation (strong regularization)")
				log("=" * 80)
				full_dataset.transform = heavy_transform
			elif CFG["aug_medium_epoch"] <= epoch < CFG["aug_heavy_epoch"] and full_dataset.transform != medium_transform:
				log("=" * 80)
				log(f"EPOCH {epoch}: Switching to MEDIUM augmentation (smooth transition)")
				log("=" * 80)
				full_dataset.transform = medium_transform
			elif epoch < CFG["aug_medium_epoch"] and full_dataset.transform != light_transform:
				log(f"EPOCH {epoch}: Using LIGHT augmentation (fast training)")
				full_dataset.transform = light_transform

			if current_aug_phase is None:
				current_aug_phase = new_aug_phase
				phase_best_auc = -1.0
			elif new_aug_phase != current_aug_phase:
				epochs_no_improve = 0
				phase_best_auc = -1.0
				lr_back_reset = CFG.get("lr_phase_reset_backbone", CFG["lr_backbone"] * 0.5)
				lr_head_reset = CFG.get("lr_phase_reset_head", CFG["lr_head"] * 0.4)
				optimizer.param_groups[0]["lr"] = lr_back_reset
				optimizer.param_groups[1]["lr"] = lr_head_reset
				optimizer.param_groups[2]["lr"] = lr_head_reset
				remaining_epochs = max(CFG["epochs"] - epoch + 1, 1)
				scheduler = optim.lr_scheduler.CosineAnnealingLR(
					optimizer, T_max=remaining_epochs, eta_min=1e-7
				)
				log(
					f"AUG PHASE changed {current_aug_phase} -> {new_aug_phase}; "
					f"early-stopping counter reset; "
					f"LR reset: backbone={lr_back_reset:.2e}, fusion/head={lr_head_reset:.2e}"
				)
				current_aug_phase = new_aug_phase

			gc.collect()
			if device.type == "cuda":
				try:
					torch.cuda.empty_cache()
				except Exception:
					pass

			mixup_prob, cutmix_prob = get_phase_mix_probs(epoch)

			backbone.train()
			fusion.train()
			head.train()
			running_loss = 0.0
			running_correct = 0
			running_total = 0

			pbar = tqdm(
				train_loader,
				desc=f"Epoch {epoch}/{CFG['epochs']}",
				dynamic_ncols=True,
				leave=False,
				ascii=True,
				bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
			)
			for batch_idx, (imgs, dcts, labels) in enumerate(pbar):
				imgs = imgs.to(device)
				dcts = dcts.to(device)
				labels = labels.to(device)

				if not torch.isfinite(imgs).all():
					log("[ERROR] Non-finite values detected in image batch")
					raise RuntimeError("Non-finite image inputs")
				if not torch.isfinite(dcts).all():
					log("[ERROR] Non-finite values detected in DCT batch")
					raise RuntimeError("Non-finite dct inputs")

				if dcts.dim() > 2:
					dcts = dcts.view(dcts.size(0), -1)

				if dct_dim > 0 and dcts.size(1) != dct_dim:
					raise RuntimeError(f"DCT dim mismatch: got {dcts.size(1)} expected {dct_dim}")

				use_mixup = np.random.rand() < mixup_prob
				use_cutmix = (not use_mixup) and (np.random.rand() < cutmix_prob) and (imgs.size(0) > 1)

				# Ref: Zhang et al., ICLR 2018 (alpha in [0.1, 0.4], use 0.4)
				if use_mixup:
					lam = float(np.random.beta(0.4, 0.4))
					idx = torch.randperm(imgs.size(0)).to(device)
					imgs_mix = lam * imgs + (1.0 - lam) * imgs[idx]
					labels_a, labels_b = labels, labels[idx]
					logits, _ = forward_pass(imgs_mix, dcts)
					loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
				elif use_cutmix:
					lam = float(np.random.beta(0.4, 0.4))
					idx = torch.randperm(imgs.size(0)).to(device)
					height, width = imgs.shape[2], imgs.shape[3]
					cut_ratio = np.sqrt(1.0 - lam)
					cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
					cy, cx = np.random.randint(0, height), np.random.randint(0, width)
					y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
					x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)
					imgs_cut = imgs.clone()
					imgs_cut[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
					lam_adj = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
					labels_a, labels_b = labels, labels[idx]
					logits, _ = forward_pass(imgs_cut, dcts)
					loss = lam_adj * criterion(logits, labels_a) + (1.0 - lam_adj) * criterion(logits, labels_b)
				else:
					logits, feats = forward_pass(imgs, dcts)
					loss = criterion(logits, labels)

				if not torch.isfinite(logits).all():
					log("[ERROR] Non-finite logits detected")
					raise RuntimeError("Non-finite logits")

				if (not isinstance(loss, torch.Tensor)) or (not torch.isfinite(loss).all()):
					log(f"[ERROR] Non-finite loss detected: {loss}")
					debug_ckpt = {
						"epoch": epoch,
						"efficientnet_state_dict": backbone.state_dict(),
						"fusion_state_dict": fusion.state_dict(),
						"head_state_dict": head.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
					}
					torch.save(debug_ckpt, CHECKPOINT_DIR / f"debug_nan_crossattn_epoch{epoch}.pth")
					raise RuntimeError("Non-finite loss encountered")

				optimizer.zero_grad()
				loss.backward()
				# Clip gradients mencakup semua modul E-3
				all_params = (
					list(backbone.parameters())
					+ list(fusion.parameters())
					+ list(head.parameters())
				)
				torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
				optimizer.step()

				bs = imgs.size(0)
				running_loss += loss.item() * bs
				preds = logits.argmax(dim=1)
				running_correct += (preds == labels).sum().item()
				running_total += bs

				train_loss = running_loss / running_total
				train_acc_epoch = running_correct / running_total
				lr_back = optimizer.param_groups[0].get("lr", 0.0)
				lr_head = optimizer.param_groups[2].get("lr", 0.0)
				if (batch_idx % 10) == 0 or (batch_idx + 1) == len(train_loader):
					pbar.set_postfix_str(
						f"loss={train_loss:.4f} acc={train_acc_epoch:.4f} lr={lr_back:.1e}/{lr_head:.1e}",
						refresh=False,
					)

				del imgs, dcts, labels, logits, loss, preds
				if not use_mixup and not use_cutmix:
					del feats
				if (batch_idx % 10) == 0:
					gc.collect()

			log(f"[EPOCH {epoch}] TRAIN loss={train_loss:.4f} acc={train_acc_epoch:.4f}")
			del pbar
			gc.collect()

			# Validasi — validate() hanya menerima (backbone, head) tapi E-3 perlu fusion.
			# Bungkus fusion+head menjadi satu modul sementara untuk validate().
			class _FusionHead(nn.Module):
				def __init__(self, _fusion, _head):
					super().__init__()
					self._fusion = _fusion
					self._head = _head

				def forward(self, x):
					# x di validate() adalah torch.cat([feats, dcts], dim=1) — intercept tidak bisa.
					# Oleh karena itu validate() di-call dengan custom wrapper di bawah.
					raise NotImplementedError("Gunakan _validate_e3()")

			# Custom validation loop untuk E-3 (inline, identik dengan validate() tapi
			# memanggil fusion sebelum head)
			@torch.no_grad()
			def _validate_e3(loader, max_batches=None):
				backbone.eval()
				fusion.eval()
				head.eval()
				_preds, _probs, _targets = [], [], []
				for i, (imgs_v, dcts_v, labels_v) in enumerate(loader):
					if max_batches is not None and i >= max_batches:
						break
					imgs_v = imgs_v.to(device)
					dcts_v = dcts_v.to(device)
					labels_v = labels_v.to(device)
					if dcts_v.dim() > 2:
						dcts_v = dcts_v.view(dcts_v.size(0), -1)
					feats_v = backbone(imgs_v)
					if dct_dim > 0:
						fused_v = fusion(feats_v, dcts_v)
					else:
						fused_v = feats_v
					logits_v = head(fused_v)
					if torch.isnan(logits_v).any() or torch.isinf(logits_v).any():
						raise RuntimeError("NaN/Inf detected in logits during validation")
					prob_v = torch.softmax(logits_v, dim=1)[:, 1]
					_preds.extend(logits_v.argmax(1).cpu().numpy().tolist())
					_probs.extend(prob_v.cpu().numpy().tolist())
					_targets.extend(labels_v.cpu().numpy().tolist())
					del imgs_v, dcts_v, labels_v, feats_v, fused_v, logits_v, prob_v
					if (i % 10) == 0:
						gc.collect()
				if not _targets:
					return float("nan"), float("nan"), [float("nan")]*2, [float("nan")]*2, [float("nan")]*2, [[0,0],[0,0]]
				from sklearn.metrics import (
					accuracy_score, roc_auc_score, precision_score,
					recall_score, f1_score, confusion_matrix,
				)
				acc_v = accuracy_score(_targets, _preds)
				try:
					auc_v = roc_auc_score(_targets, _probs)
				except Exception:
					auc_v = float("nan")
				prec_v = precision_score(_targets, _preds, labels=[0, 1], average=None, zero_division=0)
				rec_v = recall_score(_targets, _preds, labels=[0, 1], average=None, zero_division=0)
				f1_v = f1_score(_targets, _preds, labels=[0, 1], average=None, zero_division=0)
				cm_v = confusion_matrix(_targets, _preds, labels=[0, 1])
				return acc_v, auc_v, prec_v, rec_v, f1_v, cm_v

			try:
				max_train_batches = CFG.get("validate_train_max_batches", None)
				train_acc, train_auc, train_prec, train_rec, train_f1, train_cm = _validate_e3(
					train_loader, max_batches=max_train_batches
				)
			except Exception as exc:
				log(f"Train-set validate() failed: {exc}")
				train_acc, train_auc, train_prec, train_rec, train_f1, train_cm = (float("nan"),) * 2 + ([float("nan"), float("nan")],) * 3

			gc.collect()

			try:
				acc, auc, prec, rec, f1, cm = _validate_e3(val_loader)
			except Exception as exc:
				log(f"Val validate() failed: {exc}")
				acc, auc, prec, rec, f1, cm = (float("nan"),) * 2 + ([float("nan"), float("nan")],) * 3

			gc.collect()

			macro_f1_val = float(np.mean(f1)) if np.isfinite(np.array(f1, dtype=float)).all() else float("nan")
			try:
				log(f"EPOCH {epoch:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} Train AUC: {train_auc:.4f} | Val Acc: {acc:.5f} | Val AUC: {auc:.5f}")
				log(f"  Train Conf: {train_cm.tolist()} | Val Conf: {cm.tolist()}")
				log(f"  Val Prec_fake: {prec[0]:.4f} Prec_real: {prec[1]:.4f} | Val Rec_fake: {rec[0]:.4f} Rec_real: {rec[1]:.4f}")
				log(f"  Val F1_fake: {f1[0]:.4f} F1_real: {f1[1]:.4f} Macro F1: {macro_f1_val:.4f}")
			except Exception:
				log(f"[EPOCH {epoch}] TRAIN loss={train_loss:.4f} acc={train_acc} val_acc={acc}")

			# Simpan checkpoint terakhir (tiap epoch)
			last_ckpt = {
				"epoch": epoch,
				"best_acc": best_acc,
				"best_auc": best_auc,
				"phase_best_auc": phase_best_auc,
				"epochs_no_improve": epochs_no_improve,
				"current_aug_phase": current_aug_phase,
				"efficientnet_state_dict": backbone.state_dict(),
				"fusion_state_dict": fusion.state_dict(),
				"head_state_dict": head.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"dct_dim": dct_dim,
				"fusion_mode": "cross_attention",
			}
			torch.save(last_ckpt, CHECKPOINT_DIR / last_ckpt_name)
			log(f"LAST CHECKPOINT saved at epoch {epoch}")

			debug_every = CFG.get("debug_every_n_epochs", 0)
			massive_flag = CFG.get("debug_massive", False) or (os.environ.get("DEBUG_MASSIVE", "0") == "1")
			if debug_every and (epoch % debug_every == 0):
				# save_param_debug menerima (model, head) — kirim fusion sebagai pengganti model
				save_param_debug(
					fusion,
					head,
					epoch,
					CHECKPOINT_DIR,
					log_fn=log,
					bins=CFG.get("debug_hist_bins", 64),
					massive=massive_flag,
				)

			# Simpan best checkpoint hanya berdasarkan AUC validasi tertinggi
			# (BUKAN last checkpoint — sesuai catatan bug CKPT_E3)
			if np.isfinite(auc) and (auc > best_auc):
				best_acc = acc
				best_auc = auc
				best_macro_f1 = macro_f1_val
				epochs_no_improve = 0
				best_ckpt = {
					"efficientnet_state_dict": backbone.state_dict(),
					"fusion_state_dict": fusion.state_dict(),
					"head_state_dict": head.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"epoch": epoch,
					"best_acc": best_acc,
					"best_auc": best_auc,
					"best_macro_f1": best_macro_f1,
					"phase_best_auc": phase_best_auc,
					"epochs_no_improve": epochs_no_improve,
					"current_aug_phase": current_aug_phase,
					"dct_dim": dct_dim,
					"fusion_mode": "cross_attention",
				}
				torch.save(best_ckpt, CHECKPOINT_DIR / best_ckpt_name)
				log(f"NEW BEST saved: auc={best_auc:.4f} acc={best_acc:.4f} macro_f1={best_macro_f1:.4f} epoch={epoch}")
			elif (not np.isfinite(auc)) and (not (CHECKPOINT_DIR / best_ckpt_name).exists()):
				best_acc = acc if np.isfinite(acc) else best_acc
				bootstrap_ckpt = {
					"efficientnet_state_dict": backbone.state_dict(),
					"fusion_state_dict": fusion.state_dict(),
					"head_state_dict": head.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"epoch": epoch,
					"best_acc": best_acc,
					"best_auc": best_auc,
					"phase_best_auc": phase_best_auc,
					"epochs_no_improve": epochs_no_improve,
					"current_aug_phase": current_aug_phase,
					"dct_dim": dct_dim,
					"fusion_mode": "cross_attention",
				}
				torch.save(bootstrap_ckpt, CHECKPOINT_DIR / best_ckpt_name)
				log(f"NEW BEST bootstrap saved (AUC unavailable): acc={best_acc:.4f} epoch={epoch}")

			if np.isfinite(auc) and (auc > phase_best_auc):
				phase_best_auc = auc
				epochs_no_improve = 0
			else:
				epochs_no_improve += 1

			# Ref: Prechelt, 1998 (early stopping by validation performance plateau)
			if epochs_no_improve >= CFG["early_stopping_patience"] and epoch >= CFG["min_epoch_before_stop"]:
				log(
					f"EARLY STOPPING triggered at epoch {epoch}. "
					f"Phase Best AUC: {phase_best_auc:.4f} | Global Best AUC: {best_auc:.4f}"
				)
				break

			gc.collect()
			if device.type == "cuda":
				try:
					torch.cuda.empty_cache()
				except Exception:
					pass

			scheduler.step()
			current_lr_backbone = optimizer.param_groups[0]["lr"]
			current_lr_head = optimizer.param_groups[2]["lr"]
			log(f"LR updated: backbone={current_lr_backbone:.8f}, fusion/head={current_lr_head:.8f}")
	except Exception as exc:
		log_exception(exc)
	finally:
		log(f"Training finished. Best auc = {best_auc:.4f} Best acc = {best_acc:.4f}")
		log_file.close()
		print("Log file saved to:", log_path)

	return 0


if __name__ == "__main__":
	sys.exit(main())
