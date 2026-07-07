"""Entry point training pipeline deepfake detector berbasis image + DCT."""

import datetime
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
	apply_cpu_safety_overrides,
	apply_runtime_overrides,
	resolve_device,
)
from dataset import FaceOnlyDataset, detect_dct_dim
from model import build_backbone, build_head
from transforms import heavy_transform, light_transform, medium_transform
from validate import save_param_debug, validate


log_path = LOG_DIR / f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
	use_dct = bool(CFG.get("use_dct", True))
	log(f"DCT mode: {'enabled' if use_dct else 'disabled'}")
	last_ckpt_name = "last_checkpoint.pth" if use_dct else "latest_no_dct.pth"
	best_ckpt_name = "best_efficient_dct.pth" if use_dct else "best_efficient_no_dct.pth"

	if CFG["max_subset_images"] is not None or CFG["max_subset_beauty_images"] is not None:
		log(
			f"Applied per-root caps: subset={CFG['max_subset_images']} "
			f"subset_beauty={CFG['max_subset_beauty_images']}"
		)

	if use_dct:
		dct_dim = detect_dct_dim(DCT_ROOT)
		if dct_dim is None:
			log("No precomputed DCT found — proceeding with dct_dim=0 (zeros will be used)")
			dct_dim = 0
		else:
			log(f"Detected DCT dim (early) = {dct_dim}")
	else:
		dct_dim = 0
		log("DCT disabled by config/env (USE_DCT=0). Training image-only model.")

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

	backbone, feature_dim = build_backbone(log_fn=log)
	backbone = backbone.to(device)
	head = build_head(feature_dim, dct_dim).to(device)

	for name, param in backbone.named_parameters():
		if not param.requires_grad:
			log(f"[WARN] parameter frozen: {name}")
	for param in backbone.parameters():
		param.requires_grad = True

	optimizer = optim.AdamW(
		[
			{"params": backbone.parameters(), "lr": CFG["lr_backbone"]},
			{"params": head.parameters(), "lr": CFG["lr_head"]},
		],
		weight_decay=1e-2,
	)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-7)
	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

	val_ratio = float(CFG.get("val_ratio", 0.2))
	val_ratio = min(max(val_ratio, 0.01), 0.9)
	val_size = max(1, int(round(n_total * val_ratio)))
	val_size = min(val_size, n_total - 1)

	# Ref: Kohavi, IJCAI 1995 (stratified split)
	indices = np.arange(n_total)
	train_indices, val_indices = train_test_split(
		indices,
		test_size=val_size,
		random_state=CFG["seed"],
		stratify=labels_all,
		shuffle=True,
	)
	train_indices = train_indices.tolist()
	val_indices = val_indices.tolist()

	log(f"Split sizes — train: {len(train_indices)}  val: {len(val_indices)}")
	train_labels = [full_dataset.samples[idx][2] for idx in train_indices]
	val_labels = [full_dataset.samples[idx][2] for idx in val_indices]
	log(f"Stratified dist train — REAL={train_labels.count(0)}, FAKE={train_labels.count(1)}")
	log(f"Stratified dist val   — REAL={val_labels.count(0)}, FAKE={val_labels.count(1)}")

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

	def combine_features(img_feats, dct_feats):
		if dct_dim <= 0:
			return img_feats
		return torch.cat([img_feats, dct_feats], dim=1)

	try:
		imgs, dcts, _ = next(iter(train_loader))
		if torch.is_tensor(imgs):
			log(
				"DATA DEBUG — image batch mean="
				f"{imgs.mean().item():.4f} std={imgs.std().item():.4f} "
				f"min={imgs.min().item():.4f} max={imgs.max().item():.4f}"
			)
		if torch.is_tensor(dcts):
			dmean = dcts.view(dcts.size(0), -1).mean(1).mean().item()
			dstd = dcts.view(dcts.size(0), -1).std(1).mean().item()
			if dct_dim > 0:
				log(f"DATA DEBUG — dct batch mean(mean)={dmean:.4f} mean(std)={dstd:.4f}")
			else:
				log("DATA DEBUG — dct disabled/empty vectors")
	except Exception as exc:
		log("DATA DEBUG failed:")
		log_exception(exc)

	best_acc = 0.0
	best_auc = -1.0
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
			ckpt_head = ckpt.get("head_state_dict")
			ckpt_opt = ckpt.get("optimizer_state_dict")

			corrupted = False
			corrupt_name = None
			if ckpt_backbone is not None:
				corrupted, corrupt_name = state_has_nonfinite(ckpt_backbone)
			if not corrupted and ckpt_head is not None:
				corrupted, corrupt_name = state_has_nonfinite(ckpt_head)

			if corrupted:
				log(f"Checkpoint contains non-finite parameter: {corrupt_name}")
				corrupt_backup = CHECKPOINT_DIR / f"corrupt_{last_ckpt_name}"
				try:
					torch.save(ckpt, corrupt_backup)
					log(f"Backed up corrupt checkpoint to: {corrupt_backup}")
				except Exception as exc:
					log(f"Failed to backup corrupt checkpoint: {exc}")

				if os.environ.get("REPAIR_ON_CORRUPT", "0") == "1":
					log("REPAIR_ON_CORRUPT=1 -> attempting safe repair: reload pretrained backbone and reinit head, optimizer reset")
					repair_backbone, repair_feat_dim = build_backbone(log_fn=log)
					backbone = repair_backbone.to(device)
					head = build_head(repair_feat_dim, dct_dim).to(device)
					optimizer = optim.AdamW(
						[
							{"params": backbone.parameters(), "lr": CFG["lr_backbone"]},
							{"params": head.parameters(), "lr": CFG["lr_head"]},
						],
						weight_decay=1e-2,
					)
					scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-7)
					start_epoch = 1
					best_acc = 0.0
					best_auc = -1.0
					phase_best_auc = -1.0
					epochs_no_improve = 0
					current_aug_phase = None
					log("Repair successful: EfficientNet backbone reloaded, head reinitialized, optimizer reset. Starting from epoch 1.")
				else:
					log("Checkpoint appears corrupted. Set REPAIR_ON_CORRUPT=1 to attempt automatic repair, or remove the corrupt checkpoint to start fresh.")
					raise RuntimeError("Corrupt checkpoint loaded — aborting")
			else:
				if ckpt_backbone is None:
					raise KeyError("Checkpoint missing efficientnet_state_dict/backbone state")
				backbone.load_state_dict(ckpt_backbone)
				head.load_state_dict(ckpt_head)
				if ckpt_opt is not None:
					try:
						optimizer.load_state_dict(ckpt_opt)
						if len(optimizer.param_groups) >= 2:
							optimizer.param_groups[0]["lr"] = CFG["lr_backbone"]
							optimizer.param_groups[1]["lr"] = CFG["lr_head"]
					except Exception:
						log("Warning: unable to load optimizer state (parameter groups mismatch). Starting optimizer fresh.")
				else:
					optimizer.param_groups[0]["lr"] = CFG["lr_backbone"]
					optimizer.param_groups[1]["lr"] = CFG["lr_head"]

				start_epoch = ckpt.get("epoch", 0) + 1
				best_acc = ckpt.get("best_acc", 0.0)
				best_auc = ckpt.get("best_auc", -1.0)
				phase_best_auc = ckpt.get("phase_best_auc", None)
				epochs_no_improve = ckpt.get("epochs_no_improve", 0)
				current_aug_phase = ckpt.get("current_aug_phase", None)
				if current_aug_phase is None:
					phase_epoch = max(start_epoch - 1, 1)
					current_aug_phase = get_aug_phase(phase_epoch)
					log(f"Checkpoint has no current_aug_phase; derived phase='{current_aug_phase}' from epoch {phase_epoch}")
				if phase_best_auc is None:
					phase_best_auc = -1.0
					epochs_no_improve = 0
					log("Checkpoint has no phase_best_auc; resetting phase early-stopping tracker for safe resume")
				ckpt_scheduler = ckpt.get("scheduler_state_dict")
				if ckpt_scheduler is not None:
					try:
						scheduler.load_state_dict(ckpt_scheduler)
						log("Scheduler state restored from checkpoint.")
					except Exception:
						log("Warning: unable to load scheduler state. Scheduler will continue from current epoch.")
				log(
					f"Resumed from epoch {ckpt.get('epoch', 0)} "
					f"(best_acc={best_acc:.4f}, best_auc={best_auc:.4f}, "
					f"phase_best_auc={phase_best_auc:.4f}, aug_phase={current_aug_phase})"
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
				log(f"AUG PHASE changed {current_aug_phase} -> {new_aug_phase}; early-stopping counter reset")
				current_aug_phase = new_aug_phase

			gc.collect()
			if device.type == "cuda":
				try:
					torch.cuda.empty_cache()
				except Exception:
					pass

			mixup_prob, cutmix_prob = get_phase_mix_probs(epoch)

			backbone.train()
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

				if dcts.size(1) != dct_dim:
					raise RuntimeError(f"DCT dim mismatch: got {dcts.size(1)} expected {dct_dim}")

				use_mixup = np.random.rand() < mixup_prob
				use_cutmix = (not use_mixup) and (np.random.rand() < cutmix_prob) and (imgs.size(0) > 1)

				# Ref: Zhang et al., ICLR 2018 (alpha in [0.1, 0.4], use 0.4)
				if use_mixup:
					lam = float(np.random.beta(0.4, 0.4))
					idx = torch.randperm(imgs.size(0)).to(device)
					imgs_mix = lam * imgs + (1.0 - lam) * imgs[idx]
					dcts_mix = lam * dcts + (1.0 - lam) * dcts[idx]
					labels_a, labels_b = labels, labels[idx]
					logits = head(combine_features(backbone(imgs_mix), dcts_mix))
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
					dcts_cut = lam * dcts + (1.0 - lam) * dcts[idx]
					lam_adj = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
					labels_a, labels_b = labels, labels[idx]
					logits = head(combine_features(backbone(imgs_cut), dcts_cut))
					loss = lam_adj * criterion(logits, labels_a) + (1.0 - lam_adj) * criterion(logits, labels_b)
				else:
					feats = backbone(imgs)
					logits = head(combine_features(feats, dcts))
					loss = criterion(logits, labels)

				if not torch.isfinite(logits).all():
					log("[ERROR] Non-finite logits detected")
					raise RuntimeError("Non-finite logits")

				if (not isinstance(loss, torch.Tensor)) or (not torch.isfinite(loss).all()):
					log(f"[ERROR] Non-finite loss detected: {loss}")
					debug_ckpt = {
						"epoch": epoch,
						"current_aug_phase": current_aug_phase,
						"phase_best_auc": phase_best_auc,
						"efficientnet_state_dict": backbone.state_dict(),
						"head_state_dict": head.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
					}
					torch.save(debug_ckpt, CHECKPOINT_DIR / f"debug_nan_epoch{epoch}.pth")
					log(f"Saved debug checkpoint debug_nan_epoch{epoch}.pth")
					raise RuntimeError("Non-finite loss encountered")

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), max_norm=5.0)
				optimizer.step()

				bs = imgs.size(0)
				running_loss += loss.item() * bs
				preds = logits.argmax(dim=1)
				running_correct += (preds == labels).sum().item()
				running_total += bs

				train_loss = running_loss / running_total
				train_acc_epoch = running_correct / running_total
				lr_back = optimizer.param_groups[0].get("lr", 0.0)
				lr_head = optimizer.param_groups[1].get("lr", 0.0)
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

			try:
				max_train_batches = CFG.get("validate_train_max_batches", None)
				train_acc, train_auc, train_prec, train_rec, train_f1, train_cm = validate(
					train_loader,
					backbone,
					head,
					device,
					log_fn=log,
					max_batches=max_train_batches,
				)
			except Exception as exc:
				log(f"Train-set validate() failed: {exc}")
				train_acc, train_auc, train_prec, train_rec, train_f1, train_cm = (float("nan"),) * 2 + ([float("nan"), float("nan")],) * 3

			gc.collect()

			try:
				acc, auc, prec, rec, f1, cm = validate(val_loader, backbone, head, device, log_fn=log)
			except Exception as exc:
				log(f"Val validate() failed: {exc}")
				acc, auc, prec, rec, f1, cm = (float("nan"),) * 2 + ([float("nan"), float("nan")],) * 3

			gc.collect()

			try:
				log(f"EPOCH {epoch:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} Train AUC: {train_auc:.4f} | Val Acc: {acc:.5f} | Val AUC: {auc:.5f}")
				log(f"  Train Conf: {train_cm.tolist()} | Val Conf: {cm.tolist()}")
				log(f"  Val Prec_fake: {prec[0]:.4f} Prec_real: {prec[1]:.4f} | Val Rec_fake: {rec[0]:.4f} Rec_real: {rec[1]:.4f}")
				log(f"  Val F1_fake: {f1[0]:.4f} F1_real: {f1[1]:.4f}")
			except Exception:
				log(f"[EPOCH {epoch}] TRAIN loss={train_loss:.4f} acc={train_acc} val_acc={acc}")

			last_ckpt = {
				"epoch": epoch,
				"best_acc": best_acc,
				"best_auc": best_auc,
				"phase_best_auc": phase_best_auc,
				"epochs_no_improve": epochs_no_improve,
				"current_aug_phase": current_aug_phase,
				"efficientnet_state_dict": backbone.state_dict(),
				"head_state_dict": head.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"dct_dim": dct_dim,
			}
			torch.save(last_ckpt, CHECKPOINT_DIR / last_ckpt_name)
			log(f"LAST CHECKPOINT saved at epoch {epoch}")

			debug_every = CFG.get("debug_every_n_epochs", 0)
			massive_flag = CFG.get("debug_massive", False) or (os.environ.get("DEBUG_MASSIVE", "0") == "1")
			if debug_every and (epoch % debug_every == 0):
				save_param_debug(
					backbone,
					head,
					epoch,
					CHECKPOINT_DIR,
					log_fn=log,
					bins=CFG.get("debug_hist_bins", 64),
					massive=massive_flag,
				)

			if np.isfinite(auc) and (auc > best_auc):
				best_acc = acc
				best_auc = auc
				epochs_no_improve = 0
				ckpt = {
					"efficientnet_state_dict": backbone.state_dict(),
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
				}
				torch.save(ckpt, CHECKPOINT_DIR / best_ckpt_name)
				log(f"NEW BEST saved: auc={best_auc:.4f} acc={best_acc:.4f} epoch={epoch}")
			elif (not np.isfinite(auc)) and (not (CHECKPOINT_DIR / best_ckpt_name).exists()):
				best_acc = acc if np.isfinite(acc) else best_acc
				ckpt = {
					"efficientnet_state_dict": backbone.state_dict(),
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
				}
				torch.save(ckpt, CHECKPOINT_DIR / best_ckpt_name)
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
			current_lr_head = optimizer.param_groups[1]["lr"]
			log(f"LR updated: backbone={current_lr_backbone:.8f}, head={current_lr_head:.8f}")
	except Exception as exc:
		log_exception(exc)
	finally:
		log(f"Training finished. Best acc = {best_acc:.4f}")
		log_file.close()
		print("Log file saved to:", log_path)

	return 0


if __name__ == "__main__":
	sys.exit(main())
