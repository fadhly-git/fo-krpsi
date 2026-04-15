"""Fungsi validasi/evaluasi dan utilitas debug parameter model."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)


@torch.no_grad()
def validate(loader, backbone, head, device, log_fn=None, max_batches=None):
	"""Validate over DataLoader with optional max batch limit."""
	backbone.eval()
	head.eval()

	preds = []
	probs = []
	targets = []

	for i, (imgs, dcts, labels) in enumerate(loader):
		if max_batches is not None and i >= max_batches:
			break

		imgs = imgs.to(device)
		dcts = dcts.to(device)
		labels = labels.to(device)

		if dcts.dim() > 2:
			dcts = dcts.view(dcts.size(0), -1)

		feats = backbone(imgs)
		logits = head(torch.cat([feats, dcts], dim=1))

		if torch.isnan(logits).any() or torch.isinf(logits).any():
			if log_fn is not None:
				log_fn("[ERROR] validate(): logits contain NaN/Inf — dumping diagnostics")
				for name, param in list(backbone.named_parameters()) + list(head.named_parameters()):
					if torch.isnan(param).any() or torch.isinf(param).any():
						log_fn(f"  param {name} has NaN/Inf")
			raise RuntimeError("NaN/Inf detected in logits during validation")

		prob = torch.softmax(logits, dim=1)[:, 1]
		preds.extend(logits.argmax(1).cpu().numpy().tolist())
		probs.extend(prob.cpu().numpy().tolist())
		targets.extend(labels.cpu().numpy().tolist())

		del imgs, dcts, labels, feats, logits, prob

		if (i % 10) == 0:
			gc.collect()
			if device.type == "cuda":
				try:
					torch.cuda.empty_cache()
				except Exception:
					pass

	if len(targets) == 0:
		return float("nan"), float("nan"), [float("nan"), float("nan")], [float("nan"), float("nan")], [float("nan"), float("nan")], [[0, 0], [0, 0]]

	acc = accuracy_score(targets, preds)
	try:
		auc = roc_auc_score(targets, probs)
	except Exception:
		auc = float("nan")

	prec = precision_score(targets, preds, labels=[0, 1], average=None, zero_division=0)
	rec = recall_score(targets, preds, labels=[0, 1], average=None, zero_division=0)
	f1 = f1_score(targets, preds, labels=[0, 1], average=None, zero_division=0)
	cm = confusion_matrix(targets, preds, labels=[0, 1])

	preds.clear()
	probs.clear()
	targets.clear()

	gc.collect()
	if device.type == "cuda":
		try:
			torch.cuda.empty_cache()
		except Exception:
			pass

	return acc, auc, prec, rec, f1, cm


def save_param_debug(model, head, epoch, out_dir: Path, log_fn=None, bins: int = 64, massive: bool = False):
	"""Save parameter statistics/histograms to JSON for debugging."""
	out = {}
	for name, param in list(model.named_parameters()) + list(head.named_parameters()):
		arr = param.detach().cpu().numpy()
		out[name + "_mean"] = float(arr.mean())
		out[name + "_std"] = float(arr.std())
		out[name + "_min"] = float(arr.min())
		out[name + "_max"] = float(arr.max())

		try:
			hist, edges = np.histogram(arr.reshape(-1), bins=bins)
			out[name + "_hist_counts"] = hist.tolist()
			out[name + "_hist_edges"] = edges.tolist()
		except Exception:
			out[name + "_hist_counts"] = []
			out[name + "_hist_edges"] = []

		if massive:
			try:
				flat = arr.reshape(-1)
				idx = np.linspace(0, flat.size - 1, min(256, flat.size)).astype(int)
				out[name + "_sample"] = flat[idx].tolist()
			except Exception:
				out[name + "_sample"] = []

	out_path = out_dir / f"debug_params_epoch{epoch}.json"
	try:
		with open(out_path, "w", encoding="utf-8") as outfile:
			json.dump(out, outfile)
		if log_fn is not None:
			log_fn(f"Saved parameter debug JSON: {out_path}")
	except Exception as exc:
		if log_fn is not None:
			log_fn(f"Failed saving debug params: {exc}")
