# Copilot Instructions for `final-skripsi`

## Project purpose and architecture
- This is a deepfake classifier with a **dual-input pipeline**: RGB image features + optional DCT features.
- Main training entrypoint is `src/train.py`; image-only mode is `src/train_no_dct.py` (sets `USE_DCT=0` and calls `train.main()`).
- Data flow is: raw image folders (`data/raw/...`) → optional DCT `.npy` features (`data/processed/...`) → `MixedDataset` (`src/dataset.py`) → EfficientNet backbone + linear head (`src/model.py`).
- `MixedDataset` always returns `(image_tensor, dct_tensor, label)`; when DCT is disabled/missing it returns zero vectors with `dct_dim`.

## Key code paths to understand before editing
- Runtime/config knobs are centralized in `src/config.py` (`CFG`, path constants, env overrides).
- Augmentation phases are in `src/transforms.py` and scheduled in `train.py` via `get_aug_phase()` / `get_phase_mix_probs()`.
- Validation metrics and AUC logic are in `src/validate.py`.
- DCT precompute/verification is standalone in `src/precompute_dct.py`.

## Label/index convention (critical)
- Raw Twitter classes are folder names `Fake/` and `Real/` under `data/raw/true-fake/Twitter`.
- `ImageFolder` assigns class indices lexicographically; agents must verify mapping before changing inference/UI label text.
- Keep notebook/script reporting consistent with model output index order (do not assume `0=REAL, 1=FAKE` blindly).

## Checkpoint and resume behavior
- Training resumes automatically from `models/checkpoints/last_checkpoint.pth` when present.
- Checkpoints store phase-aware early-stopping state (`current_aug_phase`, `phase_best_auc`, `epochs_no_improve`) and global best metrics.
- `best_efficient_dct.pth` is saved on **global best AUC**, while early stopping uses **phase-local AUC trend**.

## Commands developers actually use
- Install deps: `pip install -r requirements.txt`
- Train (default with DCT): `python src/train.py`
- Train image-only: `python src/train_no_dct.py`
- Quick CPU/smoke run example:
  - `FORCE_CPU=1 MAX_SUBSET=512 MAX_SUBSET_BEAUTY=0 python src/train.py`
- Continue DCT generation without overwrite:
  - `python src/precompute_dct.py --img_root <raw_img_root> --dct_root <dct_root> --num_workers 4`
- Verify DCT dimensions only:
  - `python src/precompute_dct.py --img_root <raw_img_root> --dct_root <dct_root> --verify`
- Batch inference script: `python scripts/test_batch_folder.py`

## Repo-specific patterns and guardrails
- Keep edits minimal and preserve current checkpoint key names for backward compatibility.
- Preserve `try/except` + logging style in training loops; logs go to `logs/train_log_*.txt`.
- Avoid changing tensor shapes/contracts: head expects `feature_dim + dct_dim` input.
- If adding new runtime switches, wire them through `apply_runtime_overrides()` in `config.py`.
- There is no formal test suite in this repo; validate changes with focused smoke runs and log inspection.
