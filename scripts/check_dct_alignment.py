"""
check_dct_alignment.py
----------------------
Cek apakah setiap gambar faces yang akan digunakan training
sudah memiliki pasangan file DCT (.npy) yang valid.

Aturan path DCT:
  dct_features/<rel_path_dari_img_root>.npy

Contoh:
  IMG:  data/raw/true-fake/Twitter/Fake/FLUX.1/faces/00001.jpg
  DCT:  data/processed/true-fake/Twitter/dct_features/Fake/FLUX.1/faces/00001.npy

Output:
  - Summary per generator
  - Daftar gambar yang TIDAK punya DCT
  - Sample cek isi DCT (shape, nilai statistik, NaN/Inf check)
"""

import sys
from pathlib import Path

import numpy as np

# ─── Setup path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATA_ROOT, DCT_ROOT

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
STYLEGAN_GENERATORS = {"StyleGAN", "StyleGAN2", "StyleGAN3"}
SD_GENERATORS = {
    "FLUX.1",
    "StableDiffusion1.5",
    "StableDiffusion2",
    "StableDiffusion3",
    "StableDiffusionXL",
}
EXPECTED_DIM = 192


def collect_face_images():
    """Kumpulkan semua path gambar faces sesuai aturan FaceOnlyDataset."""
    fake_dir = DATA_ROOT / "Fake"
    real_dir = DATA_ROOT / "Real"

    entries = []  # (img_path, source_label)

    # FAKE
    if fake_dir.exists():
        for gen_dir in sorted(fake_dir.iterdir()):
            if not gen_dir.is_dir():
                continue
            name = gen_dir.name
            if name in STYLEGAN_GENERATORS:
                for f in sorted(gen_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in VALID_EXT:
                        entries.append((f, f"Fake/{name}", 1))
            elif name in SD_GENERATORS:
                faces_dir = gen_dir / "faces"
                if faces_dir.exists():
                    for f in sorted(faces_dir.rglob("*")):
                        if f.is_file() and f.suffix.lower() in VALID_EXT:
                            entries.append((f, f"Fake/{name}/faces", 1))

    # REAL (FFHQ saja)
    ffhq_dir = real_dir / "FFHQ"
    if ffhq_dir.exists():
        for f in sorted(ffhq_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in VALID_EXT:
                entries.append((f, "Real/FFHQ", 0))

    return entries


def check_dct_for_images(entries):
    """Cek pasangan DCT untuk setiap gambar, kembalikan statistik per sumber."""
    results = {}     # source -> {"total": int, "found": int, "missing": [], "bad_shape": [], "nan_inf": []}
    sample_checks = []  # untuk cek isi: (img_path, dct_path, arr)

    for img_path, source, label in entries:
        if source not in results:
            results[source] = {
                "total": 0, "found": 0,
                "missing": [], "bad_shape": [], "nan_inf": [],
            }

        results[source]["total"] += 1

        # Hitung path DCT yang diharapkan
        try:
            rel = img_path.relative_to(DATA_ROOT)
        except ValueError:
            results[source]["missing"].append(str(img_path))
            continue

        dct_path = DCT_ROOT / rel.with_suffix(".npy")

        if not dct_path.exists():
            results[source]["missing"].append(str(img_path))
            continue

        results[source]["found"] += 1

        # Cek isi DCT
        try:
            arr = np.load(dct_path)
            dim = int(np.prod(arr.shape))

            if dim != EXPECTED_DIM:
                results[source]["bad_shape"].append((str(dct_path), dim))

            if not np.isfinite(arr).all():
                results[source]["nan_inf"].append(str(dct_path))

            # Ambil 3 sampel per source untuk inspeksi mendalam
            if len(sample_checks) < 15 and results[source]["found"] <= 3:
                sample_checks.append((img_path, dct_path, arr.copy()))

        except Exception as exc:
            results[source]["bad_shape"].append((str(dct_path), f"ERR:{exc}"))

    return results, sample_checks


def print_report(results, sample_checks, entries):
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  DCT ALIGNMENT CHECK — Twitter Face Dataset")
    print(SEP)

    total_imgs = len(entries)
    total_found = sum(v["found"] for v in results.values())
    total_missing = total_imgs - total_found

    print(f"\n  Total gambar faces : {total_imgs:>6}")
    print(f"  DCT ditemukan      : {total_found:>6}")
    print(f"  DCT TIDAK ADA      : {total_missing:>6}")
    print(f"  DCT root           : {DCT_ROOT}")

    print(f"\n{'─' * 70}")
    print(f"  {'Source':<35} {'Total':>6} {'Found':>6} {'Missing':>8} {'BadDim':>7} {'NaN/Inf':>8}")
    print(f"{'─' * 70}")

    for source, stat in sorted(results.items()):
        miss = stat["total"] - stat["found"]
        bad = len(stat["bad_shape"])
        nan = len(stat["nan_inf"])
        flag = " ✓" if miss == 0 and bad == 0 and nan == 0 else " ✗"
        print(f"  {source:<35} {stat['total']:>6} {stat['found']:>6} {miss:>8} {bad:>7} {nan:>8}{flag}")

    print(f"{'─' * 70}")

    # Detail masalah
    any_problem = False
    for source, stat in sorted(results.items()):
        if stat["missing"] or stat["bad_shape"] or stat["nan_inf"]:
            any_problem = True
            print(f"\n[MASALAH] {source}:")
            for m in stat["missing"][:5]:
                print(f"  MISSING DCT  → {Path(m).name}  ({m})")
            if len(stat["missing"]) > 5:
                print(f"  ... dan {len(stat['missing']) - 5} gambar lainnya")
            for p, dim in stat["bad_shape"][:3]:
                print(f"  BAD SHAPE    → dim={dim}  ({p})")
            for p in stat["nan_inf"][:3]:
                print(f"  NaN/Inf      → {p}")

    if not any_problem:
        print("\n  ✓ Semua DCT valid dan sesuai dengan gambar faces!")

    # Sampel inspeksi isi DCT
    print(f"\n{SEP}")
    print("  SAMPLE INSPEKSI ISI DCT (3 pertama per source)")
    print(SEP)

    for img_path, dct_path, arr in sample_checks:
        rel_img = img_path.relative_to(DATA_ROOT)
        print(f"\n  IMG : {rel_img}")
        print(f"  DCT : {dct_path.relative_to(DCT_ROOT)}")
        print(f"        shape={arr.shape}  dim={int(np.prod(arr.shape))}")
        print(f"        mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")
        print(f"        finite={np.isfinite(arr).all()}")

    print(f"\n{SEP}\n")


def main():
    print("Mengumpulkan daftar gambar faces...", flush=True)
    entries = collect_face_images()
    print(f"  → {len(entries)} gambar ditemukan")

    print("Mengecek pasangan DCT...", flush=True)
    results, sample_checks = check_dct_for_images(entries)

    print_report(results, sample_checks, entries)

    # Exit code: 0 jika semua OK, 1 jika ada masalah
    total_missing = sum(v["total"] - v["found"] for v in results.values())
    total_bad = sum(len(v["bad_shape"]) + len(v["nan_inf"]) for v in results.values())
    return 0 if (total_missing == 0 and total_bad == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
