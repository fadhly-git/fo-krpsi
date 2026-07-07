"""
count_face_dataset.py
---------------------
Menghitung jumlah gambar wajah dari dataset Twitter deepfake detection.

Aturan pengambilan:
- Fake:
    - FLUX.1, SD1.5, SD2, SD3, SDXL  -> subfolder 'faces/'
    - StyleGAN, StyleGAN2, StyleGAN3  -> semua subfolder (isinya murni wajah)
- Real:
    - Hanya FFHQ (langsung berisi gambar wajah)
"""

import os
from pathlib import Path

BASE_DIR = Path("/home/fadhly/comvis/2kripsi/artifice-vs-nature/final-skripsi/data/raw/true-fake/Twitter")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

STYLEGAN_GENERATORS = {"StyleGAN", "StyleGAN2", "StyleGAN3"}

SD_GENERATORS = {
    "FLUX.1",
    "StableDiffusion1.5",
    "StableDiffusion2",
    "StableDiffusion3",
    "StableDiffusionXL",
}


def count_images(directory: Path) -> int:
    """Hitung jumlah gambar (rekursif) dalam direktori."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def main():
    results = {}

    # ─── FAKE ────────────────────────────────────────────────────────────────
    fake_dir = BASE_DIR / "Fake"
    results["Fake"] = {}

    for generator in sorted(fake_dir.iterdir()):
        if not generator.is_dir():
            continue
        name = generator.name

        if name in STYLEGAN_GENERATORS:
            # StyleGAN*: semua subdir adalah wajah
            count = count_images(generator)
            results["Fake"][name] = {"path": str(generator), "count": count}

        elif name in SD_GENERATORS:
            # SD-based: ambil hanya subfolder 'faces'
            faces_dir = generator / "faces"
            count = count_images(faces_dir)
            results["Fake"][name] = {"path": str(faces_dir), "count": count}

        else:
            print(f"  [WARNING] Generator tidak dikenal: {name}, dilewati.")

    # ─── REAL ────────────────────────────────────────────────────────────────
    real_dir = BASE_DIR / "Real"
    results["Real"] = {}

    ffhq_dir = real_dir / "FFHQ"
    count = count_images(ffhq_dir)
    results["Real"]["FFHQ"] = {"path": str(ffhq_dir), "count": count}

    # ─── PRINT HASIL ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET FACE COUNT  —  Twitter Deepfake Detection")
    print("=" * 60)

    total_fake = 0
    print("\n[FAKE]")
    for gen, info in sorted(results["Fake"].items()):
        print(f"  {gen:<25} : {info['count']:>6} gambar")
        total_fake += info["count"]
    print(f"  {'TOTAL FAKE':<25} : {total_fake:>6} gambar")

    total_real = 0
    print("\n[REAL]")
    for src, info in sorted(results["Real"].items()):
        print(f"  {src:<25} : {info['count']:>6} gambar")
        total_real += info["count"]
    print(f"  {'TOTAL REAL':<25} : {total_real:>6} gambar")

    print("\n" + "-" * 60)
    print(f"  {'TOTAL KESELURUHAN':<25} : {total_fake + total_real:>6} gambar")
    print(f"  {'Rasio Fake:Real':<25} : {total_fake}:{total_real}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
