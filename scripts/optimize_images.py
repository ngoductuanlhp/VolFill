"""Build web-optimized image assets for the project page.

  figures/*.jpg|png  ->  figures/opt/<stem>-{800,1600,2400}.webp  (never upscaled)

Usage:  python scripts/optimize_images.py [--quality 82]
"""
import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = FIG / "opt"
WIDTHS = (800, 1600, 2400)
SOURCES = [
    "quali_main.jpg", "quali_pixel_align.jpg", "quali_supp_mesh.jpg",
    "ablation_cond.jpg", "architecture_vae.png", "architecture_dit.png",
]


def resize_to(img, w):
    if img.width <= w:
        return img
    return img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)


def build_figures(quality):
    for name in SOURCES:
        src = Image.open(FIG / name).convert("RGB")
        stem = Path(name).stem
        prev = 0
        for w in WIDTHS:
            if prev >= src.width:
                break
            prev = w
            out = OUT / f"{stem}-{w}.webp"
            resize_to(src, w).save(out, "WEBP", quality=quality, method=6)
            print(f"{out.relative_to(ROOT)}  {out.stat().st_size // 1024} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build_figures(args.quality)


if __name__ == "__main__":
    main()
