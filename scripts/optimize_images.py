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


def build_logo(quality):
    """Hero/brand logo plus the 2x affiliation chips (displayed at 26-28 CSS px)."""
    for name, src_path in (("umass", FIG / "icons" / "umass_logo.png"), ("tucrete", FIG / "icons" / "tu_crete.png")):
        im = Image.open(src_path).convert("RGBA")
        im = im.resize((round(im.width * 52 / im.height), 52), Image.LANCZOS)
        out = OUT / f"{name}-52.webp"
        im.save(out, "WEBP", quality=90, method=6)
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size // 1024} KB")
    src = Image.open(ROOT / "figures" / "logo" / "volfill_logo.png").convert("RGBA")
    for w in (56, 420, 840):
        out = OUT / f"logo-{w}.webp"
        resize_to(src, w).save(out, "WEBP", quality=quality, method=6)
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size // 1024} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build_figures(args.quality)
    build_logo(args.quality)


if __name__ == "__main__":
    main()
