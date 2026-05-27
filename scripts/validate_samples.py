"""Scan `models/` and write `valid_samples.txt` with the sample IDs that
contain every required asset.

A sample folder is "valid" iff every method's `.glb` is present plus the
input image:

    models/<sample_id>/input.jpg
    models/<sample_id>/ours.glb
    models/<sample_id>/<baseline>.glb     (one per --methods entry)

The output file `models/valid_samples.txt` is consumed at runtime by
`js/viewer-swap.js` to populate the qualitative-results scene list.

Usage:
    python scripts/validate_samples.py
    python scripts/validate_samples.py --methods ours nova3r lari da3 moge2 vggt
    python scripts/validate_samples.py --no-input   # don't require input.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_METHODS = ["ours", "nova3r", "lari", "da3", "moge2"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    site_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--models_root", type=Path, default=site_root / "models",
                    help="Directory containing one folder per sample.")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help=f"Methods that must each have a <method>.glb in every sample "
                         f"(default: {' '.join(DEFAULT_METHODS)}).")
    ap.add_argument("--no-input", dest="require_input", action="store_false",
                    help="Don't require input.jpg.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output txt path (default: <models_root>/valid_samples.txt).")
    args = ap.parse_args()

    if not args.models_root.exists():
        print(f"models_root not found: {args.models_root}", file=sys.stderr)
        return 1

    required = [f"{m}.glb" for m in args.methods]
    if args.require_input:
        required.append("input.jpg")

    samples = sorted(p for p in args.models_root.iterdir() if p.is_dir())
    valid: list[str] = []
    invalid: dict[str, list[str]] = {}
    for s in samples:
        missing = [r for r in required if not (s / r).exists()]
        if missing:
            invalid[s.name] = missing
        else:
            valid.append(s.name)

    out_path = args.out or (args.models_root / "valid_samples.txt")
    out_path.write_text("\n".join(valid) + ("\n" if valid else ""))
    print(f"Required per sample: {', '.join(required)}")
    print(f"Wrote {len(valid)} valid samples → {out_path}")
    if invalid:
        print(f"\nSkipped {len(invalid)} samples missing files:")
        for name, miss in sorted(invalid.items()):
            print(f"  {name:<50}  missing: {', '.join(miss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
