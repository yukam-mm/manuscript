"""
Generate binary PNG masks for inpainting from LabelMe .json annotations.

Default (no args): process every .json in data/inhouse/labelme_annotations/.
Single file: pass a .json path as the first positional argument.

Output: data/inhouse/masks_final/<stem>_mask.png
- All annotated polygons filled white (255) on a black (0) background.

Usage:
    python scripts/01_generate_masks.py                 # batch all
    python scripts/01_generate_masks.py path/to/x.json  # single file
    python scripts/01_generate_masks.py --skip-existing # skip already-generated masks
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Project layout, resolved from this script's location (scripts/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOT_DIR = PROJECT_ROOT / "data" / "inhouse" / "labelme_annotations"
MASKS_DIR = PROJECT_ROOT / "data" / "inhouse" / "masks_final"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def load_labelme(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_size(data: dict, json_path: Path) -> tuple[int, int]:
    """Return (width, height). Falls back to imagePath if size fields absent."""
    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if h and w:
        return int(w), int(h)

    image_path = data.get("imagePath")
    if not image_path:
        raise ValueError(
            f"{json_path.name}: missing imageHeight/imageWidth and no imagePath."
        )
    img_file = (json_path.parent / image_path).resolve()
    if not img_file.exists():
        raise FileNotFoundError(f"Referenced image not found: {img_file}")
    with Image.open(img_file) as im:
        return im.size


def build_mask(data: dict, width: int, height: int) -> tuple[Image.Image, int]:
    """
    Rasterize all filled shapes as white on black.
    Returns (mask_image, n_shapes_drawn).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    drawn = 0

    for shape in data.get("shapes", []):
        pts = [(float(x), float(y)) for x, y in shape.get("points", [])]
        stype = shape.get("shape_type", "polygon")

        if not pts:
            continue

        if stype == "polygon":
            if len(pts) >= 3:
                draw.polygon(pts, fill=255, outline=255)
                drawn += 1
        elif stype == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            draw.rectangle(
                [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=255
            )
            drawn += 1
        elif stype == "circle" and len(pts) == 2:
            (cx, cy), (px, py) = pts
            r = float(np.hypot(px - cx, py - cy))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            drawn += 1
        elif stype in ("line", "linestrip", "point"):
            # No interior; skip.
            continue
        else:
            # Unknown shape_type: best-effort polygon fill.
            if len(pts) >= 3:
                draw.polygon(pts, fill=255, outline=255)
                drawn += 1

    return mask, drawn


def convert_one(json_path: Path, out_dir: Path) -> tuple[Path, int]:
    """Convert a single JSON to a mask PNG. Returns (out_path, n_shapes)."""
    data = load_labelme(json_path)
    width, height = resolve_size(data, json_path)
    mask, drawn = build_mask(data, width, height)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{json_path.stem}_mask.png"
    mask.save(out_path)
    return out_path, drawn


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def batch_convert(annot_dir: Path, out_dir: Path, skip_existing: bool) -> int:
    """Process every .json in annot_dir. Returns process exit code."""
    if not annot_dir.is_dir():
        print(f"error: annotations dir not found: {annot_dir}", file=sys.stderr)
        return 1

    json_files = sorted(annot_dir.glob("*.json"))
    if not json_files:
        print(f"error: no .json files in {annot_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(json_files)} annotation files in {annot_dir.name}/")
    print(f"Writing masks to {out_dir.relative_to(PROJECT_ROOT)}/\n")

    ok, skipped, failed, empty = 0, 0, 0, []
    for i, jp in enumerate(json_files, 1):
        out_path = out_dir / f"{jp.stem}_mask.png"

        if skip_existing and out_path.exists():
            print(f"[{i:3d}/{len(json_files)}] SKIP  {jp.name} (mask exists)")
            skipped += 1
            continue

        try:
            _, n = convert_one(jp, out_dir)
            flag = "  " if n > 0 else " !"  # flag empty masks
            print(f"[{i:3d}/{len(json_files)}]{flag}OK    {jp.name}  ({n} shapes)")
            if n == 0:
                empty.append(jp.name)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{i:3d}/{len(json_files)}] FAIL  {jp.name}: {e}", file=sys.stderr)
            traceback.print_exc(limit=1, file=sys.stderr)
            failed += 1

    print(f"\nDone. ok={ok}  skipped={skipped}  failed={failed}  total={len(json_files)}")
    if empty:
        print(f"WARNING: {len(empty)} mask(s) contain no shapes (empty black PNG):")
        for name in empty:
            print(f"  - {name}")
    return 0 if failed == 0 else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="LabelMe JSON -> binary PNG mask.")
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        help="Optional path to a single .json file. If omitted, all JSONs in "
             "data/inhouse/labelme_annotations/ are converted.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In batch mode, don't overwrite masks that already exist.",
    )
    parser.add_argument(
        "--annot-dir",
        type=Path,
        default=ANNOT_DIR,
        help=f"Override input directory (default: {ANNOT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MASKS_DIR,
        help=f"Override output directory (default: {MASKS_DIR.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    if args.json_path is not None:
        jp = args.json_path.expanduser().resolve()
        if not jp.is_file():
            print(f"error: not a file: {jp}", file=sys.stderr)
            return 1
        if jp.suffix.lower() != ".json":
            print(f"error: expected .json, got {jp.suffix}", file=sys.stderr)
            return 1
        out_path, n = convert_one(jp, args.out_dir)
        print(f"{out_path}  ({n} shapes)")
        return 0

    return batch_convert(args.annot_dir, args.out_dir, args.skip_existing)


if __name__ == "__main__":
    sys.exit(main())
