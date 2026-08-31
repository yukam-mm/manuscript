"""
Side-by-side visual comparison of the three SR methods on lama_dilated inputs.

"""

import os
import glob
from PIL import Image, ImageDraw, ImageFont

# (label, directory)
METHODS = [
    ("lama_dilated (input)", "outputs/inpainting/lama_dilated"),
    ("swin2sr",              "outputs/sr_inhouse/swin2sr"),
    ("realesrgan",           "outputs/sr_inhouse/realesrgan"),
    ("aesrgan",              "outputs/sr_inhouse/aesrgan"),
]

RAW_DIR = "outputs/inpainting/lama_dilated"    # enumerate stems from here
OUT_DIR = "outputs/sr_inhouse/comparisons"

PANEL_H = 500
LABEL_H = 40

os.makedirs(OUT_DIR, exist_ok=True)


def find_file(dir_path, stem):
    """Handle both `<stem>.png` and `<stem>_out.png` naming conventions."""
    for suffix in ("", "_out"):
        p = os.path.join(dir_path, f"{stem}{suffix}.png")
        if os.path.exists(p):
            return p
    return None


def load_resize(path, h):
    im = Image.open(path).convert("RGB")
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)


def make_strip(paths, labels):
    panels = [load_resize(p, PANEL_H) for p in paths]
    total_w = sum(im.width for im in panels)
    canvas = Image.new("RGB", (total_w, PANEL_H + LABEL_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()

    x = 0
    for im, label in zip(panels, labels):
        canvas.paste(im, (x, LABEL_H))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (im.width - tw) // 2, 8), label,
                  fill=(0, 0, 0), font=font)
        x += im.width
    return canvas


ok = skipped = 0
inputs = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))
print(f"Found {len(inputs)} lama_dilated inputs\n")

for i, ip in enumerate(inputs, 1):
    stem = os.path.splitext(os.path.basename(ip))[0]

    paths, missing = [], []
    for label, d in METHODS:
        p = find_file(d, stem)
        if p is None:
            missing.append(f"{label}: {stem}")
        paths.append(p)

    if any(p is None for p in paths):
        print(f"[{i:3d}/{len(inputs)}] skip {stem}: missing {missing}")
        skipped += 1
        continue

    strip = make_strip(paths, [m[0] for m in METHODS])
    out_path = os.path.join(OUT_DIR, f"{stem}_sr.png")
    strip.save(out_path)
    print(f"[{i:3d}/{len(inputs)}] saved {out_path}")
    ok += 1

print(f"\nDone. ok={ok}  skipped={skipped}")
