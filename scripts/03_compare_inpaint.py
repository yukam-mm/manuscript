"""
Full inpainting comparison grid.

2 rows x 3 columns per image:
    row 1 (reference + classical): raw | telea | navier-stokes
    row 2 (LaMa variants):         lama (manual) | lama (dilated) | lama (auto)

Same model in all three LaMa panels; only the mask differs. Same OpenCV
algorithms in telea vs. NS. This layout isolates two things at once:
- classical vs. deep inpainter (compare row-1 columns 2-3 with row-2 columns)
- mask-quality effect on LaMa (compare row-2 columns to each other)

Output: outputs/inpainting/comparisons/<stem>_grid.png

Usage (from project root):
    python scripts/03_compare_inpaint.py
"""

import os
import glob
from PIL import Image, ImageDraw, ImageFont

# 2D grid: list of rows, each row is list of (label, dir, filename_template)
GRID = [
    [
        ("raw",              "data/inhouse/raw",                    "{stem}.png"),
        ("telea",             "outputs/inpainting/telea",           "{stem}.png"),
        ("navier-stokes",     "outputs/inpainting/navier_stokes",   "{stem}.png"),
    ],
    [
        ("lama (manual)",     "outputs/inpainting/lama_manual_mask", "{stem}.png"),
        ("lama (dilated)",    "outputs/inpainting/lama_dilated",     "{stem}.png"),
        ("lama (auto)",       "outputs/inpainting/lama_auto",        "{stem}.png"),
    ],
]

RAW_DIR = "data/inhouse/raw"  # used to enumerate stems
OUT_DIR = "outputs/inpainting/comparisons"
PANEL_H = 500
LABEL_H = 40

os.makedirs(OUT_DIR, exist_ok=True)


def load_resize(path, h):
    im = Image.open(path).convert("RGB")
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)


def make_grid(paths_2d, labels_2d):
    """paths_2d and labels_2d are lists-of-lists, same shape."""
    rows_of_panels = []
    for row_paths in paths_2d:
        rows_of_panels.append([load_resize(p, PANEL_H) for p in row_paths])

    # Column widths = max width in that column (so panels align vertically)
    n_cols = len(rows_of_panels[0])
    col_widths = [
        max(rows_of_panels[r][c].width for r in range(len(rows_of_panels)))
        for c in range(n_cols)
    ]
    total_w = sum(col_widths)
    row_h   = PANEL_H + LABEL_H
    total_h = row_h * len(rows_of_panels)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()

    for r, (row_panels, row_labels) in enumerate(zip(rows_of_panels, labels_2d)):
        y0 = r * row_h
        x = 0
        for c, (im, label) in enumerate(zip(row_panels, row_labels)):
            cell_w = col_widths[c]
            # center the panel horizontally in its column
            px = x + (cell_w - im.width) // 2
            canvas.paste(im, (px, y0 + LABEL_H))
            # centered label above panel
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (cell_w - tw) // 2, y0 + 8), label,
                      fill=(0, 0, 0), font=font)
            x += cell_w
    return canvas


ok = skipped = 0
for raw_path in sorted(glob.glob(os.path.join(RAW_DIR, "*.png"))):
    stem = os.path.splitext(os.path.basename(raw_path))[0]

    paths_2d, labels_2d, missing = [], [], []
    for row in GRID:
        row_paths, row_labels = [], []
        for label, d, tpl in row:
            p = os.path.join(d, tpl.format(stem=stem))
            if not os.path.exists(p):
                missing.append(f"{label}:{os.path.basename(p)}")
            row_paths.append(p)
            row_labels.append(label)
        paths_2d.append(row_paths)
        labels_2d.append(row_labels)

    if missing:
        print(f"skip {stem}: missing {missing}")
        skipped += 1
        continue

    grid = make_grid(paths_2d, labels_2d)
    out_path = os.path.join(OUT_DIR, f"{stem}_grid.png")
    grid.save(out_path)
    print(f"saved {out_path}")
    ok += 1

print(f"\nDone. ok={ok}  skipped={skipped}")
