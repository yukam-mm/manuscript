from PIL import Image, ImageDraw, ImageFont
from pathlib import Path



methods = [
    ("data/inhouse/raw", "Raw"),
    ("outputs/inpainting/telea", "Telea"),
    ("outputs/inpainting/navier_stokes", "Navier–Stokes"),
    ("outputs/inpainting/lama_manual_mask", "LaMa Manual"),
    ("outputs/inpainting/lama_dilated", "LaMa Dilated"),
    ("outputs/inpainting/lama_auto", "LaMa Auto"),
]


filenames = [
    "sc_h_44.png",
    "sc_h_43.png",
    "sc_h_41.png",
    "sc_h_39.png",
    "sc_h_38.png",
    "sc_h_36.png",
    "sc_h_30.png",
]

target_width = 420

# Only a small gap between images

horizontal_gap = 8

vertical_gap = 8

# Larger header

header_height = 85

font_size = 34

output_file = "inpainting_comparison_10x6.png"

# ============================================================

# FONT

# ============================================================

font_paths = [

    "/System/Library/Fonts/Helvetica.ttc",          # macOS

    "/System/Library/Fonts/Supplemental/Arial.ttf",

    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",

]

font = None

for font_path in font_paths:

    try:

        font = ImageFont.truetype(font_path, font_size)

        break

    except OSError:

        continue

if font is None:

    font = ImageFont.load_default()

# ============================================================

# DETERMINE IMAGE SIZE

# ============================================================

# Use the first raw image to determine the original aspect ratio.

sample_path = Path(methods[0][0]) / filenames[0]

if not sample_path.exists():

    raise FileNotFoundError(f"Cannot find sample image: {sample_path}")

with Image.open(sample_path) as sample:

    original_width, original_height = sample.size

aspect_ratio = original_height / original_width

target_height = round(target_width * aspect_ratio)

print(f"Original size: {original_width} x {original_height}")

print(f"Display size:  {target_width} x {target_height}")

# ============================================================

# CANVAS

# ============================================================

n_cols = len(methods)

n_rows = len(filenames)

canvas_width = (

    n_cols * target_width

    + (n_cols - 1) * horizontal_gap

)

canvas_height = (

    header_height

    + n_rows * target_height

    + (n_rows - 1) * vertical_gap

)

canvas = Image.new(

    "RGB",

    (canvas_width, canvas_height),

    "white"

)

draw = ImageDraw.Draw(canvas)

# ============================================================

# COLUMN TITLES

# ============================================================

for col, (_, label) in enumerate(methods):

    bbox = draw.textbbox(

        (0, 0),

        label,

        font=font

    )

    text_width = bbox[2] - bbox[0]

    text_height = bbox[3] - bbox[1]

    column_x = col * (target_width + horizontal_gap)

    x = column_x + (target_width - text_width) // 2

    y = (header_height - text_height) // 2 - 3

    draw.text(

        (x, y),

        label,

        fill="black",

        font=font

    )

# ============================================================

# ADD IMAGES

# ============================================================

for row, filename in enumerate(filenames):

    for col, (folder, label) in enumerate(methods):

        image_path = Path(folder) / filename

        if not image_path.exists():

            print(f"WARNING: Missing: {image_path}")

            continue

        with Image.open(image_path) as img:

            img = img.convert("RGB")

            # IMPORTANT:

            # Resize the COMPLETE image.

            # No crop, no thumbnail, no removal of background.

            img = img.resize(

                (target_width, target_height),

                Image.Resampling.LANCZOS

            )

            x = col * (target_width + horizontal_gap)

            y = (

                header_height

                + row * (target_height + vertical_gap)

            )

            canvas.paste(

                img,

                (x, y)

            )

# ============================================================

# SAVE

# ============================================================

canvas.save(

    output_file,

    dpi=(300, 300)

)

print()

print(f"Saved: {output_file}")

print(f"Final size: {canvas_width} x {canvas_height} px")