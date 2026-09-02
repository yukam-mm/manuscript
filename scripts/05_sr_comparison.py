from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("outputs/inpainting/lama_dilated")

REALESRGAN_DIR = Path("outputs/sr_inhouse/realesrgan")
AESRGAN_DIR = Path("outputs/sr_inhouse/aesrgan")
SWIN2SR_DIR = Path("outputs/sr_inhouse/swin2sr")

OUTPUT_PATH = Path("outputs/sr_inhouse/sr_comparison_7images.png")


# ============================================================
# SELECT 7 IMAGES
# Use filenames that exist in all four folders
# ============================================================

IMAGE_NAMES = [
    "sc_33_13.png",
    "sc_33_17.png",
    "sc_33_31.png",
    "sc_33_36.png",
    "sc_h_44.png",
    "sc_h_41.png",      
    "sc_h_39.png",      
]


# ============================================================

# METHODS

# ============================================================

METHODS = [

    ("Input", RAW_DIR, "raw"),

    ("Real-ESRGAN", REALESRGAN_DIR, "out"),

    ("A-ESRGAN", AESRGAN_DIR, "out"),

    ("Swin2SR", SWIN2SR_DIR, "same"),

]

def get_path(folder, original_name, naming):

    original_name = Path(original_name)

    if naming == "out":

        # sc_33_33.png -> sc_33_33_out.png

        return folder / f"{original_name.stem}_out{original_name.suffix}"

    # Raw and Swin2SR keep original filename

    return folder / original_name.name

# ============================================================

# CREATE GRID

# 7 rows × 4 columns

# ============================================================

n_rows = len(IMAGE_NAMES)

n_cols = len(METHODS)

fig, axes = plt.subplots(

    n_rows,

    n_cols,

    figsize=(12, 3 * n_rows)

)

plt.subplots_adjust(

    left=0.01,

    right=0.99,

    top=0.965,

    bottom=0.01,

    wspace=0.015,

    hspace=0.015

)

# ============================================================

# LOAD IMAGES

# ============================================================

for row, image_name in enumerate(IMAGE_NAMES):

    for col, (title, folder, naming) in enumerate(METHODS):

        image_path = get_path(

            folder,

            image_name,

            naming

        )

        ax = axes[row, col]

        if not image_path.exists():

            print(f"Missing: {image_path}")

            ax.text(

                0.5,

                0.5,

                "Missing",

                ha="center",

                va="center"

            )

            ax.axis("off")

            continue

        image = Image.open(image_path).convert("RGB")

        ax.imshow(image)

        ax.axis("off")

        # Titles only above first row

        if row == 0:

            ax.set_title(

                title,

                fontsize=14,

                fontweight="bold",

                pad=8

            )

# ============================================================

# SAVE

# ============================================================

OUTPUT_PATH.parent.mkdir(

    parents=True,

    exist_ok=True

)

plt.savefig(

    OUTPUT_PATH,

    dpi=300,

    bbox_inches="tight",

    pad_inches=0.02

)

plt.close()

print(f"Saved:\n{OUTPUT_PATH}")