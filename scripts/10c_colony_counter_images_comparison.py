import os
import matplotlib.pyplot as plt
from PIL import Image

folder = "/Users/oyunb1/Desktop/mdpi_colony_enhancement/outputs/colony_counts_inhouse/colony_counter_images"

# 3 rows × 3 methods
image_files = [
    # Row 1
    ["A_1.png", "R_1.png", "S_1.png"],

    # Row 2
    ["A_2.png", "R_2.png", "S_2.png"],

    # Row 3
    ["A_3.png", "R_3.png", "S_3.png"],
]

method_names = [
    "A-ESRGAN",
    "Real-ESRGAN",
    "Swin2SR"
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row in range(3):
    for col in range(3):

        filename = image_files[row][col]
        path = os.path.join(folder, filename)

        img = Image.open(path)

        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        # Method titles only on the first row
        if row == 0:
            axes[row, col].set_title(
                method_names[col],
                fontsize=16,
                fontweight="bold",
                pad=10
            )

plt.subplots_adjust(
    left=0.01,
    right=0.99,
    top=0.94,
    bottom=0.01,
    wspace=0.03,
    hspace=0.03
)

# Save figure
output_path = os.path.join(
    folder,
    "colony_counting_comparison_9_images.png"
)

plt.savefig(
    output_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02
)

plt.show()

print(f"Saved to: {output_path}")