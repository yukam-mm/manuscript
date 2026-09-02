import pandas as pd

# Load results
df = pd.read_csv("metrics/d1_reference_free/d1_combined.csv")

# Names for paper
method_names = {
    "raw": "Raw",
    "telea": "Telea",
    "navier_stokes": "Navier–Stokes",
    "lama_manual_mask": "LaMa Manual",
    "lama_dilated": "LaMa Dilated",
    "lama_auto": "LaMa Auto",
}

rows = []

for method, paper_name in method_names.items():

    subset = df[df["method"] == method]

    # EasyOCR
    ocr_mean = subset["ocr_conf_sum ↓"].mean()
    ocr_sd = subset["ocr_conf_sum ↓"].std()

    # ReMOVE
    remove_mean = subset["remove_score ↑"].mean()
    remove_sd = subset["remove_score ↑"].std()

    # Visual
    visual_count = (
        subset["conlcusion (visual)"] == "✔️"
    ).sum()

    # Visual assessment was available for 65 images
    if method == "raw":
        visual = "—"
    else:
        visual = (
            f"{visual_count}/65 "
            f"({visual_count / 65 * 100:.1f}%)"
        )

    rows.append({
        "Method": paper_name,
        "EasyOCR ↓": f"{ocr_mean:.3f} ± {ocr_sd:.3f}",
        "ReMOVE ↑": f"{remove_mean:.3f} ± {remove_sd:.3f}",
        "Visual assessment": visual
    })


table = pd.DataFrame(rows)

print(table)

table.to_csv(
    "table_inpainting_evaluation.csv",
    index=False
)

print("\nSaved: table_inpainting_evaluation.csv")