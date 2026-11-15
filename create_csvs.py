
import csv
import random
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
XRAYS_DIR = PROJECT_DIR / "data" / "xrays"
MASKS_DIR = PROJECT_DIR / "data" / "masks"
CSVS_DIR = PROJECT_DIR / "data" / "CSVs"

CSVS_DIR.mkdir(parents=True, exist_ok=True)

# Get PNG files, sorted by their numeric filenames
xrays = sorted(
    XRAYS_DIR.glob("*.png"),
    key=lambda p: int(p.stem)
)

masks = sorted(
    MASKS_DIR.glob("*.png"),
    key=lambda p: int(p.stem)
)

# Match images and masks using their filenames
xray_ids = {p.stem for p in xrays}
mask_ids = {p.stem for p in masks}

if len(xrays) != 150 or len(masks) != 100:
    raise ValueError(
        f"Expected 150 X-rays and 100 masks. "
        f"Found {len(xrays)} X-rays and {len(masks)} masks."
    )

if not mask_ids.issubset(xray_ids):
    raise ValueError("Some masks do not have matching X-ray images.")

if len(mask_ids) != 100:
    raise ValueError("Duplicate or unexpected mask IDs detected.")

# Create the main dataset
dataset = []

for xray in xrays:
    mask_path = MASKS_DIR / f"{xray.stem}.png"

    if xray.stem in mask_ids:
        mask_value = mask_path.relative_to(PROJECT_DIR).as_posix()
    else:
        mask_value = "None"

    dataset.append([
        xray.relative_to(PROJECT_DIR).as_posix(),
        mask_value
    ])

# Save dataset.csv
def save_csv(filename, rows):
    with open(CSVS_DIR / filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["xrays", "masks"])
        writer.writerows(rows)

save_csv("dataset.csv", dataset)

# Separate labeled and test samples
labeled = [row for row in dataset if row[1] != "None"]
test = [row for row in dataset if row[1] == "None"]

# Randomly split labeled samples into training and validation
random.seed(42)
random.shuffle(labeled)

train = labeled[:80]
validation = labeled[80:]

# Save the three additional CSV files
save_csv("train.csv", train)
save_csv("validation.csv", validation)
save_csv("test.csv", test)

# Verify sample counts
counts = {
    "dataset.csv": len(dataset),
    "train.csv": len(train),
    "validation.csv": len(validation),
    "test.csv": len(test)
}

print("\nCSV files created successfully!")
print("\nSample counts:")

for filename, count in counts.items():
    print(f"{filename}: {count}")

# Create bar chart
plt.figure(figsize=(9, 5))
plt.bar(counts.keys(), counts.values())

plt.title("Number of Samples in Each CSV File")
plt.xlabel("CSV Files")
plt.ylabel("Number of Samples")
plt.tight_layout()

chart_path = PROJECT_DIR / "sample_counts.png"
plt.savefig(chart_path, dpi=300)
plt.show()

# Create ZIP archive containing the four CSV files
zip_path = PROJECT_DIR / "csv_files_Group4.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
    for filename in counts:
        file_path = CSVS_DIR / filename
        zip_file.write(file_path, arcname=filename)

print("\nBar chart saved:", chart_path)
print("ZIP archive created:", zip_path)

# Final verification
assert len(dataset) == 150
assert len(train) == 80
assert len(validation) == 20
assert len(test) == 50
assert len(train) + len(validation) == len(labeled)

print("\nAll verification checks passed!")
