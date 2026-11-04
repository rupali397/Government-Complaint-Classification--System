import pandas as pd
from pathlib import Path

# Folder containing the original dataset files
DATA_DIR = Path(
    r"C:\Users\sahil\Documents\Government_Complaints_Dataset\Government_Complaints_Dataset"
)

original_file = DATA_DIR / "metadata.csv"
additional_file = DATA_DIR / "additional_complaints.csv"
output_file = DATA_DIR / "combined_complaints.csv"

# Load both files
original_df = pd.read_csv(original_file)
additional_df = pd.read_csv(additional_file)

print("Reading additional file from:", additional_file.resolve())
print("File rows loaded:", len(additional_df))
print("Columns:", additional_df.columns.tolist())
print(
    "New row found:",
    additional_df["transcript"].astype(str).str.contains(
        "ration distribution system", case=False, na=False
    ).any()
)

# Clean column names
for df in (original_df, additional_df):
    df.columns = df.columns.map(
        lambda col: str(col).strip().strip('"').strip("'")
    )

# Check required columns
required_columns = ["category", "language", "transcript"]

for name, df in [
    ("Original", original_df),
    ("Additional", additional_df)
]:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV is missing columns: {missing}")

# Keep only matching columns, in the same order
original_df = original_df[required_columns]
additional_df = additional_df[required_columns]

# Combine without changing either source file
combined_df = pd.concat(
    [original_df, additional_df],
    ignore_index=True
)

# Remove empty complaints and categories
combined_df = combined_df.dropna(
    subset=["category", "transcript"]
)

combined_df["category"] = combined_df["category"].astype(str).str.strip()
combined_df["transcript"] = combined_df["transcript"].astype(str).str.strip()

combined_df = combined_df[
    (combined_df["category"] != "") &
    (combined_df["transcript"] != "")
].copy()

# Save combined dataset
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Original complaints:", len(original_df))
print("Additional complaints:", len(additional_df))
print("Combined complaints:", len(combined_df))
print("\nCategory counts:")
print(combined_df["category"].value_counts())
print("\nSaved to:", output_file)
