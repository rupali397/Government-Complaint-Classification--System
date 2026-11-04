
import pandas as pd
from pathlib import Path

# Dataset ka path
DATA_PATH = Path(
    r"C:\Users\sahil\Documents\Government_Complaints_Dataset\Government_Complaints_Dataset\combined_complaints.csv"
)

# CSV file read karo
df = pd.read_csv(DATA_PATH)

# Column names clean karo
df.columns = df.columns.map(
    lambda col: str(col).strip().strip('"').strip("'")
)

# Check karo required columns hain ya nahi
required_columns = ["category", "language", "transcript"]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Text values clean karo
for column in required_columns:
    df[column] = df[column].fillna("").astype(str).str.strip()

df["category"] = df["category"].str.lower()
df["language"] = df["language"].str.lower()

# Empty complaints hatao
df = df[
    (df["category"] != "") &
    (df["transcript"] != "")
].copy()

print("\n--- DATASET SUMMARY ---")
print("Total complaints:", len(df))
print("Total categories:", df["category"].nunique())
print("Languages:", df["language"].value_counts().to_dict())

print("\n--- COMPLAINTS PER CATEGORY ---")
print(df["category"].value_counts().sort_index())

print("\n--- CATEGORY vs LANGUAGE ---")
print(pd.crosstab(df["category"], df["language"]))

print("\n--- MISSING VALUES ---")
print(df[required_columns].isna().sum())

print("\n--- DUPLICATE COMPLAINTS ---")
print("Duplicate transcripts:", df["transcript"].duplicated().sum())

# Check duplicate complaints after normalizing text
df["normalized_transcript"] = (
    df["transcript"]
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print("\n--- NORMALIZED DUPLICATES ---")
print(
    "Duplicate complaints after normalization:",
    df["normalized_transcript"].duplicated().sum()
)

print("\n--- DUPLICATE CATEGORY MISMATCH ---")
duplicate_groups = df[
    df["normalized_transcript"].duplicated(keep=False)
]

if not duplicate_groups.empty:
    print(
        duplicate_groups[
            ["category", "transcript"]
        ].sort_values("normalized_transcript").to_string(index=False)
    )
else:
    print("No normalized duplicates found.")
    
print("\n--- EXAMPLE COMPLAINTS PER CATEGORY ---")
for category in sorted(df["category"].unique()):
    print(f"\n{category.upper()}")
    examples = df.loc[
        df["category"] == category, "transcript"
    ].head(2)

    for complaint in examples:
        print("-", complaint)


print("\n--- ROADS & TRANSPORT COMPLAINTS ---")
for complaint in df.loc[
    df["category"] == "roads_transport", "transcript"
]:
    print("-", complaint)

print("\n--- SANITATION COMPLAINTS ---")
for complaint in df.loc[
    df["category"] == "sanitation", "transcript"
]:
    print("-", complaint)

print("\nAnalysis complete !")
