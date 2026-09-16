import glob
import os
import numpy as np
import pandas as pd

# ============================================================
# 1. FIND AND LOAD RAW DATASET
# ============================================================
raw_files = glob.glob("*.csv") + glob.glob("Data/raw/*.csv")
raw_files = [
    f
    for f in raw_files
    if "cleaned" not in f
    and "summary" not in f
    and "statistics" not in f
    and f != "outputs"
]

if len(raw_files) == 0:
    raise FileNotFoundError(
        "No raw CSV file found in the project root or Data/raw/"
    )

file_path = raw_files[0]
df = pd.read_csv(file_path)

print("=" * 70)
print("AQI DATASET - UNDERSTANDING AND PREPROCESSING")
print("=" * 70)
print("File:", file_path)
print("Initial Rows:", df.shape[0])

# ============================================================
# 2. CLEAN COLUMN NAMES & PIVOT LONG FORMAT IF NEEDED
# ============================================================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# If dataset is in long format (pollutant_id & pollutant_avg), pivot to wide format
if "pollutant_id" in df.columns and "pollutant_avg" in df.columns:
  id_cols = [
      c
      for c in [
          "country",
          "state",
          "city",
          "station",
          "last_update",
          "latitude",
          "longitude",
      ]
      if c in df.columns
  ]
  df = df.pivot_table(
      index=id_cols, columns="pollutant_id", values="pollutant_avg"
  ).reset_index()
  df.columns.name = None

# Standardize text columns
text_cols = ["country", "state", "city", "station"]
for col in text_cols:
  if col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# Convert date column if present
if "last_update" in df.columns:
  df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")

# ============================================================
# 3. SAVE CLEANED DATASET
# ============================================================
os.makedirs("Data/processed", exist_ok=True)
output_file = "Data/processed/air_quality_cleaned.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
print(f"Cleaned dataset saved at: {output_file}")
print(f"Final processed rows: {df.shape[0]}")
print(f"Processed columns: {list(df.columns)}")