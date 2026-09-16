from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def run_analysis():
  cleaned_file = BASE_DIR / "Data" / "processed" / "air_quality_cleaned.csv"
  if not cleaned_file.exists():
    print(f"Error: {cleaned_file} not found.")
    return

  # Load directly from your cleaned dataset
  df = pd.read_csv(cleaned_file)

  # Identify pollutant columns present in your dataset
  pollutant_cols = [
      c
      for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]
      if c in df.columns
  ]

  # Calculate AQI / Pollution Index directly from dataset pollutant values (peak pollution via row-wise max)
  df["Calculated_AQI"] = (
      df[pollutant_cols].max(axis=1) if pollutant_cols else 100.0
  )

  # Data-driven categorization based on dataset distribution percentiles
  p33, p66, p90 = df["Calculated_AQI"].quantile([0.33, 0.66, 0.90])

  def categorize_aqi(val):
    if pd.isna(val):
      return "Unknown"
    elif val <= p33:
      return "Good"
    elif val <= p66:
      return "Moderate"
    elif val <= p90:
      return "Poor"
    else:
      return "Severe"

  df["AQI_Bucket"] = df["Calculated_AQI"].apply(categorize_aqi)

  # 1. City Summary
  city_col = next(
      (c for c in ["city", "City", "state", "State"] if c in df.columns), None
  )
  if city_col:
    city_summary = (
        df.groupby(city_col)
        .agg(
            Record_Count=("Calculated_AQI", "count"),
            Mean_AQI=("Calculated_AQI", "mean"),
            Max_AQI=("Calculated_AQI", "max"),
            Min_AQI=("Calculated_AQI", "min"),
        )
        .reset_index()
    )
    city_summary["Dominant_Bucket"] = city_summary["Mean_AQI"].apply(
        categorize_aqi
    )
    city_summary.to_csv(OUTPUTS_DIR / "city_summary.csv", index=False)

  # 2. Station Summary
  station_col = next(
      (c for c in ["station", "Station", "location", "Location"] if c in df.columns),
      None,
  )
  if station_col:
    station_summary = (
        df.groupby(station_col)
        .agg(
            Record_Count=("Calculated_AQI", "count"),
            Mean_AQI=("Calculated_AQI", "mean"),
            Max_AQI=("Calculated_AQI", "max"),
        )
        .reset_index()
    )
    station_summary["AQI_Bucket"] = station_summary["Mean_AQI"].apply(
        categorize_aqi
    )
    station_summary.to_csv(OUTPUTS_DIR / "station_summary.csv", index=False)

  # 3. Pollutant Statistics
  if pollutant_cols:
    pollutant_stats = (
        df[pollutant_cols].agg(["mean", "median", "max", "min", "std"]).T.reset_index()
    )
    pollutant_stats.columns = [
        "Pollutant",
        "Mean",
        "Median",
        "Max",
        "Min",
        "StdDev",
    ]
    pollutant_stats.to_csv(OUTPUTS_DIR / "pollutant_statistics.csv", index=False)

  print("✅ Successfully analyzed dataset and generated outputs.")


if __name__ == "__main__":
  run_analysis()