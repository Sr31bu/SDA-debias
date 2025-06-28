import os
import pandas as pd

# === Setup ===
base_dir = "../cleaned_data_allyears"
forecast_root = os.path.join(base_dir, "forecasts_by_feature")
obs_file = os.path.join(base_dir, "observations", "obs_mean.csv")
features = ["LAI", "AbvGrndWood", "TotSoilCarb", "SoilMoistFrac"]

# === Load observations ===
obs_df = pd.read_csv(obs_file)
obs_df.rename(columns={"Site_ID": "Site"}, inplace=True)  # unify site name

# === Loop through features ===
for feature in features:
    all_years = []
    for year in range(2012, 2025):
        year_str = f"{year}-07-15"
        forecast_path = os.path.join(forecast_root, str(year), f"{feature}.csv")
        if not os.path.exists(forecast_path):
            print(f"Missing forecast for {feature} in {year}")
            continue

        forecast_df = pd.read_csv(forecast_path)
        forecast_df["Year"] = year_str

        # Get corresponding obs for this year and feature
        obs_subset = obs_df[obs_df["Year"] == year_str][["Year", "Site", feature]]
        obs_subset.rename(columns={feature: "Observation"}, inplace=True)

        merged = pd.merge(forecast_df, obs_subset, on=["Year", "Site"], how="inner")
        merged.rename(columns={"Prediction": "Forecast"}, inplace=True)
        merged = merged[["Year", "Site", "Forecast", "Observation"]]

        all_years.append(merged)

    # === Save per feature merged dataset ===
    if all_years:
        full_df = pd.concat(all_years, ignore_index=True)
        output_path = os.path.join(base_dir, f"{feature}_with_obs_all_years.csv")
        full_df.to_csv(output_path, index=False)
        print(f"Saved merged: {output_path}")
    else:
        print(f"No data available for feature: {feature}")
