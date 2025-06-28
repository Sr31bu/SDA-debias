import os
import pandas as pd

# === Setup ===
base_dir = "../../cleaned_data_allyears"
forecast_root = os.path.join(base_dir, "forecasts_by_feature")
obs_file = os.path.join(base_dir, "observations", "obs_mean.csv")
cov_root = os.path.join(base_dir, "covariates_all_years")
output_dir = os.path.join(base_dir, "ModelingDatasets")
os.makedirs(output_dir, exist_ok=True)

features = ["LAI", "AbvGrndWood", "TotSoilCarb", "SoilMoistFrac"]

# === Step 1: Identify persistent bad covariate sites ===
print("Identifying sites with missing covariates in ALL years...")
bad_sites = set()
for year in range(2012, 2025):
    cov_path = os.path.join(cov_root, f"covariates_{year}.csv")
    if not os.path.exists(cov_path):
        print(f"❌ Missing covariates file: {cov_path}")
        continue
    df = pd.read_csv(cov_path)
    nan_sites = df[df.isnull().any(axis=1)]["Site"].tolist()
    if year == 2012:
        bad_sites = set(nan_sites)
    else:
        bad_sites &= set(nan_sites)
print(f"⚠️ {len(bad_sites)} persistent bad covariate sites will be dropped.")

# === Step 2: Load observation dataframe ===
obs_df = pd.read_csv(obs_file)
obs_df.rename(columns={"Site_ID": "Site"}, inplace=True)

# === Step 3: Process per feature ===
for feature in features:
    all_years = []
    for year in range(2012, 2025):
        year_str = f"{year}-07-15"

        # Forecast
        forecast_path = os.path.join(forecast_root, str(year), f"{feature}.csv")
        if not os.path.exists(forecast_path):
            print(f"❌ Missing forecast for {feature} in {year}")
            continue
        forecast_df = pd.read_csv(forecast_path)
        forecast_df["Year"] = year_str

        # Observation
        obs_subset = obs_df[obs_df["Year"] == year_str][["Year", "Site", feature]].copy()
        obs_subset.rename(columns={feature: "Observation"}, inplace=True)

        # Merge Forecast + Observation + Residual
        merged = pd.merge(forecast_df, obs_subset, on=["Year", "Site"], how="inner")
        merged.rename(columns={"Prediction": "Forecast"}, inplace=True)
        merged["Residual"] = merged["Observation"] - merged["Forecast"]

        # Covariates
        cov_path = os.path.join(cov_root, f"covariates_{year}.csv")
        if not os.path.exists(cov_path):
            print(f"❌ Missing covariates for {year}")
            continue
        cov_df = pd.read_csv(cov_path)

        # Merge all
        full_df = pd.merge(merged, cov_df, on="Site", how="inner")

        # Drop persistent bad sites
        full_df = full_df[~full_df["Site"].isin(bad_sites)]

        # Drop any missing Forecast/Observation/Covariates
        full_df.dropna(subset=["Forecast", "Observation"], inplace=True)
        full_df.dropna(axis=0, how="any", inplace=True)

        # Store for this year
        full_df["YearNum"] = int(year)
        all_years.append(full_df)

    # Add PrevResidual feature
    if all_years:
        final_df = pd.concat(all_years, ignore_index=True)
        final_df.sort_values(["Site", "YearNum"], inplace=True)
        final_df["PrevResidual"] = final_df.groupby("Site")["Residual"].shift(1)
        final_df.dropna(subset=["PrevResidual"], inplace=True)

        # Save dataset
        out_path = os.path.join(output_dir, f"{feature}residual_modeling_dataset.csv")
        final_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")
    else:
        print(f"⚠️ No usable data found for {feature}")
