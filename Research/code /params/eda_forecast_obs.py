import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_dir = "../cleaned_data_allyears/Data_all_years"
features = ["LAI", "AbvGrndWood", "SoilMoistFrac", "TotSoilCarb"]

for feature in features:
    print(f"\n📊 EDA for: {feature}")
    file_path = os.path.join(data_dir, f"{feature}_with_obs_all_years.csv")
    df = pd.read_csv(file_path)

    # Parse Year
    df["Year"] = pd.to_datetime(df["Year"])
    df["YearNum"] = df["Year"].dt.year

    # === Missing data summary ===
    missing_by_year = df.groupby("YearNum")["Observation"].apply(lambda x: x.isna().sum())
    total_by_year = df.groupby("YearNum")["Observation"].count() + missing_by_year
    missing_pct = (missing_by_year / total_by_year).round(3)

    print("🕳️ Missing Observations by Year:")
    print(missing_by_year)
    print("📉 Missing Percentage by Year:")
    print(missing_pct)

    # === Plot: % missing by year
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing_pct.index, y=missing_pct.values)
    plt.title(f"{feature} – % Missing Observations by Year")
    plt.ylabel("Missing %")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(f"{feature}_missing_by_year.png")
    plt.close()
    print(f"📸 Saved plot: {feature}_missing_by_year.png")

    # === Distribution of forecast vs observation (non-missing)
    valid = df.dropna(subset=["Observation"])
    plt.figure(figsize=(10, 4))
    sns.histplot(valid["Forecast"], color="blue", label="Forecast", kde=True)
    sns.histplot(valid["Observation"], color="green", label="Observation", kde=True)
    plt.title(f"{feature} – Forecast vs Observation Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{feature}_forecast_vs_obs_dist.png")
    plt.close()
    print(f"📸 Saved plot: {feature}_forecast_vs_obs_dist.png")
