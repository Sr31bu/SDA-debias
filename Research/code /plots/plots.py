import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Settings ===
features = ["LAI", "AbvGrndWood", "SoilMoistFrac", "TotSoilCarb"]
dataset_dir = "/Users/shashankramachandran/desktop/Research/cleaned_data_allyears/ModelingDatasets"
output_dir = "./forecast_relationship_plots"
os.makedirs(output_dir, exist_ok=True)

# === Loop through features ===
for feature in features:
    print(f"📈 Plotting for: {feature}")
    df = pd.read_csv(os.path.join(dataset_dir, f"{feature}_modeling_dataset.csv"))
    df = df.dropna()

    # Forecast is the predicted value before residual correction
    forecast_col = "Forecast" if "Forecast" in df.columns else "Prediction"
    if forecast_col not in df.columns:
        print(f"⚠️ Forecast column missing in {feature} dataset. Skipping.")
        continue

    # === Plot: Forecast vs Observation ===
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x=forecast_col, y="Observation", alpha=0.3)
    plt.plot([df[forecast_col].min(), df[forecast_col].max()],
             [df[forecast_col].min(), df[forecast_col].max()],
             color='red', linestyle='--', label='1:1 Line')
    plt.xlabel("Forecast")
    plt.ylabel("Observation")
    plt.title(f"{feature}: Forecast vs Observation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feature}_forecast_vs_obs.png"))
    plt.close()

    # === Plot: Forecast vs Residual ===
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x=forecast_col, y="Residual", alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.xlabel("Forecast")
    plt.ylabel("Residual (Obs - Forecast)")
    plt.title(f"{feature}: Forecast vs Residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feature}_forecast_vs_residual.png"))
    plt.close()

print(f"\n✅ Plots saved to: {output_dir}")
