import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# === Features to loop over ===
features = ["AbvGrndWood", "SoilMoistFrac", "TotSoilCarb"]
base_path = "../cleaned_data_allyears/ModelingDatasets"

for feature in features:
    print(f"\n🔍 Comparing residual models for: {feature}")

    input_path = os.path.join(base_path, f"{feature}_modeling_dataset.csv")
    output_dir = f"../cleaned_data_allyears/Results_{feature}"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    X = df.drop(columns=["Year", "Site", "Residual", "Observation"])
    y = df["Residual"]

    # Remove zero-variance columns
    X = X.loc[:, X.std() > 0]

    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
        rmse_scores = -scores
        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()

        results.append({
            "Model": name,
            "Mean_RMSE": mean_rmse,
            "Std_RMSE": std_rmse
        })

        print(f"{name}: RMSE = {mean_rmse:.3f} ± {std_rmse:.3f}")

    # Save results
    results_df = pd.DataFrame(results).sort_values("Mean_RMSE")
    results_df.to_csv(os.path.join(output_dir, "model_performance.csv"), index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(results_df["Model"], results_df["Mean_RMSE"], yerr=results_df["Std_RMSE"], capsize=5)
    plt.title(f"{feature} Residual Modeling: Model Comparison")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()

    print(f"Results saved to {output_dir}")
