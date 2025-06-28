import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def model_residuals_without_forecast_feature(feature):
    """
    Model residuals using ONLY environmental covariates (no forecast as feature)
    This avoids circular logic issues.
    """
    print(f"\n=== Modeling {feature} Residuals (Covariates Only) ===")
    
    # Load data
    df = pd.read_csv(f"../cleaned_data_allyears/ModelingDatasets/{feature}_modeling_dataset.csv")
    df = df.dropna()
    
    # Ensure temporal split
    df["Year"] = pd.to_datetime(df["Year"])
    df["YearNum"] = df["Year"].dt.year
    final_year = df["YearNum"].max()
    
    train_df = df[df["YearNum"] != final_year]
    test_df = df[df["YearNum"] == final_year]
    
    # Use ONLY environmental covariates (exclude forecast)
    exclude_cols = ["Year", "Site", "Residual", "Observation", "Forecast"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df["Residual"]
    X_test = test_df[feature_cols]
    y_test = test_df["Residual"]
    
    # Train model
    model = ExtraTreesRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 features:")
    for _, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return rmse, r2, feature_importance

def model_forecast_correction(feature):
    """
    Alternative approach: Model the full observation using forecast + covariates
    This is more direct and avoids residual modeling issues.
    """
    print(f"\n=== Modeling {feature} Observations (Forecast + Covariates) ===")
    
    # Load data
    df = pd.read_csv(f"../cleaned_data_allyears/ModelingDatasets/{feature}_modeling_dataset.csv")
    df = df.dropna()
    
    # Ensure temporal split
    df["Year"] = pd.to_datetime(df["Year"])
    df["YearNum"] = df["Year"].dt.year
    final_year = df["YearNum"].max()
    
    train_df = df[df["YearNum"] != final_year]
    test_df = df[df["YearNum"] == final_year]
    
    # Use forecast + covariates to predict observation
    exclude_cols = ["Year", "Site", "Residual", "Observation"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df["Observation"]  # Predict observation directly
    X_test = test_df[feature_cols]
    y_test = test_df["Observation"]
    
    # Train model
    model = ExtraTreesRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")
    
    # Compare with baseline (just using forecast)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, test_df["Forecast"]))
    baseline_r2 = r2_score(y_test, test_df["Forecast"])
    
    print(f"Baseline (forecast only) RMSE: {baseline_rmse:.3f}")
    print(f"Baseline (forecast only) R²: {baseline_r2:.3f}")
    print(f"Improvement in RMSE: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")
    
    return rmse, r2

def analyze_residual_patterns(feature):
    """
    Analyze whether residuals are suitable for modeling
    """
    print(f"\n=== Residual Analysis for {feature} ===")
    
    df = pd.read_csv(f"../cleaned_data_allyears/ModelingDatasets/{feature}_modeling_dataset.csv")
    df = df.dropna()
    
    # Check correlation between forecast and residual
    correlation = df["Forecast"].corr(df["Residual"])
    print(f"Forecast-Residual correlation: {correlation:.3f}")
    
    # Check if residuals are homoscedastic
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(df["Forecast"], df["Residual"], alpha=0.5, s=1)
    plt.xlabel("Forecast")
    plt.ylabel("Residual")
    plt.title("Residual vs Forecast")
    
    plt.subplot(1, 3, 2)
    plt.hist(df["Residual"], bins=50, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    
    plt.subplot(1, 3, 3)
    # Group by forecast bins and check residual variance
    df["forecast_bin"] = pd.cut(df["Forecast"], bins=10)
    bin_stats = df.groupby("forecast_bin")["Residual"].agg(['mean', 'std']).dropna()
    plt.errorbar(range(len(bin_stats)), bin_stats['mean'], yerr=bin_stats['std'], fmt='o')
    plt.xlabel("Forecast Bin")
    plt.ylabel("Mean Residual")
    plt.title("Residual by Forecast Bin")
    
    plt.tight_layout()
    plt.savefig(f"residual_analysis_{feature}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual analysis plot saved: residual_analysis_{feature}.png")

# Main execution
features = ["LAI", "AbvGrndWood", "SoilMoistFrac", "TotSoilCarb"]

print("=== CORRECTED MODELING ANALYSIS ===")
print("This analysis addresses the circular logic issue in residual modeling.")

for feature in features:
    print(f"\n{'='*50}")
    
    # Analyze residual patterns
    analyze_residual_patterns(feature)
    
    # Model 1: Residuals without forecast feature
    rmse1, r2_1, importance = model_residuals_without_forecast_feature(feature)
    
    # Model 2: Direct observation modeling
    rmse2, r2_2 = model_forecast_correction(feature)
    
    print(f"\nSummary for {feature}:")
    print(f"  Residual modeling (covariates only): R² = {r2_1:.3f}")
    print(f"  Direct observation modeling: R² = {r2_2:.3f}")

print(f"\n{'='*50}")
print("RECOMMENDATIONS:")
print("1. Use direct observation modeling instead of residual modeling")
print("2. If using residuals, exclude forecast as a feature")
print("3. Validate that residuals are suitable for modeling")
print("4. Compare against simple baseline (forecast only)") 