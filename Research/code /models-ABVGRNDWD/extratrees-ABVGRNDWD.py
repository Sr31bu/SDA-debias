import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_csv("/Users/shashankramachandran/desktop/Research/cleaned_data_allyears/ModelingDatasets/AbvGrndWood_modeling_dataset.csv")

# === Clean ===
df = df.dropna()

# If Year is a full date, extract year component
df["Year"] = pd.to_datetime(df["Year"], errors="coerce").dt.year
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# === Hold out the most recent year ===
last_year = df["Year"].max()
train_df = df[df["Year"] < last_year]
test_df = df[df["Year"] == last_year]

print(f"✅ Training on years: {sorted(train_df['Year'].unique())}")
print(f"✅ Testing on year: {last_year}")

# === Prepare features and target ===
X_train = train_df.drop(columns=["Residual", "Observation", "Year", "Site"])
y_train = train_df["Residual"]
X_test = test_df.drop(columns=["Residual", "Observation", "Year", "Site"])
y_test = test_df["Residual"]

# === Optional: Use tuned parameters ===
best_params = {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}

model = ExtraTreesRegressor(**best_params)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Final ExtraTrees Test RMSE: {rmse:.3f}")
print(f"✅ Final ExtraTrees Test R²:   {r2:.3f}")
# === Assuming model is trained and X_train is your training features ===
feature_names = X_train.columns
importances = model.feature_importances_

# === Sort and plot ===
sorted_idx = np.argsort(importances)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_importances = importances[sorted_idx]


# === Create output folder ===
output_dir = "/Users/shashankramachandran/desktop/Research/results/AbvGrndWood"
os.makedirs(output_dir, exist_ok=True)

# === Save summary stats to text file ===
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("AbvGrndWood Residual Modeling Results\n")
    f.write(f"Train Years: {sorted(train_df['Year'].unique())}\n")
    f.write(f"Test Year: {last_year}\n")
    f.write(f"Test RMSE: {rmse:.3f}\n")
    f.write(f"Test R²:   {r2:.3f}\n\n")
    f.write("Top 10 Most Important Features:\n")
    for i in range(10):
        f.write(f"{i+1}. {sorted_features[i]}: {sorted_importances[i]:.4f}\n")

# === Save feature importance plot ===
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:15][::-1], sorted_importances[:15][::-1], color="teal")
plt.xlabel("Feature Importance")
plt.title("Top 15 Feature Importances (ExtraTrees)")
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# === Save residual scatter plot ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label="Ideal Fit")
plt.xlabel("True Residuals")
plt.ylabel("Predicted Residuals")
plt.title("Predicted vs True Residuals (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pred_vs_true.png"))
plt.close()

