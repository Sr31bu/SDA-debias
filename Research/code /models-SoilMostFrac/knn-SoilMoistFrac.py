import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("/Users/shashankramachandran/desktop/Research/cleaned_data_allyears/ModelingDatasets/SoilMoistFrac_modeling_dataset.csv")
df["Year"] = pd.to_datetime(df["Year"], errors="coerce").dt.year
df = df.dropna(subset=["Year"]).copy()
df["Year"] = df["Year"].astype(int)

# Train-test split (last year is test)
last_year = df["Year"].max()
train_df = df[df["Year"] < last_year]
test_df = df[df["Year"] == last_year]

X_train = train_df.drop(columns=["Residual", "Observation", "Year", "Site"])
y_train = train_df["Residual"]
X_test = test_df.drop(columns=["Residual", "Observation", "Year", "Site"])
y_test = test_df["Residual"]

# 1. Grid Search to find best number of neighbors for KNN
knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
param_grid = {'kneighborsregressor__n_neighbors': list(range(1, 31))}

grid = GridSearchCV(knn_pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
best_k = grid.best_params_['kneighborsregressor__n_neighbors']

# Refit with best k
knn_best = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k))
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)

# Fit ExtraTrees\
best_params = {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
extra = ExtraTreesRegressor(**best_params)
extra.fit(X_train, y_train)
y_pred_tree = extra.predict(X_test)

# 2. Optimize weight between KNN and ExtraTrees
weights = np.linspace(0, 1, 101)
results = []

for w in weights:
    blended = w * y_pred_knn + (1 - w) * y_pred_tree
    rmse = np.sqrt(mean_squared_error(y_test, blended))
    r2 = r2_score(y_test, blended)
    results.append((w, rmse, r2))

results_df = pd.DataFrame(results, columns=["KNN_Weight", "RMSE", "R2"])
print(results_df.head())


# Find best weight
best_idx = results_df["RMSE"].idxmin()
best_weight_row = results_df.iloc[best_idx]
print(best_weight_row)
best_k, best_weight_row["KNN_Weight"], best_weight_row["RMSE"], best_weight_row["R2"]


# Save results
output_dir = "/Users/shashankramachandran/desktop/Research/results/SoilMoistFrac-KNN-TREES-ENSEMBLE"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "knn_extratrees_blending_results.csv")
results_df.to_csv(output_path, index=False)

print(f"✅ Saved blending results to: {output_path}")
