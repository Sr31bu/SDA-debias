import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

features = ["LAI", "AbvGrndWood", "SoilMoistFrac", "TotSoilCarb"]
dataset_dir = "/Users/shashankramachandran/desktop/Research/cleaned_data_allyears/ModelingDatasets"

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 4, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2']
}


for feature in features:
    print(f"\nGridSearch for: {feature}")
    df = pd.read_csv(os.path.join(dataset_dir, f"{feature}_modeling_dataset.csv"))
    df = df.dropna()
    # Ensure Year is int
    df["Year"] = df["Year"].astype(str).str[:4].astype(int)
    final_year = df["Year"].max()
    train_df = df[df["Year"] != final_year]
    test_df = df[df["Year"] == final_year]
    X_train = train_df.drop(columns=["Residual", "Year", "Site", "Observation"])
    y_train = train_df["Residual"]
    X_test = test_df.drop(columns=["Residual", "Year", "Site", "Observation"])
    y_test = test_df["Residual"]

    grid_search = GridSearchCV(
        estimator=ExtraTreesRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R^2: {r2:.3f}") 