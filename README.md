# Forest Carbon and Ecosystem Modeling Research Project

## Overview

This project models forest ecosystem variables (LAI, AbvGrndWood, SoilMoistFrac, TotSoilCarb) using a residual learning approach. We combine satellite forecasts, field observations, and environmental covariates to learn systematic forecast errors.

## Project Structure

```
Research/
├── cleaned_data_allyears/
│   ├── covariates/                    # Raw raster covariates (GeoTIFFs)
│   ├── covariates_all_years/          # Extracted CSV covariates
│   ├── forecasts_by_feature/          # Forecast CSVs per feature/year
│   ├── ModelingDatasets/              # Merged forecast+obs+covariates
│   ├── observations/                  # Processed observation CSV
│   └── Results_*/                     # Residual model results (by feature)
├── code/
│   ├── covariates_generation.py       # Extracts raster values at site coords
│   ├── forecasts_year.py              # Organizes forecast data
│   ├── final_dataset.py               # Merges forecast, obs, covariates
│   ├── eda_forecast_obs.py            # Missing data + residual analysis
│   ├── modelling.py                   # CV modeling scripts (linear, tree, etc)
│   ├── knn_regression.py              # KNN regression with residual targets
│   └── model-LAI/                     # Per-feature model scripts
├── observations/                      # RData + CSV observation data
├── R-data/                            # Original R output
└── ShapeFile/                         # pts.shp: 6400 site coordinates
```

## Inputs

### Forecasts

* Ensemble mean forecasts per year
* Four features, 2012–2024

### Observations

* Ground measurements per site/year
* Preprocessed into obs\_mean.csv

### Covariates

* Climate (e.g. temp, precip, rad)
* Soil (e.g. pH, nitrogen, SOC)
* Topography + land cover
* Extracted from raster `.tiff` files using shapefile locations

## Modeling Strategy

### Goal

Learn residuals = Observation - Forecast using covariates and forecast as predictors.

### Features

* Forecast (strong predictor)
* 12+ environmental covariates
* Drop Site, Year, Observation, Residual columns when training

### Models Used

* Linear (OLS, Ridge, Lasso)
* Tree-based (RandomForest, ExtraTrees)
* KNN (for spatial analog logic)
* XGBoost, LightGBM (future)

### Train/Test Split

* Train on all years except the most recent
* Hold out latest year (e.g. 2024) for testing

## Hyperparameter Tuning

We used `GridSearchCV` with 3-fold cross-validation (only on training years) to optimize tree models.

Example grid:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
```

## Output

Each feature has a `Results_*` directory with:

* Final R² and RMSE
* Top feature importances
* Scatter plot of forecast vs residual
* Residual vs predicted scatter plot

## Key Observations

* Forecast is consistently the most important feature
* Strong R² values in tree models:

  * LAI: \~0.70
  * AbvGrndWood: \~0.53
  * TotSoilCarb: \~0.88
  * SoilMoistFrac: \~0.89

## Concerns and Improvements

### ✅ Corrected Temporal Leakage

Earlier modeling scripts used random CV; we now split by year to avoid data leakage.

### ✅ Site Filtering

* Dropped sites with missing forecasts, observations, or covariates
* Ensured consistency across years

### ✅ Residual Bias

* Systematic bias in forecasts (e.g. overestimating LAI)
* Residual modeling helps learn and correct this bias

## How to Run

### Generate Covariates

```bash
python code/covariates_generation.py
```

### Create Modeling Dataset

```bash
python code/final_dataset.py
```

### Run Modeling

```bash
python code/model-LAI/extratrees-LAI.py
```

### Run Grid Search

```bash
python code/model-LAI/extratrees-gridsearch.py
```

## Dependencies

* Python: pandas, numpy, scikit-learn, matplotlib, seaborn, geopandas, rasterio
* R: dplyr, tidyr (for obs/forecast extraction)

## Contact

PI: Michael Dietze, Jonanthan Huggins
Data + modeling: Shashank Ramachandran
For questions, see the codebase or contact contributors.
