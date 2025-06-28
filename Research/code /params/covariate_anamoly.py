import os
import pandas as pd

covariate_dir = "../cleaned_data_allyears/covariates_all_years"
years = list(range(2012, 2025))

nan_sites_by_year = {}

for year in years:
    file_path = os.path.join(covariate_dir, f"covariates_{year}.csv")
    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)
    nan_rows = df[df.isnull().any(axis=1)]
    nan_sites = nan_rows["Site"].tolist()
    nan_sites_by_year[year] = set(nan_sites)
    print(f"📅 {year}: {len(nan_sites)} sites with NaNs")

# 🔍 Sites with NaNs in every year
all_nan_sites = set.intersection(*nan_sites_by_year.values())
print(f"Sites with NaNs in ALL years: {len(all_nan_sites)}")
print("Sample site IDs:", list(all_nan_sites)[:10])
