import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import os

# === Paths ===
shape_path = "/Users/shashankramachandran/Desktop/Research/ShapeFile/pts.shp"
raster_dir = "/Users/shashankramachandran/desktop/Research/cleaned_data_allyears/covariates"
output_dir = "/Users/shashankramachandran/Desktop/Research/cleaned_data_allyears/covariates_all_years"
os.makedirs(output_dir, exist_ok=True)

# === Load shapefile ===
gdf = gpd.read_file(shape_path)
print(f"Loaded shapefile: {len(gdf)} site points")
print(f"Shapefile CRS: {gdf.crs}")

# === Loop through years ===
for year in range(2012, 2025):
    raster_path = os.path.join(raster_dir, f"covariates_{year}.tiff")
    output_path = os.path.join(output_dir, f"covariates_{year}.csv")

    if not os.path.exists(raster_path):
        print(f" Missing raster for year {year}, skipping...")
        continue

    print(f"\nProcessing year: {year}")
    with rasterio.open(raster_path) as raster:
        print(f"🗺️  Raster opened: {raster_path}")
        print(f"   - CRS: {raster.crs}")
        print(f"   - Dimensions: {raster.width} x {raster.height}")
        print(f"   - Bands: {raster.count}")

        # Check and align CRS
        if raster.crs != gdf.crs:
            print("CRS mismatch — reprojecting shapefile")
            gdf_proj = gdf.to_crs(raster.crs)
        else:
            gdf_proj = gdf

        coords = [(pt.x, pt.y) for pt in gdf_proj.geometry]
        print(f"📌 Sampling at {len(coords)} site coordinates")

        # Extract band names
        band_names = []
        for i in range(1, raster.count + 1):
            desc = raster.descriptions[i - 1]
            band_names.append(desc if desc else f"covariate_{i}")

        # Sample raster
        covariate_values = list(raster.sample(coords))
        assert len(covariate_values) == len(gdf), "Sample count mismatch"

        # Check for NaNs
        nan_count = np.isnan(np.array(covariate_values)).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in sampled covariates")

        # Create DataFrame
        covariate_df = pd.DataFrame(covariate_values, columns=band_names)
        covariate_df["Site"] = gdf["FID"]  # or gdf["ID"] depending on field
        covariate_df.sort_values(by="Site", inplace=True)

        # Save
        covariate_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
