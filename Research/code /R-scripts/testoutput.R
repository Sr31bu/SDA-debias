rm(list = ls())

# === Load forecast.all object ===
load("/Users/shashankramachandran/Desktop/Research/sda.all.forecast.analysis.Rdata")

library(dplyr)

# === Feature setup ===
vars <- c("AbvGrndWood", "LAI", "SoilMoistFrac", "TotSoilCarb")
n_vars <- length(vars)


# === Function to average across 25 ensemble members ===
split_forecast_by_feature <- function(forecast_df, out_dir) {''
  n_cols  <- ncol(forecast_df)
  n_sites <- n_cols / n_vars
  stopifnot(n_sites %% 1 == 0)
  
  for (i in seq_along(vars)) {
    var <- vars[i]
    
    # Get columns for this variable across all sites
    col_idx <- seq(i, n_cols, by = n_vars)
    mat <- as.matrix(forecast_df[, col_idx])  # shape: 25 (ensembles) x 6400 (sites)
    
    # Average ensemble members per site (i.e., per column)
    var_means <- colMeans(mat, na.rm = TRUE)  # length 6400
    
    # Output dataframe
    df_out <- data.frame(Site = seq_len(length(var_means)), Prediction = var_means)
    
    # Write to CSV
    out_path <- file.path(out_dir, paste0(var, ".csv"))
    write.csv(df_out, out_path, row.names = FALSE)
  }
}

# === Output directory setup ===
output_root <- "/Users/shashankramachandran/Desktop/Research/forecasts_by_feature"
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

# === Process each year ===
for (yr in names(forecast.all)) {
  year_str <- substr(yr, 1, 4)
  out_dir <- file.path(output_root, year_str)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("📅 Averaging ensembles for:", yr, "\n")
  split_forecast_by_feature(forecast.all[[yr]], out_dir)
  cat("✅ Saved feature CSVs to:", out_dir, "\n")
}
