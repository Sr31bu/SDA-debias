rm(list = ls())

load("/Users/shashankramachandran/Desktop/Research/observations/obs.mean.Rdata")
print("Objects inside the RData file:")
print(ls())
if ("obs.mean" %in% ls()) {
  print("✅ 'obs.mean' found! Checking its structure...")
  print("Structure of obs.mean:")
  str(obs.mean)
  
  # Check class/type of object
  print("Class of obs.mean:")
  print(class(obs.mean))
  
  # Check dimensions if it's a dataframe or matrix
  if (is.data.frame(obs.mean) | is.matrix(obs.mean)) {
    print("obs.mean is a structured dataset. Checking dimensions...")
    print(dim(obs.mean))
    print("First few rows of obs.mean:")
    print(head(obs.mean))
  
    write.csv(obs.mean, "/Users/shashankramachandran/Desktop/Research/observations/obs_mean.csv", row.names = FALSE)
    print("Saved as obs_mean.csv for Python!")
  } else {
    print("obs.mean is not a dataframe or matrix. It may be a list or another structure.")
  }
} else {
  print("obs.mean' not found in the file. Check if the file is correct.")
}
length(obs.mean)
print("Names of elements in obs.mean (if available):")
print(names(obs.mean))  # Check if the list has named elements

print("Structure of the first element:")
str(obs.mean[[1]])  # Check what the first item looks like

print("First few rows (if applicable):")
print(head(obs.mean[[1]]))  # Preview the first element

# Create a global set of column names
all_columns_global <- c()

# First pass: just gather the columns from every site, every year
for (year in names(obs.mean)) {
  year_data <- obs.mean[[year]]
  
  for (site_id in seq_along(year_data)) {
    df <- year_data[[site_id]]
    if (is.null(colnames(df))) next
    all_columns_global <- unique(c(all_columns_global, colnames(df)))
  }
}

print("All possible columns across all years and sites:")
print(all_columns_global)

# We'll store each year’s standardized data here
all_years_data <- list()

# Now do the second pass with all_columns_global known
for (year in names(obs.mean)) {
  year_data <- obs.mean[[year]]  # This is a list of site dataframes
  
  # Create a temporary list to hold the standardized site data
  site_data_list <- lapply(seq_along(year_data), function(site_id) {
    df <- year_data[[site_id]]
    
    # Handle potential edge cases where df is null or has no columns
    if (is.null(df) || is.null(colnames(df))) {
      df <- data.frame(matrix(ncol = length(all_columns_global), nrow = 0))
      colnames(df) <- all_columns_global
    } else {
      # For each expected column, if it's missing, add it as NA
      missing_cols <- setdiff(all_columns_global, colnames(df))
      for (col in missing_cols) {
        df[[col]] <- NA
      }
      # Also ensure the column order matches all_columns_global
      df <- df[ , all_columns_global, drop = FALSE]
    }
    
    # Add Site_ID and convert to a data frame if needed
    df$Site_ID <- site_id
    return(df)
  })
  
  # Now safely bind rows for this year
  site_data <- do.call(rbind, site_data_list)
  # Add Year column
  site_data$Year <- year
  
  all_years_data[[year]] <- site_data
}

# Finally, combine all years into one big dataframe
obs_mean_df <- do.call(rbind, all_years_data)
print("Checking for list-type columns...")
list_columns <- sapply(obs_mean_df, is.list)
print(list_columns)

if (any(list_columns)) {
  print("⚠️ List-type columns found! Converting them to character.")
  obs_mean_df[list_columns] <- lapply(obs_mean_df[list_columns], function(x) sapply(x, toString))
}
# Write to CSV
write.csv(obs_mean_df, "/Users/shashankramachandran/Desktop/Research/observations/obs_mean.csv", row.names = FALSE)


# Quick check
print(dim(obs_mean_df))
print(head(obs_mean_df))




