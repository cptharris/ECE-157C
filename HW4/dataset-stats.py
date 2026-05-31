import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
datasets_dir = "datasets/"

# ==================== LOAD ALL CSV FILES ====================
print("=" * 70)
print("LOADING CSV FILES")
print("=" * 70)

csv_files = sorted(Path(datasets_dir).glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in {datasets_dir}")
    exit()

dataframes = {}
for file_path in csv_files:
    df = pd.read_csv(file_path)
    filename = file_path.name
    dataframes[filename] = df
    print(f"\n{filename}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

# ==================== CONCATENATE ALL DATASETS ====================
print("\n" + "=" * 70)
print("CONCATENATED DATASET")
print("=" * 70)

combined_df = pd.concat(dataframes.values(), ignore_index=True)
print(f"\nTotal rows after concatenation: {len(combined_df)}")
print(f"Total columns: {len(combined_df.columns)}")

# ==================== TICKER ANALYSIS ====================
print("\n" + "=" * 70)
print("TICKER ANALYSIS")
print("=" * 70)

if "Ticker" in combined_df.columns:
    distinct_tickers = combined_df["Ticker"].nunique()
    print(f"\nDistinct Tickers: {distinct_tickers}")
    
    # Tickers present every year
    tickers_by_file = {}
    for filename, df in dataframes.items():
        if "Ticker" in df.columns:
            tickers_by_file[filename] = set(df["Ticker"].dropna().unique())
    
    # Find intersection (tickers in all files)
    all_files = list(tickers_by_file.keys())
    if all_files:
        tickers_in_all = set.intersection(*tickers_by_file.values())
        percentage_in_all = (len(tickers_in_all) / distinct_tickers) * 100
        print(f"Tickers present in ALL files: {len(tickers_in_all)}")
        print(f"Percentage of Tickers present every year: {percentage_in_all:.2f}%")
else:
    print("'Ticker' column not found in dataset")

# ==================== SECTOR ANALYSIS ====================
print("\n" + "=" * 70)
print("SECTOR ANALYSIS")
print("=" * 70)

if "Sector" in combined_df.columns:
    sector_counts = combined_df["Sector"].value_counts()
    total_non_null = combined_df["Sector"].notna().sum()
    
    if total_non_null > 0:
        most_represented = sector_counts.index[0]
        most_represented_count = sector_counts.iloc[0]
        most_represented_pct = (most_represented_count / total_non_null) * 100
        
        print(f"\nMost represented Sector: {most_represented}")
        print(f"  Count: {most_represented_count}")
        print(f"  Percentage: {most_represented_pct:.2f}%")
        
        print(f"\nTop 5 Sectors by representation:")
        for idx, (sector, count) in enumerate(sector_counts.head(5).items(), 1):
            pct = (count / total_non_null) * 100
            print(f"  {idx}. {sector}: {count} ({pct:.2f}%)")
    else:
        print("No non-null Sector values found")
else:
    print("'Sector' column not found in dataset")

# ==================== MISSING DATA ANALYSIS ====================
print("\n" + "=" * 70)
print("MISSING DATA ANALYSIS")
print("=" * 70)

total_cells = len(combined_df) * len(combined_df.columns)
missing_cells = combined_df.isna().sum().sum()
missing_percentage = (missing_cells / total_cells) * 100

print(f"\nTotal cells: {total_cells:,}")
print(f"Missing cells (NaN): {missing_cells:,}")
print(f"Percentage of cells missing: {missing_percentage:.2f}%")

print(f"\nMissing data by column:")
missing_by_column = combined_df.isna().sum().sort_values(ascending=False)
for column, count in missing_by_column[missing_by_column > 0].items():
    col_percentage = (count / len(combined_df)) * 100
    print(f"  {column}: {count} ({col_percentage:.2f}%)")

print("\n" + "=" * 70)
