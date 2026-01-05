# Check that 
# - data/raw/weather/era5/hourly/
# - data/processed/parquet/weather/era5/hourly/ 
# contain the same number of entries (lines)
from tqdm import tqdm
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/weather/era5_land/hourly/")
INTERMEDIATE_DIR = Path("data/intermediate/parquet/weather/era5_land/hourly/")
PROCESSED_DIR = Path("data/processed/parquet/weather/era5_land/hourly/")

raw_dir_files = [f for f in RAW_DIR.glob("*.nc") if not f.name.startswith("._") and not f.name.startswith("new_")]
intermediate_dir_files = [f for f in INTERMEDIATE_DIR.glob("*.parquet") if not f.name.startswith("._")]
processed_dir_files = [f for f in PROCESSED_DIR.glob("*/*/*.parquet") if not f.name.startswith("._")]

print(f"Found {len(raw_dir_files)} raw .nc files.")
print(f"Found {len(intermediate_dir_files)} intermediate .parquet files.")
print(f"Found {len(processed_dir_files)} processed .parquet files.")

def count_lines_in_nc(file_path: Path) -> int:
    import xarray as xr
    ds = xr.open_dataset(file_path)
    return ds.sizes['valid_time'] * ds.sizes['latitude'] * ds.sizes['longitude']

def count_lines_in_csv(file_path: Path) -> int:
    df = pd.read_csv(file_path)
    return len(df)

def count_lines_in_parquet(file_path: Path) -> int:
    df = pd.read_parquet(file_path)
    return len(df)

total_raw_lines = 0
for raw_file in tqdm(raw_dir_files, desc="Counting lines in raw .nc files"):
    total_raw_lines += count_lines_in_nc(raw_file)

total_intermediate_lines = 0
for intermediate_file in tqdm(intermediate_dir_files, desc="Counting lines in intermediate .parquet files"):
    total_intermediate_lines += count_lines_in_parquet(intermediate_file)

total_processed_lines = 0
for processed_file in tqdm(processed_dir_files, desc="Counting lines in processed .parquet files"):
    total_processed_lines += count_lines_in_parquet(processed_file)

print(f"Total lines in raw .nc files: {total_raw_lines}")
print(f"Total lines in intermediate .parquet files: {total_intermediate_lines}")
print(f"Total lines in processed .parquet files: {total_processed_lines}")

if total_raw_lines == total_intermediate_lines:
    print("Health check PASSED: Raw and intermediate line counts match!")
else:
    print("Health check FAILED: Raw and intermediate line counts do not match!")

if total_intermediate_lines == total_processed_lines:
    print("Health check PASSED: Intermediate and processed line counts match!")
else:
    print("Health check FAILED: Intermediate and processed line counts do not match!")