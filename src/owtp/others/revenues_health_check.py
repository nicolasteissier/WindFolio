from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import owtp.config as config

COUNT_BIG_FOLDERS = False

def count_parquet_rows(directory: Path) -> int:
    """Fast row count using parquet metadata (no data loading)"""
    total = 0
    files = [f for f in directory.rglob("*.parquet") if not f.name.startswith("._")]
    for file in tqdm(files, desc=f"Reading {directory.name}"):
        metadata = pq.read_metadata(file)
        total += metadata.num_rows
    return total

cfg = config.load_yaml_config()

energy_dir = Path(cfg['paths']['processed_data']) / "parquet" / "energy" / "era5_land" / "hourly"       # = 1,115,035,200 in last run (15.12.2025)
prices_dir = Path(cfg['paths']['processed_data']) / "parquet" / "prices" / "hourly"                     # = 180,746 in last run (15.12.2025)
revenues_dir = Path(cfg['paths']['processed_data']) / "parquet" / "revenues" / "hourly"                 # = 1,097,424,360 in last run (15.12.2025)

print("Revenues Health Check:\n")

# Count rows (fast - only reads metadata)
print("\nCounting rows...")
if COUNT_BIG_FOLDERS:
    n_energy = count_parquet_rows(energy_dir)
    print(f"  Energy:   {n_energy:,}")
    n_revenues = count_parquet_rows(revenues_dir)
    print(f"  Revenues: {n_revenues:,}")

n_prices = count_parquet_rows(prices_dir)
print(f"  Prices:   {n_prices:,}")
    
# Check prices time range
print("\nPrices time range:")
df_prices = pd.read_parquet(prices_dir / "prices.parquet")
if 'time' not in df_prices.columns:
    df_prices = df_prices.reset_index()
df_prices['time'] = pd.to_datetime(df_prices['time'])
print(f"  First: {df_prices['time'].min()}")
print(f"  Last:  {df_prices['time'].max()}")

# Check prices missing values
n_missing_prices = df_prices['Price (EUR/MWhe)'].isna().sum()
print(f"\nNumber of missing prices entries: {n_missing_prices}")

# Find missing indices in prices
if n_missing_prices > 0:
    missing_prices = df_prices[df_prices['Price (EUR/MWhe)'].isna()]
    print("Missing prices entries:")
    print(missing_prices)

# Check for missing dates in time series (gaps)
print("\nChecking for missing dates in prices time series:")
df_prices_sorted = df_prices.sort_values('time')
date_range = pd.date_range(
    start=df_prices_sorted['time'].min(),
    end=df_prices_sorted['time'].max(),
    freq='h'
)
existing_dates = set(df_prices_sorted['time'])
expected_dates = set(date_range)
missing_dates = sorted(expected_dates - existing_dates)

if missing_dates:
    print(f"  Found {len(missing_dates):,} missing dates!")
    print(f"  Missing: {missing_dates}")
else:
    print("  ✓ No missing dates - time series is complete!")

if COUNT_BIG_FOLDERS:
    if n_revenues != n_energy:
        print("\n  Health check FAILED: Number of revenues entries does not match number of energy entries!")
    else:
        print("\n  Health check PASSED: Number of revenues entries matches number of energy entries!")

print("\nDone.")