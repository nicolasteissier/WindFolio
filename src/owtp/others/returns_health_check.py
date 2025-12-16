from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import owtp.config as config

COUNT_BIG_FOLDERS = True

def count_parquet_rows(directory: Path) -> int:
    total = 0
    files = [f for f in directory.rglob("*.parquet") if not f.name.startswith("._")]
    for file in tqdm(files, desc=f"Reading {directory.name}"):
        metadata = pq.read_metadata(file)
        total += metadata.num_rows
    return total

cfg = config.load_yaml_config()

revenues_dir = Path(cfg['paths']['processed_data']) / "parquet" / "revenues" / "hourly"
returns_dir = Path(cfg['paths']['processed_data']) / "parquet" / "returns" / "hourly"

print("Returns Health Check:\n")

print("Counting rows...")
n_returns = count_parquet_rows(returns_dir)
print(f"  Returns: {n_returns:,}")

if COUNT_BIG_FOLDERS:
    n_revenues = count_parquet_rows(revenues_dir)
    print(f"  Revenues: {n_revenues:,}")
    
    if n_returns < n_revenues:
        diff = n_revenues - n_returns
        pct = (diff / n_revenues) * 100
        print(f"\n  OK: Returns < Revenues (expected)")
        print(f"    Filtered: {diff:,} rows ({pct:.2f}%)")
    else:
        print(f"\n  /!\ WARNING: Returns >= Revenues (unexpected!)")

print("\nDone.")