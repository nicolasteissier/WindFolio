from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import owtp.config as config
from datetime import datetime

COUNT_BIG_FOLDERS = True
LOG_FILE = Path(__file__).parent / "logs" / "returns_hc.log"

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

log_lines = []
log_lines.append(f"\n{'='*60}")
log_lines.append(f"Returns Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_lines.append(f"{'='*60}")
log_lines.append(f"Returns dir: {returns_dir}")
log_lines.append(f"Revenues dir: {revenues_dir}")
log_lines.append("")

print("Returns Health Check:\n")

print("Counting rows...")
n_returns = count_parquet_rows(returns_dir)
print(f"  Returns: {n_returns:,}")
log_lines.append(f"Returns: {n_returns:,}")

if COUNT_BIG_FOLDERS:
    n_revenues = count_parquet_rows(revenues_dir)
    print(f"  Revenues: {n_revenues:,}")
    log_lines.append(f"Revenues: {n_revenues:,}")
    
    if n_returns < n_revenues:
        diff = n_revenues - n_returns
        pct = (diff / n_revenues) * 100
        print(f"\n  OK: Returns < Revenues (expected)")
        print(f"    Filtered: {diff:,} rows ({pct:.2f}%)")
        log_lines.append(f"Status: OK - Returns < Revenues")
        log_lines.append(f"Filtered: {diff:,} rows ({pct:.2f}%)")
    else:
        print(f"\n  /!\\ WARNING: Returns >= Revenues (unexpected!)")
        log_lines.append(f"Status: WARNING - Returns >= Revenues (unexpected!)")

LOG_FILE.parent.mkdir(exist_ok=True)
with open(LOG_FILE, 'a') as f:
    f.write('\n'.join(log_lines) + '\n')

print(f"\nLogged to: {LOG_FILE}")
print("Done.")