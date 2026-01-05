from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import owtp.config as config
from datetime import datetime

COUNT_BIG_FOLDERS = True
LOG_FILE = Path(__file__).parent / "logs" / "revenues_hc.log"

def count_parquet_rows(directory: Path) -> int:
    """Fast row count using parquet metadata (no data loading)"""
    total = 0
    files = [f for f in directory.rglob("*.parquet") if not f.name.startswith("._")]
    for file in tqdm(files, desc=f"Reading {directory.name}"):
        metadata = pq.read_metadata(file)
        total += metadata.num_rows
    return total

cfg = config.load_yaml_config()

energy_dir = Path(cfg['paths']['processed_data']) / "parquet" / "energy" / "era5_land" / "hourly"
revenues_dir = Path(cfg['paths']['processed_data']) / "parquet" / "revenues" / "hourly"
prices_dir = Path(cfg['paths']['processed_data']) / "parquet" / "prices" / "hourly"

log_lines = []
log_lines.append(f"\n{'='*60}")
log_lines.append(f"Revenues Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_lines.append(f"{'='*60}")
log_lines.append(f"Prices dir: {prices_dir}")
log_lines.append(f"Energy dir: {energy_dir}")
log_lines.append(f"Revenues dir: {revenues_dir}")
log_lines.append("")

print("Revenues Health Check:\n")

print("Counting rows...")
n_prices = count_parquet_rows(prices_dir)
print(f"  Prices: {n_prices:,}")
log_lines.append(f"Prices: {n_prices:,}")

if COUNT_BIG_FOLDERS:
    n_energy = count_parquet_rows(energy_dir)
    print(f"  Energy: {n_energy:,}")
    log_lines.append(f"Energy: {n_energy:,}")
    n_revenues = count_parquet_rows(revenues_dir)
    print(f"  Revenues: {n_revenues:,}")
    log_lines.append(f"Revenues: {n_revenues:,}")
    
    if n_revenues == n_energy:
        print(f"\n  OK: Revenues == Energy")
        log_lines.append("Status: OK - Revenues == Energy")
    else:
        print(f"\n  /!\\ Revenues != Energy (diff: {abs(n_revenues - n_energy):,})")
        log_lines.append(f"Status: ERROR - Revenues != Energy (diff: {abs(n_revenues - n_energy):,})")

LOG_FILE.parent.mkdir(exist_ok=True)
with open(LOG_FILE, 'a') as f:
    f.write('\n'.join(log_lines) + '\n')

print(f"\nLogged to: {LOG_FILE}")
print("Done.")