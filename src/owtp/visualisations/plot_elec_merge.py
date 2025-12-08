import matplotlib.pyplot as plt
import owtp.config
from pathlib import Path
import pandas as pd

RAW_EPEX_SPOT = [f"data/raw/elec/epexspot_{year}_repo.json" for year in range(2005, 2018 + 1)]
RAW_EPEX_SPORT_EMBER = "data/raw/elec/epexspot_ember_france.csv"

# Open and merge RAW_EPEX_SPOT files
epex_spot = pd.concat(
    pd.read_json(fp) for fp in RAW_EPEX_SPOT
).reset_index(drop=True)