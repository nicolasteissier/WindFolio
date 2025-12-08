import matplotlib.pyplot as plt
import owtp.config
from pathlib import Path
import pandas as pd

RAW_EPEX_SPOT = [f"data/raw/elec/epexspot_{year}_repo.json" for year in range(2005, 2018 + 1)]
RAW_EPEX_SPOT_EMBER = "data/raw/elec/epexspot_ember_france.csv"
MERGED = 'data/intermediate/parquet/prices/hourly/prices.parquet'

# Open and merge RAW_EPEX_SPOT files
epex_spot = pd.concat(
    pd.read_json(fp) for fp in RAW_EPEX_SPOT
).reset_index()

epex_spot["startDate"] = pd.to_datetime(
    epex_spot["startDate"].astype(str),  # make sure it's strings
    utc=True,                           # interpret offsets like +02:00
    errors="coerce"                     # invalid -> NaT instead of crash
)
epex_spot.set_index('startDate', inplace=True)

epex_spot_ember = pd.read_csv(RAW_EPEX_SPOT_EMBER)

epex_spot_ember["Datetime (UTC)"] = pd.to_datetime(
    epex_spot_ember["Datetime (UTC)"].astype(str),
    utc=True,
    errors="coerce"
)
epex_spot_ember.set_index('Datetime (UTC)', inplace=True)

merged_data = pd.read_parquet(MERGED)
merged_data.index = pd.to_datetime(merged_data.index, utc=True)

# Print date ranges
print(f"EPEX Spot data range: {epex_spot.index.min()} to {epex_spot.index.max()}")
print(f"EPEX Spot Ember data range: {epex_spot_ember.index.min()} to {epex_spot_ember.index.max()}")

plot_start = pd.to_datetime("2005-05-01 00:00:00+00:00", utc=True)
plot_end = pd.to_datetime("2025-12-01 00:00:00+00:00", utc=True)

LINEWIDTH = .5

fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharey=True, sharex=True)

axes[0].plot(epex_spot['price_euros_mwh'].loc[plot_start:].index, epex_spot['price_euros_mwh'].loc[plot_start:], label='EPEX Spot Prices (Repo)', linewidth=LINEWIDTH)
axes[0].set_title("Ewoken/EPEX-Spot-Data (GitHub Repository)")

axes[1].plot(epex_spot_ember['Price (EUR/MWhe)'].loc[:plot_end].index, epex_spot_ember['Price (EUR/MWhe)'].loc[:plot_end], label='EPEX Spot Prices (Ember)', linewidth=LINEWIDTH)
axes[1].set_title("European Wholesale Electricity Price Data - Ember (France)")
axes[1].set_ylabel("Price (EUR/MWhe)")

# Find overlap region
overlap_start = max(epex_spot.index.min(), epex_spot_ember.index.min())
overlap_end = min(epex_spot.index.max(), epex_spot_ember.index.max())

# Add pastel translucent background for overlap
axes[0].axvspan(
    overlap_start, overlap_end,
    color='lavender', alpha=0.6, label='Overlap'
)
axes[1].axvspan(
    overlap_start, overlap_end,
    color='lavender', alpha=0.6, label='Overlap'
)
axes[2].axvspan(
    overlap_start, overlap_end,
    color='lavender', alpha=0.6, label='Overlap'
)


axes[2].plot(merged_data['Price (EUR/MWhe)'].loc[plot_start:plot_end].index, merged_data['Price (EUR/MWhe)'].loc[plot_start:plot_end], color='green', label='Merged EPEX Spot Prices', linewidth=LINEWIDTH)
axes[2].set_title("Merged Prices")

import matplotlib.patches as mpatches

overlap_patch = mpatches.Patch(color='lavender', alpha=0.6, label='Overlapping Period')
fig.legend(handles=[overlap_patch], loc='lower center', ncol=1, frameon=False)

plt.xlabel("Time")
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Increase bottom margin for legend

plt.savefig("reports/figures/prices/epex_spot_prices.png", dpi=300)