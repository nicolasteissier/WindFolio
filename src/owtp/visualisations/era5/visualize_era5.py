import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import geodatasets
import numpy as np

PATH = "data/raw/weather/era5_land/hourly/2005_04.nc"

# Load data
ds = xr.open_dataset(PATH, engine="scipy")

valid_mask = ~(ds["t2m"].isnull() | ds["u10"].isnull() | ds["v10"].isnull() | ds["sp"].isnull())
ds = ds.where(valid_mask, drop=False)

print(f"Dataset loaded with dimensions:\n{ds.coords}")

var1, var2 = "u10", "v10"

# Prepare 3x3 subplots for 9 different times
fig = plt.figure(figsize=(18, 12))

time = 0
data: xr.DataArray = np.sqrt((ds[var1].isel(valid_time=time) ** 2) + (ds[var2].isel(valid_time=time) ** 2)) # type: ignore
im = data.plot(cmap='viridis', add_colorbar=False) # type: ignore
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)

# Print of one square, in degrees
latitudes = ds['latitude'].values
longitudes = ds['longitude'].values
lat_step = abs(latitudes[1] - latitudes[0])
lon_step = abs(longitudes[1] - longitudes[0])
print(f"Each grid square represents approximately {lat_step:.2f}° latitude by {lon_step:.2f}° longitude.")

spatial_dims = (ds.sizes['latitude'], ds.sizes['longitude'])
plt.suptitle(f"Wind speed map projection ({'x'.join(map(str, spatial_dims))})", fontsize=16)

plt.savefig("reports/figures/era5/map_projection.png", dpi=300, bbox_inches='tight')
plt.close()