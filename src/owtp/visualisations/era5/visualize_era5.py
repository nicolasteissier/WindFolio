import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import geodatasets
import numpy as np

PATH = "data/raw/weather/era5/hourly/2005_01.nc"

# Load world boundaries using geodatasets
world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
# Load data
ds = xr.open_dataset(PATH)
var1, var2 = list(ds.data_vars)[1:]  # Select the wind u and v components

# Prepare 3x3 subplots for 9 different times
fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
times = np.linspace(0, ds.sizes['valid_time'] - 1, 9, dtype=int)

for idx, ax in enumerate(axes.flat):
    time = times[idx]
    data = np.sqrt((ds[var1].isel(valid_time=time) ** 2) + (ds[var2].isel(valid_time=time) ** 2))
    im = data.plot(ax=ax, cmap='viridis', add_colorbar=False)
    world.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title(f"Time index: {time}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)

spatial_dims = (ds.sizes['latitude'], ds.sizes['longitude'])
plt.suptitle(f"Wind speed map projection ({'x'.join(map(str, spatial_dims))})", fontsize=16)
plt.savefig("reports/figures/era5_map_projection.png", dpi=300, bbox_inches='tight')

# Print the size in km of the spatial dimensions
latitudes = ds['latitude'].values
longitudes = ds['longitude'].values
lat_km = (latitudes.max() - latitudes.min()) * 111  # Approx. conversion factor
lon_km = (longitudes.max() - longitudes.min()) * 111 * np.cos(np.deg2rad(latitudes.mean()))  # Adjust for latitude
print(f"Spatial dimensions of one measurement: {lat_km/spatial_dims[0]:.2f} km (lat) x {lon_km/spatial_dims[1]:.2f} km (lon)")