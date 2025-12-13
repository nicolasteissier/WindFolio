import xarray as xr
import matplotlib.pyplot as plt

def plot(path: str, output_path: str):
    # Load data
    ds = xr.load_dataset(path, engine="h5netcdf")

    valid_mask = ~(ds["t2m"].isnull() | ds["u10"].isnull() | ds["v10"].isnull() | ds["sp"].isnull())
    ds = ds.where(valid_mask, drop=False)

    print(f"Dataset loaded with dimensions:\n{ds.coords}")
    print(f"Variables in dataset:\n{list(ds.data_vars)}")

    var1, var2 = "u10", "v10"

    # Prepare 3x3 subplots for 9 different times
    fig = plt.figure(figsize=(18, 12))

    print(ds.to_dataframe().head())

    time = ds.valid_time.values[0]
    data: xr.DataArray = ((ds[var1].sel(valid_time=time) ** 2) + (ds[var2].sel(valid_time=time) ** 2)) ** 0.5
    data_plot = data.plot.pcolormesh(
        ax=plt.gca(),
        x='longitude',
        y='latitude',
        cmap='viridis',
        add_colorbar=True,
        cbar_kwargs={'label': 'Wind Speed (m/s)'}
    )
    plt.title(f"Wind Speed at time index {time}")
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

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    PATH1 = "/Volumes/SSD/Epfl/Master/MA3/FBD/data/raw/weather/era5_land/hourly/2005_03_04.nc"
    PATH2 = "/Volumes/SSD/Epfl/Master/MA3/FBD/data/raw/weather/era5/hourly/2005_03.nc"
    PATH3 = "/Volumes/SSD/Epfl/Master/MA3/FBD/data/intermediate/nc/weather/era5_land/hourly/2005_03_04.nc"
    plot(PATH1, output_path="reports/figures/era5/map_projection_era5land.png")
    plot(PATH2, output_path="reports/figures/era5/map_projection_era5.png")
    plot(PATH3, output_path="reports/figures/era5/map_projection_era5land_masked.png")
    