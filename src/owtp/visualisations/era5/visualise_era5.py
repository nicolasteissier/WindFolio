import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(path: str, output_path: str, dataset):
    # Load data
    print(f"{dataset} - dataset loading from {path}...")

    if path.endswith(".nc"):
        ds = xr.load_dataset(path, engine="h5netcdf")
        print(f"{dataset} - dataset loaded.")
        var1, var2 = "u10", "v10"
        valid_mask = ~(ds["t2m"].isnull() | ds["u10"].isnull() | ds["v10"].isnull() | ds["sp"].isnull())
        ds = ds.where(valid_mask, drop=False)
        spatial_dims = (ds.sizes['latitude'], ds.sizes['longitude'])
        time = ds.valid_time.values[0]
        data: xr.DataArray = ((ds[var1].sel(valid_time=time) ** 2) + (ds[var2].sel(valid_time=time) ** 2)) ** 0.5
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
        print(f"{dataset} - dataset loaded.")
        # keep the first time only for plotting
        first_time = df['valid_time'].min()
        df_time = df[df['valid_time'] == first_time]
        
        # Round coordinates to match grid precision
        df_time['latitude'] = df_time['latitude'].round(1)
        df_time['longitude'] = df_time['longitude'].round(1)
        
        # Define full regular grid with 0.1 step
        full_lats = np.arange(41.3, 51.2, 0.1).round(1)
        full_lons = np.arange(-5.2, 9.7, 0.1).round(1)
        spatial_dims = (len(full_lats), len(full_lons))
        time = first_time
        
        # df with lat as rows and lon as columns, values as wind speed
        # Pivot the DataFrame to have shape (lat, lon), filling missing values with NaN
        interim_data_u10 = df_time.pivot(index='latitude', columns='longitude', values='u10')
        interim_data_v10 = df_time.pivot(index='latitude', columns='longitude', values='v10')
        
        # Reindex to full regular grid, fill missing with NaN
        interim_data_u10 = interim_data_u10.reindex(index=full_lats, columns=full_lons)
        interim_data_v10 = interim_data_v10.reindex(index=full_lats, columns=full_lons)
        wind_speed_values = (interim_data_u10.values ** 2 + interim_data_v10.values ** 2) ** 0.5
        data = xr.DataArray(
            wind_speed_values,
            coords={'latitude': full_lats, 'longitude': full_lons},
            dims=['latitude', 'longitude']
        )
    else:
        raise ValueError("Unsupported file format. Please provide a .nc or .parquet file.")

    # Prepare figure with explicit axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect(spatial_dims[1] / spatial_dims[0])

    print(f"{dataset} - plotting wind speed map projection...")
    data_plot = data.plot.pcolormesh(
        ax=ax,
        x='longitude',
        y='latitude',
        cmap='viridis',
        add_colorbar=False
    )
    
    # Add colorbar with proper sizing
    cbar = fig.colorbar(data_plot, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('Wind Speed (m/s)')
    
    ax.set_title(f"{dataset} Wind Speed map projection ({'x'.join(map(str, spatial_dims))} points)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)

    # Print of one square, in degrees
    latitudes = data['latitude'].values
    longitudes = data['longitude'].values
    lat_step = abs(latitudes[1] - latitudes[0])
    lon_step = abs(longitudes[1] - longitudes[0])
    print(f"{dataset} - Each square: {lat_step:.2f}° lat by {lon_step:.2f}° lon.")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    PATH1 = "/Volumes/SSD/Epfl/Master/MA3/FBD/data/raw/weather/era5_land/hourly/2005_03_04.nc"
    PATH2 = "/Volumes/SSD/Epfl/Master/MA3/FBD/data/raw/weather/era5/hourly/2005_03.nc"
    PATH3 = "/Volumes/SSD/Epfl/Master/MA3/FBD/data/intermediate/parquet/weather/era5_land/hourly/2005_03_04.parquet"
    plot(PATH3, output_path="reports/figures/era5/map_projection_era5land_masked.png", dataset="ERA5-Land Masked")
    plot(PATH1, output_path="reports/figures/era5/map_projection_era5land.png", dataset="ERA5-Land")
    plot(PATH2, output_path="reports/figures/era5/map_projection_era5.png", dataset="ERA5")
    