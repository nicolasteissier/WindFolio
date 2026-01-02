import owtp.config
from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl
from turbine_models.parser import Turbines
from turbine_models.tools.extract_power_curve import extract_power_curve
from turbine_models.tools.power_curve_tools import plot_power_curve
from turbine_models.tools.library_tools import check_turbine_library_for_turbine
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import os



class WindHeightAdjustment:
    """
    Adjust wind heights from weather data.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_weather_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather" / "era5_land" / "hourly"
        self.input_alpha_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather" / "era5_land" / "hourly_alpha"
        self.output_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather" / "era5_land_100m" / "hourly"

        self.ORIGINAL_HEIGHT = 10  # meters
        self.TARGET_HEIGHT = 100   # meters

        self.use_real_z0 = False
        self.constant_alpha = 0.143  # 1/7th power law

    def adjust_wind_height(self, n_workers=None, verbose=True):
        """Adjust wind heights from input weather data using alpha or z0 values."""

        if n_workers is None:
            n_workers = os.cpu_count() // 2 # type: ignore

        # Distributed scheduler for performance monitoring and memory management
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit='2GB',
            processes=True,
            dashboard_address=':8787'
        )
        client = Client(cluster)
        
        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")

        try:
            ddf_weather = dd.read_parquet(self.input_weather_dir, engine="pyarrow")

            if verbose:
                print("Computing wind speed.")

            wind_speed = np.sqrt(np.pow(ddf_weather['u10'], 2) + np.pow(ddf_weather['v10'], 2))

            if verbose:
                print("Adjusting wind heights using alpha values.")

            if self.use_real_z0:
                ddf_z0 = dd.read_parquet(self.input_alpha_dir, engine="pyarrow")
                # Join to have the z0 values aligned
                ordered = ddf_weather.merge(ddf_z0, on=["lat_bin", "lon_bin", "valid_time", "latitude", "longitude"], how="left")
                adjusted_wind_speed = wind_speed * (np.log(self.TARGET_HEIGHT / ordered['z0']) / np.log(self.ORIGINAL_HEIGHT / ordered['z0']))
            else:
                # Use a constant alpha value if real alpha values are not available
                adjusted_wind_speed = wind_speed * (self.TARGET_HEIGHT / self.ORIGINAL_HEIGHT) ** self.constant_alpha

            ddf_adjusted = ddf_weather[["lat_bin", "lon_bin", "valid_time", "latitude", "longitude"]].assign(ws_100m=adjusted_wind_speed)

            ddf_adjusted.to_parquet(
                self.output_dir,
                engine="pyarrow",
                compression="zstd",
                write_index=False,
                partition_on=["lat_bin", "lon_bin"],
            )

            if verbose:
                print("Done writing adjusted wind height Parquet files.")

        finally:
            client.close()
            cluster.close()
    
if __name__ == "__main__":
    adjuster = WindHeightAdjustment(target="paths_local")
    adjuster.adjust_wind_height(verbose=True)