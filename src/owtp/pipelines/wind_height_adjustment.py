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
        self.input_z0_dir = Path(self.config[target]['processed_data']) / "roughness" / "era5" / "hourly" / "roughness.parquet"
        self.output_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather_height_adjusted" / "era5_land" / "hourly"

        self.ORIGINAL_HEIGHT = 10  # meters
        self.TARGET_HEIGHT = 175   # meters

        self.use_real_z0 = True
        self.constant_alpha = 0.143  # Not used. We used it before having the real z0 data. (kept for debugging)

    def adjust_wind_height(self, n_workers=None, verbose=True):
        """Adjust wind heights from input weather data using alpha or z0 values."""

        if n_workers is None:
            n_workers = os.cpu_count() // 2 

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=4,
            memory_limit='30GB',
            processes=True,
            dashboard_address=':8787'
        )
        client = Client(cluster)
        
        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")

        try:
            ddf_weather = dd.read_parquet(
                self.input_weather_dir, 
                engine="pyarrow",
                split_row_groups="infer",
                aggregate_files="lon_bin",
                calculate_divisions=False
            )

            if verbose:
                print("Computing wind speed.")

            if verbose:
                print("Adjusting wind heights using terrain roughness values.")

            ddf_z0 = dd.read_parquet(self.input_z0_dir, engine="pyarrow")
            ddf_z0 = ddf_z0.repartition(npartitions=1)

            def merge_partition(partition_df, z0_df):
                """
                Merge function to add z0 values to the weather data partition.
                """
                if isinstance(z0_df, dd.DataFrame):
                    z0_df = z0_df.compute()

                partition_df['latitude'] = partition_df['latitude'].round(1)
                partition_df['longitude'] = partition_df['longitude'].round(1)
                z0_df['latitude'] = z0_df['latitude'].round(1)
                z0_df['longitude'] = z0_df['longitude'].round(1)

                ordered = partition_df.merge(z0_df, on=["latitude", "longitude"], how="left")
                
                wind_speed = np.sqrt(ordered['u10']**2 + ordered['v10']**2)
                
                log_ratio = np.log(self.TARGET_HEIGHT / ordered['fsr']) / np.log(self.ORIGINAL_HEIGHT / ordered['fsr'])
                adjusted_wind_speed = wind_speed * log_ratio
                
                return ordered[["valid_time", "latitude", "longitude", "lat_bin", "lon_bin"]].assign(ws=adjusted_wind_speed)

            
            meta_df = {
                'valid_time': 'datetime64[ns]',
                'latitude': 'f8',  
                'longitude': 'f8',
                'lat_bin': 'category',   
                'lon_bin': 'category',
                'ws': 'f8'
            }
            
            z0_df = ddf_z0.compute()

            ddf_adjusted = ddf_weather.map_partitions(
                merge_partition,
                z0_df=z0_df, 
                meta=meta_df
            )


            self.output_dir.mkdir(parents=True, exist_ok=True)

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


