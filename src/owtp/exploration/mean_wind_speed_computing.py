import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import os
from tqdm import tqdm
import xarray as xr


class MeanWindSpeedComputer:
    """
    Compute mean wind speed per location from partitioned weather data.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        
        self.input_weather_dir = Path(self.config[target]['intermediate_data']) / "parquet" / "weather" / "era5_land" / "hourly"

        self.output_wind_speed_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "wind_speed" / "mean"
        self.output_wind_speed_parquet_dir.mkdir(parents=True, exist_ok=True)

        self.output_wind_speed_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "wind_speed" / "mean"
        self.output_wind_speed_csv_dir.mkdir(parents=True, exist_ok=True)


    def compute_mean_wind_speed(self, n_workers=None, verbose=True):
        """
        Compute mean wind speed per location and save results.
        
        Outputs:
            - mean_wind_speed.parquet: Mean wind speed per location
            - location_mapping.parquet: Location identifiers and coordinates
        """
        
        if n_workers is None:
            n_workers = os.cpu_count() // 2

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit='30GB',
            processes=True,
            dashboard_address=':8788'
        )
        client = Client(cluster)

        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")

        try:
            if verbose:
                print(f"\nLoading weather data from {self.input_weather_dir}...")

            meta = [f for f in self.input_weather_dir.rglob("*.parquet") if f.name.startswith("._")]
            if verbose and len(meta) > 0:
                print(f"\nFound {len(meta)} system files to remove.")
            iterator = tqdm(meta, desc="Removing system files") if verbose else meta
            for meta_file in iterator:
                meta_file.unlink()

            parquet_files = sorted([f for f in self.input_weather_dir.rglob("*.parquet") if not f.name.startswith("._")])
            
            if verbose:
                print(f"\nFound {len(parquet_files)} parquet files")
            
            ddf_weather = dd.read_parquet(
                parquet_files,
                engine="pyarrow",
                blocksize="128MB",
            )

            if verbose:
                print(f"Loaded {ddf_weather.npartitions} partitions")
            
            ddf_weather['latitude'] = (ddf_weather['latitude'] * 10).round() / 10
            ddf_weather['longitude'] = (ddf_weather['longitude'] * 10).round() / 10
            
            if verbose:
                print("\nComputing wind speed from u10 and v10 components...")
            
            ddf_weather['wind_speed'] = np.sqrt(ddf_weather['u10']**2 + ddf_weather['v10']**2)
            
            if verbose:
                print("\nComputing mean wind speed per location...")
            
            def compute_partition_mean(partition_df):
                """
                Compute mean wind speed for all locations within a partition.
                This function operates on a single partition (pandas DataFrame).
                """
                mean_by_location = partition_df.groupby(['latitude', 'longitude']).agg(
                    mean_wind_speed=('wind_speed', 'mean')
                ).reset_index()
                
                return mean_by_location
            
            meta_df = {
                'latitude': np.float64,
                'longitude': np.float64,
                'mean_wind_speed': np.float64,
            }

            ddf_means = ddf_weather.map_partitions(
                compute_partition_mean,
                meta=meta_df
            )

            if verbose:
                print("\nCollecting results from all partitions...")
            
            mean_df = ddf_means.compute()
            
            if verbose:
                print(f"\nComputed mean wind speed for {len(mean_df)} locations")
            
            mean_df['location'] = mean_df['latitude'].astype(str) + '_' + mean_df['longitude'].astype(str)
            
            mean_df = mean_df[['location', 'latitude', 'longitude', 'mean_wind_speed']]
            
            mean_df = mean_df.sort_values('location').reset_index(drop=True)
            
            mean_wind_speed_path = self.output_wind_speed_parquet_dir / "mean_wind_speed.parquet"
            mean_df.to_parquet(mean_wind_speed_path, index=False)
            
            if verbose:
                print(f"\nSaved mean wind speed to {mean_wind_speed_path}")
                print(f"  - Min mean wind speed: {mean_df['mean_wind_speed'].min():.2f} m/s")
                print(f"  - Max mean wind speed: {mean_df['mean_wind_speed'].max():.2f} m/s")
                print(f"  - Avg mean wind speed: {mean_df['mean_wind_speed'].mean():.2f} m/s")
            
            mean_df.to_csv(self.output_wind_speed_csv_dir / "mean_wind_speed.csv", index=False)
            
            if verbose:
                print("\nMean wind speed also saved as CSV.")
                
        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    computer = MeanWindSpeedComputer(target="paths_local")
    computer.compute_mean_wind_speed(verbose=True)
