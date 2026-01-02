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

    def __init__(self, target: Literal["paths", "paths_local"], adjusted_height: bool = True):
        self.config = owtp.config.load_yaml_config()
        
        # Use intermediate data (masked parquet files by date)
        self.input_weather_dir = Path(self.config[target]['intermediate_data']) / "parquet" / "weather" / "era5_land" / "hourly"

        self.input_weather_100m_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather" / "era5_land_100m" / "hourly"

        self.output_wind_speed_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "wind_speed" / "mean"
        self.output_wind_speed_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.output_wind_speed_100m_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "wind_speed_100m" / "mean"
        self.output_wind_speed_100m_parquet_dir.mkdir(parents=True, exist_ok=True)

        self.output_wind_speed_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "wind_speed" / "mean"
        self.output_wind_speed_csv_dir.mkdir(parents=True, exist_ok=True)
        self.output_wind_speed_100m_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "wind_speed_100m" / "mean"
        self.output_wind_speed_100m_csv_dir.mkdir(parents=True, exist_ok=True)

        self.adjusted_height = adjusted_height

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
            exist = self.input_weather_100m_dir.exists()
            print(f"EXIST: {exist}")
        except Exception as e:
            print(f"Error checking existence of directory: {e}")

        try:
            

            # Remove system files
            if self.adjusted_height:
                meta = [f for f in self.input_weather_100m_dir.rglob("*.parquet") if f.name.startswith("._")]
            else:
                meta = [f for f in self.input_weather_dir.rglob("*.parquet") if f.name.startswith("._")]

            print(len(meta))

            if verbose:
                print(f"\nLoading weather data from {self.input_weather_100m_dir}...")

            if verbose and len(meta) > 0:
                print(f"\nFound {len(meta)} system files to remove.")
            iterator = tqdm(meta, desc="Removing system files") if verbose else meta
            for meta_file in iterator:
                meta_file.unlink()

            # Get all parquet files
            if self.adjusted_height:
                parquet_files = sorted([f for f in self.input_weather_100m_dir.rglob("*.parquet")])
            else:
                parquet_files = sorted([f for f in self.input_weather_dir.rglob("*.parquet")])
            
            if verbose:
                print(f"\nFound {len(parquet_files)} parquet files")
            
            # Read intermediate parquet files (not partitioned)
            ddf_weather = dd.read_parquet(
                parquet_files,
                engine="pyarrow",
                blocksize="128MB",
            )

            if verbose:
                print(f"Loaded {ddf_weather.npartitions} partitions")
            
            # Round coordinates to 0.1 degree for consistent precision (as done in weather_data_preprocessing.py)
            ddf_weather['latitude'] = (ddf_weather['latitude'] * 10).round() / 10
            ddf_weather['longitude'] = (ddf_weather['longitude'] * 10).round() / 10
            
            if verbose:
                print("\nComputing wind speed from u10 and v10 components...")
            
            if not self.adjusted_height:
                # Compute wind speed from u10 and v10
                ddf_weather['wind_speed'] = np.sqrt(ddf_weather['u10']**2 + ddf_weather['v10']**2)
            
            if verbose:
                print("\nComputing mean wind speed per location...")
            
            # Process each partition independently
            def compute_partition_mean(partition_df):
                """
                Compute mean wind speed for all locations within a partition.
                This function operates on a single partition (pandas DataFrame).
                """
                # Group by exact location within this partition
                if not self.adjusted_height:
                    partition_df = partition_df[['latitude', 'longitude', 'wind_speed']]
                else:
                    partition_df = partition_df[['latitude', 'longitude', 'ws_100m']].rename(columns={'ws_100m': 'wind_speed'})
                
                mean_by_location = partition_df.groupby(['latitude', 'longitude']).agg(
                    mean_wind_speed=('wind_speed', 'mean')
                ).reset_index()
                
                return mean_by_location
            
            # Define output metadata
            meta_df = {
                'latitude': np.float64,
                'longitude': np.float64,
                'mean_wind_speed': np.float64,
            }

            # Apply to all partitions
            ddf_means = ddf_weather.map_partitions(
                compute_partition_mean,
                meta=meta_df
            )

            if verbose:
                print("\nCollecting results from all partitions...")
            
            # Compute and collect all partition means
            mean_df = ddf_means.compute()
            
            if verbose:
                print(f"\nComputed mean wind speed for {len(mean_df)} locations")
            
            # Add location identifier
            mean_df['location'] = mean_df['latitude'].astype(str) + '_' + mean_df['longitude'].astype(str)
            
            # Reorder columns
            mean_df = mean_df[['location', 'latitude', 'longitude', 'mean_wind_speed']]
            
            # Sort by location for consistency
            mean_df = mean_df.sort_values('location').reset_index(drop=True)
            
            # Save mean wind speed (including bins for reference)
            if self.adjusted_height:
                mean_wind_speed_path = self.output_wind_speed_100m_parquet_dir / "mean_wind_speed.parquet"
            else:
                mean_wind_speed_path = self.output_wind_speed_parquet_dir / "mean_wind_speed.parquet"
            mean_df.to_parquet(mean_wind_speed_path, index=False)
            
            if verbose:
                print(f"\nSaved mean wind speed to {mean_wind_speed_path}")
                print(f"  - Min mean wind speed: {mean_df['mean_wind_speed'].min():.2f} m/s")
                print(f"  - Max mean wind speed: {mean_df['mean_wind_speed'].max():.2f} m/s")
                print(f"  - Avg mean wind speed: {mean_df['mean_wind_speed'].mean():.2f} m/s")
            
            if self.adjusted_height:
                mean_df.to_csv(self.output_wind_speed_100m_csv_dir / "mean_wind_speed.csv", index=False)
            else:
                mean_df.to_csv(self.output_wind_speed_csv_dir / "mean_wind_speed.csv", index=False)
            
            if verbose:
                print("\nMean wind speed also saved as CSV.")
        except Exception as e:
            print(f"Error during mean wind speed computation: {e}")
                
        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    computer = MeanWindSpeedComputer(target="paths_local", adjusted_height=True)
    computer.compute_mean_wind_speed(verbose=True)
