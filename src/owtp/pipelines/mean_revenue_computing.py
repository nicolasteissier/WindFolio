import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import os
from tqdm import tqdm

from owtp.others import rolling

class MeanRevenueComputer:
    """
    Compute mean revenue per location from partitioned revenues data.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_revenues_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "hourly"

        self.output_revenues_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "mean"
        self.output_revenues_parquet_dir.mkdir(parents=True, exist_ok=True)

        self.location_mapping_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "locations"
        self.location_mapping_parquet_dir.mkdir(parents=True, exist_ok=True)

    def compute_mean_revenue(self, n_workers=None, verbose=True):
        """
        Compute mean revenue per location and save results.
        
        Outputs:
            - mean_revenue.parquet: Mean revenue per location
            - location_mapping.parquet: Location identifiers and coordinates
        """
        
        if n_workers is None:
            n_workers = os.cpu_count() // 2 # type: ignore

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit='4GB',
            processes=True,
            dashboard_address=':8788'
        )
        client = Client(cluster)

        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")

        try:
            if verbose:
                print(f"\nLoading revenues data from {self.input_revenues_dir}...")

            meta = [f for f in self.input_revenues_dir.rglob("*.parquet") if f.name.startswith("._")]
            if verbose and len(meta) > 0:
                print(f"\nFound {len(meta)} system files to remove.")
            iterator = tqdm(meta, desc="Removing system files") if verbose else meta
            for meta_file in iterator:
                meta_file.unlink()

            # Read parquet with explicit divisions to respect partitioning
            ddf_revenues = dd.read_parquet(
                self.input_revenues_dir,
                engine="pyarrow",
                split_row_groups="infer",
                aggregate_files="lon_bin",  # Use partitioning columns to aggregate files
                calculate_divisions=False  # Faster since we don't need sorted divisions
            )

            if verbose:
                print(f"\nFound {ddf_revenues.npartitions} partitions in revenue data.")
            
            # Convert categorical partition columns back to numeric
            ddf_revenues['lat_bin'] = ddf_revenues['lat_bin'].map_partitions(
                lambda s: s.cat.categories[s.cat.codes].astype(float).astype(np.int64),
                meta=('lat_bin', np.int64)
            )
            ddf_revenues['lon_bin'] = ddf_revenues['lon_bin'].map_partitions(
                lambda s: s.cat.categories[s.cat.codes].astype(float).astype(np.int64),
                meta=('lon_bin', np.int64)
            )
            
            if verbose:
                print("\nComputing mean revenue per location...")
            
            # Check if partitioning columns exist
            has_partitioning = 'lat_bin' in ddf_revenues.columns and 'lon_bin' in ddf_revenues.columns

            if verbose and not has_partitioning:
                raise ValueError("Partitioning columns 'lat_bin' and 'lon_bin' not found in revenue data.")
            
            for window in rolling.get_windows(
                ddf_revenues['time'].min().compute(),
                ddf_revenues['time'].max().compute()
                ):
                # Process each partition independently
                def compute_partition_mean(partition_df, window_start, window_end):
                    """
                    Compute mean revenue for all locations within a partition.
                    This function operates on a single partition (pandas DataFrame).
                    """
                    mask = (partition_df['time'] >= window_start) & (partition_df['time'] <= window_end)
                    # Group by exact location within this partition
                    mean_by_location = partition_df.loc[mask].groupby(['latitude', 'longitude']).agg(
                        mean_revenue=('revenue', 'mean'),
                        lat_bin=('lat_bin', 'first'),
                        lon_bin=('lon_bin', 'first')
                    ).reset_index()
                    
                    return mean_by_location
                
                # Define output metadata
                meta_df = {
                    'latitude': np.float64,
                    'longitude': np.float64,
                    'mean_revenue': np.float64,
                    'lat_bin': np.int64,
                    'lon_bin': np.int64
                }

                # Apply to all partitions
                ddf_means = ddf_revenues.map_partitions(
                    compute_partition_mean,
                    window_start=window["train_window_start"],
                    window_end=window["train_window_end"],
                    meta=meta_df
                )

                if verbose:
                    print("\nCollecting results from all partitions...")
                
                # Compute and collect all partition means
                mean_df = ddf_means.compute()
                
                if verbose:
                    print(f"\nComputed mean revenue for {len(mean_df)} locations")
                
                # Add location identifier
                mean_df['location'] = mean_df['latitude'].astype(str) + '_' + mean_df['longitude'].astype(str)
                
                # Reorder columns
                mean_df = mean_df[['location', 'latitude', 'longitude', 'mean_revenue', 'lat_bin', 'lon_bin']]
                
                # Sort by location for consistency
                mean_df = mean_df.sort_values('location').reset_index(drop=True)
                
                # Save mean revenue (including bins for reference)
                mean_revenue_path = self.output_revenues_parquet_dir / f"{rolling.format_window_str(window['train_window_start'], window['train_window_end'])}.parquet"
                if mean_revenue_path.exists():
                    mean_revenue_path.unlink()
                    if verbose:
                        print(f"\nRemoved existing parquet file for mean revenue.")
                mean_df.to_parquet(mean_revenue_path, index=False)
                
                if verbose:
                    print(f"\nSaved mean revenue to {mean_revenue_path}")
                    print(f"  - Min mean revenue: {mean_df['mean_revenue'].min():.2f}")
                    print(f"  - Max mean revenue: {mean_df['mean_revenue'].max():.2f}")
                    print(f"  - Avg mean revenue: {mean_df['mean_revenue'].mean():.2f}")

                    print(f"\nSample of mean revenue data:")
                    print(mean_df.head())
                
                # Create and save location mapping
                location_map = mean_df[['location', 'latitude', 'longitude']].copy()
                if len(list(self.location_mapping_parquet_dir.glob("*.parquet"))) > 0:
                    for existing_file in self.location_mapping_parquet_dir.glob("*.parquet"):
                        existing_file.unlink()
                location_map.to_parquet(self.location_mapping_parquet_dir / f"location_mapping_{rolling.format_window_str(window['train_window_start'], window['train_window_end'])}.parquet", index=False)
                
                if verbose:
                    print(f"\nSaved location mapping for {len(location_map)} locations")
                    print(f"  - Number of locations: {len(location_map)}")
                    print(f"\nSample of location mapping:")
                    print(location_map.head())
                
        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    computer = MeanRevenueComputer(target="paths_local")
    computer.compute_mean_revenue(verbose=True)
