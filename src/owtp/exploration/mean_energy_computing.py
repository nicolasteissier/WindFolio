import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import os
from tqdm import tqdm


class MeanEnergyComputer:
    """
    Compute mean energy per location from partitioned energy data.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_energy_dir = Path(self.config[target]['processed_data']) / "parquet" / "energy" / "era5_land" / "hourly"

        self.output_energy_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "energy" / "mean"
        self.output_energy_parquet_dir.mkdir(parents=True, exist_ok=True)

        self.output_energy_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "energy" / "mean"
        self.output_energy_csv_dir.mkdir(parents=True, exist_ok=True)


    def compute_mean_energy(self, verbose=True):
        """
        Compute mean energy per location and save results.
        
        Outputs:
            - mean_energy.parquet: Mean energy per location
            - location_mapping.parquet: Location identifiers and coordinates
        """
        n_workers = self.config['clustering']['n_workers']
        threads_per_worker = self.config['clustering']['threads_per_worker']
        memory_limit = self.config['clustering']['memory_limit']

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,
            dashboard_address=':8788'
        )
        client = Client(cluster)

        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")

        try:
            if verbose:
                print(f"\nLoading energy data from {self.input_energy_dir}...")

            meta = [f for f in self.input_energy_dir.rglob("*.parquet") if f.name.startswith("._")]
            if verbose and len(meta) > 0:
                print(f"\nFound {len(meta)} system files to remove.")
            iterator = tqdm(meta, desc="Removing system files") if verbose else meta
            for meta_file in iterator:
                meta_file.unlink()

            ddf_energy = dd.read_parquet(
                self.input_energy_dir,
                engine="pyarrow",
                split_row_groups="infer",
                aggregate_files="lon_bin",  
                calculate_divisions=False  
            )

            if verbose:
                print(f"\nFound {ddf_energy.npartitions} partitions in energy data.")
            
            ddf_energy['lat_bin'] = ddf_energy['lat_bin'].map_partitions(
                lambda s: s.cat.categories[s.cat.codes].astype(float).astype(np.int64),
                meta=('lat_bin', np.int64)
            )
            ddf_energy['lon_bin'] = ddf_energy['lon_bin'].map_partitions(
                lambda s: s.cat.categories[s.cat.codes].astype(float).astype(np.int64),
                meta=('lon_bin', np.int64)
            )
            
            if verbose:
                print("\nComputing mean energy per location...")
            
            has_partitioning = 'lat_bin' in ddf_energy.columns and 'lon_bin' in ddf_energy.columns

            if verbose and not has_partitioning:
                raise ValueError("Partitioning columns 'lat_bin' and 'lon_bin' not found in energy data.")
            
            def compute_partition_mean(partition_df):
                """
                Compute mean energy for all locations within a partition.
                This function operates on a single partition (pandas DataFrame).
                """
                mean_by_location = partition_df.groupby(['latitude', 'longitude']).agg(
                    mean_energy=('mwh', 'mean'),
                    lat_bin=('lat_bin', 'first'),
                    lon_bin=('lon_bin', 'first')
                ).reset_index()
                
                return mean_by_location
            
            meta_df = {
                'latitude': np.float64,
                'longitude': np.float64,
                'mean_energy': np.float64,
                'lat_bin': np.int64,
                'lon_bin': np.int64
            }

            ddf_means = ddf_energy.map_partitions(
                compute_partition_mean,
                meta=meta_df
            )

            if verbose:
                print("\nCollecting results from all partitions...")
            
            mean_df = ddf_means.compute()
            
            if verbose:
                print(f"\nComputed mean energy for {len(mean_df)} locations")
            
            mean_df['location'] = mean_df['latitude'].astype(str) + '_' + mean_df['longitude'].astype(str)
            
            mean_df = mean_df[['location', 'latitude', 'longitude', 'mean_energy', 'lat_bin', 'lon_bin']]
            
            mean_df = mean_df.sort_values('location').reset_index(drop=True)
            
            mean_energy_path = self.output_energy_parquet_dir / "mean_energy.parquet"
            mean_df.to_parquet(mean_energy_path, index=False)
            
            if verbose:
                print(f"\nSaved mean energy to {mean_energy_path}")
                print(f"  - Min mean energy: {mean_df['mean_energy'].min():.4f} MWh")
                print(f"  - Max mean energy: {mean_df['mean_energy'].max():.4f} MWh")
                print(f"  - Avg mean energy: {mean_df['mean_energy'].mean():.4f} MWh")
            
            mean_df.to_csv(self.output_energy_csv_dir / "mean_energy.csv", index=False)
            
            if verbose:
                print("\nMean energy also saved as CSV.")
                
        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    computer = MeanEnergyComputer(target="paths_local")
    computer.compute_mean_energy(verbose=True)
