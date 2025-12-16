import owtp.config
from pathlib import Path
from typing import Literal
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
import os


class CovarianceMatrixComputer:
    """
    Compute covariance matrix from returns data across all locations.
    
    This computes the covariance matrix of returns where each location (lat, lon)
    is treated as a separate variable/asset.
    """

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_returns_dir = Path(self.config['paths']['processed_data']) / "parquet" / "returns" / str(freq)
        self.output_dir = Path(self.config['paths']['processed_data']) / "parquet" / "covariance" / str(freq)

    def compute_covariance_matrix(self, n_workers=None, verbose=True):
        """Compute covariance matrix of returns across all locations."""
        
        if n_workers is None:
            n_workers = os.cpu_count() // 2

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit='4GB', # 4GB --> run on Nico's mac
            processes=True,
            dashboard_address=':8787'
        )
        client = Client(cluster)

        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")

        try:
            if verbose:
                print("Loading returns data...")
            ddf_returns = dd.read_parquet(self.input_returns_dir, engine="pyarrow")
            
            ddf_returns['time'] = dd.to_datetime(ddf_returns['time'])

            if verbose:
                print("Returns data loaded.\n")
                print("Preparing data for covariance computation:\n")
                print("Creating location identifier...")
            
            # Create a unique location identifier (lat_lon_key)
            ddf_returns['location'] = (
                ddf_returns['latitude'].round(6).astype(str) + '_' + 
                ddf_returns['longitude'].round(6).astype(str)
            )
            
            if verbose:
                print("Location identifier created.\n")
                print("Dataframe before pivot:")
                print(ddf_returns.head().compute())
                print("\nPivoting data to wide format (time x locations)...")
            
            # Pivot: rows=time, columns=location, values=return
            ddf_pivot = ddf_returns.pivot_table(
                index='time',
                columns='location',
                values='return',
                aggfunc='first'  # Should be unique per (time, location)
            )
            
            if verbose:
                print("Data pivoted to wide format.\n")
                print("Dataframe after pivot:")
                print(ddf_pivot.head().compute())
                print("\nComputing covariance matrix...")
            
            cov_matrix_dask = ddf_pivot.cov()
            
            if verbose:
                print("Covariance computation scheduled, now computing...")
            
            cov_matrix = cov_matrix_dask.compute()
            
            if verbose:
                print(f"Data shape: {ddf_pivot.shape[0].compute()} timestamps × {len(ddf_pivot.columns)} locations")
                print(f"Covariance matrix shape: {cov_matrix.shape}\n")
                print("Saving covariance matrix...")
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save covariance matrix as Parquet
            cov_matrix.to_parquet(
                self.output_dir / "covariance_matrix.parquet",
                engine="pyarrow",
                compression="zstd"
            )
            
            # Also save location mapping (location string -> lat, lon)
            location_map = ddf_returns[['location', 'latitude', 'longitude']].drop_duplicates().compute()
            location_map.to_parquet(
                self.output_dir / "location_mapping.parquet",
                engine="pyarrow",
                compression="zstd"
            )
            
            if verbose:
                print("Done writing covariance matrix.")
                print(f"Covariance matrix saved to: {self.output_dir / 'covariance_matrix.parquet'}")
                print(f"Location mapping saved to: {self.output_dir / 'location_mapping.parquet'}")
                
        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    computer = CovarianceMatrixComputer(freq='hourly')
    computer.compute_covariance_matrix(verbose=True)