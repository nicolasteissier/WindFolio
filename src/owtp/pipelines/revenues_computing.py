import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import os


class RevenuesComputing:
    """
    Compute revenues from energy and prices data using Dask.
    """

    def __init__(self, target: Literal["paths", "paths_local"], adjusted_height: bool = True):
        self.config = owtp.config.load_yaml_config()
        self.input_energy_dir = Path(self.config[target]['processed_data']) / "parquet" / "energy" / "era5_land" / "hourly"
        self.input_energy_100m_dir = Path(self.config[target]['processed_data']) / "parquet" / "energy_100m" / "era5_land" / "hourly"
        self.input_prices_dir = Path(self.config[target]['processed_data']) / "parquet" / "prices" / str("hourly")
        self.output_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / str("hourly")
        self.output_100m_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues_100m" / str("hourly")
        
        self.adjusted_height = adjusted_height

    def compute_revenues(self, n_workers=None, verbose=True):
        """Compute revenues from energy and prices data"""

        if n_workers is None:
            n_workers = os.cpu_count() // 2

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
            if verbose:
                if self.adjusted_height:
                    print(f"Loading energy data at adjusted height from {self.input_energy_100m_dir}...")
                else:
                    print(f"Loading energy data from {self.input_energy_dir}...")
            
            if self.adjusted_height:
                input_dir = self.input_energy_100m_dir
            else:
                input_dir = self.input_energy_dir
            
            ddf_energy = dd.read_parquet(input_dir, engine="pyarrow")
            
            if verbose:
                print("Loading prices data...")
            df_prices = pd.read_parquet(self.input_prices_dir / "prices.parquet").reset_index() # using pandas, price data is small

            if 'valid_time' in ddf_energy.columns:
                ddf_energy = ddf_energy.rename(columns={'valid_time': 'time'})
            
            ddf_prices = dd.from_pandas(df_prices, npartitions=1)
            
            ddf_energy['time'] = dd.to_datetime(ddf_energy['time'])
            ddf_prices['time'] = dd.to_datetime(ddf_prices['time'])
            
            if verbose:
                print("Energy and prices data loaded.\n")
                print("Merging energy and prices data...\n")

            print(f"Columns in energy data: {ddf_energy.columns.tolist()}")
            print(f"Columns in prices data: {ddf_prices.columns.tolist()}")  

            ddf_merged = dd.merge(
                ddf_energy,
                ddf_prices[['time', 'Price (EUR/MWhe)']],
                on='time',
                how='inner'
            )

            if verbose:
            
                print("Computing revenues...")
            
            ddf_merged['revenue'] = ddf_merged['mwh'] * ddf_merged['Price (EUR/MWhe)']
            
            # add lat_bin and lon_bin partitioning
            if 'lat_bin' in ddf_energy.columns and 'lon_bin' in ddf_energy.columns:
                ddf_revenues = ddf_merged[['time', 'latitude', 'longitude', 'lat_bin', 'lon_bin', 'revenue']]
                partition_cols = ['lat_bin', 'lon_bin']
            else:
                print("/!\ Warning: 'lat_bin' and 'lon_bin' columns not found in energy data. Writing without partitioning.\n")
                raise ValueError("Partitioning columns 'lat_bin' and 'lon_bin' not found in energy data.")
            
            if verbose:
                print("Revenues computed.\n")

                print("Writing revenues to Parquet...")
            
            # Write to Parquet (partitioned if possible)
            if self.adjusted_height:
                output_dir = self.output_100m_dir
            else:
                output_dir = self.output_dir
            
            output_dir.mkdir(parents=True, exist_ok=True)
            ddf_revenues.to_parquet(
                output_dir,
                engine="pyarrow",
                compression="zstd",
                write_index=False,
                partition_on=partition_cols,
            )
            
            if verbose:
                print("Done writing revenues Parquet files.")
                
        finally:
            client.close()
            cluster.close()

    
if __name__ == "__main__":
    computer = RevenuesComputing(target="paths")
    computer.compute_revenues(verbose=True)
