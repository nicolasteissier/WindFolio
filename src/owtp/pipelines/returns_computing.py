from pandas import Timestamp
from tqdm import tqdm
import owtp.config
from pathlib import Path
from typing import Literal
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import os


class ReturnsComputing:
    """Compute returns from pre-computed revenue time series for each location."""

    global TIMESTAMPS_TO_IGNORE # cf src/owtp/others/extract_missing_dates.py for the list construction and report for explanation 
    TIMESTAMPS_TO_IGNORE = [
        Timestamp('2005-10-30 23:00:00'),
        Timestamp('2006-10-29 23:00:00'), 
        Timestamp('2007-10-28 23:00:00'), 
        Timestamp('2008-10-26 23:00:00'), 
        Timestamp('2009-10-25 23:00:00')
    ]

    def __init__(self, target: Literal['paths', 'paths_local']):
        self.config = owtp.config.load_yaml_config()
        self.input_revenues_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "hourly"
        self.output_dir = Path(self.config[target]['processed_data']) / "parquet" / "returns" / "hourly"

    def compute_returns(self, return_type: Literal['simple', 'log'] = 'simple', n_workers=None, verbose=True):
        """
        Compute returns from pre-computed revenue data.
        
        Parameters:
        - return_type : str
            Type of returns to compute:
            - 'simple': (revenue_t - revenue_t-1) / revenue_t-1
            - 'log': log(revenue_t / revenue_t-1)
        - n_workers : int, optional
            Number of Dask workers to use
        """
        
        if n_workers is None:
            n_workers = os.cpu_count() // 2 # type: ignore

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
                print("Loading revenue data...")

            meta = [f for f in self.input_revenues_dir.rglob("*.parquet") if f.name.startswith("._")]
            if verbose:
                print(f"Found {len(meta)} system files to remove.")
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
                print(f"Found {ddf_revenues.npartitions} partitions in revenue data.")
            
            ddf_revenues['time'] = dd.to_datetime(ddf_revenues['time'])


            # Convert categorical partition columns back to int64
            # ddf_revenues['lat_bin'] = ddf_revenues['lat_bin'].cat.codes.astype(np.int64)
            # ddf_revenues['lon_bin'] = ddf_revenues['lon_bin'].cat.codes.astype(np.int64)

            ddf_revenues['lat_bin'] = ddf_revenues['lat_bin'].map_partitions(
                lambda s: s.cat.categories[s.cat.codes].astype(float).astype(np.int64),
                meta=('lat_bin', np.int64)
            )
            ddf_revenues['lon_bin'] = ddf_revenues['lon_bin'].map_partitions(
                lambda s: s.cat.categories[s.cat.codes].astype(float).astype(np.int64),
                meta=('lon_bin', np.int64)
            )
            
            if verbose:
                print(f"Computing {return_type} returns for each location...")
            
            # Check if partitioning columns exist
            has_partitioning = 'lat_bin' in ddf_revenues.columns and 'lon_bin' in ddf_revenues.columns

            if verbose:
                if has_partitioning:
                    print("Partitioning columns 'lat_bin' and 'lon_bin' found. Using optimized groupby.")
                else:
                    print("/!\\ Warning: Partitioning columns 'lat_bin' and 'lon_bin' not found.")
                    raise ValueError("Partitioning columns 'lat_bin' and 'lon_bin' not found in revenue data.")
            
            # Process each partition independently (no shuffle since data is already partitioned)
            def compute_partition_returns(partition_df, return_type='simple'):
                """
                Compute returns for all locations within a partition.
                This function operates on a single partition (pandas DataFrame).
                """
                # Group by exact location within this partition
                grouped = partition_df.groupby(['latitude', 'longitude'], group_keys=False)
                
                def compute_location_returns(location_df):
                    # Sort by time for proper returns calculation
                    location_df = location_df.sort_values('time')
                    
                    if return_type == 'simple':
                        location_df['return'] = location_df['revenue'].pct_change()
                    elif return_type == 'log':
                        location_df['return'] = np.log(location_df['revenue'] / location_df['revenue'].shift(1))
                    else:
                        raise ValueError(f"Unknown return_type: {return_type}")
                    
                    # Filter out timestamps right after price gaps (invalid returns)
                    location_df = location_df[~location_df['time'].isin(TIMESTAMPS_TO_IGNORE)]
                    
                    # Now replace remaining NaN (from first observation) with 0.0
                    location_df.loc[location_df['return'].isna(), 'return'] = 0.0 # TODO placeholder, to remove
                    
                    return location_df.drop(columns=['revenue'])
                
                # Apply to all locations in this partition
                return grouped[['time', 'latitude', 'longitude', 'lat_bin', 'lon_bin', 'revenue']].apply(compute_location_returns)
            
            # Define output metadata (input columns + new 'return' column)
            meta_df = {
                'time': 'datetime64[ns]',
                'latitude': np.float64,
                'longitude': np.float64,
                'lat_bin': np.int64,
                'lon_bin': np.int64,
                'return': np.float64
            }
            
            # Use map_partitions to avoid shuffle - each partition is processed independently
            ddf_final = ddf_revenues.map_partitions(
                compute_partition_returns,
                return_type=return_type,
                meta=meta_df
            )

            partition_cols = ['lat_bin', 'lon_bin']
            
            # Drop NaN returns (first observation for each location)
            ddf_final = ddf_final.dropna(subset=['return'])
            
            if verbose:
                print("Writing returns to Parquet...")
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ddf_final.to_parquet(
                self.output_dir,
                engine="pyarrow",
                compression="zstd",
                write_index=False,
                partition_on=partition_cols,
            )

            if verbose:
                print(f"Done writing {return_type} returns Parquet files.")
                
        finally:
            client.close()
            cluster.close()

    
if __name__ == "__main__":
    computer = ReturnsComputing(target="paths")
    computer.compute_returns(return_type='simple', verbose=True)
