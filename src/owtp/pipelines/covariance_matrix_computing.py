import owtp.config
from pathlib import Path
from typing import Literal
from tqdm.contrib.concurrent import process_map
import pandas as pd
import itertools
import os

from owtp.others import rolling

class CovarianceMatrixComputer:
    """
    Compute covariance matrix from revenues data across all locations.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_revenues_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "hourly"
        self.output_dir = Path(self.config[target]['processed_data']) / "csv" / "covariance_matrix_long"

    def compute_covariance_matrix(self, verbose=True):
        """Compute covariance matrix of revenues across all locations."""
        
        for output_csv in self.output_dir.glob("*.csv"):
            output_csv.unlink()

        n_workers = self.config['clustering']['n_workers']
        
        bins = set(self.input_revenues_dir.glob("lat_bin=*/lon_bin=*"))

        if verbose:
            print("Processing partitions one by one")

        sample = pd.read_parquet(next(iter(bins)))
        date_min, date_max = sample['time'].min(), sample['time'].max()
        windows = rolling.get_windows(date_min, date_max)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for window in windows:
            start = window["train_window_start"]
            end = window["train_window_end"]
            output_csv = self.output_dir / f"{rolling.format_window_str(start, end)}.csv"
            pd.DataFrame(columns=["col1", "col2", "covariance"]).to_csv(output_csv, index=False)

        process_map(
            self._handle_single_partition,
            bins,
            [windows]*len(bins),
            max_workers=n_workers,
            chunksize=1,
            desc="Processing partitions",
        )

        if verbose:
            print("Processing partition pairs")

        bin_list = list(bins)
        
        partition_pairs = list(itertools.combinations(bin_list, 2))

        process_map(
            self._handle_partition_pairs,
            *zip(*partition_pairs),
            [windows]*len(partition_pairs),
            max_workers=n_workers,
            chunksize=1,
            desc="Processing partition pairs",
        )

    def _handle_single_partition(self, file_path: Path, windows: list[dict]) -> None:
        """Load a single partition of revenues data."""
        path = Path(file_path)

        df = pd.read_parquet(path)

        pivoted = self._pivot_and_clean(df)

        if pivoted.shape[1] == 0:
            print(f"No valid columns in partition {file_path}, skipping.")
            return
            
        for window in windows:
            start = window["train_window_start"]
            end = window["train_window_end"]

            windowed = pivoted.loc[(pivoted.index >= start) & (pivoted.index <= end)]
            cov = windowed.cov()
            cov_melted = cov.reset_index(names="col1").melt(id_vars="col1", var_name="col2", value_name="covariance")
            
            cov_melted = cov_melted[cov_melted["col1"] <= cov_melted["col2"]]
            
            output_file = self.output_dir / f"{rolling.format_window_str(start, end)}.csv"
            cov_melted.to_csv(output_file, mode="a", header=False, index=False)

    def _handle_partition_pairs(self, file_path1: Path, file_path2: Path, windows: list[dict]) -> None:
        """Load two partitions of revenues data and compute cross-covariance."""
        path1 = Path(file_path1)
        path2 = Path(file_path2)

        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)

        pivoted1 = self._pivot_and_clean(df1)
        pivoted2 = self._pivot_and_clean(df2)

        if pivoted1.shape[1] == 0 or pivoted2.shape[1] == 0:
            print(f"No valid columns in partition pair {file_path1}, {file_path2}, skipping.")
            return
        
        if pivoted1.shape[0] != pivoted2.shape[0]:
            raise ValueError(f"Partitions {file_path1} and {file_path2} have different number of time points.")

        for window in windows:
            start = window["train_window_start"]
            end = window["train_window_end"]

            windowed1 = pivoted1.loc[(pivoted1.index >= start) & (pivoted1.index <= end)]
            windowed2 = pivoted2.loc[(pivoted2.index >= start) & (pivoted2.index <= end)]

            combined = pd.concat([windowed1, windowed2], axis=1)

            cov_matrix = combined.cov().loc[windowed1.columns, windowed2.columns]

            cov_melted = cov_matrix.reset_index(names="col1").melt(id_vars="col1", var_name="col2", value_name="covariance")
            
            output_file = self.output_dir / f"{rolling.format_window_str(start, end)}.csv"
            cov_melted.to_csv(output_file, mode="a", header=False, index=False)

    def _pivot_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot the revenues DataFrame to have time as index and (latitude, longitude) as columns and clean invalid columns."""
        pivoted = df.pivot(index="time", columns=["latitude", "longitude"], values="revenue").sort_index()
        
        pivoted.columns = [f"{lat}_{lon}" for lat, lon in pivoted.columns]

        return pivoted

if __name__ == "__main__":
    computer = CovarianceMatrixComputer(target="paths_local")
    computer.compute_covariance_matrix(verbose=True)