import owtp.config
from pathlib import Path
from typing import Literal
from tqdm.contrib.concurrent import process_map
import pandas as pd
import itertools
import os


class CovarianceMatrixComputer:
    """
    Compute covariance matrix from revenues data across all locations.
    
    This computes the covariance matrix of revenues where each location (lat, lon)
    is treated as a separate variable/asset.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_revenues_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "hourly"
        self.output_csv = Path(self.config[target]['processed_data']) / "csv" / "covariance_matrix.csv"

    def compute_covariance_matrix(self, n_workers=None, verbose=True):
        """Compute covariance matrix of revenues across all locations."""
        
        if self.output_csv.exists():
            self.output_csv.unlink()

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["col1", "col2", "covariance"]).to_csv(self.output_csv, index=False)

        if n_workers is None:
            n_workers = os.cpu_count() // 2 # type: ignore
        
        bins = set(self.input_revenues_dir.glob("lat_bin=*/lon_bin=*"))

        if verbose:
            print("Processing partitions one by one")

        process_map(
            self._handle_single_partition,
            bins,
            max_workers=n_workers,
            chunksize=1,
            desc="Processing partitions",
        )

        if verbose:
            print("Processing partition pairs")

        bin_list = list(bins)
        
        partition_pairs = itertools.combinations(bin_list, 2)

        process_map(
            self._handle_partition_pairs,
            *zip(*partition_pairs),
            max_workers=n_workers,
            chunksize=1,
            desc="Processing partition pairs",
        )

    def _handle_single_partition(self, file_path: Path) -> None:
        """Load a single partition of revenues data."""
        path = Path(file_path)

        df = pd.read_parquet(path)

        pivoted = self._pivot_and_clean(df)

        # Skip if no valid columns remain
        if pivoted.shape[1] == 0:
            print(f"No valid columns in partition {file_path}, skipping.")
            return
        
        # Compute covariance matrix, melt to long format (col1, col2, covariance), and append to CSV
        cov = pivoted.cov()
        cov_melted = cov.reset_index(names="col1").melt(id_vars="col1", var_name="col2", value_name="covariance")
        
        # Keep only upper triangle (including diagonal) to avoid duplicates in symmetric matrix
        cov_melted = cov_melted[cov_melted["col1"] <= cov_melted["col2"]]
        
        cov_melted.to_csv(self.output_csv, mode="a", header=False, index=False)

    def _handle_partition_pairs(self, file_path1: Path, file_path2: Path) -> None:
        """Load two partitions of revenues data and compute cross-covariance."""
        path1 = Path(file_path1)
        path2 = Path(file_path2)

        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)

        pivoted1 = self._pivot_and_clean(df1)
        pivoted2 = self._pivot_and_clean(df2)

        # Skip if no valid columns remain in either partition
        if pivoted1.shape[1] == 0 or pivoted2.shape[1] == 0:
            print(f"No valid columns in partition pair {file_path1}, {file_path2}, skipping.")
            return
        
        if pivoted1.shape[0] != pivoted2.shape[0]:
            raise ValueError(f"Partitions {file_path1} and {file_path2} have different number of time points.")

        combined = pd.concat([pivoted1, pivoted2], axis=1)

        # Compute cross-covariance matrix
        cov_matrix = combined.cov().loc[pivoted1.columns, pivoted2.columns]

        # Melt to long format and append to CSV
        cov_melted = cov_matrix.reset_index(names="col1").melt(id_vars="col1", var_name="col2", value_name="covariance")
        cov_melted.to_csv(self.output_csv, mode="a", header=False, index=False)

    def _pivot_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot the revenues DataFrame and clean invalid columns."""
        # Pivot the DataFrame to have time as index and (latitude, longitude) as columns
        pivoted = df.pivot(index="time", columns=["latitude", "longitude"], values="revenue").sort_index()
        
        # Flatten column names
        pivoted.columns = [f"{lat}_{lon}" for lat, lon in pivoted.columns]

        return pivoted

if __name__ == "__main__":
    computer = CovarianceMatrixComputer(target="paths_local")
    computer.compute_covariance_matrix(verbose=True)