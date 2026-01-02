import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
from tqdm import tqdm

from owtp.others import rolling

class CovarianceMatrixPivot:
    """
    Pivot long-format covariance matrix CSV to wide matrix format (necessary for mean-var optimization).
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config[target]['processed_data']) / "csv" / "covariance_matrix_long"

        self.output_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "covariance_matrix"
        self.output_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "covariance_matrix"
        self.output_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)

        self.location_mapping_dir = Path(self.config[target]['processed_data']) / "parquet" / "locations"

    def pivot_covariance_matrices(self, chunksize=100000, verbose=True):
        """
        Pivot all long-format covariance CSVs in the input directory to wide matrix format.
        
        Args:
            chunksize: Number of rows to read at a time from CSV
            verbose: Print progress information
        """
        csv_files = [f for f in self.input_dir.glob("*.csv") if not f.name.startswith("._")]
        
        if verbose:
            print(f"Found {len(csv_files)} covariance CSV files to pivot.")
        
        for csv_file in tqdm(csv_files, desc="Pivoting covariance matrices"):
            if verbose:
                print(f"\nPivoting {csv_file.name}...")
            self.pivot_covariance_matrix(
                input_csv=csv_file,
                chunksize=chunksize,
                verbose=verbose
            )

    def pivot_covariance_matrix(self, input_csv: Path, chunksize=100000, verbose=True) -> pd.DataFrame:
        """
        Pivot long-format covariance CSV to wide matrix format.
        
        Args:
            chunksize: Number of rows to read at a time from CSV
            verbose: Print progress information
        """
        
        if not input_csv.exists():
            raise FileNotFoundError(
                f"Covariance matrix CSV not found at {input_csv}. "
                f"Please run CovarianceMatrixComputer first."
            )
        
        if verbose:
            print(f"Loading covariance matrix from {input_csv}")
            print("This may take a while for large matrices...")
        
        if verbose:
            print("\nIdentifying unique locations...")
        
        locations = set()
        for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize), desc="Reading chunks", disable=not verbose):
            locations.update(chunk['col1'].unique())
            locations.update(chunk['col2'].unique())
        
        locations = sorted(locations)
        n_locations = len(locations)
        
        if verbose:
            print(f"Found {n_locations} unique locations")
            print(f"Matrix will be {n_locations} x {n_locations}")
        
        loc_to_idx = {loc: idx for idx, loc in enumerate(locations)}
        
        # symmetric, so we fill both triangles
        cov_matrix = np.zeros((n_locations, n_locations))
        
        if verbose:
            print("\nBuilding covariance matrix...")
        
        for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize),
                         desc="Processing chunks", disable=not verbose):
            for _, row in chunk.iterrows():
                i = loc_to_idx[row['col1']]
                j = loc_to_idx[row['col2']]
                cov_value = row['covariance']
                
                cov_matrix[i, j] = cov_value # fill both (i,j) and (j,i) (symmetric matrix)
                if i != j:
                    cov_matrix[j, i] = cov_value
        
        if verbose:
            print("\nCreating DataFrame with location labels...")
        
        cov_df = pd.DataFrame(cov_matrix, index=locations, columns=locations)
        
        if verbose: # check symmetry
            max_asymmetry = np.max(np.abs(cov_matrix - cov_matrix.T))
            print(f"Matrix symmetry check - max asymmetry: {max_asymmetry:.2e}")
            if max_asymmetry > 1e-10:
                raise ValueError("Covariance matrix is not symmetric")
        
        if verbose:
            print("\nSaving wide-format covariance matrix...")
        output_parquet = self.output_parquet_dir / input_csv.with_suffix(".parquet").name
        cov_df.to_parquet(output_parquet)
        
        if verbose:
            print(f"\nSaved wide-format covariance matrix to {output_parquet}")
            print(f"Matrix shape: {cov_df.shape}")
        
        output_csv = self.output_csv_dir / input_csv.name
        cov_df.to_csv(output_csv)
        if verbose:
            print(f"Saved CSV version to {output_csv}")
        
        # verify consistency with location mapping
        location_mapping_file = self.location_mapping_dir / input_csv.with_suffix(".parquet").name
        if location_mapping_file.exists():
            if verbose:
                print("\nVerifying consistency with location mapping...")
            
            location_map = pd.read_parquet(location_mapping_file)
            expected_locations = sorted(location_map['location'].tolist())
            
            if locations == expected_locations:
                if verbose:
                    print("✓ Locations match location_mapping.parquet perfectly")
            else:
                missing_in_cov = set(expected_locations) - set(locations)
                missing_in_map = set(locations) - set(expected_locations)
                
                if missing_in_cov:
                    print(f"WARNING: {len(missing_in_cov)} locations in mapping but not in covariance matrix")
                if missing_in_map:
                    print(f"WARNING: {len(missing_in_map)} locations in covariance matrix but not in mapping")
        
        return cov_df
    
    def check_covariance_matrix(self):
        """
        Simple exploration of the covariance matrix.
        """
        for path in [f for f in self.output_parquet_dir.glob("*.parquet") if not f.name.startswith("._")]:
            cov_df = pd.read_parquet(path)
            
            print("\nCovariance Matrix Exploration:")

            print(f"\nShape: {cov_df.shape}")

            print(f"\nMean covariance: {cov_df.values.mean():.4f}")
            print(f"Max covariance: {cov_df.values.max():.4f}")
            print(f"Min covariance: {cov_df.values.min():.4f}")

            print("\nFirst few rows:")
            print(cov_df.head())
            
            eigvals = np.linalg.eigvalsh(cov_df.values)
            n_negative = np.sum(eigvals < 0)
            if n_negative == 0:
                print("\n✓ Covariance matrix is positive definite")
            else:
                print(f"WARNING: Covariance matrix has {n_negative} negative eigenvalues")


if __name__ == "__main__":
    pivoter = CovarianceMatrixPivot(target="paths_local")
    pivoter.pivot_covariance_matrices(verbose=True)
    pivoter.check_covariance_matrix()