import owtp.config
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd

from owtp.others import rolling


class RandomPortfolioAllocator:
    """
    Create random portfolio allocations as a baseline for comparison.
    Randomly allocates turbines across some locations
    """

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp, target: Literal["paths", "paths_local"] = "paths_local", random_seed: int = 42):
        self.config = owtp.config.load_yaml_config()
        self.random_seed = random_seed

        self.file_name = rolling.format_window_str(start, end)
        
        self.mean_revenue_path = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "mean" / f"{self.file_name}.parquet"
        
        self.output_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "portfolio_weights"
        self.output_parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_weights"
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)
        
        self.mean_revenue_df = None

    def load_data(self, verbose=True):
        """Load mean revenues."""
        
        if verbose:
            print(f"Loading mean revenues from {self.mean_revenue_path}...")
        
        self.mean_revenue_df = pd.read_parquet(self.mean_revenue_path)
        
        if verbose:
            print(f"  - Loaded {len(self.mean_revenue_df)} locations")
            print(f"  - Mean revenue range: {self.mean_revenue_df['mean_revenue'].min():.2f} - {self.mean_revenue_df['mean_revenue'].max():.2f}")

    def random_allocate(
        self, 
        total_turbines=10, 
        min_revenue_threshold=None,
        verbose=True
    ):
        """
        Randomly allocate turbines across eligible locations.
        
        Args:
            total_turbines: Total number of turbines to allocate
            min_revenue_threshold: Only consider locations with mean revenue above this threshold
            verbose: Print allocation details
        
        Returns:
            Tuple of (weights_continuous, weights_integer, allocation_results)
        """
        
        if self.mean_revenue_df is None:
            self.load_data(verbose=verbose)
        
        np.random.seed(self.random_seed)
        
        if min_revenue_threshold is not None:
            eligible_mask = self.mean_revenue_df['mean_revenue'] >= min_revenue_threshold
            
            if verbose:
                n_eligible = eligible_mask.sum()
                print(f"\nFiltering locations with mean revenue >= {min_revenue_threshold}")
                print(f"Eligible locations: {n_eligible} / {len(self.mean_revenue_df)}")
            
            eligible_indices = eligible_mask.values
        else:
            eligible_indices = np.ones(len(self.mean_revenue_df), dtype=bool)
        
        n_eligible = eligible_indices.sum()
        
        if n_eligible == 0:
            raise ValueError("No eligible locations after filtering")
        
        random_weights = np.random.uniform(0, 1, size=n_eligible)
        
        random_weights = random_weights / random_weights.sum() * total_turbines
        
        weights_int = np.round(random_weights).astype(int)
        
        current_total = weights_int.sum()
        difference = total_turbines - current_total
        
        if difference != 0:
            adjustment_pool = np.arange(len(weights_int))
            np.random.shuffle(adjustment_pool)
            
            for i in range(abs(difference)):
                idx = adjustment_pool[i % len(adjustment_pool)]
                if difference > 0:
                    weights_int[idx] += 1
                else:
                    if weights_int[idx] > 0:
                        weights_int[idx] -= 1
        
        weights_continuous = np.zeros(len(self.mean_revenue_df))
        weights_continuous[eligible_indices] = random_weights
        
        weights_integer = np.zeros(len(self.mean_revenue_df), dtype=int)
        weights_integer[eligible_indices] = weights_int
        
        results = {
            'expected_return': float(self.mean_revenue_df['mean_revenue'].values @ weights_integer),
            'total_turbines': int(weights_integer.sum()),
            'num_locations': int((weights_integer > 0).sum()),
            'allocation_method': 'random',
            'random_seed': self.random_seed,
        }
        
        if verbose:
            print(f"\nRandom Allocation Results:")
            print(f"  - Expected return: {results['expected_return']:.2f}")
            print(f"  - Total turbines: {results['total_turbines']}")
            print(f"  - Active locations: {results['num_locations']}")
            print(f"  - Random seed: {self.random_seed}")
        
        return weights_continuous, weights_integer, results

    def save_weights(self, weights_continuous, weights_integer, results, window_suffix="", suffix="", verbose=True):
        """Save portfolio weights to disk."""

        output_parquet_dir = self.output_parquet_dir / suffix / window_suffix
        output_parquet_dir.mkdir(parents=True, exist_ok=True)
        output_csv_dir = self.output_csv_dir / suffix / window_suffix
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        
        weights_df = self.mean_revenue_df[['location', 'latitude', 'longitude', 'mean_revenue']].copy()
        weights_df['weight_continuous'] = weights_continuous
        weights_df['weight_integer'] = weights_integer
        
        weights_active_df = weights_df[weights_df['weight_integer'] > 0].copy()
        weights_active_df = weights_active_df.sort_values('weight_integer', ascending=False)
        
        parquet_path = output_parquet_dir / f"portfolio_weights{suffix}.parquet"
        csv_path = output_csv_dir / f"portfolio_weights{suffix}.csv"
        
        weights_df.to_parquet(parquet_path, index=False)
        weights_df.to_csv(csv_path, index=False)
        
        parquet_path_active = output_parquet_dir / f"portfolio_weights_active{suffix}.parquet"
        csv_path_active = output_csv_dir / f"portfolio_weights_active{suffix}.csv"
        
        weights_active_df.to_parquet(parquet_path_active, index=False)
        weights_active_df.to_csv(csv_path_active, index=False)
        
        results_df = pd.DataFrame([results])
        results_path = output_csv_dir / f"allocation_results{suffix}.csv"
        results_df.to_csv(results_path, index=False)
        
        if verbose:
            print(f"\nSaved portfolio weights to:")
            print(f"  - {parquet_path}")
            print(f"  - {csv_path}")
            print(f"  - {parquet_path_active} (active only)")
            print(f"  - {results_path} (summary)")


if __name__ == "__main__":

    config = owtp.config.load_yaml_config()
    start = pd.Timestamp(config['rolling_calibrations']['start_date'])
    end = pd.Timestamp(config['rolling_calibrations']['end_date'])

    windows = rolling.get_windows(start, end)

    total_turbines = config['mean_variance_optimization']['total_turbines']
    min_revenue_threshold = config['mean_variance_optimization']['min_revenue_threshold']
    random_seed = 42

    for window in windows:
        print(f"\nProcessing window: {rolling.format_window_str(window['train_window_start'], window['train_window_end'])}")

        allocator = RandomPortfolioAllocator(
            start=window['train_window_start'], 
            end=window['train_window_end'], 
            target="paths_local",
            random_seed=random_seed
        )
        
        allocator.load_data(verbose=True)

        window_suffix = rolling.format_window_str(window['train_window_start'], window['train_window_end'])
        
        weights_cont, weights_int, results = allocator.random_allocate(
            total_turbines=total_turbines,
            min_revenue_threshold=min_revenue_threshold,
            verbose=True
        )
    
        suffix = f"_({total_turbines})_(random)_({min_revenue_threshold})"
        allocator.save_weights(weights_cont, weights_int, results, window_suffix=window_suffix, suffix=suffix, verbose=True)
        
        print("\n" + "="*60)
        print("Top 10 locations by turbine allocation:")
        print("="*60)
        
        top_locations = weights_int.argsort()[::-1][:10]

        for i, idx in enumerate(top_locations):
            if weights_int[idx] > 0:
                loc = allocator.mean_revenue_df.loc[idx, 'location']
                lat = allocator.mean_revenue_df.loc[idx, 'latitude']
                lon = allocator.mean_revenue_df.loc[idx, 'longitude']
                rev = allocator.mean_revenue_df.loc[idx, 'mean_revenue']
                n_turbines = weights_int[idx]
                print(f"{i+1}. {loc} ({lat:.1f}, {lon:.1f}): {n_turbines} turbines, {rev:.2f} EUR/hour mean revenue")
