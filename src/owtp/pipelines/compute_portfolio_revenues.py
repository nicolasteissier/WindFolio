import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import os

from owtp.others import rolling


class PortfolioRevenueComputer:
    """
    Compute portfolio revenues using saved portfolio weights from mean-variance optimization.
    Evaluates portfolio performance on the eval window for each rolling window.
    """

    def __init__(self, target: Literal["paths", "paths_local"] = "paths_local"):
        self.config = owtp.config.load_yaml_config()
        
        self.input_revenues_dir = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "hourly"
        self.input_weights_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_weights"
        
        self.output_hourly_dir = Path(self.config[target]['processed_data']) / "parquet" / "portfolio_revenues"
        self.output_summary_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_revenues_summary"
        
        self.total_turbines = self.config['mean_variance_optimization']['total_turbines']
        self.lambda_values = self.config['mean_variance_optimization']['lambda_values']
        self.min_revenue_threshold = self.config['mean_variance_optimization']['min_revenue_threshold']

    def load_portfolio_weights(self, window_suffix: str, param_suffix: str, verbose: bool = True) -> pd.DataFrame:
        """
        Load portfolio weights for a specific window and parameter configuration.
        
        Args:
            window_suffix: Window identifier (e.g., "20050424-220000_20100423-215959")
            param_suffix: Parameter suffix (e.g., "_(100)_(0.1)_(10.0)")
            verbose: Print loading information
            
        Returns:
            DataFrame with columns: location, latitude, longitude, mean_revenue, weight_continuous, weight_integer
        """
        weights_path = self.input_weights_dir / param_suffix / window_suffix / f"portfolio_weights_active{param_suffix}.csv"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Portfolio weights not found at {weights_path}")
        
        weights_df = pd.read_csv(weights_path)
        
        if verbose:
            print(f"  Loaded {len(weights_df)} locations with active weights")
            print(f"    Total turbines: {weights_df['weight_integer'].sum()}")
            print(f"    Top location: {weights_df.iloc[0]['location']} with {weights_df.iloc[0]['weight_integer']} turbines")
        
        return weights_df

    def compute_portfolio_revenues(
        self, 
        window: dict,
        param_suffix: str,
        n_workers: int = None,
        verbose: bool = True
    ):
        """
        Compute portfolio revenues for a specific window and parameter configuration.
        
        Args:
            window: Dictionary with train_window_start, train_window_end, eval_window_start, eval_window_end
            param_suffix: Parameter suffix (e.g., "_(100)_(0.1)_(10.0)")
            n_workers: Number of Dask workers
            verbose: Print computation information
        """
        
        train_window_str = rolling.format_window_str(window['train_window_start'], window['train_window_end'])
        eval_window_str = rolling.format_window_str(window['eval_window_start'], window['eval_window_end'])
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Computing portfolio revenues for:")
            print(f"  Training window: {train_window_str}")
            print(f"  Evaluation window: {eval_window_str}")
            print(f"  Parameters: {param_suffix}")
            print(f"{'='*80}\n")
        
        try:
            weights_df = self.load_portfolio_weights(train_window_str, param_suffix, verbose=verbose)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            return
        
        location_weights = dict(zip(weights_df['location'], weights_df['weight_integer']))
        
        if verbose:
            print(f"\n  Loading hourly revenues for eval window...")
        
        if n_workers is None:
            n_workers = os.cpu_count() // 2
        
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=4,
            memory_limit='20GB',
            processes=True,
            dashboard_address=':8789'
        )
        client = Client(cluster)
        
        try:
            ddf_revenues = dd.read_parquet(
                self.input_revenues_dir,
                engine="pyarrow",
                split_row_groups="infer",
                calculate_divisions=False
            )
            
            eval_start = window['eval_window_start']
            eval_end = window['eval_window_end']
            
            ddf_revenues['time'] = dd.to_datetime(ddf_revenues['time'])
            ddf_eval = ddf_revenues[
                (ddf_revenues['time'] >= eval_start) & 
                (ddf_revenues['time'] <= eval_end)
            ]
            
            ddf_eval['location'] = ddf_eval['latitude'].astype(str) + '_' + ddf_eval['longitude'].astype(str)
            
            portfolio_locations = set(location_weights.keys())
            ddf_portfolio = ddf_eval[ddf_eval['location'].isin(portfolio_locations)]
            
            if verbose:
                print(f"  Computing portfolio revenues...")
            
            def apply_weights(partition_df, weights_dict):
                """Apply portfolio weights to revenues in a partition"""
                partition_df = partition_df.copy()
                partition_df['weight'] = partition_df['location'].map(weights_dict)
                partition_df['portfolio_revenue'] = partition_df['revenue'] * partition_df['weight']
                return partition_df[['time', 'location', 'revenue', 'weight', 'portfolio_revenue']]
            
            meta_df = {
                'time': 'datetime64[ns]',
                'location': str,
                'revenue': np.float64,
                'weight': np.int64,
                'portfolio_revenue': np.float64
            }
            
            ddf_weighted = ddf_portfolio.map_partitions(
                apply_weights,
                weights_dict=location_weights,
                meta=meta_df
            )
            
            weighted_df = ddf_weighted.compute()
            
            if verbose:
                print(f"  Computed revenues for {len(weighted_df)} location-hour combinations")
                print(f"    Time range: {weighted_df['time'].min()} to {weighted_df['time'].max()}")
            
            hourly_portfolio = weighted_df.groupby('time').agg({
                'portfolio_revenue': 'sum',
                'revenue': 'sum',  # Total revenue if we had 1 turbine at each location
                'weight': 'sum'     
            }).reset_index()
            
            hourly_portfolio = hourly_portfolio.sort_values('time').reset_index(drop=True)
            
            # cumulative revenue computation
            hourly_portfolio['cumulative_revenue'] = hourly_portfolio['portfolio_revenue'].cumsum()
            
            summary = {
                'train_window_start': window['train_window_start'],
                'train_window_end': window['train_window_end'],
                'eval_window_start': window['eval_window_start'],
                'eval_window_end': window['eval_window_end'],
                'total_turbines': int(weights_df['weight_integer'].sum()),
                'num_locations': len(weights_df),
                'num_hours': len(hourly_portfolio),
                'total_revenue': float(hourly_portfolio['portfolio_revenue'].sum()),
                'mean_hourly_revenue': float(hourly_portfolio['portfolio_revenue'].mean()),
                'std_hourly_revenue': float(hourly_portfolio['portfolio_revenue'].std()),
                'min_hourly_revenue': float(hourly_portfolio['portfolio_revenue'].min()),
                'max_hourly_revenue': float(hourly_portfolio['portfolio_revenue'].max()),
                'median_hourly_revenue': float(hourly_portfolio['portfolio_revenue'].median()),
                'final_cumulative_revenue': float(hourly_portfolio['cumulative_revenue'].iloc[-1]),
            }
            
            if verbose:
                print(f"\n  Portfolio Performance Summary:")
                print(f"    Total revenue (eval period): {summary['total_revenue']:,.2f} EUR")
                print(f"    Mean hourly revenue: {summary['mean_hourly_revenue']:.2f} EUR/hour")
                print(f"    Std hourly revenue: {summary['std_hourly_revenue']:.2f} EUR/hour")
                print(f"    Revenue range: [{summary['min_hourly_revenue']:.2f}, {summary['max_hourly_revenue']:.2f}]")
            
            self._save_results(
                hourly_portfolio,
                summary,
                train_window_str,
                eval_window_str,
                param_suffix,
                verbose=verbose
            )
            
        finally:
            client.close()
            cluster.close()

    def _save_results(
        self,
        hourly_df: pd.DataFrame,
        summary: dict,
        train_window_str: str,
        eval_window_str: str,
        param_suffix: str,
        verbose: bool = True
    ):
        """Save hourly revenues and summary statistics"""
        
        hourly_dir = self.output_hourly_dir / param_suffix / train_window_str
        hourly_dir.mkdir(parents=True, exist_ok=True)
        
        summary_dir = self.output_summary_dir / param_suffix
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        hourly_path = hourly_dir / f"portfolio_revenues_hourly{param_suffix}.parquet"
        hourly_df.to_parquet(hourly_path, index=False)
        
        summary_df = pd.DataFrame([summary])
        summary_path = summary_dir / f"portfolio_revenues_summary{param_suffix}.csv"
        
        if summary_path.exists():
            existing_summary = pd.read_csv(summary_path)
            mask = (
                (existing_summary['train_window_start'] == str(summary['train_window_start'])) &
                (existing_summary['train_window_end'] == str(summary['train_window_end']))
            )
            if mask.any():
                existing_summary.loc[mask] = summary_df.iloc[0]
                summary_df = existing_summary
            else:
                summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
        
        summary_df.to_csv(summary_path, index=False)
        
        if verbose:
            print(f"\n  Saved results:")
            print(f"    Hourly: {hourly_path}")
            print(f"    Summary: {summary_path}")

    def compute_all_windows(self, n_workers: int = None, verbose: bool = True, include_random: bool = True):
        """
        Compute portfolio revenues for all windows and parameter combinations.
        
        Args:
            n_workers: Number of Dask workers
            verbose: Print information
            include_random: Whether to include random portfolio baseline
        """
        
        start = pd.Timestamp(self.config['rolling_calibrations']['start_date'])
        end = pd.Timestamp(self.config['rolling_calibrations']['end_date'])
        
        windows = rolling.get_windows(start, end)
        
        param_configs = []
        for lambda_risk in self.lambda_values:
            param_configs.append(('lambda', lambda_risk))
        if include_random:
            param_configs.append(('random', 'random'))
        
        if verbose:
            print(f"Processing {len(windows)} windows for {len(param_configs)} configurations = {len(windows) * len(param_configs)} combinations\n")
        
        for i, window in enumerate(windows, 1):
            for config_type, config_value in param_configs:
                param_suffix = f"_({self.total_turbines})_({config_value})_({self.min_revenue_threshold})"
                
                if verbose:
                    if config_type == 'lambda':
                        print(f"\n[{i}/{len(windows)}] Window {i}, _lambda={config_value}")
                    else:
                        print(f"\n[{i}/{len(windows)}] Window {i}, Random baseline")

                try:
                    self.compute_portfolio_revenues(
                        window=window,
                        param_suffix=param_suffix,
                        n_workers=n_workers,
                        verbose=verbose
                    )
                except Exception as e:
                    print(f"Error processing window {i} with config={config_value}: {e}")
                    continue


if __name__ == "__main__":
    computer = PortfolioRevenueComputer(target="paths_local")
    computer.compute_all_windows(verbose=True, include_random=True)
