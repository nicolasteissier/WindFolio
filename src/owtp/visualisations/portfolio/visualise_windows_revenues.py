import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class WindowRevenueVisualiser:
    """
    Visualise portfolio revenues across rolling windows for different lambda values.
    """

    def __init__(self, target: Literal["paths", "paths_local"] = "paths_local"):
        self.config = owtp.config.load_yaml_config()
        
        self.input_summary_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_revenues_summary"
        self.output_dir = Path(self.config[target]['visualisations']) / "portfolio_revenues"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.total_turbines = self.config['mean_variance_optimization']['total_turbines']
        self.lambda_values = self.config['mean_variance_optimization']['lambda_values']
        self.min_revenue_threshold = self.config['mean_variance_optimization']['min_revenue_threshold']
        self.include_random = True  

    def load_all_summaries(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load all portfolio revenue summaries for all lambda values and random baseline.
        
        Returns:
            DataFrame with all summaries combined, including a 'lambda' column
        """
        all_summaries = []
        
        for lambda_risk in self.lambda_values:
            param_suffix = f"_({self.total_turbines})_({lambda_risk})_({self.min_revenue_threshold})"
            summary_path = self.input_summary_dir / param_suffix / f"portfolio_revenues_summary{param_suffix}.csv"
            
            if not summary_path.exists():
                if verbose:
                    print(f"Warning: Summary file not found for _lambda={lambda_risk} at {summary_path}")
                continue
            
            summary_df = pd.read_csv(summary_path)
            summary_df = summary_df.iloc[:-1]
            summary_df['lambda'] = lambda_risk
            summary_df['portfolio_type'] = 'optimized'
            all_summaries.append(summary_df)
            
            if verbose:
                print(f"  Loaded {len(summary_df)} windows for _lambda={lambda_risk}")
        
        if self.include_random:
            param_suffix = f"_({self.total_turbines})_(random)_({self.min_revenue_threshold})"
            summary_path = self.input_summary_dir / param_suffix / f"portfolio_revenues_summary{param_suffix}.csv"
            
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                summary_df = summary_df.iloc[:-1]

                summary_df['lambda'] = 'random'
                summary_df['portfolio_type'] = 'random'
                all_summaries.append(summary_df)
                
                if verbose:
                    print(f"  Loaded {len(summary_df)} windows for random baseline")
            elif verbose:
                print(f"Warning: Random baseline not found at {summary_path}")
        
        if not all_summaries:
            raise FileNotFoundError("No summary files found. Run compute_portfolio_revenues.py first.")
        
        combined_df = pd.concat(all_summaries, ignore_index=True)
        
        combined_df['eval_window_end'] = pd.to_datetime(combined_df['eval_window_end'])
        combined_df['eval_window_start'] = pd.to_datetime(combined_df['eval_window_start'])
        combined_df['train_window_start'] = pd.to_datetime(combined_df['train_window_start'])
        combined_df['train_window_end'] = pd.to_datetime(combined_df['train_window_end'])
        
        combined_df = combined_df.sort_values(['lambda', 'eval_window_end']).reset_index(drop=True)
        
        if verbose:
            print(f"\n  Total: {len(combined_df)} data points across {len(self.lambda_values)} lambda values")
        
        return combined_df

    def plot_revenues_over_time(self, verbose: bool = True, figsize: tuple = (14, 8)):
        """
        Plot portfolio total revenues over time for different lambda values.
        
        Args:
            verbose: Print information
            figsize: Figure size (width, height)
        """
        
        if verbose:
            print("Loading portfolio revenue summaries...")
        
        df = self.load_all_summaries(verbose=verbose)
        
        if verbose:
            print("\nCreating visualization...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_lambdas = sorted(df['lambda'].unique(), key=lambda x: (isinstance(x, str), x))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lambdas)))
        
        for i, lambda_risk in enumerate(unique_lambdas):
            df_lambda = df[df['lambda'] == lambda_risk].copy()
            
            if len(df_lambda) == 0:
                continue
            
            df_lambda = df_lambda.sort_values('eval_window_end')
            
            ax.plot(
                df_lambda['eval_window_end'],
                df_lambda['total_revenue'],
                marker='o',
                markersize=6,
                linewidth=2,
                label=f'_lambda = {lambda_risk}',
                color=colors[i],
                alpha=0.8
            )
        
        ax.set_xlabel('Evaluation Period End', fontsize=14)
        ax.set_ylabel('Total Revenue (EUR)', fontsize=14)
        ax.set_title(
            f'Portfolio Revenue Over Time\n({self.total_turbines} turbines)', 
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.tick_params(axis='y', labelsize=12)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fontsize=13,
            title='Risk Aversion',
            title_fontsize=14
        )
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"portfolio_revenues_over_time_({self.total_turbines})_({self.min_revenue_threshold}).png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"\nSaved figure to: {output_path}")
        
        plt.show()
        
        return fig, ax

    def plot_mean_hourly_revenues_over_time(self, verbose: bool = True, figsize: tuple = (14, 8)):
        """
        Plot portfolio mean hourly revenues over time for different lambda values.
        
        Args:
            verbose: Print information
            figsize: Figure size (width, height)
        """
        
        if verbose:
            print("Loading portfolio revenue summaries...")
        
        df = self.load_all_summaries(verbose=verbose)
        
        if verbose:
            print("\nCreating visualization...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_lambdas = sorted(df['lambda'].unique(), key=lambda x: (isinstance(x, str), x))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lambdas)))
        
        for i, lambda_risk in enumerate(unique_lambdas):
            df_lambda = df[df['lambda'] == lambda_risk].copy()
            
            if len(df_lambda) == 0:
                continue
            
            df_lambda = df_lambda.sort_values('eval_window_end')
            
            ax.plot(
                df_lambda['eval_window_end'],
                df_lambda['mean_hourly_revenue'],
                marker='o',
                markersize=6,
                linewidth=2,
                label=f'_lambda = {lambda_risk}',
                color=colors[i],
                alpha=0.8
            )
        
        ax.set_xlabel('Evaluation Period End', fontsize=14)
        ax.set_ylabel('Mean Hourly Revenue (EUR/hour)', fontsize=14)
        ax.set_title(
            f'Portfolio Mean Hourly Revenue Over Time\n({self.total_turbines} turbines)', 
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.2f}'))
        ax.tick_params(axis='y', labelsize=12)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fontsize=13,
            title='Risk Aversion',
            title_fontsize=14
        )
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"portfolio_mean_hourly_revenues_over_time_({self.total_turbines})_({self.min_revenue_threshold}).png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"\nSaved figure to: {output_path}")
        
        plt.show()
        
        return fig, ax

    def plot_revenues_over_time_logscale(self, verbose: bool = True, figsize: tuple = (14, 8)):
        """
        Plot portfolio total revenues over time for different lambda values (log scale).
        
        Args:
            verbose: Print information
            figsize: Figure size (width, height)
        """
        
        if verbose:
            print("Loading portfolio revenue summaries...")
        
        df = self.load_all_summaries(verbose=verbose)
        
        if verbose:
            print("\nCreating visualization (log scale)...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_lambdas = sorted(df['lambda'].unique(), key=lambda x: (isinstance(x, str), x))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lambdas)))
        
        for i, lambda_risk in enumerate(unique_lambdas):
            df_lambda = df[df['lambda'] == lambda_risk].copy()
            
            if len(df_lambda) == 0:
                continue
            
            df_lambda = df_lambda.sort_values('eval_window_end')
            
            ax.plot(
                df_lambda['eval_window_end'],
                df_lambda['total_revenue'],
                marker='o',
                markersize=6,
                linewidth=2,
                label=f'_lambda = {lambda_risk}',
                color=colors[i],
                alpha=0.8
            )
        
        ax.set_yscale('log')
        
        ax.set_xlabel('Evaluation Period End', fontsize=14)
        ax.set_ylabel('Total Revenue (EUR) - Log Scale', fontsize=14)
        ax.set_title(
            f'Portfolio Revenue Over Time (Log Scale)\n({self.total_turbines} turbines)', 
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax.set_axisbelow(True)
        
        ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fontsize=13,
            title='Risk Aversion',
            title_fontsize=14
        )
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"portfolio_revenues_over_time_logscale_({self.total_turbines})_({self.min_revenue_threshold}).png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"\nSaved figure to: {output_path}")
        
        plt.show()
        
        return fig, ax

    def plot_mean_hourly_revenues_over_time_logscale(self, verbose: bool = True, figsize: tuple = (14, 8)):
        """
        Plot portfolio mean hourly revenues over time for different lambda values (log scale).
        
        Args:
            verbose: Print information
            figsize: Figure size (width, height)
        """
        
        if verbose:
            print("Loading portfolio revenue summaries...")
        
        df = self.load_all_summaries(verbose=verbose)
        
        if verbose:
            print("\nCreating visualization (log scale)...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_lambdas = sorted(df['lambda'].unique(), key=lambda x: (isinstance(x, str), x))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lambdas)))
        
        for i, lambda_risk in enumerate(unique_lambdas):
            df_lambda = df[df['lambda'] == lambda_risk].copy()
            
            if len(df_lambda) == 0:
                continue
            
            df_lambda = df_lambda.sort_values('eval_window_end')
            
            ax.plot(
                df_lambda['eval_window_end'],
                df_lambda['mean_hourly_revenue'],
                marker='o',
                markersize=6,
                linewidth=2,
                label=f'_lambda = {lambda_risk}',
                color=colors[i],
                alpha=0.8
            )
        
        ax.set_yscale('log')
        
        ax.set_xlabel('Evaluation Period End', fontsize=14)
        ax.set_ylabel('Mean Hourly Revenue (EUR/hour) - Log Scale', fontsize=14)
        ax.set_title(
            f'Portfolio Mean Hourly Revenue Over Time (Log Scale)\n({self.total_turbines} turbines)',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax.set_axisbelow(True)
        
        ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fontsize=13,
            title='Risk Aversion',
            title_fontsize=14
        )
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"portfolio_mean_hourly_revenues_over_time_logscale_({self.total_turbines})_({self.min_revenue_threshold}).png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"\nSaved figure to: {output_path}")
        
        plt.show()
        
        return fig, ax


if __name__ == "__main__":
    visualiser = WindowRevenueVisualiser(target="paths_local")
    
    print("=" * 80)
    print("Creating Total Revenue Plot")
    print("=" * 80)
    visualiser.plot_revenues_over_time(verbose=True)
    
    print("\n" + "=" * 80)
    print("Creating Total Revenue Plot (Log Scale)")
    print("=" * 80)
    visualiser.plot_revenues_over_time_logscale(verbose=True)
    
    print("\n" + "=" * 80)
    print("Creating Mean Hourly Revenue Plot")
    print("=" * 80)
    visualiser.plot_mean_hourly_revenues_over_time(verbose=True)
    
    print("\n" + "=" * 80)
    print("Creating Mean Hourly Revenue Plot (Log Scale)")
    print("=" * 80)
    visualiser.plot_mean_hourly_revenues_over_time_logscale(verbose=True)
