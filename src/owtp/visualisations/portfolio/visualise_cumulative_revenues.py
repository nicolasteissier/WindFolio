import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class CumulativeRevenueVisualiser:
    """
    Visualise cumulative portfolio revenues across rolling windows for different lambda values.
    """

    def __init__(self, target: Literal["paths", "paths_local"] = "paths_local"):
        self.config = owtp.config.load_yaml_config()
        
        self.input_summary_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_revenues_summary"
        self.output_dir = Path(self.config[target]['visualisations']) / "portfolio_revenues"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimization parameters from config
        self.total_turbines = self.config['mean_variance_optimization']['total_turbines']
        self.lambda_values = self.config['mean_variance_optimization']['lambda_values']
        self.min_revenue_threshold = self.config['mean_variance_optimization']['min_revenue_threshold']
        self.include_random = True  # Include random baseline in plots

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
                    print(f"⚠ Warning: Summary file not found for λ={lambda_risk} at {summary_path}")
                continue
            
            summary_df = pd.read_csv(summary_path)
            summary_df['lambda'] = lambda_risk
            summary_df['portfolio_type'] = 'optimized'
            all_summaries.append(summary_df)
            
            if verbose:
                print(f"  Loaded {len(summary_df)} windows for λ={lambda_risk}")
        
        # Load random baseline if requested
        if self.include_random:
            param_suffix = f"_({self.total_turbines})_(random)_({self.min_revenue_threshold})"
            summary_path = self.input_summary_dir / param_suffix / f"portfolio_revenues_summary{param_suffix}.csv"
            
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                summary_df['lambda'] = 'random'
                summary_df['portfolio_type'] = 'random'
                all_summaries.append(summary_df)
                
                if verbose:
                    print(f"  Loaded {len(summary_df)} windows for random baseline")
            elif verbose:
                print(f"⚠ Warning: Random baseline not found at {summary_path}")
        
        if not all_summaries:
            raise FileNotFoundError("No summary files found. Run compute_portfolio_revenues.py first.")
        
        combined_df = pd.concat(all_summaries, ignore_index=True)
        
        # Convert date columns to datetime
        combined_df['eval_window_end'] = pd.to_datetime(combined_df['eval_window_end'])
        combined_df['eval_window_start'] = pd.to_datetime(combined_df['eval_window_start'])
        combined_df['train_window_start'] = pd.to_datetime(combined_df['train_window_start'])
        combined_df['train_window_end'] = pd.to_datetime(combined_df['train_window_end'])
        
        # Sort by eval window end and lambda
        combined_df = combined_df.sort_values(['lambda', 'eval_window_end']).reset_index(drop=True)
        
        if verbose:
            print(f"\n  Total: {len(combined_df)} data points across {len(self.lambda_values)} lambda values")
        
        return combined_df

    def plot_cumulative_revenues(self, verbose: bool = True, figsize: tuple = (14, 8)):
        """
        Plot cumulative portfolio revenues over time for different lambda values.
        
        Args:
            verbose: Print information
            figsize: Figure size (width, height)
        """
        
        if verbose:
            print("Loading portfolio revenue summaries...")
        
        df = self.load_all_summaries(verbose=verbose)
        
        if verbose:
            print("\nCreating cumulative revenue visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique lambda values from data (includes random if available)
        unique_lambdas = sorted(df['lambda'].unique(), key=lambda x: (isinstance(x, str), x))
        
        # Define colors for different lambda values
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lambdas)))
        
        # Plot each lambda as a separate line
        for i, lambda_risk in enumerate(unique_lambdas):
            df_lambda = df[df['lambda'] == lambda_risk].copy()
            
            if len(df_lambda) == 0:
                continue
            
            # Sort by eval window end
            df_lambda = df_lambda.sort_values('eval_window_end').reset_index(drop=True)
            
            # Calculate cumulative revenue across windows
            df_lambda['cumulative_total_revenue'] = df_lambda['total_revenue'].cumsum()
            
            if verbose:
                print(f"\nλ={lambda_risk}:")
                print(f"  Total cumulative revenue: {df_lambda['cumulative_total_revenue'].iloc[-1]:,.2f} EUR")
                print(f"  Number of windows: {len(df_lambda)}")
            
            # Plot line and markers
            ax.plot(
                df_lambda['eval_window_end'],
                df_lambda['cumulative_total_revenue'],
                marker='o',
                markersize=6,
                linewidth=2,
                label=f'λ = {lambda_risk}',
                color=colors[i],
                alpha=0.8
            )
        
        # Formatting
        ax.set_xlabel('Evaluation Period End', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Total Revenue (EUR)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Cumulative Portfolio Revenue Over Time\n({self.total_turbines} turbines, min revenue ≥ {self.min_revenue_threshold} EUR/hour)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Format x-axis to show year-month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fontsize=11,
            title='Risk Aversion',
            title_fontsize=12
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"cumulative_revenues_over_time_({self.total_turbines})_({self.min_revenue_threshold}).png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"\n✓ Saved figure to: {output_path}")
        
        plt.show()
        
        return fig, ax

    def plot_cumulative_revenues_logscale(self, verbose: bool = True, figsize: tuple = (14, 8)):
        """
        Plot cumulative portfolio revenues over time for different lambda values (log scale).
        
        Args:
            verbose: Print information
            figsize: Figure size (width, height)
        """
        
        if verbose:
            print("Loading portfolio revenue summaries...")
        
        df = self.load_all_summaries(verbose=verbose)
        
        if verbose:
            print("\nCreating cumulative revenue visualization (log scale)...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique lambda values from data (includes random if available)
        unique_lambdas = sorted(df['lambda'].unique(), key=lambda x: (isinstance(x, str), x))
        
        # Define colors for different lambda values
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lambdas)))
        
        # Plot each lambda as a separate line
        for i, lambda_risk in enumerate(unique_lambdas):
            df_lambda = df[df['lambda'] == lambda_risk].copy()
            
            if len(df_lambda) == 0:
                continue
            
            # Sort by eval window end
            df_lambda = df_lambda.sort_values('eval_window_end').reset_index(drop=True)
            
            # Calculate cumulative revenue across windows
            df_lambda['cumulative_total_revenue'] = df_lambda['total_revenue'].cumsum()
            
            # Plot line and markers
            ax.plot(
                df_lambda['eval_window_end'],
                df_lambda['cumulative_total_revenue'],
                marker='o',
                markersize=6,
                linewidth=2,
                label=f'λ = {lambda_risk}',
                color=colors[i],
                alpha=0.8
            )
        
        # Set log scale
        ax.set_yscale('log')
        
        # Formatting
        ax.set_xlabel('Evaluation Period End', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Total Revenue (EUR) - Log Scale', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Cumulative Portfolio Revenue Over Time (Log Scale)\n({self.total_turbines} turbines, min revenue ≥ {self.min_revenue_threshold} EUR/hour)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Format x-axis to show year-month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fontsize=11,
            title='Risk Aversion',
            title_fontsize=12
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"cumulative_revenues_over_time_logscale_({self.total_turbines})_({self.min_revenue_threshold}).png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"\n✓ Saved figure to: {output_path}")
        
        plt.show()
        
        return fig, ax


if __name__ == "__main__":
    visualiser = CumulativeRevenueVisualiser(target="paths_local")
    
    # Plot cumulative revenues
    print("=" * 80)
    print("Creating Cumulative Revenue Plot")
    print("=" * 80)
    visualiser.plot_cumulative_revenues(verbose=True)
    
    # Plot cumulative revenues (log scale)
    print("\n" + "=" * 80)
    print("Creating Cumulative Revenue Plot (Log Scale)")
    print("=" * 80)
    visualiser.plot_cumulative_revenues_logscale(verbose=True)
