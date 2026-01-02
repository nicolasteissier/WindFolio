import matplotlib.pyplot as plt
import owtp.config
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
import geopandas as gpd

from owtp.others import rolling


def plot_turbine_allocation_single(
    ax,
    all_locations: pd.DataFrame,
    df: pd.DataFrame,
    lambda_value: float,
    total_turbines: int,
    show_colorbar: bool = True,
    show_ylabel: bool = True
):
    """
    Plot turbine allocation for a single lambda value on a given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        all_locations: DataFrame with all locations for background
        df: DataFrame with active portfolio weights
        lambda_value: Lambda risk parameter value
        total_turbines: Total number of turbines
        show_colorbar: Whether to show colorbar
        show_ylabel: Whether to show ylabel
    """
    
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No turbines allocated', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'λ = {lambda_value}', fontsize=11, fontweight='bold')
        return
    
    # Plot ALL locations as background (Mean Revenue Spatial Distribution)
    background_scatter = ax.scatter(
        all_locations['longitude'], 
        all_locations['latitude'],
        c=all_locations['mean_revenue'],
        cmap='viridis',
        s=10,
        alpha=0.6,
        zorder=1
    )
    
    # Plot allocated turbine locations on top
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'],
        s=df['weight_integer'] * 30,  # Scale point size
        c=df['mean_revenue'],
        cmap='YlOrRd',
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        zorder=5
    )
    
    # Add text labels for number of turbines
    for _, row in df.iterrows():
        ax.annotate(
            f"{int(row['weight_integer'])}",
            xy=(row['longitude'], row['latitude']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=7,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=10
        )
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=10)
    if show_ylabel:
        ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(f'λ = {lambda_value}', fontsize=11, fontweight='bold')
    
    # Set proper aspect ratio
    mid_lat = all_locations['latitude'].mean()
    ax.set_aspect(1.0 / np.cos(np.radians(mid_lat)))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f"Turbines: {df['weight_integer'].sum()}\n"
    stats_text += f"Locations: {len(df)}\n"
    stats_text += f"Avg/Loc: {df['weight_integer'].mean():.1f}"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black'))
    
    return background_scatter, scatter


def plot_turbine_allocation_window(
    window_suffix: str,
    target: Literal["paths", "paths_local"] = "paths_local",
    total_turbines: int = 100,
    min_revenue_threshold: float = 10.0,
    lambda_values: list[float] = [0, 0.1, 0.5, 1.0]
):
    """
    Visualise turbine allocation for all lambda values in a single window.
    
    Args:
        window_suffix: Window identifier string (e.g., "2005-04-24_2010-04-23")
        target: Configuration target ('paths' or 'paths_local')
        total_turbines: Total number of turbines
        min_revenue_threshold: Minimum revenue threshold
        lambda_values: List of lambda risk parameters
    """
    # Load configuration
    config = owtp.config.load_yaml_config()
    
    # Load mean revenue data for this window (for background)
    mean_revenue_path = Path(config[target]['processed_data']) / "parquet" / "revenues" / "mean" / f"{window_suffix}.parquet"
    all_locations = pd.read_parquet(mean_revenue_path)
    mean_revenue_path = Path(config[target]['processed_data']) / "parquet" / "revenues" / "mean" / f"{window_suffix}.parquet"
    all_locations = pd.read_parquet(mean_revenue_path)
    
    # Output path for this window
    output_path = Path(config[target]['visualisations']) / "portfolio" / window_suffix / f"turbine_allocation_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing window: {window_suffix}")
    print(f"  Total locations in window: {len(all_locations)}")
    
    # Create figure with subplots (2x2 grid for 4 lambda values)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    # Load and plot for each lambda value
    portfolio_dfs = []
    for idx, lambda_val in enumerate(lambda_values):
        suffix = f"_({total_turbines})_({lambda_val})_({min_revenue_threshold})"
        data_path = Path(config[target]['processed_data']) / "parquet" / "portfolio_weights" / suffix / window_suffix / f"portfolio_weights_active{suffix}.parquet"
        
        if not data_path.exists():
            print(f"  Warning: No data found for λ={lambda_val} at {data_path}")
            axes[idx].text(0.5, 0.5, f'No data for λ = {lambda_val}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'λ = {lambda_val}', fontsize=11, fontweight='bold')
            continue
        
        df = pd.read_parquet(data_path)
        portfolio_dfs.append(df)
        
        print(f"  λ={lambda_val}: {len(df)} locations, {df['weight_integer'].sum()} turbines")
        
        # Plot on corresponding subplot
        show_ylabel = (idx % 2 == 0)  # Only show ylabel on left column
        scatters = plot_turbine_allocation_single(
            axes[idx], 
            all_locations, 
            df, 
            lambda_val, 
            total_turbines,
            show_colorbar=True,
            show_ylabel=show_ylabel
        )
    
    # Add overall title
    fig.suptitle(f'Wind Turbine Portfolio Allocation - Window {window_suffix}\n'
                 f'({total_turbines} turbines, min revenue ≥ {min_revenue_threshold} EUR/hour)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved to {output_path}")
    
    plt.close(fig)
    
    return fig, axes


def plot_all_windows(
    target: Literal["paths", "paths_local"] = "paths_local"
):
    """
    Plot turbine allocation for all windows defined in the configuration.
    
    Args:
        target: Configuration target ('paths' or 'paths_local')
    """
    config = owtp.config.load_yaml_config()
    
    # Get parameters from config
    total_turbines = config['mean_variance_optimization']['total_turbines']
    lambda_values = config['mean_variance_optimization']['lambda_values']
    min_revenue_threshold = config['mean_variance_optimization']['min_revenue_threshold']
    
    # Get rolling windows
    start = pd.Timestamp(config['rolling_calibrations']['start_date'])
    end = pd.Timestamp(config['rolling_calibrations']['end_date'])
    windows = rolling.get_windows(start, end)
    
    print(f"Processing {len(windows)} windows...")
    print(f"Parameters: {total_turbines} turbines, λ ∈ {lambda_values}, min revenue ≥ {min_revenue_threshold}")
    
    for window in windows:
        window_suffix = rolling.format_window_str(window['train_window_start'], window['train_window_end'])
        plot_turbine_allocation_window(
            window_suffix=window_suffix,
            target=target,
            total_turbines=total_turbines,
            min_revenue_threshold=min_revenue_threshold,
            lambda_values=lambda_values
        )
    
    print(f"\n✓ All {len(windows)} windows processed successfully!")


if __name__ == "__main__":
    plot_all_windows(target="paths_local")