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
    vmin_bg: float = None,
    vmax_bg: float = None,
    vmin_alloc: float = None,
    vmax_alloc: float = None,
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
        vmin_bg: Minimum value for background colormap
        vmax_bg: Maximum value for background colormap
        vmin_alloc: Minimum value for allocated locations colormap
        vmax_alloc: Maximum value for allocated locations colormap
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
        vmin=vmin_bg,
        vmax=vmax_bg,
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
        vmin=vmin_alloc,
        vmax=vmax_alloc,
        zorder=5
    )
    
    # Add text labels for number of turbines
    for _, row in df.iterrows():
        ax.annotate(
            f"{int(row['weight_integer'])}",
            xy=(row['longitude'], row['latitude']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=10
        )
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
    ax.set_title(f'λ = {lambda_value}', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    
    # Set proper aspect ratio
    mid_lat = all_locations['latitude'].mean()
    ax.set_aspect(1.0 / np.cos(np.radians(mid_lat)))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f"Turbines: {df['weight_integer'].sum() if df['weight_integer'].sum() <= 100 else 100}\n"
    stats_text += f"Locations: {len(df)}\n"
    stats_text += f"Avg/Loc: {df['weight_integer'].mean():.1f}"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=2))
    
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
    output_path = Path(config[target]['visualisations']) / "portfolio" / window_suffix / f"turbine_allocation_comparison_{lambda_values[0]}_{lambda_values[1]}_{lambda_values[2]}_{lambda_values[3]}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing window: {window_suffix}")
    print(f"  Total locations in window: {len(all_locations)}")
    
    # Create figure with subplots (2x2 grid for 4 lambda values)
    fig = plt.figure(figsize=(22, 16))
    
    # Create GridSpec for better layout control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.88, top=0.93, bottom=0.07, 
                          hspace=0.25, wspace=0.05)
    
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    
    # First pass: Load all data to determine global color scale
    portfolio_dfs = []
    all_allocated_revenues = []
    
    for lambda_val in lambda_values:
        suffix = f"_({total_turbines})_({lambda_val})_({min_revenue_threshold})"
        data_path = Path(config[target]['processed_data']) / "parquet" / "portfolio_weights" / suffix / window_suffix / f"portfolio_weights_active{suffix}.parquet"
        
        if data_path.exists():
            df = pd.read_parquet(data_path)
            portfolio_dfs.append(df)
            if len(df) > 0:
                all_allocated_revenues.extend(df['mean_revenue'].tolist())
        else:
            portfolio_dfs.append(pd.DataFrame())
    
    # Calculate global color scale limits
    vmin_bg = all_locations['mean_revenue'].min()
    vmax_bg = all_locations['mean_revenue'].max()
    
    if all_allocated_revenues:
        vmin_alloc = min(all_allocated_revenues)
        vmax_alloc = max(all_allocated_revenues)
    else:
        vmin_alloc = vmin_bg
        vmax_alloc = vmax_bg
    
    print(f"  Global color scale - Background: [{vmin_bg:.2f}, {vmax_bg:.2f}]")
    print(f"  Global color scale - Allocated: [{vmin_alloc:.2f}, {vmax_alloc:.2f}]")
    
    # Second pass: Plot with global color scale
    scatter_objects_bg = []
    scatter_objects_alloc = []
    
    for idx, lambda_val in enumerate(lambda_values):
        df = portfolio_dfs[idx]
        
        if len(df) == 0:
            print(f"  Warning: No data found for λ={lambda_val}")
            axes[idx].text(0.5, 0.5, f'No data for λ = {lambda_val}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'λ = {lambda_val}', fontsize=11, fontweight='bold')
            continue
        
        print(f"  λ={lambda_val}: {len(df)} locations, {df['weight_integer'].sum()} turbines")
        
        # Plot on corresponding subplot with global color scale
        show_ylabel = (idx % 2 == 0)  # Only show ylabel on left column
        bg_scatter, alloc_scatter = plot_turbine_allocation_single(
            axes[idx], 
            all_locations, 
            df, 
            lambda_val, 
            total_turbines,
            vmin_bg=vmin_bg,
            vmax_bg=vmax_bg,
            vmin_alloc=vmin_alloc,
            vmax_alloc=vmax_alloc,
            show_ylabel=show_ylabel
        )
        scatter_objects_bg.append(bg_scatter)
        scatter_objects_alloc.append(alloc_scatter)
    
    # Add shared colorbars for the entire figure
    if scatter_objects_bg and scatter_objects_alloc:
        # Colorbar for background (all France locations) - on the right, top position
        cbar_ax_bg = fig.add_axes([0.90, 0.52, 0.02, 0.38])
        cbar_bg = fig.colorbar(scatter_objects_bg[0], cax=cbar_ax_bg, orientation='vertical')
        cbar_bg.set_label('All Locations\n(EUR/hour)', fontsize=14, weight='bold')
        cbar_bg.ax.tick_params(labelsize=11)
        
        # Colorbar for allocated locations - on the right, bottom position
        cbar_ax_alloc = fig.add_axes([0.90, 0.08, 0.02, 0.38])
        cbar_alloc = fig.colorbar(scatter_objects_alloc[0], cax=cbar_ax_alloc, orientation='vertical')
        cbar_alloc.set_label('Allocated Locations\n(EUR/hour)', fontsize=14, weight='bold')
        cbar_alloc.ax.tick_params(labelsize=11)
    
    # Add overall title
    fig.suptitle(f'Wind Turbine Portfolio Allocation - Window {window_suffix} - {total_turbines} turbines', 
                 fontsize=20, fontweight='bold', y=0.98)
    
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
    
    print(f"\nAll {len(windows)} windows processed successfully!")


if __name__ == "__main__":
    plot_all_windows(target="paths_local")