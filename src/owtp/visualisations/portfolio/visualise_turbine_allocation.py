import matplotlib.pyplot as plt
import owtp.config
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
import geopandas as gpd


def plot_turbine_allocation(
    target: Literal["paths", "paths_local"] = "paths_local",
    suffix: str = "unknown"
):
    """
    Visualise turbine allocation across France from portfolio optimization.
    
    Args:
        target: Configuration target ('paths' or 'paths_local')
        suffix: Suffix to identify which optimization results to load
    """
    # Load configuration
    config = owtp.config.load_yaml_config()
    
    # Load ALL mean revenue data for background
    mean_revenue_path = Path(config[target]['processed_data']) / "parquet" / "revenues" / "mean" / "mean_revenue.parquet"
    all_locations = pd.read_parquet(mean_revenue_path)
    
    # Load active portfolio weights
    data_path = Path(config[target]['processed_data']) / "parquet" / "portfolio_weights" / f"{suffix}" / f"portfolio_weights_active{suffix}.parquet"
    output_path = Path(config[target]['visualisations']) / "portfolio" / f"turbine_allocation{suffix}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading turbine allocation from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if len(df) == 0:
        print("No turbines allocated! Check optimization results.")
        return
    
    print(f"Loaded {len(all_locations)} total locations")
    print(f"Loaded {len(df)} locations with turbine allocations")
    print(f"Total turbines: {df['weight_integer'].sum()}")
    print(f"Turbine range: {df['weight_integer'].min()} - {df['weight_integer'].max()}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    # Size proportional to number of turbines, color by mean revenue
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'],
        s=df['weight_integer'] * 30,  # Scale point size
        c=df['mean_revenue'],
        cmap='YlOrRd',
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        zorder=5  # Ensure allocated points are on top
    )
    
    # Add text labels for number of turbines
    for _, row in df.iterrows():
        ax.annotate(
            f"{int(row['weight_integer'])}",
            xy=(row['longitude'], row['latitude']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=10
        )
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Wind Turbine Portfolio Allocation\n(overlaid on Mean Revenue Spatial Distribution)', fontsize=14, fontweight='bold')
    
    # Set proper aspect ratio for France's latitude
    mid_lat = all_locations['latitude'].mean()
    ax.set_aspect(1.0 / np.cos(np.radians(mid_lat)))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar for background mean revenue
    cbar_bg = plt.colorbar(background_scatter, ax=ax, label='Mean Revenue (EUR/hour) - All Locations', 
                          shrink=0.6, pad=0.06, location='left')
    
    # Add colorbar for allocated turbines
    cbar = plt.colorbar(scatter, ax=ax, label='Mean Revenue (EUR/hour) - Allocated', 
                       shrink=0.6, pad=0.02)
    
    # Add statistics text box
    stats_text = f"Total Turbines: {df['weight_integer'].sum()}\n"
    stats_text += f"Active Locations: {len(df)}\n"
    stats_text += f"Avg Turbines / Active Location: {df['weight_integer'].mean():.1f}\n"
    stats_text += f"Total Expected Return: {(df['mean_revenue'] * df['weight_integer']).sum():.2f} EUR/hour"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black'),
            fontsize=10)
    
    # Add legend for point sizes
    # Create dummy scatter points for legend
    legend_sizes = [10, 25, 50] if df['weight_integer'].max() >= 50 else [5, 15, df['weight_integer'].max()]
    legend_points = []
    for size in legend_sizes:
        if size <= df['weight_integer'].max():
            legend_points.append(ax.scatter([], [], s=size*30, c='gray', alpha=0.7, edgecolors='black', linewidths=1.5))
    
    # if legend_points:
    #     ax.legend(legend_points, [f'{int(s)} turbines' for s in legend_sizes if s <= df['weight_integer'].max()],
    #              loc='lower right', title='Allocation', framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    
    return fig, ax


if __name__ == "__main__":
    plot_turbine_allocation(target="paths_local", suffix="_(100)_(0.0001)_(10.0)")