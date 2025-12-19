import matplotlib.pyplot as plt
import owtp.config
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal


def plot_mean_revenue_distribution(
    target: Literal["paths", "paths_local"] = "paths_local"
):
    """
    Load and plot the mean revenue distribution across all locations.
    """
    # Load configuration and data
    config = owtp.config.load_yaml_config()
    data_path = Path(config[target]['processed_data']) / "parquet" / "revenues" / "mean" / "mean_revenue.parquet"
    output_path = Path(config[target]['visualisations']) / "revenues" / "mean_revenue_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading mean revenue data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    print(f"Loaded {len(df)} locations")
    print(f"Mean revenue range: {df['mean_revenue'].min():.2f} - {df['mean_revenue'].max():.2f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of mean revenue distribution
    axes[0].hist(df['mean_revenue'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Mean Revenue')
    axes[0].set_ylabel('Number of Locations')
    axes[0].set_title('Mean Revenue Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Total Locations: {len(df)}\n"
    stats_text += f"Mean: {df['mean_revenue'].mean():.2f}\n"
    stats_text += f"Median: {df['mean_revenue'].median():.2f}\n"
    stats_text += f"Std: {df['mean_revenue'].std():.2f}"
    axes[0].text(0.98, 0.97, stats_text,
                transform=axes[0].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    # Spatial distribution of mean revenue
    scatter = axes[1].scatter(
        df['longitude'], 
        df['latitude'],
        c=df['mean_revenue'],
        cmap='viridis',
        s=10,
        alpha=0.6
    )
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Mean Revenue Spatial Distribution')

    mid_lat = df['latitude'].mean()
    axes[1].set_aspect(1.0 / np.cos(np.radians(mid_lat)))
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Mean Revenue')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    plot_mean_revenue_distribution(target="paths_local")
