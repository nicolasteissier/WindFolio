import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import box
from pathlib import Path

def visualize_bounding_box():
    """Visualize the ERA5 download bounding box"""
    
    # Define bounding box from weather.py
    # area = [North, West, South, East]
    area = [51.1, -5.2, 41.3, 9.6]
    north, west, south, east = area
    
    # Create bounding box polygon
    bbox = box(west, south, east, north)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")
    
    # Load world map
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot world (focus on Europe)
    world.plot(ax=ax, color='lightgray', edgecolor='white')
    
    # Plot bounding box
    bbox_gdf.boundary.plot(ax=ax, color='red', linewidth=2, label='ERA5 Download Area')
    bbox_gdf.plot(ax=ax, color='red', alpha=0.2)
    
    # Set limits to show France and surrounding area
    ax.set_xlim(-10, 15)
    ax.set_ylim(38, 54)
    
    plt.title("ERA5 Download Bounding Box", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    
    # Save
    output_path = Path("reports/figures/era5_bounding_box.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bounding box visualization saved to {output_path}")
    plt.close()
    
    # Print info
    print(f"\nBounding Box Coordinates:")
    print(f"  North: {north}°")
    print(f"  South: {south}°")
    print(f"  West:  {west}°")
    print(f"  East:  {east}°")
    print(f"\nArea dimensions:")
    print(f"  Width:  {east - west:.1f}° longitude")
    print(f"  Height: {north - south:.1f}° latitude")

if __name__ == "__main__":
    visualize_bounding_box()
