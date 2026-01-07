import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import box
from pathlib import Path

from owtp.data.fetching.weather import ERA5WeatherDataFetcher
import itertools

def visualize_all_bounding_boxes():
    """Visualize all ERA5 download bounding boxes"""

    available_zones = list(ERA5WeatherDataFetcher.areas.keys())
    colors = plt.cm.get_cmap('tab10', len(available_zones))
    color_cycle = itertools.cycle([colors(i) for i in range(len(available_zones))])

    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))

    fig, ax = plt.subplots(figsize=(12, 10))
    world.plot(ax=ax, color='lightgray', edgecolor='white')

    bbox_gdfs = []
    for zone in available_zones:
        area = ERA5WeatherDataFetcher.areas[zone]
        north, west, south, east = area
        bbox = box(west, south, east, north)
        bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")
        bbox_gdf.boundary.plot(ax=ax, linewidth=2, label=f'{zone} boundary', color=next(color_cycle))
        bbox_gdf.plot(ax=ax, alpha=0.2, label=f'{zone} area')
        bbox_gdfs.append((zone, bbox_gdf, area))

    global_zone = [
        max(ERA5WeatherDataFetcher.areas[z][0] for z in available_zones),  # North
        min(ERA5WeatherDataFetcher.areas[z][1] for z in available_zones),  # West
        min(ERA5WeatherDataFetcher.areas[z][2] for z in available_zones),  # South
        max(ERA5WeatherDataFetcher.areas[z][3] for z in available_zones),  # East
    ]

    global_bbox = box(global_zone[1], global_zone[2], global_zone[3], global_zone[0])
    global_bbox_gdf = gpd.GeoDataFrame({'geometry': [global_bbox]}, crs="EPSG:4326")
    global_bbox_gdf.boundary.plot(ax=ax, linewidth=3, edgecolor='black', label='Global bounding box')

    ax.set_xlim(-12, 30)
    ax.set_ylim(35, 65)
    plt.title("ERA5 Download Bounding Boxes", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)

    output_path = Path("reports/figures/era5/all_bounding_boxes.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All bounding boxes visualization saved to {output_path}")
    plt.close()

    for zone, _, area in bbox_gdfs:
        north, west, south, east = area
        print(f"\nZone: {zone}")
        print(f"  North: {north}°")
        print(f"  South: {south}°")
        print(f"  West:  {west}°")
        print(f"  East:  {east}°")
        print(f"  Width:  {east - west:.1f}° longitude")
        print(f"  Height: {north - south:.1f}° latitude")

if __name__ == "__main__":
    visualize_all_bounding_boxes()
