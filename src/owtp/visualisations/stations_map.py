import pandas as pd
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import owtp.config
from tqdm import tqdm

def plot_stations_map():
    """Plots all pertinent weather stations on a map."""
    config = owtp.config.load_yaml_config()
    filtered_stations_dir = Path(config['paths']['intermediate_data']) / "parquet" / "weather" / "hourly"
    
    all_files = sorted(filtered_stations_dir.glob("*.parquet"))
    print(f"Found {len(all_files)} station files")

    locations: dict[str, Union[tuple[float, float], None]] = {file.stem: None for file in all_files}
    for file in tqdm(all_files, desc="Extracting station locations"):
        df = pd.read_parquet(file, columns=['lat', 'lon'])
        lat = df['lat'].iloc[0]
        lon = df['lon'].iloc[0]
        locations[file.stem] = (lon, lat)

    print(f"Extracted {len(locations)} station locations")

    gdf = gpd.GeoDataFrame(
        {'station_id': list(locations.keys()),
         'geometry': gpd.points_from_xy(
             [loc[0] for loc in locations.values() if loc is not None],
             [loc[1] for loc in locations.values() if loc is not None]
            )},
        crs="EPSG:4326"
    )
    
    print(f"Created GeoDataFrame with {len(gdf)} stations")
    
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    
    # Global map
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='lightgray', edgecolor='white')
    gdf.plot(ax=ax, color='red', markersize=20, alpha=0.6)
    
    plt.title("Weather Stations Map", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    
    output_path = Path("reports/figures/weather_stations_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_path}")
    plt.close()
    
    # France-focused map
    fig, ax = plt.subplots(figsize=(12, 10))
    world.plot(ax=ax, color='lightgray', edgecolor='white')
    gdf.plot(ax=ax, color='red', markersize=40, alpha=0.6)
    
    # Set limits to focus on France (approximate bounds)
    ax.set_xlim(-5, 10)
    ax.set_ylim(41, 52)
    
    plt.title("Weather Stations Map - France", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    
    output_path_france = Path("reports/figures/weather_stations_map_france.png")
    plt.savefig(output_path_france, dpi=300, bbox_inches='tight')
    print(f"France map saved to {output_path_france}")
    plt.close()

if __name__ == "__main__":
    plot_stations_map()