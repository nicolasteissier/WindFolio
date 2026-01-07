import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import geopandas as gpd
import geodatasets
from shapely.geometry import Point
from typing import Union


class AerisMissingValuesVisualisation:
    """
    Visualize missing wind speed values in AERIS datasets (hourly and 6-minute).
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()

        self.input_dir_hourly = Path(self.config[target]['intermediate_data']) / "parquet" / "weather" / "hourly"
        self.input_dir_6minute = Path(self.config[target]['intermediate_data']) / "parquet" / "weather" / "6minute"

        self.output_dir = Path(self.config[target]['visualisations']) / "aeris"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.missing_values_df_hourly = None
        self.missing_values_df_with_loc_hourly= None

        self.missing_values_df_6minute = None
        self.missing_values_df_with_loc_6minute= None

    def check_missing_ws_values(self, df):
        """Check missing values in wind speed column named 'ws'."""
        try:
            df = df[['ws']]
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
        except KeyError:
            raise KeyError("Error checking missing values: The dataframe does not contain a 'ws' column.")
        
        return missing_df
    
    def compute_missing_values_df(self, freq: Literal['hourly', '6minute'] = 'hourly', verbose=True) -> pd.DataFrame:
        """Compute missing wind speed values for all stations and return as a DataFrame."""
        missing_ws_stats_list = []

        weather_files = [f for f in Path(self.input_dir_hourly if freq == 'hourly' else self.input_dir_6minute).glob("*.parquet") if not f.name.startswith("._")]
                
        if verbose:
            print("\n")
        iterator = tqdm(weather_files, desc=f"Computing missing values pct for all stations ({freq})", disable=not verbose)
        for energy_file in iterator:
            station_id = energy_file.stem
            df_weather = pd.read_parquet(energy_file)

            missing_ws_df = self.check_missing_ws_values(df_weather)
            missing_ws_values = missing_ws_df.loc['ws', 'Missing Values']
            missing_ws_percentage = missing_ws_df.loc['ws', 'Percentage']

            missing_ws_stats_list.append({
                'station_id': station_id,
                'missing_ws_values': missing_ws_values,
                'missing_ws_percentage': missing_ws_percentage
            })

        if freq == 'hourly':
            self.missing_values_df_hourly = pd.DataFrame(missing_ws_stats_list)
        else:
            self.missing_values_df_6minute = pd.DataFrame(missing_ws_stats_list)

        nb_stations = len(missing_ws_stats_list)

        if verbose:
            print(f"\nComputed missing wind speed values for {nb_stations} stations ({freq})")

        return (self.missing_values_df_hourly if freq == 'hourly' else self.missing_values_df_6minute), nb_stations
    
    def load_missing_values_df_with_locations(self, freq: Literal['hourly', '6minute'] = 'hourly', verbose=True) -> pd.DataFrame:
        """Merge missing values DataFrame with station location data (lon, lat)."""
        meta_list = []
        weather_files = [f for f in Path(self.input_dir_hourly if freq == 'hourly' else self.input_dir_6minute).glob("*.parquet") if not f.name.startswith("._")]

        if verbose:
            print("\n")
        iterator = tqdm(weather_files, desc=f"Loading station metadata ({freq})", disable=not verbose)
        for f in iterator:
            station_id = f.stem
            try:
                meta = pd.read_parquet(f, columns=['lon', 'lat']).iloc[0]
                meta_list.append({'station_id': station_id, 'lon': float(meta['lon']), 'lat': float(meta['lat'])})
            except Exception:
                meta_list.append({'station_id': station_id, 'lon': None, 'lat': None})
                if verbose:
                    print(f"Could not read metadata for station {station_id}")

        meta_df = pd.DataFrame(meta_list)

        missing_ws_with_loc = (self.missing_values_df_hourly if freq == 'hourly' else self.missing_values_df_6minute).merge(meta_df, on='station_id', how='left')
        missing_ws_with_loc = missing_ws_with_loc.sort_values('missing_ws_percentage', ascending=False)

        if freq == 'hourly':
            self.missing_values_df_with_loc_hourly = missing_ws_with_loc
        else:
            self.missing_values_df_with_loc_6minute = missing_ws_with_loc

        nb_stations = len(missing_ws_with_loc)

        if verbose:
            print(f"\nMerged and loaded missing values with location data for {nb_stations} stations ({freq})")

        return missing_ws_with_loc, nb_stations
    
    def plot_nans_bar_chart(self, missing_ws_df_hourly: pd.DataFrame, missing_ws_df_6min: pd.DataFrame, verbose=True):
        """Plot histogram of missing wind speed percentages for both hourly and 6-minute data."""

        bins = np.linspace(0, 100, 31)  # 30 bins from 0 to 100
        counts_hourly, bin_edges = np.histogram(missing_ws_df_hourly['missing_ws_percentage'], bins=bins)
        counts_6min, _ = np.histogram(missing_ws_df_6min['missing_ws_percentage'], bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        bar_width = bin_width * 0.4
        offset = bar_width / 2

        plt.figure(figsize=(12, 6))
        plt.bar(bin_centers - offset, counts_hourly, width=bar_width, alpha=0.8, label='Hourly Data', color='blue')
        plt.bar(bin_centers + offset, counts_6min, width=bar_width, alpha=0.8, label='6-Minute Data', color='orange')
        plt.yscale('log')
        plt.xlabel('Percentage of Missing Wind Speed Values')
        plt.ylabel('Number of Stations')
        plt.title('Comparison of Missing Wind Speed Values Across Stations')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'missing_ws_bar_chart.png', dpi=300)
        if verbose:
            print(f"\nSaved bar chart for missing wind speed values.")

        plt.close()

    def plot_nans_cdf(self, missing_ws_df_hourly: pd.DataFrame, missing_ws_df_6min: pd.DataFrame, verbose=True):
        """Plot CDF of missing wind speed percentages for both hourly and 6-minute data."""

        fig, ax = plt.subplots(figsize=(10, 6))

        sorted_hourly = np.sort(missing_ws_df_hourly['missing_ws_percentage'])
        sorted_6min = np.sort(missing_ws_df_6min['missing_ws_percentage'])

        cdf_hourly = np.arange(1, len(sorted_hourly) + 1) / len(sorted_hourly) * 100
        cdf_6min = np.arange(1, len(sorted_6min) + 1) / len(sorted_6min) * 100

        ax.plot(sorted_hourly, cdf_hourly, linewidth=2, label='Hourly Data', color='blue')
        ax.plot(sorted_6min, cdf_6min, linewidth=2, label='6-Minute Data', color='orange')

        ax.set_xlabel('Percentage of Missing Wind Speed Values')
        ax.set_ylabel('Cumulative Percentage of Stations (%)')
        ax.set_title('CDF: What % of Stations Have ≤X% Missing Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'missing_ws_cdf.png', dpi=300)
        if verbose:
            print(f"\nSaved CDF plot for missing wind speed values.")

        plt.close()

    def plot_nans_violin(self, missing_ws_df_hourly: pd.DataFrame, missing_ws_df_6min: pd.DataFrame, verbose=True):
        """Plot violin plot of missing wind speed percentages for both hourly and 6-minute data."""

        fig, ax = plt.subplots(figsize=(8, 6))

        positions = [1, 2]
        parts = ax.violinplot([missing_ws_df_hourly['missing_ws_percentage'],
                            missing_ws_df_6min['missing_ws_percentage']], 
                            positions=positions, showmeans=True, showmedians=True)

        ax.set_xticks(positions)
        ax.set_xticklabels(['Hourly Data', '6-Minute Data'])
        ax.set_ylabel('Percentage of Missing Wind Speed Values')
        ax.set_title('Distribution Shape of Missing Values')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'missing_ws_violin_plot.png', dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\nSaved violin plot for missing wind speed values.")

        plt.close()

    def plot_nans_spatial_distribution(self, freq: Literal['hourly', '6minute'] = 'hourly', verbose=True):
        """Plot spatial distribution of missing wind speed values for the specified frequency."""
        if freq == 'hourly':
            missing_values_df_with_loc = self.missing_values_df_with_loc_hourly
        else:
            missing_values_df_with_loc = self.missing_values_df_with_loc_6minute

        geometry = [Point(xy) for xy in zip(missing_values_df_with_loc['lon'], 
                                            missing_values_df_with_loc['lat'])]
        gdf = gpd.GeoDataFrame(missing_values_df_with_loc, 
                            geometry=geometry, 
                            crs='EPSG:4326')  
        
        metro_france_bounds = {
            'lon_min': -5.5,
            'lon_max': 10.0,
            'lat_min': 41.0,
            'lat_max': 51.5
        }

        gdf_metro_france = gdf[
            (gdf['lon'] >= metro_france_bounds['lon_min']) &
            (gdf['lon'] <= metro_france_bounds['lon_max']) &
            (gdf['lat'] >= metro_france_bounds['lat_min']) &
            (gdf['lat'] <= metro_france_bounds['lat_max'])
        ]

        gdf_proj = gdf_metro_france.to_crs('EPSG:2154')
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        )
        france = world[world.NAME == 'France']
        france_proj = france.to_crs('EPSG:2154')

        fig, ax = plt.subplots(figsize=(9, 8))

        norm = Normalize(vmin=0, vmax=100)

        france_proj.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

        gdf_proj.plot(ax=ax, 
                      column='missing_ws_percentage',
                      cmap=cm.RdYlGn_r,
                      markersize=25,
                      alpha=1,
                      edgecolor='black',
                      linewidth=0.2,
                      legend=False,
                      norm=norm)

        ax.set_xlim(0, 1.3e6)
        ax.set_ylim(6e6, 7.2e6)
        ax.set_xlabel('Easting (m)', fontsize=11)
        ax.set_ylabel('Northing (m)', fontsize=11)
        ax.grid(True, alpha=0.3)

        sm = ScalarMappable(cmap=cm.RdYlGn_r, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Missing Wind Speed Values (%)', fontsize=12)

        plt.savefig(self.output_dir / f'missing_ws_spatial_distribution_{freq}_no_title.png', dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\nSaved spatial distribution plot for missing wind speed values ({freq}).")

        fig.suptitle(f'Spatial Distribution of Missing Wind Speed Data ({freq}) \n in Metropolitan France from ground-based stations dataset from Météo-France', 
                    fontsize=16, y=0.98)

        plt.savefig(self.output_dir / f'missing_ws_spatial_distribution_{freq}.png', dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved spatial distribution plot with title for missing wind speed values ({freq}).")

        plt.close()

    def plot_stations_map(self):
        """Plots all pertinent weather stations on a map."""
        filtered_stations_dir = self.input_dir_hourly
        
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
        
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color='lightgray', edgecolor='white')
        gdf.plot(ax=ax, color='red', markersize=20, alpha=0.6)
        
        plt.title("Weather Stations Map", fontsize=16)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)

        output_path = self.output_dir / 'weather_stations_map.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
        plt.close()
        
        # The France focused map:
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
        
        output_path_france = self.output_dir / "weather_stations_map_france.png"
        plt.savefig(output_path_france, dpi=300, bbox_inches='tight')
        print(f"France map saved to {output_path_france}")
        plt.close()


if __name__ == "__main__":
    visualiser = AerisMissingValuesVisualisation(target="paths_local")

    VISUALISE_HOURLY = True
    VISUALISE_6MINUTE = True

    if VISUALISE_HOURLY:
        missing_ws_df_hourly, nb_stations_hourly = visualiser.compute_missing_values_df(freq='hourly', verbose=True)
        if nb_stations_hourly == 0:
            VISUALISE_HOURLY = False
            print("\n /!\ Warning: No hourly stations found, skipping hourly visualisations. Check input data.")
    if VISUALISE_6MINUTE:
        missing_ws_df_6min, nb_stations_6min = visualiser.compute_missing_values_df(freq='6minute', verbose=True)
        if nb_stations_6min == 0:
            VISUALISE_6MINUTE = False
            print("\n /!\ Warning: No 6-minute stations found, skipping 6-minute visualisations. Check input data.")

    if VISUALISE_HOURLY:
        missing_ws_with_loc_hourly, nb_stations_with_loc_hourly = visualiser.load_missing_values_df_with_locations(freq='hourly', verbose=True)
        if nb_stations_with_loc_hourly == 0:
            VISUALISE_HOURLY = False
            print("\n /!\ Warning: No hourly stations with location data found, skipping hourly spatial visualisations. Check input data.")
    if VISUALISE_6MINUTE:
        missing_ws_with_loc_6min, nb_stations_with_loc_6min = visualiser.load_missing_values_df_with_locations(freq='6minute', verbose=True)
        if nb_stations_with_loc_6min == 0:
            VISUALISE_6MINUTE = False
            print("\n /!\ Warning: No 6-minute stations with location data found, skipping 6-minute spatial visualisations. Check input data.")

    if VISUALISE_HOURLY and VISUALISE_6MINUTE:
        visualiser.plot_nans_bar_chart(missing_ws_df_hourly, missing_ws_df_6min, verbose=True)
        visualiser.plot_nans_cdf(missing_ws_df_hourly, missing_ws_df_6min, verbose=True)
        visualiser.plot_nans_violin(missing_ws_df_hourly, missing_ws_df_6min, verbose=True)
    else:
        print("\n /!\ Warning: Skipping comparative visualisations due to lack of data in one or both frequencies.")
    
    if VISUALISE_HOURLY:
        visualiser.plot_nans_spatial_distribution(freq='hourly', verbose=True)
    if VISUALISE_6MINUTE:
        visualiser.plot_nans_spatial_distribution(freq='6minute', verbose=True)