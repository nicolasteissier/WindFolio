import owtp.config
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import xarray as xr
from typing import Literal
import dask
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class WeatherDataPreprocessor:
    """
    Converts all .nc weather data files to Parquet format in parallel (year-by-year, `num_workers` processes at a time) using Dask.
    Reorganizes output into intermediate/parquet/weather/{freq}/{station_id}/{year}
    """

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly', verbose: bool = True):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config['paths']['raw_data']) / "weather" / str(freq)
        self.output_dir = Path(self.config['paths']['intermediate_data']) / "parquet" / "weather" / str(freq)
        self.freq = freq

        if verbose:
            print(f"\nInitialized WeatherDataPreprocessor with frequency: {self.freq}")
            print(f"Input directory: {self.input_dir}")
            print(f"Output directory: {self.output_dir}")

    def convert_and_restructure(self, verbose: bool = True):
        """Convert all .nc files to Parquet and restructure by station ID and year"""
        YEAR_RANGE = range(2023, 2025)
        all_nc_files = list(file for file in sorted(self.input_dir.glob("*/*.nc")) if file.parent.name.isdigit() and int(file.parent.name) in YEAR_RANGE and not file.name.startswith("._"))

        if verbose:
            print(f"\nFound {len(all_nc_files)} .nc files for years {YEAR_RANGE.start}-{YEAR_RANGE.stop - 1}")

        stations = set(map(self.get_station_id, all_nc_files))

        station_files = {station: [] for station in stations}
        for nc_file in all_nc_files:
            station_files[self.get_station_id(nc_file)].append(nc_file)
        
        if verbose:
            print(f"\nGrouped .nc files by station. Number of stations: {len(stations)}")

        years_count = {station: len(files) for station, files in station_files.items()}
        stations_per_year = {year: sum(1 for files in station_files.values() if any(int(file.parent.name) == year for file in files)) for year in YEAR_RANGE}
        
        self.plot_histogram_of_years_per_station(years_count, verbose=verbose)
        self.plot_histogram_of_stations_per_year(stations_per_year, verbose=verbose)

        expected_years = len(YEAR_RANGE)
        filtered_stations = {station: files for station, files in station_files.items() if years_count[station] == expected_years}

        if verbose:
            print(f"\nFiltered stations to those with complete data ({expected_years} years). Number of remaining stations: {len(filtered_stations)}")

        # years_count = {station: len(files) for station, files in filtered_stations.items()}

        # if len(filtered_stations) < len(station_files) * 0.25:
        #     if verbose:
        #         print("\nLess than 25% of stations have complete data. Applying fallback filtering method.")
        #     filtered_stations = self.fallback_filtering_stations(station_files, expected_years, verbose=verbose)

        # Process each station in parallel
        process_map(self.process_station, filtered_stations.keys(),
                    [filtered_stations[station] for station in filtered_stations.keys()],
                    max_workers=4, chunksize=1)

    def process_station(self, station: str, nc_files: list[Path]):
        """Process all .nc files for a given station"""

        try:
            output_path = self.output_dir / f"{station}.parquet"
            if output_path.exists():
                return
            dfs = [self.convert_nc_into_df(nc_file) for nc_file in nc_files]
            combined_df = pd.concat(dfs)
            self.save_df_as_parquet(combined_df, output_path)
        except Exception as e:
            print(f"\nFailed to process station {station}: {e}")

    def convert_nc_into_df(self, nc_file: Path) -> pd.DataFrame:
        """Convert a single .nc file into a DataFrame"""
        try:
            ds = xr.open_dataset(nc_file)
            return ds.to_dataframe()
        except Exception as e:
            raise RuntimeError(f"Failed to convert {nc_file} to DataFrame.")
    
    def save_df_as_parquet(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame as Parquet file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)

    def build_output_path(self, nc_file: Path, year: int) -> Path:
        """Build output path based on station ID and year"""
        station_id = self.get_station_id(nc_file)
        return self.output_dir / station_id / f"{year}.parquet"

    def get_station_id(self, nc_file: Path) -> str:
        """Extract station ID from .nc filename"""
        return nc_file.stem.split('_MTO_')[0]
    
    def plot_histogram_of_years_per_station(self, years_count: dict, verbose: bool = True):
        """Plot histogram of number of years per station"""
        sns.histplot(list(years_count.values()), bins=50)
        plt.title(f"Distribution of Number of Years per Station in {self.freq} Data")
        plt.xlabel("Number of Years")
        plt.ylabel("Count of Stations")
        plt.yscale('linear')
        plt.grid(True, which="both", ls="--", lw=0.5, axis='y')
        plt.savefig(f"reports/figures/weather_station_years_distribution_{self.freq}.png")
        plt.close()

        if verbose:
            print(f"\nSaved weather station years distribution plot to reports/figures/weather_station_years_distribution_{self.freq}.png")

    def plot_histogram_of_stations_per_year(self, year_count: dict, verbose: bool = True):
        """Plot number of stations per year"""
        years = sorted(year_count.keys())
        counts = [year_count[year] for year in years]
        
        plt.figure(figsize=(12, 6))
        plt.bar(years, counts)
        plt.title(f"Number of Stations per Year in {self.freq} Data")
        plt.xlabel("Year")
        plt.ylabel("Number of Stations")
        plt.yscale('linear')
        plt.grid(True, which="both", ls="--", lw=0.5, axis='y')
        plt.xticks(years, rotation=45)
        plt.tight_layout()
        plt.savefig(f"reports/figures/weather_years_stations_distribution_{self.freq}.png")
        plt.close()

        if verbose:
            print(f"\nSaved weather years stations distribution plot to reports/figures/weather_years_stations_distribution_{self.freq}.png")

    def fallback_filtering_stations(self, station_files: dict, expected_years: int, verbose: bool = True) -> dict:
        """Fallback method to filter stations when data is incomplete (not all years present)"""
        filtered_stations = {}

        for i in range(expected_years):
            return # TODO

        if verbose:
            print(f"\nFallback filtering retained {len(filtered_stations)} stations with complete data.")

        return filtered_stations


if __name__ == "__main__":
    # preprocessor_hourly = WeatherDataPreprocessor(freq='hourly')
    # preprocessor_hourly.convert_and_restructure()

    preprocessor_6minute = WeatherDataPreprocessor(freq='6minute')
    preprocessor_6minute.convert_and_restructure()
