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

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config['paths']['raw_data']) / "weather" / str(freq)
        self.output_dir = Path(self.config['paths']['intermediate_data']) / "parquet" / "weather" / str(freq)

    def convert_and_restructure(self):
        """Convert all .nc files to Parquet and restructure by station ID and year"""
        YEAR_RANGE = range(2005, 2025)
        all_nc_files = list(file for file in sorted(self.input_dir.glob("*/*.nc")) if file.parent.name.isdigit() and int(file.parent.name) in YEAR_RANGE)
        stations = set(map(self.get_station_id, all_nc_files))
        
        station_files = {station: [] for station in stations}
        for nc_file in all_nc_files:
            station_files[self.get_station_id(nc_file)].append(nc_file)

        years_count = {station: len(files) for station, files in station_files.items()}
        sns.histplot(list(years_count.values()), bins=50)
        plt.title("Distribution of Number of Years per Station")
        plt.xlabel("Number of Years")
        plt.ylabel("Count of Stations")
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", lw=0.5, axis='y')
        plt.savefig("reports/figures/weather_station_years_distribution.png")

        expected_years = len(YEAR_RANGE)
        filtered_stations = {station: files for station, files in station_files.items() if years_count[station] == expected_years}

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
            print(f"Failed to process station {station}: {e}")

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

if __name__ == "__main__":
    preprocessor = WeatherDataPreprocessor(freq='hourly')
    preprocessor.convert_and_restructure()