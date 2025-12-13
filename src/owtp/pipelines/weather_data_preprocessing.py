import owtp.config
from typing import Literal
from pathlib import Path

# Processing
from tqdm.contrib.concurrent import process_map
import xarray as xr
import pandas as pd
import geopandas as gpd
import regionmask
import dask
import dask.dataframe as dd
from dask.diagnostics.progress import ProgressBar

# Plotting
import seaborn as sns
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

class Era5WeatherDataPreprocessor:
    """
    Converts all .nc weather data files to Parquet format in parallel (year-by-year, `num_workers` processes at a time) using Dask.
    Reorganizes output into intermediate/parquet/weather/{freq}/{station_id}/{year}
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        
        self.era5land_input_dir = Path(self.config[target]['raw_data']) / "weather" / "era5_land" / "hourly"
        self.era5land_input_dir.mkdir(parents=True, exist_ok=True)

        self.era5land_intermediate_dir = Path(self.config[target]['intermediate_data']) / "nc" / "weather" / "era5_land" / "hourly"
        self.era5land_intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.era5land_processed_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather" / "era5_land" / "hourly"
        self.era5land_processed_dir.mkdir(parents=True, exist_ok=True)

        self.france_mask_path = Path(self.config[target]['masks']) / "france_land_mask.nc"
        self.france_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.france_mask_path.exists():
            print("Creating France land mask...")
            self._create_france_mask(
                output_path=self.france_mask_path,
            )
            print(f"France land mask saved to {self.france_mask_path}")

    def _get_era5land_files(self) -> list[Path]:
        """Get all raw ERA5-Land .nc files"""
        return list(file for file in sorted(self.era5land_input_dir.glob("*.nc")) if not file.name.startswith("._"))
    
    def _get_masked_era5land_files(self) -> list[Path]:
        """Get all masked ERA5-Land .nc files in intermediate directory"""
        return list(file for file in sorted(self.era5land_intermediate_dir.glob("*.nc")) if not file.name.startswith("._"))

    def mask_raw_files(self) -> None:
        """Apply France land mask to all raw ERA5-Land .nc files and save to intermediate directory"""
        print("Loading France land mask...")
        mask_ds = self._load_france_mask()

        print("Applying France land mask to raw ERA5-Land files...")
        input_files = self._get_era5land_files()

        process_map(
            self._apply_mask_on,
            input_files,
            [mask_ds] * len(input_files),
            max_workers=4,
            chunksize=1,
        )

    def _apply_mask_on(self, input_path: Path, mask_ds: xr.Dataset) -> None:
        """Apply France land mask to a dataset"""
        try:
            relative_path = input_path.relative_to(self.era5land_input_dir)
            
            output_path = self.era5land_intermediate_dir / relative_path
            if output_path.exists():
                return
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            ds = xr.open_dataset(input_path, engine="h5netcdf")
            ds_masked = self._apply_france_mask(ds, mask_ds)
            
            ds_masked.to_netcdf(output_path, engine="h5netcdf")
        except Exception as e:
            print(f"\nFailed to apply mask on {input_path}: {e}")

    def _create_france_mask(self, output_path: Path) -> None:
        # Source "https://simplemaps.com/static/svg/country/fr/all/fr.json"
        json_path = output_path.parent / "fr.json"
        mask_geodf = gpd.read_file(json_path)

        france_shape = regionmask.from_geopandas(mask_geodf)

        # Grid from reference file
        reference_path = self._get_era5land_files()[0]
        ds_ref = xr.open_dataset(reference_path, engine="h5netcdf")
        lons = ds_ref["longitude"]
        lats = ds_ref["latitude"]

        # Region index mask on this grid
        mask_array = france_shape.mask(lons, lats)  # 2D (lat, lon)

        # Assume single region, take its index explicitly
        fr_idx = france_shape.numbers[0]
        inside_france = (mask_array == fr_idx)

        # Save boolean mask
        mask_ds = xr.Dataset(
            data_vars={"mask_france": (("latitude", "longitude"), inside_france.data)},
            coords={"latitude": lats, "longitude": lons},
        )
        mask_ds.to_netcdf(output_path, engine="h5netcdf")


    def _load_france_mask(self) -> xr.Dataset:
        """Load the France land mask dataset"""
        if not self.france_mask_path.exists():
            self._create_france_mask(
                output_path=self.france_mask_path,
            )

        return xr.open_dataset(self.france_mask_path)

    def _apply_france_mask(self, ds: xr.Dataset, mask_ds: xr.Dataset) -> xr.Dataset:
        mask = mask_ds["mask_france"]
        return ds.where(mask, other=float("nan"))
    

    def convert_and_restructure(self, verbose: bool = True) -> None:
        """
        Convert all masked .nc files to Parquet with only:
            ['valid_time', 'latitude', 'longitude', 't2m', 'u10', 'v10', 'sp']
        and write to era5land_processed_dir, dropping all-NaN rows.
        """
        ddf = self._load_as_dask_df()

        vars_to_keep = ["t2m", "u10", "v10", "sp"]
        keep_cols = ["valid_time", "latitude", "longitude"] + vars_to_keep

        missing = [c for c in keep_cols if c not in ddf.columns]
        if missing:
            raise RuntimeError(f"Missing expected columns in DataFrame: {missing}")

        ddf = ddf[keep_cols]

        if verbose:
            print("Columns at write time:", list(ddf.columns))
            print(f"Writing Parquet dataset to {self.era5land_processed_dir}")

        with ProgressBar():
            ddf.to_parquet(
                self.era5land_processed_dir,
                engine="pyarrow",
                compression="snappy",
                write_index=False,
                partition_on=["latitude", "longitude"],  # directory per (lat, lon)
            )

        if verbose:
            print("Done.")



    def _load_as_dask_df(self) -> dd.DataFrame:
        """Load all masked .nc files, flatten to a Dask DataFrame and drop all-NaN rows."""
        masked_files = self._get_masked_era5land_files()

        print("Loading masked ERA5-Land files...")
        ds = xr.open_mfdataset(
            [str(p) for p in masked_files],
            combine="by_coords",
            parallel=True,
            engine="h5netcdf",
            chunks={"valid_time": 2 * 31 * 24, "latitude": -1, "longitude": -1},
        )

        print("Converting to Dask DataFrame...")
        ddf = ds.to_dask_dataframe()        # dims/coords -> index/columns
        ddf = ddf.reset_index()             # ensure valid_time, latitude, longitude are columns

        ddf["longitude"] = ddf["longitude"].round(1)
        ddf["latitude"] = ddf["latitude"].round(1)

        # Drop rows where all vars of interest are NaN
        vars_to_keep = ["t2m", "u10", "v10", "sp"]
        ddf = ddf.dropna(subset=vars_to_keep, how="all")

        return ddf

if __name__ == "__main__":
    era5_preprocessor = Era5WeatherDataPreprocessor(target="paths")
    #era5_preprocessor.mask_raw_files()
    era5_preprocessor.convert_and_restructure()
