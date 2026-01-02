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
from dask.distributed import Client, LocalCluster
import os

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

        self.era5land_intermediate_dir = Path(self.config[target]['intermediate_data']) / "parquet" / "weather" / "era5_land" / "hourly"
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
        """Get all masked ERA5-Land .parquet files in intermediate directory"""
        return list(file for file in sorted(self.era5land_intermediate_dir.glob("*.parquet")) if not file.name.startswith("._"))

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
        """Apply France land mask to a dataset and save as Parquet"""
        try:
            # Change extension to .parquet
            output_path = self.era5land_intermediate_dir / input_path.with_suffix(".parquet").name
            if output_path.exists():
                return
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            ds = xr.open_dataset(input_path, engine="h5netcdf")
            ds_masked = self._apply_france_mask(ds, mask_ds)
            
            df = ds_masked.to_dataframe().reset_index()
            
            vars_to_keep = ["t2m", "u10", "v10", "sp"]
            keep_cols = ["valid_time", "latitude", "longitude"] + vars_to_keep
            
            df = df[keep_cols].dropna(subset=vars_to_keep, how="all", axis=0)

            df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
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
    

    def restructure(self, verbose: bool = True, n_workers: int = 4, spatial_resolution: float = 1.0) -> None:
        """
        Convert all masked files to Parquet with optimized parallelization.
        Uses distributed scheduler for better performance and memory management.
        Partitions by coarse spatial grid for efficient time-series access per location.
        
        Args:
            verbose: Print progress information
            n_workers: Number of Dask workers (defaults to CPU count)
            spatial_resolution: Grid resolution in degrees for partitioning (default: 1.0°)
                            Smaller = more files but faster spatial queries
                            Larger = fewer files but slower spatial queries
        """        
        if n_workers is None:
            n_workers = os.cpu_count() // 2
        
        # Distributed scheduler for performance monitoring and memory management
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=4,
            memory_limit='30GB',
            processes=True,
            dashboard_address=':8787'
        )
        client = Client(cluster)
        
        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")
        
        try:
            ddf = self._load_as_dask_df()
            
            # Create coarse spatial grid for partitioning
            ddf['lat_bin'] = (ddf['latitude'] / spatial_resolution).round().astype(int) * spatial_resolution
            ddf['lon_bin'] = (ddf['longitude'] / spatial_resolution).round().astype(int) * spatial_resolution
            
            if verbose:
                print(f"Spatial grid: {spatial_resolution}° resolution")
                print(f"Columns at write time: {list(ddf.columns)}")
                print(f"Number of partitions: {ddf.npartitions}")
                print(f"Writing Parquet dataset to {self.era5land_processed_dir}")
            
            with ProgressBar():
                ddf.to_parquet(
                    self.era5land_processed_dir,
                    engine="pyarrow",
                    compression="zstd",
                    write_index=False,
                    partition_on=["lat_bin", "lon_bin"],
                )
            
            if verbose:
                print("Done writing Parquet files.")
        finally:
            client.close()
            cluster.close()


    def _load_as_dask_df(self) -> dd.DataFrame:
        """Load all masked Parquet files with optimized configuration."""
        masked_files = self._get_masked_era5land_files()

        print(f"Loading {len(masked_files)} masked ERA5-Land Parquet files...")
        # blocksize controls partition size (smaller = more parallelism)
        ddf = dd.read_parquet(
            [str(file) for file in masked_files],
            engine="pyarrow",
            blocksize="128MB",
        )

        # Round lat/lon to 0.1 degree for consistent precision
        ddf["longitude"] = (ddf["longitude"] * 10).round() / 10
        ddf["latitude"] = (ddf["latitude"] * 10).round() / 10
        
        # Ensure valid_time is datetime type
        if ddf['valid_time'].dtype == 'object':
            ddf['valid_time'] = dd.to_datetime(ddf['valid_time'])
        
        ddf = ddf.map_partitions(lambda df: df.sort_values('valid_time'), meta=ddf)

        return ddf

if __name__ == "__main__":
    era5_preprocessor = Era5WeatherDataPreprocessor(target="paths_local")
    era5_preprocessor.restructure()
