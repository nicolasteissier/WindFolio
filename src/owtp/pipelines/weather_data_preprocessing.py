import owtp.config
from pathlib import Path
from tqdm import tqdm
import xarray as xr
from typing import Literal
import dask

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
        for year in tqdm(range(2000, 2025), desc="Converting yearly weather data"):
            nc_files = [f for f in (self.input_dir / str(year)).glob("*.nc")]
            
            def process_file(nc_file):
                output_path = self.build_output_path(nc_file, year)
                if output_path.exists():
                    return
                self.convert_file(nc_file, output_path)
            
            # Dask parallelizes across ALL files simultaneously
            tasks = [dask.delayed(process_file)(f) for f in nc_files] # type: ignore
            dask.compute(tasks, scheduler='processes', num_workers=4) # type: ignore

    def convert_file(self, nc_file: Path, output_path: Path):
        """Convert a single .nc file to Parquet format"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ds = xr.open_dataset(nc_file)
            df = ds.to_dataframe().reset_index()
            df.to_parquet(output_path)
            ds.close()
        except Exception as e:
            print(f"Failed to convert {nc_file}: {e}")

    def build_output_path(self, nc_file: Path, year: int) -> Path:
        """Build output path based on station ID and year"""
        station_id = nc_file.stem.split('_MTO_')[0]
        return self.output_dir / station_id / f"{year}.parquet"

if __name__ == "__main__":
    preprocessor = WeatherDataPreprocessor(freq='hourly')
    preprocessor.convert_and_restructure()