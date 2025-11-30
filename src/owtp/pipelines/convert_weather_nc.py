import owtp.config
from pathlib import Path
from tqdm import tqdm
import xarray as xr
from typing import Literal
import dask

class WeatherDataConverter:
    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config['paths']['raw_data']) / "weather" / str(freq)
        self.output_dir = Path(self.config['paths']['intermediate_data']) / "parquet" / "weather" / str(freq)

    def convert_files(self):
        for year in tqdm(range(1845, 2025), desc="Converting yearly weather data"):
            nc_files = [f for f in self.input_dir.glob(f"{year}/*.nc") if not (self.output_dir / f.relative_to(self.input_dir).with_suffix('.parquet')).exists()]
            
            def process_file(nc_file):
                relative_path = nc_file.relative_to(self.input_dir).with_suffix('.parquet')
                output_path = self.output_dir / relative_path
                self.convert_file(nc_file, output_path)  # Your existing method
            
            # Dask parallelizes across ALL files simultaneously
            tasks = [dask.delayed(process_file)(f) for f in nc_files]
            dask.compute(tasks, scheduler='processes', num_workers=4)

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


if __name__ == "__main__":
    converter = WeatherDataConverter(freq='hourly')
    converter.convert_files()