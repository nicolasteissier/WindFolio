import owtp.config
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import xarray as xr
from typing import Literal
import dask
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Era5DataPreprocessor:
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