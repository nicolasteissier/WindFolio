import owtp.config
import requests
from siphon.catalog import TDSCatalog
from pathlib import Path
from tqdm import tqdm
import time
from ecmwf.datastores import Client
from typing import Literal

class AerisWeatherDataFetcher:
    def __init__(self, target: Literal["paths", "paths_local"], freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.base_dir = Path(self.config[target]['raw_data']) / "weather" / "aeris" / str(freq)
        
        if freq == 'hourly':
            self.catalog_base_url = "https://thredds-su.ipsl.fr/thredds/catalog/aeris_thredds/actrisfr_data/665029c8-82b8-4754-9ff4-d558e640b0ba"
            self.file_base_url = "https://thredds-su.ipsl.fr/thredds/fileServer/aeris_thredds/actrisfr_data/665029c8-82b8-4754-9ff4-d558e640b0ba"
        elif freq == '6minute':
            self.catalog_base_url = "https://thredds-su.ipsl.fr/thredds/catalog/aeris_thredds/actrisfr_data/cbe74172-66e4-4e18-b2cc-31ad11ed934d"
            self.file_base_url = "https://thredds-su.ipsl.fr/thredds/fileServer/aeris_thredds/actrisfr_data/cbe74172-66e4-4e18-b2cc-31ad11ed934d"
        else:
            raise ValueError("Frequency must be either 'hourly' or '6minute'")
        
        self.year_range = range(2005, 2025)

    def fetch_weather_data(self):
        for year in tqdm(self.year_range, desc="Fetching yearly weather data"):
            year_catalog_url = f"{self.catalog_base_url}/{year}/catalog.xml"
            
            try:
                catalog = TDSCatalog(year_catalog_url)
                stations = self.get_nc_datasets(catalog)
            except Exception as e:
                print(f"Failed to fetch catalog for year {year}: {e}")
                continue

            for station in tqdm(stations, desc=f"Fetching station data for year {year}", leave=False):
                station_file = self.base_dir / str(year) / station
                if station_file.exists():
                    continue
                station_file.parent.mkdir(parents=True, exist_ok=True)

                station_url = f"{self.file_base_url}/{year}/{station}"
                self.download_file(station_url, station_file)
                
                time.sleep(0.05)  # Polite rate limiting
    
    def get_nc_datasets(self, catalog):
        """Extract all .nc datasets from THREDDS catalog"""
        nc_files = []
        for ds in catalog.datasets.values():
            if ds.name.endswith('.nc'):
                nc_files.append(ds.name)
        return nc_files
    
    def download_file(self, url, filepath):
        """Download with retry logic"""
        for attempt in range(3):
            try:
                resp = requests.get(url, stream=True, timeout=30)
                resp.raise_for_status()
                with open(filepath, 'w+b') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {filepath.name}: {e}")
                time.sleep(1)
        print(f"Failed to download {filepath.name} after 3 attempts")

class ERA5WeatherDataFetcher:
    """Fetcher for ERA5 weather data using CDS API"""

    areas = {
        "france": [51.1, -5.2, 41.3, 9.6],  # North, West, South, East
        "germany": [55.1, 5.9, 47.3, 15.0],
        "spain": [43.8, -9.3, 36.0, 3.3],
        "italy": [47.1, 6.6, 36.6, 18.5],
        "poland": [55.0, 14.1, 49.0, 24.2],
    }
    
    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.base_dir = Path(self.config[target]['raw_data']) / "weather" / "era5" / "hourly"
        
        self.client = Client(sleep_max=30)

        self.year_range = range(2005, 2025)

    def fetch_weather_data(self, target_zone: Literal["france"] = "france", verbose: bool = False):
        """Batch download ERA5 single-levels for France (2005-2024)"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        area = self.areas[target_zone]

        variables = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'surface_pressure',
        ]
        
        years = list(self.year_range)
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        days = [f'{d:02d}' for d in range(1, 32)]
        # Generate times based on frequency (e.g., every 1, 5, 10, 15, 30, or 60 minutes)
        times = [f'{h:02d}:00' for h in range(0, 24)]
        
        # Download by month to reduce number of requests
        for year in tqdm(years, desc="Downloading ERA5 data by year"):
            for month in tqdm(months, desc=f"Year {year} months"):
                    filename = self.base_dir / f'{year}_{month}.nc'
                    
                    if filename.exists():
                        if verbose:
                            print(f"Skipping {filename} (already exists)")
                        continue
                        
                    if verbose:
                        print(f"Queueing ERA5 {year} {month} download...")
                    
                    self.client.retrieve(
                        'reanalysis-era5-land',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': variables,
                            'year': str(year),
                            'month': str(month),
                            'day': days,
                            'time': times,
                            'area': area,
                        },
                        str(filename)
                    )
                    if verbose:
                        print(f"Queued: {filename}")


if __name__ == "__main__":
    fetcher = ERA5WeatherDataFetcher(target="paths_local")
    fetcher.fetch_weather_data(verbose=True)