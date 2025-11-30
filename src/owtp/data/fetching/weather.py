import owtp.config
import requests
from siphon.catalog import TDSCatalog
from pathlib import Path
from tqdm import tqdm
import time
from typing import Literal

class WeatherDataFetcher:
    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.base_dir = Path(self.config['paths']['raw_data']) / "weather" / str(freq)
        
        if freq == 'hourly':
            self.catalog_base_url = "https://thredds-su.ipsl.fr/thredds/catalog/aeris_thredds/actrisfr_data/665029c8-82b8-4754-9ff4-d558e640b0ba"
            self.file_base_url = "https://thredds-su.ipsl.fr/thredds/fileServer/aeris_thredds/actrisfr_data/665029c8-82b8-4754-9ff4-d558e640b0ba"
        elif freq == '6minute':
            self.catalog_base_url = "https://thredds-su.ipsl.fr/thredds/catalog/aeris_thredds/actrisfr_data/cbe74172-66e4-4e18-b2cc-31ad11ed934d"
            self.file_base_url = "https://thredds-su.ipsl.fr/thredds/fileServer/aeris_thredds/actrisfr_data/cbe74172-66e4-4e18-b2cc-31ad11ed934d"
        else:
            raise ValueError("Frequency must be either 'hourly' or '6minute'")
        
        self.year_range = range(2000, 2025)

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

if __name__ == "__main__":
    fetcher = WeatherDataFetcher(freq='hourly')
    fetcher.fetch_weather_data()