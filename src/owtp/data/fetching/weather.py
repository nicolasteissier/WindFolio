import owtp.config
import requests
from requests import Response
import regex
from pathlib import Path
from tqdm import tqdm

class HourlyWeatherDataFetcher:
    def __init__(self):
        self.config = owtp.config.load_yaml_config()

        self.base_dir = Path(self.config['paths']['raw_data']) / "weather/hourly"

        self.catalog_base_url = "https://thredds-su.ipsl.fr/thredds/catalog/aeris_thredds/actrisfr_data/665029c8-82b8-4754-9ff4-d558e640b0ba"
        self.catalog_url_suffix = "catalog.html"

        self.file_base_url = "https://thredds-su.ipsl.fr/thredds/fileServer/aeris_thredds/actrisfr_data/665029c8-82b8-4754-9ff4-d558e640b0ba"

#        self.year_range = range(1845, 2024)
        self.year_range = range(2000, 2002)


    def fetch_weather_data(self):
        for year in tqdm(self.year_range, desc="Fetching yearly weather data"):
            year_url = f"{self.catalog_base_url}/{year}/{self.catalog_url_suffix}"
            response = requests.get(year_url)
            stations = self.parse_year_available_stations(response)
            if response.status_code != 200:
                print(f"Failed to fetch data for year {year}: {response.status_code}. Error message: {response.text}")
                continue

            for station in tqdm(stations, desc=f"Fetching station data for year {year}", leave=False):
                station_file = self.base_dir / str(year) / station
                if station_file.exists():
                    continue
                station_file.parent.mkdir(parents=True, exist_ok=True)

                station_url = f"{self.file_base_url}/{year}/{station}"
                station_response = requests.get(station_url)
                if station_response.status_code != 200:
                    print(f"Failed to fetch station file {station} for year {year}: {station_response.status_code}. Error message: {station_response.text}")
                    continue

                with open(station_file, "w+b") as f:
                    f.write(station_response.content)
    
    def parse_year_available_stations(self, reponse: Response) -> list[str]:
        PATTERN = r"<a\s+[^>]*href=[\"'][^\"']*/([^/]+\.nc)[\"'][^>]*>"

        return regex.findall(PATTERN, reponse.text, regex.IGNORECASE)

if __name__ == "__main__":
    config = owtp.config.load_yaml_config()

    fetcher = HourlyWeatherDataFetcher()

    fetcher.fetch_weather_data()

    print(f"Fetched data to {config['paths']['raw_data']}/weather/hourly")