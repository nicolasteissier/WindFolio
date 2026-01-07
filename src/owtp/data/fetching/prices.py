import owtp.config
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
from typing import Literal

class EpexSpotRepoHourlyDataFetcher:
    """
    Fetch EPEX spot price data from the GitHub repository.
    """
    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()

        self.base_dir = Path(self.config[target]["raw_data"]) / "elec"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.repo_api_url = (
            "https://api.github.com/repos/ewoken/epex-spot-data/contents/historicData"
        )
        self.raw_base_url = (
            "https://raw.githubusercontent.com/ewoken/epex-spot-data/master/historicData"
        )

    def fetch_epexspot_data_from_repo(self):
        try :
            response = requests.get(self.repo_api_url)
            if response.status_code != 200:
                print(
                    f"Failed to list repo contents: {response.status_code} {response.text}"
                )
                return
            
            print("Fetching list of EPEX spot data files...")

            entries = response.json()
            json_files = [
                e["name"]
                for e in entries
                if e.get("type") == "file" and e["name"].endswith(".json")
            ]

            for fname in tqdm(json_files, desc="Fetching EPEX spot data from GitHub repository"):
                year = fname.split(".")[0]
                url = f"{self.raw_base_url}/{fname}"

                resp = requests.get(url)
                if resp.status_code != 200:
                    print(f"Failed to fetch {fname}: {resp.status_code} {resp.text}")
                    continue

                out_file = self.base_dir / f"epexspot_{year}_repo.json"
                out_file.write_text(resp.text, encoding="utf-8")
        except Exception as e:
            print(f"An error occurred while fetching EPEX spot data from GitHub repository: {e}")

class EpexSpotEmberDataFetcher:
    """
    Fetch EPEX spot price data from Ember dataset.
    """
    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()

        self.base_dir = Path(self.config[target]["raw_data"]) / "elec"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.ember_file_url = "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/price/outputs/european_wholesale_electricity_price_data_hourly.zip"

        self.countries = ["France"]

    def fetch_epexspot_data_from_ember(self):
        try:
            response = requests.get(self.ember_file_url)
            if response.status_code != 200:
                print(
                    f"Failed to fetch Ember EPEX spot data: {response.status_code} {response.text}"
                )
                return

            out_file = self.base_dir / "epexspot_ember_hourly.zip"
            with open(out_file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded Ember EPEX spot data to {out_file}")
        except Exception as e:
            print(f"An error occurred while fetching Ember EPEX spot data: {e}")

    def unzip_ember_data_file(self):
        zip_path = self.base_dir / "epexspot_ember_hourly.zip"
        if not zip_path.exists():
            print(f"Ember EPEX spot data zip file does not exist: {zip_path}")
            return

        for country in self.countries:
            country_file = country + ".csv"

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(country_file, path=self.base_dir)
            
            extracted_file = self.base_dir / country_file
            renamed_file = self.base_dir / f"epexspot_ember_{country.lower()}.csv"
            extracted_file.rename(renamed_file)
            print(f"Extracted and renamed {country_file} to {renamed_file}")

if __name__ == "__main__":
    # Fetch EPEX spot data from GitHub repository
    repo_fetcher = EpexSpotRepoHourlyDataFetcher(target="paths_local")
    print("Fetching EPEX spot data from GitHub repository...")
    repo_fetcher.fetch_epexspot_data_from_repo()
    print("Fetching EPEX spot data from Ember...")

    # Fetch EPEX spot data from Ember
    ember_fetcher = EpexSpotEmberDataFetcher(target="paths_local")
    ember_fetcher.fetch_epexspot_data_from_ember()
    ember_fetcher.unzip_ember_data_file()