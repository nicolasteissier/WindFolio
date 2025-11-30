import owtp.config
import requests
import zipfile
import regex
from requests import Response
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class RteEnergyDataFetcher:
    """
    Fetch hourly energy data from RTE Eco2Mix.

    Currently not implemented.
    TODO: Implement data fetching logic if interested in RTE energy data cf. docu: https://assets.rte-france.com/prod/public/2025-06/Eco2mix%20-%20Spécifications%20des%20fichiers%20en%20puissance_0.pdf
    """
    def __init__(self):
        self.config = owtp.config.load_yaml_config()

        self.base_dir = Path(self.config['paths']['raw_data']) / "energy"

        self.base_url = "https://www.rte-france.com/en/data-publications/eco2mix/download-indicators"
        self.download_base = "https://eco2mix.rte-france.com"

        self.year_range = range(2012, 2023)


    def fetch_rte_energy_data(self):
        # TODO
        return


if __name__ == "__main__":
    config = owtp.config.load_yaml_config()

    fetcher = RteEnergyDataFetcher()

    fetcher.fetch_rte_energy_data()

    print(f"ERROR: Energy data fetcher is not yet implemented.")