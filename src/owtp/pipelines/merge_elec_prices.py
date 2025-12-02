import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import owtp.config
from pathlib import Path
from typing import Literal


class PriceMerger:
    """
    Comment
    """

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config['paths_local']['raw_data']) / "elec"
        self.output_dir = Path(self.config['paths_local']['intermediate_data']) / "parquet" / "prices" / str(freq)

    def merge_prices(self):

        data = {}   # or use a list if you prefer
        files = sorted(os.listdir(self.input_dir))
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(self.input_dir, filename)
                with open(filepath, "r") as f:
                    data[filename] = json.load(f)

        dataframes = []
        for filename in files:
            if filename.endswith(".json"):
                dataframes.append(pd.DataFrame(data[filename]))

        epex_merged = pd.concat(dataframes) 

        epex_merged["startDate"] = pd.to_datetime(
            epex_merged["startDate"].astype(str),  # make sure it's strings
            utc=True,                              # interpret offsets like +02:00
            errors="coerce"                        # invalid -> NaT instead of crash
        )

        epex_merged["time"] = epex_merged["startDate"].dt.tz_convert(None)
        epex_merged = epex_merged.rename(columns={"price_euros_mwh": "Price (EUR/MWhe)"})
        epex_merged = epex_merged.drop(columns=["startDate", "endDate", "volume_mwh"])
        epex_merged = epex_merged.reindex(columns=['time', 'Price (EUR/MWhe)'])
        
        ember_data_path = self.input_dir / "epexspot_ember_france.csv"
        ember_data = pd.read_csv(ember_data_path)
        ember_data = ember_data.drop(columns=["Country", "ISO3 Code", "Datetime (Local)"])
        ember_data = ember_data.rename(columns={"Datetime (UTC)": "time"})
        ember_data['time'] = pd.to_datetime(ember_data['time'])

        merged = (
            pd.concat([epex_merged, ember_data])
            .drop_duplicates(subset="time", keep="first")
            .sort_values("time")
            .reset_index(drop=True)
        ).set_index('time')

        # Save merged data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "prices.parquet"
        merged.to_parquet(output_path)


    def plot_prices(prices):
        plt.figure(figsize=(12, 6))
        plt.plot(prices['time'], prices['Price (EUR/MWhe)'])
        plt.xlabel('time')
        plt.ylabel('Price (EUR/MWhe)')
        plt.title('EPEX Electricity Prices Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    #plot_prices(epex_merged[epex_merged['Date']>= '2015-01-01 00:00:00'])
    #plot_prices(ember_data[ember_data['Date']< '2019-01-01 00:00:00'])
    #plot_prices(merged)
    # 
    # """

if __name__ == "__main__":
    merger = PriceMerger()
    merger.merge_prices()