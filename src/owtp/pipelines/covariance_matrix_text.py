import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
from tqdm import tqdm
import os
import polars as pl

class covariance_matrix:
    def __init__(self, target: Literal["paths", "paths_local"], freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config['paths']['processed_data']) / "parquet" / "returns" / str(freq)

    def covariance(self):  
        
        filenames = sorted(file for file in os.listdir(self.input_dir) if not file.startswith('._'))
        lazy_frames = []

        for filename in filenames:
            p = os.path.join(self.input_dir, filename)
            station_name = Path(p).stem  # e.g. "station_001"
            lf = (
                pl.scan_parquet(p)            # use scan_parquet(...) if you have parquet
                .select(
                    pl.col("time").alias("time"),
                    pl.lit(station_name).alias("station"),
                    pl.col("revenue").alias("rev")
                )
            )
            lazy_frames.append(lf)
        # 1) Stack all (long format): columns: [time, station, rev]
        lf_all = pl.concat(lazy_frames)

        # 2) Collect first, then pivot to wide: one column per station
        df_all = lf_all.collect()
        
        df_wide = (
            df_all
            .with_columns(pl.col("time").str.strptime(pl.Datetime, strict=False))
            .pivot(
                index="time",           # rows
                on="station",      # becomes one column per station
                values="rev",           # cell values
                aggregate_function="first",
            )
            .sort("time")
        )

        df_revenue = df_wide.drop("time")
        print("computing covariance matrix")
        cov_df = df_revenue.cov()
       # Polars DataFrame: station x station
        print(cov_df)



if __name__ == '__main__':
    covariance_matrix(target='paths', freq='hourly').covariance()