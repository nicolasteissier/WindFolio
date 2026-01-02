import owtp.config
from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl
from turbine_models.parser import Turbines
from turbine_models.tools.extract_power_curve import extract_power_curve
from turbine_models.tools.power_curve_tools import plot_power_curve
from turbine_models.tools.library_tools import check_turbine_library_for_turbine
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import os



class EnergyComputing:
    """
    Compute energy from weather data.
    """

    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config[target]['processed_data']) / "parquet" / "weather" / "era5_land" / "hourly"
        self.output_dir = Path(self.config[target]['processed_data']) / "parquet" / "energy" / "era5_land" / "hourly"

    def compute_energy(self, turbine_model="NREL_7MW", n_workers=None, verbose=True):
        """Compute energy from wind speed data using the specified turbine model."""
        
        if n_workers is None:
            n_workers = os.cpu_count() // 2 # type: ignore
        
        # Distributed scheduler for performance monitoring and memory management
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=4,
            memory_limit='20GB',
            processes=True,
            dashboard_address=':8787'
        )
        client = Client(cluster)
        
        if verbose:
            print(f"Dask cluster initialized with {n_workers} workers")
            print(f"Dashboard: {client.dashboard_link}")
        
        try:
            ddf = dd.read_parquet(self.input_dir, engine="pyarrow")

            if verbose:
                print(f"Computing energy using turbine model: {turbine_model}")
        
            wind_speed = np.sqrt(np.pow(ddf['u10'], 2) + np.pow(ddf['v10'], 2))

            mwh = self.windspeed_to_MWh(wind_speed, turbine_model=turbine_model)
            
            ddf_energy = ddf[["lat_bin", "lon_bin", "valid_time", "latitude", "longitude"]].assign(mwh=mwh)

            ddf_energy.to_parquet(
                self.output_dir,
                engine="pyarrow",
                compression="zstd",
                write_index=False,
                partition_on=["lat_bin", "lon_bin"],
            )
        
            if verbose:
                print("Done writing Parquet files.")
        finally:
            client.close()
            cluster.close()

    def get_power_curve(self, turbine_model="GE_1.5MW"):
        """Get the power curve for a specified turbine model."""

        t_lib = Turbines()

        if not check_turbine_library_for_turbine(turbine_model):
            raise ValueError(f"Turbine model '{turbine_model}' not found in the turbine library.")
    
        turb_group = t_lib.find_group_for_turbine(turbine_model)
        turbine_specs = t_lib.specs(turbine_model, group = turb_group) # type: ignore

        power_curve = extract_power_curve(turbine_specs) # type: ignore
        power_curve = pl.DataFrame({
            "wind_speed": power_curve["wind_speed"].tolist(), # type: ignore
            "power_curve_kw": power_curve["power_curve_kw"].tolist(), # type: ignore
        })
        return power_curve

    def interp_power(self, ws_values: np.ndarray, ws_curve, p_curve) -> np.ndarray:
        """
        Linearly interpolate power from wind speed.
        Extrapolation:
        - below min ws -> 0
        - above max ws -> 0 (or last value)
        """
        # Use np.interp with 0 outside domain
        # np.interp(x, xp, fp) returns fp[0] for x<xp[0] and fp[-1] for x>xp[-1]
        # We'll override >max to 0 after.
        p = np.interp(ws_values, ws_curve, p_curve)
        # Set power to 0 below cut-in (already mostly 0 from curve) and above cut-out
        cut_in = ws_curve.min()
        cut_out = ws_curve.max()
        p = np.where((ws_values < cut_in) | (ws_values > cut_out), 0.0, p)
        return p

    def windspeed_to_MWh(self, wind_speed_df, turbine_model):

        # first load the power curve
        power_curve = self.get_power_curve(turbine_model)

        ws_curve = power_curve["wind_speed"].to_numpy()
        p_curve = power_curve["power_curve_kw"].to_numpy()
        
        def process_partition(ws_series):
            return self.interp_power(ws_series.values, ws_curve, p_curve) / 1000
        
        return wind_speed_df.map_partitions(process_partition, meta=('mwh', 'f8'))
    
if __name__ == "__main__":
    computer = EnergyComputing(target="paths_local")
    computer.compute_energy(turbine_model="NREL_7MW", verbose=True)