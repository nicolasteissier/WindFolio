import owtp.config
from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl
from turbine_models.parser import Turbines
from turbine_models.tools.extract_power_curve import extract_power_curve
from turbine_models.tools.power_curve_tools import plot_power_curve
from turbine_models.tools.library_tools import check_turbine_library_for_turbine
from tqdm import tqdm



class EnergyComputing:
    """
    Compute energy from weather data.
    """

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_dir = Path(self.config['paths']['processed_data']) / "parquet" / "weather" / "era5" / str(freq)
        self.output_dir = Path(self.config['paths']['processed_data']) / "parquet" / "energy" / "era5" / str(freq)

    def compute_energy(self, turbine_model="GE_1.5MW", verbose=True):
        """Compute energy from wind speed data using the specified turbine model."""
        
        station_files = list(self.input_dir.glob("*.parquet.gz"))
        
        if verbose:
            print(f"Computing energy using turbine model: {turbine_model}")
            print(f"Found {len(station_files)} stations to process\n")
        
        iterator = tqdm(station_files, desc="Processing stations", disable=not verbose)

        sum_of_nans = 0
        max_number_of_nan = 0
        
        for weather_file in iterator:
            station_id = weather_file.stem
            
            # Update progress bar with current station
            if verbose:
                iterator.set_postfix({"station": station_id})
            
            df_weather = pl.read_parquet(weather_file)
            wind_speed = np.sqrt(df_weather["u10"].to_numpy()**2 + df_weather["u10"].to_numpy()**2)


        

            mwh = self.windspeed_to_MWh(wind_speed, turbine_model=turbine_model)
            
            df_energy = pl.DataFrame({
                "time": df_weather["valid_time"],
                "latitude": df_weather["latitude"],
                "longitude": df_weather["longitude"],
                "mwh": mwh
            })

            
            output_path = self.output_dir / f"{station_id}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_energy.write_parquet(output_path, compression='gzip')
        
        # if verbose:
        #     print("\nAll stations processed successfully")


    def get_power_curve(self, turbine_model="GE_1.5MW"):
        """Get the power curve for a specified turbine model."""

        t_lib = Turbines()

        if not check_turbine_library_for_turbine(turbine_model):
            raise ValueError(f"Turbine model '{turbine_model}' not found in the turbine library.")
    
        turb_group = t_lib.find_group_for_turbine(turbine_model)
        turbine_specs = t_lib.specs(turbine_model, group = turb_group)

        power_curve = extract_power_curve(turbine_specs)
        power_curve = pl.DataFrame({
            "wind_speed": power_curve["wind_speed"].tolist(),
            "power_curve_kw": power_curve["power_curve_kw"].tolist(),
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

    def windspeed_to_MWh(self, wind_speed_df, turbine_model=None):

        # first load the power curve
        power_curve = self.get_power_curve(turbine_model)

        ws_curve = power_curve["wind_speed"].to_numpy()
        p_curve = power_curve["power_curve_kw"].to_numpy()
        
        # Divide by 1000 to get MWh instead of KWh
        return self.interp_power(wind_speed_df, ws_curve, p_curve)/1000
    
if __name__ == "__main__":
    computer = EnergyComputing(freq='hourly')
    computer.compute_energy(turbine_model="GE_1.5MW")