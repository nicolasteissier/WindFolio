import polars as pl
import numpy as np
from pathlib import Path
import owtp.config

from turbine_models.parser import Turbines
from turbine_models.tools.extract_power_curve import extract_power_curve
from turbine_models.tools.power_curve_tools import plot_power_curve
from turbine_models.tools.library_tools import check_turbine_library_for_turbine
import turbine_models

def get_power_curve(turbine_model="GE_1.5MW"):

    turbine = "GE_1.5MW"
    t_lib = Turbines()

    is_valid = check_turbine_library_for_turbine(turbine)

    turb_group = t_lib.find_group_for_turbine(turbine)
    turbine_specs = t_lib.specs(turbine,group = turb_group)

    power_curve = extract_power_curve(turbine_specs)
    power_curve = pl.DataFrame({
        "wind_speed": power_curve["wind_speed"].tolist(),
        "power_curve_kw": power_curve["power_curve_kw"].tolist(),
    })
    return power_curve

def interp_power(ws_values: np.ndarray, ws_curve, p_curve) -> np.ndarray:
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

def windspeed_to_kWh(wind_speed_df,turbine_model=None):
    # first load the power curve
    power_curve = get_power_curve(turbine_model)

    ws_curve = power_curve["wind_speed"].to_numpy()
    p_curve = power_curve["power_curve_kw"].to_numpy()
    
    return interp_power(wind_speed_df, ws_curve, p_curve)


# Example of how to use the function 
print("In test mode (get_power.py)")
# test wind_speed dataframe
df_ws = pl.DataFrame({
    "timestamp": [
        "2024-01-01 00:00:00",
        "2024-01-01 00:10:00",
        "2024-01-01 00:20:00",
        "2024-01-01 00:30:00",
        "2024-01-01 00:40:00",
        "2024-01-01 00:50:00",
        "2024-01-01 01:00:00",
    ],
    "wind_speed": [
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
    ]
})
print(windspeed_to_kWh(df_ws["wind_speed"]))
