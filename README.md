# WindFolio: A Mean-Variance Approach To Optimal Wind Turbine Placement

Mean-variance approach to optimal wind turbine site selection and portfolio backtesting in France using large-scale weather data and EPEX spot electricity prices.

## Demo Run

As a demo run, we want to

1. Reduce the amount of data to process
2. Avoid relying on the ERA5-Land API (bottleneck)

Download the [roughness](https://drive.switch.ch/index.php/s/cE58Ul3hUnZhLw6?path=%2Fprocessed%2Froughness%2Fera5%2Fhourly) data and place it in `data/processed/roughness/era5/hourly/`.

Then, download a subset (one 'lat_bin=X/lon_bin=Y' folder) of the [pre-processed weather data](https://drive.switch.ch/index.php/s/cE58Ul3hUnZhLw6?path=%2Fprocessed%2Fparquet%2Fweather%2Fera5_land%2Fhourly) and place it in `data/processed/parquet/weather/era5_land/hourly/` (for example, `data/processed/parquet/weather/era5_land/hourly/lat_bin=42.0/lon_bin=2.0/` should contain several `part.*.parquet` files).

Finally, set the `demo` variable to `true` in `run.sh`, and run:

```bash
chmod +x run.sh # (if not already executable)

./run.sh
```

#### Troubleshooting: 
If you get the following error: 
````
ModuleNotFoundError: No module named 'owtp'
````

Just delete `.venv` folder and re-run `./run.sh`

## Quick Start

Configure parameters in `config/config.yaml` and run:

```bash
./run.sh
```

This executes the complete pipeline: data fetching, preprocessing, energy computation, revenue calculation, mean-variance optimization, backtesting, and visualization generation. Results are saved to `data/processed/` and `reports/figures/`.

## Data

This project processes large-scale spatiotemporal datasets:

- **Weather data**: Hourly wind speed from [ERA5 reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) (Hersbach et al., 2020) and [Aeris meteorological stations](https://en.aeris-data.fr/) across France
- **Temporal range**: 2005-2024 (175,320 hourly observations per location)
- **Spatial coverage**: Grid-based locations covering French territory
- **Electricity prices**: Hourly day-ahead prices from [EPEX Spot France](https://github.com/ewoken/epex-spot-data) and [Ember Climate](https://ember-energy.org/data/european-wholesale-electricity-price-data/)

### Data Access

**Pre-processed data and full report**: [Data on Switch Drive](https://drive.switch.ch/index.php/s/cE58Ul3hUnZhLw6)

**For raw data collection**, API credentials are required:

- **ERA5 weather data**: Requires ECMWF CDS API key

  - Register at [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
  - Navigate to your account’s API Key section.
  - ⁠Locate the «  To easily use your API key, you can configure it inside your .cdsapirc file » text, and copy the text block that is right below.
  - Run the following commands in your terminal:
  - ⁠Create a file named « .ecmwfdatastoresrc » at the root and paste the copied contents inside it.

    This process will allow the ⁠ecmwf ⁠library to access the API on your behalf and automatically download the weather data.

## Project Structure

```
├── config/
│   └── config.yaml           # Configuration: paths, time windows, optimization parameters
├── data/
│   ├── raw/                  # Downloaded weather and price raw data
│   ├── intermediate/         # Intermediate files during processing
│   ├── processed/            # Final processed datasets
│   └── masks/                # France land mask for filtering locations
├── notebooks/                # Data exploration notebooks (legacy)
├── reports/figures/          # Generated plots and visualizations
├── src/owtp/
│   ├── data/
│   │   └── fetching/         # Data fetching scripts
│   ├── exploration/          # Scripts for dev stage data analysis (legacy)
│   ├── others/               # Utilities scripts
│   ├── pipelines/            # Processing steps (see Pipeline Details)
│   ├── visualisations/       # Visualisation scripts
│   └── config.py             # Configuration loader
└── run.sh                    # Main pipeline execution script
```

## Pipeline Details

The `run.sh` script executes these steps sequentially:

1. **Data fetching** (`src/owtp/data/fetching/`)

   - Electricity prices from EPEX Spot GitHub repository and Ember
   - Weather data from ERA5 and Aeris
   - Surface roughness from ERA5

2. **Preprocessing** (`src/owtp/pipelines/`)

   - `merge_elec_prices.py`: Consolidate price data
   - `weather_data_preprocessing.py`: Clean and standardize weather data
   - `wind_height_adjustment.py`: Adjust wind speed to turbine hub height using logarithmic profile
   - `energy_computing.py`: Convert wind speed to power output using turbine power curves

3. **Revenue and optimization** (`src/owtp/pipelines/`)

   - `revenues_computing.py`: Calculate hourly revenues
   - `covariance_matrix_computing.py`: Compute revenue covariance across locations
   - `covariance_matrix_pivot.py`: Reshape covariance matrix for optimization
   - `mean_revenue_computing.py`: Calculate mean revenue vectors
   - `mean_variance_optimization.py`: Solve mean-variance portfolio optimization using CVXPY

4. **Backtesting and visualization**
   - `random_portfolio_allocation.py`: Generate baseline comparison
   - `visualise_turbine_allocation.py`: Create allocation maps and performance plots
   - `visualise_windows_revenues.py`: Plot revenue time series for rolling windows
   - `visualise_cumulative_revenues.py`: Plot cumulative revenues over time

## Configuration

Key parameters in `config/config.yaml`:

- Optimization problem parameters:

  ```yaml
  rolling_calibrations:
    window_size: 1825 # Training window (days)
    step_size: 365 # Rolling step (days)
    eval_size: 365 # Test window (days)

  mean_variance_optimization:
    total_turbines: 100 # Total number of turbines to allocate
    lambda_values: [0.0, 0.00001, 0.0001, ...] # Risk aversion parameters for mean-variance optimisation
  ```

- Resource management parameters for Dask parallel operations (adjust based on computational resources):
  ```yaml
  clustering:
    n_workers: 4 # Number of parallel workers
    threads_per_worker: 1 # Threads per worker
    memory_limit: 4GB # Memory limit per worker
  ```

## Environment Setup

This project uses UV for dependency management:

All dependencies are installed automatically using uv when the `./run.sh` script is ran.

Instructions for Windows OS execution:

```sh
# MacOS/ Linux:
# Comment this line:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
# Uncomment this line:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Re-using This Code

**Different time periods**: Adjust `rolling_calibrations` parameters in `config/config.yaml`.

**Different turbine models**: The `energy_computing.py` pipeline uses the `turbine-models` library. Change turbine specifications there.

## Authors

This project was developed by Nathan Gromb, Christopher Soriano, and Nicolas Teissier as part of FIN-525 course at EPFL (2025-2026).
