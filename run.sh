#!/bin/bash

echo "STARTING FBD PROJECT PIPELINE"

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv sync

set -e

# echo "--------------------------------"
# echo "Fetching prices data..."
# echo "--------------------------------"
# uv run src/owtp/data/fetching/prices.py
# echo ""

# echo "Merging prices data..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/merge_elec_prices.py
# echo ""

# echo "--------------------------------"
# echo "Fetching weather data..."
# echo "--------------------------------"
# uv run src/owtp/data/fetching/weather.py
# echo ""

# echo "--------------------------------"
# echo "Preprocessing data..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/weather_data_preprocessing.py
# echo ""

# echo "--------------------------------"
# echo "Fetching roughness data..."
# echo "--------------------------------"
# uv run src/owtp/data/fetching/roughness.py
# echo ""

# echo "--------------------------------"
# echo "Adjusting wind speed to turbine hub height..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/wind_height_adjustment.py
# echo ""

# echo "--------------------------------"
# echo "Computing energy production..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/energy_computing.py
# echo ""

# echo "--------------------------------"
# echo "Computing revenues..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/revenues_computing.py
# echo ""

# echo "--------------------------------"
# echo "Computing Covariance matrix..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/covariance_matrix_computing.py
# echo ""

# echo "--------------------------------"
# echo "Pivoting Covariance matrix..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/covariance_matrix_pivot.py
# echo ""

# echo "--------------------------------"
# echo "Computing Mean Revenue vector..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/mean_revenue_computing.py
# echo ""

# echo "--------------------------------"
# echo "Running Mean-Variance Optimization..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/mean_variance_optimization.py
# echo ""

# echo "--------------------------------"
# echo "Creating visualisation of mean-variance optimization results..."
# echo "--------------------------------"
# uv run src/owtp/visualisations/portfolio/visualise_turbine_allocation.py
# echo ""

# echo "--------------------------------"
# echo "Computing random allocation portfolio revenues..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/random_portfolio_allocation.py
# echo ""

# echo "--------------------------------"
# echo "Computing portfolio revenues..."
# echo "--------------------------------"
# uv run src/owtp/pipelines/compute_portfolio_revenues.py
# echo ""

echo "--------------------------------"
echo "Ploting portfolio revenues..."
echo "--------------------------------"
uv run src/owtp/visualisations/portfolio/visualise_windows_revenues.py
echo ""

echo "--------------------------------"
echo "Ploting cumulative portfolio revenues..."
echo "--------------------------------"
uv run src/owtp/visualisations/portfolio/visualise_cumulative_revenues.py
echo ""


