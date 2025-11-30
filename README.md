# Optimal Wind Turbine Placement

A Financial Big Data course project at EPFL focused on optimizing wind turbine placement using publicly available data from France. This project aims to establish a big-data-driven strategy for optimal wind turbine site selection.

## Project Structure

```
├── data/
│   ├── raw/          # Raw data collected from various sources
│   └── processed/    # Processed and cleaned data
├── notebooks/        # Jupyter notebooks for analysis
├── reports/          # Project reports and documentation
│   └── figures/      # Generated visualizations and plots          
├── src/
│   └── owtp/         # Main package for Optimal Wind Turbine Placement project
│       ├── data/
│       │   ├── fetching/    # Data collection scripts
│       │   └── loading/     # Data loading utilities
└──     └── pipelines/       # Data processing pipelines
```

## Environment Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management. Make sure you have UV installed:

```bash
brew install uv
```

To set-up / update the environment, run:

```bash
uv sync
```

## Data Collection

To collect all the data used for this project, run:

```bash
uv run src/owtp/data/fetching/data_collection.py
```

The data will be collected and stored in the `data/raw` folder.