import owtp.config
from pathlib import Path
import xarray as xr
from ecmwf.datastores import Client
from typing import Literal
import geopandas as gpd
import pandas as pd
import numpy as np
import regionmask

class ERA5RoughnessDataFetcher:
    """Fetcher for ERA5 roughness data using CDS API"""

    france = [51.1, -5.2, 41.3, 9.6],  # North, West, South, East
    
    def __init__(self, target: Literal["paths", "paths_local"]):
        self.config = owtp.config.load_yaml_config()
        self.roughness_base_dir = Path(self.config[target]['raw_data']) / "roughness" / "era5" / "hourly"
        self.roughness_intermediate_dir = Path(self.config[target]['intermediate_data']) / "roughness" / "era5" / "hourly"
        self.roughness_processed_dir = Path(self.config[target]['processed_data']) / "roughness" / "era5" / "hourly"
        self.era5land_base_dir = Path(self.config[target]['raw_data']) / "weather" / "era5_land" / "hourly"
        
        self.france_mask_path = Path(self.config[target]['masks']) / "france_land_mask.nc"
        self.france_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.france_mask_path.exists():
            print("Creating France land mask...")
            self._create_france_mask(
                output_path=self.france_mask_path,
            )
            print(f"France land mask saved to {self.france_mask_path}")


        self.client = Client(sleep_max=30)

    def _get_era5land_file(self) -> Path:
        """Get list of all raw ERA5-Land .nc files"""
        return sorted(self.era5land_base_dir.glob("*.nc"))[0]
    
    def _get_roughness_file(self, stage: Literal["raw", "intermediate", "processed"] = "raw") -> Path:
        """Get list of all raw ERA5 roughness .nc files"""
        if stage == "raw":
            base_dir = self.roughness_base_dir
        elif stage == "intermediate":
            base_dir = self.roughness_intermediate_dir
        elif stage == "processed":
            base_dir = self.roughness_processed_dir
        else:
            raise ValueError("stage must be one of 'raw', 'intermediate', or 'processed'.")
        
        return sorted(base_dir.glob("*.nc" if stage == "raw" else "*.parquet"))[0]

    def fetch_roughness(
            self, 
            year: str, 
            month: str,
            verbose: bool = False):
        """Batch download ERA5 and ERA5-Land data from CDS API"""
        if not isinstance(year, str) or not isinstance(month, str):
            raise ValueError("Year and month must be strings.")
        
        if not month.isdigit() or not (1 <= int(month) <= 12) or len(month) != 2:
            raise ValueError("Month must be a string representing a number between 01 and 12.")
        
        active_dir = self.roughness_base_dir
        active_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = 'reanalysis-era5-single-levels'

        variables = [
            'forecast_surface_roughness'
        ]
        
        days = [f'{d:02d}' for d in range(1, 32)]
        times = [f'{h:02d}:00' for h in range(0, 24)]
        
        filename = active_dir / f'roughness_{year}_{month}.nc'
        
        if filename.exists():
            if verbose:
                print(f"Skipping {filename} (already exists)")
            return
            
        if verbose:
            print(f"Queueing ERA5 {year} {month} download...")
        
        self.client.retrieve(
            dataset_id,
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'download_format': 'zip',
                'variable': variables,
                'year': year,
                'month': month,
                'day': days,
                'time': times,
                'area': self.france
            },
            str(filename)
        )
        if verbose:
            print(f"Queued: {filename}")

    def interpolate_roughness(self) -> None:
        """Interpolate roughness data onto regular grid and save to intermediate directory"""
        print("Loading reference ERA5-Land dataset for interpolation grid...")
        reference_path = self._get_era5land_file()
        ds_ref = xr.open_dataset(reference_path, engine="h5netcdf") 
        input_files = self._get_roughness_file("raw")
        self._interpolate_on(input_files, ds_ref)
        print("Interpolation complete.")

    def _interpolate_on(self, input_path: Path, ds_ref: xr.Dataset) -> None:
        """Interpolate roughness dataset onto reference grid and save as Parquet"""
        output_path = self.roughness_intermediate_dir / "roughness.parquet"
        if output_path.exists():
            print("Interpolated file already exists, skipping.")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ds = xr.open_dataset(input_path, engine="h5netcdf")
        ds_interp = ds.interp(
            longitude=ds_ref["longitude"],
            latitude=ds_ref["latitude"],
            method="nearest",
        )

        roughness_variance = ds_interp["fsr"].var(dim="valid_time").mean().item()
        print(f"Average roughness variance over time: {roughness_variance:.6f}")
        
        ds_interp = ds_interp.mean(dim="valid_time")

        df = ds_interp.to_dataframe().reset_index()
        
        vars_to_keep = ["fsr"]
        keep_cols = ["latitude", "longitude"] + vars_to_keep
        
        df = df[keep_cols].dropna(subset=vars_to_keep, how="all", axis=0)

        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        
    def mask_roughness(self) -> None:
        """Apply France land mask to all raw ERA5-Land .nc files and save to intermediate directory"""
        print("Loading France land mask...")
        mask_ds = self._load_france_mask()

        print("Applying France land mask to raw ERA5-Land files...")
        input_path = self._get_roughness_file("intermediate")

        try:
            output_path = self.roughness_processed_dir / "roughness.parquet"
            if output_path.exists():
                print("Masked file already exists, skipping.")
                return
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = self._apply_france_mask(input_path, mask_ds)
            
            vars_to_keep = ["fsr"]
            keep_cols = ["latitude", "longitude"] + vars_to_keep
            
            df = df[keep_cols].dropna(subset=vars_to_keep, how="all", axis=0)

            df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        except Exception as e:
            print(f"\nFailed to apply mask on {input_path}: {e}")

        print("Masking complete.")

    def _create_france_mask(self, output_path: Path) -> None:
        # Source "https://simplemaps.com/static/svg/country/fr/all/fr.json"
        json_path = output_path.parent / "fr.json"
        mask_geodf = gpd.read_file(json_path)

        france_shape = regionmask.from_geopandas(mask_geodf)

        reference_path = self._get_era5land_file()
        ds_ref = xr.open_dataset(reference_path, engine="h5netcdf")
        lons = ds_ref["longitude"]
        lats = ds_ref["latitude"]

        mask_array = france_shape.mask(lons, lats)  

        fr_idx = france_shape.numbers[0]
        inside_france = (mask_array == fr_idx)

        mask_ds = xr.Dataset(
            data_vars={"mask_france": (("latitude", "longitude"), inside_france.data)},
            coords={"latitude": lats, "longitude": lons},
        )
        mask_ds.to_netcdf(output_path, engine="h5netcdf")


    def _load_france_mask(self) -> xr.Dataset:
        """Load the France land mask dataset"""
        if not self.france_mask_path.exists():
            self._create_france_mask(
                output_path=self.france_mask_path,
            )

        return xr.open_dataset(self.france_mask_path)

    def _apply_france_mask(self, input_path: Path, mask_ds: xr.Dataset) -> pd.DataFrame:
        df = pd.read_parquet(input_path, engine="pyarrow")

        mask = mask_ds["mask_france"].to_dataframe()

        return df.merge(
            mask.reset_index(),
            on=["latitude", "longitude"],
            how="left",
        ).loc[lambda x: x["mask_france"] == True].drop(columns=["mask_france"])
    
    def plot_roughness(self) -> None:
        """Plot roughness data for visual inspection"""
        import matplotlib.pyplot as plt

        raw = self._get_roughness_file("raw")
        intermediate = self._get_roughness_file("intermediate")
        processed = self._get_roughness_file("processed")

        ds_raw = xr.open_dataset(raw, engine="h5netcdf")
        df_intermediate = pd.read_parquet(intermediate, engine="pyarrow")
        df_processed = pd.read_parquet(processed, engine="pyarrow")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        print('Raw data dimensions:', ds_raw.dims)
        ds_raw["fsr"].isel(valid_time=0).plot(ax=axes[0], cmap="viridis")
        df_intermediate_pivot = df_intermediate.pivot_table(index="latitude", columns="longitude", values="fsr")
        im1 = axes[1].imshow(df_intermediate_pivot, origin="lower", cmap="viridis",
                             extent=[df_intermediate["longitude"].min(), df_intermediate["longitude"].max(),
                                     df_intermediate["latitude"].min(), df_intermediate["latitude"].max()])
        axes[1].set_title("Interpolated Roughness")
        fig.colorbar(im1, ax=axes[1], orientation='vertical', label='fsr')
        df_processed_pivot = df_processed.pivot_table(index="latitude", columns="longitude", values="fsr")
        im2 = axes[2].imshow(df_processed_pivot, origin="lower", cmap="viridis",
                             extent=[df_processed["longitude"].min(), df_processed["longitude"].max(),
                                     df_processed["latitude"].min(), df_processed["latitude"].max()])
        axes[2].set_title("Masked Roughness")
        fig.colorbar(im2, ax=axes[2], orientation='vertical', label='fsr')
        for ax in axes:
            mid_lat = (df_processed["latitude"].min() + df_processed["latitude"].max()) / 2
            ax.set_aspect(1.0 / np.cos(np.radians(mid_lat)))
        plt.tight_layout()
        plt.savefig("reports/figures/era5/roughness_plot.png")

        print('Resulting dimensions:')
        print(f"Masked data shape: {df_processed.shape}")
        print(f"Number of unique latitudes: {df_processed['latitude'].nunique()}")
        print(f"Number of unique longitudes: {df_processed['longitude'].nunique()}")
        print(df_processed.head())

if __name__ == "__main__":
    fetcher = ERA5RoughnessDataFetcher(target="paths_local")
    fetcher.fetch_roughness(
        year='2005',
        month='01',
        verbose=True
    )
    fetcher.interpolate_roughness()
    fetcher.mask_roughness()
    fetcher.plot_roughness()