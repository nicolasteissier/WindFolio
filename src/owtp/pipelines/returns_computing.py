import owtp.config
from pathlib import Path
from typing import Literal
import pandas as pd
from tqdm import tqdm


class ReturnsComputing:
    """
    Compute returns from energy and prices data.
    """

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        self.input_energy_dir = Path(self.config['paths_local']['intermediate_data']) / "parquet" / "energy" / str(freq)
        self.input_prices_dir = Path(self.config['paths_local']['intermediate_data']) / "parquet" / "prices" / str(freq)
        self.output_dir = Path(self.config['paths_local']['intermediate_data']) / "parquet" / "returns" / str(freq)

    def compute_returns(self, verbose=True):
        """Compute returns from energy and prices data"""
        
        # Load the global prices data
        if verbose:
            print("Loading prices data...")
        df_prices = self.load_prices_data(self.input_prices_dir / "prices.parquet")
        print(f"Prices data loaded with {len(df_prices)} entries.")
        
        # Process each station's energy data
        station_files = list(self.input_energy_dir.glob("*.parquet"))
        
        iterator = tqdm(station_files, desc="Computing returns for all stations") if verbose else station_files
        
        for energy_file in iterator:
            station_id = energy_file.stem
            df_energy = self.load_energy_data(energy_file)
            df_returns = self.compute_station_returns(df_energy, df_prices)
            
            output_path = self.output_dir / f"{station_id}_returns.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_returns.to_frame().to_parquet(output_path)

    def compute_station_returns(self, df_energy, df_prices):
        """
        Computes the returns based on energy generation and electricity prices.

        Parameters:
        df_energy (pd.DataFrame): DataFrame containing energy data with a datetime index.
                                Expected format: index as datetime, column 'mwh'.
        df_prices (pd.DataFrame): DataFrame containing prices data with a datetime index.
                                Expected format: index as datetime, column 'Price (EUR/MWhe)'.
        Returns:
        pd.Series: Series containing the computed returns indexed by time.
        """
        df_returns = df_energy['mwh'] * df_prices['Price (EUR/MWhe)']
        return df_returns

    def load_prices_data(self, filepath: Path):
        """Load prices data from the prices Parquet file"""
        return pd.read_parquet(filepath)
    
    def load_energy_data(self, filepath: Path):
        """Load energy data for a specific station from the corresponding energy Parquet file"""
        return pd.read_parquet(filepath)
    
if __name__ == "__main__":
    computer = ReturnsComputing(freq='hourly')
    computer.compute_returns(verbose=True)






