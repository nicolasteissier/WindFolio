import pandas as pd

from owtp.config import load_yaml_config

def get_windows(df: pd.DataFrame):
    """Generate sliding windows of the DataFrame based on the configured parameters."""
    config = load_yaml_config()

    window_size = config['rolling_calibrations']['window_size']
    step_size = config['rolling_calibrations']['step_size']
    gap_size = config['rolling_calibrations']['gap_size']
    eval_size = config['rolling_calibrations']['eval_size']

    start = df.index.min()
    end = df.index.max()

    windows = []
    current_start = start
    while current_start + pd.Timedelta(days=window_size + gap_size + eval_size) <= end:
        train_start = current_start
        train_end = current_start + pd.Timedelta(days=window_size) - pd.Timedelta(seconds=1)
        eval_start = train_end + pd.Timedelta(seconds=1) + pd.Timedelta(days=gap_size)
        eval_end = eval_start + pd.Timedelta(days=eval_size) - pd.Timedelta(seconds=1)

        train_window = df.loc[train_start:train_end]
        eval_window = df.loc[eval_start:eval_end]

        windows.append((train_window, eval_window))

        current_start += pd.Timedelta(days=step_size)

    return windows