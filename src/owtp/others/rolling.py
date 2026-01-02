import pandas as pd

from owtp.config import load_yaml_config

def get_windows(start, end):
    """Generate sliding windows of the DataFrame based on the configured parameters."""
    config = load_yaml_config()

    window_size = config['rolling_calibrations']['window_size']
    step_size = config['rolling_calibrations']['step_size']
    gap_size = config['rolling_calibrations']['gap_size']
    eval_size = config['rolling_calibrations']['eval_size']

    final_window_size = config['rolling_calibrations']['final_window_size']
    final_eval_size = config['rolling_calibrations']['final_eval_size']

    windows = []
    current_start = start
    while current_start + pd.Timedelta(days=window_size + gap_size + eval_size) <= end:
        train_start = current_start
        train_end = current_start + pd.Timedelta(days=window_size) - pd.Timedelta(seconds=1)
        eval_start = train_end + pd.Timedelta(seconds=1) + pd.Timedelta(days=gap_size)
        eval_end = eval_start + pd.Timedelta(days=eval_size) - pd.Timedelta(seconds=1)

        windows.append({
            "train_window_start": train_start,
            "train_window_end": train_end,
            "eval_window_start": eval_start,
            "eval_window_end": eval_end,
        })

        current_start += pd.Timedelta(days=step_size)
    
    # Add final window (whole time range)
    final_train_start = start
    final_train_end = final_train_start + pd.Timedelta(days=final_window_size) - pd.Timedelta(seconds=1)
    final_eval_start = final_train_end + pd.Timedelta(seconds=1) + pd.Timedelta(days=gap_size)
    final_eval_end = final_eval_start + pd.Timedelta(days=final_eval_size) - pd.Timedelta(seconds=1)

    windows.append({
        "train_window_start": final_train_start,
        "train_window_end": final_train_end,
        "eval_window_start": final_eval_start,
        "eval_window_end": final_eval_end,
    })

    print(f"Generated {len(windows)} windows for rolling computations.")

    return windows

def format_window_str(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format window start and end times into a string."""
    return f"{start.strftime('%Y%m%d-%H%M%S')}_{end.strftime('%Y%m%d-%H%M%S')}"