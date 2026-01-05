"""
Analyzes the index of price data to identify and print missing hourly dates ranges.

Prints the Timestamp objects that follow each missing date range, as they need to be dropped
because their computed returns will be invalid (based on more than one hour gap).
"""

import pandas as pd

# Replace with the actual path to your prices Parquet file
PATH = '~/Downloads/prices.parquet'

df_prices = pd.read_parquet(PATH)

df_prices_sorted = df_prices.sort_index()
date_range = pd.date_range(
    start=df_prices_sorted.index.min(),
    end=df_prices_sorted.index.max(),
    freq='h'
)
existing_dates = set(df_prices_sorted.index)
expected_dates = set(date_range)
missing_dates = sorted(expected_dates - existing_dates)

# Group contiguous missing dates
grouped_missing_dates = []
if missing_dates:
    current_group = [missing_dates[0]]
    for i in range(1, len(missing_dates)):
        if (missing_dates[i] - missing_dates[i - 1]) == pd.Timedelta(hours=1):
            current_group.append(missing_dates[i])
        else:
            grouped_missing_dates.append(current_group)
            current_group = [missing_dates[i]]
    grouped_missing_dates.append(current_group)

    # Create a more readable format for missing date ranges
    formatted_missing_dates = []
    for group in grouped_missing_dates:
        if len(group) == 1:
            formatted_missing_dates.append(group[0])
        else:
            formatted_missing_dates.append(group[-1])
else:
    formatted_missing_dates = []

print("Missing dates:")
ts1h = pd.Timedelta(hours=1)

print("[")
for date in formatted_missing_dates:
    print(f"\t{(date + ts1h).__repr__()},")
print("]")