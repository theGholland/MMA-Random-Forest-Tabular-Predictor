import numpy as np
import pandas as pd

# Columns containing "X of Y" counts
COUNT_COLS = [
    'total_strikes_1', 'total_strikes_2',
    'significant_strikes_1', 'significant_strikes_2',
    'head_strikes_1', 'head_strikes_2',
    'body_strikes_1', 'body_strikes_2',
    'leg_strikes_1', 'leg_strikes_2',
    'distance_strikes_1', 'distance_strikes_2',
    'clinch_strikes_1', 'clinch_strikes_2',
    'ground_strikes_1', 'ground_strikes_2',
    'takedowns_1', 'takedowns_2'
]

# Columns representing times in mm:ss format
TIME_COLS = ['control_time_1', 'control_time_2', 'time']

# Numeric output columns the model predicts
NUMERIC_COLS = [
    'knockdowns_1', 'knockdowns_2',
    'total_strikes_1', 'total_strikes_2',
    'significant_strikes_1', 'significant_strikes_2',
    'head_strikes_1', 'head_strikes_2',
    'body_strikes_1', 'body_strikes_2',
    'leg_strikes_1', 'leg_strikes_2',
    'distance_strikes_1', 'distance_strikes_2',
    'clinch_strikes_1', 'clinch_strikes_2',
    'ground_strikes_1', 'ground_strikes_2',
    'takedowns_1', 'takedowns_2',
    'submission_attempts_1', 'submission_attempts_2',
    'reversals_1', 'reversals_2',
    'control_time_1', 'control_time_2',
    'time'
]

def parse_count(value):
    """Return the landed count from strings like '19 of 31'."""
    try:
        return int(str(value).split(' of ')[0])
    except Exception:
        return np.nan

def parse_time(ts):
    """Convert mm:ss strings to total seconds."""
    try:
        m, s = map(int, str(ts).split(':'))
        return m * 60 + s
    except Exception:
        return np.nan

def load_data(path: str) -> pd.DataFrame:
    """Load and clean the UFC stats dataset."""
    df = pd.read_csv(path)
    for col in COUNT_COLS:
        df[col] = df[col].apply(parse_count)
    for col in TIME_COLS:
        df[col] = df[col].apply(parse_time)
    numeric_cols = NUMERIC_COLS + ['round']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['round'] = df['round'].astype(int)
    return df
