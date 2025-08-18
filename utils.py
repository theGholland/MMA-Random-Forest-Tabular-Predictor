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

# Base stat names used for historical aggregates
BASE_STATS = [c[:-2] for c in NUMERIC_COLS if c.endswith('_1')]
HISTORIC_STATS = BASE_STATS + ['time']

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

    # ------------------------------------------------------------------
    # Historical fighter statistics
    # ------------------------------------------------------------------
    # Build a per-fighter record of wins, losses and average metrics so
    # that each row contains information about both competitors based on
    # all fights present in the dataset.  The `result` column reflects the
    # outcome for `fighter_1`; invert it for `fighter_2` when aggregating.

    stat_cols_1 = [f"{s}_1" for s in BASE_STATS]
    stat_cols_2 = [f"{s}_2" for s in BASE_STATS]
    f1 = df[["fighter_1", "result"] + stat_cols_1 + ["time"]].rename(
        columns={**{"fighter_1": "fighter"}, **{f"{s}_1": s for s in BASE_STATS}}
    )
    f2 = df[["fighter_2", "result"] + stat_cols_2 + ["time"]].rename(
        columns={**{"fighter_2": "fighter"}, **{f"{s}_2": s for s in BASE_STATS}}
    )
    # Invert result for the second fighter (W -> L, L -> W, others unchanged)
    f2["result"] = f2["result"].map({"W": "L", "L": "W"}).fillna(f2["result"])

    fighter_stats = pd.concat([f1, f2], ignore_index=True)

    wins = fighter_stats["result"].eq("W").groupby(fighter_stats["fighter"]).sum()
    losses = fighter_stats["result"].eq("L").groupby(fighter_stats["fighter"]).sum()
    winloss = wins - losses
    avg_stats = fighter_stats.groupby("fighter")[HISTORIC_STATS].mean()

    fighter_agg = avg_stats.add_prefix("avg_")
    fighter_agg["winloss"] = winloss

    df = df.merge(
        fighter_agg.add_prefix("fighter_1_"),
        left_on="fighter_1",
        right_index=True,
        how="left",
    )
    df = df.merge(
        fighter_agg.add_prefix("fighter_2_"),
        left_on="fighter_2",
        right_index=True,
        how="left",
    )

    new_cols = []
    for prefix in ["fighter_1", "fighter_2"]:
        new_cols.append(f"{prefix}_winloss")
        for stat in HISTORIC_STATS:
            new_cols.append(f"{prefix}_avg_{stat}")
    df[new_cols] = df[new_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df
