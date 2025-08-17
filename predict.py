import argparse
import os
import difflib

import joblib
import pandas as pd
from scipy import sparse

from utils import NUMERIC_COLS, load_data


def load_models(model_dir: str = "models"):
    encoder = joblib.load(os.path.join(model_dir, "encoder.joblib"))
    regressors = {
        col: joblib.load(os.path.join(model_dir, f"regressor_{col}.joblib"))
        for col in NUMERIC_COLS
    }
    classifiers = {
        col: joblib.load(os.path.join(model_dir, f"classifier_{col}.joblib"))
        for col in ["result", "method", "round"]
    }
    return encoder, regressors, classifiers


def build_lookup(df: pd.DataFrame) -> dict:
    """Create a case-insensitive mapping of fighter names to stats."""
    f1 = df[
        [
            "fighter_1",
            "fighter_1_winloss",
            "fighter_1_avg_total_strikes",
            "fighter_1_avg_control_time",
            "fighter_1_avg_time",
        ]
    ].rename(
        columns={
            "fighter_1": "fighter",
            "fighter_1_winloss": "winloss",
            "fighter_1_avg_total_strikes": "avg_total_strikes",
            "fighter_1_avg_control_time": "avg_control_time",
            "fighter_1_avg_time": "avg_time",
        }
    )
    f2 = df[
        [
            "fighter_2",
            "fighter_2_winloss",
            "fighter_2_avg_total_strikes",
            "fighter_2_avg_control_time",
            "fighter_2_avg_time",
        ]
    ].rename(
        columns={
            "fighter_2": "fighter",
            "fighter_2_winloss": "winloss",
            "fighter_2_avg_total_strikes": "avg_total_strikes",
            "fighter_2_avg_control_time": "avg_control_time",
            "fighter_2_avg_time": "avg_time",
        }
    )
    lookup_df = (
        pd.concat([f1, f2], ignore_index=True)
        .drop_duplicates(subset=["fighter"])
        .assign(key=lambda x: x["fighter"].str.lower())
        .set_index("key")
    )
    return lookup_df.to_dict("index")


def get_fighter_stats(name: str, lookup: dict):
    """Return canonical name and stats for a fighter using fuzzy matching."""
    key = name.lower()
    if key not in lookup:
        matches = difflib.get_close_matches(key, lookup.keys(), n=1, cutoff=0.6)
        if matches:
            key = matches[0]
        else:
            return name, {}
    data = lookup[key]
    return data.get("fighter", name), data


def predict(
    fighter1: str,
    fighter2: str,
    referee: str,
    encoder,
    regressors,
    classifiers,
    lookup,
):
    cat_features = ["fighter_1", "fighter_2", "referee"]
    num_features = [
        "fighter_1_winloss",
        "fighter_1_avg_total_strikes",
        "fighter_1_avg_control_time",
        "fighter_1_avg_time",
        "fighter_2_winloss",
        "fighter_2_avg_total_strikes",
        "fighter_2_avg_control_time",
        "fighter_2_avg_time",
    ]

    f1_name, f1_stats = get_fighter_stats(fighter1, lookup)
    f2_name, f2_stats = get_fighter_stats(fighter2, lookup)
    row = {
        "fighter_1": f1_name,
        "fighter_2": f2_name,
        "referee": referee,
        "fighter_1_winloss": f1_stats.get("winloss", 0),
        "fighter_1_avg_total_strikes": f1_stats.get("avg_total_strikes", 0),
        "fighter_1_avg_control_time": f1_stats.get("avg_control_time", 0),
        "fighter_1_avg_time": f1_stats.get("avg_time", 0),
        "fighter_2_winloss": f2_stats.get("winloss", 0),
        "fighter_2_avg_total_strikes": f2_stats.get("avg_total_strikes", 0),
        "fighter_2_avg_control_time": f2_stats.get("avg_control_time", 0),
        "fighter_2_avg_time": f2_stats.get("avg_time", 0),
    }

    X_new = pd.DataFrame([row])
    X_cat = encoder.transform(X_new[cat_features])
    X = sparse.hstack(
        [X_cat, sparse.csr_matrix(X_new[num_features].values)], format="csr"
    )
    num_pred = {col: float(model.predict(X)[0]) for col, model in regressors.items()}
    cat_pred = {col: model.predict(X)[0] for col, model in classifiers.items()}
    result = dict(num_pred)
    result.update(cat_pred)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict fight statistics and outcome")
    parser.add_argument("--fighter1", required=True)
    parser.add_argument("--fighter2", required=True)
    parser.add_argument("--referee", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--csv-path", default="ufc_fight_stats.csv")
    args = parser.parse_args()

    encoder, regressors, classifiers = load_models(args.model_dir)
    df = load_data(args.csv_path)
    lookup = build_lookup(df)

    prediction = predict(
        args.fighter1,
        args.fighter2,
        args.referee,
        encoder,
        regressors,
        classifiers,
        lookup,
    )
    print(prediction)


if __name__ == "__main__":
    main()

