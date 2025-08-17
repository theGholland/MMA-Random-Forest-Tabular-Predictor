import argparse
import os

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

    f1_stats = lookup.get(fighter1, {})
    f2_stats = lookup.get(fighter2, {})
    row = {
        "fighter_1": fighter1,
        "fighter_2": fighter2,
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
        .set_index("fighter")
    )
    lookup = lookup_df.to_dict("index")

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

