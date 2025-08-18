import argparse
import os
import difflib

import joblib
import pandas as pd
from scipy import sparse
import numpy as np

from utils import NUMERIC_COLS, HISTORIC_STATS, load_data


def load_models(model_dir: str = "models", num_passes: int = 1):
    encoder = joblib.load(os.path.join(model_dir, "encoder.joblib"))
    regressors = []
    classifiers = []
    for p in range(num_passes):
        regressors.append(
            {
                col: joblib.load(
                    os.path.join(model_dir, f"regressor_{col}_pass{p+1}.joblib")
                )
                for col in NUMERIC_COLS
            }
        )
        classifiers.append(
            {
                col: joblib.load(
                    os.path.join(model_dir, f"classifier_{col}_pass{p+1}.joblib")
                )
                for col in ["result", "method", "round"]
            }
        )
    label_encoders = {
        col: joblib.load(os.path.join(model_dir, f"label_encoder_{col}.joblib"))
        for col in ["result", "method", "round"]
    }
    return encoder, regressors, classifiers, label_encoders


def build_lookup(df: pd.DataFrame) -> dict:
    """Create a case-insensitive mapping of fighter names to stats."""
    f1_cols = ["fighter_1", "fighter_1_winloss"] + [
        f"fighter_1_avg_{stat}" for stat in HISTORIC_STATS
    ]
    f2_cols = ["fighter_2", "fighter_2_winloss"] + [
        f"fighter_2_avg_{stat}" for stat in HISTORIC_STATS
    ]
    f1 = df[f1_cols].rename(
        columns={
            "fighter_1": "fighter",
            "fighter_1_winloss": "winloss",
            **{f"fighter_1_avg_{stat}": f"avg_{stat}" for stat in HISTORIC_STATS},
        }
    )
    f2 = df[f2_cols].rename(
        columns={
            "fighter_2": "fighter",
            "fighter_2_winloss": "winloss",
            **{f"fighter_2_avg_{stat}": f"avg_{stat}" for stat in HISTORIC_STATS},
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
    label_encoders,
    num_passes: int = 1,
):
    cat_features = ["fighter_1", "fighter_2", "referee"]
    num_features = ["fighter_1_winloss", "fighter_2_winloss"]
    for stat in HISTORIC_STATS:
        num_features.extend(
            [f"fighter_1_avg_{stat}", f"fighter_2_avg_{stat}"]
        )

    f1_name, f1_stats = get_fighter_stats(fighter1, lookup)
    f2_name, f2_stats = get_fighter_stats(fighter2, lookup)
    row = {
        "fighter_1": f1_name,
        "fighter_2": f2_name,
        "referee": referee,
        "fighter_1_winloss": f1_stats.get("winloss", 0),
        "fighter_2_winloss": f2_stats.get("winloss", 0),
    }
    for stat in HISTORIC_STATS:
        row[f"fighter_1_avg_{stat}"] = f1_stats.get(f"avg_{stat}", 0)
        row[f"fighter_2_avg_{stat}"] = f2_stats.get(f"avg_{stat}", 0)

    X_new = pd.DataFrame([row])
    X_cat = encoder.transform(X_new[cat_features])
    X = sparse.hstack(
        [X_cat, sparse.csr_matrix(X_new[num_features].values)], format="csr"
    )
    X_curr = X
    for p in range(num_passes):
        num_models = regressors[p]
        cat_models = classifiers[p]
        num_pred = {
            col: float(num_models[col].predict(X_curr)[0]) for col in NUMERIC_COLS
        }
        cat_pred = {
            col: cat_models[col].predict(X_curr)[0]
            for col in ["result", "method", "round"]
        }
        if p < num_passes - 1:
            feat_vals = []
            for col in NUMERIC_COLS:
                feat_vals.append(num_pred[col])
            for col in ["result", "method", "round"]:
                le = label_encoders[col]
                feat_vals.append(le.transform([cat_pred[col]])[0])
            X_curr = sparse.hstack(
                [X_curr, sparse.csr_matrix(np.array(feat_vals).reshape(1, -1))],
                format="csr",
            )
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
    parser.add_argument(
        "--fighter-stats-csv",
        default="fighter_stats.csv",
        help="Path to cached fighter averages",
    )
    parser.add_argument(
        "--num-passes", type=int, default=1, help="Number of iterative passes"
    )
    args = parser.parse_args()

    encoder, regressors, classifiers, label_encoders = load_models(
        args.model_dir, args.num_passes
    )
    df = load_data(args.csv_path, args.fighter_stats_csv)
    lookup = build_lookup(df)

    prediction = predict(
        args.fighter1,
        args.fighter2,
        args.referee,
        encoder,
        regressors,
        classifiers,
        lookup,
        label_encoders,
        num_passes=args.num_passes,
    )
    print(prediction)


if __name__ == "__main__":
    main()

