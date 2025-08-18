import os
from typing import Optional

import joblib
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import numpy as np

from utils import NUMERIC_COLS, HISTORIC_STATS, load_data


def train_models(
    csv_path: str,
    model_dir: str = "models",
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42,
    fighter_stats_csv: str = "fighter_stats.csv",
    num_passes: int = 1,
) -> None:
    """Train scikit-learn Random Forest models for fight statistics and outcomes."""

    df = load_data(csv_path, fighter_stats_csv)
    cat_features = ["fighter_1", "fighter_2", "referee"]
    num_features = ["fighter_1_winloss", "fighter_2_winloss"]
    for stat in HISTORIC_STATS:
        num_features.extend(
            [f"fighter_1_avg_{stat}", f"fighter_2_avg_{stat}"]
        )
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.iloc[:split]
    test_df = df_shuffled.iloc[split:]

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(train_df[cat_features])

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(encoder, os.path.join(model_dir, "encoder.joblib"))

    cat_label_encoders = {
        col: LabelEncoder().fit(train_df[col]) for col in ["result", "method", "round"]
    }
    for col, le in cat_label_encoders.items():
        joblib.dump(le, os.path.join(model_dir, f"label_encoder_{col}.joblib"))

    X_train_cat = encoder.transform(train_df[cat_features])
    X_test_cat = encoder.transform(test_df[cat_features])
    X_train = sparse.hstack(
        [X_train_cat, sparse.csr_matrix(train_df[num_features].values)], format="csr"
    )
    X_test = sparse.hstack(
        [X_test_cat, sparse.csr_matrix(test_df[num_features].values)], format="csr"
    )

    X_train_curr = X_train
    X_test_curr = X_test
    for p in range(num_passes):
        train_mses, test_mses = [], []
        train_accs, test_accs = [], []
        train_feats, test_feats = [], []

        for col in tqdm(NUMERIC_COLS, desc=f"Regressors Pass {p+1}"):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train_curr, train_df[col])
            train_pred = model.predict(X_train_curr)
            test_pred = model.predict(X_test_curr)
            train_mse = mean_squared_error(train_df[col], train_pred)
            test_mse = mean_squared_error(test_df[col], test_pred)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            joblib.dump(
                model, os.path.join(model_dir, f"regressor_{col}_pass{p+1}.joblib")
            )
            if p == num_passes - 1:
                print(f"{col} Train MSE: {train_mse:.4f} Test MSE: {test_mse:.4f}")
            train_feats.append(train_pred.reshape(-1, 1))
            test_feats.append(test_pred.reshape(-1, 1))

        for col in tqdm(["result", "method", "round"], desc=f"Classifiers Pass {p+1}"):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train_curr, train_df[col])
            train_pred = model.predict(X_train_curr)
            test_pred = model.predict(X_test_curr)
            train_acc = accuracy_score(train_df[col], train_pred)
            test_acc = accuracy_score(test_df[col], test_pred)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            joblib.dump(
                model, os.path.join(model_dir, f"classifier_{col}_pass{p+1}.joblib")
            )
            if p == num_passes - 1:
                print(f"{col} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}")
            le = cat_label_encoders[col]
            train_feats.append(le.transform(train_pred).reshape(-1, 1))
            test_feats.append(le.transform(test_pred).reshape(-1, 1))

        avg_train_acc = sum(train_accs) / len(train_accs) if train_accs else 0.0
        avg_test_acc = sum(test_accs) / len(test_accs) if test_accs else 0.0
        avg_train_mse = sum(train_mses) / len(train_mses) if train_mses else 0.0
        avg_test_mse = sum(test_mses) / len(test_mses) if test_mses else 0.0
        print(
            f"Pass {p+1}/{num_passes} Train Acc: {avg_train_acc:.4f} Test Acc: {avg_test_acc:.4f} "
            f"Train MSE: {avg_train_mse:.4f} Test MSE: {avg_test_mse:.4f}"
        )

        if p < num_passes - 1:
            X_train_curr = sparse.hstack(
                [X_train_curr, sparse.csr_matrix(np.column_stack(train_feats))],
                format="csr",
            )
            X_test_curr = sparse.hstack(
                [X_test_curr, sparse.csr_matrix(np.column_stack(test_feats))],
                format="csr",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fight outcome models")
    parser.add_argument("--csv-path", default="ufc_fight_stats.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--fighter-stats-csv",
        default="fighter_stats.csv",
        help="Path to cached fighter averages",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of trees in each forest"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Maximum depth of each tree"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-passes", type=int, default=1, help="Number of iterative passes"
    )
    args = parser.parse_args()

    train_models(
        args.csv_path,
        args.model_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        fighter_stats_csv=args.fighter_stats_csv,
        num_passes=args.num_passes,
    )

