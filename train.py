import os
from typing import Optional

import joblib
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from utils import NUMERIC_COLS, HISTORIC_STATS, load_data


def train_models(
    csv_path: str,
    model_dir: str = "models",
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42,
    fighter_stats_csv: str = "fighter_stats.csv",
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

    X_train_cat = encoder.transform(train_df[cat_features])
    X_test_cat = encoder.transform(test_df[cat_features])
    X_train = sparse.hstack(
        [X_train_cat, sparse.csr_matrix(train_df[num_features].values)], format="csr"
    )
    X_test = sparse.hstack(
        [X_test_cat, sparse.csr_matrix(test_df[num_features].values)], format="csr"
    )

    train_mses, test_mses = [], []
    train_accs, test_accs = [], []

    # Train regressors for numeric columns
    for col in tqdm(NUMERIC_COLS, desc="Regressors"):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, train_df[col])
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_mse = mean_squared_error(train_df[col], train_pred)
        test_mse = mean_squared_error(test_df[col], test_pred)
        train_mses.append(train_mse)
        test_mses.append(test_mse)
        print(f"{col} Train MSE: {train_mse:.4f} Test MSE: {test_mse:.4f}")
        joblib.dump(model, os.path.join(model_dir, f"regressor_{col}.joblib"))

    # Train classifiers for categorical columns
    for col in tqdm(["result", "method", "round"], desc="Classifiers"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, train_df[col])
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(train_df[col], train_pred)
        test_acc = accuracy_score(test_df[col], test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"{col} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}")
        joblib.dump(model, os.path.join(model_dir, f"classifier_{col}.joblib"))

    avg_train_acc = sum(train_accs) / len(train_accs) if train_accs else 0.0
    avg_test_acc = sum(test_accs) / len(test_accs) if test_accs else 0.0
    avg_train_mse = sum(train_mses) / len(train_mses) if train_mses else 0.0
    avg_test_mse = sum(test_mses) / len(test_mses) if test_mses else 0.0

    print(
        f"Train Acc: {avg_train_acc:.4f} Test Acc: {avg_test_acc:.4f} "
        f"Train MSE: {avg_train_mse:.4f} Test MSE: {avg_test_mse:.4f}"
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
    args = parser.parse_args()

    train_models(
        args.csv_path,
        args.model_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        fighter_stats_csv=args.fighter_stats_csv,
    )

