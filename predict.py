import argparse
import os

import joblib
import pandas as pd

from utils import NUMERIC_COLS


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


def predict(fighter1: str, fighter2: str, referee: str, encoder, regressors, classifiers):
    X_new = pd.DataFrame(
        {"fighter_1": [fighter1], "fighter_2": [fighter2], "referee": [referee]}
    )
    X = encoder.transform(X_new)
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
    args = parser.parse_args()

    encoder, regressors, classifiers = load_models(args.model_dir)
    prediction = predict(
        args.fighter1, args.fighter2, args.referee, encoder, regressors, classifiers
    )
    print(prediction)


if __name__ == "__main__":
    main()

