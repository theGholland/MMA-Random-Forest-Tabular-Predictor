import argparse
import os
import joblib
import pandas as pd

from utils import NUMERIC_COLS


def load_models(model_dir: str = 'models'):
    regressor = joblib.load(os.path.join(model_dir, 'regressor.joblib'))
    classifier = joblib.load(os.path.join(model_dir, 'classifier.joblib'))
    label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.joblib'))
    return regressor, classifier, label_encoders


def predict(fighter1: str, fighter2: str, referee: str,
            regressor, classifier, label_encoders):
    X_new = pd.DataFrame({
        'fighter_1': [fighter1],
        'fighter_2': [fighter2],
        'referee': [referee]
    })
    num_pred = regressor.predict(X_new)[0]
    cat_pred_codes = classifier.predict(X_new)[0]
    cat_pred = {
        col: label_encoders[col].inverse_transform([code])[0]
        for col, code in zip(label_encoders, cat_pred_codes)
    }
    result = dict(zip(NUMERIC_COLS, num_pred))
    result.update(cat_pred)
    return result


def main():
    parser = argparse.ArgumentParser(description='Predict fight statistics and outcome')
    parser.add_argument('--fighter1', required=True)
    parser.add_argument('--fighter2', required=True)
    parser.add_argument('--referee', required=True)
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()

    regressor, classifier, label_encoders = load_models(args.model_dir)
    prediction = predict(args.fighter1, args.fighter2, args.referee,
                         regressor, classifier, label_encoders)
    print(prediction)


if __name__ == '__main__':
    main()
