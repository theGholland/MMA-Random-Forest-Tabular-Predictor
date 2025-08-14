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
            regressor, classifier, label_encoders, use_cuda: bool = False):
    X_new = pd.DataFrame({
        'fighter_1': [fighter1],
        'fighter_2': [fighter2],
        'referee': [referee]
    })

    def _to_numpy(arr):
        """Convert cupy/cudf objects to numpy arrays."""
        if hasattr(arr, 'to_pandas'):
            arr = arr.to_pandas().to_numpy()
        if hasattr(arr, 'get'):
            arr = arr.get()
        return arr

    num_pred = _to_numpy(regressor.predict(X_new))[0]
    num_pred = [float(x) for x in num_pred]
    cat_pred_codes = _to_numpy(classifier.predict(X_new))[0]
    cat_pred_codes = [int(x) for x in cat_pred_codes]
    cat_pred = {
        col: label_encoders[col].inverse_transform([code])[0]
        for col, code in zip(label_encoders, cat_pred_codes)
    }
    cat_pred = {
        col: int(val) if hasattr(val, 'item') and isinstance(val.item(), int) else val
        for col, val in cat_pred.items()
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
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use GPU-accelerated models if available')
    args = parser.parse_args()

    regressor, classifier, label_encoders = load_models(args.model_dir)
    prediction = predict(args.fighter1, args.fighter2, args.referee,
                         regressor, classifier, label_encoders,
                         use_cuda=args.use_cuda)
    print(prediction)


if __name__ == '__main__':
    main()
