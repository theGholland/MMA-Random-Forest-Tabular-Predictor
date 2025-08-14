import argparse
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from utils import NUMERIC_COLS, launch_tensorboard, configure_device


def load_models(model_dir: str = 'models'):
    regressors = {}
    for col in NUMERIC_COLS:
        path = os.path.join(model_dir, f'regressor_{col}')
        regressors[col] = tf.keras.models.load_model(path)
    classifiers = {}
    for col in ['result', 'method', 'round']:
        path = os.path.join(model_dir, f'classifier_{col}')
        classifiers[col] = tf.keras.models.load_model(path)
    with open(os.path.join(model_dir, 'class_info.json')) as f:
        class_info = json.load(f)
    return regressors, classifiers, class_info


def predict(fighter1: str, fighter2: str, referee: str,
            regressors, classifiers, class_info):
    X_new = pd.DataFrame({
        'fighter_1': [fighter1],
        'fighter_2': [fighter2],
        'referee': [referee]
    })
    ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_new)
    num_pred = {
        col: float(model.predict(ds)[0][0])
        for col, model in regressors.items()
    }
    cat_pred = {}
    for col, model in classifiers.items():
        proba = model.predict(ds)[0]
        classes = class_info[col]
        if proba.ndim == 0 or len(proba) == 1:
            pred = classes[1] if float(proba[0]) >= 0.5 else classes[0]
        else:
            pred = classes[int(proba.argmax())]
        cat_pred[col] = pred
    result = dict(num_pred)
    result.update(cat_pred)
    return result


def main():
    parser = argparse.ArgumentParser(description='Predict fight statistics and outcome')
    parser.add_argument('--fighter1', required=True)
    parser.add_argument('--fighter2', required=True)
    parser.add_argument('--referee', required=True)
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()

    configure_device()
    launch_tensorboard('runs')
    regressors, classifiers, class_info = load_models(args.model_dir)
    prediction = predict(args.fighter1, args.fighter2, args.referee,
                         regressors, classifiers, class_info)
    print(prediction)


if __name__ == '__main__':
    main()
