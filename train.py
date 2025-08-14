import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow logging

import tensorflow as tf
import tensorflow_decision_forests as tfdf
from absl import logging as absl_logging
from tqdm.auto import tqdm

from utils import NUMERIC_COLS, load_data, launch_tensorboard, configure_device


def train_models(csv_path: str, model_dir: str = 'models', epochs: int = 1, debuglevel: int = 0) -> None:
    """Train TensorFlow Decision Forest models and log metrics to TensorBoard."""
    configure_device()
    if debuglevel >= 2:
        tf.get_logger().setLevel("INFO")
        absl_logging.set_verbosity(absl_logging.INFO)
        absl_logging.set_stderrthreshold('info')
    else:
        tf.get_logger().setLevel("ERROR")
        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold('error')
    launch_tensorboard("runs")
    df = load_data(csv_path)
    features = ['fighter_1', 'fighter_2', 'referee']
    df_shuffled = df.sample(frac=1, random_state=42)
    split = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.iloc[:split].reset_index(drop=True)
    test_df = df_shuffled.iloc[split:].reset_index(drop=True)

    writer = tf.summary.create_file_writer('runs')
    os.makedirs(model_dir, exist_ok=True)

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        train_mses, test_mses = [], []
        train_accs, test_accs = [], []

        # Train regressors for numeric columns
        for col in NUMERIC_COLS:
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                train_df[features + [col]], label=col, task=tfdf.keras.Task.REGRESSION
            )
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                test_df[features + [col]], label=col, task=tfdf.keras.Task.REGRESSION
            )
            model = tfdf.keras.RandomForestModel(
                task=tfdf.keras.Task.REGRESSION, num_trees=100, verbose=0
            )
            fit_verbose = 1 if debuglevel >= 2 else 0
            model.fit(train_ds, verbose=fit_verbose)
            model.compile(metrics=['mse'])
            eval_verbose = 1 if debuglevel >= 2 else 0
            train_eval = model.evaluate(train_ds, return_dict=True, verbose=eval_verbose)
            test_eval = model.evaluate(test_ds, return_dict=True, verbose=eval_verbose)
            train_mses.append(train_eval['mse'])
            test_mses.append(test_eval['mse'])
            model.save(os.path.join(model_dir, f'regressor_{col}'))

        # Train classifiers for categorical columns
        class_info = {}
        for col in ['result', 'method', 'round']:
            classes = sorted(train_df[col].unique().tolist())
            class_info[col] = classes
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                train_df[features + [col]], label=col, task=tfdf.keras.Task.CLASSIFICATION
            )
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                test_df[features + [col]], label=col, task=tfdf.keras.Task.CLASSIFICATION
            )
            model = tfdf.keras.RandomForestModel(
                task=tfdf.keras.Task.CLASSIFICATION, num_trees=100, verbose=0
            )
            fit_verbose = 1 if debuglevel >= 2 else 0
            model.fit(train_ds, verbose=fit_verbose)
            model.compile(metrics=['accuracy'])
            eval_verbose = 1 if debuglevel >= 2 else 0
            train_eval = model.evaluate(train_ds, return_dict=True, verbose=eval_verbose)
            test_eval = model.evaluate(test_ds, return_dict=True, verbose=eval_verbose)
            train_accs.append(train_eval['accuracy'])
            test_accs.append(test_eval['accuracy'])
            model.save(os.path.join(model_dir, f'classifier_{col}'))

        with open(os.path.join(model_dir, 'class_info.json'), 'w') as f:
            json.dump(class_info, f)

        avg_train_acc = sum(train_accs) / len(train_accs) if train_accs else 0.0
        avg_test_acc = sum(test_accs) / len(test_accs) if test_accs else 0.0
        avg_train_mse = sum(train_mses) / len(train_mses) if train_mses else 0.0
        avg_test_mse = sum(test_mses) / len(test_mses) if test_mses else 0.0

        if debuglevel >= 1:
            tqdm.write(
                f"Epoch {epoch}/{epochs} - Train Acc: {avg_train_acc:.4f} Test Acc: {avg_test_acc:.4f} "
                f"Train MSE: {avg_train_mse:.4f} Test MSE: {avg_test_mse:.4f}"
            )
        with writer.as_default():
            tf.summary.scalar('Accuracy/train', avg_train_acc, step=epoch)
            tf.summary.scalar('Accuracy/test', avg_test_acc, step=epoch)
            tf.summary.scalar('MSE/train', avg_train_mse, step=epoch)
            tf.summary.scalar('MSE/test', avg_test_mse, step=epoch)
    writer.close()
    if debuglevel >= 1:
        tqdm.write(f'Models saved to {model_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train fight outcome models')
    parser.add_argument('--csv-path', default='ufc_fight_stats.csv')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--debuglevel', type=int, default=0, choices=[0, 1, 2],
                        help='0=silent, 1=metrics, 2=verbose')
    args = parser.parse_args()

    train_models(args.csv_path, args.model_dir, epochs=args.epochs, debuglevel=args.debuglevel)
