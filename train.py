import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tensorboardX import SummaryWriter

from utils import NUMERIC_COLS, load_data


def _get_estimators(use_cuda: bool):
    """Return RandomForest estimators for regression and classification.

    If ``use_cuda`` is True and RAPIDS cuML is available, GPU-accelerated
    estimators are used. Otherwise, fall back to scikit-learn.
    """
    if use_cuda:
        try:
            from cuml.ensemble import (
                RandomForestClassifier as cuRFClassifier,
                RandomForestRegressor as cuRFRegressor,
            )
            return cuRFRegressor, cuRFClassifier
        except Exception:
            print("cuML is not available. Falling back to CPU scikit-learn models.")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    return RandomForestRegressor, RandomForestClassifier


def train_models(csv_path: str, model_dir: str = 'models', epochs: int = 1, use_cuda: bool = False) -> None:
    """Train random forest models.

    Parameters
    ----------
    csv_path: str
        Path to the training data CSV file.
    model_dir: str, optional
        Directory where the trained models will be saved.
    epochs: int, optional
        Number of epochs to train for. Models are re-fit each epoch.
    """

    df = load_data(csv_path)
    # create explicit copies to avoid SettingWithCopyWarning when modifying
    X = df[['fighter_1', 'fighter_2', 'referee']].copy()
    y_num = df[NUMERIC_COLS].copy()
    y_cat = df[['result', 'method', 'round']].copy()

    label_encoders = {}
    for col in y_cat.columns:
        le = LabelEncoder()
        y_cat[col] = le.fit_transform(y_cat[col])
        label_encoders[col] = le

    preproc = ColumnTransformer([
        ('names', OneHotEncoder(handle_unknown='ignore'), ['fighter_1', 'fighter_2', 'referee'])
    ])

    RFRegressor, RFClassifier = _get_estimators(use_cuda)

    regressor = Pipeline([
        ('prep', preproc),
        ('rf', RFRegressor(n_estimators=200, random_state=42))
    ])

    classifier = Pipeline([
        ('prep', preproc),
        ('rf', MultiOutputClassifier(RFClassifier(n_estimators=200, random_state=42)))
    ])

    X_train, X_test, y_num_train, y_num_test, y_cat_train, y_cat_test = train_test_split(
        X, y_num, y_cat, test_size=0.2, random_state=42
    )

    writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        regressor.fit(X_train, y_num_train)
        classifier.fit(X_train, y_cat_train)

        train_acc = classifier.score(X_train, y_cat_train)
        test_acc = classifier.score(X_test, y_cat_test)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f"Train Accuracy: {train_acc:.4f}  Test Accuracy: {test_acc:.4f}")

    writer.close()

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(regressor, os.path.join(model_dir, 'regressor.joblib'))
    joblib.dump(classifier, os.path.join(model_dir, 'classifier.joblib'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))
    print(f'Models saved to {model_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train fight outcome models')
    parser.add_argument('--csv-path', default='ufc_fight_stats.csv')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--use-cuda', action='store_true', help='Use GPU-accelerated training if available')
    args = parser.parse_args()

    train_models(args.csv_path, args.model_dir, epochs=args.epochs, use_cuda=args.use_cuda)
