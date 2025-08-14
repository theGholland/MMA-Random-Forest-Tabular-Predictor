import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import NUMERIC_COLS, load_data


def train_models(csv_path: str, model_dir: str = 'models') -> None:
    df = load_data(csv_path)
    X = df[['fighter_1', 'fighter_2', 'referee']]
    y_num = df[NUMERIC_COLS]
    y_cat = df[['result', 'method', 'round']]

    label_encoders = {}
    for col in y_cat.columns:
        le = LabelEncoder()
        y_cat[col] = le.fit_transform(y_cat[col])
        label_encoders[col] = le

    preproc = ColumnTransformer([
        ('names', OneHotEncoder(handle_unknown='ignore'), ['fighter_1', 'fighter_2', 'referee'])
    ])

    regressor = Pipeline([
        ('prep', preproc),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    classifier = Pipeline([
        ('prep', preproc),
        ('rf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42)))
    ])

    X_train, X_test, y_num_train, y_num_test, y_cat_train, y_cat_test = train_test_split(
        X, y_num, y_cat, test_size=0.2, random_state=42
    )

    regressor.fit(X_train, y_num_train)
    classifier.fit(X_train, y_cat_train)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(regressor, os.path.join(model_dir, 'regressor.joblib'))
    joblib.dump(classifier, os.path.join(model_dir, 'classifier.joblib'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))
    print(f'Models saved to {model_dir}')


if __name__ == '__main__':
    train_models('ufc_fight_stats.csv')
