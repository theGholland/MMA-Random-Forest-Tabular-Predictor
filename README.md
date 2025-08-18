# MMA-Random-Forest-Tabular-Predictor
Using Tabular UFC fight data to predict winners from a given matchup

## Model Overview
The training pipeline cleans raw fight statistics (for example converting "19 of 31" and "2:39" to numbers) and encodes fighter and referee names for use with [scikit-learn](https://scikit-learn.org/). Separate Random Forest models predict numeric fight statistics and categorical outcomes (result, method and round). Models and the feature encoder are saved with `joblib` for later prediction. Fighter features now include historical averages for all tracked fight metrics, giving the models richer context for both training and prediction.

## Setup
Install dependencies:

```
pip install -r requirements.txt
```

## Training
Train models and save them to the `models/` directory:

```
python train.py --csv-path ufc_fight_stats.csv --model-dir models
```

### Configurable Parameters
The training script exposes several arguments that can improve accuracy and precision:

* `--n-estimators` – number of trees in each forest (default: 100)
* `--max-depth` – maximum depth of each tree (default: unlimited)
* `--random-state` – seed for reproducible splits and model initialization (default: 42)

Tune these parameters to balance bias and variance for your dataset.

## Prediction
Load the saved models and generate predictions for a matchup:

```
python predict.py --fighter1 "Alexa Grasso" --fighter2 "Valentina Shevchenko" --referee "Herb Dean"
```

The script prints predicted fight statistics along with the result, method and round.
