# MMA-Random-Forest-Tabular-Predictor
Using Tabular UFC fight data to predict winners from a given matchup

## Model Overview
The training pipeline cleans raw fight statistics (for example converting "19 of 31" and "2:39" to numbers) and one-hot encodes fighter and referee names. It builds two Random Forest models with 200 trees each: a `RandomForestRegressor` that predicts multiple numeric fight stats and a `MultiOutputClassifier` wrapping a `RandomForestClassifier` that predicts the fight result, method and round. All models and label encoders are saved so that, during prediction, the script loads them, assembles a one-row DataFrame for the matchup, translates predictions back to labels and returns a combined dictionary of fight statistics and outcome.

## Setup
Install dependencies:

```
pip install -r requirements.txt
```

## Training
Train models and save them to the `models/` directory:

```
python train.py [--use-cuda]
```

You can adjust the number of epochs and data/model paths:

```
python train.py --epochs 5 --csv-path ufc_fight_stats.csv --model-dir models
```

Training logs, including per-epoch accuracy, are written for TensorBoard. To
visualize them, run:

```
tensorboard --logdir runs
```

## Prediction
Load the saved models and generate predictions for a matchup:

```
python predict.py --fighter1 "Alexa Grasso" --fighter2 "Valentina Shevchenko" --referee "Herb Dean" [--use-cuda]
```

The script prints predicted fight statistics along with the result, method and round.
