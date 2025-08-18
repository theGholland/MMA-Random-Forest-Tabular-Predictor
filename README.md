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
Train models and save them to the `models/` directory. Historical per-fighter
averages are cached to a separate CSV (`fighter_stats.csv` by default) and
reused for both training and prediction:

```
python train.py --csv-path ufc_fight_stats.csv --model-dir models --fighter-stats-csv fighter_stats.csv
```

### Configurable Parameters
The training script exposes several arguments that can improve accuracy and precision:

* `--n-estimators` – number of trees in each forest (default: 100)
* `--max-depth` – maximum depth of each tree (default: unlimited)
* `--random-state` – seed for reproducible splits and model initialization (default: 42)
* `--num-passes` – number of iterative passes where model predictions are fed back as features (default: 1)

Tune these parameters to balance bias and variance for your dataset.

## Prediction
Load the saved models and generate predictions for a matchup (the same cached
fighter statistics file is required):

```
python predict.py --fighter1 "Alexa Grasso" --fighter2 "Valentina Shevchenko" --referee "Herb Dean" --fighter-stats-csv fighter_stats.csv
```

The script prints predicted fight statistics along with the result, method and round. Use `--num-passes` with `predict.py` if the models were trained with more than one pass.
