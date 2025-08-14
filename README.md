# MMA-Random-Forest-Tabular-Predictor
Using Tabular UFC fight data to predict winners from a given matchup

## Model Overview
The training pipeline cleans raw fight statistics (for example converting "19 of 31" and "2:39" to numbers) and feeds fighter and referee names directly into [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests). Separate Random Forest models predict numeric fight statistics and categorical outcomes (result, method and round). Models are saved in TensorFlow's SavedModel format for use during prediction.

## Setup
Install dependencies (pinned to versions compatible with TensorFlow Decision Forests):

```
pip install -r requirements.txt
```

## Training
Train models and save them to the `models/` directory:

```
python train.py
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
python predict.py --fighter1 "Alexa Grasso" --fighter2 "Valentina Shevchenko" --referee "Herb Dean"
```

The script prints predicted fight statistics along with the result, method and round.
