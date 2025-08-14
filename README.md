# MMA-Random-Forest-Tabular-Predictor
Using Tabular UFC fight data to predict winners from a given matchup

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

## Prediction
Load the saved models and generate predictions for a matchup:

```
python predict.py --fighter1 "Alexa Grasso" --fighter2 "Valentina Shevchenko" --referee "Herb Dean" [--use-cuda]
```

The script prints predicted fight statistics along with the result, method and round.
