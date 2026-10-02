# Freight Rate Prediction Challenge

## Approach
- Labeled development data: 48,000 rows from January-October 2025.
- Final validation set: 12,000 rows from November-December 2025.
- Development split: chronological; January-August for training and September-October for holdout validation.
- Missing values: CatBoost handles numeric missing values; categorical missing values are represented as `MISSING`.
- Final model: CatBoost regression.
- Features: pickup, delivery, distance, equipment, weight, route, weight-per-mile, year/month/day/day-of-week/day-of-year/week-of-year.
- `load_id` is excluded from modeling.
- Coordinates, market_index and quote_signal were not used in the final model because the model using the December-compatible feature schema performed slightly better on the chronological holdout.

## Holdout results
- RMSE: 633.69
- MAE: 122.17
- R²: 0.8276
- Best iteration from early stopping: 178

## Run
```bash
python -m pip install -r requirements.txt
python train_model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Outputs
- `validation_predictions.csv`
- `data/december_chart_inputs.csv`
- `scorer_results/candidate_december.png`
- `freight_rate_catboost.cbm`

## Submission
Submit the GitHub repository, `validation_predictions.csv`, the PDF report, the December chart, and the 2-3 minute Loom walkthrough.
