from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def make_features(df):
    x = df[["pickup","delivery","distance","equipment","weight","date"]].copy()
    x["date"] = pd.to_datetime(x["date"])
    x["year"] = x["date"].dt.year
    x["month"] = x["date"].dt.month
    x["dayofmonth"] = x["date"].dt.day
    x["dayofweek"] = x["date"].dt.dayofweek
    x["dayofyear"] = x["date"].dt.dayofyear
    x["weekofyear"] = x["date"].dt.isocalendar().week.astype(int)
    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)
    x["weight_per_mile"] = x["weight"] / x["distance"].replace(0, np.nan)
    x = x.drop(columns=["date"])
    cats = [c for c in x.columns if x[c].dtype == "object"]
    x[cats] = x[cats].fillna("MISSING")
    return x, cats

train = pd.read_csv(DATA/"train_test.csv")
validation = pd.read_csv(DATA/"validation.csv")
template = pd.read_csv(DATA/"validation_predictions_template.csv")
december = pd.read_csv(DATA/"december_chart_inputs.csv")

train["date"] = pd.to_datetime(train["date"])
X, cats = make_features(train)
y = train["posted_rate"].astype(float)

# 177 was selected by chronological early stopping during development.
model = CatBoostRegressor(
    iterations=177, depth=7, learning_rate=0.04,
    loss_function="RMSE", l2_leaf_reg=8,
    random_seed=42, verbose=False, allow_writing_files=False
)
model.fit(X, y, cat_features=cats, verbose=False)

validation_X, _ = make_features(validation)
pred = np.maximum(model.predict(validation_X), 0.01)
prediction_map = pd.Series(pred, index=validation["load_id"].astype(str))
out = template[["load_id"]].copy()
out["predicted_rate"] = out["load_id"].astype(str).map(prediction_map)
out.to_csv(ROOT/"validation_predictions.csv", index=False)

december_X, _ = make_features(december)
december["predicted_rate"] = np.maximum(model.predict(december_X), 0.01)
december.to_csv(DATA/"december_chart_inputs.csv", index=False)

model.save_model(str(ROOT/"freight_rate_catboost.cbm"))
print("Created validation_predictions.csv and completed data/december_chart_inputs.csv")
