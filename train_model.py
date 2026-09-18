"""
train_model.py — House price prediction, Seattle-area housing dataset.

Cleans Housing_price_dataset.csv, removes outliers with a single-pass IQR
filter, trains and compares five regression models, and saves the selected
production model to house_price_production_model.pkl.

Script form of notebooks/House_Price_Analysis.ipynb.

Run with:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "Housing_price_dataset.csv"
MODEL_PATH = HERE / "house_price_production_model.pkl"

NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "month",
]
# IQR outlier removal only makes sense on genuinely continuous distributions.
# waterfront (binary flag) and view (skewed 0-4 ordinal) are excluded — see
# load_and_clean's docstring for why applying it to them is a real bug, not
# a style choice.
IQR_COLS = [
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
]
CATEGORICAL_FEATURES = ["city", "statezip"]


def load_and_clean(path: Path = DATA_PATH, remove_outliers: bool = True) -> pd.DataFrame:
    """Load the raw export, engineer date parts, drop unusable columns, and
    remove outliers with a single-pass IQR filter restricted to genuinely
    continuous columns.

    Two real problems in the original preprocessing, found by inspecting
    what it actually produced rather than assumed:

    1. It looped over *every* numeric column and re-filtered the
       already-shrunk dataframe on each pass, so later columns' bounds were
       computed on data already trimmed by earlier ones. That compounds,
       and it's why the original run dropped 28% of rows (4,600 -> 3,316)
       instead of the 25% a single upfront pass removes.
    2. It ran IQR filtering on `waterfront` (a 0/1 flag, ~99% zero) and
       `view` (an ordinal 0-4 score, heavily skewed toward 0) along with
       every genuinely continuous column. For a column that's almost all
       one value, the IQR bounds collapse to that value — so this silently
       deleted *every single waterfront property and every non-zero view
       score* from the dataset (confirmed: 33 waterfront homes in the raw
       data, 0 after the original's filter). That's not noise removal,
       it's erasing two of the more plausible "luxury premium" price
       drivers before the model ever saw them. This version restricts IQR
       filtering to the columns it's actually valid for.

    `year` is engineered from `date` but dropped as a feature below — the
    entire dataset is a single 3-month window in 2014, so `year` is
    constant (zero information) and `month` only carries a thin signal.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df = df.drop(columns=["date", "street", "country"])

    if not remove_outliers:
        return df

    mask = pd.Series(True, index=df.index)
    for col in IQR_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)

    n_before = len(df)
    df = df[mask].reset_index(drop=True)
    print(f"Single-pass IQR filter (continuous columns only): {n_before} -> {len(df)} rows "
          f"({n_before - len(df)} removed, {(n_before - len(df)) / n_before:.1%}). "
          f"Waterfront homes retained: {(df['waterfront'] == 1).sum()}.")
    return df


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def evaluate(name, pipeline, X_train, y_train, X_test, y_test) -> dict:
    cv_rmse = -np.mean(cross_val_score(
        pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5))
    y_pred = pipeline.predict(X_test)
    return {
        "name": name,
        "cv_rmse": cv_rmse,
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def main():
    df = load_and_clean()
    y = df["price"]
    X = df.drop(columns=["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    if XGB_AVAILABLE:
        candidates["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        metrics = evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(metrics)
        print(f"\n{name}")
        print(f"  CV RMSE={metrics['cv_rmse']:,.0f}  Test RMSE={metrics['test_rmse']:,.0f}  "
              f"MAE={metrics['test_mae']:,.0f}  R2={metrics['r2']:.3f}")

    results_sorted = sorted(results, key=lambda r: r["test_rmse"])
    chosen_name = results_sorted[0]["name"]
    chosen_pipeline = fitted[chosen_name]
    chosen_metrics = results_sorted[0]

    artifact = {
        "pipeline": chosen_pipeline,
        "model_name": chosen_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": chosen_metrics,
        "all_results": results,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nBest model on test RMSE: {chosen_name}")
    print(f"Saved to {MODEL_PATH}")

    return results, chosen_metrics


if __name__ == "__main__":
    main()
