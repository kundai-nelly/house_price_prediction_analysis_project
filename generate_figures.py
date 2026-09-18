"""
generate_figures.py — regenerates every chart used in README.md and the
notebook, plus metrics.json. Run after train_model.py.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

from train_model import (
    CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, load_and_clean,
)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)
sns.set_style("whitegrid")


def main():
    df_raw = load_and_clean(remove_outliers=False)
    df = load_and_clean(remove_outliers=True)

    metrics = {
        "n_rows_raw": len(df_raw),
        "n_rows_clean": len(df),
        "n_removed": len(df_raw) - len(df),
        "pct_removed": round((len(df_raw) - len(df)) / len(df_raw) * 100, 1),
    }

    # --- price distribution, before/after outlier removal -------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df_raw["price"], bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_title(f"Before ({len(df_raw)} rows)")
    axes[0].set_xlabel("Price (USD)")
    axes[1].hist(df["price"], bins=60, color="#55A868", edgecolor="white")
    axes[1].set_title(f"After single-pass IQR filter ({len(df)} rows)")
    axes[1].set_xlabel("Price (USD)")
    plt.suptitle("Price distribution, before and after outlier removal")
    plt.tight_layout()
    plt.savefig(ASSETS / "price_distribution.png", dpi=140)
    plt.close(fig)
    metrics["price_stats_raw"] = {
        "mean": round(float(df_raw["price"].mean()), 0),
        "median": round(float(df_raw["price"].median()), 0),
        "max": round(float(df_raw["price"].max()), 0),
    }
    metrics["price_stats_clean"] = {
        "mean": round(float(df["price"].mean()), 0),
        "median": round(float(df["price"].median()), 0),
        "max": round(float(df["price"].max()), 0),
    }

    # --- correlation heatmap --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    corr_cols = NUMERIC_FEATURES + ["price"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation heatmap, numeric features + price")
    plt.tight_layout()
    plt.savefig(ASSETS / "correlation_heatmap.png", dpi=140)
    plt.close(fig)
    price_corr = corr["price"].drop("price").sort_values(ascending=False)
    metrics["price_correlation"] = {k: round(float(v), 2) for k, v in price_corr.items()}

    # --- mutual information (feature relevance, non-linear-aware) -----------
    X_num = df[NUMERIC_FEATURES]
    mi = mutual_info_regression(X_num, df["price"], random_state=42)
    mi_series = pd.Series(mi, index=NUMERIC_FEATURES).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    mi_series.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information with price")
    ax.set_title("Feature relevance (mutual information)")
    plt.tight_layout()
    plt.savefig(ASSETS / "mutual_information.png", dpi=140)
    plt.close(fig)
    metrics["mutual_information"] = {k: round(float(v), 3) for k, v in mi_series.items()}

    # --- model comparison bar chart + actual-vs-predicted -------------------
    artifact = joblib.load(MODEL_PATH)
    results = artifact["all_results"]
    results_df = pd.DataFrame(results).sort_values("test_rmse")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(results_df["name"], results_df["test_rmse"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (USD, lower is better)")
    ax.set_title("Model comparison")
    plt.tight_layout()
    plt.savefig(ASSETS / "model_comparison.png", dpi=140)
    plt.close(fig)

    X = df.drop(columns=["price"])
    y = df["price"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = artifact["pipeline"]
    y_pred = pipeline.predict(X_test)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Predicted price (USD)")
    ax.set_title(f"Actual vs. predicted — {artifact['model_name']}")
    plt.tight_layout()
    plt.savefig(ASSETS / "actual_vs_predicted.png", dpi=140)
    plt.close(fig)

    # --- Ridge coefficients (only meaningful for the linear model) ----------
    if hasattr(pipeline.named_steps["model"], "coef_"):
        ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = NUMERIC_FEATURES + cat_names
        coefs = pipeline.named_steps["model"].coef_
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})

        # A one-hot dummy for a category with a handful of rows (or fewer)
        # isn't estimating a location effect, it's fitting noise from those
        # specific rows. Drop any city/statezip dummy backed by <10 rows in
        # the cleaned data before ranking, so the chart doesn't present a
        # single weird data point as a "finding". (Confirmed case: Medina /
        # WA 98039 both reduce to exactly 1 row after outlier removal, and
        # both dummies carry the identical, unreliable coefficient.)
        min_support = 10
        reliable = set()
        for col in CATEGORICAL_FEATURES:
            counts = df[col].value_counts()
            reliable.update(f"{col}_{v}" for v in counts[counts >= min_support].index)
        for feat in NUMERIC_FEATURES:
            reliable.add(feat)
        coef_df = coef_df[coef_df["feature"].isin(reliable)]

        top15 = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index).head(15)
        top15 = top15.sort_values("coef")

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#55A868" if c > 0 else "#C44E52" for c in top15["coef"]]
        ax.barh(top15["feature"], top15["coef"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient, USD per std. dev. (standardized numeric / one-hot city+zip)")
        ax.set_title(f"Top 15 price drivers — {artifact['model_name']}")
        plt.tight_layout()
        plt.savefig(ASSETS / "coefficients.png", dpi=140)
        plt.close(fig)
        metrics["top_positive_coefs"] = (
            coef_df[coef_df["coef"] > 0].sort_values("coef", ascending=False)
            [["feature", "coef"]].round(0).head(8).to_dict("records")
        )
        metrics["top_negative_coefs"] = (
            coef_df[coef_df["coef"] < 0].sort_values("coef")
            [["feature", "coef"]].round(0).head(8).to_dict("records")
        )

    metrics["model_metrics"] = {r["name"]: {k: round(v, 3) if isinstance(v, float) else v
                                             for k, v in r.items() if k != "name"} for r in results}
    metrics["model_name"] = artifact["model_name"]

    with open(HERE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {ASSETS} charts and metrics.json")


if __name__ == "__main__":
    main()
