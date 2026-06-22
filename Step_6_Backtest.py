"""
Step 6: Backtest - Predictions vs Actuals
==========================================
This script produces a side-by-side comparison of what each model
predicted vs what actually happened, for every day in the test set.

This is possible because the test set is historical data:
  - We know the actual outcome for every row (target column)
  - We can ask the model what it WOULD have predicted on each row
  - We compare the two

This is called backtesting. It is the standard way to evaluate
whether a model's predictions have any real-world value.

Outputs:
  reports/backtest_{TICKER}.csv          one row per day per model,
                                         enriched with actual prices
  reports/backtest_summary_{TICKER}.csv  accuracy rolled up by month
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob

from Step_2_Train import FEATURE_COLUMNS
from Step_3_Compare import load_all_models, load_test_set


# ------------------------------------------------------------
# STEP 6A: Load prices for enrichment
#
# We join the raw OHLCV prices back onto the comparison table
# so the output CSV shows the actual Close price and the price
# change alongside each prediction. This makes it much easier
# to audit whether a correct "UP" call was a big move or a tiny
# one, and whether wrong calls happened on volatile days.
# ------------------------------------------------------------

def load_prices(output_path):
    """
    Loads the prices CSV saved by step1.py.
    Returns a DataFrame indexed by Date with OHLCV columns.
    """
    df = pd.read_csv(output_path, index_col="Date", parse_dates=True)
    return df


# ------------------------------------------------------------
# STEP 6B: Build the day-by-day comparison table
#
# For each model and each day in the test set we record:
#   date             - the trading date
#   model            - which model made this prediction
#   actual           - 1 if price went up, 0 if down/flat
#   actual_label     - "UP" or "DOWN" (human readable)
#   predicted        - what the model said (1 or 0)
#   predicted_label  - "UP" or "DOWN" (human readable)
#   correct          - True if predicted == actual
#   prob_up          - model's confidence that price goes up
#   prob_down        - model's confidence that price goes down
#   close            - actual closing price that day (from prices.csv)
#   price_change_pct - actual % price change that day
# ------------------------------------------------------------

def build_comparison_table(models, X_test, y_test, TICKER, prices_df=None):
    """
    Returns a long-format DataFrame with one row per day per model.
    Long format means each model gets its own rows rather than its
    own columns. This makes it easy to filter, group, and chart.

    TICKER is stored as a column so that when multiple tickers are
    combined into one file, each row is still identifiable.

    If prices_df is provided, Close and price_change_pct are joined in.
    """
    all_rows = []

    for model_name, pipeline in models.items():
        predictions   = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        for i, date in enumerate(X_test.index):
            actual    = int(y_test.iloc[i])
            predicted = int(predictions[i])
            prob_up   = round(float(probabilities[i][1]), 4)
            prob_down = round(float(probabilities[i][0]), 4)

            all_rows.append({
                "date":            date.date(),
                "ticker":          TICKER,
                "model":           model_name,
                "actual":          actual,
                "actual_label":    "UP" if actual == 1 else "DOWN",
                "predicted":       predicted,
                "predicted_label": "UP" if predicted == 1 else "DOWN",
                "correct":         actual == predicted,
                "prob_up":         prob_up,
                "prob_down":       prob_down,
            })

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "model"]).reset_index(drop=True)

    # Join prices if provided
    if prices_df is not None:
        prices_slim = prices_df[["Close"]].copy()
        prices_slim["price_change_pct"] = prices_slim["Close"].pct_change().round(4)
        prices_slim.index = pd.to_datetime(prices_slim.index)

        df = df.merge(
            prices_slim,
            left_on="date",
            right_index=True,
            how="left"
        )
        df["Close"] = df["Close"].round(2)

    return df


# ------------------------------------------------------------
# STEP 6C: Print a readable daily view for the first N days
# ------------------------------------------------------------

def print_daily_view(df, n_days=20):
    """
    Prints a human-readable table of the first N dates,
    showing all three models side by side with the actual close price.
    """
    dates = sorted(df["date"].unique())[:n_days]

    has_prices = "Close" in df.columns

    print("\n" + "=" * 96)
    print("  Day-by-day: Actual vs Predicted  (first 20 test days)")
    print("=" * 96)

    if has_prices:
        print(f"  {'Date':<12} {'Close':>7} {'Chg%':>6} {'Actual':<6} {'Random Forest':<18} {'Log Reg':<18} {'Grad Boost':<14}")
    else:
        print(f"  {'Date':<12} {'Actual':<8} {'Random Forest':<18} {'Log Reg':<18} {'Grad Boost':<14}")

    print("  " + "-" * 92)

    model_order = ["random_forest", "logistic_regression", "gradient_boosting"]

    for date in dates:
        day_rows     = df[df["date"] == date]
        actual_label = day_rows.iloc[0]["actual_label"]

        cells = []
        for mname in model_order:
            row = day_rows[day_rows["model"] == mname]
            if row.empty:
                cells.append("N/A")
                continue
            row  = row.iloc[0]
            tick = "✓" if row["correct"] else "✗"
            conf = f"{max(row['prob_up'], row['prob_down']):.0%}"
            cells.append(f"{tick} {row['predicted_label']:<5} ({conf})")

        if has_prices:
            first_row  = day_rows.iloc[0]
            close      = f"${first_row['Close']:>6.2f}" if pd.notna(first_row.get("Close")) else "     N/A"
            chg        = f"{first_row['price_change_pct']:>+.1%}" if pd.notna(first_row.get("price_change_pct")) else "   N/A"
            print(
                f"  {str(date.date()):<12} {close} {chg}  "
                f"{actual_label:<6} "
                f"{cells[0]:<18} "
                f"{cells[1]:<18} "
                f"{cells[2]}"
            )
        else:
            print(
                f"  {str(date.date()):<12} "
                f"{actual_label:<8} "
                f"{cells[0]:<18} "
                f"{cells[1]:<18} "
                f"{cells[2]}"
            )

    print("=" * 96)


# ------------------------------------------------------------
# STEP 6D: Roll up accuracy by month
# ------------------------------------------------------------

def build_monthly_summary(df):
    """
    Returns accuracy per model per calendar month.
    """
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    summary = (
        df.groupby(["month", "model"])["correct"]
        .agg(total="count", correct_count="sum")
        .assign(accuracy=lambda x: (x["correct_count"] / x["total"]).round(3))
        .reset_index()
    )

    return summary


def print_monthly_summary(summary):
    pivot      = summary.pivot(index="month", columns="model", values="accuracy")
    pivot      = pivot.sort_index()
    model_cols = sorted(pivot.columns)

    print("\n" + "=" * 72)
    print("  Monthly accuracy by model")
    print("=" * 72)

    header = f"  {'Month':<10}"
    for col in model_cols:
        short = col.replace("_", " ").replace("gradient boosting", "grad boost")[:16]
        header += f"  {short:>16}"
    print(header)
    print("  " + "-" * 68)

    for month, row in pivot.iterrows():
        line = f"  {str(month):<10}"
        for col in model_cols:
            val  = row[col]
            flag = " ▲" if val >= 0.55 else (" ▼" if val <= 0.45 else "  ")
            line += f"  {val:>14.1%}{flag}"
        print(line)

    print("=" * 72)
    print("  ▲ = above 55% accuracy   ▼ = below 45% accuracy")


# ------------------------------------------------------------
# STEP 6E: Overall summary stats
# ------------------------------------------------------------

def print_overall_summary(df):
    summary = (
        df.groupby("model")["correct"]
        .agg(total="count", correct="sum")
        .assign(accuracy=lambda x: (x["correct"] / x["total"]).round(3))
    )

    direction_counts = (
        df.groupby(["model", "predicted_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"UP": "pred_up", "DOWN": "pred_down"})
    )

    summary = summary.join(direction_counts)

    print("\n" + "=" * 58)
    print("  Overall backtest summary")
    print("=" * 58)
    print(f"  {'Model':<26} {'Acc':>6} {'Pred UP':>8} {'Pred DN':>8}")
    print("  " + "-" * 54)

    for model_name, row in summary.iterrows():
        print(
            f"  {model_name:<26}"
            f"  {row['accuracy']:>5.1%}"
            f"  {int(row.get('pred_up', 0)):>8}"
            f"  {int(row.get('pred_down', 0)):>8}"
        )

    print("=" * 58)


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

# def fMain(MODELS_DIR, TEST_SET_PATH, OUTPUT_PATH, TICKER):
def fMain(cfg):
    """
    Parameters
    ----------
    MODELS_DIR    : str   e.g. "models"
    TEST_SET_PATH : str   e.g. "data/test_set_MSFT.csv"
    OUTPUT_PATH   : str   e.g. "data/prices_MSFT.csv"
    TICKER        : str   e.g. "MSFT"
    """
    backtest_dir   = "reports/backtesting"
    output_detail  = f"{backtest_dir}/backtest_{cfg.ticker}.csv"
    output_summary = f"{backtest_dir}/backtest_summary_{cfg.ticker}.csv"

    print("Loading models and test set...")
    models         = load_all_models(cfg.models_dir, cfg.ticker)
    X_test, y_test = load_test_set(cfg.test_set_path)
    print(f"  {len(models)} models, {len(X_test)} test days")

    print("\nLoading prices for enrichment...")
    prices_df = load_prices(cfg.output_path)

    print("\nBuilding comparison table...")
    comparison = build_comparison_table(models, X_test, y_test, cfg.ticker, prices_df)

    print_daily_view(comparison, n_days=20)

    monthly = build_monthly_summary(comparison)
    print_monthly_summary(monthly)

    print_overall_summary(comparison)

    os.makedirs(backtest_dir, exist_ok=True)
    comparison.to_csv(output_detail, index=False)
    monthly.to_csv(output_summary, index=False)

    print(f"\nDetailed results saved to : {output_detail}")
    print(f"Monthly summary saved to  : {output_summary}")
    print("\nOpen the backtest CSV in a spreadsheet to explore further.")


if __name__ == "__main__":
    print("Call fMain() with:")
    print("  MODELS_DIR, TEST_SET_PATH, OUTPUT_PATH, TICKER")


# ------------------------------------------------------------
# COMBINE: Merge all backtest files into one
#
# When you have run backtests for multiple tickers, this
# function reads every backtest_{TICKER}.csv from the
# backtesting folder and stacks them into a single file.
#
# The ticker column on each row is what lets you tell them
# apart after combining. In Excel or Power BI you can then
# filter or slice by ticker just like any other column.
#
# Two combined files are produced:
#   combined_detail.csv   all individual day/model rows
#   combined_summary.csv  all monthly accuracy rows
#
# Run this after you have backtested at least two tickers.
# ------------------------------------------------------------

def combine_backtest_files():
    """
    Reads all backtest_*.csv and backtest_summary_*.csv files
    from reports/backtesting/ and combines each set into one file.

    Output:
      reports/backtesting/combined_detail.csv
      reports/backtesting/combined_summary.csv
    """
    backtest_dir = "reports/backtesting"

    # --- Detail files ---
    # Match only ticker-specific files e.g. backtest_MSFT.csv
    # The combined file is named combined_detail.csv so it will never match
    detail_files = glob.glob(os.path.join(backtest_dir, "backtest_[A-Z]*.csv"))

    if not detail_files:
        print("No backtest detail files found. Run fMain() for at least one ticker first.")
        return

    detail_frames = []
    for path in sorted(detail_files):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        detail_frames.append(df)
        print(f"  Loaded {os.path.basename(path)}  ({len(df)} rows)")

    combined_detail = pd.concat(detail_frames, ignore_index=True)
    combined_detail = combined_detail.sort_values(["date", "ticker", "model"]).reset_index(drop=True)

    detail_out = os.path.join(backtest_dir, "combined_detail.csv")
    combined_detail.to_csv(detail_out, index=False)
    print(f"\nCombined detail saved to  : {detail_out}  ({len(combined_detail)} rows)")

    # --- Summary files ---
    # Match only ticker-specific summary files e.g. backtest_summary_MSFT.csv
    summary_files = glob.glob(os.path.join(backtest_dir, "backtest_summary_[A-Z]*.csv"))

    if summary_files:
        summary_frames = []
        for path in sorted(summary_files):
            df = pd.read_csv(path)

            # Derive ticker from filename: backtest_summary_MSFT.csv -> MSFT
            ticker = os.path.basename(path).replace("backtest_summary_", "").replace(".csv", "")
            df.insert(0, "ticker", ticker)
            summary_frames.append(df)

        combined_summary = pd.concat(summary_frames, ignore_index=True)
        summary_out = os.path.join(backtest_dir, "combined_summary.csv")
        combined_summary.to_csv(summary_out, index=False)
        print(f"Combined summary saved to : {summary_out}  ({len(combined_summary)} rows)")
