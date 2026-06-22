"""
Step 7: Walk-Forward Simulation
=================================
Simulates how models would have evolved through progressive retraining
by training on growing windows of historical data and scoring each one.

Instead of one fixed train/test split, this script:
  1. Starts with 60% of the data as the first training window
  2. Trains all three models on that window
  3. Scores each model on the NEXT 10% of data (unseen)
  4. Grows the window by 5% and repeats
  5. Exports all results to reports/walkforward/ for Power BI

This answers: "Is adding more data actually improving my models,
or are they just memorizing noise?"

Outputs:
  reports/walkforward/walkforward_{TICKER}.csv     one row per model per window
  reports/walkforward/walkforward_summary_{TICKER}.csv  best model per window
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Step_2_Train import FEATURE_COLUMNS, build_models


# ------------------------------------------------------------
# STEP 7A: Define the walk-forward windows
#
# We slide the training cutoff forward in steps, holding the
# test window fixed at 10% of total data each time.
#
# Example with 1,651 rows:
#   Window 1: train rows 0-989,   test rows 990-1154   (60% / 10%)
#   Window 2: train rows 0-1072,  test rows 1073-1237  (65% / 10%)
#   Window 3: train rows 0-1154,  test rows 1155-1319  (70% / 10%)
#   Window 4: train rows 0-1237,  test rows 1238-1402  (75% / 10%)
#   Window 5: train rows 0-1319,  test rows 1320-1484  (80% / 10%)
#
# Why a fixed test size rather than "everything after the cutoff"?
# Because we want scores that are comparable across windows.
# If the test set grew with each window, a later window's lower
# score might just reflect a harder test period, not a worse model.
# ------------------------------------------------------------

def build_windows(n_rows, start_frac=0.60, step_frac=0.05, test_frac=0.10):
    """
    Returns a list of (train_end_idx, test_end_idx) tuples.

    Parameters
    ----------
    n_rows     : total number of rows in the dataset
    start_frac : fraction of data used in the first training window
    step_frac  : how much to grow the training window each step
    test_frac  : fixed fraction of data used as the test window
    """
    windows = []
    frac = start_frac

    # while frac <= 0.80:
    while frac <= 0.90: #Changed to account for tail ends of the dataset that would otherwise be disgarded.
        train_end = int(n_rows * frac)
        test_end  = min(train_end + int(n_rows * test_frac), n_rows)

        if test_end > train_end:
            windows.append((train_end, test_end))

        frac = round(frac + step_frac, 2)

    return windows


# ------------------------------------------------------------
# STEP 7B: Train and score one window
#
# For each window we train fresh model instances from scratch.
# We do NOT load the saved .pkl files because those were trained
# on a fixed split. Walk-forward needs models trained on exactly
# the rows up to the cutoff for that window.
# ------------------------------------------------------------

def run_window(X, y, train_end, test_end, ticker, window_num):
    """
    Trains all models on rows 0..train_end and scores on
    rows train_end..test_end.

    Returns a list of result dicts, one per model.
    """
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test  = X.iloc[train_end:test_end]
    y_test  = y.iloc[train_end:test_end]

    train_start_date = X.index[0].strftime("%Y-%m-%d")
    train_end_date   = X.index[train_end - 1].strftime("%Y-%m-%d")
    test_start_date  = X.index[train_end].strftime("%Y-%m-%d")
    test_end_date    = X.index[test_end - 1].strftime("%Y-%m-%d")

    rows = []
    models = build_models()

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        rows.append({
            "ticker":           ticker,
            "window":           window_num,
            "train_start":      train_start_date,
            "train_end":        train_end_date,
            "test_start":       test_start_date,
            "test_end":         test_end_date,
            "train_rows":       train_end,
            "test_rows":        test_end - train_end,
            "model":            name,
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
            "pred_up_count":    int((y_pred == 1).sum()),
            "pred_down_count":  int((y_pred == 0).sum()),
            "actual_up_count":  int((y_test == 1).sum()),
            "actual_down_count":int((y_test == 0).sum()),
        })

        print(f"    {name:<26} AUC {rows[-1]['roc_auc']:.3f}  Acc {rows[-1]['accuracy']:.3f}")

    return rows


# ------------------------------------------------------------
# STEP 7C: Build the summary table
#
# For each window, identify the best model by AUC and record
# whether it improved over the previous window. This is the
# key signal for the retraining safeguard in step5:
#   improved = True  -> safe to swap in new model
#   improved = False -> keep the previous model
# ------------------------------------------------------------

def build_summary(df):
    """
    Returns one row per window showing the best model, its scores,
    and whether it improved over the prior window.
    """
    rows = []
    prev_auc = None

    for window_num in sorted(df["window"].unique()):
        window_df = df[df["window"] == window_num]
        best_row  = window_df.loc[window_df["roc_auc"].idxmax()]

        improved = None
        if prev_auc is not None:
            improved = float(best_row["roc_auc"]) >= prev_auc

        rows.append({
            "ticker":       best_row["ticker"],
            "window":       window_num,
            "train_end":    best_row["train_end"],
            "test_start":   best_row["test_start"],
            "test_end":     best_row["test_end"],
            "train_rows":   best_row["train_rows"],
            "best_model":   best_row["model"],
            "best_auc":     best_row["roc_auc"],
            "best_accuracy":best_row["accuracy"],
            "improved":     improved,
        })

        prev_auc = float(best_row["roc_auc"])

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# STEP 7D: Print results to console
# ------------------------------------------------------------

def print_results(df, summary):
    print("\n" + "=" * 72)
    print("  Walk-forward results by window")
    print("=" * 72)
    print(f"  {'Win':<5} {'Train end':<12} {'Test window':<24} {'Model':<26} {'AUC':>6} {'Acc':>6}")
    print("  " + "-" * 68)

    for _, row in df.iterrows():
        print(
            f"  {int(row['window']):<5}"
            f"  {row['train_end']:<12}"
            f"  {row['test_start']} to {row['test_end']}"
            f"  {row['model']:<26}"
            f"  {row['roc_auc']:>5.3f}"
            f"  {row['accuracy']:>5.3f}"
        )

    print("\n" + "=" * 58)
    print("  Best model per window")
    print("=" * 58)
    print(f"  {'Win':<5} {'Train end':<12} {'Best model':<26} {'AUC':>6} {'Better?':>8}")
    print("  " + "-" * 54)

    for _, row in summary.iterrows():
        improved_str = ""
        if row["improved"] is None:
            improved_str = "  (first)"
        elif row["improved"]:
            improved_str = "  ▲ yes"
        else:
            improved_str = "  ▼ no"

        print(
            f"  {int(row['window']):<5}"
            f"  {row['train_end']:<12}"
            f"  {row['best_model']:<26}"
            f"  {row['best_auc']:>5.3f}"
            f"  {improved_str}"
        )

    print("=" * 58)


# ------------------------------------------------------------
# STEP 7E: Save outputs for Power BI
#
# Two CSV files are produced:
#
#   walkforward_{TICKER}.csv
#     Granular data: one row per model per window. Use this in
#     Power BI to build line charts comparing all three models
#     across windows, or to filter by model and see score trends.
#
#   walkforward_summary_{TICKER}.csv
#     One row per window showing only the best model. Use this
#     to build a simple table visual or card showing which model
#     won each retraining cycle and whether things improved.
#
# Power BI tips:
#   - "window" is an integer, good for a numeric X axis
#   - "train_end" and "test_start" are dates, use as axis labels
#   - "improved" is boolean, convert to "Yes/No" in Power BI
#     using a calculated column:
#       Improved Label = IF([improved] = TRUE, "Yes", "No")
#   - Slice by "ticker" when you load multiple tickers
# ------------------------------------------------------------

def save_outputs(df, summary, walkforward_dir, ticker):
    os.makedirs(walkforward_dir, exist_ok=True)

    detail_path  = os.path.join(walkforward_dir, f"walkforward_{ticker}.csv")
    summary_path = os.path.join(walkforward_dir, f"walkforward_summary_{ticker}.csv")

    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nDetail saved to  : {detail_path}")
    print(f"Summary saved to : {summary_path}")


# ------------------------------------------------------------
# COMBINE: Merge all walk-forward files across tickers
#
# Same pattern as step6.combine_backtest_files().
# Run after processing multiple tickers to get one file
# you can load directly into Power BI.
# ------------------------------------------------------------

def combine_walkforward_files():
    """
    Combines all walkforward_*.csv and walkforward_summary_*.csv
    files into single combined files.

    Output:
      reports/walkforward/combined_detail.csv
      reports/walkforward/combined_summary.csv
    """
    walkforward_dir = "reports/walkforward"

    detail_files = glob.glob(os.path.join(walkforward_dir, "walkforward_[A-Z]*.csv"))
    if not detail_files:
        print("No walk-forward detail files found. Run fMain() for at least one ticker first.")
        return

    detail_frames = []
    for path in sorted(detail_files):
        df = pd.read_csv(path)
        detail_frames.append(df)
        print(f"  Loaded {os.path.basename(path)}  ({len(df)} rows)")

    combined_detail = pd.concat(detail_frames, ignore_index=True)
    combined_detail = combined_detail.sort_values(["ticker", "window", "model"]).reset_index(drop=True)

    detail_out = os.path.join(walkforward_dir, "combined_detail.csv")
    combined_detail.to_csv(detail_out, index=False)
    print(f"\nCombined detail saved to  : {detail_out}  ({len(combined_detail)} rows)")

    summary_files = glob.glob(os.path.join(walkforward_dir, "walkforward_summary_[A-Z]*.csv"))
    if summary_files:
        summary_frames = []
        for path in sorted(summary_files):
            df = pd.read_csv(path)
            summary_frames.append(df)

        combined_summary = pd.concat(summary_frames, ignore_index=True)
        combined_summary = combined_summary.sort_values(["ticker", "window"]).reset_index(drop=True)

        summary_out = os.path.join(walkforward_dir, "combined_summary.csv")
        combined_summary.to_csv(summary_out, index=False)
        print(f"Combined summary saved to : {summary_out}  ({len(combined_summary)} rows)")


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def fMain(cfg):
    """
    Parameters pulled from cfg:
      cfg.ticker        : e.g. "MSFT"
      cfg.features_path : e.g. "data/features_MSFT.csv"
    """
    ticker        = cfg.ticker
    features_path = cfg.features_path
    walkforward_dir = "reports/walkforward"

    print("=" * 52)
    print(f"  Walk-forward simulation: {ticker}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 52)

    print(f"\nLoading features from {features_path}...")
    df = pd.read_csv(features_path, index_col="Date", parse_dates=True)

    missing = [c for c in FEATURE_COLUMNS + ["target"] if c not in df.columns]
    if missing:
        raise ValueError(f"features CSV is missing columns: {missing}. Re-run step1.py first.")

    X = df[FEATURE_COLUMNS]
    y = df["target"]
    print(f"  {len(df)} rows loaded")

    windows = build_windows(len(df))
    print(f"  {len(windows)} windows to process\n")

    all_rows = []
    for i, (train_end, test_end) in enumerate(windows, start=1):
        train_pct = round(train_end / len(df) * 100)
        print(f"Window {i}: training on {train_end} rows ({train_pct}%), testing on {test_end - train_end} rows")
        window_rows = run_window(X, y, train_end, test_end, ticker, i)
        all_rows.extend(window_rows)

    results_df = pd.DataFrame(all_rows)
    summary_df = build_summary(results_df)

    print_results(results_df, summary_df)
    save_outputs(results_df, summary_df, walkforward_dir, ticker)

    print("\nWalk-forward simulation complete.")
    print("Run combine_walkforward_files() after processing all tickers.")

    return results_df, summary_df


if __name__ == "__main__":
    print("Call fMain(cfg) with an MLConfig instance.")
    print("Call combine_walkforward_files() to merge results across tickers.")
