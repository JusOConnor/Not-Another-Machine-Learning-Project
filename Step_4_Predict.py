import pandas as pd
import numpy as np
import pickle
import os
import yfinance as yf

from Step_1_Data import engineer_features, pull_raw_data
from Step_2_Train import FEATURE_COLUMNS


# ------------------------------------------------------------
# STEP 4A: Read which model is the best
# ------------------------------------------------------------

def load_best_model(models_dir, ptr_path, TICKER):
    """
    Reads the best model name from the pointer file,
    reconstructs the full filename using TICKER,
    then loads and returns that model pipeline.

    The pointer file stores just the base name (e.g. "random_forest").
    The actual file on disk is "random_forest_MSFT.pkl".
    Keeping them separate means the pointer file stays ticker-agnostic
    and we only need TICKER in one place when loading.
    """
    with open(ptr_path, "r") as f:
        best_name = f.read().strip()

    model_path = os.path.join(models_dir, f"{best_name}_{TICKER}.pkl")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    print(f"Loaded model: {best_name}")
    return best_name, pipeline


# ------------------------------------------------------------
# STEP 4B: Pull the most recent data
# ------------------------------------------------------------

def pull_recent_data(ticker, lookback_days):
    """
    Pulls the last N days of price data.
    Falls back to the last N rows of raw_prices_{ticker}.csv if offline.
    """
    from datetime import date, timedelta

    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days * 2)).isoformat()

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if len(raw) == 0:
            raise ValueError("Empty download")
        print(f"Pulled {len(raw)} recent rows from yfinance")
    except Exception as e:
        print(f"yfinance unavailable ({e}), using last {lookback_days} rows of prices_{ticker}.csv")
        raw = pd.read_csv(f"data/prices_{ticker}.csv", index_col="Date", parse_dates=True)
        raw = raw.tail(lookback_days + 20)

    return raw


# ------------------------------------------------------------
# STEP 4C: Build features for the latest row
#
# We call engineer_features() on the full recent window,
# then take only the LAST row.
#
# Why pass the full window and not just one row?
# Because features like the 50-day moving average need 50 prior
# rows to compute. We need the history to produce a valid
# feature value for today.
#
# After engineer_features() runs we take the last row, which
# represents "today" (the most recent trading day).
# We drop the target column because we don't know tomorrow yet.
# ------------------------------------------------------------

def build_latest_features(raw_df):
    features, _ = engineer_features(raw_df)

    if len(features) == 0:
        raise ValueError("Not enough data to compute features. Need at least 50 rows.")

    latest      = features.iloc[[-1]]       # keep as DataFrame, not Series
    latest_date = latest.index[0]

    # Select only the known feature columns in the correct order.
    # This guarantees the input to predict() matches what the model
    # was trained on, even if engineer_features() column order changes.
    X_latest = latest[FEATURE_COLUMNS]

    print(f"Features built for date: {latest_date.date()}")
    return X_latest, latest_date


# ------------------------------------------------------------
# STEP 4D: Make the prediction
#
# pipeline.predict()       returns [0] or [1]
# pipeline.predict_proba() returns [[prob_down, prob_up]]
#
# We show both so you can see not just the direction call but
# also how confident the model is.
# ------------------------------------------------------------

def make_prediction(pipeline, X_latest):
    prediction    = pipeline.predict(X_latest)[0]
    probabilities = pipeline.predict_proba(X_latest)[0]

    direction  = "UP" if prediction == 1 else "DOWN / FLAT"
    confidence = probabilities[prediction]

    return prediction, direction, confidence, probabilities


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

# def fMain(MODELS_DIR, BEST_MODEL_PTR, LOOKBACK_DAYS, TICKER):
def fMain(cfg):
    best_name, pipeline = load_best_model(cfg.models_dir, cfg.best_model_ptr, cfg.ticker)

    raw = pull_recent_data(cfg.ticker, cfg.lookback_days)
    X_latest, latest_date = build_latest_features(raw)

    prediction, direction, confidence, probs = make_prediction(pipeline, X_latest)

    print("\n" + "=" * 46)
    print("  Prediction")
    print("=" * 46)
    print(f"  Model          : {best_name}")
    print(f"  As of date     : {latest_date.date()}")
    print(f"  Ticker         : {cfg.ticker}")
    print(f"  Direction      : {direction}")
    print(f"  Confidence     : {confidence:.1%}")
    print(f"  P(up)          : {probs[1]:.1%}")
    print(f"  P(down/flat)   : {probs[0]:.1%}")
    print("=" * 46)
    print("\nNote: this is a learning project, not financial advice.")


if __name__ == "__main__":
    print("Call fMain() with:")
    print("  MODELS_DIR, BEST_MODEL_PTR, LOOKBACK_DAYS, TICKER")
