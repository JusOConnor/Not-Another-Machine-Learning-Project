"""
Step 1: Data Pipeline
=====================
This script does two things:
  1. Pulls historical stock price data from Yahoo Finance
  2. Engineers features (the columns the model will learn from)

Call fMain() from other scripts or run directly for a quick test.

Outputs:
  data/features.csv   engineered columns only, used for training
  data/prices.csv     raw OHLCV columns, saved for reference only
"""

import yfinance as yf
import pandas as pd
import os


# ------------------------------------------------------------
# STEP 1A: Pull raw price data
# ------------------------------------------------------------

def pull_raw_data(ticker, start, end):
    """
    Downloads daily OHLCV data from Yahoo Finance.

    OHLCV stands for:
      Open  - price at market open
      High  - highest price of the day
      Low   - lowest price of the day
      Close - price at market close
      Volume - number of shares traded

    Falls back to data/raw_prices.csv if network is unavailable.
    On your own machine, the yfinance download will work directly.

    Returns a DataFrame with one row per trading day.
    """
    print(f"Pulling data for {ticker} from {start} to {end}...")

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if len(raw) == 0:
            raise ValueError("Empty download result")

    except Exception as e:
        print(f"  yfinance unavailable ({e}), loading from data/raw_prices.csv")
        raw = pd.read_csv("data/raw_prices.csv", index_col="Date", parse_dates=True)

    print(f"  Loaded {len(raw)} trading days")
    return raw


# ------------------------------------------------------------
# STEP 1B: Engineer features
#
# Features are the input columns the model will learn from.
# Raw price data alone is not useful because the model would
# just learn that higher prices come after lower prices, which
# is trivially true but tells it nothing predictive.
#
# Instead we build RELATIVE and MOMENTUM features that capture
# how the price is behaving, not just what it is.
# ------------------------------------------------------------

def engineer_features(df):
    """
    Takes raw OHLCV data and adds engineered columns.

    Each feature is explained inline below.
    Returns a tuple of (features DataFrame, prices DataFrame).
    """
    data = df.copy()

    # -- Moving averages --
    # A moving average smooths out day-to-day noise.
    # We use the ratio of the current close to the moving average
    # so the value is always relative (e.g. 1.02 = 2% above average)
    # rather than absolute (e.g. $430). Relative values generalize
    # much better across different time periods and price levels.

    data["ma_5"]  = data["Close"].rolling(5).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()
    data["ma_50"] = data["Close"].rolling(50).mean()

    data["close_to_ma5"]  = data["Close"] / data["ma_5"]
    data["close_to_ma20"] = data["Close"] / data["ma_20"]
    data["close_to_ma50"] = data["Close"] / data["ma_50"]

    # -- Daily return --
    # How much did the price move today, as a percentage.
    # pct_change() computes (today - yesterday) / yesterday.

    data["daily_return"] = data["Close"].pct_change()

    # -- Momentum (return over a window) --
    # How much has price moved over the past N days.
    # shift(N) looks back N rows.

    data["momentum_5"]  = data["Close"] / data["Close"].shift(5)  - 1
    data["momentum_20"] = data["Close"] / data["Close"].shift(20) - 1

    # -- Volatility --
    # Standard deviation of daily returns over the past 20 days.
    # High volatility = price is moving around a lot lately.

    data["volatility_20"] = data["daily_return"].rolling(20).std()

    # -- Volume ratio --
    # Today's volume relative to the 20-day average volume.
    # Spikes in volume often precede or accompany large moves.

    data["volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()

    # -- High/Low range --
    # How wide was today's price range, relative to the close.
    # A big range day signals indecision or news.

    data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]

    # ------------------------------------------------------------
    # STEP 1C: Build the target (what we are trying to predict)
    #
    # The target column is what the model learns to predict.
    # We want to know: will tomorrow's price be higher than today's?
    #   1 = yes (price goes up)
    #   0 = no  (price goes flat or down)
    #
    # shift(-1) looks FORWARD one row (i.e. tomorrow's close).
    # We compare tomorrow's close to today's close.
    #
    # IMPORTANT: This is only safe because we never let the model
    # see tomorrow's close as a feature. It only sees today and past.
    # ------------------------------------------------------------

    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # -- Drop rows with NaN values --
    # Rolling calculations need a warm-up period.
    # The first 50 rows will have NaN values (no 50-day MA yet).
    # We also drop the last row because target = NaN (no tomorrow).

    data = data.dropna()

    # -- Select only the columns we need --
    # We keep the feature columns and the target.
    # We drop raw OHLCV columns because the model should not learn
    # from raw price levels directly.
    #
    # OHLCV is returned separately so the caller can save it to its
    # own reference file. This keeps the training data pure while
    # still preserving the raw prices for auditing and joining to
    # backtest results.

    feature_columns = [
        "close_to_ma5",
        "close_to_ma20",
        "close_to_ma50",
        "daily_return",
        "momentum_5",
        "momentum_20",
        "volatility_20",
        "volume_ratio",
        "hl_range",
        "target"
    ]

    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]

    result = data[feature_columns]
    prices = data[ohlcv_columns]

    print(f"  Engineered {len(result)} rows with {len(feature_columns) - 1} features")
    print(f"  Target distribution: {result['target'].value_counts().to_dict()}")

    return result, prices


# ------------------------------------------------------------
# STEP 1D: Save to CSV
#
# We save two files:
#
#   features.csv  - engineered columns only, used for training
#   prices.csv    - raw OHLCV columns, saved for reference
#
# Keeping them separate means nothing stops the training scripts
# from accidentally picking up a raw price column. If you ever
# want to audit a prediction or join prices back to the backtest
# output, load prices.csv and merge on the Date index.
#
# prices.csv lives in the same folder as features.csv, derived
# automatically from the features path so callers only need to
# pass one path.
# ------------------------------------------------------------

def save_features(df, features_path):
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df.to_csv(features_path)
    print(f"  Saved features to {features_path}")


def save_prices(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"  Saved prices to   {output_path}")


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def fMain(cfg):
    """
    Parameters
    ----------
    ticker        : str   e.g. "SPY"
    start_date    : str   e.g. "2018-01-01"
    end_date      : str   e.g. "2024-12-31"
    features_path : str   e.g. "data/features_SPY.csv"
    output_path   : str   e.g. "data/prices_SPY.csv"
    """
    raw = pull_raw_data(cfg.ticker, cfg.start_date, cfg.end_date)
    features, prices = engineer_features(raw)
    save_features(features, cfg.features_path)
    save_prices(prices, cfg.output_path)

    print("\nFirst 5 rows of features:")
    print(features.head())
    print("\nFirst 5 rows of prices:")
    print(prices.head())

    return features, prices


if __name__ == "__main__":
    print("Call fMain() with:")
    print("  ticker        e.g. 'SPY'")
    print("  start_date    e.g. '2018-01-01'")
    print("  end_date      e.g. '2024-12-31'")
    print("  features_path e.g. 'data/features.csv'")
    print("")
    print("Example:")
    print("  from step_1_data import fMain")
    print("  fMain('SPY', '2018-01-01', '2024-12-31', 'data/features.csv')")
