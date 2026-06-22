import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# These are the only columns the model should ever see.
# Selecting them explicitly means load_and_split is safe even if
# features.csv contains extra columns (e.g. OHLCV from an older run).

FEATURE_COLUMNS = [
    "close_to_ma5",
    "close_to_ma20",
    "close_to_ma50",
    "daily_return",
    "momentum_5",
    "momentum_20",
    "volatility_20",
    "volume_ratio",
    "hl_range",
]


# ------------------------------------------------------------
# STEP 2A: Load and split the data
# ------------------------------------------------------------

def load_and_split(path, train_fraction):
    """
    Loads the features CSV and performs a time-based train/test split.

    Explicitly selects only the known feature columns rather than
    dropping target and taking everything else. This means the function
    is safe even if the CSV contains extra columns (e.g. OHLCV that
    crept in from an older version of step1.py).

    Returns:
      X_train, y_train  - features and labels the models learn from
      X_test,  y_test   - held-out features and labels for scoring
    """
    print(f"Loading features from {path}...")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)

    # Verify all expected columns are present before proceeding
    missing = [c for c in FEATURE_COLUMNS + ["target"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"features CSV is missing columns: {missing}. Re-run step1.py first."
        )

    # Select only the known feature columns and the target.
    # Any other columns in the file (e.g. OHLCV) are intentionally ignored.
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    split_index = int(len(df) * train_fraction)
    split_date  = df.index[split_index]

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_test  = y.iloc[split_index:]

    print(f"  Total rows    : {len(df)}")
    print(f"  Training rows : {len(X_train)}  (up to {split_date.date()})")
    print(f"  Test rows     : {len(X_test)}  (after {split_date.date()})")

    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------
# STEP 2B: Define the models
#
# Each model is wrapped in a Pipeline. A Pipeline chains together
# a sequence of steps that are applied in order.
#
# Here each pipeline has two steps:
#   1. StandardScaler  - rescales features to have mean=0, std=1
#   2. The classifier  - the actual ML model
#
# Why scale? Logistic Regression is sensitive to the scale of
# input features. A feature ranging 0..1000 will dominate one
# ranging 0..1 unless we normalize first. Random Forest and
# Gradient Boosting are not sensitive to scale, but including
# the scaler in the pipeline costs nothing and keeps things
# consistent across all models.
#
# Why a Pipeline and not just the model alone?
# Because the pipeline treats scaling + model as ONE object.
# When we save it and later call pipeline.predict(new_data),
# the scaler automatically transforms the new data the same
# way it transformed the training data. We never have to
# manually scale anything later.
# ------------------------------------------------------------

def build_models():
    """
    Returns a dictionary of named model pipelines.
    Each value is an untrained Pipeline ready to be fit.
    """
    models = {

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators=100,   # number of trees in the forest
                max_depth=5,        # how deep each tree can grow
                                    # (limits overfitting)
                random_state=42     # makes results reproducible
            ))
        ]),

        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                max_iter=1000,      # allow enough iterations to converge
                random_state=42
            ))
        ]),

        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingClassifier(
                n_estimators=100,   # number of sequential trees
                max_depth=3,        # shallow trees work better here
                                    # (deeper = more overfitting risk)
                learning_rate=0.05, # how much each tree corrects the last
                                    # (lower = more stable, needs more trees)
                random_state=42
            ))
        ]),
    }
    return models


# ------------------------------------------------------------
# STEP 2C: Train and save each model
#
# pickle is Python's built-in way to serialize objects to disk.
# Serializing means converting a Python object (the trained
# model) into a stream of bytes that can be written to a file
# and loaded back later into an identical Python object.
#
# .pkl is the conventional file extension for pickle files.
# ------------------------------------------------------------

def train_and_save(models, X_train, y_train, models_dir, TICKER):
    """
    Trains each model on the training data and saves it to disk.
    Returns a dict of {name: trained_pipeline}.
    """
    os.makedirs(models_dir, exist_ok=True)
    trained = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")

        # fit() is the core training call.
        # It passes X_train and y_train to the model and lets it
        # learn the mapping from features to labels.
        # After fit(), the pipeline holds the learned parameters.
        pipeline.fit(X_train, y_train)

        # Save the trained pipeline to a .pkl file
        save_path = os.path.join(models_dir, f"{name}_{TICKER}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pipeline, f)

        print(f"  Saved to {save_path}")
        trained[name] = pipeline

    return trained


# ------------------------------------------------------------
# STEP 2D: Save the test set
#
# We save X_test and y_test together as a single CSV.
# This lets step3.py load it independently.
# We are explicit about this being the SAME held-out rows
# every model will be scored against.
# ------------------------------------------------------------

def save_test_set(X_test, y_test, path):
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(path)
    print(f"\nTest set saved to {path} ({len(test_df)} rows)")


# def fMain(FEATURES_PATH, TRAIN_FRACTION, MODELS_DIR, TEST_SET_PATH, TICKER):
def fMain(cfg):
    X_train, y_train, X_test, y_test = load_and_split(cfg.features_path, cfg.train_fraction)

    models = build_models()
    trained_models = train_and_save(models, X_train, y_train, cfg.models_dir, cfg.ticker)

    save_test_set(X_test, y_test, cfg.test_set_path)

    print("\nAll models trained. Ready for step3.py")


if __name__ == "__main__":
    print("Call fMain() with:")
    print("  FEATURES_PATH, TRAIN_FRACTION, MODELS_DIR, TEST_SET_PATH, TICKER")
