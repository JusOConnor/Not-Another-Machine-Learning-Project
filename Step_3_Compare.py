import pandas as pd
import numpy as np
import pickle
import os
import glob
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from Step_2_Train import FEATURE_COLUMNS


# ------------------------------------------------------------
# STEP 3A: Load all saved models from the models/ folder
# ------------------------------------------------------------

def load_all_models(models_dir, TICKER):
    """
    Scans models/ for ticker-specific .pkl files and loads each one.
    Returns a dict of {model_name: trained_pipeline}.
    """
    model_files = glob.glob(os.path.join(models_dir, f"*_{TICKER}.pkl"))

    if not model_files:
        raise FileNotFoundError(
            f"No .pkl files found for {TICKER} in {models_dir}. Run step2.py first."
        )

    models = {}
    for path in sorted(model_files):
        # Strip the ticker suffix to get a clean model name.
        # e.g. "random_forest_MSFT" -> "random_forest"
        basename = os.path.splitext(os.path.basename(path))[0]
        name = basename.replace(f"_{TICKER}", "")
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded {name}")

    return models


# ------------------------------------------------------------
# STEP 3B: Load the test set
# ------------------------------------------------------------

def load_test_set(path):
    df = pd.read_csv(path, index_col="Date", parse_dates=True)

    # Select only the known feature columns, same defensive pattern as step2.
    # This ensures test set column order always matches what the model expects.
    X_test = df[FEATURE_COLUMNS]
    y_test = df["target"]
    return X_test, y_test


# ------------------------------------------------------------
# STEP 3C: Score each model
#
# predict()       returns hard labels (0 or 1)
# predict_proba() returns probability scores [prob_0, prob_1]
#
# We use hard labels for accuracy, precision, and recall.
# We use probability scores for ROC AUC because AUC measures
# how well the model RANKS days (does it assign higher
# probability to days that actually go up?) rather than just
# how often the hard prediction is right.
# ------------------------------------------------------------

def score_model(pipeline, X_test, y_test):
    """
    Returns a dict of metric scores for a single model.
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of class 1

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


# ------------------------------------------------------------
# STEP 3D: Write results to the log
#
# We append a row to results_log.csv every time this script runs.
# This gives you a history of model performance over time so you
# can see whether retraining is improving or degrading things.
# Each row records the timestamp and the scores for all models.
# ------------------------------------------------------------

def append_results_log(results, log_path):
    """
    Appends one row per model to the results log CSV.
    Creates the file (with headers) if it does not exist yet.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for model_name, scores in results.items():
        row = {"timestamp": timestamp, "model": model_name}
        row.update(scores)
        rows.append(row)

    df_new = pd.DataFrame(rows)

    if os.path.exists(log_path):
        df_new.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(log_path, index=False)

    print(f"\nResults appended to {log_path}")


# ------------------------------------------------------------
# STEP 3E: Print comparison table and pick the best model
#
# We use ROC AUC as the tiebreaker metric because it is the
# most robust single-number summary for a binary classifier.
# You could change this to accuracy or any other metric.
# ------------------------------------------------------------

def print_comparison(results):
    df = pd.DataFrame(results).T
    df = df.sort_values("roc_auc", ascending=False)

    print("\n" + "=" * 58)
    print("  Model comparison on held-out test set")
    print("=" * 58)
    print(f"  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
    print("  " + "-" * 54)

    for model_name, row in df.iterrows():
        print(
            f"  {model_name:<28}"
            f"  {row['accuracy']:>5.3f}"
            f"  {row['precision']:>5.3f}"
            f"  {row['recall']:>5.3f}"
            f"  {row['roc_auc']:>5.3f}"
        )

    print("=" * 58)

    best_name = df.index[0]
    best_auc  = df.iloc[0]["roc_auc"]
    print(f"\n  Best model: {best_name}  (AUC {best_auc})")
    return best_name


def save_best_model_pointer(best_name, ptr_path, TICKER):
    """
    Writes the best model name (without ticker suffix) to a text file.
    step4.py reads this and reconstructs the full filename using TICKER.
    """
    with open(ptr_path, "w") as f:
        f.write(best_name)
    print(f"  Best model pointer saved to {ptr_path}")


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

# def fMain(MODELS_DIR, TEST_SET_PATH, RESULTS_LOG, BEST_MODEL_PTR, TICKER):
def fMain(cfg):
    print("Loading models...")
    models = load_all_models(cfg.models_dir, cfg.ticker)

    print("\nLoading test set...")
    X_test, y_test = load_test_set(cfg.test_set_path)
    print(f"  {len(X_test)} rows, {len(X_test.columns)} features")

    print("\nScoring models...")
    results = {}
    for name, pipeline in models.items():
        scores = score_model(pipeline, X_test, y_test)
        results[name] = scores
        print(f"  {name}: {scores}")

    best_name = print_comparison(results)
    append_results_log(results, cfg.results_log)
    save_best_model_pointer(best_name, cfg.best_model_ptr, cfg.ticker)

    print("\nReady for step4.py")


if __name__ == "__main__":
    print("Call fMain() with:")
    print("  MODELS_DIR, TEST_SET_PATH, RESULTS_LOG, BEST_MODEL_PTR, TICKER")
