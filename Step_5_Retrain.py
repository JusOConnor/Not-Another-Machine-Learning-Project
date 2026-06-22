import pandas as pd
import numpy as np
import pickle
import os
import shutil
from datetime import date, datetime

from Step_1_Data import pull_raw_data, engineer_features, save_features, save_prices
from Step_2_Train import load_and_split, build_models, train_and_save, save_test_set
from Step_3_Compare import load_all_models, load_test_set, score_model, print_comparison
from Step_3_Compare import append_results_log, save_best_model_pointer


# ------------------------------------------------------------
# STEP 5A: Pull fresh data up to today
# ------------------------------------------------------------

def pull_fresh_data(TICKER, START_DATE):
    end_date = date.today().isoformat()
    print(f"Pulling fresh data from {START_DATE} to {end_date}...")
    raw = pull_raw_data(TICKER, START_DATE, end_date)
    return raw


# ------------------------------------------------------------
# STEP 5B: Read the previous best score from the results log
#
# We look at the most recent row in the log for each model
# and find the best AUC score from the last run.
# This is what new models must beat to replace the old ones.
# ------------------------------------------------------------

def get_previous_best_auc(log_path):
    """
    Returns the best ROC AUC seen in the most recent log entry,
    or 0.0 if the log does not exist yet.
    """
    if not os.path.exists(log_path):
        print("  No previous results log found. All models will be accepted.")
        return 0.0

    df = pd.read_csv(log_path)
    if df.empty:
        return 0.0

    latest_ts   = df["timestamp"].max()
    latest_rows = df[df["timestamp"] == latest_ts]
    best_auc    = latest_rows["roc_auc"].max()

    print(f"  Previous best AUC ({latest_ts}): {best_auc:.4f}")
    return best_auc


# ------------------------------------------------------------
# STEP 5C: Back up current models before overwriting
# ------------------------------------------------------------

def backup_current_models(models_dir, backup_dir, TICKER):
    """
    Copies all ticker-specific .pkl files to a timestamped backup folder.
    Lets you roll back if needed.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(backup_dir, timestamp)
    os.makedirs(dest, exist_ok=True)

    backed_up = 0
    for fname in os.listdir(models_dir):
        if fname.endswith(f"_{TICKER}.pkl"):
            shutil.copy(os.path.join(models_dir, fname), dest)
            backed_up += 1

    if backed_up > 0:
        print(f"  Backed up {backed_up} models to {dest}")
    else:
        print("  No existing models to back up")


# ------------------------------------------------------------
# STEP 5D: Decide whether to accept new models
# ------------------------------------------------------------

def should_replace(new_best_auc, previous_best_auc, min_improvement):
    """
    Returns True if the new models should replace the old ones.
    """
    improvement = new_best_auc - previous_best_auc
    if improvement >= min_improvement:
        print(f"\n  New best AUC {new_best_auc:.4f} vs previous {previous_best_auc:.4f}")
        print(f"  Improvement: {improvement:+.4f} — accepting new models")
        return True
    else:
        print(f"\n  New best AUC {new_best_auc:.4f} vs previous {previous_best_auc:.4f}")
        print(f"  Improvement: {improvement:+.4f} — keeping existing models")
        return False


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

# def fMain(TICKER, START_DATE, FEATURES_PATH, OUTPUT_PATH, MODELS_DIR, BACKUP_DIR,
#           TEST_SET_PATH, RESULTS_LOG, BEST_MODEL_PTR, TRAIN_FRACTION, MIN_IMPROVEMENT):
def fMain(cfg):
    print("=" * 52)
    print("  Scheduled retraining")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 52)

    # Pull previous best score before we overwrite the log
    previous_best_auc = get_previous_best_auc(cfg.results_log)

    # Pull fresh data and rebuild features
    raw = pull_fresh_data(cfg.ticker, cfg.start_date)
    features, prices = engineer_features(raw)
    save_features(features, cfg.features_path)
    save_prices(prices, cfg.output_path)

    # Retrain all models on the updated data
    new_models_dir = os.path.join(cfg.models_dir, "new")
    X_train, y_train, X_test, y_test = load_and_split(cfg.features_path, cfg.train_fraction)
    models         = build_models()
    trained_models = train_and_save(models, X_train, y_train, new_models_dir, cfg.ticker)
    save_test_set(X_test, y_test, cfg.test_set_path)

    # Score new models
    print("\nScoring new models...")
    results = {}
    for name, pipeline in trained_models.items():
        results[name] = score_model(pipeline, X_test, y_test)

    best_name    = print_comparison(results)
    new_best_auc = results[best_name]["roc_auc"]

    # Decide whether to accept
    if should_replace(new_best_auc, previous_best_auc, cfg.min_improvement):
        backup_current_models(cfg.models_dir, cfg.backup_dir, cfg.ticker)

        for fname in os.listdir(new_models_dir):
            if fname.endswith(f"_{cfg.ticker}.pkl"):
                shutil.move(
                    os.path.join(new_models_dir, fname),
                    os.path.join(cfg.models_dir, fname)
                )

        shutil.rmtree(new_models_dir)
        append_results_log(results, cfg.results_log)
        save_best_model_pointer(best_name, cfg.best_model_ptr, cfg.ticker)
        print("\n  Models updated successfully.")
    else:
        shutil.rmtree(new_models_dir, ignore_errors=True)
        print("\n  Existing models retained. Retraining run logged but not applied.")
        append_results_log(results, cfg.results_log)

    print("\nRetraining run complete.")


if __name__ == "__main__":
    print("Call fMain() with:")
    print("  TICKER, START_DATE, FEATURES_PATH, OUTPUT_PATH, MODELS_DIR, BACKUP_DIR,")
    print("  TEST_SET_PATH, RESULTS_LOG, BEST_MODEL_PTR, TRAIN_FRACTION, MIN_IMPROVEMENT")
