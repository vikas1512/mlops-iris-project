import numpy as np
import ast
import os

LOG_PATH = "logs/predictions.log"
TRAIN_STATS = "features/train_stats.npy"   # optional, if you saved train stats

def load_live_stats():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError("No predictions.log found yet.")

    features_list = []

    with open(LOG_PATH, "r") as f:
        for line in f:
            try:
                _, features_str, _ = line.strip().split(",", 2)

                # Safely parse feature list
                if not features_str.endswith("]"):
                    continue  # skip broken rows

                features = ast.literal_eval(features_str)
                features_list.append(features)

            except Exception:
                continue  # skip any bad/malformed rows

    if len(features_list) == 0:
        raise ValueError("No valid feature rows found in log.")

    arr = np.array(features_list)
    return arr.mean(axis=0), arr.std(axis=0)

def simple_drift_check():
    live_mean, live_std = load_live_stats()

    print("Live mean:", live_mean)
    print("Live std:", live_std)
    print("\nOK — Monitoring working without crash.")

if __name__ == "__main__":
    simple_drift_check()
