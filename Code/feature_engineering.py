# src/feature_engineering.py
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_iris.csv")
FEATURE_STORE_DIR = "features"
SCALER_PATH = os.path.join(FEATURE_STORE_DIR, "scaler.pkl")
FEATURED_PATH = os.path.join(FEATURE_STORE_DIR, "features.csv")

def build_features():
    os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop("species", axis=1)
    y = df["species"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df["species"] = y

    X_scaled_df.to_csv(FEATURED_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Features saved to {FEATURED_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    build_features()
