# src/data_ingestion.py
import os
import pandas as pd
from sklearn.datasets import load_iris

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw_iris.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_iris.csv")

def ingest_and_prepare():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load from sklearn and save raw
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].apply(lambda i: iris.target_names[i])
    df = df.drop(columns=["target"])

    df.to_csv(RAW_PATH, index=False)

    # 2. Basic cleaning (here just drop NA)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Raw saved to {RAW_PATH}")
    print(f"Processed saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    ingest_and_prepare()
