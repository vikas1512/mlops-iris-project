# src/train_mlflow.py
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

FEATURED_PATH = "features/features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def train_with_mlflow(n_neighbors: int = 5):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(FEATURED_PATH)
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("iris_mlops")

    with mlflow.start_run():
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params & metrics
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("accuracy", acc)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # Save locally
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}, accuracy={acc:.4f}")


if __name__ == "__main__":
    train_with_mlflow()
