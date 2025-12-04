# FULL WORKING FASTAPI + IRIS MODEL

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

MODEL_PATH = "model.pkl"

def train_model():
    if os.path.exists(MODEL_PATH):
        return

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)


model = load_model()

app = FastAPI()

class IrisInput(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: IrisInput):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr)[0]

    names = ["setosa", "versicolor", "virginica"]
    return {"prediction": names[pred]}


if __name__ == "__main__":
    uvicorn.run("mmm:app", host="0.0.0.0", port=8000, reload=True)
