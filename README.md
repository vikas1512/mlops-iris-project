# mlops-iris-project
🌸 MLOps Iris Classification Project

A complete end-to-end MLOps pipeline built using FastAPI, Streamlit, MLflow, Scikit-learn, and Monitoring with Drift Detection.

🚀 Project Overview

This project demonstrates a full MLOps workflow using the Iris dataset.
It covers everything from data ingestion → model training → deployment → monitoring.

The system allows users to enter iris flower measurements through a Streamlit web UI, which sends the data to a FastAPI backend that hosts the trained ML model.
All predictions are logged and monitored for data drift.

🧱 Project Structure
mlops-iris-project/
│
├── Code/                     # All scripts and ML code
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── train_mlflow.py
│   ├── mmm.py                # FastAPI backend
│   ├── frontend.py           # Streamlit UI
│   ├── model.pkl             # Trained ML model
│   ├── scaler.pkl            # Preprocessing scaler
│   └── monitor.py            # Drift detection script
│
├── Datasets/                 # Raw + processed datasets
│   ├── raw_iris.csv
│   ├── processed_iris.csv
│   └── features.csv
│
├── Results/                  # Screenshots + logs for evaluation
│   ├── fastapi_running.png
│   ├── streamlit_prediction.png
│   ├── mlflow_accuracy.png
│   ├── drift_detection.png
│   └── predictions.log
│
└── README.md                 # Project documentation

📊 1. Data Ingestion

Raw Iris dataset collected from sklearn.datasets or CSV.

Stored in Datasets/raw_iris.csv

Script used → Code/data_ingestion.py

Tasks performed:

Load raw data

Remove duplicates

Save cleaned dataset

🧪 2. Feature Engineering

Located in Code/feature_engineing.py.

Steps applied:

Standardization

Train/test split

Feature extraction

Save features.csv and scaler.pkl

🤖 3. Model Training with MLflow

Training code → Code/train_mlflow.py

Logistic Regression model used

MLflow used for:

Experiment tracking

Logging accuracy

Storing artifacts (model, scaler)

Trained model saved as:

Code/model.pkl

🚀 4. Model Deployment (FastAPI)

Backend code → mmm.py

Start the API:

uvicorn mmm:app --host 0.0.0.0 --port 8000 --reload


API Features:

/predict endpoint

Accepts flower measurements as JSON

Returns predicted species

Logs every prediction to Results/predictions.log

Example request:

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

🌐 5. Frontend (Streamlit UI)

Streamlit app → frontend.py

Run using:

streamlit run frontend.py


Allows users to:

Input flower measurements

Call FastAPI backend

Display prediction

🔍 6. Monitoring & Drift Detection

Monitoring script → monitor.py

Tracks drift by:

Watching new inputs in predictions.log

Comparing live means vs training means

Alerts when drift threshold is exceeded

Run manually using:

python monitor.py

📁 7. Results (Screenshots & Evidence)

Included in the Results/ folder:

File	Description
fastapi_running.png	API started successfully
streamlit_prediction.png	Prediction from UI
mlflow_accuracy.png	MLflow model metrics
drift_detection.png	Drift alert example
predictions.log	Logged live predictions
🛠️ Tech Stack Used
Component	Tool / Library
Programming	Python
Model Training	Scikit-learn
Tracking	MLflow
Deployment	FastAPI + Uvicorn
UI	Streamlit
Logging	Python Logging / CSV
Monitoring	Custom drift detection
Version Control	Git + GitHub
📦 How to Run the Whole Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Start FastAPI backend
uvicorn mmm:app --host 0.0.0.0 --port 8000 --reload

3️⃣ Start Streamlit frontend
streamlit run frontend.py

4️⃣ Run drift monitoring
python monitor.py

🎯 Conclusion

This project demonstrates the complete lifecycle of an ML system, automated and production-ready:

Data Pipeline

Feature Engineering

Model Training & Tracking

API Deployment

Web Interface

Monitoring & Drift Detection

A perfect end-to-end MLOps project for learning and showcasing skills.
