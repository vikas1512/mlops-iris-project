import streamlit as st
import requests

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict the species.")

sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

if st.button("Predict"):
    data = {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=data)
        
        if res.status_code == 200:
            st.success(f"🌼 Prediction: {res.json()['prediction']}")
        else:
            st.error(f"API Error: {res.text}")

    except Exception as e:
        st.error(f"❌ Cannot connect to API: {e}")
