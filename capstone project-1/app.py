import streamlit as st
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.markdown("Enter patient details below to check the risk of heart disease.")

with st.form("heart_form"):
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 220)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)", [0, 1, 2])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    sex_num = 1 if sex == "Male" else 0
    input_data = {
        "age": age, "sex": sex_num, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    try:
        response = requests.post(FASTAPI_URL, json={"data": input_data})
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            if prediction >= 0.5:
                st.error("⚠️ High Risk of Heart Disease Detected!")
            else:
                st.success("✅ Low Risk of Heart Disease.")
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚨 Failed to connect to FastAPI.\n\nError: {e}")

st.write("---")
st.caption("Developed by Sanjana 💻 | Powered by FastAPI + Streamlit + Machine Learning")
