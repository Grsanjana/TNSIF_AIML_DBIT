import streamlit as st
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("### Enter patient details to check the possibility of heart disease.")

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
        "age": age,
        "sex": sex_num,
        "chest_pain_type": cp,
        "resting_blood_pressure": trestbps,
        "cholesterol": chol,
        "fasting_blood_sugar": fbs,
        "resting_ecg": restecg,
        "max_heart_rate": thalach,
        "exercise_induced_angina": exang,
        "st_depression": oldpeak,
        "st_slope": slope,
        "num_major_vessels": ca,
        "thalassemia": thal
    }

    try:
        response = requests.post(FASTAPI_URL, json={"data": input_data})

        if response.status_code == 200:
            result = response.json()
            prediction_value = result.get("prediction_value", 0)
            probability = result.get("probability", None)

            st.info(f"🧩 **Prediction Value:** {prediction_value}")

            if prediction_value == 1:
                st.error("⚠️ **Heart Disease Detected!**")
                if probability:
                    st.write(f"🧠 Probability: **{probability * 100:.2f}%**")
                st.warning("🩺 Risk Level: **High Risk** – Immediate consultation recommended.")
            else:
                st.success("✅ **No Heart Disease Detected.**")
                if probability:
                    st.write(f"🧠 Probability: **{probability * 100:.2f}%**")
                st.info("💚 Risk Level: **Low Risk** – Continue healthy habits.")

        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"🚨 Failed to connect to FastAPI.\n\n**Error:** {e}")

st.write("---")
st.caption("Developed by **Sanjana 💻** | Powered by FastAPI ⚙️ + Streamlit 🎨 + Machine Learning 🧠")
