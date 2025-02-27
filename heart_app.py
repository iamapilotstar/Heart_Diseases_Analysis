import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# Streamlit Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
st.markdown(""" Note: This is a test app, Please dont use for real Diagnosis. Always Consult a doctor at all times.
""")

# ✅ Load Model and Scaler Efficiently
@st.cache_data
def load_model_and_scaler():
    model_path = "LogisticRegression.pkl"  
    scaler_path = "StandardScalar.pkl"

    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please upload 'LogisticRegression.pkl'.")
        return None, None

    if not os.path.exists(scaler_path):
        st.error("⚠️ Scaler file not found! Please upload 'StandardScalar.pkl'.")
        return None, None

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    return model, scaler

# Load once and use globally
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.stop()  # Stop execution if model or scaler is missing

# ✅ Streamlit App Title & Info
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the likelihood of a patient having heart disease based on clinical parameters.
Simply enter the patient details and get an instant prediction!
""")

# ✅ Input Form UI
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Patient Information")

    age = st.number_input("Age (years)", min_value=20, max_value=80, value=50)
    sex = st.radio("Sex", ["Male", "Female"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.slider("Max Heart Rate Achieved", 50, 220, 150)
    exercise_angina = st.radio("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    cholesterol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 200)

# ✅ Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
fasting_bs = 1 if fasting_bs == "Yes" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0

# ✅ One-hot encoding mappings
chest_pain_mapping = {
    "Typical Angina": "TA", "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP", "Asymptomatic": "Asym"
}

chest_pain_types = ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA']
chest_pain_encoded = {ctype: 0 for ctype in chest_pain_types}
if f"ChestPainType_{chest_pain_mapping[chest_pain_type]}" in chest_pain_types:
    chest_pain_encoded[f"ChestPainType_{chest_pain_mapping[chest_pain_type]}"] = 1

resting_ecg_mapping = {"Normal": "Normal", "ST-T Wave Abnormality": "ST", "Left Ventricular Hypertrophy": "LVH"}
resting_ecg_types = ['RestingECG_Normal', 'RestingECG_ST']
resting_ecg_encoded = {recg: 0 for recg in resting_ecg_types}
if f"RestingECG_{resting_ecg_mapping[resting_ecg]}" in resting_ecg_types:
    resting_ecg_encoded[f"RestingECG_{resting_ecg_mapping[resting_ecg]}"] = 1

st_slope_mapping = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}
st_slope_types = ['ST_Slope_Flat']
st_slope_encoded = {stype: 0 for stype in st_slope_types}
if f"ST_Slope_{st_slope_mapping[st_slope]}" in st_slope_types:
    st_slope_encoded[f"ST_Slope_{st_slope_mapping[st_slope]}"] = 1

# ✅ Feature Engineering
cholesterol_age_ratio = cholesterol / age
max_hr_age_ratio = max_hr / age

# ✅ Age Bracket Feature
if age <= 40:
    age_bracket = 0  
elif 40 < age <= 60:
    age_bracket = 1  
else:
    age_bracket = 2  

# ✅ Prepare Input Features in Correct Order
input_features = np.array([[ 
    sex, fasting_bs, exercise_angina, oldpeak, 
    cholesterol_age_ratio, max_hr_age_ratio, age_bracket,
    chest_pain_encoded['ChestPainType_ATA'], 
    chest_pain_encoded['ChestPainType_NAP'], 
    chest_pain_encoded['ChestPainType_TA'], 
    resting_ecg_encoded['RestingECG_Normal'], 
    resting_ecg_encoded['RestingECG_ST'], 
    st_slope_encoded['ST_Slope_Flat']
]])

# ✅ Prediction Button
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("❤️ Predict Heart Disease"):
        try:
            input_data_scaled = scaler.transform(input_features)

            if input_data_scaled.shape[1] != model.n_features_in_:
                st.error(f"⚠️ Model expects {model.n_features_in_} features, but received {input_data_scaled.shape[1]}.")
            else:
                prediction = model.predict(input_data_scaled)
                probabilities = model.predict_proba(input_data_scaled)[0]

            # ✅ New Threshold-Based Prediction
            if probabilities[1] >= 0.50:
                outcome = "Likely to Have Heart Disease"
            elif 0.30 <= probabilities[1] < 0.50:
                outcome = "Borderline Risk"
            else:
                outcome = "Unlikely to Have Heart Disease"

            st.success(f"❤️ Prediction: **{outcome}**")
            st.write(f"### 📊 Probability of Heart Disease: {probabilities[1] * 100:.2f}%")

            # ✅ Bar Chart for Probabilities
            prob_df = pd.DataFrame({
                'Outcome': ['No Heart Disease', 'Heart Disease'],
                'Probability': probabilities * 100
            })

            fig = px.bar(prob_df, x='Outcome', y='Probability',
                         title='Prediction Probabilities',
                         labels={'Probability': 'Probability (%)'},
                         color='Probability',
                         color_continuous_scale='Reds')

            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")

# ✅ Sidebar Information
st.sidebar.header("📌 About")
st.sidebar.info("""
This application predicts the likelihood of heart disease based on patient medical data using **Machine Learning**.

### **Model Information**
- **Algorithm:** Logistic Regression (Best Performing)
- **Trained on:** Heart Disease Dataset  
""")

st.sidebar.header("📊 Model Performance")
st.sidebar.markdown("""
- **Algorithm Used:** Logistic Regression
- **Best Accuracy:** 86.9%
- **ROC AUC Score:** 92.5%
""")

# ✅ Feature Importance Display
st.subheader("📊 Top Features Impacting Heart Disease Prediction")
feature_importance_path = "feature_importance.png"
if os.path.exists(feature_importance_path):
    st.image(Image.open(feature_importance_path), caption="Feature Importance", width=700)  
