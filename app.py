import streamlit as st
import pandas as pd
import joblib
import ollama
import matplotlib.pyplot as plt
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------- Load Model --------
@st.cache_resource
def load_model():
    return joblib.load("cvd_rf_model.pkl")

rf_model = load_model()

st.title("🫀 Cardiovascular Risk Prediction System")


# -------- MEDICAL DOCUMENT VALIDATION --------
def is_medical_report(text):

    medical_keywords = [
        "age",
        "blood pressure",
        "cholesterol",
        "heart",
        "patient",
        "hospital",
        "report",
        "ecg"
    ]

    text = text.lower()

    count = sum(keyword in text for keyword in medical_keywords)

    return count >= 2


# -------- INPUT METHOD --------
method = st.sidebar.radio(
    "Choose Input Method",
    ["Manual Input","Upload Medical Report Image"]
)


# -------- MANUAL INPUT --------
if method == "Manual Input":

    age = st.sidebar.slider("Age",1,120,50)

    sex = st.sidebar.selectbox("Sex",["Male","Female"])
    sex = 1 if sex=="Male" else 0

    cp = st.sidebar.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps = st.sidebar.slider("Blood Pressure",80,250,120)
    chol = st.sidebar.slider("Cholesterol",100,600,200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar >120",["False","True"])
    fbs = 1 if fbs=="True" else 0
    restecg = st.sidebar.selectbox("Rest ECG",[0,1,2])
    thalach = st.sidebar.slider("Max Heart Rate",60,220,150)
    exang = st.sidebar.selectbox("Exercise Angina",["No","Yes"])
    exang = 1 if exang=="Yes" else 0
    oldpeak = st.sidebar.slider("Oldpeak",0.0,10.0,1.0)
    slope = st.sidebar.selectbox("Slope",[0,1,2])
    ca = st.sidebar.selectbox("Major Vessels",[0,1,2,3])
    thal = st.sidebar.selectbox("Thal",[1,2,3])

    patient = {
        "age":age,
        "sex":sex,
        "cp":cp,
        "trestbps":trestbps,
        "chol":chol,
        "fbs":fbs,
        "restecg":restecg,
        "thalach":thalach,
        "exang":exang,
        "oldpeak":oldpeak,
        "slope":slope,
        "ca":ca,
        "thal":thal
    }

    patient_df = pd.DataFrame([patient])


# -------- IMAGE UPLOAD ---------
else:

    uploaded_file = st.file_uploader(
        "Upload Medical Report Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image,caption="Uploaded Medical Report")

        img = np.array(image)

        text = pytesseract.image_to_string(img)

        st.subheader("Extracted Text")
        st.text(text)


        # - CHECK IF MEDICAL DOCUMENT 
        if not is_medical_report(text):

            st.error("❌ Please upload a valid medical report image.")

            st.stop()


        # -------- Extract Values using Regex 
        def find_value(pattern,text,default):
            match = re.search(pattern,text,re.I)
            return float(match.group(1)) if match else default


        age = find_value(r'age\s*[:\-]?\s*(\d+)',text,50)
        trestbps = find_value(r'pressure\s*[:\-]?\s*(\d+)',text,120)
        chol = find_value(r'cholesterol\s*[:\-]?\s*(\d+)',text,200)
        thalach = find_value(r'heart\s*rate\s*[:\-]?\s*(\d+)',text,150)


        #  VALIDATE EXTRACTION 
        if age is None or trestbps is None or chol is None:

            st.error("⚠️ Could not extract required medical values. Upload clearer report.")

            st.stop()


        # default values for others
        sex=1
        cp=1
        fbs=0
        restecg=1
        exang=0
        oldpeak=1
        slope=1
        ca=0
        thal=2


        patient = {
            "age":age,
            "sex":sex,
            "cp":cp,
            "trestbps":trestbps,
            "chol":chol,
            "fbs":fbs,
            "restecg":restecg,
            "thalach":thalach,
            "exang":exang,
            "oldpeak":oldpeak,
            "slope":slope,
            "ca":ca,
            "thal":thal
        }

        patient_df = pd.DataFrame([patient])

    else:
        st.stop()


#  Ensure Feature Order
patient_df = patient_df[rf_model.feature_names_in_]


#  Prediction 
if st.button("Predict Cardiovascular Risk"):

    prediction = rf_model.predict(patient_df)[0]
    probability = rf_model.predict_proba(patient_df)[0][1]

    risk_text = "High Cardiovascular Risk" if prediction==1 else "Low Cardiovascular Risk"

    st.subheader("Prediction Result")

    st.write("Prediction:",risk_text)
    st.write("Confidence:",round(probability*100,2),"%")


    #  PIE CHART 
    st.subheader("Risk Probability")

    fig,ax=plt.subplots()

    ax.pie(
        [1-probability,probability],
        labels=["Low Risk","High Risk"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)


    # -------- FEATURE IMPORTANCE --------
    st.subheader("Feature Influence")

    importances = rf_model.feature_importances_
    features = rf_model.feature_names_in_

    df = pd.DataFrame({
        "Feature":features,
        "Importance":importances
    }).sort_values("Importance")

    fig2,ax2=plt.subplots()

    ax2.barh(df["Feature"],df["Importance"])

    st.pyplot(fig2)


    # -------- MEDICAL THRESHOLD COMPARISON --------
    st.subheader("Health Factors vs Normal Thresholds")

    thresholds = {
        "age": 50,
        "trestbps": 120,
        "chol": 200,
        "thalach": 150,
        "oldpeak": 1
    }

    labels = {
        "age": "Age",
        "trestbps": "Blood Pressure",
        "chol": "Cholesterol",
        "thalach": "Max Heart Rate",
        "oldpeak": "ST Depression"
    }

    normal_values = []
    patient_values = []
    factor_names = []

    for key in thresholds:
        normal_values.append(thresholds[key])
        patient_values.append(patient[key])
        factor_names.append(labels[key])

    fig3, ax3 = plt.subplots()

    x = range(len(factor_names))

    ax3.bar(x, normal_values, width=0.4, label="Normal")
    ax3.bar([i + 0.4 for i in x], patient_values, width=0.4, label="Patient")

    ax3.set_xticks([i + 0.2 for i in x])
    ax3.set_xticklabels(factor_names, rotation=30)

    ax3.set_ylabel("Measurement Value")
    ax3.set_title("Patient Health Indicators vs Normal Levels")

    ax3.legend()

    st.pyplot(fig3)


    # -------- PATIENT DATA TABLE --------
    st.subheader("Patient Data Summary")
    st.table(patient_df)


    # -------- LLM PROMPT --------
    prompt = f"""
A machine learning model predicted cardiovascular disease risk.

Prediction: {risk_text}
Confidence: {round(probability*100,2)}%

Patient data:
Age: {patient_df['age'][0]}
Blood Pressure: {patient_df['trestbps'][0]}
Cholesterol: {patient_df['chol'][0]}
Maximum Heart Rate: {patient_df['thalach'][0]}

Explain:
1. Why the patient may have cardiovascular risk
2. Which factors are important
3. Provide lifestyle advice
"""

    try:

        response = ollama.chat(
            model="llama3.2:1b",#ollama LLM 
            messages=[{"role":"user","content":prompt}]
        )

        explanation = response["message"]["content"]

        st.subheader("AI Explanation")

        st.write(explanation)

    except:

        st.warning("Ollama not running.")