import streamlit as st
import pandas as pd
import joblib
import ollama
import matplotlib.pyplot as plt


# -------- Load ML Model --------
@st.cache_resource
def load_model():
    return joblib.load("cvd_rf_model.pkl")

rf_model = load_model()


# -------- Title --------
st.title("🫀 Cardiovascular Risk Prediction System")
st.write("Enter patient medical details to predict cardiovascular disease risk.")


# -------- Sidebar Inputs --------
st.sidebar.header("Patient Medical Details")

age = st.sidebar.slider("Age", 1, 120, 50)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])

trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 250, 120)

chol = st.sidebar.slider("Cholesterol Level", 100, 600, 200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dl", ["False","True"])
fbs = 1 if fbs == "True" else 0

restecg = st.sidebar.selectbox("Rest ECG", [0,1,2])

thalach = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)

exang = st.sidebar.selectbox("Exercise Induced Angina", ["No","Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 10.0, 1.0)

slope = st.sidebar.selectbox("Slope", [0,1,2])

ca = st.sidebar.selectbox("Major Vessels (0-3)", [0,1,2,3])

thal = st.sidebar.selectbox("Thal", [1,2,3])


# -------- Patient Data --------
patient = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

patient_df = pd.DataFrame([patient])
patient_df = patient_df[rf_model.feature_names_in_]


# -------- Prediction Button --------
if st.button("Predict Cardiovascular Risk"):

    prediction = rf_model.predict(patient_df)[0]
    probability = rf_model.predict_proba(patient_df)[0][1]

    risk_text = "High Cardiovascular Risk" if prediction == 1 else "Low Cardiovascular Risk"

    st.subheader("Prediction Result")
    st.write("Prediction:", risk_text)
    st.write("Confidence:", round(probability*100,2), "%")


    # -------- PIE CHART --------
    st.subheader("Risk Probability")

    low = 1 - probability
    high = probability

    fig1, ax1 = plt.subplots()

    ax1.pie(
        [low, high],
        labels=["Low Risk", "High Risk"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax1.axis("equal")

    st.pyplot(fig1)


    # -------- FEATURE IMPORTANCE CHART --------
    st.subheader("Factors Influencing Risk Prediction")

    try:

        importances = rf_model.feature_importances_
        features = rf_model.feature_names_in_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()

        ax2.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax2.set_xlabel("Importance Score")
        ax2.set_title("Feature Influence on Prediction")

        st.pyplot(fig2)

    except:
        st.warning("Feature importance chart unavailable.")


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


    # -------- LLM Prompt (YOUR ORIGINAL PROMPT) --------
    prompt = f"""
A machine learning model predicted cardiovascular disease risk.

Prediction: {risk_text}
Confidence: {round(probability*100,2)}%

Patient data:
Age: {age}
Blood Pressure: {trestbps}
Cholesterol: {chol}
Maximum Heart Rate: {thalach}

Explain:
1. Why the patient may have cardiovascular risk
2. Which factors are important
3. Provide lifestyle advice
"""

    try:

        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response["message"]["content"]

        st.subheader("AI Explanation")
        st.write(explanation)

    except:
        st.warning("LLM explanation unavailable. Ensure Ollama is running.")