import joblib
import pandas as pd
import ollama


rf_model = joblib.load("cvd_rf_model.pkl")



patient = {
    "age": 54,
    "sex": 1,
    "cp": 2,
    "trestbps": 148,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
}


# Convert to dataframe
patient_df = pd.DataFrame([patient])

# Ensure correct feature order
patient_df = patient_df[rf_model.feature_names_in_]


# -----------------------------
# ML Prediction
# -----------------------------
prediction = rf_model.predict(patient_df)[0]
probability = rf_model.predict_proba(patient_df)[0][1]

risk_text = "High cardiovascular risk" if prediction == 1 else "Low cardiovascular risk"


# -----------------------------
# Create prompt for LLM
# -----------------------------
prompt = f"""
You are a medical assistant.

A machine learning model predicted cardiovascular risk.

Prediction: {risk_text}
Confidence: {round(probability*100,2)}%

Patient information:
Age: {patient['age']}
Blood Pressure: {patient['trestbps']}
Cholesterol: {patient['chol']}
Max Heart Rate: {patient['thalach']}

Explain in simple human language:
1. Why the patient may have cardiovascular risk
2. Which factors are important
3. Give simple lifestyle advice
"""


# -----------------------------
# Ask Ollama
# -----------------------------
response = ollama.chat(
    model="tinyllama",
    messages=[{"role": "user", "content": prompt}]
)


explanation = response["message"]["content"]


# -----------------------------
# Output
# -----------------------------
print("\n==============================")
print("Cardiovascular Risk Prediction")
print("==============================")

print("\nPrediction:", risk_text)
print("Confidence:", round(probability*100,2), "%")

print("\nExplanation:\n")
print(explanation)