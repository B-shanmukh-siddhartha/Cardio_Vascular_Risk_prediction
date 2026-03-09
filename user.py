import joblib
import pandas as pd
import ollama


# Load trained RandomForest model
rf_model = joblib.load("cvd_rf_model.pkl")


print("\nEnter Patient Details\n")


age = int(input("Age: "))
sex = int(input("Sex (1=Male, 0=Female): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol Level: "))
fbs = int(input("Fasting Blood Sugar (1=True, 0=False): "))
restecg = int(input("Rest ECG (0-2): "))
thalach = int(input("Maximum Heart Rate: "))
exang = int(input("Exercise Induced Angina (1=Yes, 0=No): "))
oldpeak = float(input("Oldpeak (ST Depression): "))
slope = int(input("Slope (0-2): "))
ca = int(input("Number of Major Vessels (0-3): "))
thal = int(input("Thal (1=Normal,2=Fixed defect,3=Reversible defect): "))


# Create patient dictionary
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


# Convert to dataframe
patient_df = pd.DataFrame([patient])

# Ensure correct feature order
patient_df = patient_df[rf_model.feature_names_in_]


# ML Prediction
prediction = rf_model.predict(patient_df)[0]
probability = rf_model.predict_proba(patient_df)[0][1]

risk_text = "High cardiovascular risk" if prediction == 1 else "Low cardiovascular risk"


# Prompt for Ollama
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
3. Give lifestyle advice
"""


# Ask Ollama
response = ollama.chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": prompt}]
)

explanation = response["message"]["content"]


# Final Output
print("\n==============================")
print("Cardiovascular Risk Prediction")
print("==============================")

print("\nPrediction:", risk_text)
print("Confidence:", round(probability*100,2), "%")

print("\nExplanation:\n")
print(explanation)