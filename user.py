import joblib
import pandas as pd
import ollama


# -------- Input Validation Function --------
def get_input(prompt, dtype=int, min_val=None, max_val=None):
    while True:
        try:
            value = input(prompt).strip()

            if value == "":
                print("Value cannot be empty. Please try again.")
                continue

            value = dtype(value)

            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter the correct type.")


# -------- Load ML Model --------
try:
    rf_model = joblib.load("cvd_rf_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    exit()


print("\nEnter Patient Details\n")


# -------- Safe User Inputs --------
age = get_input("Age: ", int, 1, 120)
sex = get_input("Sex (1=Male, 0=Female): ", int, 0, 1)
cp = get_input("Chest Pain Type (0-3): ", int, 0, 3)
trestbps = get_input("Resting Blood Pressure: ", int, 50, 250)
chol = get_input("Cholesterol Level: ", int, 50, 600)
fbs = get_input("Fasting Blood Sugar (1=True, 0=False): ", int, 0, 1)
restecg = get_input("Rest ECG (0-2): ", int, 0, 2)
thalach = get_input("Maximum Heart Rate: ", int, 40, 220)
exang = get_input("Exercise Induced Angina (1=Yes, 0=No): ", int, 0, 1)
oldpeak = get_input("Oldpeak (ST Depression): ", float, 0, 10)
slope = get_input("Slope (0-2): ", int, 0, 2)
ca = get_input("Number of Major Vessels (0-3): ", int, 0, 3)
thal = get_input("Thal (1=Normal,2=Fixed defect,3=Reversible defect): ", int, 1, 3)


# -------- Create Patient Data --------
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

# Ensure correct feature order
try:
    patient_df = patient_df[rf_model.feature_names_in_]
except Exception as e:
    print("Feature mismatch error:", e)
    exit()


# -------- ML Prediction --------
try:
    prediction = rf_model.predict(patient_df)[0]
    probability = rf_model.predict_proba(patient_df)[0][1]
except Exception as e:
    print("Prediction error:", e)
    exit()


risk_text = "High cardiovascular risk" if prediction == 1 else "Low cardiovascular risk"


# -------- LLM Prompt --------
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


# -------- Ask Ollama --------
try:
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    explanation = response["message"]["content"]

except Exception as e:
    explanation = "Could not generate explanation. Ensure Ollama is running."


# -------- Final Output --------
print("\n==============================")
print("Cardiovascular Risk Prediction")
print("==============================")

print("\nPrediction:", risk_text)
print("Confidence:", round(probability*100,2), "%")

print("\nExplanation:\n")
print(explanation)