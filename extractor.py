import pandas as pd
import joblib

from info_extractor import extract_text, extract_values


image_path = input("Enter report image path: ")

text = extract_text(image_path)

values = extract_values(text)

print("\nExtracted Values:", values)


rf_model = joblib.load("cvd_rf_model.pkl")

patient_df = pd.DataFrame([values])

patient_df = patient_df[rf_model.feature_names_in_]


prediction = rf_model.predict(patient_df)[0]

probability = rf_model.predict_proba(patient_df)[0][1]


risk_text = "High cardiovascular risk" if prediction == 1 else "Low cardiovascular risk"


print("\n==============================")
print("Cardiovascular Risk Prediction")
print("==============================")

print("\nPrediction:", risk_text)

print("Confidence:", round(probability*100, 2), "%")