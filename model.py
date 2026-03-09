import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from info_extractor import extract_text, extract_values

image_path = input("Enter report image path: ")

text = extract_text(image_path)

values = extract_values(text)

print(values)


# Load dataset
df = pd.read_csv("../datasets/Cardio_synthetic.csv")


# Remove duplicates
df = df.drop_duplicates()


# Fill missing values
df = df.fillna(df.mean(numeric_only=True))


# Check class distribution
print("Class Distribution:\n", df["target"].value_counts())


# Split features and target
X = df.drop("target", axis=1)
y = df["target"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)


# Train model
rf_model.fit(X_train_res, y_train_res)


# Predictions
y_pred = rf_model.predict(X_test)


# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save model
joblib.dump(rf_model, "cvd_rf_model.pkl")

print("\nModel saved successfully!")