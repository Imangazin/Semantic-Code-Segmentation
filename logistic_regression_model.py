import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load BOC data
boc_data_path = "data/boc_data.json"
with open(boc_data_path, "r", encoding="utf-8") as f:
    boc_data = json.load(f)

# Convert BOC data into NumPy arrays
X_boc = np.array([entry["vector"] for entry in boc_data])
y_boc = np.array([entry["label"] for entry in boc_data])

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_boc, y_boc, test_size=0.2, random_state=42, stratify=y_boc)

# Train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)

# Save trained model
model_path = "data/logistic_regression_model.pkl"
joblib.dump(log_reg_model, model_path)

print(f"Trained model saved to {model_path}")

# Make predictions
y_pred = log_reg_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print results
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)

# Save classification report to file
report_path = "data/logistic_regression_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)

print(f"Classification report saved to {report_path}")
