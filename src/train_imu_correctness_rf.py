# ================================================================
#  train_imu_correctness_rf.py
#  ---------------------------------------------------------------
#  ML Three — IMU-based Correctness Classifier (Windowed RF)
#
#  Task:
#       Binary classification of exercise execution correctness
#       (Correct vs Incorrect) using windowed IMU features.
#
#  Dataset:
#       PTExercises IMU Windows Dataset (Kaggle)
#
#  Outputs:
#       - imu_correctness_model.pkl
#       - imu_correctness_scaler.pkl
#       - imu_correctness_confusion_matrix.png
#       - imu_correctness_accuracy_pie.png
#
#  Evaluation Metrics Computed:
#       - Accuracy
#       - Precision
#       - Recall
#       - F1-score
#       - Confusion Matrix
#
#  Workflow:
#       1. Load dataset
#       2. Drop metadata columns
#       3. Clean NaN / Inf values
#       4. Feature scaling
#       5. Stratified train-test split
#       6. Random Forest + GridSearchCV
#       7. Model evaluation (all metrics)
#       8. Save evaluation plots
#       9. Save trained model and scaler
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# ------------------------------------------------
# Load dataset
# ------------------------------------------------
df = pd.read_csv("data/imu_windows.csv")
print("Loaded IMU dataset:", df.shape)

# ------------------------------------------------
# Feature selection
# ------------------------------------------------
drop_cols = [
    "subject", "exercise", "u_folder",
    "correctness", "file", "exercise_label"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["correctness_label"]  # Binary label: 0 = incorrect, 1 = correct

# ------------------------------------------------
# Data cleaning
# ------------------------------------------------
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

print("Cleaned dataset shape:", X.shape)

# ------------------------------------------------
# Feature scaling
# ------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------
# Train-test split (stratified)
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# Model definition + hyperparameter tuning
# ------------------------------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [10, 20, None]
}

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid.best_params_)

best_model = grid.best_estimator_

# ------------------------------------------------
# Prediction
# ------------------------------------------------
y_pred = best_model.predict(X_test)

# ------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")
cm = confusion_matrix(y_test, y_pred)

print("\n================ Evaluation Metrics ================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n", cm)

# ------------------------------------------------
# Save Confusion Matrix Image
# ------------------------------------------------
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Incorrect", "Correct"],
    yticklabels=["Incorrect", "Correct"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("IMU Correctness Classifier – Confusion Matrix")
plt.tight_layout()
plt.savefig("imu_correctness_confusion_matrix.png")
plt.close()

# ------------------------------------------------
# Save Accuracy Pie Chart
# ------------------------------------------------
plt.figure(figsize=(5, 5))
plt.pie(
    [accuracy, 1 - accuracy],
    labels=[
        f"Correct Predictions ({accuracy:.2%})",
        f"Incorrect Predictions ({1 - accuracy:.2%})"
    ],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("IMU Correctness Classifier – Accuracy Breakdown")
plt.tight_layout()
plt.savefig("imu_correctness_accuracy_pie.png")
plt.close()

print("\nSaved imu_correctness_confusion_matrix.png")
print("Saved imu_correctness_accuracy_pie.png")

# ------------------------------------------------
# Save trained model and scaler
# ------------------------------------------------
joblib.dump(best_model, "imu_correctness_model.pkl")
joblib.dump(scaler, "imu_correctness_scaler.pkl")

print("\nSaved imu_correctness_model.pkl")
print("Saved imu_correctness_scaler.pkl")