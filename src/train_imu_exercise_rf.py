# ================================================================
#  train_imu_exercise_rf.py
#  ---------------------------------------------------------------
#  ML Two — IMU-based Exercise Classifier (Windowed RandomForest)
#
#  Task:
#       Multi-class classification of rehabilitation exercises
#       using windowed IMU features.
#
#  Dataset:
#       PTExercises IMU Windows Dataset (Kaggle)
#
#  Outputs:
#       - imu_exercise_model.pkl
#       - imu_exercise_scaler.pkl
#       - imu_exercise_confusion_matrix.png
#       - imu_exercise_accuracy.png
#
#  Evaluation Metrics Computed:
#       - Accuracy
#       - Precision (weighted)
#       - Recall (weighted)
#       - F1-score (weighted)
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
    "subject", "exercise", "u_folder", "correctness",
    "file", "exercise_label", "correctness_label"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["exercise_label"]  # Multi-class exercise label

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
    "max_depth": [10, 20, None],
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
# Evaluation Metrics (Multi-class)
# ------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print("\n================ Evaluation Metrics ================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f} (weighted)")
print(f"Recall    : {recall:.4f} (weighted)")
print(f"F1-score  : {f1:.4f} (weighted)")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n", cm)

# ------------------------------------------------
# Save Confusion Matrix Image
# ------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("IMU Exercise Classifier – Confusion Matrix")
plt.tight_layout()
plt.savefig("imu_exercise_confusion_matrix.png")
plt.close()

# ------------------------------------------------
# Save Accuracy Bar Chart
# ------------------------------------------------
plt.figure(figsize=(5, 5))
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("IMU Exercise Classifier Accuracy")
plt.tight_layout()
plt.savefig("imu_exercise_accuracy.png")
plt.close()

print("\nSaved imu_exercise_confusion_matrix.png")
print("Saved imu_exercise_accuracy.png")

# ------------------------------------------------
# Save trained model and scaler
# ------------------------------------------------
joblib.dump(best_model, "imu_exercise_model.pkl")
joblib.dump(scaler, "imu_exercise_scaler.pkl")

print("\nSaved imu_exercise_model.pkl")
print("Saved imu_exercise_scaler.pkl")