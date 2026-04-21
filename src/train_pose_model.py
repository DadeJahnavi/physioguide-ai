# ================================================================
#  train_pose_model.py
#  ---------------------------------------------------------------
#  Vision-based Exercise Classifier (Pose Model)
#
#  Task:
#       Multi-class classification of exercises using vision-based
#       pose landmarks extracted from video frames.
#
#  Dataset:
#       Fitness Poses Dataset (CSV) – clean_pose.csv
#
#  Outputs:
#       - pose_model.pkl
#       - pose_label_encoder.pkl
#       - pose_confusion_matrix.png
#       - pose_accuracy.png
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
#       2. Encode pose labels
#       3. Stratified train-test split
#       4. Train RandomForest classifier
#       5. Evaluate model performance (all metrics)
#       6. Save evaluation plots
#       7. Save trained model and label encoder
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
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
INPUT = Path("data/clean_pose.csv")
df = pd.read_csv(INPUT)
print("Loaded clean_pose.csv:", df.shape)

# ------------------------------------------------
# Prepare features and labels
# ------------------------------------------------
y_cat = df["label"].astype("category")
X = df.drop(columns=["label"])

# Encode labels
label_encoder = dict(enumerate(y_cat.cat.categories))
print("Pose classes:", list(y_cat.cat.categories))

y = y_cat.cat.codes  # numeric labels

# ------------------------------------------------
# Train-test split (stratified)
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# Train RandomForest model
# ------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ------------------------------------------------
# Prediction
# ------------------------------------------------
y_pred = model.predict(X_test)

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
plt.title("Pose Model – Confusion Matrix")
plt.tight_layout()
plt.savefig("pose_confusion_matrix.png")
plt.close()

# ------------------------------------------------
# Save Accuracy Bar Chart
# ------------------------------------------------
plt.figure(figsize=(5, 5))
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("Pose Model Accuracy")
plt.tight_layout()
plt.savefig("pose_accuracy.png")
plt.close()

print("\nSaved pose_confusion_matrix.png")
print("Saved pose_accuracy.png")

# ------------------------------------------------
# Save trained model and label encoder
# ------------------------------------------------
joblib.dump(model, "pose_model.pkl")
joblib.dump(label_encoder, "pose_label_encoder.pkl")

print("\nSaved pose_model.pkl")
print("Saved pose_label_encoder.pkl")
