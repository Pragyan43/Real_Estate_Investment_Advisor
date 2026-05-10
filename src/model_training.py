import pandas as pd
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

# -----------------------------------
# Set MLflow Experiment
# -----------------------------------

mlflow.set_experiment("Real_Estate_Investment_Advisor")

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("data/cleaned_real_estate.csv")

print("\nDataset Loaded Successfully!\n")

print(df.head())

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------

categorical_columns = df.select_dtypes(include="object").columns

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nCategorical Encoding Completed!")

# -----------------------------------
# Features
# -----------------------------------

features = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Parking_Space"
]

# ===================================
# CLASSIFICATION MODEL
# ===================================

print("\n==============================")
print("TRAINING CLASSIFICATION MODEL")
print("==============================")

X_class = df[features]
y_class = df["Good_Investment"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run(run_name="Classification_Model"):

    classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    classifier.fit(X_train_c, y_train_c)

    y_pred_c = classifier.predict(X_test_c)

    accuracy = accuracy_score(y_test_c, y_pred_c)

    precision = precision_score(y_test_c, y_pred_c)

    recall = recall_score(y_test_c, y_pred_c)

    f1 = f1_score(y_test_c, y_pred_c)

    cm = confusion_matrix(y_test_c, y_pred_c)

    print("\n----- Classification Results -----")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:\n")
    print(cm)

    # -------------------------------
    # Log Metrics to MLflow
    # -------------------------------

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # -------------------------------
    # Log Parameters
    # -------------------------------

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    # -------------------------------
    # Log Model
    # -------------------------------

    mlflow.sklearn.log_model(
        classifier,
        "classification_model"
    )

# ===================================
# REGRESSION MODEL
# ===================================

print("\n==========================")
print("TRAINING REGRESSION MODEL")
print("==========================")

X_reg = df[features]
y_reg = df["Future_Price_5Y"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run(run_name="Regression_Model"):

    regressor = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    regressor.fit(X_train_r, y_train_r)

    y_pred_r = regressor.predict(X_test_r)

    mae = mean_absolute_error(y_test_r, y_pred_r)

    rmse = np.sqrt(
        mean_squared_error(y_test_r, y_pred_r)
    )

    r2 = r2_score(y_test_r, y_pred_r)

    print("\n----- Regression Results -----")

    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")

    # -------------------------------
    # Log Metrics
    # -------------------------------

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    # -------------------------------
    # Log Parameters
    # -------------------------------

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)

    # -------------------------------
    # Log Model
    # -------------------------------

    mlflow.sklearn.log_model(
        regressor,
        "regression_model"
    )

# ===================================
# FEATURE IMPORTANCE
# ===================================

print("\n====================")
print("FEATURE IMPORTANCE")
print("====================")

importance = classifier.feature_importances_

for feature, score in zip(features, importance):
    print(f"{feature} : {score:.4f}")

# ===================================
# SAVE MODELS
# ===================================

pickle.dump(
    classifier,
    open("models/classification_model.pkl", "wb")
)

pickle.dump(
    regressor,
    open("models/regression_model.pkl", "wb")
)

print("\nModels Saved Successfully!")

print("\nProject Training Completed Successfully!")