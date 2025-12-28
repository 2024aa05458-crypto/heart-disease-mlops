import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn


# ------------------------------------------------------------------
# PATH SETUP (ROBUST & CI/CD SAFE)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "heart_cleaned.csv")

print("Loading data from:", DATA_PATH)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# FORCE BINARY TARGET (CRITICAL FOR UCI VARIANTS)
df["target"] = df["target"].apply(lambda x: 1 if int(x) > 0 else 0)
print("Target unique values:", df["target"].unique())

# ------------------------------------------------------------------
# SPLIT FEATURES & TARGET
# ------------------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

X.columns = X.columns.str.strip()

# ------------------------------------------------------------------
# AUTOMATIC FEATURE TYPE INFERENCE (ROBUST)
# ------------------------------------------------------------------
categorical_features = [col for col in X.columns if X[col].nunique() < 10]
numerical_features = [col for col in X.columns if X[col].nunique() >= 10]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# ------------------------------------------------------------------
# PREPROCESSING PIPELINES (WITH IMPUTATION)
# ------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------------
# TRAIN–TEST SPLIT
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------------
# MLFLOW SETUP
# ------------------------------------------------------------------
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("Heart Disease Classification")

# ------------------------------------------------------------------
# LOGISTIC REGRESSION EXPERIMENT
# ------------------------------------------------------------------
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

with mlflow.start_run(run_name="Logistic_Regression"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    log_reg_pipeline.fit(X_train, y_train)

    y_pred_lr = log_reg_pipeline.predict(X_test)
    y_prob_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred_lr)
    prec = precision_score(y_test, y_pred_lr, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred_lr, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_prob_lr)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(log_reg_pipeline, "logistic_regression_model")

    print("\nLogistic Regression Performance:")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("ROC-AUC  :", auc)

# ------------------------------------------------------------------
# RANDOM FOREST EXPERIMENT
# ------------------------------------------------------------------
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

with mlflow.start_run(run_name="Random_Forest"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)

    rf_pipeline.fit(X_train, y_train)

    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred_rf)
    prec = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_prob_rf)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(rf_pipeline, "random_forest_model")

    print("\nRandom Forest Performance:")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("ROC-AUC  :", auc)

# ------------------------------------------------------------------
# CROSS-VALIDATION (OUTSIDE MLFLOW RUNS)
# ------------------------------------------------------------------
cv_lr = cross_val_score(log_reg_pipeline, X, y, cv=5, scoring="roc_auc")
cv_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring="roc_auc")

print("\nLogistic Regression CV ROC-AUC:", cv_lr.mean())
print("Random Forest CV ROC-AUC:", cv_rf.mean())

import joblib

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_pipeline.joblib")
joblib.dump(rf_pipeline, MODEL_PATH)

print(f"Model saved at: {MODEL_PATH}")
