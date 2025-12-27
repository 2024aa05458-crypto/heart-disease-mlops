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



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "heart_cleaned.csv")

print("Loading data from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
# FORCE binary target (safety for all UCI variants)


df.columns = df.columns.str.strip()
df["target"] = df["target"].apply(lambda x: 1 if int(x) > 0 else 0)

print("Target unique values:", df["target"].unique())

X = df.drop("target", axis=1)
y = df["target"]

X.columns = X.columns.str.strip()


categorical_features = [
    col for col in X.columns if X[col].nunique() < 10
]

numerical_features = [
    col for col in X.columns if X[col].nunique() >= 10
]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)


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


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_reg_pipeline.fit(X_train, y_train)

y_pred_lr = log_reg_pipeline.predict(X_test)
y_prob_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]


rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]


def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n{model_name} Performance:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print(
        "Precision:",
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    print(
        "Recall   :",
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    print("ROC-AUC  :", roc_auc_score(y_true, y_prob))

evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")

cv_lr = cross_val_score(log_reg_pipeline, X, y, cv=5, scoring="roc_auc")
cv_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring="roc_auc")

print("\nLogistic Regression CV ROC-AUC:", cv_lr.mean())
print("Random Forest CV ROC-AUC:", cv_rf.mean())

