from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
import logging
import time

# ---------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts risk of heart disease from patient data",
    version="1.0"
)

# ---------------------------------------------------
# REQUEST LOGGING MIDDLEWARE
# ---------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"Status={response.status_code} "
        f"Time={process_time:.4f}s"
    )
    return response

# ---------------------------------------------------
# INPUT SCHEMA
# ---------------------------------------------------
class PatientData(BaseModel):
    id: int
    age: int
    sex: int
    dataset: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalch: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# ---------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------
@app.post("/predict")
def predict(data: PatientData):

    input_dict = {
        "id": data.id,
        "age": data.age,
        "sex": data.sex,
        "dataset": data.dataset,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalch": data.thalch,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "heart_disease_risk": int(prediction),
        "confidence": round(float(probability), 4)
    }

# ---------------------------------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}
