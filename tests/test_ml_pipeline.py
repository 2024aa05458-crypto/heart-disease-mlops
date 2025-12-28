import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "heart_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_pipeline.joblib")


def test_data_processing():
    """
    Test that the cleaned dataset loads correctly
    and target values follow UCI specification (0–4).
    """
    assert os.path.exists(DATA_PATH), "Processed dataset not found"

    df = pd.read_csv(DATA_PATH)

    assert "target" in df.columns, "Target column missing"

    # UCI heart disease dataset target values range from 0 to 4
    assert df["target"].min() >= 0, "Target has invalid negative values"
    assert df["target"].max() <= 4, "Target has invalid values > 4"



def test_model_inference():
    """
    Test that the trained model pipeline can
    generate predictions on a sample input.
    """
    assert os.path.exists(MODEL_PATH), "Trained model not found"

    model = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame([{
        "id": 1,
        "age": 55,
        "sex": 1,
        "dataset": 0,
        "cp": 2,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalch": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }])

    prediction = model.predict(sample_input)

    assert prediction.shape == (1,), "Model prediction failed"
