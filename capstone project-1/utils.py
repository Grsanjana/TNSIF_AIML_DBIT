import pickle
import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Load the model, scaler, and feature columns
def load_assets():
    model_path = MODEL_DIR / "linear_regression_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    features_path = MODEL_DIR / "feature_columns.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(features_path, "r") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


def make_prediction(model, scaler, feature_columns, input_data: dict):
    try:
        # Ensure order of features matches training
        X = np.array([[input_data[feature] for feature in feature_columns]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
