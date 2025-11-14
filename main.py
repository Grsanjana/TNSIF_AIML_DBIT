from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, json, numpy as np

# -------------------------------------------------------
# FASTAPI APP SETUP
# -------------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")

# Enable CORS (so Streamlit frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your Streamlit URL when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# LOAD MODEL, SCALER & FEATURES
# -------------------------------------------------------
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    # Ensure only first 13 features (if target got saved accidentally)
    if len(feature_columns) > 13:
        feature_columns = feature_columns[:13]

    load_error = None
except Exception as e:
    model = scaler = feature_columns = None
    load_error = str(e)

# -------------------------------------------------------
# REQUEST MODEL
# -------------------------------------------------------
class PredictionRequest(BaseModel):
    data: dict

# -------------------------------------------------------
# ROOT & HEALTH ENDPOINTS
# -------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "✅ Heart Disease Prediction API is running!"}

@app.get("/health")
async def health():
    return {
        "status": "ok" if model else "error",
        "model_loaded": model is not None,
        "error": load_error,
        "features": feature_columns if feature_columns else None,
    }

# -------------------------------------------------------
# PREDICTION ENDPOINT
# -------------------------------------------------------
@app.post("/predict")
async def predict(request: PredictionRequest):
    if not model or not scaler or not feature_columns:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    try:
        input_data = request.data

        # Arrange inputs in the correct order
        values = [input_data.get(feat, 0) for feat in feature_columns]

        # Validate feature count
        if len(values) != len(feature_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_columns)} features, got {len(values)}"
            )

        # Scale and predict
        scaled_data = scaler.transform([values])
        prediction = int(model.predict(scaled_data)[0])  # 0 or 1

        # Try to get probability (if supported)
        try:
            probability = float(model.predict_proba(scaled_data)[0][1])
        except Exception:
            probability = None

        return {"prediction_value": prediction, "probability": probability}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
