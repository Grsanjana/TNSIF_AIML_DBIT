from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, json, numpy as np, uvicorn

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    load_error = None
except Exception as e:
    model = scaler = feature_columns = None
    load_error = str(e)

class PredictionRequest(BaseModel):
    data: dict
@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.get("/health")
async def health():
    return {
        "status": "ok" if model else "error",
        "model_loaded": model is not None,
        "error": load_error,
        "features": feature_columns if feature_columns else None,
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not model or not scaler or not feature_columns:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    try:
        input_data = request.data
        values = [input_data.get(feat, 0) for feat in feature_columns]
        scaled_data = scaler.transform([values])
        prediction = model.predict(scaled_data)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
