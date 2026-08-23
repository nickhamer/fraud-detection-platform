from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Dict
import joblib, json


app = FastAPI()

BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"
bundle = joblib.load(BUNDLE_PATH)

model             = bundle["model"]
feature_order     = bundle["feature_order"]
category_levels   = bundle["category_levels"]
pr_curve          = bundle["pr_curve"]
default_threshold = bundle["default_threshold"]

class PredictionData(BaseModel):
    TransactionAmt: float
    TransactionDT: float
    ProductCD: str
    
    features: Dict[str, str | float]


@app.post("/predict")
def predict(prediction_data: PredictionData):

    features = [[
        transaction.Time,
        transaction.Amount
    ]]

    probability = model.predict_proba(features)[0][1]

    return {
        "fraud_probability": float(probability)
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
