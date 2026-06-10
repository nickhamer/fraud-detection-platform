from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib, json

app = FastAPI()

model = joblib.load("../models/fraud_model.joblib")
feature_types = json.load("../models/fraud_model_feature_types.json")

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
