from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, Union
from pathlib import Path
import joblib, json


app = FastAPI()

BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"
bundle = joblib.load(BUNDLE_PATH)

model             = bundle["model"]
feature_order     = bundle["feature_order"]
category_levels   = bundle["category_levels"]
pr_curve          = bundle["pr_curve"]
default_threshold = bundle["default_threshold"]

class Transaction(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    ProductCD: str = Field(..., min_length=1, max_length=4)
    features: Dict[str, Union[float, str, None]] = Field(default_factory=dict)

    @field_validator("TransactionAmt")
    @classmethod
    def finite_amount(cls, v):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("TransactionAmt must be finite")
        return v

    @field_validator("features")
    @classmethod
    def known_features(cls, v):
        unknown = set(v) - set(feature_order)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)[:5]}")
        return v


class PredictionRequest(BaseModel):
    transaction: Transaction
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


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
