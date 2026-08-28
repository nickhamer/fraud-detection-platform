import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Union
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, confusion_matrix,
)
import joblib


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

class EvaluateRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=100_000)
    labels: List[int] = Field(..., min_length=1)
    target_precision: Optional[float] = Field(None, gt=0.0, lt=1.0)
    target_recall: Optional[float] = Field(None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def check(self):
        if len(self.labels) != len(self.transactions):
            raise ValueError("labels and transactions must be the same length")
        if set(self.labels) - {0, 1}:
            raise ValueError("labels must be 0 or 1")
        if sum(self.labels) == 0:
            raise ValueError("at least one positive label required; metrics undefined without them")
        if self.target_precision is not None and self.target_recall is not None:
            raise ValueError("specify at most one of target_precision / target_recall")
        return self


class PredictionRequest(BaseModel):
    transaction: Transaction
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    
class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=10_000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

def to_model_frame(txns: List[Transaction]) -> tuple[pd.DataFrame, list[int]]:
    rows = [{**t.features,
             "TransactionAmt": t.TransactionAmt,
             "ProductCD": t.ProductCD} for t in txns]

    df = pd.DataFrame(rows).reindex(columns=feature_order)

    for col, levels in category_levels.items():
        df[col] = pd.Categorical(df[col], categories=levels)

    n_missing = df.isna().sum(axis=1).astype(int).tolist()
    return df, n_missing


@app.post("/predict")
def predict(request: PredictionRequest):
    df, n_missing = to_model_frame([request.transaction])

    probability = float(model.predict_proba(df)[0][1])
    threshold = request.threshold if request.threshold is not None else default_threshold

    return {
        "fraud_probability": probability,
        "threshold": threshold,
        "is_fraud": probability >= threshold,
        "n_features_missing": n_missing[0],
    }

@app.post("/predict-batch")
def predict_batch(request: BatchRequest):
    df, n_missing = to_model_frame(request.transactions)

    probs = model.predict_proba(df)[:, 1]
    threshold = request.threshold if request.threshold is not None else default_threshold
    flags = probs >= threshold

    return {
        "threshold": threshold,
        "n_transactions": len(probs),
        "n_flagged": int(flags.sum()),
        "predictions": [
            {
                "fraud_probability": float(p),
                "is_fraud": bool(f),
                "n_features_missing": m,
            }
            for p, f, m in zip(probs, flags, n_missing)
        ],
    }

@app.get("/thresholds")
def thresholds(n_points: int = Query(20, ge=2, le=1000)):
    thr = np.asarray(pr_curve["thresholds"])
    p   = np.asarray(pr_curve["precision"])
    r   = np.asarray(pr_curve["recall"])

    grid = np.linspace(thr.min(), thr.max(), n_points)

    return {
        "thresholds": grid.tolist(),
        "precision": np.interp(grid, thr, p).tolist(),
        "recall":    np.interp(grid, thr, r).tolist(),
    }

def _threshold_for(curve, precision=None, recall=None):
    if (precision is None) == (recall is None):
        raise ValueError("specify exactly one of precision or recall")

    thr = np.asarray(curve["thresholds"])
    p   = np.asarray(curve["precision"])
    r   = np.asarray(curve["recall"])

    if precision is not None:
        hits = np.flatnonzero(p >= precision)
        if not len(hits):
            raise ValueError(f"precision {precision} unreachable (max {p.max():.3f})")
        i = hits[0]
    else:
        hits = np.flatnonzero(r >= recall)
        if not len(hits):
            raise ValueError(f"recall {recall} unreachable (max {r.max():.3f})")
        i = hits[-1]

    return {"threshold": float(thr[i]),
            "precision": float(p[i]),
            "recall": float(r[i])}


@app.get("/threshold-for")
def threshold_for(precision: float = None, recall: float = None):
    try:
        return _threshold_for(pr_curve, precision=precision, recall=recall)
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    df, _ = to_model_frame(request.transactions)
    y = np.array(request.labels)
    probs = model.predict_proba(df)[:, 1]

    p_arr, r_arr, t_arr = precision_recall_curve(y, probs)
    local_curve = {"thresholds": t_arr.tolist(),
                   "precision": p_arr[:-1].tolist(),
                   "recall": r_arr[:-1].tolist()}

    try:
        if request.target_precision is not None:
            op = _threshold_for(local_curve, precision=request.target_precision)
        elif request.target_recall is not None:
            op = _threshold_for(local_curve, recall=request.target_recall)
        else:
            op = {"threshold": default_threshold}
    except ValueError as e:
        raise HTTPException(422, str(e))


    thr = op["threshold"]
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "operating_point": {
            "threshold": float(thr),
            "requested_precision": request.target_precision,
            "requested_recall": request.target_recall,
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "n_flagged": int(preds.sum()),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "trained_at": bundle["trained_at"],
        "metrics": bundle["metrics"],
        "default_threshold": default_threshold,
        "n_features": len(feature_order),
        "features": feature_order,
        "categorical_features": list(category_levels),
        "fraud_prevalence_baseline": 0.034,
    }