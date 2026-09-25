from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

try:
    from .feature_engineering import add_engineered_features
except ImportError:
    from feature_engineering import add_engineered_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "churn_model.joblib"
METADATA_PATH = ROOT / "model_metadata.json"
IMPORTANCE_PATH = ROOT / "feature_importance.csv"

app = FastAPI(
    title="European Bank Customer Churn API",
    description="Predictive modeling and churn risk scoring API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For college/demo use. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
metadata = {}


import os

def load_artifacts():
    global model, metadata
    try:
        print("=== DEBUG START ===")
        print(f"ROOT folder: {ROOT}")
        print(f"Files in ROOT: {os.listdir(ROOT)}")
        print(f"Checking: {MODEL_PATH}")
        print(f"Exists?: {MODEL_PATH.exists()}")
        
        if MODEL_PATH.exists():
            print(f"File size: {MODEL_PATH.stat().st_size} bytes")
            model = joblib.load(MODEL_PATH)
            print("MODEL LOADED OK!!!")
        else:
            print("MODEL FILE NOT FOUND - LIST FAILED")
            
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        print("=== DEBUG END ===")
    except Exception as e:
        print(f"!!! MODEL LOAD FAILED: {e}")
        import traceback
        traceback.print_exc()
        model = None
load_artifacts()

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    CreditScore: int = Field(..., ge=300, le=900)
    Geography: str
    Gender: str
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=20)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=10)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)


def risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def recommendations(data: CustomerInput, probability: float):
    recs = []

    if probability >= 0.60:
        recs.append("Prioritize this customer for a proactive retention campaign.")
    elif probability >= 0.30:
        recs.append("Monitor engagement and consider a personalized retention offer.")
    else:
        recs.append("Maintain regular engagement and relationship-building activities.")

    if data.IsActiveMember == 0:
        recs.append("Increase customer engagement through targeted digital or branch outreach.")

    if data.NumOfProducts <= 1:
        recs.append("Consider relevant cross-sell opportunities based on customer needs.")

    if data.Balance > 0 and data.EstimatedSalary > 0:
        ratio = data.Balance / data.EstimatedSalary
        if ratio > 1.5:
            recs.append("Review service quality and product value for this high-balance customer.")

    return recs


@app.get("/")
def home():
    return {
        "message": "European Bank Customer Churn API is running.",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "best_model": metadata.get("best_model"),
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Run backend/train_model.py first.",
        )

    row = pd.DataFrame([customer.model_dump()])
    row = add_engineered_features(row)

    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata.get("threshold", 0.50))
    churn_flag = int(probability >= threshold)

    return {
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(probability * 100, 2),
        "churn_flag": churn_flag,
        "risk_level": risk_band(probability),
        "threshold": threshold,
        "prediction": "Likely to Churn" if churn_flag else "Likely to Stay",
        "recommendations": recommendations(customer, probability),
    }


@app.get("/model-info")
def model_info():
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Train the model first.",
        )

    # Avoid returning the large classification report unless needed.
    compact_results = {}
    for name, metrics in metadata.get("results", {}).items():
        compact_results[name] = {
            k: v for k, v in metrics.items()
            if k in {"accuracy", "precision", "recall", "f1", "roc_auc"}
        }

    return {
        "best_model": metadata.get("best_model"),
        "selection_metric": metadata.get("selection_metric"),
        "threshold": metadata.get("threshold"),
        "results": compact_results,
    }


@app.get("/feature-importance")
def feature_importance(limit: int = 15):
    if not IMPORTANCE_PATH.exists():
        return {
            "available": False,
            "message": "Feature importance file is not available for the selected model."
        }

    df = pd.read_csv(IMPORTANCE_PATH).head(max(1, min(limit, 50)))
    return {
        "available": True,
        "items": df.to_dict(orient="records"),
    }


@app.post("/what-if")
def what_if(customer: CustomerInput):
    """
    Same scoring engine as /predict.
    The frontend can repeatedly call this endpoint after changing
    activity/product values to simulate scenarios.
    """
    return predict(customer)
