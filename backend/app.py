# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import torch
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os


from .dkn_model import DKN, preprocess


MODEL_PATH = os.path.join(os.path.dirname(__file__), "dkn_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"                


app = FastAPI(title="DKN API")


origins = [
    "https://stingersonx228.github.io",
    "https://stingersonx228.github.io/DKN-Site",
    "https://dkn-newback.onrender.com",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    features: Dict[str, float]   # {"age":45, "sex":1, ...} вроде так :)

class PredictOut(BaseModel):
    probability: float
    raw: Dict[str, Any] = {}


print("Loading model...")
model_data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if isinstance(model_data, dict) and "model_state" in model_data:
    fit_objs = model_data.get("fit_objs", None)
    numeric_cols = model_data.get("numeric_cols", [])
    categorical_cols = model_data.get("categorical_cols", [])
    if categorical_cols and fit_objs and fit_objs.get("ohe") is not None:
        try:
            cat_dims = len(fit_objs["ohe"].get_feature_names_out(categorical_cols))
        except Exception:
            cat_dims = sum(len(xs) for xs in fit_objs["ohe"].categories_)
        input_dim = len(numeric_cols) + cat_dims
    else:
        input_dim = len(numeric_cols)

    model = DKN(input_dim=input_dim)
    model.load_state_dict(model_data["model_state"], strict=False)
    model.eval()
    print(f"Model loaded. Expecting {input_dim} features.")
else:
    
    model = model_data
    fit_objs = None
    numeric_cols = []
    categorical_cols = []
    print("Loaded full model object (no fit_objs present).")

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    try:
        vals = payload.features
        
        if fit_objs is not None and numeric_cols:
            
            row = {}
            for c in numeric_cols + categorical_cols:
                row[c] = vals.get(c, np.nan)
            df = pd.DataFrame([row])
            X, _ = preprocess(df, numeric_cols, categorical_cols, fit_objects=fit_objs)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().ravel()[0]
            return {"probability": float(pred), "raw": {"features_used": numeric_cols + categorical_cols}}
        else:
            if "vector" not in vals:
                raise HTTPException(status_code=400, detail="No preprocessors available; send {'vector':[...]} or provide model with fit_objs.")
            vector = np.array(vals["vector"], dtype=float).reshape(1, -1)
            X_tensor = torch.tensor(vector, dtype=torch.float32)
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().ravel()[0]
            return {"probability": float(pred), "raw": {"vector_len": vector.shape[1]}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def root():
    return {"status": "ok", "message": "DKN API is running"}

