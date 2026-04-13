"""
score.py

Scoring script for the AML Managed Online Endpoint.

AML calls two functions:
  init()      — runs once when the endpoint starts up; loads the model into memory
  run(data)   — called on every prediction request; returns the classification result

The endpoint receives a JSON payload with the 9 feature columns and returns
the predicted anomaly type and the model's confidence probability.
"""

import os
import json
import numpy as np
import lightgbm as lgb

# Loaded once at startup, reused for every request
model = None

# Must match FEATURE_COLS in train.py — order matters
FEATURE_COLS = [
    "forecast_bias",
    "forecast_accuracy",
    "volume_of_error",
    "pct_error",
    "weeks_affected",
    "prior_anomaly_count",
    "dc_region_encoded",
    "product_lifecycle_encoded",
    "is_seasonal_encoded",
]


def init():
    """Load model from the mounted model directory. Called once on endpoint startup."""
    global model
    # AZUREML_MODEL_DIR is set by AML to the directory containing the registered model files
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.txt")
    model = lgb.Booster(model_file=model_path)
    print(f"Model loaded from {model_path}")


def run(raw_data: str) -> str:
    """
    Handle a prediction request.

    Expected input JSON:
    {
        "forecast_bias": 0.85,
        "forecast_accuracy": 0.3,
        "volume_of_error": 0.7,
        "pct_error": 0.4,
        "weeks_affected": 6,
        "prior_anomaly_count": 2,
        "dc_region_encoded": 1,
        "product_lifecycle_encoded": 1,
        "is_seasonal_encoded": 0
    }

    Returns:
    {
        "prediction": "baseline_shift",
        "probability": 0.94
    }
    """
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}"})

    # Validate all required features are present
    missing = [col for col in FEATURE_COLS if col not in data]
    if missing:
        return json.dumps({"error": f"Missing required features: {missing}"})

    # Build feature vector in the correct column order
    features = np.array([[data[col] for col in FEATURE_COLS]])

    # Model outputs probability of class 1 (baseline_shift)
    prob = float(model.predict(features)[0])
    prediction = "baseline_shift" if prob >= 0.5 else "temporary"

    return json.dumps({
        "prediction": prediction,
        "probability": round(prob, 4),
    })
