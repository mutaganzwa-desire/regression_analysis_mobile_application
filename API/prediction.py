"""
Prediction pipeline execution script. Handles live pipeline decompression 
and deterministic forward-pass transformation executions.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "pipeline_artifact.joblib")

_cached_artifacts: Any = None

def load_prediction_artifacts() -> Any:
    """Loads operational transformation parameters into thread memory."""
    global _cached_artifacts
    if _cached_artifacts is not None:
        return _cached_artifacts

    if not os.path.exists(ARTIFACT_PATH):
        logger.error(f"Operational artifacts missing at path target: {ARTIFACT_PATH}")
        raise FileNotFoundError(f"Artifact payload not localized at {ARTIFACT_PATH}")

    try:
        _cached_artifacts = joblib.load(ARTIFACT_PATH)
        logger.info("Successfully bound deployment artifacts into execution context.")
        return _cached_artifacts
    except Exception as e:
        logger.critical(f"Failed to extract deployment artifact serialization: {str(e)}")
        raise e

def invalidate_artifact_cache() -> None:
    """Clears thread memory cache to force hot reloading after retraining."""
    global _cached_artifacts
    _cached_artifacts = None
    logger.info("Artifact memory cache invalidated successfully.")

def run_inference(input_data: Dict[str, Any]) -> float:
    """
    Ingests raw structural parameters, performs functional transforms,
    and runs a model forward pass.
    """
    artifacts = load_prediction_artifacts()
    
    # 1. Extract raw model estimator safely
    if isinstance(artifacts, dict):
        model = artifacts.get("model", artifacts.get("pipeline", next(iter(artifacts.values()))))
        scaler = artifacts.get("scaler")
        encoder = artifacts.get("encoder")
        
        # Safely extract features dictionary if it exists
        features_dict = artifacts.get("features", {})
        feature_names = features_dict.get("all_features") if isinstance(features_dict, dict) else None
    else:
        model = artifacts
        scaler = None
        encoder = None
        feature_names = None

    # 2. Check model's intrinsic feature names if available
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    # 3. Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Year -> Car Age derivation if needed
    current_year = datetime.now().year
    if "model_year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = current_year - df["model_year"]
    elif "year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = current_year - df["year"]

    # 4. Perform Encoding & Scaling if fitted transformers exist in artifacts
    if encoder is not None and scaler is not None and isinstance(features_dict, dict) and "numerical" in features_dict:
        num_cols = features_dict["numerical"]
        cat_cols = features_dict["categorical"]
        
        num_scaled = scaler.transform(df[num_cols])
        df_num = pd.DataFrame(num_scaled, columns=num_cols)

        cat_encoded = encoder.transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=encoded_cols)

        df_final = pd.concat([df_num, df_cat], axis=1)
    else:
        # Fallback One-Hot Encoding
        df_final = pd.get_dummies(df, drop_first=False)

    # 5. Align feature matrix shape and feature names
    if feature_names is not None:
        for col in feature_names:
            if col not in df_final.columns:
                df_final[col] = 0.0
        df_final = df_final[feature_names]
    elif hasattr(model, "n_features_in_"):
        expected_n = model.n_features_in_
        current_n = df_final.shape[1]
        if current_n < expected_n:
            for i in range(expected_n - current_n):
                df_final[f"dummy_feature_{i}"] = 0.0

    # 6. Perform prediction
    raw_prediction = model.predict(df_final)[0]
    
    return float(np.maximum(0.0, raw_prediction))