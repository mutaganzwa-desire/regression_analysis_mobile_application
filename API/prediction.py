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
    
    # 1. Unpack artifacts dynamically
    if isinstance(artifacts, dict):
        model = artifacts.get("model", artifacts.get("pipeline", next(iter(artifacts.values()))))
        scaler = artifacts.get("scaler")
        encoder = artifacts.get("encoder")
        features_dict = artifacts.get("features", {}) if isinstance(artifacts.get("features"), dict) else {}
    else:
        model = artifacts
        scaler = None
        encoder = None
        features_dict = {}

    df = pd.DataFrame([input_data])
    
    # Year -> Car Age derivation
    current_year = datetime.now().year
    if "model_year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = current_year - df["model_year"]
    elif "year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = current_year - df["year"]

    # 2. Extract fitted transformers if available
    if encoder is not None and scaler is not None and "numerical" in features_dict and "categorical" in features_dict:
        num_cols = features_dict["numerical"]
        cat_cols = features_dict["categorical"]
        
        num_scaled = scaler.transform(df[num_cols])
        df_num = pd.DataFrame(num_scaled, columns=num_cols)

        cat_encoded = encoder.transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=encoded_cols)

        df_final = pd.concat([df_num, df_cat], axis=1)
    else:
        # Fallback categorical encoding
        df_final = pd.get_dummies(df, drop_first=False)

    # 3. Align feature matrix shape to match expected model input
    feature_names = features_dict.get("all_features")
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if feature_names is not None:
        for col in feature_names:
            if col not in df_final.columns:
                df_final[col] = 0.0
        df_final = df_final[feature_names]
        X_input = df_final.values
    else:
        # Pad numeric matrix directly to match model.n_features_in_
        X_input = df_final.select_dtypes(include=[np.number]).values
        expected_n = getattr(model, "n_features_in_", 67)
        current_n = X_input.shape[1]
        
        if current_n < expected_n:
            padding = np.zeros((X_input.shape[0], expected_n - current_n))
            X_input = np.hstack([X_input, padding])
        elif current_n > expected_n:
            X_input = X_input[:, :expected_n]

    # 4. Predict
    raw_prediction = model.predict(X_input)[0]
    logger.info(f"Raw Model Prediction Output: {raw_prediction}")
    
    return float(np.maximum(0.0, raw_prediction))