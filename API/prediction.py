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

# Point directly to the current API folder where pipeline_artifact.joblib lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "pipeline_artifact.joblib")

_cached_artifacts: Dict[str, Any] = {}

def load_prediction_artifacts() -> Dict[str, Any]:
    """Loads operational transformation parameters and weights into thread memory."""
    global _cached_artifacts
    if _cached_artifacts:
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
    _cached_artifacts.clear()
    logger.info("Artifact memory cache invalidated successfully.")

def run_inference(input_data: Dict[str, Any]) -> float:
    """
    Ingests raw structural parameters, performs functional transforms,
    and runs a model forward pass.
    """
    artifacts = load_prediction_artifacts()
    
    # If your joblib payload is a full pipeline object (e.g. sklearn Pipeline):
    if not isinstance(artifacts, dict):
        df = pd.DataFrame([input_data])
        if "year" in df.columns:
            df["car_age"] = datetime.now().year - df["year"]
        raw_prediction = artifacts.predict(df)[0]
        return float(np.maximum(0.0, raw_prediction))

    # If your joblib payload is a dictionary containing individual components:
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    numerical_columns = artifacts["features"]["numerical"]
    categorical_columns = artifacts["features"]["categorical"]
    expected_features = artifacts["features"]["all_features"]

    df = pd.DataFrame([input_data])
    
    # Calculate car age if required by feature pipeline
    if "year" in df.columns:
        df["car_age"] = datetime.now().year - df["year"]
    
    # Feature transformations
    num_data = df[numerical_columns].copy()
    num_scaled = scaler.transform(num_data)
    df_scaled = pd.DataFrame(num_scaled, columns=numerical_columns)

    cat_data = df[categorical_columns].copy()
    cat_encoded = encoder.transform(cat_data)
    encoded_cols = encoder.get_feature_names_out(categorical_columns)
    df_encoded = pd.DataFrame(cat_encoded, columns=encoded_cols)

    # Matrix unification alignment
    final_features_df = pd.concat([df_scaled, df_encoded], axis=1)
    for col in expected_features:
        if col not in final_features_df.columns:
            final_features_df[col] = 0.0
            
    final_features_df = final_features_df[expected_features]
    
    raw_prediction = model.predict(final_features_df)[0]
    
    return float(np.maximum(0.0, raw_prediction))