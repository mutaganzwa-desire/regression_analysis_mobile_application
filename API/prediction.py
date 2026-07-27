"""
Prediction pipeline execution script. Handles live pipeline decompression 
and deterministic forward-pass transformation executions.
"""

import os
import logging
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "linear_regression", "models"))
ARTIFACT_PATH = os.path.join(MODEL_DIR, "production_artifacts.joblib")

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
    
    # 1. Convert input dictionary into a 1-row Pandas DataFrame
    df = pd.DataFrame([input_data])
    
    # 2. Derive engineered features if needed by the model
    current_year = datetime.now().year
    if "model_year" in df.columns:
        df["car_age"] = current_year - df["model_year"]
    elif "year" in df.columns:
        df["car_age"] = current_year - df["year"]

    # 3. Handle object or dictionary payload
    if not isinstance(artifacts, dict):
        model = artifacts
    elif "pipeline" in artifacts:
        model = artifacts["pipeline"]
    elif "model" in artifacts:
        model = artifacts["model"]
    else:
        # Fallback: take the first value stored in the dictionary
        model = list(artifacts.values())[0]

    # 4. Check if manual feature decomposition is needed
    if isinstance(artifacts, dict) and "features" in artifacts:
        scaler = artifacts.get("scaler")
        encoder = artifacts.get("encoder")
        numerical_columns = artifacts["features"].get("numerical", [])
        categorical_columns = artifacts["features"].get("categorical", [])
        expected_features = artifacts["features"].get("all_features", [])

        if scaler and encoder:
            num_data = df[numerical_columns].copy()
            num_scaled = scaler.transform(num_data)
            df_scaled = pd.DataFrame(num_scaled, columns=numerical_columns)

            cat_data = df[categorical_columns].copy()
            cat_encoded = encoder.transform(cat_data)
            encoded_cols = encoder.get_feature_names_out(categorical_columns)
            df_encoded = pd.DataFrame(cat_encoded, columns=encoded_cols)

            final_features_df = pd.concat([df_scaled, df_encoded], axis=1)
            for col in expected_features:
                if col not in final_features_df.columns:
                    final_features_df[col] = 0.0
            
            df = final_features_df[expected_features]

    # 5. Execute prediction pass
    raw_prediction = model.predict(df)[0]
    
    return float(np.maximum(0.0, raw_prediction))