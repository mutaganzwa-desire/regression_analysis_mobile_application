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
    
    # 1. Prepare Pandas DataFrame input
    df = pd.DataFrame([input_data])
    
    current_year = datetime.now().year
    if "model_year" in df.columns:
        df["car_age"] = current_year - df["model_year"]
    elif "year" in df.columns:
        df["car_age"] = current_year - df["year"]

    # 2. Extract model / pipeline estimator safely
    model = None
    if not isinstance(artifacts, dict):
        model = artifacts
    elif "pipeline" in artifacts:
        model = artifacts["pipeline"]
    elif "model" in artifacts:
        model = artifacts["model"]
    else:
        # Fallback: take the first item in dictionary
        model = next(iter(artifacts.values()))

    # 3. Perform prediction
    try:
        raw_prediction = model.predict(df)[0]
    except Exception as err:
        logger.error(f"Prediction execution failed on model object: {str(err)}")
        raise err

    return float(np.maximum(0.0, raw_prediction))