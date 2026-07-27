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
        features_dict = artifacts.get("features", {})
    else:
        model = artifacts
        scaler = None
        encoder = None
        features_dict = {}

    df = pd.DataFrame([input_data])
    
    # Derivations
    current_year = datetime.now().year
    if "model_year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = current_year - df["model_year"]
    elif "year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = current_year - df["year"]

    # 2. Extract fitted transformers if available
    if encoder is not None and scaler is not None and isinstance(features_dict, dict) and "numerical" in features_dict:
        num_cols = features_dict["numerical"]
        cat_cols = features_dict["categorical"]
        
        # Scale numericals
        num_scaled = scaler.transform(df[num_cols])
        df_num = pd.DataFrame(num_scaled, columns=num_cols)

        # Encode categoricals
        cat_encoded = encoder.transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=encoded_cols)

        df_final = pd.concat([df_num, df_cat], axis=1)
        
        if "all_features" in features_dict:
            for col in features_dict["all_features"]:
                if col not in df_final.columns:
                    df_final[col] = 0.0
            df_final = df_final[features_dict["all_features"]]
    else:
        # Fallback processing if artifacts dictionary doesn't contain separate preprocessors
        df_final = pd.get_dummies(df, drop_first=False)
        
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            for col in expected_cols:
                if col not in df_final.columns:
                    df_final[col] = 0.0
            df_final = df_final[expected_cols]

    # 3. Predict valuation
    raw_prediction = model.predict(df_final)[0]
    
    # Log raw output to Render logs so we can monitor accuracy
    logger.info(f"Raw Model Prediction: {raw_prediction}")
    
    # Return formatted valuation
    return float(np.maximum(0.0, raw_prediction))