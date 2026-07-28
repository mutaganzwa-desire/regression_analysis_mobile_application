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
    
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    model = artifacts["model"]
    num_cols = artifacts["numeric_features"]
    cat_cols = artifacts["categorical_features"]
    all_feature_names = artifacts["all_feature_names"]

    # 1. Standardize / Map Incoming Frontend Keys to Expecting Pipeline Columns
    raw_brand = input_data.get("brand", input_data.get("manufacturer_brand", "Toyota"))
    raw_year = input_data.get("model_year", input_data.get("year", 2018))
    raw_mileage = input_data.get("odometer", input_data.get("mileage", input_data.get("odometer_miles", 65000)))
    raw_fuel = input_data.get("fuel_type", input_data.get("combustible_fuel_type", "Gasoline"))
    raw_accident = input_data.get("accident", "None reported")

    # Calculate Car Age from Model Year
    try:
        year_val = float(raw_year)
        age_val = float(datetime.now().year - year_val)
    except (ValueError, TypeError):
        age_val = 6.0

    try:
        mileage_val = float(raw_mileage)
    except (ValueError, TypeError):
        mileage_val = 65000.0

    # Fuel type normalization to match trained categories ('Gasoline', 'Diesel', etc.)
    if "petrol" in str(raw_fuel).lower() or "gas" in str(raw_fuel).lower():
        norm_fuel = "Gasoline"
    elif "diesel" in str(raw_fuel).lower():
        norm_fuel = "Diesel"
    elif "hybrid" in str(raw_fuel).lower():
        norm_fuel = "Hybrid"
    else:
        norm_fuel = str(raw_fuel)

    # 2. Build Single-Row DataFrame matching fitted input
    df_raw = pd.DataFrame([{
        "Age": age_val,
        "Mileage": mileage_val,
        "Brand": str(raw_brand),
        "FuelType": norm_fuel,
        "Accident": str(raw_accident)
    }])

    # 3. Transform Numerical Features using Scaler
    num_scaled = scaler.transform(df_raw[num_cols])
    df_num = pd.DataFrame(num_scaled, columns=num_cols)

    # 4. Transform Categorical Features using OneHotEncoder
    cat_encoded = encoder.transform(df_raw[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    df_cat = pd.DataFrame(cat_encoded, columns=encoded_cols)

    # Combine Numerical + Categorical
    df_processed = pd.concat([df_num, df_cat], axis=1)

    # 5. Reindex to match the EXACT 67 features expected by LinearRegression
    df_final = df_processed.reindex(columns=all_feature_names, fill_value=0.0)

    # 6. Perform Prediction
    raw_prediction = model.predict(df_final)[0]
    logger.info(f"Calculated Valuation: {raw_prediction}")

    return float(np.maximum(1000.0, raw_prediction))