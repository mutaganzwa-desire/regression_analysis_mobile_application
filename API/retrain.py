"""
Retraining engine subsystem. Executes shadow evaluation loops 
and manages atomic pipeline asset updates.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from summative.API.prediction import ARTIFACT_PATH, invalidate_artifact_cache

logger = logging.getLogger(__name__)

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "linear_regression", "dataset", "used_cars.csv"))

def execute_pipeline_retraining(uploaded_csv_path: str) -> Tuple[bool, float, float]:
    """
    Ingests update files, joins histories, updates pipeline baselines, 
    and determines deployment viability.
    """
    if not os.path.exists(DATASET_PATH):
        logger.warning("Historical master repository baseline absent. Creating new file repository.")
        master_df = pd.DataFrame()
    else:
        master_df = pd.read_csv(DATASET_PATH)

    update_df = pd.read_csv(uploaded_csv_path)
    combined_df = pd.concat([master_df, update_df], ignore_index=True).drop_duplicates()

    # Feature preparation
    combined_df["car_age"] = 2026 - combined_df["year"]
    num_cols = ["car_age", "mileage", "engine_size", "hp"]
    cat_cols = ["brand", "transmission", "fuel_type"]
    target_col = "price"

    X = combined_df[num_cols + cat_cols]
    y = combined_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Re-fit preprocessing pipelines
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(cat_feature_names)

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    # Evaluate new champion candidate model
    candidate_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    candidate_model.fit(X_train_final, y_train)
    
    candidate_preds = candidate_model.predict(X_test_final)
    new_r2 = float(r2_score(y_test, candidate_preds))

    # Evaluate against active production model baseline
    current_r2 = -float('inf')
    if os.path.exists(ARTIFACT_PATH):
        try:
            current_artifacts = joblib.load(ARTIFACT_PATH)
            current_model = current_artifacts["model"]
            current_scaler = current_artifacts["scaler"]
            current_encoder = current_artifacts["encoder"]
            
            # Match transformations to production structure
            s_num = current_scaler.transform(X_test[current_artifacts["features"]["numerical"]])
            e_cat = current_encoder.transform(X_test[current_artifacts["features"]["categorical"]])
            curr_final = np.hstack([s_num, e_cat])
            
            curr_preds = current_model.predict(curr_final)
            current_r2 = float(r2_score(y_test, curr_preds))
        except Exception as e:
            logger.error(f"Error evaluating active deployment baseline: {str(e)}")

    # Update production model if candidate out-performs it
    if new_r2 > current_r2:
        artifact_payload = {
            "model": candidate_model,
            "scaler": scaler,
            "encoder": encoder,
            "features": {
                "numerical": num_cols,
                "categorical": cat_cols,
                "all_features": all_feature_names
            }
        }
        # Atomic file write to avoid file corruption
        temp_artifact_path = ARTIFACT_PATH + ".tmp"
        joblib.dump(artifact_payload, temp_artifact_path)
        os.replace(temp_artifact_path, ARTIFACT_PATH)
        
        # Consolidate analytical data store
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        combined_df.to_csv(DATASET_PATH, index=False)
        
        invalidate_artifact_cache()
        return True, current_r2 if current_r2 != -float('inf') else 0.0, new_r2

    return False, current_r2, new_r2