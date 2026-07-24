"""
Main FastAPI Application Entrypoint. Exposes application endpoints, 
orchestrates middleware, and implements access policies.
"""

import os
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from schemas import CarPredictionInput, PredictionResponse, RetrainResponse
from prediction import run_inference
from retrain import execute_pipeline_retraining

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_GATEWAY")

app = FastAPI(
    title="Automobile Valuator Engine API",
    description="Production machine learning backend providing real-time pricing intelligence.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Expanded origin fallbacks for seamless local mobile/web testing
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/", tags=["Root"])
async def root_redirect():
    """Returns basic service branding details."""
    return {
        "engine": "Automobile Valuator Machine Learning Pipeline",
        "status": "Operational",
        "documentation": "/docs"
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Verifies infrastructure status and loaded state dependencies."""
    try:
        from summative.API.prediction import load_prediction_artifacts
        artifacts = load_prediction_artifacts()
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifacts_loaded": list(artifacts.keys())
        }
    except Exception as e:
        logger.critical(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline unhealthy: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_price(payload: CarPredictionInput):
    """Ingests application features, runs data checks, and generates predictions."""
    try:
        input_dict = payload.model_dump()
        predicted_value = run_inference(input_dict)
        return PredictionResponse(
            predicted_price=round(predicted_value, 2),
            status="SUCCESS",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except FileNotFoundError as fnf:
        logger.error(f"Inference called without model initialization: {str(fnf)}")
        raise HTTPException(status_code=503, detail="Prediction engine uninitialized.")
    except Exception as e:
        logger.error(f"Prediction failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal inference error: {str(e)}")

@app.post("/retrain", response_model=RetrainResponse, tags=["Pipeline Management"])
async def retrain_engine(file: UploadFile = File(...)):
    """Ingests fresh raw logs, evaluates updating candidate pipelines, and triggers atomic hot-swaps."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid format. Supply standard CSV files only.")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        is_updated, prior_r2, fresh_r2 = execute_pipeline_retraining(temp_path)
        
        status_msg = "Candidate exceeded quality thresholds. Core deployment pipeline replaced." if is_updated \
            else "Candidate rejected. Deployment retains baseline architecture."
            
        return RetrainResponse(
            status=status_msg,
            previous_r2=prior_r2,
            new_r2=fresh_r2,
            model_updated=is_updated
        )
    except Exception as e:
        logger.error(f"Retraining lifecycle aborted: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining engine encountered exceptions: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)