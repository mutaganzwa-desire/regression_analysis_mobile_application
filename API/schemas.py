"""
API Data Contract Schema Layer.
Defines Pydantic models with strict range guardrails for input payload validation.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class CarPredictionInput(BaseModel):
    brand: str = Field(..., example="Ford")
    age: float = Field(..., ge=0, example=11.0)
    mileage: float = Field(..., ge=0, example=51000.0)
    fuel_type: str = Field(..., example="Gasoline")
    accident: str = Field(..., example="None reported")

    @field_validator('year')
    @classmethod
    def validate_year(cls, v: int) -> int:
        current_year = datetime.now().year + 1
        if v < 1980 or v > current_year:
            raise ValueError(f"Year must be between 1980 and {current_year}")
        return v

    @field_validator('mileage')
    @classmethod
    def validate_mileage(cls, v: float) -> float:
        if v < 0.0 or v > 1000000.0:
            raise ValueError("Mileage must be non-negative and less than 1,000,000")
        return v

    @field_validator('engine_size')
    @classmethod
    def validate_engine(cls, v: float) -> float:
        if v <= 0.0 or v > 10.0:
            raise ValueError("Engine size must be between 0.1 and 10.0 liters")
        return v

    @field_validator('hp')
    @classmethod
    def validate_hp(cls, v: float) -> float:
        if v <= 0.0 or v > 2000.0:
            raise ValueError("Horsepower must be between 1 and 2000")
        return v


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated market price of the asset")
    status: str = Field(..., description="Execution confirmation state")
    timestamp: str = Field(..., description="ISO formatted transactional timestamp")


class RetrainResponse(BaseModel):
    status: str = Field(..., description="Status message detailing evaluation outcome")
    previous_r2: float = Field(..., description="R-squared metric of the unseated operational model")
    new_r2: float = Field(..., description="R-squared metric of the newly validated model variant")
    model_updated: bool = Field(..., description="Flag indicating if the performance bound triggered a hot-swap")