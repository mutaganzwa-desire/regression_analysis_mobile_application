from pydantic import BaseModel, Field, field_validator
from typing import Optional

class CarPredictionInput(BaseModel):
    brand: str = Field(..., example="Toyota")
    model_year: int = Field(..., example=2018)
    milage: float = Field(..., example=65000.0)
    engine_size: float = Field(..., example=2.5)
    horsepower: int = Field(..., example=203)
    transmission: str = Field(..., example="Automatic")
    fuel_type: str = Field(..., example="Petrol")
    accident: Optional[str] = Field(default="None reported")

    @field_validator("model_year")
    def validate_model_year(cls, v):
        if v < 1900 or v > 2026:
            raise ValueError("Model year must be between 1900 and 2026.")
        return v

    @field_validator("milage")
    def validate_milage(cls, v):
        if v < 0:
            raise ValueError("Mileage cannot be negative.")
        return v


class PredictionResponse(BaseModel):
    predicted_price: float
    status: str = "SUCCESS"
    timestamp: str


class RetrainResponse(BaseModel):
    status: str
    previous_r2: Optional[float] = None
    new_r2: Optional[float] = None
    model_updated: bool