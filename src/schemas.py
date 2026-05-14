"""
Pydantic schemas for the FastAPI endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    data_rows: Optional[int] = None
    latest_date: Optional[str] = None
    model_version: Optional[str] = None
    walk_forward_accuracy: Optional[float] = None


class PredictionResponse(BaseModel):
    signal: str
    prediction: int
    probability_up: float
    probability_down: float
    confidence: str                   # "high" | "medium" | "low"
    latest_close: float
    latest_timestamp: str
    threshold_used: float
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    feature_columns: List[str]
    feature_count: int
    metadata: Dict[str, Any]


class FetchResponse(BaseModel):
    success: bool
    rows_fetched: int
    rows_total: int
    latest_date: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
