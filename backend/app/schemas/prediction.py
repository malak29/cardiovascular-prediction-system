from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import re

from pydantic import (
    BaseModel,
    Field,
    validator,
    root_validator,
    Extra,
    conint,
    confloat,
    constr
)

from app.models.prediction import GenderEnum, RaceEthnicityEnum


class PatientDataSchema(BaseModel):
    """Schema for patient data input in predictions"""
    
    # Demographics
    age: conint(ge=18, le=120) = Field(..., description="Patient age in years (18-120)")
    gender: GenderEnum = Field(..., description="Patient gender")
    race_ethnicity: Optional[RaceEthnicityEnum] = Field(None, description="Patient race/ethnicity")
    
    # Geographic
    state_code: Optional[constr(regex=r"^[A-Z]{2}$")] = Field(None, description="US state code (e.g., 'CA')")
    zip_code: Optional[constr(regex=r"^\d{5}(-\d{4})?$")] = Field(None, description="ZIP code")
    urban_rural_code: Optional[constr(regex=r"^[1-6]$")] = Field(None, description="Urban/Rural code (1-6)")
    
    # Medical history - cardiovascular conditions
    has_hypertension: bool = Field(False, description="History of hypertension")
    has_diabetes: bool = Field(False, description="History of diabetes")
    has_heart_disease: bool = Field(False, description="History of heart disease")
    has_stroke_history: bool = Field(False, description="History of stroke")
    has_heart_attack_history: bool = Field(False, description="History of heart attack")
    has_atrial_fibrillation: bool = Field(False, description="History of atrial fibrillation")
    has_heart_failure: bool = Field(False, description="History of heart failure")
    
    # Other conditions
    has_kidney_disease: bool = Field(False, description="History of kidney disease")
    has_copd: bool = Field(False, description="History of COPD")
    has_depression: bool = Field(False, description="History of depression")
    
    # Risk factors
    smoking_status: Optional[constr(regex=r"^(never|former|current|unknown)$")] = Field(
        None, description="Smoking status"
    )
    bmi: Optional[confloat(ge=10.0, le=100.0)] = Field(None, description="Body Mass Index (10-100)")
    
    # Vital signs (latest available)
    systolic_bp: Optional[conint(ge=50, le=300)] = Field(None, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[conint(ge=30, le=200)] = Field(None, description="Diastolic blood pressure (mmHg)")
    total_cholesterol: Optional[confloat(ge=50.0, le=500.0)] = Field(None, description="Total cholesterol (mg/dL)")
    hdl_cholesterol: Optional[confloat(ge=10.0, le=150.0)] = Field(None, description="HDL cholesterol (mg/dL)")
    
    # Laboratory values
    blood_glucose: Optional[confloat(ge=30.0, le=600.0)] = Field(None, description="Blood glucose (mg/dL)")
    hba1c: Optional[confloat(ge=3.0, le=20.0)] = Field(None, description="HbA1c percentage")
    creatinine: Optional[confloat(ge=0.1, le=15.0)] = Field(None, description="Serum creatinine (mg/dL)")
    
    # Hospitalizations
    cardiovascular_hospitalizations_last_year: Optional[conint(ge=0, le=50)] = Field(
        0, description="CV hospitalizations in last year"
    )
    total_hospitalizations_last_year: Optional[conint(ge=0, le=50)] = Field(
        0, description="Total hospitalizations in last year"
    )
    
    # Medicare information
    medicare_part_a: bool = Field(True, description="Medicare Part A coverage")
    medicare_part_b: bool = Field(True, description="Medicare Part B coverage")
    medicare_part_c: bool = Field(False, description="Medicare Part C (Advantage)")
    medicare_part_d: bool = Field(False, description="Medicare Part D (prescription)")
    dual_eligible: bool = Field(False, description="Medicare + Medicaid dual eligible")
    
    # Additional features (flexible for model evolution)
    additional_features: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        None, description="Additional patient features"
    )
    
    class Config:
        extra = Extra.forbid
        schema_extra = {
            "example": {
                "age": 72,
                "gender": "male",
                "race_ethnicity": "white_non_hispanic",
                "state_code": "MA",
                "zip_code": "02101",
                "has_hypertension": True,
                "has_diabetes": True,
                "has_heart_disease": False,
                "smoking_status": "former",
                "bmi": 28.5,
                "systolic_bp": 145,
                "diastolic_bp": 90,
                "total_cholesterol": 220.0,
                "hdl_cholesterol": 40.0,
                "blood_glucose": 126.0,
                "hba1c": 7.2,
                "cardiovascular_hospitalizations_last_year": 0,
                "medicare_part_a": True,
                "medicare_part_b": True
            }
        }
    
    @validator('total_hospitalizations_last_year')
    def validate_total_hospitalizations(cls, v, values):
        """Ensure total hospitalizations >= cardiovascular hospitalizations"""
        cv_hospitalizations = values.get('cardiovascular_hospitalizations_last_year', 0)
        if v is not None and cv_hospitalizations is not None and v < cv_hospitalizations:
            raise ValueError('Total hospitalizations must be >= cardiovascular hospitalizations')
        return v


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    
    patient_data: PatientDataSchema = Field(..., description="Patient data for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    include_confidence: bool = Field(False, description="Include confidence intervals")
    include_features: bool = Field(False, description="Include feature importance")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_data": {
                    "age": 72,
                    "gender": "male",
                    "has_hypertension": True,
                    "has_diabetes": True,
                    "systolic_bp": 145,
                    "diastolic_bp": 90
                },
                "include_confidence": True,
                "include_features": True
            }
        }


class FeatureImportanceSchema(BaseModel):
    """Schema for feature importance information"""
    
    feature_name: str = Field(..., description="Name of the feature")
    importance_score: confloat(ge=0.0, le=1.0) = Field(..., description="Importance score (0-1)")
    description: Optional[str] = Field(None, description="Human-readable feature description")
    category: Optional[str] = Field(None, description="Feature category")


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    
    # Prediction results
    risk_score: confloat(ge=0.0, le=1.0) = Field(..., description="Cardiovascular risk score (0-1)")
    risk_category: constr(regex=r"^(low|medium|high)$") = Field(..., description="Risk category")
    confidence_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Prediction confidence")
    
    # Confidence intervals (if requested)
    confidence_lower: Optional[float] = Field(None, description="Lower confidence bound")
    confidence_upper: Optional[float] = Field(None, description="Upper confidence bound")
    confidence_level: Optional[float] = Field(0.95, description="Confidence level (default 95%)")
    
    # Feature importance (if requested)
    feature_importance: Optional[List[FeatureImportanceSchema]] = Field(
        None, description="Top contributing features"
    )
    
    # Model information
    model_version: str = Field(..., description="Model version used")
    model_type: str = Field(..., description="Type of model used")
    
    # Metadata
    prediction_id: UUID = Field(..., description="Unique prediction identifier")
    prediction_time_ms: float = Field(..., description="Time taken for prediction (milliseconds)")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    
    # Recommendations (optional)
    recommendations: Optional[List[str]] = Field(None, description="Clinical recommendations")
    risk_factors: Optional[List[str]] = Field(None, description="Identified risk factors")
    
    class Config:
        schema_extra = {
            "example": {
                "risk_score": 0.75,
                "risk_category": "high",
                "confidence_score": 0.92,
                "confidence_lower": 0.68,
                "confidence_upper": 0.82,
                "model_version": "ridge_v1.2.0",
                "model_type": "ridge_regression",
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "prediction_time_ms": 245.7,
                "timestamp": "2024-08-22T10:30:00Z",
                "feature_importance": [
                    {"feature_name": "age", "importance_score": 0.35},
                    {"feature_name": "has_diabetes", "importance_score": 0.28}
                ],
                "recommendations": [
                    "Regular cardiovascular monitoring recommended",
                    "Consider lifestyle modifications"
                ]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    
    patients_data: List[PatientDataSchema] = Field(
        ..., 
        min_items=1, 
        max_items=1000,
        description="List of patient data (max 1000)"
    )
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    include_confidence: bool = Field(False, description="Include confidence intervals")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    batch_name: Optional[str] = Field(None, description="Optional name for the batch")
    
    class Config:
        schema_extra = {
            "example": {
                "patients_data": [
                    {
                        "age": 72,
                        "gender": "male",
                        "has_hypertension": True
                    },
                    {
                        "age": 68,
                        "gender": "female",
                        "has_diabetes": True
                    }
                ],
                "include_confidence": True,
                "parallel_processing": True,
                "batch_name": "Q4_2024_screening"
            }
        }


class BatchPredictionItem(BaseModel):
    """Individual prediction item in batch response"""
    
    patient_index: int = Field(..., description="Index of patient in original request")
    prediction: Optional[PredictionResponse] = Field(None, description="Prediction result")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_index": 0,
                "prediction": {
                    "risk_score": 0.75,
                    "risk_category": "high",
                    "model_version": "ridge_v1.2.0"
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    # Results
    predictions: List[BatchPredictionItem] = Field(..., description="List of prediction results")
    failed_predictions: List[BatchPredictionItem] = Field(..., description="Failed predictions with errors")
    
    # Summary
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    
    # Metadata
    batch_id: UUID = Field(..., description="Unique batch identifier")
    total_requested: int = Field(..., description="Total predictions requested")
    successful_count: int = Field(..., description="Number of successful predictions")
    failed_count: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time")
    
    # Model information
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Batch completion timestamp")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "patient_index": 0,
                        "prediction": {
                            "risk_score": 0.75,
                            "risk_category": "high"
                        }
                    }
                ],
                "failed_predictions": [],
                "summary": {
                    "risk_distribution": {"low": 30, "medium": 45, "high": 25},
                    "average_risk_score": 0.52
                },
                "batch_id": "123e4567-e89b-12d3-a456-426614174000",
                "total_requested": 100,
                "successful_count": 100,
                "failed_count": 0,
                "processing_time_ms": 2450.7
            }
        }


class ModelInfoSchema(BaseModel):
    """Schema for model information"""
    
    id: UUID
    version: str = Field(..., description="Model version identifier")
    name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Type of ML model")
    description: Optional[str] = Field(None, description="Model description")
    
    # Status
    is_active: bool = Field(..., description="Whether model is currently active")
    is_default: bool = Field(..., description="Whether this is the default model")
    status: str = Field(..., description="Model status")
    
    # Performance metrics
    r2_score: Optional[float] = Field(None, description="R-squared score")
    mse_score: Optional[float] = Field(None, description="Mean Squared Error")
    cross_val_score: Optional[float] = Field(None, description="Cross-validation score")
    
    # Training information
    training_data_size: Optional[int] = Field(None, description="Size of training dataset")
    created_at: datetime = Field(..., description="Model creation timestamp")
    deployed_at: Optional[datetime] = Field(None, description="Model deployment timestamp")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "version": "ridge_v1.2.0",
                "name": "Ridge Regression CV Predictor",
                "model_type": "ridge_regression",
                "is_active": True,
                "is_default": True,
                "status": "active",
                "r2_score": 0.892,
                "mse_score": 0.0234,
                "training_data_size": 50000,
                "created_at": "2024-08-01T10:00:00Z"
            }
        }


class ModelPerformanceResponse(BaseModel):
    """Response schema for model performance metrics"""
    
    # Model identification
    model_id: UUID
    model_version: str
    model_type: str
    
    # Core performance metrics
    r2_score: float = Field(..., description="R-squared score")
    mse_score: float = Field(..., description="Mean Squared Error")
    mae_score: float = Field(..., description="Mean Absolute Error")
    rmse_score: float = Field(..., description="Root Mean Squared Error")
    
    # Classification metrics (if applicable)
    accuracy_score: Optional[float] = Field(None, description="Accuracy score")
    precision_score: Optional[float] = Field(None, description="Precision score")
    recall_score: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    auc_score: Optional[float] = Field(None, description="AUC-ROC score")
    
    # Cross-validation results
    cross_validation: Dict[str, Any] = Field(..., description="Cross-validation results")
    
    # Feature importance
    feature_importance: List[FeatureImportanceSchema] = Field(..., description="Feature importance rankings")
    
    # Training information
    training_info: Dict[str, Any] = Field(..., description="Training metadata")
    
    # Evaluation metadata
    evaluation_date: datetime = Field(..., description="When metrics were calculated")
    evaluation_method: str = Field(..., description="Evaluation methodology")
    test_data_size: int = Field(..., description="Size of test dataset")
    
    class Config:
        from_attributes = True


class PredictionHistoryItem(BaseModel):
    """Schema for individual prediction history item"""
    
    prediction_id: UUID
    risk_score: float
    risk_category: str
    model_version: str
    created_at: datetime
    
    # Patient information (anonymized)
    patient_age: int
    patient_gender: str
    
    # Optional feedback
    has_feedback: bool = Field(False, description="Whether feedback was provided")
    actual_outcome: Optional[bool] = Field(None, description="Actual outcome if known")
    
    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    """Response schema for prediction history"""
    
    predictions: List[PredictionHistoryItem] = Field(..., description="List of historical predictions")
    total_count: int = Field(..., description="Total predictions matching filters")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    
    # Pagination
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether more items are available")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                        "risk_score": 0.75,
                        "risk_category": "high",
                        "model_version": "ridge_v1.2.0",
                        "created_at": "2024-08-22T10:30:00Z",
                        "patient_age": 72,
                        "patient_gender": "male"
                    }
                ],
                "total_count": 1250,
                "summary": {
                    "risk_distribution": {"low": 400, "medium": 550, "high": 300},
                    "average_risk_score": 0.52
                },
                "limit": 50,
                "offset": 0,
                "has_more": True
            }
        }


class ModelTrainingRequest(BaseModel):
    """Request schema for model training"""
    
    model_name: str = Field(..., description="Name for the new model")
    model_type: constr(regex=r"^(linear|ridge|lasso|xgboost|lightgbm|random_forest)$") = Field(
        ..., description="Type of model to train"
    )
    training_dataset_id: UUID = Field(..., description="ID of training dataset")
    validation_dataset_id: Optional[UUID] = Field(None, description="ID of validation dataset")
    
    # Training parameters
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    cross_validation_folds: conint(ge=3, le=10) = Field(5, description="Number of CV folds")
    test_size: confloat(ge=0.1, le=0.5) = Field(0.2, description="Test set size ratio")
    
    # Training options
    enable_feature_selection: bool = Field(True, description="Enable automatic feature selection")
    enable_hyperparameter_tuning: bool = Field(False, description="Enable hyperparameter optimization")
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "CVD Risk Predictor v2",
                "model_type": "ridge",
                "training_dataset_id": "123e4567-e89b-12d3-a456-426614174000",
                "hyperparameters": {"alpha": 1.0},
                "cross_validation_folds": 5,
                "enable_feature_selection": True
            }
        }


class ModelTrainingResponse(BaseModel):
    """Response schema for model training job"""
    
    job_id: UUID = Field(..., description="Training job identifier")
    job_status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    
    # Timing estimates
    estimated_duration_minutes: Optional[float] = Field(None, description="Estimated training duration")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Progress tracking
    progress_percentage: float = Field(0.0, description="Training progress percentage")
    current_step: Optional[str] = Field(None, description="Current training step")
    
    # Job metadata
    created_at: datetime = Field(..., description="Job creation timestamp")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        from_attributes = True


class PredictionFeedbackRequest(BaseModel):
    """Request schema for prediction feedback"""
    
    prediction_id: UUID = Field(..., description="ID of the prediction to provide feedback on")
    feedback_type: constr(regex=r"^(outcome|rating|correction|note)$") = Field(
        ..., description="Type of feedback"
    )
    
    # Outcome feedback
    actual_outcome: Optional[bool] = Field(None, description="Actual hospitalization outcome")
    outcome_date: Optional[date] = Field(None, description="Date of outcome")
    
    # Rating feedback
    provider_rating: Optional[conint(ge=1, le=5)] = Field(None, description="Provider rating (1-5)")
    provider_comments: Optional[str] = Field(None, description="Provider comments")
    
    # Correction feedback
    corrected_risk_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Corrected risk score")
    corrected_risk_category: Optional[constr(regex=r"^(low|medium|high)$")] = Field(None, description="Corrected category")
    correction_reason: Optional[str] = Field(None, description="Reason for correction")
    
    # General feedback
    notes: Optional[str] = Field(None, description="Additional notes")
    
    @root_validator
    def validate_feedback_data(cls, values):
        """Validate feedback data based on feedback type"""
        feedback_type = values.get('feedback_type')
        
        if feedback_type == 'outcome':
            if values.get('actual_outcome') is None:
                raise ValueError('actual_outcome is required for outcome feedback')
        elif feedback_type == 'rating':
            if values.get('provider_rating') is None:
                raise ValueError('provider_rating is required for rating feedback')
        elif feedback_type == 'correction':
            if not any([values.get('corrected_risk_score'), values.get('corrected_risk_category')]):
                raise ValueError('corrected_risk_score or corrected_risk_category required for correction feedback')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "feedback_type": "outcome",
                "actual_outcome": False,
                "outcome_date": "2024-12-01",
                "provider_comments": "Patient managed with medication, no hospitalization required"
            }
        }


class PredictionFeedbackResponse(BaseModel):
    """Response schema for prediction feedback submission"""
    
    feedback_id: UUID = Field(..., description="Feedback record identifier")
    message: str = Field(..., description="Confirmation message")
    prediction_id: UUID = Field(..., description="Associated prediction ID")
    feedback_type: str = Field(..., description="Type of feedback provided")
    created_at: datetime = Field(..., description="Feedback submission timestamp")
    
    class Config:
        from_attributes = True


# Utility schemas
class ErrorResponse(BaseModel):
    """Standard error response schema"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class SuccessResponse(BaseModel):
    """Standard success response schema"""
    
    message: str = Field(..., description="Success message")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")


# Export all schemas
__all__ = [
    "PatientDataSchema",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionItem",
    "BatchPredictionResponse",
    "ModelInfoSchema",
    "ModelPerformanceResponse",
    "PredictionHistoryItem",
    "PredictionHistoryResponse",
    "ModelTrainingRequest",
    "ModelTrainingResponse",
    "PredictionFeedbackRequest",
    "PredictionFeedbackResponse",
    "FeatureImportanceSchema",
    "ErrorResponse",
    "SuccessResponse"
]