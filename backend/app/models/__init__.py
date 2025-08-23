from app.models.prediction import (
    PredictionModel,
    Prediction,
    PredictionBatch,
    ModelPerformanceHistory,
    PredictionFeedback,
    ModelTrainingJob,
    PredictionAuditLog,
    Dataset,
    User,
    APIKey,
    GenderEnum,
    RaceEthnicityEnum
)

from app.models.patient import (
    Patient,
    PatientMedicalHistory,
    PatientVitals,
    PatientMedication,
    PatientRiskFactor,
    PatientOutcome,
    HealthcareProvider,
    PatientProviderRelationship
)

# Base model for all tables
from app.core.database import Base

# Model collections for easy iteration
PREDICTION_MODELS = [
    PredictionModel,
    Prediction,
    PredictionBatch,
    ModelPerformanceHistory,
    PredictionFeedback,
    ModelTrainingJob,
    PredictionAuditLog
]

PATIENT_MODELS = [
    Patient,
    PatientMedicalHistory,
    PatientVitals,
    PatientMedication,
    PatientRiskFactor,
    PatientOutcome,
    HealthcareProvider,
    PatientProviderRelationship
]

DATA_MODELS = [
    Dataset,
    User,
    APIKey
]

ALL_MODELS = PREDICTION_MODELS + PATIENT_MODELS + DATA_MODELS

# Model registry for dynamic access
MODEL_REGISTRY = {
    # Prediction models
    "prediction_model": PredictionModel,
    "prediction": Prediction,
    "prediction_batch": PredictionBatch,
    "model_performance_history": ModelPerformanceHistory,
    "prediction_feedback": PredictionFeedback,
    "model_training_job": ModelTrainingJob,
    "prediction_audit_log": PredictionAuditLog,
    
    # Patient models
    "patient": Patient,
    "patient_medical_history": PatientMedicalHistory,
    "patient_vitals": PatientVitals,
    "patient_medication": PatientMedication,
    "patient_risk_factor": PatientRiskFactor,
    "patient_outcome": PatientOutcome,
    "healthcare_provider": HealthcareProvider,
    "patient_provider_relationship": PatientProviderRelationship,
    
    # Data models
    "dataset": Dataset,
    "user": User,
    "api_key": APIKey
}

# Utility functions
def get_model_by_name(model_name: str):
    """Get model class by name"""
    return MODEL_REGISTRY.get(model_name)

def get_all_table_names() -> list[str]:
    """Get all table names in the system"""
    return [model.__tablename__ for model in ALL_MODELS]

def get_models_by_category(category: str) -> list:
    """Get models by category"""
    category_map = {
        "prediction": PREDICTION_MODELS,
        "patient": PATIENT_MODELS,
        "data": DATA_MODELS
    }
    return category_map.get(category, [])

# Export everything
__all__ = [
    # Base
    "Base",
    
    # Prediction models
    "PredictionModel",
    "Prediction",
    "PredictionBatch", 
    "ModelPerformanceHistory",
    "PredictionFeedback",
    "ModelTrainingJob",
    "PredictionAuditLog",
    
    # Patient models
    "Patient",
    "PatientMedicalHistory",
    "PatientVitals", 
    "PatientMedication",
    "PatientRiskFactor",
    "PatientOutcome",
    "HealthcareProvider",
    "PatientProviderRelationship",
    
    # Data models
    "Dataset",
    "User",
    "APIKey",
    
    # Enums
    "GenderEnum",
    "RaceEthnicityEnum",
    
    # Collections
    "PREDICTION_MODELS",
    "PATIENT_MODELS", 
    "DATA_MODELS",
    "ALL_MODELS",
    "MODEL_REGISTRY",
    
    # Utilities
    "get_model_by_name",
    "get_all_table_names",
    "get_models_by_category"
]