from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from app.core.database import Base


class PredictionModel(Base):
    """Model versions and metadata for cardiovascular prediction models"""
    
    __tablename__ = "prediction_models"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Model identification
    version = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    model_type = Column(String(50), nullable=False)  # linear, ridge, lasso, xgboost, etc.
    
    # Model status
    is_active = Column(Boolean, default=False, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    status = Column(String(20), default="training", nullable=False)  # training, active, deprecated
    
    # Model files and configuration
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500))
    feature_columns = Column(JSONB)  # List of feature column names
    target_column = Column(String(100), default="Data_Value")
    
    # Performance metrics
    r2_score = Column(Float)
    mse_score = Column(Float)
    mae_score = Column(Float)
    cross_val_score = Column(Float)
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    
    # Training information
    training_data_size = Column(Integer)
    training_features_count = Column(Integer)
    training_duration_minutes = Column(Float)
    hyperparameters = Column(JSONB)
    
    # Versioning and tracking
    parent_model_id = Column(UUID(as_uuid=True), ForeignKey("prediction_models.id"))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Deployment information
    deployed_at = Column(DateTime(timezone=True))
    deprecated_at = Column(DateTime(timezone=True))
    
    # Relationships
    parent_model = relationship("PredictionModel", remote_side=[id])
    predictions = relationship("Prediction", back_populates="model")
    created_by_user = relationship("User", foreign_keys=[created_by])
    
    # Constraints
    __table_args__ = (
        CheckConstraint("r2_score >= 0 AND r2_score <= 1", name="check_r2_score_range"),
        CheckConstraint("mse_score >= 0", name="check_mse_positive"),
        CheckConstraint("mae_score >= 0", name="check_mae_positive"),
        CheckConstraint("training_data_size > 0", name="check_training_data_size"),
        CheckConstraint("status IN ('training', 'active', 'deprecated', 'failed')", name="check_status_values"),
        Index("idx_model_version_status", "version", "status"),
        Index("idx_model_active_default", "is_active", "is_default"),
        Index("idx_model_created_at", "created_at"),
    )
    
    @validates('model_type')
    def validate_model_type(self, key, model_type):
        """Validate model type"""
        allowed_types = ['linear', 'ridge', 'lasso', 'xgboost', 'lightgbm', 'random_forest', 'neural_network']
        if model_type not in allowed_types:
            raise ValueError(f"Model type must be one of: {allowed_types}")
        return model_type
    
    def __repr__(self):
        return f"<PredictionModel(version='{self.version}', type='{self.model_type}', active={self.is_active})>"


class Prediction(Base):
    """Individual prediction records with patient data and results"""
    
    __tablename__ = "predictions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Prediction metadata
    correlation_id = Column(String(100), index=True)
    prediction_type = Column(String(20), default="single", nullable=False)  # single, batch, csv
    
    # Model information
    model_id = Column(UUID(as_uuid=True), ForeignKey("prediction_models.id"), nullable=False)
    model_version = Column(String(50), nullable=False, index=True)
    
    # Patient identification (optional - for tracking)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    external_patient_id = Column(String(100))  # For external system integration
    
    # Input features (stored as JSON for flexibility)
    input_features = Column(JSONB, nullable=False)
    processed_features = Column(JSONB)  # Features after preprocessing
    
    # Prediction results
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)  # low, medium, high
    confidence_score = Column(Float)
    
    # Confidence intervals
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    confidence_level = Column(Float, default=0.95)
    
    # Feature importance (top contributing factors)
    feature_importance = Column(JSONB)
    
    # Prediction metadata
    prediction_time_ms = Column(Float)  # Time taken for prediction
    api_version = Column(String(20), default="v1")
    
    # User and session tracking
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    session_id = Column(String(100))
    ip_address = Column(String(45))  # Supports IPv6
    user_agent = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Feedback and validation (for model improvement)
    actual_outcome = Column(Boolean)  # True if patient was actually hospitalized
    outcome_date = Column(DateTime(timezone=True))
    feedback_score = Column(Integer)  # 1-5 rating from healthcare provider
    feedback_notes = Column(Text)
    
    # Relationships
    model = relationship("PredictionModel", back_populates="predictions")
    patient = relationship("Patient", back_populates="predictions")
    user = relationship("User", foreign_keys=[user_id])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="check_risk_score_range"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_range"),
        CheckConstraint("confidence_level > 0 AND confidence_level < 1", name="check_confidence_level"),
        CheckConstraint("feedback_score >= 1 AND feedback_score <= 5", name="check_feedback_score_range"),
        CheckConstraint("risk_category IN ('low', 'medium', 'high')", name="check_risk_category"),
        CheckConstraint("prediction_type IN ('single', 'batch', 'csv')", name="check_prediction_type"),
        Index("idx_prediction_created_at", "created_at"),
        Index("idx_prediction_risk_category", "risk_category"),
        Index("idx_prediction_user_created", "user_id", "created_at"),
        Index("idx_prediction_model_created", "model_id", "created_at"),
        Index("idx_prediction_correlation_id", "correlation_id"),
    )
    
    @validates('risk_category')
    def validate_risk_category(self, key, risk_category):
        """Validate risk category based on risk score"""
        if hasattr(self, 'risk_score') and self.risk_score is not None:
            if self.risk_score < 0.3 and risk_category != 'low':
                raise ValueError("Low risk score should have 'low' category")
            elif 0.3 <= self.risk_score < 0.7 and risk_category != 'medium':
                raise ValueError("Medium risk score should have 'medium' category")
            elif self.risk_score >= 0.7 and risk_category != 'high':
                raise ValueError("High risk score should have 'high' category")
        return risk_category
    
    def __repr__(self):
        return f"<Prediction(id='{self.id}', risk_score={self.risk_score}, category='{self.risk_category}')>"


class PredictionBatch(Base):
    """Batch prediction jobs for tracking multiple predictions"""
    
    __tablename__ = "prediction_batches"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Batch metadata
    batch_name = Column(String(200))
    correlation_id = Column(String(100), nullable=False, index=True)
    
    # Job information
    job_status = Column(String(20), default="queued", nullable=False)  # queued, running, completed, failed
    total_records = Column(Integer, nullable=False)
    processed_records = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)
    failed_predictions = Column(Integer, default=0)
    
    # File information (for CSV uploads)
    original_filename = Column(String(500))
    processed_filename = Column(String(500))
    file_size_bytes = Column(Integer)
    
    # Model and processing info
    model_id = Column(UUID(as_uuid=True), ForeignKey("prediction_models.id"), nullable=False)
    model_version = Column(String(50), nullable=False)
    processing_parameters = Column(JSONB)
    
    # Timing information
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    processing_duration_seconds = Column(Float)
    
    # User tracking
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # Results summary
    results_summary = Column(JSONB)  # Summary statistics of batch results
    
    # Relationships
    model = relationship("PredictionModel")
    user = relationship("User", foreign_keys=[user_id])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("total_records > 0", name="check_total_records_positive"),
        CheckConstraint("processed_records >= 0", name="check_processed_records_non_negative"),
        CheckConstraint("successful_predictions >= 0", name="check_successful_predictions_non_negative"),
        CheckConstraint("failed_predictions >= 0", name="check_failed_predictions_non_negative"),
        CheckConstraint("job_status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="check_job_status"),
        Index("idx_batch_status_created", "job_status", "created_at"),
        Index("idx_batch_user_created", "user_id", "created_at"),
        Index("idx_batch_correlation_id", "correlation_id"),
    )
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.processed_records == 0:
            return 0.0
        return (self.successful_predictions / self.processed_records) * 100
    
    def __repr__(self):
        return f"<PredictionBatch(id='{self.id}', status='{self.job_status}', progress={self.progress_percentage:.1f}%)>"


class ModelPerformanceHistory(Base):
    """Historical performance tracking for models"""
    
    __tablename__ = "model_performance_history"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Model reference
    model_id = Column(UUID(as_uuid=True), ForeignKey("prediction_models.id"), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Evaluation period
    evaluation_date = Column(DateTime(timezone=True), nullable=False)
    evaluation_period_start = Column(DateTime(timezone=True))
    evaluation_period_end = Column(DateTime(timezone=True))
    
    # Performance metrics
    r2_score = Column(Float)
    mse_score = Column(Float)
    mae_score = Column(Float)
    rmse_score = Column(Float)
    accuracy_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    
    # Cross-validation results
    cv_mean_score = Column(Float)
    cv_std_score = Column(Float)
    cv_scores = Column(JSONB)  # Individual fold scores
    
    # Prediction statistics
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    # Risk category distribution
    low_risk_predictions = Column(Integer, default=0)
    medium_risk_predictions = Column(Integer, default=0)
    high_risk_predictions = Column(Integer, default=0)
    
    # Feature importance
    feature_importance = Column(JSONB)
    top_features = Column(JSONB)  # Top 10 most important features
    
    # Evaluation metadata
    evaluation_method = Column(String(50), default="holdout")  # holdout, cross_validation, temporal
    test_data_size = Column(Integer)
    evaluation_notes = Column(Text)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    model = relationship("PredictionModel")
    created_by_user = relationship("User", foreign_keys=[created_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("r2_score >= 0 AND r2_score <= 1", name="check_r2_score_range"),
        CheckConstraint("total_predictions >= 0", name="check_total_predictions_non_negative"),
        CheckConstraint("correct_predictions >= 0", name="check_correct_predictions_non_negative"),
        Index("idx_performance_model_date", "model_id", "evaluation_date"),
        Index("idx_performance_version_date", "model_version", "evaluation_date"),
        UniqueConstraint("model_id", "evaluation_date", name="uq_model_evaluation_date"),
    )
    
    def __repr__(self):
        return f"<ModelPerformanceHistory(model_version='{self.model_version}', r2={self.r2_score:.4f})>"


class PredictionFeedback(Base):
    """Feedback on predictions for model improvement"""
    
    __tablename__ = "prediction_feedback"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Reference to prediction
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False)
    
    # Feedback information
    feedback_type = Column(String(20), nullable=False)  # outcome, rating, correction
    feedback_value = Column(JSONB, nullable=False)  # Flexible feedback data
    
    # Outcome tracking (for model validation)
    actual_outcome = Column(Boolean)  # True if hospitalization occurred
    outcome_date = Column(DateTime(timezone=True))
    outcome_source = Column(String(100))  # hospital, insurance, patient, etc.
    
    # Provider feedback
    provider_rating = Column(Integer)  # 1-5 rating
    provider_comments = Column(Text)
    clinical_notes = Column(Text)
    
    # Correction data (if prediction was wrong)
    corrected_risk_score = Column(Float)
    corrected_risk_category = Column(String(20))
    correction_reason = Column(Text)
    
    # Tracking
    feedback_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    provided_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    provider_role = Column(String(50))  # doctor, nurse, patient, admin
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verified_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    verified_at = Column(DateTime(timezone=True))
    
    # Relationships
    prediction = relationship("Prediction")
    provided_by_user = relationship("User", foreign_keys=[provided_by])
    verified_by_user = relationship("User", foreign_keys=[verified_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("provider_rating >= 1 AND provider_rating <= 5", name="check_provider_rating_range"),
        CheckConstraint("corrected_risk_score >= 0 AND corrected_risk_score <= 1", name="check_corrected_risk_range"),
        CheckConstraint("feedback_type IN ('outcome', 'rating', 'correction', 'note')", name="check_feedback_type"),
        CheckConstraint("corrected_risk_category IN ('low', 'medium', 'high')", name="check_corrected_category"),
        Index("idx_feedback_prediction_date", "prediction_id", "feedback_date"),
        Index("idx_feedback_outcome", "actual_outcome", "outcome_date"),
        Index("idx_feedback_provider", "provided_by", "feedback_date"),
    )
    
    def __repr__(self):
        return f"<PredictionFeedback(prediction_id='{self.prediction_id}', type='{self.feedback_type}')>"


class ModelTrainingJob(Base):
    """Training jobs for tracking model development"""
    
    __tablename__ = "model_training_jobs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Job identification
    job_name = Column(String(200), nullable=False)
    correlation_id = Column(String(100), nullable=False, index=True)
    
    # Job status
    status = Column(String(20), default="queued", nullable=False)  # queued, running, completed, failed
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(100))
    total_steps = Column(Integer)
    
    # Training configuration
    model_type = Column(String(50), nullable=False)
    hyperparameters = Column(JSONB)
    training_config = Column(JSONB)
    
    # Data information
    training_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    validation_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    training_data_size = Column(Integer)
    validation_data_size = Column(Integer)
    
    # Results
    resulting_model_id = Column(UUID(as_uuid=True), ForeignKey("prediction_models.id"))
    final_metrics = Column(JSONB)
    training_history = Column(JSONB)  # Training progress over epochs/iterations
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    total_duration_minutes = Column(Float)
    
    # Resource usage
    cpu_usage_avg = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_usage_avg = Column(Float)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSONB)
    retry_count = Column(Integer, default=0)
    
    # User tracking
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    training_dataset = relationship("Dataset", foreign_keys=[training_dataset_id])
    validation_dataset = relationship("Dataset", foreign_keys=[validation_dataset_id])
    resulting_model = relationship("PredictionModel")
    created_by_user = relationship("User", foreign_keys=[created_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100", name="check_progress_range"),
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="check_job_status"),
        CheckConstraint("retry_count >= 0", name="check_retry_count_non_negative"),
        Index("idx_training_job_status_created", "status", "created_at"),
        Index("idx_training_job_user_created", "created_by", "created_at"),
        Index("idx_training_job_correlation_id", "correlation_id"),
    )
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self.status in ["queued", "running"]
    
    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully"""
        return self.status == "completed"
    
    @property
    def estimated_remaining_minutes(self) -> Optional[float]:
        """Estimate remaining time for running jobs"""
        if not self.is_running or not self.started_at or self.progress_percentage == 0:
            return None
        
        elapsed_minutes = (datetime.utcnow() - self.started_at).total_seconds() / 60
        total_estimated_minutes = elapsed_minutes / (self.progress_percentage / 100)
        remaining_minutes = total_estimated_minutes - elapsed_minutes
        
        return max(0, remaining_minutes)
    
    def __repr__(self):
        return f"<ModelTrainingJob(id='{self.id}', status='{self.status}', progress={self.progress_percentage:.1f}%)>"


class PredictionAuditLog(Base):
    """Audit log for prediction system operations"""
    
    __tablename__ = "prediction_audit_logs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Audit information
    event_type = Column(String(50), nullable=False)  # prediction, model_update, data_sync, etc.
    event_description = Column(String(500), nullable=False)
    
    # Related entities
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"))
    model_id = Column(UUID(as_uuid=True), ForeignKey("prediction_models.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Event details
    event_data = Column(JSONB)  # Flexible event data
    correlation_id = Column(String(100), index=True)
    
    # Request information
    ip_address = Column(String(45))
    user_agent = Column(Text)
    endpoint = Column(String(200))
    http_method = Column(String(10))
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Risk and compliance
    risk_level = Column(String(20), default="low")  # low, medium, high
    compliance_flags = Column(JSONB)  # HIPAA, GDPR, etc.
    
    # Relationships
    prediction = relationship("Prediction")
    model = relationship("PredictionModel")
    user = relationship("User", foreign_keys=[user_id])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="check_risk_level"),
        Index("idx_audit_event_type_date", "event_type", "created_at"),
        Index("idx_audit_user_date", "user_id", "created_at"),
        Index("idx_audit_correlation_id", "correlation_id"),
        Index("idx_audit_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<PredictionAuditLog(event_type='{self.event_type}', created_at='{self.created_at}')>"


# Model utility functions
def get_risk_category_from_score(risk_score: float) -> str:
    """Convert numeric risk score to category"""
    if risk_score < 0.3:
        return "low"
    elif risk_score < 0.7:
        return "medium"
    else:
        return "high"


def validate_risk_score_category_consistency(risk_score: float, risk_category: str) -> bool:
    """Validate that risk score and category are consistent"""
    expected_category = get_risk_category_from_score(risk_score)
    return expected_category == risk_category


# Export all models
__all__ = [
    "PredictionModel",
    "Prediction",
    "PredictionBatch",
    "ModelPerformanceHistory",
    "PredictionFeedback",
    "ModelTrainingJob",
    "PredictionAuditLog",
    "get_risk_category_from_score",
    "validate_risk_score_category_consistency"
]