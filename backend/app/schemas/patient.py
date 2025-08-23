"""
Cardiovascular Disease Prediction System - Additional Patient Models

Additional SQLAlchemy models for comprehensive patient data management
including risk factors, outcomes, and healthcare provider relationships.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Date,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint,
    Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class RiskFactorTypeEnum(str, enum.Enum):
    """Risk factor type enumeration"""
    BEHAVIORAL = "behavioral"
    CLINICAL = "clinical"
    DEMOGRAPHIC = "demographic"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"


class OutcomeTypeEnum(str, enum.Enum):
    """Patient outcome type enumeration"""
    HOSPITALIZATION = "hospitalization"
    EMERGENCY_VISIT = "emergency_visit"
    PROCEDURE = "procedure"
    MEDICATION_CHANGE = "medication_change"
    DEATH = "death"
    RECOVERY = "recovery"


class PatientRiskFactor(Base):
    """Patient-specific risk factors for cardiovascular disease"""
    
    __tablename__ = "patient_risk_factors"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient reference
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Risk factor identification
    risk_factor_name = Column(String(100), nullable=False)
    risk_factor_type = Column(SQLEnum(RiskFactorTypeEnum), nullable=False)
    risk_factor_code = Column(String(20))  # ICD-10 or other standard code
    
    # Risk assessment
    risk_level = Column(String(20), nullable=False)  # low, moderate, high, severe
    risk_score = Column(Float)  # Numerical risk score if available
    
    # Clinical details
    onset_date = Column(Date)
    severity = Column(String(20))  # mild, moderate, severe
    is_controlled = Column(Boolean, default=False)
    control_method = Column(String(200))  # medication, lifestyle, procedure
    
    # Quantitative measures
    baseline_value = Column(Float)
    current_value = Column(Float)
    target_value = Column(Float)
    unit_of_measure = Column(String(20))
    
    # Provider assessment
    assessed_by = Column(UUID(as_uuid=True), ForeignKey("healthcare_providers.id"))
    assessment_date = Column(Date, nullable=False, default=date.today)
    next_assessment_date = Column(Date)
    
    # Risk factor status
    is_active = Column(Boolean, default=True)
    resolution_date = Column(Date)
    resolution_method = Column(String(200))
    
    # Documentation
    clinical_notes = Column(Text)
    evidence_source = Column(String(100))  # lab_result, physical_exam, patient_report
    confidence_level = Column(String(20))  # definite, probable, possible
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient")
    assessed_by_provider = relationship("HealthcareProvider", foreign_keys=[assessed_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("risk_level IN ('low', 'moderate', 'high', 'severe')", name="check_risk_level"),
        CheckConstraint("severity IN ('mild', 'moderate', 'severe')", name="check_severity"),
        CheckConstraint("confidence_level IN ('definite', 'probable', 'possible')", name="check_confidence"),
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="check_risk_score_range"),
        Index("idx_risk_factor_patient_type", "patient_id", "risk_factor_type"),
        Index("idx_risk_factor_active", "is_active", "assessment_date"),
        Index("idx_risk_factor_level", "risk_level", "risk_factor_type"),
    )
    
    @property
    def is_improving(self) -> Optional[bool]:
        """Check if risk factor is improving based on values"""
        if self.baseline_value and self.current_value:
            # Logic depends on the type of measurement
            # For most CV risk factors, lower is better
            return self.current_value < self.baseline_value
        return None
    
    def __repr__(self):
        return f"<PatientRiskFactor(patient_id='{self.patient_id}', factor='{self.risk_factor_name}', level='{self.risk_level}')>"


class PatientOutcome(Base):
    """Patient outcomes and events for model validation"""
    
    __tablename__ = "patient_outcomes"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient reference
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Related prediction (if this outcome validates a prediction)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"))
    
    # Outcome identification
    outcome_type = Column(SQLEnum(OutcomeTypeEnum), nullable=False)
    outcome_description = Column(String(500), nullable=False)
    
    # Clinical coding
    icd10_primary = Column(String(10))  # Primary ICD-10 diagnosis code
    icd10_secondary = Column(JSONB)  # List of secondary ICD-10 codes
    drg_code = Column(String(10))  # Diagnosis Related Group
    cpt_codes = Column(JSONB)  # Current Procedural Terminology codes
    
    # Temporal information
    outcome_date = Column(DateTime(timezone=True), nullable=False)
    admission_date = Column(DateTime(timezone=True))
    discharge_date = Column(DateTime(timezone=True))
    length_of_stay_days = Column(Integer)
    
    # Severity and urgency
    severity_score = Column(Integer)  # 1-5 scale
    urgency_level = Column(String(20))  # routine, urgent, emergent
    
    # Cardiovascular specific
    is_cardiovascular_related = Column(Boolean, default=False)
    cardiovascular_category = Column(String(50))  # acute_mi, stroke, heart_failure, etc.
    
    # Financial information
    total_cost = Column(Float)
    medicare_payment = Column(Float)
    patient_payment = Column(Float)
    
    # Facility information
    facility_name = Column(String(200))
    facility_type = Column(String(50))  # hospital, clinic, emergency_department
    facility_npi = Column(String(20))  # National Provider Identifier
    
    # Provider information
    attending_provider_id = Column(UUID(as_uuid=True), ForeignKey("healthcare_providers.id"))
    referring_provider_id = Column(UUID(as_uuid=True), ForeignKey("healthcare_providers.id"))
    
    # Outcome details
    disposition = Column(String(50))  # home, nursing_facility, rehabilitation, deceased
    readmission_30_day = Column(Boolean, default=False)
    readmission_date = Column(DateTime(timezone=True))
    
    # Data quality
    data_source = Column(String(100), nullable=False)  # medicare_claims, ehr, registry
    verified = Column(Boolean, default=False)
    verification_date = Column(DateTime(timezone=True))
    data_completeness = Column(Float)  # 0-1 score
    
    # Follow-up
    follow_up_required = Column(Boolean, default=False)
    follow_up_date = Column(Date)
    follow_up_completed = Column(Boolean, default=False)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient")
    prediction = relationship("Prediction")
    attending_provider = relationship("HealthcareProvider", foreign_keys=[attending_provider_id])
    referring_provider = relationship("HealthcareProvider", foreign_keys=[referring_provider_id])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("severity_score >= 1 AND severity_score <= 5", name="check_severity_score"),
        CheckConstraint("length_of_stay_days >= 0", name="check_los_non_negative"),
        CheckConstraint("data_completeness >= 0 AND data_completeness <= 1", name="check_completeness_range"),
        CheckConstraint("total_cost >= 0", name="check_total_cost_non_negative"),
        CheckConstraint("urgency_level IN ('routine', 'urgent', 'emergent')", name="check_urgency_level"),
        Index("idx_outcome_patient_date", "patient_id", "outcome_date"),
        Index("idx_outcome_cv_related", "is_cardiovascular_related", "outcome_date"),
        Index("idx_outcome_prediction", "prediction_id"),
        Index("idx_outcome_type_date", "outcome_type", "outcome_date"),
        Index("idx_outcome_readmission", "readmission_30_day", "outcome_date"),
    )
    
    @property
    def is_predicted_outcome(self) -> bool:
        """Check if this outcome validates a prediction"""
        return self.prediction_id is not None
    
    @property
    def was_readmitted(self) -> bool:
        """Check if patient was readmitted within 30 days"""
        return self.readmission_30_day
    
    def __repr__(self):
        return f"<PatientOutcome(patient_id='{self.patient_id}', type='{self.outcome_type}', date='{self.outcome_date}')>"


class HealthcareProvider(Base):
    """Healthcare providers and facilities"""
    
    __tablename__ = "healthcare_providers"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Provider identification
    npi = Column(String(20), unique=True, index=True)  # National Provider Identifier
    name = Column(String(200), nullable=False)
    provider_type = Column(String(50), nullable=False)  # individual, organization
    
    # Individual provider details
    first_name = Column(String(100))
    last_name = Column(String(100))
    middle_name = Column(String(100))
    credentials = Column(String(100))  # MD, DO, NP, PA, etc.
    
    # Specialty information
    primary_specialty = Column(String(100))
    secondary_specialties = Column(JSONB)  # List of additional specialties
    board_certifications = Column(JSONB)
    
    # Practice information
    practice_name = Column(String(200))
    organization_name = Column(String(200))
    tax_id = Column(String(20))  # Employer Identification Number
    
    # Contact information
    phone = Column(String(20))
    fax = Column(String(20))
    email = Column(String(320))
    website = Column(String(500))
    
    # Address
    address_line1 = Column(String(200))
    address_line2 = Column(String(200))
    city = Column(String(100))
    state = Column(String(2))
    zip_code = Column(String(10))
    county = Column(String(100))
    
    # Provider status
    is_active = Column(Boolean, default=True)
    medicare_enrolled = Column(Boolean, default=False)
    medicaid_enrolled = Column(Boolean, default=False)
    
    # Quality metrics
    star_rating = Column(Float)  # CMS star rating
    patient_satisfaction_score = Column(Float)
    quality_measures = Column(JSONB)
    
    # Cardiovascular expertise
    cardiovascular_specialist = Column(Boolean, default=False)
    cardiovascular_procedures = Column(JSONB)  # List of procedures performed
    annual_cv_cases = Column(Integer)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_verified = Column(DateTime(timezone=True))
    
    # Relationships
    patient_relationships = relationship("PatientProviderRelationship", back_populates="provider")
    assessed_risk_factors = relationship("PatientRiskFactor", foreign_keys="PatientRiskFactor.assessed_by")
    attended_outcomes = relationship("PatientOutcome", foreign_keys="PatientOutcome.attending_provider_id")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("star_rating >= 1 AND star_rating <= 5", name="check_star_rating_range"),
        CheckConstraint("patient_satisfaction_score >= 0 AND patient_satisfaction_score <= 100", name="check_satisfaction_range"),
        CheckConstraint("annual_cv_cases >= 0", name="check_cv_cases_non_negative"),
        CheckConstraint("provider_type IN ('individual', 'organization')", name="check_provider_type"),
        Index("idx_provider_specialty", "primary_specialty"),
        Index("idx_provider_cv_specialist", "cardiovascular_specialist"),
        Index("idx_provider_state_zip", "state", "zip_code"),
        Index("idx_provider_active", "is_active"),
    )
    
    @property
    def full_name(self) -> str:
        """Get provider's full name"""
        if self.provider_type == "individual":
            parts = [self.first_name, self.middle_name, self.last_name]
            return " ".join(part for part in parts if part)
        return self.name
    
    @property
    def display_name(self) -> str:
        """Get display name with credentials"""
        full_name = self.full_name
        if self.credentials:
            return f"{full_name}, {self.credentials}"
        return full_name
    
    def __repr__(self):
        return f"<HealthcareProvider(npi='{self.npi}', name='{self.display_name}')>"


class PatientProviderRelationship(Base):
    """Relationship between patients and healthcare providers"""
    
    __tablename__ = "patient_provider_relationships"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Relationship references
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("healthcare_providers.id"), nullable=False)
    
    # Relationship details
    relationship_type = Column(String(50), nullable=False)  # primary_care, specialist, consulting
    care_category = Column(String(50))  # cardiovascular, primary_care, emergency, etc.
    
    # Relationship period
    relationship_start = Column(Date, nullable=False)
    relationship_end = Column(Date)
    is_active = Column(Boolean, default=True)
    
    # Care details
    primary_provider = Column(Boolean, default=False)
    care_level = Column(String(20))  # primary, secondary, tertiary
    referral_source = Column(String(200))
    
    # Communication preferences
    preferred_contact_method = Column(String(20))  # phone, email, portal, mail
    language_preference = Column(String(20))
    
    # Care coordination
    care_plan_id = Column(String(100))
    shared_decision_making = Column(Boolean, default=False)
    
    # Quality metrics
    satisfaction_rating = Column(Integer)  # 1-5 scale
    communication_rating = Column(Integer)  # 1-5 scale
    
    # Visit tracking
    total_visits = Column(Integer, default=0)
    last_visit_date = Column(Date)
    next_appointment_date = Column(Date)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient")
    provider = relationship("HealthcareProvider", back_populates="patient_relationships")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("satisfaction_rating >= 1 AND satisfaction_rating <= 5", name="check_satisfaction_rating"),
        CheckConstraint("communication_rating >= 1 AND communication_rating <= 5", name="check_communication_rating"),
        CheckConstraint("total_visits >= 0", name="check_total_visits_non_negative"),
        CheckConstraint("relationship_type IN ('primary_care', 'specialist', 'consulting', 'emergency')", name="check_relationship_type"),
        CheckConstraint("care_level IN ('primary', 'secondary', 'tertiary')", name="check_care_level"),
        UniqueConstraint("patient_id", "provider_id", "relationship_type", name="uq_patient_provider_relationship"),
        Index("idx_patient_provider_active", "patient_id", "provider_id", "is_active"),
        Index("idx_provider_patient_active", "provider_id", "patient_id", "is_active"),
        Index("idx_relationship_primary", "primary_provider", "is_active"),
    )
    
    def __repr__(self):
        return f"<PatientProviderRelationship(patient_id='{self.patient_id}', provider_id='{self.provider_id}', type='{self.relationship_type}')>"


class PatientCareEpisode(Base):
    """Episodes of care for comprehensive patient management"""
    
    __tablename__ = "patient_care_episodes"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient reference
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Episode identification
    episode_number = Column(String(50), unique=True)
    episode_type = Column(String(50), nullable=False)  # acute, chronic, preventive, follow_up
    
    # Clinical information
    primary_diagnosis = Column(String(10))  # ICD-10 code
    secondary_diagnoses = Column(JSONB)  # List of ICD-10 codes
    chief_complaint = Column(Text)
    
    # Episode timeline
    episode_start_date = Column(DateTime(timezone=True), nullable=False)
    episode_end_date = Column(DateTime(timezone=True))
    is_ongoing = Column(Boolean, default=True)
    
    # Care settings
    care_settings = Column(JSONB)  # List of care settings (inpatient, outpatient, emergency, etc.)
    primary_facility_id = Column(UUID(as_uuid=True), ForeignKey("healthcare_providers.id"))
    
    # Clinical outcomes
    clinical_outcome = Column(String(50))  # resolved, improved, stable, deteriorated
    functional_status_change = Column(String(20))  # improved, same, declined
    
    # Quality measures
    care_coordination_score = Column(Float)  # 0-1
    patient_experience_score = Column(Float)  # 0-1
    clinical_quality_score = Column(Float)  # 0-1
    
    # Cost and utilization
    total_episode_cost = Column(Float)
    number_of_encounters = Column(Integer, default=0)
    emergency_visits = Column(Integer, default=0)
    hospitalizations = Column(Integer, default=0)
    
    # Risk assessment during episode
    admission_risk_score = Column(Float)
    discharge_risk_score = Column(Float)
    risk_change = Column(Float)  # Change in risk during episode
    
    # Care plan
    care_plan_goals = Column(JSONB)
    goals_achieved = Column(Integer, default=0)
    goals_total = Column(Integer, default=0)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient")
    primary_facility = relationship("HealthcareProvider", foreign_keys=[primary_facility_id])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("care_coordination_score >= 0 AND care_coordination_score <= 1", name="check_care_coordination_range"),
        CheckConstraint("admission_risk_score >= 0 AND admission_risk_score <= 1", name="check_admission_risk_range"),
        CheckConstraint("number_of_encounters >= 0", name="check_encounters_non_negative"),
        CheckConstraint("goals_achieved <= goals_total", name="check_goals_consistency"),
        CheckConstraint("episode_type IN ('acute', 'chronic', 'preventive', 'follow_up')", name="check_episode_type"),
        Index("idx_episode_patient_date", "patient_id", "episode_start_date"),
        Index("idx_episode_ongoing", "is_ongoing", "episode_start_date"),
        Index("idx_episode_type", "episode_type", "episode_start_date"),
    )
    
    @property
    def episode_duration_days(self) -> Optional[int]:
        """Calculate episode duration in days"""
        if self.episode_end_date:
            return (self.episode_end_date - self.episode_start_date).days
        elif not self.is_ongoing:
            return None
        else:
            return (datetime.utcnow() - self.episode_start_date).days
    
    @property
    def goals_achievement_rate(self) -> float:
        """Calculate care plan goals achievement rate"""
        if self.goals_total == 0:
            return 0.0
        return self.goals_achieved / self.goals_total
    
    def __repr__(self):
        return f"<PatientCareEpisode(id='{self.id}', patient_id='{self.patient_id}', type='{self.episode_type}')>"


class DataSyncJob(Base):
    """Data synchronization jobs for external data sources"""
    
    __tablename__ = "data_sync_jobs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Job identification
    job_name = Column(String(200), nullable=False)
    data_source = Column(String(100), nullable=False)  # cdc, cms, ehr_system
    sync_type = Column(String(50), nullable=False)  # full, incremental, validation
    
    # Job status
    status = Column(String(20), default="queued", nullable=False)  # queued, running, completed, failed
    progress_percentage = Column(Float, default=0.0)
    
    # Data processing
    records_requested = Column(Integer)
    records_received = Column(Integer, default=0)
    records_processed = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_skipped = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    
    # Time tracking
    scheduled_time = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    
    # Configuration
    sync_parameters = Column(JSONB)  # API endpoints, filters, etc.
    date_range_start = Column(Date)
    date_range_end = Column(Date)
    
    # Results and errors
    success_summary = Column(JSONB)
    error_summary = Column(JSONB)
    validation_results = Column(JSONB)
    
    # File tracking
    source_files = Column(JSONB)  # List of source files processed
    output_files = Column(JSONB)  # List of generated output files
    
    # User tracking
    triggered_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    correlation_id = Column(String(100), index=True)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    triggered_by_user = relationship("User", foreign_keys=[triggered_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100", name="check_progress_range"),
        CheckConstraint("records_received >= 0", name="check_records_received_non_negative"),
        CheckConstraint("records_processed >= 0", name="check_records_processed_non_negative"),
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="check_sync_status"),
        CheckConstraint("sync_type IN ('full', 'incremental', 'validation')", name="check_sync_type"),
        Index("idx_sync_job_status_created", "status", "created_at"),
        Index("idx_sync_job_source_type", "data_source", "sync_type"),
        Index("idx_sync_job_correlation_id", "correlation_id"),
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of data processing"""
        if self.records_processed == 0:
            return 0.0
        return ((self.records_inserted + self.records_updated) / self.records_processed) * 100
    
    @property
    def estimated_remaining_minutes(self) -> Optional[float]:
        """Estimate remaining time for running jobs"""
        if self.status != "running" or not self.started_at or self.progress_percentage == 0:
            return None
        
        elapsed_minutes = (datetime.utcnow() - self.started_at).total_seconds() / 60
        total_estimated_minutes = elapsed_minutes / (self.progress_percentage / 100)
        remaining_minutes = total_estimated_minutes - elapsed_minutes
        
        return max(0, remaining_minutes)
    
    def __repr__(self):
        return f"<DataSyncJob(id='{self.id}', source='{self.data_source}', status='{self.status}')>"


class DataProcessingJob(Base):
    """Data processing and transformation jobs"""
    
    __tablename__ = "data_processing_jobs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Job identification
    job_name = Column(String(200), nullable=False)
    job_type = Column(String(50), nullable=False)  # validation, cleaning, transformation, feature_engineering
    
    # Job status
    status = Column(String(20), default="queued", nullable=False)
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(100))
    
    # Input/Output datasets
    input_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    output_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    
    # Processing configuration
    processing_config = Column(JSONB, nullable=False)
    validation_rules = Column(JSONB)
    transformation_steps = Column(JSONB)
    
    # Results
    processing_results = Column(JSONB)
    validation_results = Column(JSONB)
    quality_metrics = Column(JSONB)
    
    # Records processing
    input_records_count = Column(Integer, nullable=False)
    output_records_count = Column(Integer, default=0)
    records_modified = Column(Integer, default=0)
    records_dropped = Column(Integer, default=0)
    
    # Error handling
    errors_count = Column(Integer, default=0)
    warnings_count = Column(Integer, default=0)
    error_details = Column(JSONB)
    
    # Performance metrics
    memory_usage_mb = Column(Float)
    processing_time_seconds = Column(Float)
    cpu_time_seconds = Column(Float)
    
    # Time tracking
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    
    # User tracking
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    correlation_id = Column(String(100), index=True)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    input_dataset = relationship("Dataset", foreign_keys=[input_dataset_id])
    output_dataset = relationship("Dataset", foreign_keys=[output_dataset_id])
    created_by_user = relationship("User", foreign_keys=[created_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100", name="check_processing_progress_range"),
        CheckConstraint("input_records_count > 0", name="check_input_records_positive"),
        CheckConstraint("output_records_count >= 0", name="check_output_records_non_negative"),
        CheckConstraint("job_type IN ('validation', 'cleaning', 'transformation', 'feature_engineering')", name="check_job_type"),
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="check_processing_status"),
        Index("idx_processing_job_status", "status", "created_at"),
        Index("idx_processing_job_type", "job_type", "created_at"),
        Index("idx_processing_job_dataset", "input_dataset_id"),
        Index("idx_processing_job_correlation", "correlation_id"),
    )
    
    @property
    def processing_rate_records_per_second(self) -> Optional[float]:
        """Calculate processing rate"""
        if self.processing_time_seconds and self.processing_time_seconds > 0:
            return self.output_records_count / self.processing_time_seconds
        return None
    
    @property
    def data_quality_improvement(self) -> Optional[float]:
        """Calculate data quality improvement if available"""
        if self.processing_results and 'quality_before' in self.processing_results and 'quality_after' in self.processing_results:
            before = self.processing_results['quality_before']
            after = self.processing_results['quality_after']
            return after - before
        return None
    
    def __repr__(self):
        return f"<DataProcessingJob(id='{self.id}', type='{self.job_type}', status='{self.status}')>"


# Additional utility models
class SystemConfiguration(Base):
    """System-wide configuration settings"""
    
    __tablename__ = "system_configurations"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration identification
    config_key = Column(String(100), nullable=False, unique=True, index=True)
    config_category = Column(String(50), nullable=False)
    
    # Configuration value
    config_value = Column(JSONB, nullable=False)
    default_value = Column(JSONB)
    
    # Metadata
    description = Column(Text)
    data_type = Column(String(20), nullable=False)  # string, integer, float, boolean, json
    is_sensitive = Column(Boolean, default=False)  # For secrets/passwords
    
    # Validation
    validation_rules = Column(JSONB)  # Min/max values, regex patterns, etc.
    
    # Management
    is_active = Column(Boolean, default=True)
    can_be_modified = Column(Boolean, default=True)
    requires_restart = Column(Boolean, default=False)
    
    # Change tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    updated_by_user = relationship("User", foreign_keys=[updated_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("data_type IN ('string', 'integer', 'float', 'boolean', 'json')", name="check_data_type"),
        Index("idx_config_category", "config_category"),
        Index("idx_config_active", "is_active"),
    )
    
    def __repr__(self):
        return f"<SystemConfiguration(key='{self.config_key}', category='{self.config_category}')>"


# Export all patient models
__all__ = [
    "PatientRiskFactor",
    "PatientOutcome", 
    "HealthcareProvider",
    "PatientProviderRelationship",
    "PatientCareEpisode",
    "DataSyncJob",
    "DataProcessingJob",
    "SystemConfiguration",
    "RiskFactorTypeEnum",
    "OutcomeTypeEnum"
]