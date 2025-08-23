from datetime import datetime, date
from typing import Optional, List, Dict, Any
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


class GenderEnum(str, enum.Enum):
    """Gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class RaceEthnicityEnum(str, enum.Enum):
    """Race/Ethnicity enumeration based on CDC categories"""
    WHITE_NON_HISPANIC = "white_non_hispanic"
    BLACK_NON_HISPANIC = "black_non_hispanic"
    HISPANIC = "hispanic"
    ASIAN_NON_HISPANIC = "asian_non_hispanic"
    AMERICAN_INDIAN_ALASKA_NATIVE = "american_indian_alaska_native"
    NATIVE_HAWAIIAN_PACIFIC_ISLANDER = "native_hawaiian_pacific_islander"
    MULTIRACIAL = "multiracial"
    OTHER = "other"
    UNKNOWN = "unknown"


class Patient(Base):
    """Patient demographics and basic information"""
    
    __tablename__ = "patients"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient identification
    external_id = Column(String(100), unique=True, index=True)  # External system ID
    medicare_id = Column(String(50), unique=True, index=True)  # Medicare beneficiary ID
    mrn = Column(String(50))  # Medical Record Number (if available)
    
    # Demographics
    age = Column(Integer, nullable=False)
    date_of_birth = Column(Date)
    gender = Column(SQLEnum(GenderEnum), nullable=False)
    race_ethnicity = Column(SQLEnum(RaceEthnicityEnum))
    
    # Geographic information
    state_code = Column(String(2))  # US state abbreviation
    zip_code = Column(String(10))
    county_fips = Column(String(5))  # FIPS county code
    urban_rural_code = Column(String(1))  # Urban/Rural classification
    
    # Insurance information
    medicare_part_a = Column(Boolean, default=True)
    medicare_part_b = Column(Boolean, default=True)
    medicare_part_c = Column(Boolean, default=False)  # Medicare Advantage
    medicare_part_d = Column(Boolean, default=False)  # Prescription drugs
    dual_eligible = Column(Boolean, default=False)  # Medicare + Medicaid
    
    # Basic health indicators
    height_cm = Column(Float)
    weight_kg = Column(Float)
    bmi = Column(Float)
    smoking_status = Column(String(20))  # never, former, current, unknown
    
    # Contact and consent
    consent_date = Column(DateTime(timezone=True))
    data_sharing_consent = Column(Boolean, default=False)
    
    # Data quality and completeness
    data_completeness_score = Column(Float)  # 0-1 score
    last_verified = Column(DateTime(timezone=True))
    verification_source = Column(String(100))
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Privacy and security
    is_anonymized = Column(Boolean, default=True)
    anonymization_date = Column(DateTime(timezone=True))
    retention_until = Column(DateTime(timezone=True))
    
    # Relationships
    predictions = relationship("Prediction", back_populates="patient")
    medical_history = relationship("PatientMedicalHistory", back_populates="patient")
    vitals = relationship("PatientVitals", back_populates="patient")
    medications = relationship("PatientMedication", back_populates="patient")
    created_by_user = relationship("User", foreign_keys=[created_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("age >= 0 AND age <= 150", name="check_age_range"),
        CheckConstraint("bmi >= 10 AND bmi <= 100", name="check_bmi_range"),
        CheckConstraint("data_completeness_score >= 0 AND data_completeness_score <= 1", name="check_completeness_range"),
        CheckConstraint("height_cm >= 50 AND height_cm <= 300", name="check_height_range"),
        CheckConstraint("weight_kg >= 20 AND weight_kg <= 500", name="check_weight_range"),
        Index("idx_patient_age_gender", "age", "gender"),
        Index("idx_patient_state_zip", "state_code", "zip_code"),
        Index("idx_patient_created_at", "created_at"),
        Index("idx_patient_external_id", "external_id"),
    )
    
    @validates('smoking_status')
    def validate_smoking_status(self, key, smoking_status):
        """Validate smoking status"""
        allowed_statuses = ['never', 'former', 'current', 'unknown']
        if smoking_status and smoking_status not in allowed_statuses:
            raise ValueError(f"Smoking status must be one of: {allowed_statuses}")
        return smoking_status
    
    @property
    def calculated_bmi(self) -> Optional[float]:
        """Calculate BMI from height and weight"""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 2)
        return None
    
    def __repr__(self):
        return f"<Patient(id='{self.id}', age={self.age}, gender='{self.gender}')>"


class PatientMedicalHistory(Base):
    """Patient medical history and conditions"""
    
    __tablename__ = "patient_medical_history"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient reference
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Cardiovascular conditions
    has_hypertension = Column(Boolean, default=False)
    has_diabetes = Column(Boolean, default=False)
    has_heart_disease = Column(Boolean, default=False)
    has_stroke_history = Column(Boolean, default=False)
    has_heart_attack_history = Column(Boolean, default=False)
    has_atrial_fibrillation = Column(Boolean, default=False)
    has_heart_failure = Column(Boolean, default=False)
    
    # Other relevant conditions
    has_kidney_disease = Column(Boolean, default=False)
    has_copd = Column(Boolean, default=False)
    has_depression = Column(Boolean, default=False)
    has_dementia = Column(Boolean, default=False)
    has_osteoporosis = Column(Boolean, default=False)
    
    # Condition details
    hypertension_diagnosed_date = Column(Date)
    diabetes_type = Column(String(20))  # type1, type2, gestational, other
    diabetes_diagnosed_date = Column(Date)
    heart_disease_type = Column(String(100))
    
    # Family history
    family_history_heart_disease = Column(Boolean, default=False)
    family_history_stroke = Column(Boolean, default=False)
    family_history_diabetes = Column(Boolean, default=False)
    
    # Risk factors
    cholesterol_level = Column(String(20))  # normal, borderline, high
    blood_pressure_category = Column(String(20))  # normal, elevated, stage1, stage2
    
    # Hospitalization history
    cardiovascular_hospitalizations_last_year = Column(Integer, default=0)
    total_hospitalizations_last_year = Column(Integer, default=0)
    last_hospitalization_date = Column(Date)
    last_hospitalization_reason = Column(String(200))
    
    # Tracking
    record_date = Column(Date, nullable=False, default=date.today)
    data_source = Column(String(100))  # medicare_claims, ehr, patient_report, etc.
    verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_history")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("cardiovascular_hospitalizations_last_year >= 0", name="check_cv_hospitalizations"),
        CheckConstraint("total_hospitalizations_last_year >= 0", name="check_total_hospitalizations"),
        CheckConstraint("cardiovascular_hospitalizations_last_year <= total_hospitalizations_last_year", name="check_cv_vs_total_hospitalizations"),
        Index("idx_medical_history_patient_date", "patient_id", "record_date"),
        Index("idx_medical_history_conditions", "has_hypertension", "has_diabetes", "has_heart_disease"),
        UniqueConstraint("patient_id", "record_date", name="uq_patient_medical_history_date"),
    )
    
    @property
    def cardiovascular_conditions_count(self) -> int:
        """Count of cardiovascular conditions"""
        conditions = [
            self.has_hypertension,
            self.has_diabetes,
            self.has_heart_disease,
            self.has_stroke_history,
            self.has_heart_attack_history,
            self.has_atrial_fibrillation,
            self.has_heart_failure
        ]
        return sum(1 for condition in conditions if condition)
    
    def __repr__(self):
        return f"<PatientMedicalHistory(patient_id='{self.patient_id}', cv_conditions={self.cardiovascular_conditions_count})>"


class PatientVitals(Base):
    """Patient vital signs and measurements"""
    
    __tablename__ = "patient_vitals"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient reference
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Vital signs
    systolic_bp = Column(Integer)  # mmHg
    diastolic_bp = Column(Integer)  # mmHg
    heart_rate = Column(Integer)  # bpm
    temperature_celsius = Column(Float)
    respiratory_rate = Column(Integer)  # breaths per minute
    oxygen_saturation = Column(Float)  # percentage
    
    # Laboratory values
    total_cholesterol = Column(Float)  # mg/dL
    ldl_cholesterol = Column(Float)  # mg/dL
    hdl_cholesterol = Column(Float)  # mg/dL
    triglycerides = Column(Float)  # mg/dL
    blood_glucose = Column(Float)  # mg/dL
    hba1c = Column(Float)  # percentage
    
    # Kidney function
    creatinine = Column(Float)  # mg/dL
    egfr = Column(Float)  # mL/min/1.73m²
    
    # Cardiac markers
    troponin = Column(Float)  # ng/mL
    bnp = Column(Float)  # pg/mL
    nt_probnp = Column(Float)  # pg/mL
    
    # Measurement metadata
    measurement_date = Column(DateTime(timezone=True), nullable=False)
    measurement_type = Column(String(50), default="routine")  # routine, emergency, followup
    measured_by = Column(String(100))  # Provider or system that took measurement
    
    # Data quality
    data_source = Column(String(100))  # ehr, lab, device, patient_report
    verified = Column(Boolean, default=False)
    abnormal_flags = Column(JSONB)  # Flags for abnormal values
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="vitals")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("systolic_bp >= 50 AND systolic_bp <= 300", name="check_systolic_bp_range"),
        CheckConstraint("diastolic_bp >= 30 AND diastolic_bp <= 200", name="check_diastolic_bp_range"),
        CheckConstraint("heart_rate >= 30 AND heart_rate <= 220", name="check_heart_rate_range"),
        CheckConstraint("oxygen_saturation >= 50 AND oxygen_saturation <= 100", name="check_oxygen_sat_range"),
        CheckConstraint("total_cholesterol >= 50 AND total_cholesterol <= 500", name="check_cholesterol_range"),
        CheckConstraint("blood_glucose >= 30 AND blood_glucose <= 600", name="check_glucose_range"),
        CheckConstraint("hba1c >= 3 AND hba1c <= 20", name="check_hba1c_range"),
        Index("idx_vitals_patient_date", "patient_id", "measurement_date"),
        Index("idx_vitals_measurement_date", "measurement_date"),
        Index("idx_vitals_abnormal", "abnormal_flags"),
    )
    
    @property
    def blood_pressure_category(self) -> str:
        """Categorize blood pressure based on AHA guidelines"""
        if not self.systolic_bp or not self.diastolic_bp:
            return "unknown"
        
        if self.systolic_bp < 120 and self.diastolic_bp < 80:
            return "normal"
        elif self.systolic_bp < 130 and self.diastolic_bp < 80:
            return "elevated"
        elif (120 <= self.systolic_bp <= 129) or (80 <= self.diastolic_bp <= 89):
            return "stage1_hypertension"
        elif self.systolic_bp >= 130 or self.diastolic_bp >= 90:
            return "stage2_hypertension"
        else:
            return "unknown"
    
    @property
    def cholesterol_ratio(self) -> Optional[float]:
        """Calculate total cholesterol to HDL ratio"""
        if self.total_cholesterol and self.hdl_cholesterol and self.hdl_cholesterol > 0:
            return round(self.total_cholesterol / self.hdl_cholesterol, 2)
        return None
    
    def __repr__(self):
        return f"<PatientVitals(patient_id='{self.patient_id}', date='{self.measurement_date}')>"


class PatientMedication(Base):
    """Patient medication history and current medications"""
    
    __tablename__ = "patient_medications"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Patient reference
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    
    # Medication information
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200))
    ndc_code = Column(String(20))  # National Drug Code
    therapeutic_class = Column(String(100))
    
    # Dosage and administration
    dosage = Column(String(100))
    frequency = Column(String(50))
    route = Column(String(50))  # oral, injection, topical, etc.
    
    # Medication period
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    is_current = Column(Boolean, default=True)
    
    # Cardiovascular relevance
    is_cardiovascular_medication = Column(Boolean, default=False)
    medication_category = Column(String(50))  # ace_inhibitor, beta_blocker, diuretic, statin, etc.
    
    # Adherence and effectiveness
    adherence_percentage = Column(Float)  # 0-100
    side_effects = Column(JSONB)
    effectiveness_rating = Column(Integer)  # 1-5 scale
    
    # Prescriber information
    prescribed_by = Column(String(200))
    prescriber_npi = Column(String(20))  # National Provider Identifier
    prescription_date = Column(Date)
    
    # Data source and quality
    data_source = Column(String(100))  # medicare_part_d, ehr, patient_report
    verified = Column(Boolean, default=False)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("adherence_percentage >= 0 AND adherence_percentage <= 100", name="check_adherence_range"),
        CheckConstraint("effectiveness_rating >= 1 AND effectiveness_rating <= 5", name="check_effectiveness_range"),
        Index("idx_medication_patient_current", "patient_id", "is_current"),
        Index("idx_medication_category", "medication_category"),
        Index("idx_medication_cv_relevant", "is_cardiovascular_medication"),
        Index("idx_medication_start_end", "start_date", "end_date"),
    )
    
    def __repr__(self):
        return f"<PatientMedication(patient_id='{self.patient_id}', medication='{self.medication_name}')>"


class Dataset(Base):
    """Datasets used for training and validation"""
    
    __tablename__ = "datasets"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Dataset identification
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text)
    version = Column(String(50), default="1.0")
    
    # Dataset type and purpose
    dataset_type = Column(String(20), nullable=False)  # training, validation, test
    source_type = Column(String(50), nullable=False)  # cdc, medicare, upload, manual
    
    # File information
    original_filename = Column(String(500))
    stored_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer)
    file_format = Column(String(20))  # csv, json, parquet
    
    # Data characteristics
    total_records = Column(Integer, nullable=False)
    total_columns = Column(Integer, nullable=False)
    column_names = Column(JSONB)
    data_types = Column(JSONB)
    
    # Data quality metrics
    completeness_score = Column(Float)  # 0-1
    missing_values_count = Column(Integer, default=0)
    duplicate_records_count = Column(Integer, default=0)
    outliers_count = Column(Integer, default=0)
    
    # Processing status
    processing_status = Column(String(20), default="uploaded")  # uploaded, processing, processed, failed
    validation_status = Column(String(20), default="pending")  # pending, passed, failed
    validation_results = Column(JSONB)
    
    # Date ranges in data
    data_start_date = Column(Date)
    data_end_date = Column(Date)
    geographic_coverage = Column(JSONB)  # States/regions covered
    
    # Usage tracking
    used_for_training = Column(Boolean, default=False)
    last_used_date = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Metadata and tags
    tags = Column(JSONB)
    metadata = Column(JSONB)
    schema_version = Column(String(20), default="1.0")
    
    # Access control
    is_public = Column(Boolean, default=False)
    access_level = Column(String(20), default="private")  # private, internal, public
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Data lineage
    parent_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    derivation_method = Column(String(100))  # filtered, aggregated, joined, etc.
    
    # Relationships
    created_by_user = relationship("User", foreign_keys=[created_by])
    parent_dataset = relationship("Dataset", remote_side=[id])
    training_jobs = relationship("ModelTrainingJob", foreign_keys="ModelTrainingJob.training_dataset_id")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("total_records > 0", name="check_total_records_positive"),
        CheckConstraint("total_columns > 0", name="check_total_columns_positive"),
        CheckConstraint("completeness_score >= 0 AND completeness_score <= 1", name="check_completeness_range"),
        CheckConstraint("missing_values_count >= 0", name="check_missing_values_non_negative"),
        CheckConstraint("usage_count >= 0", name="check_usage_count_non_negative"),
        CheckConstraint("dataset_type IN ('training', 'validation', 'test')", name="check_dataset_type"),
        CheckConstraint("processing_status IN ('uploaded', 'processing', 'processed', 'failed')", name="check_processing_status"),
        CheckConstraint("validation_status IN ('pending', 'passed', 'failed')", name="check_validation_status"),
        Index("idx_dataset_type_status", "dataset_type", "processing_status"),
        Index("idx_dataset_created_by", "created_by", "created_at"),
        Index("idx_dataset_name", "name"),
        Index("idx_dataset_source_type", "source_type"),
    )
    
    @property
    def data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        scores = []
        
        # Completeness score
        if self.completeness_score is not None:
            scores.append(self.completeness_score)
        
        # Missing values penalty
        if self.total_records and self.missing_values_count is not None:
            missing_rate = self.missing_values_count / (self.total_records * self.total_columns)
            scores.append(1.0 - missing_rate)
        
        # Duplicate records penalty
        if self.total_records and self.duplicate_records_count is not None:
            duplicate_rate = self.duplicate_records_count / self.total_records
            scores.append(1.0 - duplicate_rate)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', type='{self.dataset_type}', records={self.total_records})>"


class User(Base):
    """System users for authentication and authorization"""
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User identification
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(320), nullable=False, unique=True, index=True)
    full_name = Column(String(200))
    
    # Authentication
    hashed_password = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Authorization
    roles = Column(JSONB, default=list)  # List of role names
    permissions = Column(JSONB, default=list)  # List of specific permissions
    
    # Profile information
    organization = Column(String(200))
    department = Column(String(100))
    job_title = Column(String(100))
    phone_number = Column(String(20))
    
    # Account management
    last_login = Column(DateTime(timezone=True))
    password_changed_at = Column(DateTime(timezone=True))
    account_locked_until = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    
    # API access
    api_quota_daily = Column(Integer, default=1000)
    api_usage_today = Column(Integer, default=0)
    api_usage_reset_date = Column(Date, default=date.today)
    
    # Privacy and compliance
    data_retention_until = Column(DateTime(timezone=True))
    gdpr_consent = Column(Boolean, default=False)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("failed_login_attempts >= 0", name="check_failed_attempts_non_negative"),
        CheckConstraint("api_quota_daily > 0", name="check_api_quota_positive"),
        CheckConstraint("api_usage_today >= 0", name="check_api_usage_non_negative"),
        Index("idx_user_email", "email"),
        Index("idx_user_active", "is_active"),
        Index("idx_user_last_login", "last_login"),
    )
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin role"""
        return "admin" in (self.roles or [])
    
    @property
    def can_retrain_models(self) -> bool:
        """Check if user can trigger model retraining"""
        allowed_roles = ["admin", "ml_engineer", "data_scientist"]
        return any(role in (self.roles or []) for role in allowed_roles)
    
    @property
    def api_quota_remaining(self) -> int:
        """Calculate remaining API quota for today"""
        # Reset usage if it's a new day
        if self.api_usage_reset_date != date.today():
            return self.api_quota_daily
        
        return max(0, self.api_quota_daily - self.api_usage_today)
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}', active={self.is_active})>"


class APIKey(Base):
    """API keys for external system access"""
    
    __tablename__ = "api_keys"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Key information
    name = Column(String(200), nullable=False)
    description = Column(Text)
    key_hash = Column(String(128), nullable=False, unique=True, index=True)
    key_prefix = Column(String(20), nullable=False)  # First few characters for identification
    
    # Permissions and access
    permissions = Column(JSONB, default=list)  # List of allowed permissions
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    allowed_ips = Column(JSONB)  # List of allowed IP addresses
    
    # Status and lifecycle
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Tracking
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    created_by_user = relationship("User", foreign_keys=[created_by])
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("rate_limit > 0", name="check_rate_limit_positive"),
        CheckConstraint("usage_count >= 0", name="check_usage_count_non_negative"),
        Index("idx_api_key_active", "is_active"),
        Index("idx_api_key_expires", "expires_at"),
        Index("idx_api_key_created_by", "created_by"),
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if API key is valid for use"""
        return self.is_active and not self.is_expired
    
    def __repr__(self):
        return f"<APIKey(name='{self.name}', prefix='{self.key_prefix}', active={self.is_active})>"


# Import User model to ensure it's available for relationships
# Note: In a real application, User might be in a separate auth module
# but for this example, we'll include it here to complete the relationships

# Export all models
__all__ = [
    "PredictionModel",
    "Prediction", 
    "PredictionBatch",
    "ModelPerformanceHistory",
    "PredictionFeedback",
    "ModelTrainingJob",
    "PredictionAuditLog",
    "Patient",
    "PatientMedicalHistory",
    "PatientVitals",
    "PatientMedication",
    "Dataset",
    "User",
    "APIKey",
    "GenderEnum",
    "RaceEthnicityEnum"
]