import secrets
from functools import lru_cache
from typing import List, Optional, Union
from pathlib import Path

from pydantic import (
    BaseModel,
    BaseSettings,
    Field,
    PostgresDsn,
    RedisDsn,
    validator,
    root_validator
)
from pydantic.env_settings import SettingsSourceCallable


class DatabaseSettings(BaseModel):
    """Database configuration settings"""
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_USER: str = Field(default="cvd_user", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="cvd_password", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="cardiovascular_db", env="POSTGRES_DB")
    DATABASE_URL: Optional[PostgresDsn] = Field(default=None, env="DATABASE_URL")
    
    # Connection pool settings
    DB_POOL_SIZE: int = Field(default=10, ge=1, le=100)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100)
    DB_POOL_TIMEOUT: int = Field(default=30, ge=1, le=300)
    DB_POOL_RECYCLE: int = Field(default=3600, ge=300, le=86400)
    
    @root_validator
    def assemble_db_connection(cls, values):
        if isinstance(values.get("DATABASE_URL"), str):
            return values
        values["DATABASE_URL"] = PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            port=str(values.get("POSTGRES_PORT")),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
        return values


class RedisSettings(BaseModel):
    """Redis configuration settings"""
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_URL: Optional[RedisDsn] = Field(default=None, env="REDIS_URL")
    
    # Redis specific settings
    REDIS_TIMEOUT: int = Field(default=5, ge=1, le=60)
    REDIS_MAX_CONNECTIONS: int = Field(default=50, ge=1, le=1000)
    CACHE_TTL: int = Field(default=3600, ge=60, le=86400)  # 1 hour default
    
    @root_validator
    def assemble_redis_connection(cls, values):
        if isinstance(values.get("REDIS_URL"), str):
            return values
        
        password = values.get("REDIS_PASSWORD")
        auth_part = f":{password}@" if password else ""
        
        values["REDIS_URL"] = RedisDsn.build(
            scheme="redis",
            host=values.get("REDIS_HOST"),
            port=str(values.get("REDIS_PORT")),
            path=f"/{values.get('REDIS_DB') or 0}",
            password=password
        )
        return values


class SecuritySettings(BaseModel):
    """Security and authentication settings"""
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1, le=1440)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=30)
    
    # Password hashing
    BCRYPT_ROUNDS: int = Field(default=12, ge=4, le=20)
    
    # API Key settings
    API_KEY_EXPIRE_DAYS: int = Field(default=365, ge=1, le=3650)
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, ge=1, le=1000)
    RATE_LIMIT_BURST: int = Field(default=10, ge=1, le=100)


class MLSettings(BaseModel):
    """Machine Learning configuration settings"""
    MODEL_VERSION: str = Field(default="1.0.0")
    MODEL_PATH: Path = Field(default=Path("ml_models"))
    MODEL_RETRAIN_SCHEDULE: str = Field(default="0 2 * * 0")  # Weekly on Sunday at 2 AM
    MODEL_PERFORMANCE_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)
    AUTO_RETRAIN: bool = Field(default=False)
    
    # Model serving settings
    MODEL_CACHE_SIZE: int = Field(default=10, ge=1, le=100)
    PREDICTION_BATCH_SIZE: int = Field(default=1000, ge=1, le=10000)
    MODEL_TIMEOUT: int = Field(default=30, ge=1, le=300)
    
    # Feature engineering
    FEATURE_SELECTION_METHOD: str = Field(default="lasso", regex="^(lasso|ridge|mutual_info)$")
    CROSS_VALIDATION_FOLDS: int = Field(default=5, ge=3, le=10)
    
    @validator("MODEL_PATH")
    def validate_model_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class Settings(BaseSettings):
    """Main application settings"""
    
    # Basic application settings
    ENVIRONMENT: str = Field(default="development", regex="^(development|staging|production)$")
    DEBUG: bool = Field(default=False)
    API_VERSION: str = Field(default="v1")
    LOG_LEVEL: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000, ge=1000, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=32)
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    ALLOWED_HOSTS: List[str] = Field(default=["*"])
    
    # File upload settings
    UPLOAD_MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024)  # 50MB
    ALLOWED_EXTENSIONS: List[str] = Field(default=["csv", "json", "parquet"])
    DATA_STORAGE_PATH: Path = Field(default=Path("data"))
    
    # External API settings
    CDC_API_BASE_URL: str = Field(default="https://data.cdc.gov/api")
    CDC_API_KEY: Optional[str] = Field(default=None)
    MEDICARE_API_BASE_URL: Optional[str] = Field(default=None)
    MEDICARE_API_KEY: Optional[str] = Field(default=None)
    
    # Monitoring settings
    PROMETHEUS_ENABLED: bool = Field(default=True)
    METRICS_PORT: int = Field(default=8001, ge=1000, le=65535)
    HEALTH_CHECK_INTERVAL: int = Field(default=30, ge=5, le=300)
    SENTRY_DSN: Optional[str] = Field(default=None)
    
    # Email settings (for alerts)
    SMTP_HOST: Optional[str] = Field(default=None)
    SMTP_PORT: int = Field(default=587, ge=1, le=65535)
    SMTP_USERNAME: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    FROM_EMAIL: Optional[str] = Field(default=None)
    
    # Nested configuration objects
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("DATA_STORAGE_PATH", "ml")
    def validate_paths(cls, v):
        if hasattr(v, 'MODEL_PATH'):  # For MLSettings
            if isinstance(v.MODEL_PATH, str):
                v.MODEL_PATH = Path(v.MODEL_PATH)
        elif isinstance(v, str):  # For DATA_STORAGE_PATH
            return Path(v)
        return v
    
    @root_validator
    def validate_production_settings(cls, values):
        """Validate critical settings for production environment"""
        if values.get("ENVIRONMENT") == "production":
            # Ensure critical security settings are properly configured
            if values.get("DEBUG", False):
                raise ValueError("DEBUG must be False in production")
            
            security = values.get("security", {})
            if isinstance(security, dict):
                if security.get("SECRET_KEY") == "your-super-secret-key-change-this-in-production":
                    raise ValueError("SECRET_KEY must be changed from default in production")
                if security.get("JWT_SECRET_KEY") == "jwt-secret-key-change-in-production":
                    raise ValueError("JWT_SECRET_KEY must be changed from default in production")
            
            # Ensure CORS is properly configured
            cors_origins = values.get("CORS_ORIGINS", [])
            if "*" in cors_origins or "http://localhost:3000" in cors_origins:
                raise ValueError("CORS_ORIGINS must not include localhost or wildcard in production")
        
        return values
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = True
        validate_assignment = True
        
        # Custom settings source to handle nested objects
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


# Global settings instance cache
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience function to get database URL
def get_database_url() -> str:
    """Get database connection URL"""
    settings = get_settings()
    return str(settings.database.DATABASE_URL)


# Convenience function to get redis URL
def get_redis_url() -> str:
    """Get Redis connection URL"""
    settings = get_settings()
    return str(settings.redis.REDIS_URL)


# Application constants
class AppConstants:
    """Application-wide constants"""
    
    # API Response Messages
    API_SUCCESS_MESSAGE = "Operation completed successfully"
    API_ERROR_MESSAGE = "An error occurred while processing your request"
    
    # Model Constants
    SUPPORTED_MODEL_TYPES = ["linear", "ridge", "lasso", "xgboost", "lightgbm"]
    DEFAULT_MODEL_TYPE = "ridge"
    
    # Data Processing Constants
    MAX_PREDICTION_BATCH_SIZE = 1000
    MIN_TRAINING_DATA_SIZE = 100
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    
    # Health Check Constants
    HEALTH_CHECK_TIMEOUT = 5
    CRITICAL_SERVICES = ["database", "redis", "ml_model"]
    
    # Cache Keys
    CACHE_KEY_MODEL = "ml_model:{version}"
    CACHE_KEY_FEATURES = "features:{version}"
    CACHE_KEY_PREDICTIONS = "predictions:{hash}"
    
    # File Processing
    CHUNK_SIZE = 10000  # For processing large CSV files
    MAX_CONCURRENT_PREDICTIONS = 100


# Export settings and constants
__all__ = [
    "Settings",
    "DatabaseSettings", 
    "RedisSettings",
    "SecuritySettings",
    "MLSettings",
    "get_settings",
    "get_database_url",
    "get_redis_url",
    "AppConstants"
]