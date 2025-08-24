import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# FastAPI and database imports
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Application imports
from app.main import app
from app.core.database import Base, get_db
from app.core.config import get_settings
from app.services.ml_service import MLService
from app.services.prediction_service import PredictionService
from app.services.data_service import DataService

# Configure logging for tests
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Test database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 20
    },
    poolclass=StaticPool,
    echo=False  # Set to True for SQL debugging
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override database dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Set up test database before all tests."""
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up after all tests
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Create a fresh database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def async_client():
    """Create an async test client for the FastAPI application."""
    from httpx import AsyncClient
    
    async def _async_client():
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    return _async_client


# Sample data fixtures
@pytest.fixture
def sample_patient_data():
    """Provide sample patient data for testing."""
    return {
        "Age": 65,
        "Sex": "Male",
        "BMI": 28.5,
        "Smoking": True,
        "HighBP": True,
        "HighChol": True,
        "Diabetes": False,
        "PhysActivity": True,
        "HvyAlcoholConsump": False,
        "PhysHlth": 5,
        "MentHlth": 2,
        "NoDocbcCost": False,
        "GenHlth": "Good",
        "DiffWalk": False,
        "Stroke": False
    }


@pytest.fixture
def sample_batch_patient_data():
    """Provide batch patient data for testing."""
    base_data = {
        "Age": 65,
        "Sex": "Male", 
        "BMI": 28.5,
        "Smoking": True,
        "HighBP": True,
        "HighChol": True,
        "Diabetes": False,
        "PhysActivity": True,
        "HvyAlcoholConsump": False
    }
    
    return [
        base_data,
        {**base_data, "Age": 45, "Sex": "Female", "HighBP": False},
        {**base_data, "Age": 70, "Diabetes": True, "Smoking": False},
        {**base_data, "Age": 55, "BMI": 35.0, "PhysActivity": False}
    ]


@pytest.fixture
def sample_prediction_request():
    """Provide sample prediction request for API testing."""
    return {
        "patient_data": {
            "Age": 65,
            "Sex": "Male",
            "BMI": 28.5,
            "Smoking": True,
            "HighBP": True,
            "HighChol": True,
            "Diabetes": False,
            "PhysActivity": True,
            "HvyAlcoholConsump": False
        },
        "model_version": "v1.0.0",
        "return_explanation": True,
        "save_to_database": False
    }


@pytest.fixture
def sample_batch_prediction_request():
    """Provide sample batch prediction request."""
    base_patient = {
        "Age": 65,
        "Sex": "Male",
        "BMI": 28.5,
        "Smoking": True,
        "HighBP": True,
        "HighChol": True,
        "Diabetes": False,
        "PhysActivity": True,
        "HvyAlcoholConsump": False
    }
    
    return {
        "predictions": [
            {"patient_data": base_patient},
            {"patient_data": {**base_patient, "Age": 45, "Sex": "Female"}},
            {"patient_data": {**base_patient, "Age": 70, "Diabetes": True}}
        ],
        "batch_id": "test_batch_001",
        "model_version": "v1.0.0",
        "return_explanations": True
    }


# Mock fixtures for ML components
@pytest.fixture
def mock_trained_model():
    """Provide a mock trained ML model."""
    model = Mock()
    model.predict.return_value = np.array([1, 0, 1])
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
    model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01])
    return model


@pytest.fixture
def mock_preprocessing_artifacts():
    """Provide mock preprocessing artifacts."""
    scaler = Mock()
    scaler.transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.2]])
    
    return {
        'scaler': scaler,
        'label_encoders': {
            'Sex': Mock(),
            'GenHlth': Mock()
        },
        'feature_names': [
            'Age', 'BMI', 'HighBP', 'Smoking', 'Sex', 'HighChol', 
            'Diabetes', 'PhysActivity', 'HvyAlcoholConsump', 'PhysHlth'
        ],
        'best_model_name': 'random_forest',
        'model_version': 'v1.0.0'
    }


@pytest.fixture
def mock_ml_service(mock_trained_model, mock_preprocessing_artifacts):
    """Provide a mock ML service."""
    with patch('joblib.load', return_value=mock_trained_model):
        with patch('builtins.open', create=True):
            with patch('pickle.load', return_value=mock_preprocessing_artifacts):
                service = MLService()
                service.model = mock_trained_model
                service.artifacts = mock_preprocessing_artifacts
                service.is_loaded = True
                yield service


@pytest.fixture
def mock_prediction_service(db_session):
    """Provide a mock prediction service."""
    service = PredictionService(db_session)
    return service


@pytest.fixture
def mock_data_service():
    """Provide a mock data service."""
    service = Mock(spec=DataService)
    
    # Mock data statistics
    service.get_data_statistics.return_value = {
        "total_records": 100000,
        "date_range": {
            "start": "2020-01-01",
            "end": "2024-01-01"
        },
        "feature_statistics": {
            "Age": {"mean": 58.2, "std": 12.5, "min": 18, "max": 100},
            "BMI": {"mean": 26.8, "std": 4.2, "min": 15.0, "max": 50.0}
        },
        "target_distribution": {
            "positive_cases": 15000,
            "negative_cases": 85000,
            "positive_rate": 0.15
        }
    }
    
    # Mock data validation
    service.validate_data_quality.return_value = {
        "is_valid": True,
        "quality_score": 0.92,
        "issues": [],
        "recommendations": [
            "Consider additional feature engineering for Age variable"
        ]
    }
    
    return service


# Test data fixtures
@pytest.fixture
def sample_cardiovascular_dataset():
    """Provide a sample cardiovascular dataset for testing."""
    np.random.seed(42)  # For reproducible test data
    
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 100, n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.random.normal(26.8, 4.2, n_samples),
        'Smoking': np.random.choice([True, False], n_samples),
        'HighBP': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'HighChol': np.random.choice([True, False], n_samples, p=[0.25, 0.75]),
        'Diabetes': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'PhysActivity': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
        'HvyAlcoholConsump': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'PhysHlth': np.random.randint(0, 31, n_samples),
        'MentHlth': np.random.randint(0, 31, n_samples)
    }
    
    # Create target variable with some logic
    heart_disease_prob = (
        (data['Age'] > 60).astype(int) * 0.3 +
        data['Smoking'].astype(int) * 0.2 +
        data['HighBP'].astype(int) * 0.15 +
        data['Diabetes'].astype(int) * 0.1 +
        (data['BMI'] > 30).astype(int) * 0.1
    )
    
    data['HeartDisease'] = np.random.binomial(1, np.clip(heart_disease_prob, 0, 1), n_samples)
    
    return pd.DataFrame(data)


# File system fixtures
@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_model_file(temp_directory, mock_trained_model):
    """Provide a temporary model file for testing."""
    import joblib
    
    model_path = temp_directory / "test_model.pkl"
    joblib.dump(mock_trained_model, model_path)
    yield model_path


@pytest.fixture
def temp_artifacts_file(temp_directory, mock_preprocessing_artifacts):
    """Provide a temporary artifacts file for testing."""
    import pickle
    
    artifacts_path = temp_directory / "test_artifacts.pkl"
    with open(artifacts_path, 'wb') as f:
        pickle.dump(mock_preprocessing_artifacts, f)
    yield artifacts_path


@pytest.fixture
def temp_data_file(temp_directory, sample_cardiovascular_dataset):
    """Provide a temporary data file for testing."""
    data_path = temp_directory / "test_data.csv"
    sample_cardiovascular_dataset.to_csv(data_path, index=False)
    yield data_path


# Environment and configuration fixtures
@pytest.fixture
def test_settings():
    """Provide test settings configuration."""
    settings = get_settings()
    # Override settings for testing
    settings.TESTING = True
    settings.DATABASE_URL = SQLALCHEMY_DATABASE_URL
    settings.MODEL_PATH = "test_models/"
    settings.LOG_LEVEL = "WARNING"
    return settings


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", SQLALCHEMY_DATABASE_URL)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MODEL_PATH", "test_models/")


# API testing fixtures
@pytest.fixture
def api_headers():
    """Provide standard headers for API testing."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Add API key if authentication is required
        # "X-API-Key": "test-api-key-123"
    }


@pytest.fixture
def authenticated_headers(api_headers):
    """Provide authenticated headers for API testing."""
    headers = api_headers.copy()
    headers["Authorization"] = "Bearer test-token-123"
    return headers


# Time-based fixtures
@pytest.fixture
def freeze_time():
    """Freeze time for predictable testing."""
    frozen_time = datetime(2024, 1, 15, 12, 0, 0)
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = frozen_time
        mock_datetime.now.return_value = frozen_time
        yield frozen_time


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Provide a performance timer for testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Error simulation fixtures
@pytest.fixture
def database_error_simulation():
    """Simulate database errors for testing."""
    def simulate_error():
        raise Exception("Database connection failed")
    return simulate_error


@pytest.fixture
def model_error_simulation():
    """Simulate model errors for testing."""
    def simulate_error():
        raise Exception("Model prediction failed")
    return simulate_error


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests as ML/model tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add markers to tests based on their location/name
    for item in items:
        # Mark API tests
        if "test_api" in item.nodeid or "api" in item.name.lower():
            item.add_marker(pytest.mark.api)
        
        # Mark ML tests
        if "test_ml" in item.nodeid or "ml" in item.name.lower():
            item.add_marker(pytest.mark.ml)
        
        # Mark database tests
        if "db" in item.name.lower() or "database" in item.name.lower():
            item.add_marker(pytest.mark.database)
        
        # Mark slow tests
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Utility functions for tests
def assert_prediction_response_structure(response_data):
    """Assert that prediction response has correct structure."""
    required_fields = ['prediction', 'probability', 'model_version', 'timestamp']
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    assert isinstance(response_data['prediction'], int)
    assert response_data['prediction'] in [0, 1]
    assert isinstance(response_data['probability'], (int, float))
    assert 0 <= response_data['probability'] <= 1
    assert isinstance(response_data['model_version'], str)


def assert_batch_prediction_response_structure(response_data):
    """Assert that batch prediction response has correct structure."""
    required_fields = ['batch_id', 'predictions', 'summary', 'timestamp']
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    assert isinstance(response_data['predictions'], list)
    assert len(response_data['predictions']) > 0
    
    # Check each individual prediction
    for prediction in response_data['predictions']:
        assert_prediction_response_structure(prediction)
    
    # Check summary structure
    summary = response_data['summary']
    summary_fields = ['total_predictions', 'high_risk_count', 'average_probability']
    for field in summary_fields:
        assert field in summary, f"Missing summary field: {field}"


def create_mock_prediction_result(prediction=1, probability=0.75, model_version="v1.0.0"):
    """Create a mock prediction result for testing."""
    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat(),
        "correlation_id": "test-correlation-id-123"
    }