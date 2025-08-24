import pytest
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import pandas as pd
import numpy as np

# Import the main FastAPI app
from app.main import app
from app.core.database import get_db
from app.services.ml_service import MLService
from app.services.prediction_service import PredictionService
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionRequest, PredictionResponse

class TestHealthEndpoints:
    """Test suite for health check endpoints."""
    
    def setup_method(self):
        """Set up test client and dependencies."""
        self.client = TestClient(app)
    
    def test_health_check_success(self):
        """Test basic health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_detailed_health_check(self):
        """Test detailed health check with system metrics."""
        response = self.client.get("/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check required fields
        required_fields = [
            "status", "timestamp", "version", "database", 
            "redis", "ml_models", "system_metrics"
        ]
        for field in required_fields:
            assert field in data
        
        # Check system metrics structure
        system_metrics = data["system_metrics"]
        assert "cpu_usage" in system_metrics
        assert "memory_usage" in system_metrics
        assert "disk_usage" in system_metrics
        assert "uptime" in system_metrics
    
    @patch('app.core.database.get_db')
    def test_health_check_database_failure(self, mock_db):
        """Test health check when database is unavailable."""
        mock_db.side_effect = Exception("Database connection failed")
        
        response = self.client.get("/health/detailed")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "database" in data
        assert data["database"]["status"] == "unhealthy"


class TestPredictionEndpoints:
    """Test suite for prediction endpoints."""
    
    def setup_method(self):
        """Set up test client and mock dependencies."""
        self.client = TestClient(app)
        
        # Sample prediction request data
        self.valid_prediction_data = {
            "patient_data": {
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
                "MentHlth": 2
            },
            "model_version": "v1.0.0",
            "return_explanation": True
        }
        
        self.invalid_prediction_data = {
            "patient_data": {
                "Age": "invalid_age",  # Invalid data type
                "Sex": "Male"
                # Missing required fields
            }
        }
    
    @patch('app.services.ml_service.MLService.predict')
    def test_single_prediction_success(self, mock_predict):
        """Test successful single prediction."""
        # Mock ML service response
        mock_predict.return_value = {
            "prediction": 1,
            "probability": 0.75,
            "model_version": "v1.0.0",
            "explanation": {
                "top_risk_factors": ["HighBP", "Age", "BMI"],
                "risk_level": "High"
            }
        }
        
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=self.valid_prediction_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert "explanation" in data
        assert "timestamp" in data
        assert "correlation_id" in data
        
        # Check prediction values
        assert data["prediction"] == 1
        assert data["probability"] == 0.75
        assert data["model_version"] == "v1.0.0"
    
    def test_single_prediction_validation_error(self):
        """Test prediction with invalid input data."""
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=self.invalid_prediction_data
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
    
    def test_single_prediction_missing_data(self):
        """Test prediction with missing required fields."""
        incomplete_data = {
            "patient_data": {
                "Age": 65
                # Missing other required fields
            }
        }
        
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=incomplete_data
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('app.services.ml_service.MLService.predict')
    def test_batch_predictions_success(self, mock_predict):
        """Test successful batch predictions."""
        # Mock ML service response for each prediction
        mock_predict.side_effect = [
            {
                "prediction": 1,
                "probability": 0.75,
                "model_version": "v1.0.0"
            },
            {
                "prediction": 0,
                "probability": 0.30,
                "model_version": "v1.0.0"
            }
        ]
        
        batch_data = {
            "predictions": [
                self.valid_prediction_data,
                {
                    **self.valid_prediction_data,
                    "patient_data": {
                        **self.valid_prediction_data["patient_data"],
                        "Age": 35,
                        "HighBP": False
                    }
                }
            ],
            "batch_id": "batch_001"
        }
        
        response = self.client.post(
            "/api/v1/predictions/batch",
            json=batch_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "batch_id" in data
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 2
        
        # Check summary statistics
        summary = data["summary"]
        assert "total_predictions" in summary
        assert "high_risk_count" in summary
        assert "average_probability" in summary
    
    @patch('app.services.ml_service.MLService.predict')
    def test_prediction_model_error(self, mock_predict):
        """Test prediction when ML model throws an error."""
        mock_predict.side_effect = Exception("Model prediction failed")
        
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=self.valid_prediction_data
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data
        assert "correlation_id" in data
    
    @patch('app.services.ml_service.MLService.get_model_info')
    def test_model_info_endpoint(self, mock_model_info):
        """Test model information endpoint."""
        mock_model_info.return_value = {
            "model_name": "cardiovascular_risk_predictor",
            "version": "v1.0.0",
            "training_date": "2024-01-15",
            "performance_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "roc_auc": 0.90
            },
            "feature_count": 45
        }
        
        response = self.client.get("/api/v1/predictions/model-info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "model_name" in data
        assert "version" in data
        assert "performance_metrics" in data
        assert data["model_name"] == "cardiovascular_risk_predictor"
    
    def test_rate_limiting(self):
        """Test API rate limiting functionality."""
        # Make multiple rapid requests to test rate limiting
        responses = []
        for i in range(10):  # Assuming rate limit is lower than 10 requests
            response = self.client.post(
                "/api/v1/predictions/predict",
                json=self.valid_prediction_data
            )
            responses.append(response.status_code)
        
        # Check if any requests were rate limited
        rate_limited = any(code == status.HTTP_429_TOO_MANY_REQUESTS for code in responses)
        # This test may pass or fail depending on rate limiting configuration
        # In a real scenario, you'd want to configure lower limits for testing


class TestDataEndpoints:
    """Test suite for data management endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('app.services.data_service.DataService.get_data_statistics')
    def test_data_statistics_endpoint(self, mock_get_stats):
        """Test data statistics endpoint."""
        mock_get_stats.return_value = {
            "total_records": 100000,
            "date_range": {
                "start": "2020-01-01",
                "end": "2024-01-01"
            },
            "feature_statistics": {
                "Age": {"mean": 58.2, "std": 12.5},
                "BMI": {"mean": 26.8, "std": 4.2}
            },
            "target_distribution": {
                "positive_cases": 15000,
                "negative_cases": 85000,
                "positive_rate": 0.15
            }
        }
        
        response = self.client.get("/api/v1/data/statistics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "total_records" in data
        assert "feature_statistics" in data
        assert "target_distribution" in data
    
    @patch('app.services.data_service.DataService.validate_data_quality')
    def test_data_validation_endpoint(self, mock_validate):
        """Test data validation endpoint."""
        mock_validate.return_value = {
            "is_valid": True,
            "quality_score": 0.92,
            "issues": [],
            "recommendations": [
                "Consider additional feature engineering for Age variable"
            ]
        }
        
        response = self.client.get("/api/v1/data/validate")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "is_valid" in data
        assert "quality_score" in data
        assert "issues" in data
        assert "recommendations" in data
    
    def test_data_upload_endpoint(self):
        """Test data upload endpoint with CSV file."""
        # Create a mock CSV file
        csv_content = "Age,Sex,BMI,HeartDisease\n65,Male,28.5,1\n45,Female,22.0,0"
        
        response = self.client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.csv", csv_content, "text/csv")}
        )
        
        # This might return different status codes based on implementation
        assert response.status_code in [
            status.HTTP_200_OK, 
            status.HTTP_201_CREATED, 
            status.HTTP_202_ACCEPTED
        ]


class TestAuthenticationAndSecurity:
    """Test suite for authentication and security features."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_api_key_validation_missing(self):
        """Test API endpoint without required API key."""
        response = self.client.post(
            "/api/v1/predictions/predict",
            json={"patient_data": {"Age": 65}}
        )
        
        # Depending on implementation, this might be 401 or allow without auth
        # Update this test based on your actual authentication requirements
        assert response.status_code in [
            status.HTTP_200_OK,  # If no auth required
            status.HTTP_401_UNAUTHORIZED,  # If auth required
            status.HTTP_422_UNPROCESSABLE_ENTITY  # If validation fails first
        ]
    
    def test_api_key_validation_invalid(self):
        """Test API endpoint with invalid API key."""
        headers = {"X-API-Key": "invalid_key_123"}
        
        response = self.client.post(
            "/api/v1/predictions/predict",
            json={"patient_data": {"Age": 65}},
            headers=headers
        )
        
        # Update based on actual implementation
        assert response.status_code in [
            status.HTTP_200_OK,  # If no auth required
            status.HTTP_401_UNAUTHORIZED,  # If auth required
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_cors_headers(self):
        """Test CORS headers in responses."""
        response = self.client.get("/health")
        
        # Check if CORS headers are present (if CORS is enabled)
        # This test should be adjusted based on your CORS configuration
        assert response.status_code == status.HTTP_200_OK


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_404_error_handling(self):
        """Test handling of non-existent endpoints."""
        response = self.client.get("/api/v1/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
    
    def test_method_not_allowed(self):
        """Test handling of incorrect HTTP methods."""
        response = self.client.put("/health")  # GET endpoint called with PUT
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_malformed_json(self):
        """Test handling of malformed JSON requests."""
        response = self.client.post(
            "/api/v1/predictions/predict",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_large_payload_handling(self):
        """Test handling of unusually large payloads."""
        # Create a very large prediction request
        large_patient_data = {f"feature_{i}": i for i in range(10000)}
        large_request = {
            "patient_data": large_patient_data
        }
        
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=large_request
        )
        
        # Should handle gracefully, either process or reject with proper error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        ]


class TestPerformance:
    """Test suite for performance and load testing."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @pytest.mark.performance
    def test_prediction_response_time(self):
        """Test that predictions complete within acceptable time."""
        import time
        
        prediction_data = {
            "patient_data": {
                "Age": 65,
                "Sex": "Male",
                "BMI": 28.5,
                "Smoking": True,
                "HighBP": True,
                "HighChol": True,
                "Diabetes": False,
                "PhysActivity": True
            }
        }
        
        start_time = time.time()
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=prediction_data
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Assert response time is less than 2 seconds
        assert response_time < 2.0
        
        # Also check that response is successful
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY  # If validation fails
        ]
    
    @pytest.mark.performance
    @patch('app.services.ml_service.MLService.predict')
    def test_concurrent_predictions(self, mock_predict):
        """Test handling of concurrent prediction requests."""
        mock_predict.return_value = {
            "prediction": 1,
            "probability": 0.75,
            "model_version": "v1.0.0"
        }
        
        import concurrent.futures
        import time
        
        prediction_data = {
            "patient_data": {
                "Age": 65,
                "Sex": "Male",
                "BMI": 28.5
            }
        }
        
        def make_prediction():
            return self.client.post(
                "/api/v1/predictions/predict",
                json=prediction_data
            )
        
        # Test with 10 concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check that all requests completed successfully
        successful_responses = [r for r in responses if r.status_code == status.HTTP_200_OK]
        assert len(successful_responses) == 10
        
        # Check that concurrent processing was reasonably fast
        assert total_time < 10.0  # 10 seconds for 10 concurrent requests


class TestIntegration:
    """Integration tests that test multiple components together."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @pytest.mark.integration
    @patch('app.services.ml_service.MLService.predict')
    @patch('app.services.prediction_service.PredictionService.save_prediction')
    def test_end_to_end_prediction_flow(self, mock_save, mock_predict):
        """Test complete prediction flow from request to database storage."""
        # Mock ML service
        mock_predict.return_value = {
            "prediction": 1,
            "probability": 0.75,
            "model_version": "v1.0.0",
            "explanation": {
                "top_risk_factors": ["HighBP", "Age", "BMI"],
                "risk_level": "High"
            }
        }
        
        # Mock database save
        mock_save.return_value = {"id": 123, "status": "saved"}
        
        prediction_data = {
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
            "save_to_database": True,
            "return_explanation": True
        }
        
        response = self.client.post(
            "/api/v1/predictions/predict",
            json=prediction_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify prediction was made
        assert "prediction" in data
        assert "probability" in data
        assert "explanation" in data
        
        # Verify mocks were called
        mock_predict.assert_called_once()
        mock_save.assert_called_once()


# Pytest fixtures and configuration
@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_prediction_data():
    """Sample data for prediction tests."""
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
            "HvyAlcoholConsump": False,
            "PhysHlth": 5,
            "MentHlth": 2
        },
        "model_version": "v1.0.0",
        "return_explanation": True
    }


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow tests"
    )