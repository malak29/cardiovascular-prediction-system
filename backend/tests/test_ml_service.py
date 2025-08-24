import pytest
import numpy as np
import pandas as pd
import joblib
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import logging

from app.services.ml_service import MLService
from app.utils.data_preprocessing import DataPreprocessor
from app.core.config import get_settings

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestMLService:
    """Test suite for MLService class."""
    
    def setup_method(self):
        """Set up test environment and mock data."""
        self.settings = get_settings()
        
        # Create mock model and artifacts
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        self.mock_artifacts = {
            'scaler': Mock(),
            'label_encoders': {},
            'feature_names': ['Age', 'BMI', 'HighBP', 'Smoking', 'Sex'],
            'best_model_name': 'random_forest'
        }
        
        # Sample patient data
        self.sample_patient_data = {
            'Age': 65,
            'BMI': 28.5,
            'HighBP': True,
            'Smoking': True,
            'Sex': 'Male',
            'HighChol': True,
            'Diabetes': False,
            'PhysActivity': True,
            'HvyAlcoholConsump': False
        }
    
    def test_ml_service_initialization(self):
        """Test MLService initialization."""
        with patch('joblib.load') as mock_load:
            with patch('builtins.open', create=True) as mock_open:
                mock_load.return_value = self.mock_model
                mock_open.return_value.__enter__.return_value = Mock()
                
                with patch('pickle.load', return_value=self.mock_artifacts):
                    ml_service = MLService()
                    
                    assert ml_service.model is not None
                    assert ml_service.artifacts is not None
                    assert ml_service.is_loaded is True
    
    def test_ml_service_initialization_failure(self):
        """Test MLService initialization when model loading fails."""
        with patch('joblib.load', side_effect=FileNotFoundError("Model file not found")):
            ml_service = MLService()
            
            assert ml_service.model is None
            assert ml_service.artifacts is None
            assert ml_service.is_loaded is False
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_preprocess_patient_data(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test patient data preprocessing."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock scaler transformation
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        processed_data = ml_service._preprocess_patient_data(self.sample_patient_data)
        
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape == (1, 5)  # 1 sample, 5 features
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_predict_success(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test successful prediction."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock scaler transformation
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        result = ml_service.predict(self.sample_patient_data)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert 'model_version' in result
        assert result['prediction'] == 1
        assert result['probability'] == 0.7
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_predict_with_explanation(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test prediction with explanation."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock feature importance for explanation
        self.mock_model.feature_importances_ = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        result = ml_service.predict(self.sample_patient_data, return_explanation=True)
        
        assert 'explanation' in result
        assert 'top_risk_factors' in result['explanation']
        assert 'risk_level' in result['explanation']
        assert isinstance(result['explanation']['top_risk_factors'], list)
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        ml_service = MLService()
        # Don't load model
        ml_service.is_loaded = False
        ml_service.model = None
        
        with pytest.raises(Exception) as exc_info:
            ml_service.predict(self.sample_patient_data)
        
        assert "Model not loaded" in str(exc_info.value)
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_predict_missing_features(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test prediction with missing required features."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        incomplete_data = {
            'Age': 65,
            'BMI': 28.5
            # Missing other required features
        }
        
        ml_service = MLService()
        
        with pytest.raises(Exception) as exc_info:
            ml_service.predict(incomplete_data)
        
        assert "Missing required features" in str(exc_info.value) or "KeyError" in str(type(exc_info.value))
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_batch_predict(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test batch prediction functionality."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock multiple predictions
        self.mock_model.predict.return_value = np.array([1, 0, 1])
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]] * 3)
        
        ml_service = MLService()
        
        batch_data = [
            self.sample_patient_data,
            {**self.sample_patient_data, 'Age': 45, 'HighBP': False},
            {**self.sample_patient_data, 'Age': 70, 'Diabetes': True}
        ]
        
        results = ml_service.batch_predict(batch_data)
        
        assert len(results) == 3
        for result in results:
            assert 'prediction' in result
            assert 'probability' in result
            assert 'model_version' in result
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_model_info(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test getting model information."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        ml_service = MLService()
        model_info = ml_service.get_model_info()
        
        assert 'model_name' in model_info
        assert 'version' in model_info
        assert 'feature_count' in model_info
        assert 'model_type' in model_info
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_feature_importance(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test feature importance extraction."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock feature importance
        self.mock_model.feature_importances_ = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
        
        ml_service = MLService()
        importance = ml_service.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(self.mock_artifacts['feature_names'])
        assert all(isinstance(v, float) for v in importance.values())


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.sample_data = pd.DataFrame({
            'Age': [65, 45, 70, 55],
            'BMI': [28.5, 22.0, 30.2, 26.1],
            'HighBP': [True, False, True, False],
            'Smoking': [True, False, True, False],
            'Sex': ['Male', 'Female', 'Male', 'Female']
        })
        
        self.mock_encoders = {
            'Sex': Mock()
        }
        self.mock_encoders['Sex'].transform.return_value = np.array([0, 1, 0, 1])
        
        self.preprocessor = DataPreprocessor()
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'BMI'] = np.nan
        data_with_missing.loc[1, 'Age'] = np.nan
        
        cleaned_data = self.preprocessor.handle_missing_values(data_with_missing)
        
        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
        assert len(cleaned_data) == len(data_with_missing)  # Same number of rows
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        encoded_data = self.preprocessor.encode_categorical_features(
            self.sample_data, 
            encoders=self.mock_encoders
        )
        
        # Check that categorical columns were processed
        assert 'Sex' in encoded_data.columns
        # The exact assertion depends on your encoding implementation
    
    def test_scale_numerical_features(self):
        """Test numerical feature scaling."""
        scaled_data = self.preprocessor.scale_numerical_features(self.sample_data)
        
        # Check that numerical columns were scaled
        numerical_cols = ['Age', 'BMI']
        for col in numerical_cols:
            if col in scaled_data.columns:
                # Check that values are scaled (mean close to 0, std close to 1)
                assert abs(scaled_data[col].mean()) < 0.1
                assert abs(scaled_data[col].std() - 1.0) < 0.1
    
    def test_validate_feature_schema(self):
        """Test feature schema validation."""
        required_features = ['Age', 'BMI', 'HighBP', 'Smoking', 'Sex']
        
        # Valid data
        is_valid, missing = self.preprocessor.validate_feature_schema(
            self.sample_data, required_features
        )
        assert is_valid is True
        assert len(missing) == 0
        
        # Missing features
        incomplete_data = self.sample_data.drop(columns=['BMI', 'Smoking'])
        is_valid, missing = self.preprocessor.validate_feature_schema(
            incomplete_data, required_features
        )
        assert is_valid is False
        assert 'BMI' in missing
        assert 'Smoking' in missing
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        features_with_interactions = self.preprocessor.create_interaction_features(
            self.sample_data
        )
        
        # Should have more columns than original
        assert features_with_interactions.shape[1] >= self.sample_data.shape[1]
        # Check for some expected interaction features
        interaction_cols = [col for col in features_with_interactions.columns if '_X_' in col]
        assert len(interaction_cols) > 0


class TestModelPredictionEdgeCases:
    """Test suite for edge cases in model predictions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_model = Mock()
        self.mock_artifacts = {
            'scaler': Mock(),
            'label_encoders': {},
            'feature_names': ['Age', 'BMI', 'HighBP', 'Smoking', 'Sex'],
            'best_model_name': 'random_forest'
        }
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_extreme_age_values(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test prediction with extreme age values."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        
        # Test very young age
        young_patient = {
            'Age': 18,  # Minimum age
            'BMI': 25.0,
            'HighBP': False,
            'Smoking': False,
            'Sex': 'Female'
        }
        
        result = ml_service.predict(young_patient)
        assert 'prediction' in result
        
        # Test very old age
        old_patient = {
            'Age': 100,  # Very high age
            'BMI': 25.0,
            'HighBP': True,
            'Smoking': False,
            'Sex': 'Male'
        }
        
        result = ml_service.predict(old_patient)
        assert 'prediction' in result
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_extreme_bmi_values(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test prediction with extreme BMI values."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        
        # Test very low BMI
        underweight_patient = {
            'Age': 30,
            'BMI': 15.0,  # Severely underweight
            'HighBP': False,
            'Smoking': False,
            'Sex': 'Female'
        }
        
        result = ml_service.predict(underweight_patient)
        assert 'prediction' in result
        
        # Test very high BMI
        obese_patient = {
            'Age': 50,
            'BMI': 50.0,  # Severely obese
            'HighBP': True,
            'Smoking': False,
            'Sex': 'Male'
        }
        
        result = ml_service.predict(obese_patient)
        assert 'prediction' in result
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_model_prediction_error(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test handling of model prediction errors."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock model to raise an exception
        self.mock_model.predict.side_effect = Exception("Model prediction failed")
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        
        patient_data = {
            'Age': 65,
            'BMI': 28.5,
            'HighBP': True,
            'Smoking': True,
            'Sex': 'Male'
        }
        
        with pytest.raises(Exception) as exc_info:
            ml_service.predict(patient_data)
        
        assert "Model prediction failed" in str(exc_info.value)
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_preprocessing_error(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test handling of preprocessing errors."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock scaler to raise an exception
        self.mock_artifacts['scaler'].transform.side_effect = Exception("Scaling failed")
        
        ml_service = MLService()
        
        patient_data = {
            'Age': 65,
            'BMI': 28.5,
            'HighBP': True,
            'Smoking': True,
            'Sex': 'Male'
        }
        
        with pytest.raises(Exception) as exc_info:
            ml_service.predict(patient_data)
        
        assert "Scaling failed" in str(exc_info.value)


class TestModelPerformanceMonitoring:
    """Test suite for model performance monitoring."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_model = Mock()
        self.mock_artifacts = {
            'scaler': Mock(),
            'label_encoders': {},
            'feature_names': ['Age', 'BMI', 'HighBP', 'Smoking', 'Sex'],
            'best_model_name': 'random_forest'
        }
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_prediction_confidence_scoring(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test prediction confidence scoring."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        # Mock different confidence levels
        test_cases = [
            np.array([[0.1, 0.9]]),   # High confidence
            np.array([[0.45, 0.55]]), # Low confidence
            np.array([[0.2, 0.8]])    # Medium confidence
        ]
        
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        
        for i, proba in enumerate(test_cases):
            self.mock_model.predict.return_value = np.array([np.argmax(proba)])
            self.mock_model.predict_proba.return_value = proba
            
            patient_data = {
                'Age': 65,
                'BMI': 28.5,
                'HighBP': True,
                'Smoking': True,
                'Sex': 'Male'
            }
            
            result = ml_service.predict(patient_data, include_confidence=True)
            
            assert 'confidence' in result
            assert 0 <= result['confidence'] <= 1
    
    @patch('joblib.load')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_prediction_latency_tracking(self, mock_pickle_load, mock_open, mock_joblib_load):
        """Test prediction latency tracking."""
        mock_joblib_load.return_value = self.mock_model
        mock_pickle_load.return_value = self.mock_artifacts
        
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
        
        ml_service = MLService()
        
        patient_data = {
            'Age': 65,
            'BMI': 28.5,
            'HighBP': True,
            'Smoking': True,
            'Sex': 'Male'
        }
        
        result = ml_service.predict(patient_data, track_latency=True)
        
        assert 'latency_ms' in result
        assert isinstance(result['latency_ms'], (int, float))
        assert result['latency_ms'] > 0


# Fixtures for testing
@pytest.fixture
def sample_patient_data():
    """Fixture providing sample patient data."""
    return {
        'Age': 65,
        'BMI': 28.5,
        'HighBP': True,
        'Smoking': True,
        'Sex': 'Male',
        'HighChol': True,
        'Diabetes': False,
        'PhysActivity': True,
        'HvyAlcoholConsump': False
    }


@pytest.fixture
def mock_trained_model():
    """Fixture providing a mock trained model."""
    model = Mock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.feature_importances_ = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
    return model


@pytest.fixture
def mock_preprocessing_artifacts():
    """Fixture providing mock preprocessing artifacts."""
    return {
        'scaler': Mock(),
        'label_encoders': {},
        'feature_names': ['Age', 'BMI', 'HighBP', 'Smoking', 'Sex'],
        'best_model_name': 'random_forest'
    }


# Performance benchmarks
@pytest.mark.benchmark
class TestMLServicePerformance:
    """Performance benchmarks for ML service."""
    
    def test_prediction_speed_benchmark(self, benchmark, sample_patient_data):
        """Benchmark prediction speed."""
        with patch('joblib.load') as mock_load:
            with patch('builtins.open', create=True):
                with patch('pickle.load') as mock_pickle:
                    mock_model = Mock()
                    mock_model.predict.return_value = np.array([1])
                    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
                    
                    mock_artifacts = {
                        'scaler': Mock(),
                        'label_encoders': {},
                        'feature_names': ['Age', 'BMI', 'HighBP', 'Smoking', 'Sex'],
                        'best_model_name': 'random_forest'
                    }
                    mock_artifacts['scaler'].transform.return_value = np.array([[0.5, 0.3, 1.0, 1.0, 0.0]])
                    
                    mock_load.return_value = mock_model
                    mock_pickle.return_value = mock_artifacts
                    
                    ml_service = MLService()
                    
                    # Benchmark the prediction
                    result = benchmark(ml_service.predict, sample_patient_data)
                    
                    assert 'prediction' in result
                    assert 'probability' in result


if __name__ == "__main__":
    pytest.main([__file__])