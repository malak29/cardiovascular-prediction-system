import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import logging

from app.utils.data_preprocessing import DataPreprocessor
from app.services.data_service import DataService
from data.processed.feature_engineering import CVDFeatureEngineer
from backend.scripts.data_ingestion import CVDDataIngestion

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.preprocessor = DataPreprocessor()
        
        # Sample cardiovascular data
        self.sample_data = pd.DataFrame({
            'Age': [65, 45, 70, 55, np.nan],
            'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'BMI': [28.5, 22.0, 30.2, 26.1, 24.8],
            'Smoking': [True, False, True, False, True],
            'HighBP': [True, False, True, False, True],
            'HighChol': [True, False, True, True, False],
            'Diabetes': [False, False, True, False, False],
            'PhysActivity': [True, True, False, True, False],
            'HvyAlcoholConsump': [False, False, True, False, False],
            'PhysHlth': [5, 0, 15, 2, np.nan],
            'MentHlth': [2, 1, 10, 0, 3],
            'HeartDisease': [1, 0, 1, 0, 1]
        })
        
        # Sample data with missing values
        self.data_with_missing = self.sample_data.copy()
        self.data_with_missing.loc[0, 'BMI'] = np.nan
        self.data_with_missing.loc[1, 'Sex'] = np.nan
        self.data_with_missing.loc[2, 'Smoking'] = np.nan
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'scaler')
        assert hasattr(preprocessor, 'encoders')
    
    def test_handle_missing_values_numerical(self):
        """Test handling missing values in numerical columns."""
        result = self.preprocessor.handle_missing_values(self.data_with_missing)
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
        
        # Check that Age was filled (originally had NaN at index 4)
        assert not pd.isna(result.loc[4, 'Age'])
        
        # Check that BMI was filled (originally had NaN at index 0)
        assert not pd.isna(result.loc[0, 'BMI'])
    
    def test_handle_missing_values_categorical(self):
        """Test handling missing values in categorical columns."""
        result = self.preprocessor.handle_missing_values(self.data_with_missing)
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
        
        # Check that Sex was filled (originally had NaN at index 1)
        assert not pd.isna(result.loc[1, 'Sex'])
        assert result.loc[1, 'Sex'] in ['Male', 'Female']
    
    def test_handle_missing_values_with_strategy(self):
        """Test different missing value handling strategies."""
        # Test median strategy
        result_median = self.preprocessor.handle_missing_values(
            self.data_with_missing, strategy='median'
        )
        assert result_median.isnull().sum().sum() == 0
        
        # Test mean strategy
        result_mean = self.preprocessor.handle_missing_values(
            self.data_with_missing, strategy='mean'
        )
        assert result_mean.isnull().sum().sum() == 0
        
        # Test mode strategy
        result_mode = self.preprocessor.handle_missing_values(
            self.data_with_missing, strategy='mode'
        )
        assert result_mode.isnull().sum().sum() == 0
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        # Prepare test data with categorical features
        data_with_categorical = self.sample_data.copy()
        data_with_categorical['GenHlth'] = ['Good', 'Excellent', 'Fair', 'Poor', 'Good']
        
        result = self.preprocessor.encode_categorical_features(data_with_categorical)
        
        # Sex should be encoded (binary)
        assert 'Sex' in result.columns
        sex_unique = result['Sex'].unique()
        assert len(sex_unique) <= 2  # Should be binary encoded
        assert all(isinstance(x, (int, float)) for x in sex_unique)
    
    def test_scale_numerical_features(self):
        """Test numerical feature scaling."""
        result = self.preprocessor.scale_numerical_features(self.sample_data)
        
        # Numerical columns should be scaled
        numerical_cols = ['Age', 'BMI', 'PhysHlth', 'MentHlth']
        
        for col in numerical_cols:
            if col in result.columns:
                col_mean = result[col].mean()
                col_std = result[col].std()
                
                # Check if scaled (mean close to 0, std close to 1)
                assert abs(col_mean) < 0.1, f"Column {col} mean not close to 0: {col_mean}"
                assert abs(col_std - 1.0) < 0.1, f"Column {col} std not close to 1: {col_std}"
    
    def test_validate_feature_schema(self):
        """Test feature schema validation."""
        required_features = ['Age', 'Sex', 'BMI', 'Smoking', 'HighBP']
        
        # Test with valid schema
        is_valid, missing = self.preprocessor.validate_feature_schema(
            self.sample_data, required_features
        )
        assert is_valid is True
        assert len(missing) == 0
        
        # Test with missing features
        incomplete_data = self.sample_data.drop(columns=['BMI', 'Smoking'])
        is_valid, missing = self.preprocessor.validate_feature_schema(
            incomplete_data, required_features
        )
        assert is_valid is False
        assert 'BMI' in missing
        assert 'Smoking' in missing
    
    def test_outlier_detection(self):
        """Test outlier detection functionality."""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'Age'] = 200  # Outlier
        data_with_outliers.loc[1, 'BMI'] = 100  # Outlier
        
        outliers = self.preprocessor.detect_outliers(data_with_outliers, columns=['Age', 'BMI'])
        
        assert isinstance(outliers, dict)
        assert 'Age' in outliers
        assert 'BMI' in outliers
        assert len(outliers['Age']) > 0  # Should detect Age outlier
        assert len(outliers['BMI']) > 0  # Should detect BMI outlier
    
    def test_remove_outliers(self):
        """Test outlier removal functionality."""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'Age'] = 200  # Extreme outlier
        
        original_length = len(data_with_outliers)
        cleaned_data = self.preprocessor.remove_outliers(
            data_with_outliers, columns=['Age'], method='iqr'
        )
        
        # Should have fewer rows after outlier removal
        assert len(cleaned_data) <= original_length
        
        # Extreme age value should be removed
        assert cleaned_data['Age'].max() < 200
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        result = self.preprocessor.create_interaction_features(self.sample_data)
        
        # Should have more columns than original
        assert result.shape[1] > self.sample_data.shape[1]
        
        # Check for some expected interaction features
        interaction_cols = [col for col in result.columns if '_X_' in col or '_interaction' in col.lower()]
        assert len(interaction_cols) > 0
    
    def test_feature_binning(self):
        """Test feature binning functionality."""
        result = self.preprocessor.create_binned_features(self.sample_data)
        
        # Should contain binned versions of continuous features
        expected_binned = ['Age_binned', 'BMI_binned']
        
        for col in expected_binned:
            if col in result.columns:
                # Binned features should have fewer unique values
                assert result[col].nunique() < self.sample_data[col.replace('_binned', '')].nunique()
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        X = self.sample_data.drop(columns=['HeartDisease'])
        y = self.sample_data['HeartDisease']
        
        X_processed, y_processed = self.preprocessor.preprocess(X, y)
        
        # Should return processed data
        assert isinstance(X_processed, (pd.DataFrame, np.ndarray))
        assert isinstance(y_processed, (pd.Series, np.ndarray))
        
        # Should have no missing values
        if isinstance(X_processed, pd.DataFrame):
            assert X_processed.isnull().sum().sum() == 0
        else:
            assert not np.isnan(X_processed).any()
        
        # Should preserve number of samples
        assert len(X_processed) == len(X)
        assert len(y_processed) == len(y)


class TestCVDFeatureEngineer:
    """Test suite for CVDFeatureEngineer class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.feature_engineer = CVDFeatureEngineer()
        
        # Sample data for feature engineering
        self.sample_data = pd.DataFrame({
            'Age': [65, 45, 70, 55, 40],
            'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'BMI': [28.5, 22.0, 30.2, 26.1, 24.8],
            'Smoking': [1, 0, 1, 0, 1],
            'HighBP': [1, 0, 1, 0, 1],
            'HighChol': [1, 0, 1, 1, 0],
            'Diabetes': [0, 0, 1, 0, 0],
            'PhysActivity': [1, 1, 0, 1, 0],
            'HvyAlcoholConsump': [0, 0, 1, 0, 0],
            'PhysHlth': [5, 0, 15, 2, 8],
            'MentHlth': [2, 1, 10, 0, 3],
            'HeartDisease': [1, 0, 1, 0, 1]
        })
    
    def test_create_basic_features(self):
        """Test basic feature engineering."""
        result = self.feature_engineer.create_basic_features(self.sample_data)
        
        # Should have more columns than original
        assert result.shape[1] > self.sample_data.shape[1]
        
        # Check for specific engineered features
        expected_features = [
            'BMI_Category', 'BMI_High_Risk', 'Age_Group', 'Age_High_Risk',
            'Lifestyle_Risk_Score', 'Medical_Risk_Score'
        ]
        
        for feature in expected_features:
            if feature in result.columns:
                assert not result[feature].isnull().all(), f"Feature {feature} is all null"
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        basic_features_data = self.feature_engineer.create_basic_features(self.sample_data)
        result = self.feature_engineer.create_interaction_features(basic_features_data)
        
        # Should have interaction features
        interaction_features = [col for col in result.columns if '_X_' in col]
        assert len(interaction_features) > 0
    
    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        result = self.feature_engineer.create_polynomial_features(self.sample_data)
        
        # Should have polynomial features
        poly_features = [col for col in result.columns if '_Poly_' in col or '_Sqrt' in col or '_Log' in col]
        assert len(poly_features) > 0
    
    def test_create_binning_features(self):
        """Test binning feature creation."""
        result = self.feature_engineer.create_binning_features(self.sample_data)
        
        # Should have binned features
        binned_features = [col for col in result.columns if '_Decile' in col or '_Quintile' in col or '_Binned' in col]
        assert len(binned_features) > 0
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Create some engineered features first
        engineered_data = self.feature_engineer.create_basic_features(self.sample_data)
        
        X = engineered_data.drop(columns=['HeartDisease'])
        y = engineered_data['HeartDisease']
        
        X_selected, selected_features = self.feature_engineer.select_features(
            X, y, method='univariate', k=5
        )
        
        assert len(selected_features) == 5
        assert X_selected.shape[1] == 5
        assert all(feature in X.columns for feature in selected_features)
    
    def test_handle_missing_values_strategies(self):
        """Test different missing value handling strategies."""
        # Add missing values to test data
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'BMI'] = np.nan
        data_with_missing.loc[1, 'Age'] = np.nan
        
        # Test smart strategy
        result_smart = self.feature_engineer.handle_missing_values(data_with_missing, strategy='smart')
        assert result_smart.isnull().sum().sum() == 0
        
        # Test KNN strategy
        result_knn = self.feature_engineer.handle_missing_values(data_with_missing, strategy='knn')
        assert result_knn.isnull().sum().sum() == 0
    
    def test_encode_categorical_features(self):
        """Test categorical encoding in feature engineering."""
        # Add categorical features
        data_with_categorical = self.sample_data.copy()
        data_with_categorical['GenHlth'] = ['Good', 'Excellent', 'Fair', 'Poor', 'Good']
        data_with_categorical['Education'] = ['College', 'High School', 'Graduate', 'College', 'High School']
        
        result = self.feature_engineer.encode_categorical_features(data_with_categorical)
        
        # Should handle different types of categorical variables appropriately
        assert 'Sex' in result.columns  # Should be encoded
        # One-hot encoded features should appear for multi-class categoricals
        onehot_cols = [col for col in result.columns if col.startswith('GenHlth_') or col.startswith('Education_')]
        assert len(onehot_cols) > 0


class TestDataService:
    """Test suite for DataService class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_db = Mock()
        self.data_service = DataService(self.mock_db)
        
        # Sample statistics data
        self.sample_stats = {
            "total_records": 100000,
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
    
    @patch('app.services.data_service.pd.read_sql')
    def test_get_data_statistics(self, mock_read_sql):
        """Test data statistics retrieval."""
        # Mock database query result
        mock_data = pd.DataFrame({
            'Age': np.random.normal(58, 12, 1000),
            'BMI': np.random.normal(27, 4, 1000),
            'HeartDisease': np.random.binomial(1, 0.15, 1000)
        })
        mock_read_sql.return_value = mock_data
        
        result = self.data_service.get_data_statistics()
        
        assert isinstance(result, dict)
        assert 'total_records' in result
        assert 'feature_statistics' in result
        assert 'target_distribution' in result
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Mock data quality validation
        with patch.object(self.data_service, '_check_missing_values', return_value={'Age': 0.05}):
            with patch.object(self.data_service, '_check_outliers', return_value={'BMI': 10}):
                with patch.object(self.data_service, '_check_data_consistency', return_value=[]):
                    
                    result = self.data_service.validate_data_quality()
                    
                    assert isinstance(result, dict)
                    assert 'is_valid' in result
                    assert 'quality_score' in result
                    assert 'issues' in result
                    assert 'recommendations' in result
    
    def test_data_upload_validation(self):
        """Test data upload validation."""
        # Create sample upload data
        upload_data = pd.DataFrame({
            'Age': [65, 45, 70],
            'Sex': ['Male', 'Female', 'Male'],
            'BMI': [28.5, 22.0, 30.2],
            'HeartDisease': [1, 0, 1]
        })
        
        is_valid, issues = self.data_service.validate_upload_data(upload_data)
        
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        # Mock feature importance data
        mock_importance = {
            'Age': 0.25,
            'BMI': 0.20,
            'HighBP': 0.18,
            'Smoking': 0.15,
            'Diabetes': 0.12,
            'HighChol': 0.10
        }
        
        with patch.object(self.data_service, '_calculate_feature_importance', return_value=mock_importance):
            result = self.data_service.get_feature_importance()
            
            assert isinstance(result, dict)
            assert len(result) > 0
            assert all(isinstance(v, (int, float)) for v in result.values())


class TestCVDDataIngestion:
    """Test suite for CVDDataIngestion class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'raw_data_path': f'{self.temp_dir}/raw/',
            'processed_data_path': f'{self.temp_dir}/processed/',
            'backup_path': f'{self.temp_dir}/backup/',
            'database_url': 'sqlite:///:memory:'
        }
        self.ingestion_system = CVDDataIngestion(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('requests.get')
    def test_fetch_cdc_data_success(self, mock_get):
        """Test successful CDC data fetching."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'LocationDesc,DataValue,Question,Response',
            'Alabama,15.2,Heart Disease Mortality,Rate',
            'Alaska,12.8,Heart Disease Mortality,Rate'
        ]
        mock_get.return_value = mock_response
        
        result = self.ingestion_system.fetch_cdc_data('heart_disease_mortality', force_refresh=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 data rows
        assert 'LocationDesc' in result.columns
        assert 'DataValue' in result.columns
    
    @patch('requests.get')
    def test_fetch_cdc_data_failure(self, mock_get):
        """Test CDC data fetching failure handling."""
        # Mock failed HTTP response
        mock_get.side_effect = Exception("Connection failed")
        
        result = self.ingestion_system.fetch_cdc_data('heart_disease_mortality', force_refresh=True)
        
        assert result is None
    
    def test_validate_data(self):
        """Test data validation functionality."""
        # Create sample data
        valid_data = pd.DataFrame({
            'LocationDesc': ['Alabama', 'Alaska'],
            'DataValue': [15.2, 12.8],
            'Question': ['Heart Disease', 'Heart Disease'],
            'Response': ['Rate', 'Rate'],
            'YearStart': [2020, 2020]
        })
        
        is_valid, issues = self.ingestion_system.validate_data(valid_data, 'test_source')
        
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_clean_and_standardize_data(self):
        """Test data cleaning and standardization."""
        # Create sample messy data
        messy_data = pd.DataFrame({
            'Location Desc': ['  Alabama  ', 'ALASKA', ''],
            'Data Value': [15.2, 12.8, np.nan],
            'Question   ': ['Heart Disease', 'heart disease', 'Heart Disease'],
            'YearStart': [2020, 2020, 2019]
        })
        
        cleaned_data = self.ingestion_system.clean_and_standardize_data(messy_data, 'test_source')
        
        # Should have standardized column names
        assert all('_' in col.lower() or col.islower() for col in cleaned_data.columns)
        
        # Should have data_source and timestamp columns
        assert 'data_source' in cleaned_data.columns
        assert 'ingestion_timestamp' in cleaned_data.columns
        
        # Should handle missing values appropriately
        assert cleaned_data.isnull().sum().sum() <= messy_data.isnull().sum().sum()
    
    def test_merge_data_sources(self):
        """Test merging multiple data sources."""
        # Create sample data from different sources
        source1_data = pd.DataFrame({
            'LocationDesc': ['Alabama', 'Alaska'],
            'DataValue': [15.2, 12.8],
            'YearStart': [2020, 2020],
            'Topic': ['Heart Disease', 'Heart Disease']
        })
        
        source2_data = pd.DataFrame({
            'LocationDesc': ['Alabama', 'Alaska'],
            'DataValue': [25.3, 22.1],
            'YearStart': [2020, 2020],
            'Topic': ['Diabetes', 'Diabetes']
        })
        
        dataframes = {
            'heart_disease': source1_data,
            'diabetes': source2_data
        }
        
        merged_data = self.ingestion_system.merge_data_sources(dataframes)
        
        assert isinstance(merged_data, pd.DataFrame)
        assert len(merged_data) >= len(source1_data)  # Should have at least original rows
        assert 'LocationDesc' in merged_data.columns
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        # Create sample processed data
        processed_data = pd.DataFrame({
            'LocationDesc': ['Alabama', 'Alaska'],
            'DataValue': [15.2, 12.8],
            'data_source': ['test', 'test'],
            'ingestion_timestamp': [pd.Timestamp.now(), pd.Timestamp.now()]
        })
        
        self.ingestion_system.save_processed_data(processed_data, 'test_dataset')
        
        # Check that files were created
        processed_path = Path(self.config['processed_data_path'])
        csv_files = list(processed_path.glob('test_dataset_*.csv'))
        parquet_files = list(processed_path.glob('test_dataset_*.parquet'))
        
        assert len(csv_files) > 0
        assert len(parquet_files) > 0
        
        # Check latest files exist
        assert (processed_path / 'test_dataset_latest.csv').exists()
        assert (processed_path / 'test_dataset_latest.parquet').exists()


class TestDataPipelineIntegration:
    """Integration tests for complete data processing pipelines."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline."""
        # Create sample raw data
        raw_data = pd.DataFrame({
            'Age': np.random.randint(18, 100, 100),
            'Sex': np.random.choice(['Male', 'Female'], 100),
            'BMI': np.random.normal(26, 4, 100),
            'Smoking': np.random.choice([0, 1], 100),
            'HighBP': np.random.choice([0, 1], 100),
            'HighChol': np.random.choice([0, 1], 100),
            'Diabetes': np.random.choice([0, 1], 100),
            'PhysActivity': np.random.choice([0, 1], 100),
            'HeartDisease': np.random.choice([0, 1], 100)
        })
        
        # Save raw data
        raw_data_path = Path(self.temp_dir) / 'raw_data.csv'
        raw_data.to_csv(raw_data_path, index=False)
        
        # Run feature engineering pipeline
        feature_engineer = CVDFeatureEngineer()
        results = feature_engineer.run_feature_engineering_pipeline(
            data_path=raw_data_path,
            target_column='HeartDisease',
            output_path=Path(self.temp_dir) / 'engineered_data.csv'
        )
        
        # Verify pipeline results
        assert results['success'] is True
        assert results['final_shape'][0] == raw_data.shape[0]  # Same number of rows
        assert results['final_shape'][1] > raw_data.shape[1]   # More features
        
        # Verify output files exist
        assert Path(self.temp_dir, 'engineered_data.csv').exists()
        assert Path(self.temp_dir, 'feature_engineering_artifacts.pkl').exists()
    
    def test_data_quality_monitoring(self):
        """Test data quality monitoring throughout pipeline."""
        # Create data with various quality issues
        problematic_data = pd.DataFrame({
            'Age': [25, 150, -5, 65, np.nan],  # Outliers and missing
            'BMI': [22, 80, 15, np.nan, 28],   # Outliers and missing
            'Sex': ['Male', 'Female', 'Other', 'Male', ''],  # Unexpected category
            'HighBP': [1, 0, 1, np.nan, 'Yes'],  # Mixed types
            'HeartDisease': [0, 1, 0, 1, 1]
        })
        
        # Test quality assessment
        feature_engineer = CVDFeatureEngineer()
        quality_report = feature_engineer.analyze_data_quality(problematic_data)
        
        # Should identify quality issues
        assert quality_report['missing_data']['counts']['Age'] > 0
        assert quality_report['outliers']['Age'] > 0
        assert quality_report['outliers']['BMI'] > 0
    
    def test_preprocessing_pipeline_robustness(self):
        """Test preprocessing pipeline with edge cases."""
        # Create challenging dataset
        edge_case_data = pd.DataFrame({
            'Age': [18, 100, 25, 65, 45],  # Min/max values
            'BMI': [15.0, 50.0, 22.5, 28.0, 35.2],  # Min/max BMI
            'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Smoking': [True, False, True, False, True],
            'HighBP': [True, True, False, False, True],
            'HeartDisease': [0, 1, 0, 1, 1]
        })
        
        preprocessor = DataPreprocessor()
        
        # Should handle edge cases without errors
        try:
            X = edge_case_data.drop(columns=['HeartDisease'])
            y = edge_case_data['HeartDisease']
            X_processed, y_processed = preprocessor.preprocess(X, y)
            
            assert X_processed is not None
            assert y_processed is not None
            assert len(X_processed) == len(edge_case_data)
            
        except Exception as e:
            pytest.fail(f"Preprocessing pipeline failed on edge cases: {str(e)}")


# Utility functions for testing
def create_sample_cvd_data(n_samples=1000, include_missing=False, include_outliers=False):
    """Create sample cardiovascular disease data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'Age': np.random.randint(18, 100, n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.random.normal(26.8, 4.2, n_samples),
        'Smoking': np.random.choice([0, 1], n_samples),
        'HighBP': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'HighChol': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'Diabetes': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'PhysActivity': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'HvyAlcoholConsump': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    }
    
    # Create target with some logic
    heart_disease_prob = (
        (data['Age'] > 60).astype(int) * 0.3 +
        np.array(data['Smoking']) * 0.2 +
        np.array(data['HighBP']) * 0.15 +
        np.array(data['Diabetes']) * 0.1
    )
    data['HeartDisease'] = np.random.binomial(1, np.clip(heart_disease_prob, 0, 1), n_samples)
    
    df = pd.DataFrame(data)
    
    if include_missing:
        # Add missing values
        missing_mask = np.random.random(n_samples) < 0.05
        df.loc[missing_mask, 'BMI'] = np.nan
        
        missing_mask = np.random.random(n_samples) < 0.02
        df.loc[missing_mask, 'Age'] = np.nan
    
    if include_outliers:
        # Add outliers
        outlier_mask = np.random.random(n_samples) < 0.01
        df.loc[outlier_mask, 'Age'] = np.random.choice([150, 200, 5, 10])
        
        outlier_mask = np.random.random(n_samples) < 0.01
        df.loc[outlier_mask, 'BMI'] = np.random.choice([80, 100, 5, 8])
    
    return df


# Performance benchmarks
@pytest.mark.benchmark
class TestDataProcessingPerformance:
    """Performance benchmarks for data processing."""
    
    def test_preprocessing_speed_benchmark(self, benchmark):
        """Benchmark preprocessing speed."""
        # Create large dataset
        large_dataset = create_sample_cvd_data(n_samples=10000)
        preprocessor = DataPreprocessor()
        
        X = large_dataset.drop(columns=['HeartDisease'])
        y = large_dataset['HeartDisease']
        
        # Benchmark preprocessing
        result = benchmark(preprocessor.preprocess, X, y)
        
        X_processed, y_processed = result
        assert len(X_processed) == len(large_dataset)
        assert len(y_processed) == len(large_dataset)
    
    def test_feature_engineering_speed_benchmark(self, benchmark):
        """Benchmark feature engineering speed."""
        # Create dataset for feature engineering
        dataset = create_sample_cvd_data(n_samples=5000)
        feature_engineer = CVDFeatureEngineer()
        
        # Benchmark feature engineering
        result = benchmark(feature_engineer.create_basic_features, dataset)
        
        assert result.shape[1] > dataset.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])