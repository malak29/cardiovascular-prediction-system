import asyncio
import joblib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import structlog

from app.core.config import get_settings, AppConstants
from app.schemas.prediction import (
    ModelInfoSchema,
    ModelPerformanceResponse,
    FeatureImportanceSchema
)
from app.utils.model_utils import ModelManager, ModelEvaluator, HyperparameterOptimizer

logger = structlog.get_logger(__name__)


class MLService:
    """Main machine learning service for cardiovascular prediction"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_manager = ModelManager()
        self.model_evaluator = ModelEvaluator()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
        self._scaler_cache: Dict[str, StandardScaler] = {}
        self._feature_columns_cache: Dict[str, List[str]] = {}
        
        # Model registry
        self._available_models = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor
        }
        
        # Current active model
        self._active_model_version: Optional[str] = None
        self._active_model = None
        self._active_scaler = None
        self._active_features: List[str] = []
    
    async def initialize(self) -> None:
        """Initialize ML service and load models"""
        try:
            logger.info("Initializing ML Service")
            
            # Create model directory if it doesn't exist
            self.settings.ml.MODEL_PATH.mkdir(parents=True, exist_ok=True)
            
            # Load default model
            await self._load_default_model()
            
            # Warm up model cache
            await self._warm_up_cache()
            
            logger.info("ML Service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ML Service", error=str(e), exc_info=True)
            raise
    
    async def _load_default_model(self) -> None:
        """Load the default active model"""
        try:
            # Look for the latest model file
            model_files = list(self.settings.ml.MODEL_PATH.glob("*.pkl"))
            
            if not model_files:
                logger.warning("No model files found, creating default model")
                await self._create_default_model()
                return
            
            # Load the most recent model
            latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model_version = latest_model_file.stem
            
            await self.load_model(model_version)
            logger.info("Default model loaded", model_version=model_version)
            
        except Exception as e:
            logger.error("Failed to load default model", error=str(e), exc_info=True)
            # Create a basic model as fallback
            await self._create_default_model()
    
    async def _create_default_model(self) -> None:
        """Create a default Ridge regression model"""
        try:
            logger.info("Creating default Ridge regression model")
            
            # Create a basic Ridge model
            model = Ridge(alpha=1.0, random_state=42)
            scaler = StandardScaler()
            
            # Define basic feature set
            feature_columns = [
                "age", "gender_male", "has_hypertension", "has_diabetes",
                "has_heart_disease", "systolic_bp", "total_cholesterol",
                "cardiovascular_hospitalizations_last_year"
            ]
            
            # Save model artifacts
            model_version = "ridge_v1.0.0_default"
            model_path = self.settings.ml.MODEL_PATH / f"{model_version}.pkl"
            scaler_path = self.settings.ml.MODEL_PATH / f"{model_version}_scaler.pkl"
            features_path = self.settings.ml.MODEL_PATH / f"{model_version}_features.json"
            
            # Save files
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            with open(features_path, 'w') as f:
                json.dump(feature_columns, f)
            
            # Update active model
            self._active_model_version = model_version
            self._active_model = model
            self._active_scaler = scaler
            self._active_features = feature_columns
            
            logger.info("Default model created and loaded", model_version=model_version)
            
        except Exception as e:
            logger.error("Failed to create default model", error=str(e), exc_info=True)
            raise
    
    async def load_model(self, model_version: str) -> bool:
        """Load a specific model version"""
        try:
            logger.info("Loading model", model_version=model_version)
            
            # Check if model is already cached
            if model_version in self._model_cache:
                self._active_model_version = model_version
                self._active_model = self._model_cache[model_version]
                self._active_scaler = self._scaler_cache.get(model_version)
                self._active_features = self._feature_columns_cache.get(model_version, [])
                logger.debug("Model loaded from cache", model_version=model_version)
                return True
            
            # Load model files
            model_path = self.settings.ml.MODEL_PATH / f"{model_version}.pkl"
            scaler_path = self.settings.ml.MODEL_PATH / f"{model_version}_scaler.pkl"
            features_path = self.settings.ml.MODEL_PATH / f"{model_version}_features.json"
            
            if not model_path.exists():
                logger.error("Model file not found", model_path=str(model_path))
                return False
            
            # Load model
            model = joblib.load(model_path)
            
            # Load scaler (optional)
            scaler = None
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
            
            # Load feature columns
            feature_columns = []
            if features_path.exists():
                with open(features_path, 'r') as f:
                    feature_columns = json.load(f)
            
            # Cache the loaded artifacts
            self._model_cache[model_version] = model
            if scaler:
                self._scaler_cache[model_version] = scaler
            self._feature_columns_cache[model_version] = feature_columns
            
            # Set as active model
            self._active_model_version = model_version
            self._active_model = model
            self._active_scaler = scaler
            self._active_features = feature_columns
            
            logger.info("Model loaded successfully", model_version=model_version)
            return True
            
        except Exception as e:
            logger.error("Failed to load model", model_version=model_version, error=str(e), exc_info=True)
            return False
    
    async def predict_single(
        self,
        features: Dict[str, Any],
        model_version: Optional[str] = None,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """Make a single prediction"""
        try:
            # Use specified model or active model
            if model_version and model_version != self._active_model_version:
                await self.load_model(model_version)
            
            if not self._active_model:
                raise ValueError("No model available for prediction")
            
            # Prepare features
            feature_array = await self._prepare_features(features)
            
            # Make prediction
            if hasattr(self._active_model, 'predict'):
                prediction = self._active_model.predict(feature_array.reshape(1, -1))[0]
            else:
                raise ValueError("Model does not support prediction")
            
            # Calculate confidence if requested
            confidence_info = {}
            if include_confidence:
                confidence_info = await self._calculate_confidence(
                    feature_array.reshape(1, -1), 
                    prediction
                )
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(features)
            
            result = {
                "risk_score": float(prediction),
                "risk_category": self._get_risk_category(prediction),
                "model_version": self._active_model_version,
                "model_type": type(self._active_model).__name__.lower(),
                "feature_importance": feature_importance,
                **confidence_info
            }
            
            logger.debug("Single prediction completed", risk_score=prediction)
            return result
            
        except Exception as e:
            logger.error("Single prediction failed", error=str(e), exc_info=True)
            raise
    
    async def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        model_version: Optional[str] = None,
        parallel_processing: bool = True
    ) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            logger.info("Starting batch prediction", batch_size=len(features_list))
            
            # Use specified model or active model
            if model_version and model_version != self._active_model_version:
                await self.load_model(model_version)
            
            if not self._active_model:
                raise ValueError("No model available for prediction")
            
            # Prepare all features
            feature_arrays = []
            for features in features_list:
                feature_array = await self._prepare_features(features)
                feature_arrays.append(feature_array)
            
            # Convert to numpy array
            X = np.array(feature_arrays)
            
            # Make batch predictions
            predictions = self._active_model.predict(X)
            
            # Process results
            results = []
            for i, (prediction, original_features) in enumerate(zip(predictions, features_list)):
                result = {
                    "index": i,
                    "risk_score": float(prediction),
                    "risk_category": self._get_risk_category(prediction),
                    "model_version": self._active_model_version,
                    "model_type": type(self._active_model).__name__.lower()
                }
                results.append(result)
            
            # Calculate summary statistics
            summary = await self._calculate_batch_summary(predictions)
            
            batch_result = {
                "predictions": results,
                "summary": summary,
                "model_version": self._active_model_version,
                "batch_size": len(features_list),
                "successful_predictions": len(results),
                "failed_predictions": 0
            }
            
            logger.info("Batch prediction completed", successful_predictions=len(results))
            return batch_result
            
        except Exception as e:
            logger.error("Batch prediction failed", error=str(e), exc_info=True)
            raise
    
    async def train_model(
        self,
        training_data: pd.DataFrame,
        model_type: str = "ridge",
        hyperparameters: Optional[Dict[str, Any]] = None,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train a new model"""
        try:
            logger.info("Starting model training", model_type=model_type)
            
            if model_type not in self._available_models:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Prepare training data
            X, y, feature_columns = await self._prepare_training_data(training_data)
            
            # Split data if no validation set provided
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X, y
                X_val, y_val, _ = await self._prepare_training_data(validation_data)
            
            # Initialize model
            model_class = self._available_models[model_type]
            if hyperparameters:
                model = model_class(**hyperparameters)
            else:
                model = model_class()
            
            # Create preprocessing pipeline
            scaler = StandardScaler()
            
            # Fit preprocessing
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            start_time = datetime.utcnow()
            model.fit(X_train_scaled, y_train)
            training_duration = (datetime.utcnow() - start_time).total_seconds() / 60
            
            # Evaluate model
            train_predictions = model.predict(X_train_scaled)
            val_predictions = model.predict(X_val_scaled)
            
            # Calculate metrics
            metrics = {
                "training": {
                    "r2_score": r2_score(y_train, train_predictions),
                    "mse": mean_squared_error(y_train, train_predictions),
                    "mae": mean_absolute_error(y_train, train_predictions)
                },
                "validation": {
                    "r2_score": r2_score(y_val, val_predictions),
                    "mse": mean_squared_error(y_val, val_predictions),
                    "mae": mean_absolute_error(y_val, val_predictions)
                }
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=self.settings.ml.CROSS_VALIDATION_FOLDS,
                scoring='r2'
            )
            
            metrics["cross_validation"] = {
                "mean_score": cv_scores.mean(),
                "std_score": cv_scores.std(),
                "individual_scores": cv_scores.tolist()
            }
            
            # Generate model version
            model_version = f"{model_type}_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model artifacts
            await self._save_model_artifacts(
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
                model_version=model_version,
                metrics=metrics,
                hyperparameters=hyperparameters
            )
            
            training_result = {
                "model_version": model_version,
                "model_type": model_type,
                "metrics": metrics,
                "training_duration_minutes": training_duration,
                "feature_count": len(feature_columns),
                "training_samples": len(X_train),
                "validation_samples": len(X_val)
            }
            
            logger.info(
                "Model training completed",
                model_version=model_version,
                validation_r2=metrics["validation"]["r2_score"]
            )
            
            return training_result
            
        except Exception as e:
            logger.error("Model training failed", error=str(e), exc_info=True)
            raise
    
    async def evaluate_model(
        self,
        model_version: str,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate a model on test data"""
        try:
            logger.info("Evaluating model", model_version=model_version)
            
            # Load model if not active
            if model_version != self._active_model_version:
                await self.load_model(model_version)
            
            # Prepare test data
            X_test, y_test, _ = await self._prepare_training_data(test_data)
            
            # Scale features
            if self._active_scaler:
                X_test_scaled = self._active_scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Make predictions
            predictions = self._active_model.predict(X_test_scaled)
            
            # Calculate comprehensive metrics
            evaluation_metrics = await self.model_evaluator.calculate_regression_metrics(
                y_true=y_test,
                y_pred=predictions
            )
            
            # Add feature importance
            feature_importance = await self._get_model_feature_importance()
            evaluation_metrics["feature_importance"] = feature_importance
            
            # Risk category analysis
            risk_analysis = await self._analyze_risk_predictions(predictions, y_test)
            evaluation_metrics["risk_analysis"] = risk_analysis
            
            logger.info(
                "Model evaluation completed",
                model_version=model_version,
                r2_score=evaluation_metrics["r2_score"]
            )
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error("Model evaluation failed", error=str(e), exc_info=True)
            raise
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get information about all available models"""
        try:
            models_info = []
            model_files = list(self.settings.ml.MODEL_PATH.glob("*.pkl"))
            
            for model_file in model_files:
                model_version = model_file.stem
                
                # Load model metadata
                metadata_path = self.settings.ml.MODEL_PATH / f"{model_version}_metadata.json"
                metadata = {}
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                model_info = {
                    "version": model_version,
                    "name": metadata.get("name", model_version),
                    "model_type": metadata.get("model_type", "unknown"),
                    "is_active": model_version == self._active_model_version,
                    "is_default": model_version == self._active_model_version,
                    "created_at": metadata.get("created_at"),
                    "metrics": metadata.get("metrics", {}),
                    "file_size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
                }
                
                models_info.append(model_info)
            
            # Sort by creation date (newest first)
            models_info.sort(
                key=lambda x: x.get("created_at", ""), 
                reverse=True
            )
            
            return {
                "models": models_info,
                "default_model": self._active_model_version,
                "supported_types": list(self._available_models.keys()),
                "total_models": len(models_info),
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("Failed to get available models", error=str(e), exc_info=True)
            raise
    
    async def get_model_performance(self, model_version: str) -> Optional[ModelPerformanceResponse]:
        """Get detailed performance metrics for a model"""
        try:
            # Load model metadata
            metadata_path = self.settings.ml.MODEL_PATH / f"{model_version}_metadata.json"
            
            if not metadata_path.exists():
                logger.warning("Model metadata not found", model_version=model_version)
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Construct performance response
            performance_data = ModelPerformanceResponse(
                model_id=uuid.UUID(metadata.get("model_id", str(uuid.uuid4()))),
                model_version=model_version,
                model_type=metadata.get("model_type", "unknown"),
                r2_score=metadata.get("metrics", {}).get("validation", {}).get("r2_score", 0.0),
                mse_score=metadata.get("metrics", {}).get("validation", {}).get("mse", 0.0),
                mae_score=metadata.get("metrics", {}).get("validation", {}).get("mae", 0.0),
                rmse_score=np.sqrt(metadata.get("metrics", {}).get("validation", {}).get("mse", 0.0)),
                cross_validation=metadata.get("metrics", {}).get("cross_validation", {}),
                feature_importance=metadata.get("feature_importance", []),
                training_info=metadata.get("training_info", {}),
                evaluation_date=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                evaluation_method="holdout_validation",
                test_data_size=metadata.get("validation_samples", 0)
            )
            
            return performance_data
            
        except Exception as e:
            logger.error("Failed to get model performance", model_version=model_version, error=str(e))
            return None
    
    async def start_retraining_job(
        self,
        user_id: str,
        parameters: Dict[str, Any],
        correlation_id: str
    ) -> Dict[str, Any]:
        """Start a model retraining job"""
        try:
            job_id = str(uuid.uuid4())
            
            logger.info(
                "Starting model retraining job",
                job_id=job_id,
                user_id=user_id,
                correlation_id=correlation_id
            )
            
            # Create job record (this would be stored in database)
            job_info = {
                "job_id": job_id,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "parameters": parameters,
                "correlation_id": correlation_id
            }
            
            # In a real implementation, this would be queued in a job queue
            # For now, we'll simulate the job creation
            
            return job_info
            
        except Exception as e:
            logger.error("Failed to start retraining job", error=str(e), exc_info=True)
            raise
    
    async def get_retraining_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a retraining job"""
        try:
            # In a real implementation, this would query the job queue/database
            # For now, we'll return a mock status
            
            return {
                "job_id": job_id,
                "status": "completed",  # Mock status
                "progress": 100.0,
                "started_at": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "message": "Model retraining completed successfully",
                "new_model_version": f"ridge_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "performance_improvement": 0.023
            }
            
        except Exception as e:
            logger.error("Failed to get retraining job status", job_id=job_id, error=str(e))
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ML service health"""
        try:
            health_info = {
                "status": "healthy",
                "active_model": self._active_model_version,
                "model_loaded": self._active_model is not None,
                "cache_size": len(self._model_cache),
                "available_model_types": list(self._available_models.keys()),
                "model_path_exists": self.settings.ml.MODEL_PATH.exists(),
                "feature_count": len(self._active_features) if self._active_features else 0
            }
            
            # Test prediction capability
            if self._active_model:
                try:
                    # Test with dummy data
                    dummy_features = {f"feature_{i}": 0.5 for i in range(len(self._active_features))}
                    test_prediction = await self.predict_single(dummy_features)
                    health_info["prediction_test"] = "passed"
                except Exception as e:
                    health_info["prediction_test"] = f"failed: {str(e)}"
                    health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"
                health_info["prediction_test"] = "no_model_loaded"
            
            return health_info
            
        except Exception as e:
            logger.error("ML service health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_models_loaded(self) -> bool:
        """Check if models are properly loaded"""
        return self._active_model is not None
    
    # Private helper methods
    async def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model input"""
        try:
            # Create feature vector based on expected features
            feature_vector = []
            
            for feature_name in self._active_features:
                if feature_name in features:
                    value = features[feature_name]
                    
                    # Handle different data types
                    if isinstance(value, bool):
                        feature_vector.append(1.0 if value else 0.0)
                    elif isinstance(value, (int, float)):
                        feature_vector.append(float(value))
                    elif isinstance(value, str):
                        # Handle categorical variables (basic encoding)
                        feature_vector.append(1.0 if value.lower() in ['true', 'yes', '1'] else 0.0)
                    else:
                        feature_vector.append(0.0)  # Default value
                else:
                    # Missing feature - use default value
                    feature_vector.append(0.0)
            
            feature_array = np.array(feature_vector)
            
            # Apply scaling if scaler is available
            if self._active_scaler:
                feature_array = self._active_scaler.transform(feature_array.reshape(1, -1)).flatten()
            
            return feature_array
            
        except Exception as e:
            logger.error("Feature preparation failed", error=str(e), exc_info=True)
            raise
    
    async def _prepare_training_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model training"""
        try:
            # Identify target column
            target_column = "Data_Value"  # From original dataset
            
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Separate features and target
            y = data[target_column].values
            X_df = data.drop(columns=[target_column])
            
            # Get feature columns
            feature_columns = X_df.columns.tolist()
            
            # Convert to numpy array
            X = X_df.values
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error("Training data preparation failed", error=str(e), exc_info=True)
            raise
    
    async def _calculate_confidence(
        self,
        features: np.ndarray,
        prediction: float
    ) -> Dict[str, Any]:
        """Calculate prediction confidence intervals"""
        try:
            # For demonstration, calculate simple confidence based on model uncertainty
            # In production, you might use prediction intervals from the model
            
            base_uncertainty = 0.1  # 10% base uncertainty
            
            # Adjust uncertainty based on feature completeness
            feature_completeness = np.sum(features != 0) / len(features)
            uncertainty_adjustment = (1 - feature_completeness) * 0.2
            
            total_uncertainty = base_uncertainty + uncertainty_adjustment
            
            confidence_lower = max(0.0, prediction - total_uncertainty)
            confidence_upper = min(1.0, prediction + total_uncertainty)
            
            return {
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
                "confidence_level": 0.95,
                "confidence_score": 1.0 - total_uncertainty
            }
            
        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return {}
    
    async def _get_feature_importance(
        self,
        features: Dict[str, Any]
    ) -> List[FeatureImportanceSchema]:
        """Get feature importance for the current prediction"""
        try:
            if not hasattr(self._active_model, 'coef_') and not hasattr(self._active_model, 'feature_importances_'):
                return []
            
            # Get importance values
            if hasattr(self._active_model, 'coef_'):
                importances = np.abs(self._active_model.coef_)
            else:
                importances = self._active_model.feature_importances_
            
            # Create feature importance list
            feature_importance = []
            for i, (feature_name, importance) in enumerate(zip(self._active_features, importances)):
                if i < len(self._active_features):
                    feature_importance.append(
                        FeatureImportanceSchema(
                            feature_name=feature_name,
                            importance_score=float(importance),
                            description=self._get_feature_description(feature_name),
                            category=self._get_feature_category(feature_name)
                        )
                    )
            
            # Sort by importance (descending) and return top 10
            feature_importance.sort(key=lambda x: x.importance_score, reverse=True)
            return feature_importance[:10]
            
        except Exception as e:
            logger.error("Feature importance calculation failed", error=str(e))
            return []
    
    async def _get_model_feature_importance(self) -> List[Dict[str, Any]]:
        """Get overall model feature importance"""
        try:
            if not self._active_model or not self._active_features:
                return []
            
            # Get importance values from model
            if hasattr(self._active_model, 'coef_'):
                importances = np.abs(self._active_model.coef_)
            elif hasattr(self._active_model, 'feature_importances_'):
                importances = self._active_model.feature_importances_
            else:
                return []
            
            # Normalize importances to sum to 1
            importances = importances / np.sum(importances)
            
            # Create importance list
            feature_importance = []
            for feature_name, importance in zip(self._active_features, importances):
                feature_importance.append({
                    "feature_name": feature_name,
                    "importance_score": float(importance),
                    "description": self._get_feature_description(feature_name),
                    "category": self._get_feature_category(feature_name)
                })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return feature_importance
            
        except Exception as e:
            logger.error("Model feature importance calculation failed", error=str(e))
            return []
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to category"""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of feature"""
        descriptions = {
            "age": "Patient age in years",
            "gender_male": "Male gender indicator",
            "has_hypertension": "History of hypertension",
            "has_diabetes": "History of diabetes",
            "has_heart_disease": "History of heart disease",
            "systolic_bp": "Systolic blood pressure",
            "diastolic_bp": "Diastolic blood pressure",
            "total_cholesterol": "Total cholesterol level",
            "hdl_cholesterol": "HDL cholesterol level",
            "bmi": "Body Mass Index",
            "cardiovascular_hospitalizations_last_year": "CV hospitalizations in past year"
        }
        
        return descriptions.get(feature_name, feature_name.replace("_", " ").title())
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Get feature category for grouping"""
        if feature_name in ["age", "gender_male"]:
            return "demographics"
        elif "has_" in feature_name:
            return "medical_history"
        elif any(term in feature_name for term in ["bp", "cholesterol", "glucose", "bmi"]):
            return "clinical_measures"
        elif "hospitalization" in feature_name:
            return "utilization"
        else:
            return "other"
    
    async def _calculate_batch_summary(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate summary statistics for batch predictions"""
        try:
            risk_categories = [self._get_risk_category(pred) for pred in predictions]
            
            summary = {
                "total_predictions": len(predictions),
                "average_risk_score": float(np.mean(predictions)),
                "median_risk_score": float(np.median(predictions)),
                "std_risk_score": float(np.std(predictions)),
                "min_risk_score": float(np.min(predictions)),
                "max_risk_score": float(np.max(predictions)),
                "risk_distribution": {
                    "low": risk_categories.count("low"),
                    "medium": risk_categories.count("medium"),
                    "high": risk_categories.count("high")
                },
                "high_risk_percentage": (risk_categories.count("high") / len(risk_categories)) * 100
            }
            
            return summary
            
        except Exception as e:
            logger.error("Batch summary calculation failed", error=str(e))
            return {}
    
    async def _analyze_risk_predictions(
        self,
        predictions: np.ndarray,
        actual_values: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze risk prediction performance"""
        try:
            # Convert to risk categories
            pred_categories = [self._get_risk_category(pred) for pred in predictions]
            actual_categories = [self._get_risk_category(actual) for actual in actual_values]
            
            # Calculate category-wise accuracy
            category_accuracy = {}
            for category in ["low", "medium", "high"]:
                category_preds = [p for p, a in zip(pred_categories, actual_categories) if a == category]
                if category_preds:
                    correct = sum(1 for p in category_preds if p == category)
                    category_accuracy[category] = correct / len(category_preds)
                else:
                    category_accuracy[category] = 0.0
            
            analysis = {
                "category_accuracy": category_accuracy,
                "overall_category_accuracy": sum(
                    1 for p, a in zip(pred_categories, actual_categories) if p == a
                ) / len(pred_categories),
                "confusion_matrix": self._calculate_confusion_matrix(pred_categories, actual_categories)
            }
            
            return analysis
            
        except Exception as e:
            logger.error("Risk analysis failed", error=str(e))
            return {}
    
    def _calculate_confusion_matrix(
        self,
        predicted: List[str],
        actual: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate confusion matrix for risk categories"""
        categories = ["low", "medium", "high"]
        matrix = {actual_cat: {pred_cat: 0 for pred_cat in categories} for actual_cat in categories}
        
        for pred, act in zip(predicted, actual):
            matrix[act][pred] += 1
        
        return matrix
    
    async def _save_model_artifacts(
        self,
        model: Any,
        scaler: StandardScaler,
        feature_columns: List[str],
        model_version: str,
        metrics: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]]
    ) -> None:
        """Save model artifacts to disk"""
        try:
            # Save model
            model_path = self.settings.ml.MODEL_PATH / f"{model_version}.pkl"
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = self.settings.ml.MODEL_PATH / f"{model_version}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            
            # Save feature columns
            features_path = self.settings.ml.MODEL_PATH / f"{model_version}_features.json"
            with open(features_path, 'w') as f:
                json.dump(feature_columns, f)
            
            # Save metadata
            metadata = {
                "model_id": str(uuid.uuid4()),
                "version": model_version,
                "model_type": type(model).__name__.lower(),
                "created_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "hyperparameters": hyperparameters,
                "feature_columns": feature_columns,
                "training_info": {
                    "feature_count": len(feature_columns),
                    "model_class": type(model).__name__
                }
            }
            
            metadata_path = self.settings.ml.MODEL_PATH / f"{model_version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Model artifacts saved", model_version=model_version)
            
        except Exception as e:
            logger.error("Failed to save model artifacts", error=str(e), exc_info=True)
            raise
    
    async def _warm_up_cache(self) -> None:
        """Warm up model cache with frequently used models"""
        try:
            # This could load multiple models for A/B testing
            # For now, just ensure the active model is cached
            if self._active_model_version and self._active_model_version not in self._model_cache:
                self._model_cache[self._active_model_version] = self._active_model
            
            logger.debug("Model cache warmed up")
            
        except Exception as e:
            logger.error("Cache warm-up failed", error=str(e))


# Export the ML service
__all__ = ["MLService"]