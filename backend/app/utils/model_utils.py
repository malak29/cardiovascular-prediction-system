import json
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    validation_curve,
    learning_curve,
    KFold,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class ModelManager:
    """Comprehensive model management and versioning"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_registry = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "elastic_net": ElasticNet,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor
        }
        
        # Model metadata cache
        self._model_metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    async def save_model(
        self,
        model: BaseEstimator,
        model_version: str,
        model_name: str,
        model_type: str,
        training_metadata: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        feature_columns: List[str],
        scaler: Optional[StandardScaler] = None
    ) -> Dict[str, Any]:
        """Save model with comprehensive metadata"""
        try:
            logger.info("Saving model", version=model_version, type=model_type)
            
            # Create model directory
            model_dir = self.settings.ml.MODEL_PATH / model_version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Save scaler if provided
            scaler_path = None
            if scaler:
                scaler_path = model_dir / "scaler.pkl"
                joblib.dump(scaler, scaler_path)
            
            # Save feature columns
            features_path = model_dir / "features.json"
            with open(features_path, 'w') as f:
                json.dump(feature_columns, f, indent=2)
            
            # Create comprehensive metadata
            metadata = {
                "model_info": {
                    "id": str(uuid.uuid4()),
                    "version": model_version,
                    "name": model_name,
                    "type": model_type,
                    "algorithm": type(model).__name__,
                    "created_at": datetime.utcnow().isoformat(),
                    "created_by": "system"  # Could be enhanced with user tracking
                },
                "training_info": training_metadata,
                "performance_metrics": performance_metrics,
                "model_artifacts": {
                    "model_path": str(model_path),
                    "scaler_path": str(scaler_path) if scaler_path else None,
                    "features_path": str(features_path),
                    "feature_count": len(feature_columns)
                },
                "feature_columns": feature_columns,
                "hyperparameters": self._extract_hyperparameters(model),
                "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                "status": "active",
                "validation": {
                    "cross_validation_performed": "cross_validation" in performance_metrics,
                    "holdout_validation_performed": "validation" in performance_metrics,
                    "performance_threshold_met": performance_metrics.get("validation", {}).get("r2_score", 0) >= self.settings.ml.MODEL_PERFORMANCE_THRESHOLD
                }
            }
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update cache
            self._model_metadata_cache[model_version] = metadata
            
            logger.info(
                "Model saved successfully",
                version=model_version,
                performance_r2=performance_metrics.get("validation", {}).get("r2_score")
            )
            
            return {
                "model_version": model_version,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "model_size_mb": metadata["model_size_mb"]
            }
            
        except Exception as e:
            logger.error("Model save failed", version=model_version, error=str(e), exc_info=True)
            raise
    
    async def load_model_metadata(self, model_version: str) -> Optional[Dict[str, Any]]:
        """Load model metadata"""
        try:
            # Check cache first
            if model_version in self._model_metadata_cache:
                return self._model_metadata_cache[model_version]
            
            # Load from disk
            metadata_path = self.settings.ml.MODEL_PATH / model_version / "metadata.json"
            
            if not metadata_path.exists():
                logger.warning("Model metadata not found", version=model_version)
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Cache metadata
            self._model_metadata_cache[model_version] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error("Failed to load model metadata", version=model_version, error=str(e))
            return None
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata"""
        try:
            models = []
            
            # Scan model directory
            if not self.settings.ml.MODEL_PATH.exists():
                return models
            
            for model_dir in self.settings.ml.MODEL_PATH.iterdir():
                if model_dir.is_dir():
                    metadata = await self.load_model_metadata(model_dir.name)
                    if metadata:
                        models.append(metadata)
            
            # Sort by creation date (newest first)
            models.sort(
                key=lambda x: x.get("model_info", {}).get("created_at", ""),
                reverse=True
            )
            
            return models
            
        except Exception as e:
            logger.error("Failed to list available models", error=str(e))
            return []
    
    async def compare_models(
        self,
        model_versions: List[str],
        comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple models across various metrics"""
        try:
            if comparison_metrics is None:
                comparison_metrics = ["r2_score", "mse", "mae", "training_time"]
            
            logger.info("Comparing models", versions=model_versions)
            
            comparison_data = {}
            
            for version in model_versions:
                metadata = await self.load_model_metadata(version)
                if metadata:
                    model_data = {
                        "version": version,
                        "type": metadata.get("model_info", {}).get("type"),
                        "created_at": metadata.get("model_info", {}).get("created_at"),
                        "performance": metadata.get("performance_metrics", {}),
                        "training_info": metadata.get("training_info", {})
                    }
                    
                    comparison_data[version] = model_data
            
            # Create comparison summary
            summary = {
                "models_compared": len(comparison_data),
                "comparison_date": datetime.utcnow().isoformat(),
                "metrics_compared": comparison_metrics,
                "model_details": comparison_data
            }
            
            # Find best model for each metric
            best_models = {}
            for metric in comparison_metrics:
                best_version = None
                best_value = None
                
                for version, data in comparison_data.items():
                    metric_value = self._extract_metric_value(data, metric)
                    
                    if metric_value is not None:
                        if best_value is None or self._is_better_metric(metric, metric_value, best_value):
                            best_value = metric_value
                            best_version = version
                
                if best_version:
                    best_models[metric] = {"version": best_version, "value": best_value}
            
            summary["best_models_by_metric"] = best_models
            
            return summary
            
        except Exception as e:
            logger.error("Model comparison failed", error=str(e), exc_info=True)
            raise
    
    def _extract_hyperparameters(self, model: BaseEstimator) -> Dict[str, Any]:
        """Extract hyperparameters from a trained model"""
        try:
            # Get all parameters from the model
            params = model.get_params()
            
            # Filter out non-serializable parameters
            serializable_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_params[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value):
                    serializable_params[key] = list(value)
                else:
                    serializable_params[key] = str(value)
            
            return serializable_params
            
        except Exception as e:
            logger.error("Hyperparameter extraction failed", error=str(e))
            return {}
    
    def _extract_metric_value(self, model_data: Dict[str, Any], metric: str) -> Optional[float]:
        """Extract metric value from model data"""
        try:
            performance = model_data.get("performance", {})
            
            # Try validation metrics first, then training
            for subset in ["validation", "training", "cross_validation"]:
                if subset in performance and metric in performance[subset]:
                    return performance[subset][metric]
            
            # Try direct metric access
            if metric in performance:
                return performance[metric]
            
            return None
            
        except Exception:
            return None
    
    def _is_better_metric(self, metric_name: str, value1: float, value2: float) -> bool:
        """Determine if value1 is better than value2 for the given metric"""
        # Higher is better for these metrics
        higher_is_better = ["r2_score", "accuracy", "precision", "recall", "f1_score", "auc"]
        
        # Lower is better for these metrics
        lower_is_better = ["mse", "mae", "rmse", "training_time"]
        
        if any(metric in metric_name.lower() for metric in higher_is_better):
            return value1 > value2
        elif any(metric in metric_name.lower() for metric in lower_is_better):
            return value1 < value2
        else:
            # Default: higher is better
            return value1 > value2


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive regression metrics"""
        try:
            logger.debug("Calculating regression metrics")
            
            # Basic regression metrics
            metrics = {
                "r2_score": r2_score(y_true, y_pred, sample_weight=sample_weight),
                "mse": mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
                "mae": mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
                "explained_variance": explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
            }
            
            # Additional metrics
            try:
                metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
            except:
                metrics["mape"] = None
            
            # Residual analysis
            residuals = y_true - y_pred
            metrics["residual_analysis"] = {
                "mean_residual": np.mean(residuals),
                "std_residual": np.std(residuals),
                "residual_skewness": self._calculate_skewness(residuals),
                "residual_kurtosis": self._calculate_kurtosis(residuals)
            }
            
            # Prediction intervals analysis
            metrics["prediction_analysis"] = {
                "mean_prediction": np.mean(y_pred),
                "std_prediction": np.std(y_pred),
                "min_prediction": np.min(y_pred),
                "max_prediction": np.max(y_pred),
                "prediction_range": np.max(y_pred) - np.min(y_pred)
            }
            
            # Error distribution analysis
            absolute_errors = np.abs(residuals)
            metrics["error_distribution"] = {
                "mean_absolute_error": np.mean(absolute_errors),
                "median_absolute_error": np.median(absolute_errors),
                "q75_absolute_error": np.percentile(absolute_errors, 75),
                "q95_absolute_error": np.percentile(absolute_errors, 95),
                "max_absolute_error": np.max(absolute_errors)
            }
            
            return metrics
            
        except Exception as e:
            logger.error("Regression metrics calculation failed", error=str(e), exc_info=True)
            raise
    
    async def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        risk_thresholds: List[float] = [0.3, 0.7]
    ) -> Dict[str, Any]:
        """Calculate classification metrics for risk categories"""
        try:
            logger.debug("Calculating classification metrics")
            
            # Convert continuous predictions to risk categories
            y_true_cat = self._continuous_to_categories(y_true, risk_thresholds)
            y_pred_cat = self._continuous_to_categories(y_pred, risk_thresholds)
            
            # Calculate classification metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            
            metrics = {
                "accuracy": accuracy_score(y_true_cat, y_pred_cat),
                "confusion_matrix": confusion_matrix(y_true_cat, y_pred_cat).tolist()
            }
            
            # Precision, recall, F1 for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_cat, y_pred_cat, average=None, labels=["low", "medium", "high"]
            )
            
            for i, category in enumerate(["low", "medium", "high"]):
                metrics[f"{category}_precision"] = precision[i] if i < len(precision) else 0.0
                metrics[f"{category}_recall"] = recall[i] if i < len(recall) else 0.0
                metrics[f"{category}_f1"] = f1[i] if i < len(f1) else 0.0
                metrics[f"{category}_support"] = int(support[i]) if i < len(support) else 0
            
            # Macro and weighted averages
            metrics["macro_precision"] = np.mean(precision)
            metrics["macro_recall"] = np.mean(recall)
            metrics["macro_f1"] = np.mean(f1)
            
            # Weighted averages
            total_support = np.sum(support)
            if total_support > 0:
                metrics["weighted_precision"] = np.average(precision, weights=support)
                metrics["weighted_recall"] = np.average(recall, weights=support)
                metrics["weighted_f1"] = np.average(f1, weights=support)
            
            return metrics
            
        except Exception as e:
            logger.error("Classification metrics calculation failed", error=str(e), exc_info=True)
            return {}
    
    async def evaluate_model_robustness(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Evaluate model robustness using cross-validation and stability tests"""
        try:
            logger.info("Evaluating model robustness", cv_folds=cv_folds)
            
            # Cross-validation with multiple metrics
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Calculate CV scores for multiple metrics
            cv_metrics = {}
            for metric_name, scoring in [
                ("r2", "r2"),
                ("neg_mse", "neg_mean_squared_error"),
                ("neg_mae", "neg_mean_absolute_error")
            ]:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                cv_metrics[metric_name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "scores": scores.tolist()
                }
            
            # Learning curve analysis
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring="r2",
                random_state=42
            )
            
            learning_curve_data = {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
                "train_scores_std": np.std(train_scores, axis=1).tolist(),
                "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
                "val_scores_std": np.std(val_scores, axis=1).tolist()
            }
            
            # Stability analysis (prediction consistency)
            stability_metrics = await self._analyze_prediction_stability(model, X, y)
            
            robustness_report = {
                "cross_validation": cv_metrics,
                "learning_curve": learning_curve_data,
                "stability": stability_metrics,
                "evaluation_date": datetime.utcnow().isoformat(),
                "sample_size": len(X),
                "feature_count": X.shape[1] if len(X.shape) > 1 else 1
            }
            
            return robustness_report
            
        except Exception as e:
            logger.error("Model robustness evaluation failed", error=str(e), exc_info=True)
            raise
    
    async def _analyze_prediction_stability(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 10
    ) -> Dict[str, Any]:
        """Analyze prediction stability using bootstrap sampling"""
        try:
            predictions_list = []
            
            # Bootstrap sampling
            for i in range(n_bootstrap):
                # Sample with replacement
                n_samples = len(X)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                
                # Clone and train model
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_bootstrap, y_bootstrap)
                
                # Make predictions on original data
                predictions = model_clone.predict(X)
                predictions_list.append(predictions)
            
            # Calculate stability metrics
            predictions_array = np.array(predictions_list)
            prediction_std = np.std(predictions_array, axis=0)
            prediction_mean = np.mean(predictions_array, axis=0)
            
            # Coefficient of variation as stability measure
            cv = prediction_std / (prediction_mean + 1e-8)  # Add small value to avoid division by zero
            
            stability_metrics = {
                "mean_prediction_std": np.mean(prediction_std),
                "median_prediction_std": np.median(prediction_std),
                "mean_coefficient_variation": np.mean(cv),
                "max_coefficient_variation": np.max(cv),
                "stability_score": 1.0 / (1.0 + np.mean(cv)),  # Higher is more stable
                "bootstrap_iterations": n_bootstrap
            }
            
            return stability_metrics
            
        except Exception as e:
            logger.error("Prediction stability analysis failed", error=str(e))
            return {}
    
    def _continuous_to_categories(
        self,
        values: np.ndarray,
        thresholds: List[float] = [0.3, 0.7]
    ) -> List[str]:
        """Convert continuous risk scores to risk categories"""
        categories = []
        
        for value in values:
            if value < thresholds[0]:
                categories.append("low")
            elif value < thresholds[1]:
                categories.append("medium")
            else:
                categories.append("high")
        
        return categories
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            # Fallback calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            # Fallback calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3


class HyperparameterOptimizer:
    """Hyperparameter optimization utilities"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Default parameter grids for different model types
        self.param_grids = {
            "ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr"]
            },
            "lasso": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "max_iter": [1000, 2000, 5000]
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0]
            }
        }
    
    async def optimize_hyperparameters(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        optimization_method: str = "grid_search",
        cv_folds: int = 5,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for given model type"""
        try:
            logger.info(
                "Starting hyperparameter optimization",
                model_type=model_type,
                method=optimization_method
            )
            
            if model_type not in self.param_grids:
                raise ValueError(f"No parameter grid defined for model type: {model_type}")
            
            # Get model class and parameter grid
            from app.services.ml_service import MLService
            ml_service = MLService()
            model_class = ml_service._available_models.get(model_type)
            
            if not model_class:
                raise ValueError(f"Unknown model type: {model_type}")
            
            param_grid = self.param_grids[model_type]
            
            # Create base model
            base_model = model_class()
            
            # Set up cross-validation
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Choose optimization method
            if optimization_method == "grid_search":
                optimizer = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=1
                )
            elif optimization_method == "random_search":
                optimizer = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            # Perform optimization
            start_time = datetime.utcnow()
            optimizer.fit(X, y)
            optimization_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract results
            optimization_results = {
                "best_params": optimizer.best_params_,
                "best_score": optimizer.best_score_,
                "best_estimator": optimizer.best_estimator_,
                "cv_results": {
                    "mean_test_scores": optimizer.cv_results_["mean_test_score"].tolist(),
                    "std_test_scores": optimizer.cv_results_["std_test_score"].tolist(),
                    "params": optimizer.cv_results_["params"]
                },
                "optimization_info": {
                    "method": optimization_method,
                    "cv_folds": cv_folds,
                    "optimization_time_seconds": optimization_time,
                    "n_combinations_tried": len(optimizer.cv_results_["params"])
                }
            }
            
            logger.info(
                "Hyperparameter optimization completed",
                model_type=model_type,
                best_score=optimizer.best_score_,
                optimization_time=optimization_time
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error("Hyperparameter optimization failed", error=str(e), exc_info=True)
            raise
    
    async def validate_hyperparameters(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate hyperparameters for a given model type"""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # Get expected parameters for model type
            expected_params = self.param_grids.get(model_type, {})
            
            # Validate each parameter
            for param, value in hyperparameters.items():
                if param in expected_params:
                    expected_values = expected_params[param]
                    
                    # Check if value is in expected range/list
                    if isinstance(expected_values, list):
                        if value not in expected_values:
                            validation_result["warnings"].append(
                                f"Parameter '{param}' value '{value}' not in typical range: {expected_values}"
                            )
                    
                    # Type validation
                    if isinstance(expected_values, list) and expected_values:
                        expected_type = type(expected_values[0])
                        if not isinstance(value, expected_type):
                            validation_result["errors"].append(
                                f"Parameter '{param}' expected type {expected_type.__name__}, got {type(value).__name__}"
                            )
                            validation_result["valid"] = False
                else:
                    validation_result["warnings"].append(
                        f"Parameter '{param}' is not a standard parameter for {model_type}"
                    )
            
            # Suggest improvements
            if model_type == "ridge" and "alpha" in hyperparameters:
                alpha = hyperparameters["alpha"]
                if alpha < 0.01:
                    validation_result["suggestions"].append("Consider higher alpha value for better regularization")
                elif alpha > 1000:
                    validation_result["suggestions"].append("Very high alpha may lead to underfitting")
            
            return validation_result
            
        except Exception as e:
            logger.error("Hyperparameter validation failed", error=str(e))
            return {"valid": False, "errors": [str(e)]}


class ModelMonitor:
    """Model performance monitoring and drift detection"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def monitor_model_performance(
        self,
        model_version: str,
        recent_predictions: List[Dict[str, Any]],
        performance_window_days: int = 30
    ) -> Dict[str, Any]:
        """Monitor model performance over time"""
        try:
            logger.info(
                "Monitoring model performance",
                model_version=model_version,
                predictions_count=len(recent_predictions)
            )
            
            if not recent_predictions:
                return {
                    "status": "no_data",
                    "message": "No recent predictions available for monitoring"
                }
            
            # Extract prediction data
            risk_scores = [p.get("risk_score", 0) for p in recent_predictions]
            timestamps = [
                datetime.fromisoformat(p.get("timestamp", datetime.utcnow().isoformat()))
                for p in recent_predictions
            ]
            
            # Performance metrics over time
            performance_metrics = {
                "prediction_volume": {
                    "total_predictions": len(recent_predictions),
                    "daily_average": len(recent_predictions) / performance_window_days,
                    "date_range": {
                        "start": min(timestamps).isoformat(),
                        "end": max(timestamps).isoformat()
                    }
                },
                "risk_distribution": {
                    "mean_risk_score": np.mean(risk_scores),
                    "std_risk_score": np.std(risk_scores),
                    "min_risk_score": np.min(risk_scores),
                    "max_risk_score": np.max(risk_scores),
                    "risk_score_percentiles": {
                        "25th": np.percentile(risk_scores, 25),
                        "50th": np.percentile(risk_scores, 50),
                        "75th": np.percentile(risk_scores, 75),
                        "95th": np.percentile(risk_scores, 95)
                    }
                }
            }
            
            # Drift detection
            drift_analysis = await self._detect_prediction_drift(recent_predictions)
            performance_metrics["drift_analysis"] = drift_analysis
            
            # Performance alerts
            alerts = await self._generate_performance_alerts(performance_metrics)
            performance_metrics["alerts"] = alerts
            
            return performance_metrics
            
        except Exception as e:
            logger.error("Model performance monitoring failed", error=str(e), exc_info=True)
            raise
    
    async def _detect_prediction_drift(
        self,
        predictions: List[Dict[str, Any]],
        reference_window_days: int = 30
    ) -> Dict[str, Any]:
        """Detect drift in prediction distributions"""
        try:
            # Split predictions into reference and current windows
            now = datetime.utcnow()
            reference_cutoff = now - timedelta(days=reference_window_days)
            
            reference_predictions = []
            current_predictions = []
            
            for pred in predictions:
                pred_time = datetime.fromisoformat(pred.get("timestamp", now.isoformat()))
                
                if pred_time < reference_cutoff:
                    reference_predictions.append(pred.get("risk_score", 0))
                else:
                    current_predictions.append(pred.get("risk_score", 0))
            
            if len(reference_predictions) < 10 or len(current_predictions) < 10:
                return {
                    "status": "insufficient_data",
                    "message": "Insufficient data for drift detection"
                }
            
            # Statistical drift tests
            drift_metrics = {}
            
            # Mean shift detection
            ref_mean = np.mean(reference_predictions)
            curr_mean = np.mean(current_predictions)
            mean_shift = abs(curr_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
            
            drift_metrics["mean_shift"] = {
                "reference_mean": ref_mean,
                "current_mean": curr_mean,
                "relative_change": mean_shift,
                "significant": mean_shift > 0.1  # 10% threshold
            }
            
            # Distribution shift detection using KS test
            try:
                from scipy.stats import ks_2samp
                ks_stat, ks_p_value = ks_2samp(reference_predictions, current_predictions)
                
                drift_metrics["distribution_shift"] = {
                    "ks_statistic": ks_stat,
                    "p_value": ks_p_value,
                    "significant": ks_p_value < 0.05
                }
            except ImportError:
                logger.warning("scipy not available for KS test")
            
            # Overall drift assessment
            drift_detected = any([
                drift_metrics.get("mean_shift", {}).get("significant", False),
                drift_metrics.get("distribution_shift", {}).get("significant", False)
            ])
            
            return {
                "drift_detected": drift_detected,
                "drift_metrics": drift_metrics,
                "analysis_date": datetime.utcnow().isoformat(),
                "reference_period_days": reference_window_days
            }
            
        except Exception as e:
            logger.error("Drift detection failed", error=str(e))
            return {"status": "error", "message": str(e)}
    
    async def _generate_performance_alerts(
        self,
        performance_metrics: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate performance alerts based on metrics"""
        alerts = []
        
        try:
            # Volume alerts
            daily_average = performance_metrics.get("prediction_volume", {}).get("daily_average", 0)
            if daily_average < 10:
                alerts.append({
                    "level": "info",
                    "message": f"Low prediction volume: {daily_average:.1f} predictions/day"
                })
            elif daily_average > 1000:
                alerts.append({
                    "level": "warning", 
                    "message": f"High prediction volume: {daily_average:.1f} predictions/day"
                })
            
            # Risk distribution alerts
            mean_risk = performance_metrics.get("risk_distribution", {}).get("mean_risk_score", 0)
            if mean_risk > 0.7:
                alerts.append({
                    "level": "warning",
                    "message": f"High average risk score: {mean_risk:.3f} - investigate patient population"
                })
            
            # Drift alerts
            drift_analysis = performance_metrics.get("drift_analysis", {})
            if drift_analysis.get("drift_detected", False):
                alerts.append({
                    "level": "critical",
                    "message": "Prediction drift detected - model retraining may be needed"
                })
            
            return alerts
            
        except Exception as e:
            logger.error("Performance alert generation failed", error=str(e))
            return []


class ModelVersionManager:
    """Model versioning and lifecycle management"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def create_model_version(
        self,
        base_version: str,
        increment_type: str = "patch"  # major, minor, patch
    ) -> str:
        """Create new model version following semantic versioning"""
        try:
            # Parse current version (e.g., "ridge_v1.2.3")
            if "_v" in base_version:
                model_type, version_part = base_version.split("_v")
                version_numbers = version_part.split(".")
            else:
                model_type = base_version
                version_numbers = ["1", "0", "0"]
            
            # Ensure we have major.minor.patch
            while len(version_numbers) < 3:
                version_numbers.append("0")
            
            major, minor, patch = [int(x) for x in version_numbers[:3]]
            
            # Increment based on type
            if increment_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif increment_type == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1
            
            new_version = f"{model_type}_v{major}.{minor}.{patch}"
            
            logger.info("New model version created", old_version=base_version, new_version=new_version)
            return new_version
            
        except Exception as e:
            logger.error("Model version creation failed", error=str(e))
            # Fallback to timestamp-based versioning
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            return f"{base_version}_{timestamp}"
    
    async def archive_old_models(
        self,
        keep_latest_n: int = 5,
        archive_older_than_days: int = 90
    ) -> Dict[str, Any]:
        """Archive old model versions to save space"""
        try:
            logger.info("Archiving old models", keep_latest=keep_latest_n)
            
            models_path = self.settings.ml.MODEL_PATH
            if not models_path.exists():
                return {"archived_count": 0, "message": "No models directory found"}
            
            # Get all model directories
            model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
            
            # Sort by creation time (newest first)
            model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Determine which models to archive
            models_to_archive = []
            cutoff_date = datetime.utcnow() - timedelta(days=archive_older_than_days)
            
            for i, model_dir in enumerate(model_dirs):
                model_age = datetime.fromtimestamp(model_dir.stat().st_mtime)
                
                # Archive if it's beyond the keep limit OR too old
                if i >= keep_latest_n or model_age < cutoff_date:
                    models_to_archive.append(model_dir)
            
            # Create archive directory
            archive_path = models_path / "archived"
            archive_path.mkdir(exist_ok=True)
            
            archived_count = 0
            for model_dir in models_to_archive:
                try:
                    # Move to archive
                    archive_target = archive_path / model_dir.name
                    model_dir.rename(archive_target)
                    archived_count += 1
                    
                    logger.debug("Model archived", version=model_dir.name)
                    
                except Exception as e:
                    logger.error("Failed to archive model", version=model_dir.name, error=str(e))
            
            archive_summary = {
                "archived_count": archived_count,
                "total_models_before": len(model_dirs),
                "active_models_remaining": len(model_dirs) - archived_count,
                "archive_criteria": {
                    "keep_latest_n": keep_latest_n,
                    "archive_older_than_days": archive_older_than_days
                },
                "archive_path": str(archive_path)
            }
            
            logger.info("Model archiving completed", archived_count=archived_count)
            return archive_summary
            
        except Exception as e:
            logger.error("Model archiving failed", error=str(e), exc_info=True)
            raise


class ModelComparator:
    """Compare and benchmark different models"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def compare_model_performance(
        self,
        models_data: List[Dict[str, Any]],
        comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        try:
            if comparison_metrics is None:
                comparison_metrics = ["r2_score", "mse", "mae", "cross_val_score"]
            
            logger.info("Comparing model performance", models_count=len(models_data))
            
            comparison_results = {
                "models": [],
                "rankings": {},
                "summary": {},
                "comparison_date": datetime.utcnow().isoformat()
            }
            
            # Extract metrics for each model
            for model_data in models_data:
                model_info = {
                    "version": model_data.get("version"),
                    "type": model_data.get("type"),
                    "metrics": {}
                }
                
                # Extract metrics
                performance = model_data.get("performance_metrics", {})
                for metric in comparison_metrics:
                    value = self._extract_nested_metric(performance, metric)
                    model_info["metrics"][metric] = value
                
                comparison_results["models"].append(model_info)
            
            # Rank models by each metric
            for metric in comparison_metrics:
                model_scores = [
                    (model["version"], model["metrics"].get(metric))
                    for model in comparison_results["models"]
                    if model["metrics"].get(metric) is not None
                ]
                
                # Sort based on metric type (higher or lower is better)
                reverse_sort = metric in ["r2_score", "accuracy", "precision", "recall", "f1_score"]
                model_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
                
                comparison_results["rankings"][metric] = [
                    {"rank": i + 1, "version": version, "value": score}
                    for i, (version, score) in enumerate(model_scores)
                ]
            
            # Generate summary insights
            comparison_results["summary"] = await self._generate_comparison_summary(
                comparison_results["models"],
                comparison_results["rankings"]
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error("Model comparison failed", error=str(e), exc_info=True)
            raise
    
    def _extract_nested_metric(self, performance_data: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from nested performance data"""
        try:
            # Try different paths where the metric might be stored
            paths_to_try = [
                ["validation", metric_name],
                ["training", metric_name],
                ["cross_validation", metric_name],
                [metric_name]
            ]
            
            for path in paths_to_try:
                current_level = performance_data
                
                for key in path:
                    if isinstance(current_level, dict) and key in current_level:
                        current_level = current_level[key]
                    else:
                        current_level = None
                        break
                
                if isinstance(current_level, (int, float)):
                    return float(current_level)
            
            return None
            
        except Exception:
            return None
    
    async def _generate_comparison_summary(
        self,
        models: List[Dict[str, Any]],
        rankings: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate summary insights from model comparison"""
        try:
            summary = {
                "total_models_compared": len(models),
                "best_overall_model": None,
                "metric_leaders": {},
                "insights": []
            }
            
            # Find best model for each metric
            for metric, ranking in rankings.items():
                if ranking:
                    best_model = ranking[0]
                    summary["metric_leaders"][metric] = {
                        "version": best_model["version"],
                        "value": best_model["value"]
                    }
            
            # Determine overall best model (based on R² score primarily)
            if "r2_score" in rankings and rankings["r2_score"]:
                summary["best_overall_model"] = rankings["r2_score"][0]["version"]
            
            # Generate insights
            if len(models) > 1:
                # Check for significant performance differences
                r2_scores = [
                    model["metrics"].get("r2_score")
                    for model in models
                    if model["metrics"].get("r2_score") is not None
                ]
                
                if r2_scores:
                    r2_range = max(r2_scores) - min(r2_scores)
                    if r2_range > 0.1:
                        summary["insights"].append(
                            f"Significant performance variation detected (R² range: {r2_range:.3f})"
                        )
                    
                    if max(r2_scores) < 0.7:
                        summary["insights"].append("All models show moderate performance - consider feature engineering")
                    elif max(r2_scores) > 0.95:
                        summary["insights"].append("Very high performance detected - check for overfitting")
            
            return summary
            
        except Exception as e:
            logger.error("Comparison summary generation failed", error=str(e))
            return {}


# Utility functions
def calculate_model_signature(
    model_type: str,
    hyperparameters: Dict[str, Any],
    feature_columns: List[str]
) -> str:
    """Calculate a unique signature for a model configuration"""
    try:
        # Create a deterministic string representation
        signature_data = {
            "model_type": model_type,
            "hyperparameters": sorted(hyperparameters.items()) if hyperparameters else [],
            "feature_columns": sorted(feature_columns)
        }
        
        signature_string = json.dumps(signature_data, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
        
    except Exception as e:
        logger.error("Model signature calculation failed", error=str(e))
        return f"unknown_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def validate_model_performance(
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Validate model performance against thresholds"""
    try:
        if thresholds is None:
            thresholds = {
                "r2_score": 0.7,      # Minimum R² score
                "mse": 1000000,       # Maximum MSE (depends on data scale)
                "cross_val_std": 0.1  # Maximum CV standard deviation
            }
        
        validation_results = {
            "meets_requirements": True,
            "failed_checks": [],
            "warnings": [],
            "performance_grade": "A"
        }
        
        # Check each threshold
        for metric, threshold in thresholds.items():
            metric_value = metrics.get(metric)
            
            if metric_value is not None:
                if metric in ["r2_score", "accuracy"] and metric_value < threshold:
                    validation_results["meets_requirements"] = False
                    validation_results["failed_checks"].append(
                        f"{metric}: {metric_value:.4f} < {threshold}"
                    )
                elif metric in ["mse", "mae"] and metric_value > threshold:
                    validation_results["meets_requirements"] = False
                    validation_results["failed_checks"].append(
                        f"{metric}: {metric_value:.4f} > {threshold}"
                    )
        
        # Determine performance grade
        if not validation_results["meets_requirements"]:
            validation_results["performance_grade"] = "F"
        elif len(validation_results["warnings"]) > 2:
            validation_results["performance_grade"] = "C"
        elif len(validation_results["warnings"]) > 0:
            validation_results["performance_grade"] = "B"
        
        return validation_results
        
    except Exception as e:
        logger.error("Performance validation failed", error=str(e))
        return {
            "meets_requirements": False,
            "failed_checks": [str(e)],
            "performance_grade": "F"
        }


# Export all model utilities
__all__ = [
    "ModelManager",
    "ModelEvaluator",
    "HyperparameterOptimizer", 
    "ModelMonitor",
    "ModelComparator",
    "ModelVersionManager",
    "calculate_model_signature",
    "validate_model_performance"
]