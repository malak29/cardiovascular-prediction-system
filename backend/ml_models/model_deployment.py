"""
Cardiovascular Disease Prediction System - Model Deployment Script

This module handles model deployment, versioning, A/B testing, and 
automated model monitoring in production environments.
"""

import logging
import os
import json
import pickle
import joblib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import boto3
from botocore.exceptions import ClientError
import docker
import requests
import redis
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import mlflow
import mlflow.sklearn
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
import yaml
import hashlib
import psutil
import time
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()

class ModelDeployment(Base):
    """Database model for tracking model deployments."""
    __tablename__ = 'model_deployments'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    deployment_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='active')  # active, inactive, testing
    model_path = Column(String(500))
    model_hash = Column(String(64))
    performance_metrics = Column(Text)  # JSON string
    traffic_split = Column(Float, default=100.0)  # Percentage of traffic
    rollback_version = Column(String(50))
    created_by = Column(String(100))
    environment = Column(String(50))  # dev, staging, production
    notes = Column(Text)

class ModelPerformanceMetrics(Base):
    """Database model for storing model performance metrics over time."""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String(50))
    metric_value = Column(Float)
    environment = Column(String(50))
    batch_id = Column(String(100))

class CVDModelDeployment:
    """
    Comprehensive model deployment system for cardiovascular disease prediction models.
    
    This class handles model deployment, versioning, A/B testing, rollback,
    and automated monitoring in production environments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model deployment system.
        
        Args:
            config: Configuration dictionary containing deployment settings
        """
        self.config = config
        self.models_path = Path(config.get('models_path', 'backend/ml_models/'))
        self.deployment_path = Path(config.get('deployment_path', 'backend/ml_models/deployed/'))
        self.backup_path = Path(config.get('backup_path', 'backend/ml_models/backup/'))
        
        # Create directories
        self.deployment_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        db_url = config.get('database_url', 'sqlite:///model_deployment.db')
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Redis for model caching and feature store
        self.redis_client = None
        if config.get('redis_url'):
            self.redis_client = redis.from_url(config['redis_url'])
        
        # AWS S3 for model storage
        self.s3_client = None
        if config.get('aws_s3_bucket'):
            self.s3_client = boto3.client('s3')
            self.s3_bucket = config['aws_s3_bucket']
        
        # Docker client for containerized deployments
        self.docker_client = None
        if config.get('use_docker'):
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker client initialization failed: {str(e)}")
        
        # MLflow tracking
        if config.get('mlflow_tracking_uri'):
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        
        # Prometheus metrics
        self.metrics_registry = CollectorRegistry()
        self.setup_prometheus_metrics()
        
        # Current active models
        self.active_models = {}
        self.load_active_models()
    
    def setup_prometheus_metrics(self):
        """Set up Prometheus metrics for monitoring."""
        self.prediction_counter = Counter(
            'cvd_predictions_total',
            'Total number of predictions made',
            ['model_name', 'version', 'environment'],
            registry=self.metrics_registry
        )
        
        self.prediction_latency = Histogram(
            'cvd_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_name', 'version'],
            registry=self.metrics_registry
        )
        
        self.model_accuracy = Gauge(
            'cvd_model_accuracy',
            'Current model accuracy',
            ['model_name', 'version', 'environment'],
            registry=self.metrics_registry
        )
        
        self.model_drift = Gauge(
            'cvd_model_drift_score',
            'Model drift detection score',
            ['model_name', 'version'],
            registry=self.metrics_registry
        )
    
    def load_active_models(self):
        """Load currently active models from the database."""
        active_deployments = self.session.query(ModelDeployment).filter_by(status='active').all()
        
        for deployment in active_deployments:
            try:
                model_path = Path(deployment.model_path)
                if model_path.exists():
                    model = joblib.load(model_path)
                    self.active_models[deployment.model_name] = {
                        'model': model,
                        'version': deployment.version,
                        'deployment': deployment
                    }
                    logger.info(f"Loaded active model: {deployment.model_name} v{deployment.version}")
            except Exception as e:
                logger.error(f"Failed to load model {deployment.model_name}: {str(e)}")
    
    def calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of the model file for integrity checking."""
        with open(model_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def validate_model_before_deployment(self, model_path: Path, test_data_path: Path) -> Dict[str, Any]:
        """
        Validate model performance before deployment.
        
        Args:
            model_path: Path to the model file
            test_data_path: Path to test dataset
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating model before deployment...")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Load test data
            test_df = pd.read_csv(test_data_path)
            
            # Prepare test data (assuming similar preprocessing as training)
            y_test = test_df['HeartDisease']  # Adjust column name as needed
            X_test = test_df.drop(columns=['HeartDisease'])
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'prediction_time': prediction_time,
                'predictions_per_second': len(X_test) / prediction_time,
                'test_samples': len(X_test)
            }
            
            # Performance thresholds
            performance_requirements = {
                'min_accuracy': 0.75,
                'min_precision': 0.70,
                'min_recall': 0.70,
                'min_f1_score': 0.70,
                'min_roc_auc': 0.80,
                'max_prediction_time_per_sample': 0.1  # 100ms per sample
            }
            
            # Check performance requirements
            validation_passed = True
            issues = []
            
            for metric, min_value in performance_requirements.items():
                if metric.startswith('min_'):
                    actual_metric = metric.replace('min_', '')
                    if metrics.get(actual_metric, 0) < min_value:
                        validation_passed = False
                        issues.append(f"{actual_metric} ({metrics[actual_metric]:.3f}) below threshold ({min_value})")
                elif metric == 'max_prediction_time_per_sample':
                    time_per_sample = metrics['prediction_time'] / metrics['test_samples']
                    if time_per_sample > min_value:
                        validation_passed = False
                        issues.append(f"Prediction time per sample ({time_per_sample:.3f}s) exceeds threshold ({min_value}s)")
            
            return {
                'passed': validation_passed,
                'metrics': metrics,
                'issues': issues,
                'model_size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return {
                'passed': False,
                'metrics': {},
                'issues': [f"Validation error: {str(e)}"],
                'model_size_mb': 0
            }
    
    def deploy_model(self, model_name: str, model_path: Path, version: str,
                    environment: str = 'production', traffic_split: float = 100.0,
                    validation_data_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Deploy a model to the specified environment.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            version: Model version
            environment: Deployment environment (dev, staging, production)
            traffic_split: Percentage of traffic to route to this model (for A/B testing)
            validation_data_path: Path to validation dataset
            
        Returns:
            Dictionary containing deployment results
        """
        logger.info(f"Deploying model {model_name} version {version} to {environment}")
        
        deployment_result = {
            'success': False,
            'model_name': model_name,
            'version': version,
            'environment': environment,
            'deployment_time': datetime.utcnow(),
            'issues': []
        }
        
        try:
            # Validate model if validation data provided
            if validation_data_path and validation_data_path.exists():
                validation_result = self.validate_model_before_deployment(model_path, validation_data_path)
                if not validation_result['passed']:
                    deployment_result['issues'].extend(validation_result['issues'])
                    logger.error(f"Model validation failed: {validation_result['issues']}")
                    return deployment_result
                
                deployment_result['validation_metrics'] = validation_result['metrics']
            
            # Calculate model hash
            model_hash = self.calculate_model_hash(model_path)
            
            # Create deployment directory
            deployment_dir = self.deployment_path / environment / model_name / version
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model to deployment directory
            deployed_model_path = deployment_dir / 'model.pkl'
            shutil.copy2(model_path, deployed_model_path)
            
            # Copy artifacts if they exist
            artifacts_path = model_path.parent / 'training_artifacts.pkl'
            if artifacts_path.exists():
                shutil.copy2(artifacts_path, deployment_dir / 'training_artifacts.pkl')
            
            # Create model metadata
            metadata = {
                'model_name': model_name,
                'version': version,
                'deployment_date': datetime.utcnow().isoformat(),
                'model_hash': model_hash,
                'environment': environment,
                'model_path': str(deployed_model_path),
                'traffic_split': traffic_split,
                'performance_metrics': deployment_result.get('validation_metrics', {}),
                'system_info': {
                    'python_version': os.sys.version,
                    'cpu_count': psutil.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total / (1024**3)
                }
            }
            
            # Save metadata
            with open(deployment_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Upload to S3 if configured
            if self.s3_client:
                try:
                    s3_key = f"models/{environment}/{model_name}/{version}"
                    self.s3_client.upload_file(str(deployed_model_path), self.s3_bucket, f"{s3_key}/model.pkl")
                    self.s3_client.upload_file(str(deployment_dir / 'metadata.json'), self.s3_bucket, f"{s3_key}/metadata.json")
                    logger.info(f"Model uploaded to S3: {s3_key}")
                except ClientError as e:
                    deployment_result['issues'].append(f"S3 upload failed: {str(e)}")
            
            # Update database
            # Check if deployment already exists
            existing_deployment = self.session.query(ModelDeployment).filter_by(
                model_name=model_name, version=version, environment=environment
            ).first()
            
            if existing_deployment:
                # Update existing deployment
                existing_deployment.status = 'active'
                existing_deployment.deployment_date = datetime.utcnow()
                existing_deployment.model_path = str(deployed_model_path)
                existing_deployment.model_hash = model_hash
                existing_deployment.performance_metrics = json.dumps(deployment_result.get('validation_metrics', {}))
                existing_deployment.traffic_split = traffic_split
            else:
                # Create new deployment record
                new_deployment = ModelDeployment(
                    model_name=model_name,
                    version=version,
                    deployment_date=datetime.utcnow(),
                    status='active',
                    model_path=str(deployed_model_path),
                    model_hash=model_hash,
                    performance_metrics=json.dumps(deployment_result.get('validation_metrics', {})),
                    traffic_split=traffic_split,
                    environment=environment,
                    created_by=os.getenv('USER', 'system'),
                    notes=f"Deployed via deployment script"
                )
                self.session.add(new_deployment)
            
            self.session.commit()
            
            # Load model into memory for immediate use
            if environment == 'production':
                model = joblib.load(deployed_model_path)
                self.active_models[model_name] = {
                    'model': model,
                    'version': version,
                    'deployment': existing_deployment or new_deployment
                }
            
            # Cache model in Redis if available
            if self.redis_client and environment == 'production':
                try:
                    with open(deployed_model_path, 'rb') as f:
                        model_data = f.read()
                    self.redis_client.set(f"model:{model_name}:{version}", model_data, ex=86400)  # 24 hours
                    logger.info(f"Model cached in Redis: {model_name}:{version}")
                except Exception as e:
                    deployment_result['issues'].append(f"Redis caching failed: {str(e)}")
            
            # Update Prometheus metrics
            if deployment_result.get('validation_metrics'):
                metrics = deployment_result['validation_metrics']
                self.model_accuracy.labels(
                    model_name=model_name, 
                    version=version, 
                    environment=environment
                ).set(metrics.get('accuracy', 0))
            
            # Create Docker container if configured
            if self.docker_client and self.config.get('use_docker'):
                try:
                    self._create_docker_deployment(model_name, version, deployment_dir, environment)
                except Exception as e:
                    deployment_result['issues'].append(f"Docker deployment failed: {str(e)}")
            
            deployment_result['success'] = True
            deployment_result['deployed_path'] = str(deployed_model_path)
            deployment_result['model_hash'] = model_hash
            
            logger.info(f"Successfully deployed {model_name} v{version} to {environment}")
            
        except Exception as e:
            deployment_result['issues'].append(f"Deployment error: {str(e)}")
            logger.error(f"Deployment failed: {str(e)}")
        
        return deployment_result
    
    def _create_docker_deployment(self, model_name: str, version: str, deployment_dir: Path, environment: str):
        """Create Docker container for model deployment."""
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and artifacts
COPY model.pkl .
COPY training_artifacts.pkl .
COPY metadata.json .

# Copy prediction service
COPY predict_service.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start service
CMD ["uvicorn", "predict_service:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        dockerfile_path = deployment_dir / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create prediction service
        service_content = '''
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import json

app = FastAPI(title="CVD Prediction Service")

# Load model and artifacts
model = joblib.load("model.pkl")
with open("training_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
        
        service_path = deployment_dir / 'predict_service.py'
        with open(service_path, 'w') as f:
            f.write(service_content.replace('version', f'"{version}"'))
        
        # Build Docker image
        image_name = f"cvd-model-{model_name}:{version}"
        try:
            self.docker_client.images.build(
                path=str(deployment_dir),
                tag=image_name,
                rm=True
            )
            logger.info(f"Built Docker image: {image_name}")
            
            # Run container
            container_name = f"cvd-{model_name}-{version}-{environment}"
            port = 8000 + hash(container_name) % 1000  # Dynamic port assignment
            
            self.docker_client.containers.run(
                image_name,
                name=container_name,
                ports={'8000/tcp': port},
                detach=True,
                restart_policy={'Name': 'unless-stopped'}
            )
            logger.info(f"Started container: {container_name} on port {port}")
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {str(e)}")
            raise
    
    def rollback_model(self, model_name: str, environment: str = 'production') -> Dict[str, Any]:
        """
        Rollback to the previous version of a model.
        
        Args:
            model_name: Name of the model to rollback
            environment: Environment to rollback in
            
        Returns:
            Dictionary containing rollback results
        """
        logger.info(f"Rolling back model {model_name} in {environment}")
        
        try:
            # Get current active deployment
            current_deployment = self.session.query(ModelDeployment).filter_by(
                model_name=model_name, 
                environment=environment,
                status='active'
            ).first()
            
            if not current_deployment:
                return {'success': False, 'message': 'No active deployment found'}
            
            # Find previous deployment
            previous_deployment = self.session.query(ModelDeployment).filter(
                ModelDeployment.model_name == model_name,
                ModelDeployment.environment == environment,
                ModelDeployment.deployment_date < current_deployment.deployment_date
            ).order_by(ModelDeployment.deployment_date.desc()).first()
            
            if not previous_deployment:
                return {'success': False, 'message': 'No previous deployment found'}
            
            # Update database - deactivate current, activate previous
            current_deployment.status = 'inactive'
            current_deployment.rollback_version = previous_deployment.version
            
            previous_deployment.status = 'active'
            previous_deployment.deployment_date = datetime.utcnow()
            
            self.session.commit()
            
            # Reload model into memory
            if environment == 'production':
                model_path = Path(previous_deployment.model_path)
                if model_path.exists():
                    model = joblib.load(model_path)
                    self.active_models[model_name] = {
                        'model': model,
                        'version': previous_deployment.version,
                        'deployment': previous_deployment
                    }
            
            logger.info(f"Rolled back {model_name} from {current_deployment.version} to {previous_deployment.version}")
            
            return {
                'success': True,
                'previous_version': current_deployment.version,
                'current_version': previous_deployment.version,
                'rollback_time': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def monitor_model_performance(self, model_name: str, batch_predictions: List[Dict],
                                 actual_outcomes: List[int]) -> Dict[str, Any]:
        """
        Monitor model performance in production.
        
        Args:
            model_name: Name of the model
            batch_predictions: List of prediction results
            actual_outcomes: List of actual outcomes (for comparison)
            
        Returns:
            Dictionary containing performance monitoring results
        """
        if model_name not in self.active_models:
            return {'error': 'Model not active'}
        
        model_info = self.active_models[model_name]
        version = model_info['version']
        
        # Extract predictions and probabilities
        predictions = [pred['prediction'] for pred in batch_predictions]
        probabilities = [pred['probability'] for pred in batch_predictions]
        
        # Calculate performance metrics
        metrics = {
            'accuracy': accuracy_score(actual_outcomes, predictions),
            'precision': precision_score(actual_outcomes, predictions, average='weighted'),
            'recall': recall_score(actual_outcomes, predictions, average='weighted'),
            'f1_score': f1_score(actual_outcomes, predictions, average='weighted'),
            'roc_auc': roc_auc_score(actual_outcomes, probabilities),
            'sample_size': len(predictions)
        }
        
        # Store metrics in database
        batch_id = f"{model_name}_{version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        for metric_name, metric_value in metrics.items():
            if metric_name != 'sample_size':
                performance_record = ModelPerformanceMetrics(
                    model_name=model_name,
                    version=version,
                    timestamp=datetime.utcnow(),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    environment='production',
                    batch_id=batch_id
                )
                self.session.add(performance_record)
        
        self.session.commit()
        
        # Update Prometheus metrics
        self.model_accuracy.labels(
            model_name=model_name,
            version=version,
            environment='production'
        ).set(metrics['accuracy'])
        
        # Check for performance degradation
        performance_alerts = []
        
        # Get recent performance history
        recent_metrics = self.session.query(ModelPerformanceMetrics).filter(
            ModelPerformanceMetrics.model_name == model_name,
            ModelPerformanceMetrics.version == version,
            ModelPerformanceMetrics.metric_name == 'accuracy',
            ModelPerformanceMetrics.timestamp > datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if len(recent_metrics) > 1:
            recent_accuracy = [m.metric_value for m in recent_metrics]
            avg_recent_accuracy = np.mean(recent_accuracy)
            
            # Alert if accuracy dropped significantly
            if metrics['accuracy'] < avg_recent_accuracy * 0.95:  # 5% degradation
                performance_alerts.append({
                    'type': 'performance_degradation',
                    'message': f"Accuracy dropped from {avg_recent_accuracy:.3f} to {metrics['accuracy']:.3f}",
                    'severity': 'warning'
                })
        
        # Push metrics to Prometheus gateway if configured
        if self.config.get('prometheus_gateway'):
            try:
                push_to_gateway(
                    self.config['prometheus_gateway'],
                    job='cvd_model_monitoring',
                    registry=self.metrics_registry
                )
            except Exception as e:
                logger.error(f"Failed to push metrics to Prometheus: {str(e)}")
        
        return {
            'metrics': metrics,
            'batch_id': batch_id,
            'alerts': performance_alerts,
            'timestamp': datetime.utcnow()
        }
    
    def list_deployments(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all model deployments.
        
        Args:
            environment: Filter by environment (optional)
            
        Returns:
            List of deployment information
        """
        query = self.session.query(ModelDeployment)
        
        if environment:
            query = query.filter_by(environment=environment)
        
        deployments = query.order_by(ModelDeployment.deployment_date.desc()).all()
        
        deployment_list = []
        for deployment in deployments:
            try:
                performance_metrics = json.loads(deployment.performance_metrics or '{}')
            except:
                performance_metrics = {}
            
            deployment_list.append({
                'model_name': deployment.model_name,
                'version': deployment.version,
                'environment': deployment.environment,
                'status': deployment.status,
                'deployment_date': deployment.deployment_date.isoformat(),
                'traffic_split': deployment.traffic_split,
                'performance_metrics': performance_metrics,
                'model_path': deployment.model_path,
                'created_by': deployment.created_by
            })
        
        return deployment_list
    
    def cleanup_old_deployments(self, keep_versions: int = 5):
        """
        Clean up old model deployments to save space.
        
        Args:
            keep_versions: Number of versions to keep per model
        """
        logger.info(f"Cleaning up old deployments, keeping {keep_versions} versions per model")
        
        # Get all model names
        model_names = self.session.query(ModelDeployment.model_name).distinct().all()
        
        for (model_name,) in model_names:
            # Get deployments for this model, ordered by deployment date
            deployments = self.session.query(ModelDeployment).filter_by(
                model_name=model_name
            ).order_by(ModelDeployment.deployment_date.desc()).all()
            
            # Keep the most recent versions, mark others for cleanup
            deployments_to_cleanup = deployments[keep_versions:]
            
            for deployment in deployments_to_cleanup:
                if deployment.status != 'active':  # Don't cleanup active deployments
                    # Remove files
                    model_path = Path(deployment.model_path)
                    if model_path.exists():
                        # Move to backup instead of deleting
                        backup_dir = self.backup_path / f"{model_name}_{deployment.version}"
                        backup_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(model_path.parent), str(backup_dir))
                        logger.info(f"Moved {model_name} v{deployment.version} to backup")
                    
                    # Update database status
                    deployment.status = 'archived'
        
        self.session.commit()
        logger.info("Deployment cleanup completed")


def main():
    """Main function to run model deployment operations."""
    # Configuration
    config = {
        'models_path': 'backend/ml_models/trained_models/',
        'deployment_path': 'backend/ml_models/deployed/',
        'backup_path': 'backend/ml_models/backup/',
        'database_url': 'sqlite:///model_deployment.db',
        # 'redis_url': 'redis://localhost:6379',
        # 'aws_s3_bucket': 'your-cvd-models-bucket',
        # 'use_docker': True,
        # 'prometheus_gateway': 'localhost:9091'
    }
    
    # Initialize deployment system
    deployment_system = CVDModelDeployment(config)
    
    # Example: Deploy a model
    model_path = Path(config['models_path']) / 'random_forest_model.pkl'
    validation_data_path = Path('data/processed/test_data.csv')
    
    if model_path.exists():
        result = deployment_system.deploy_model(
            model_name='cardiovascular_risk_predictor',
            model_path=model_path,
            version='v1.0.0',
            environment='production',
            traffic_split=100.0,
            validation_data_path=validation_data_path if validation_data_path.exists() else None
        )
        
        print("\n" + "="*50)
        print("MODEL DEPLOYMENT SUMMARY")
        print("="*50)
        print(f"Success: {result['success']}")
        print(f"Model: {result['model_name']} v{result['version']}")
        print(f"Environment: {result['environment']}")
        
        if result.get('validation_metrics'):
            metrics = result['validation_metrics']
            print(f"\nValidation Metrics:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
            print(f"  Predictions/sec: {metrics.get('predictions_per_second', 0):.0f}")
        
        if result.get('issues'):
            print(f"\nIssues:")
            for issue in result['issues']:
                print(f"  - {issue}")
    
    # List all deployments
    deployments = deployment_system.list_deployments()
    if deployments:
        print(f"\nActive Deployments: {len([d for d in deployments if d['status'] == 'active'])}")
        for deployment in deployments[:3]:  # Show first 3
            print(f"  {deployment['model_name']} v{deployment['version']} ({deployment['status']})")


if __name__ == "__main__":
    main()