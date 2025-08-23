import logging
import pickle
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVDModelTrainer:
    """
    Comprehensive trainer for cardiovascular disease prediction models.
    
    This class handles data preprocessing, model selection, hyperparameter tuning,
    training, and evaluation for cardiovascular disease prediction.
    """
    
    def __init__(self, data_path: str, model_output_path: str = "models/"):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to the training dataset
            model_output_path: Directory to save trained models
        """
        self.data_path = Path(data_path)
        self.model_output_path = Path(model_output_path)
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.feature_names = []
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'classifier__C': [0.01, 0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['rbf', 'linear'],
                    'classifier__gamma': ['scale', 'auto']
                }
            }
        }
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess the cardiovascular disease dataset.
        
        Returns:
            Tuple of features DataFrame and target Series
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load the dataset
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Handle missing values
        logger.info("Handling missing values...")
        
        # Fill numerical missing values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'HeartDisease' and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        logger.info("Encoding categorical variables...")
        for col in categorical_cols:
            if col != 'HeartDisease':  # Don't encode the target variable yet
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Prepare features and target
        target_col = 'HeartDisease'  # Assuming this is the target column
        if target_col not in df.columns:
            # If HeartDisease not present, create based on cardiovascular conditions
            cardiovascular_indicators = ['CHD', 'MI', 'Angina', 'Stroke']
            available_indicators = [col for col in cardiovascular_indicators if col in df.columns]
            if available_indicators:
                df[target_col] = df[available_indicators].any(axis=1).astype(int)
            else:
                raise ValueError("No cardiovascular disease indicators found in dataset")
        
        # Encode target variable
        if df[target_col].dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(df[target_col])
            self.label_encoders['target'] = le_target
        else:
            y = df[target_col].values
        
        # Prepare features
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, pd.Series(y)
    
    def create_pipeline(self, model) -> Pipeline:
        """
        Create a preprocessing and modeling pipeline.
        
        Args:
            model: The sklearn model to use
            
        Returns:
            Complete pipeline with preprocessing and model
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        return pipeline
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter tuning.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary containing trained models and their scores
        """
        logger.info("Starting model training and hyperparameter tuning...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        best_score = 0
        
        # Start MLflow experiment
        mlflow.set_experiment("cardiovascular_prediction")
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                try:
                    # Create pipeline
                    pipeline = self.create_pipeline(config['model'])
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        pipeline,
                        config['params'],
                        cv=5,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    
                    # Make predictions
                    y_pred = best_model.predict(X_test)
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1': f1_score(y_test, y_pred, average='weighted'),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    }
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
                    metrics['cv_roc_auc_mean'] = cv_scores.mean()
                    metrics['cv_roc_auc_std'] = cv_scores.std()
                    
                    # Log parameters and metrics to MLflow
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metrics(metrics)
                    
                    # Save model
                    model_path = self.model_output_path / f"{model_name}_model.pkl"
                    joblib.dump(best_model, model_path)
                    mlflow.sklearn.log_model(best_model, f"{model_name}_model")
                    
                    # Store results
                    results[model_name] = {
                        'model': best_model,
                        'metrics': metrics,
                        'best_params': grid_search.best_params_,
                        'cv_scores': cv_scores
                    }
                    
                    logger.info(f"{model_name} - ROC AUC: {metrics['roc_auc']:.4f}")
                    
                    # Track best model
                    if metrics['roc_auc'] > best_score:
                        best_score = metrics['roc_auc']
                        self.best_model = best_model
                        self.best_model_name = model_name
                
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
        
        return results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return evaluation
    
    def save_artifacts(self):
        """Save all training artifacts."""
        artifacts = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name
        }
        
        artifacts_path = self.model_output_path / 'training_artifacts.pkl'
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Training artifacts saved to {artifacts_path}")
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Training results and model performance metrics
        """
        logger.info("Starting cardiovascular disease prediction model training pipeline...")
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Train models
        results = self.train_models(X, y)
        
        # Save artifacts
        self.save_artifacts()
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Best model: {self.best_model_name}")
        
        return results


def main():
    """Main function to run the training pipeline."""
    # Configuration
    DATA_PATH = "data/processed/cardiovascular_disease_data.csv"
    MODEL_OUTPUT_PATH = "backend/ml_models/trained_models/"
    
    # Initialize trainer
    trainer = CVDModelTrainer(DATA_PATH, MODEL_OUTPUT_PATH)
    
    # Run training pipeline
    results = trainer.run_training_pipeline()
    
    # Print results summary
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ROC AUC: {result['metrics']['roc_auc']:.4f}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  F1 Score: {result['metrics']['f1']:.4f}")
        print(f"  CV ROC AUC: {result['metrics']['cv_roc_auc_mean']:.4f} ± {result['metrics']['cv_roc_auc_std']:.4f}")


if __name__ == "__main__":
    main()