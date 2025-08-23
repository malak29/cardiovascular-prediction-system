import logging
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import learning_curve, validation_curve
import shap
from lime.lime_tabular import LimeTabularExplainer
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVDModelEvaluator:
    """
    Comprehensive evaluator for cardiovascular disease prediction models.
    
    This class provides detailed model evaluation including performance metrics,
    fairness analysis, calibration assessment, and interpretability analysis.
    """
    
    def __init__(self, model_path: str, artifacts_path: str, results_path: str = "evaluation_results/"):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
            artifacts_path: Path to training artifacts (scaler, encoders, etc.)
            results_path: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.artifacts_path = Path(artifacts_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Load model and artifacts
        self.model = None
        self.artifacts = None
        self.load_model_and_artifacts()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_model_and_artifacts(self):
        """Load the trained model and preprocessing artifacts."""
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            # Load artifacts
            logger.info(f"Loading artifacts from {self.artifacts_path}")
            with open(self.artifacts_path, 'rb') as f:
                self.artifacts = pickle.load(f)
                
            logger.info("Model and artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model or artifacts: {str(e)}")
            raise
    
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary containing basic metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'specificity': self._compute_specificity(y_true, y_pred),
            'npv': self._compute_npv(y_true, y_pred),  # Negative Predictive Value
            'sensitivity': recall_score(y_true, y_pred)  # Same as recall
        }
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _compute_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute negative predictive value."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return 0.0
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> str:
        """
        Plot ROC curve and save to file.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Path to saved plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.results_path / 'roc_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> str:
        """
        Plot precision-recall curve and save to file.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Path to saved plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5,
                   label=f'Baseline (AP = {np.mean(y_true):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.results_path / 'precision_recall_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Plot confusion matrix and save to file.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No CVD', 'CVD'],
                   yticklabels=['No CVD', 'CVD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plot_path = self.results_path / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def calibration_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyze model calibration.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Calibration analysis results
        """
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='quantile'
        )
        
        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                linewidth=2, label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        calibration_plot_path = self.results_path / 'calibration_plot.png'
        plt.savefig(calibration_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Brier score (lower is better)
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        
        return {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'brier_score': brier_score,
            'calibration_plot_path': str(calibration_plot_path)
        }
    
    def fairness_analysis(self, X_test: pd.DataFrame, y_true: np.ndarray, 
                         y_pred: np.ndarray, y_pred_proba: np.ndarray,
                         sensitive_features: List[str]) -> Dict[str, Any]:
        """
        Analyze model fairness across different demographic groups.
        
        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            sensitive_features: List of sensitive feature names
            
        Returns:
            Fairness analysis results
        """
        fairness_results = {}
        
        for feature in sensitive_features:
            if feature not in X_test.columns:
                logger.warning(f"Sensitive feature {feature} not found in data")
                continue
            
            # Get unique values for this sensitive feature
            unique_values = X_test[feature].unique()
            group_metrics = {}
            
            for value in unique_values:
                # Filter data for this group
                group_mask = X_test[feature] == value
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                group_y_pred_proba = y_pred_proba[group_mask]
                
                if len(group_y_true) > 0:
                    # Compute metrics for this group
                    group_metrics[str(value)] = {
                        'sample_size': len(group_y_true),
                        'accuracy': accuracy_score(group_y_true, group_y_pred),
                        'precision': precision_score(group_y_true, group_y_pred, 
                                                   average='binary', zero_division=0),
                        'recall': recall_score(group_y_true, group_y_pred, 
                                             average='binary', zero_division=0),
                        'f1_score': f1_score(group_y_true, group_y_pred, 
                                           average='binary', zero_division=0),
                        'roc_auc': roc_auc_score(group_y_true, group_y_pred_proba) 
                                  if len(np.unique(group_y_true)) > 1 else 0.0
                    }
            
            fairness_results[feature] = group_metrics
        
        return fairness_results
    
    def feature_importance_analysis(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X_test: Test features
            
        Returns:
            Feature importance analysis results
        """
        importance_results = {}
        
        # Model-based feature importance (if available)
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            feature_names = self.artifacts['feature_names']
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Sort features by importance
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            importance_results['model_based'] = {
                'features': [f[0] for f in feature_importance],
                'importances': [f[1] for f in feature_importance]
            }
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = feature_importance[:20]  # Top 20 features
            features, scores = zip(*top_features)
            
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            
            importance_plot_path = self.results_path / 'feature_importance.png'
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            importance_results['importance_plot_path'] = str(importance_plot_path)
        
        # SHAP analysis (if dataset is not too large)
        if len(X_test) <= 1000:  # Limit for computational efficiency
            try:
                # Initialize SHAP explainer
                if self.shap_explainer is None:
                    self.shap_explainer = shap.Explainer(self.model.predict, X_test[:100])
                
                # Compute SHAP values for a sample
                sample_size = min(100, len(X_test))
                shap_values = self.shap_explainer(X_test.iloc[:sample_size])
                
                # SHAP summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test.iloc[:sample_size], 
                                show=False, max_display=20)
                
                shap_plot_path = self.results_path / 'shap_summary.png'
                plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                importance_results['shap_plot_path'] = str(shap_plot_path)
                
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {str(e)}")
        
        return importance_results
    
    def threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyze optimal threshold for classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Threshold analysis results
        """
        # Calculate metrics for different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred_thresh),
                'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_thresh, zero_division=0),
                'specificity': self._compute_specificity(y_true, y_pred_thresh)
            }
            threshold_metrics.append(metrics)
        
        # Find optimal threshold based on F1 score
        best_f1_idx = max(range(len(threshold_metrics)), 
                         key=lambda i: threshold_metrics[i]['f1_score'])
        optimal_threshold = threshold_metrics[best_f1_idx]['threshold']
        
        # Plot threshold analysis
        df_thresh = pd.DataFrame(threshold_metrics)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(df_thresh['threshold'], df_thresh['accuracy'], label='Accuracy')
        plt.plot(df_thresh['threshold'], df_thresh['f1_score'], label='F1 Score')
        plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Accuracy and F1 Score vs Threshold')
        
        plt.subplot(2, 2, 2)
        plt.plot(df_thresh['threshold'], df_thresh['precision'], label='Precision')
        plt.plot(df_thresh['threshold'], df_thresh['recall'], label='Recall')
        plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Precision and Recall vs Threshold')
        
        plt.subplot(2, 2, 3)
        plt.plot(df_thresh['threshold'], df_thresh['specificity'], label='Specificity')
        plt.plot(df_thresh['threshold'], df_thresh['recall'], label='Sensitivity')
        plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Specificity and Sensitivity vs Threshold')
        
        plt.tight_layout()
        
        threshold_plot_path = self.results_path / 'threshold_analysis.png'
        plt.savefig(threshold_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'optimal_threshold': optimal_threshold,
            'threshold_metrics': threshold_metrics,
            'threshold_plot_path': str(threshold_plot_path)
        }
    
    def comprehensive_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series,
                               sensitive_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            sensitive_features: List of sensitive features for fairness analysis
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Initialize results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(X_test),
            'positive_class_ratio': np.mean(y_test)
        }
        
        # Basic metrics
        logger.info("Computing basic metrics...")
        evaluation_results['basic_metrics'] = self.compute_basic_metrics(
            y_test.values, y_pred, y_pred_proba
        )
        
        # Generate plots
        logger.info("Generating evaluation plots...")
        evaluation_results['roc_curve_path'] = self.plot_roc_curve(y_test.values, y_pred_proba)
        evaluation_results['pr_curve_path'] = self.plot_precision_recall_curve(y_test.values, y_pred_proba)
        evaluation_results['confusion_matrix_path'] = self.plot_confusion_matrix(y_test.values, y_pred)
        
        # Calibration analysis
        logger.info("Performing calibration analysis...")
        evaluation_results['calibration'] = self.calibration_analysis(y_test.values, y_pred_proba)
        
        # Feature importance analysis
        logger.info("Analyzing feature importance...")
        evaluation_results['feature_importance'] = self.feature_importance_analysis(X_test)
        
        # Threshold analysis
        logger.info("Performing threshold analysis...")
        evaluation_results['threshold_analysis'] = self.threshold_analysis(y_test.values, y_pred_proba)
        
        # Fairness analysis (if sensitive features provided)
        if sensitive_features:
            logger.info("Performing fairness analysis...")
            evaluation_results['fairness'] = self.fairness_analysis(
                X_test, y_test.values, y_pred, y_pred_proba, sensitive_features
            )
        
        # Classification report
        evaluation_results['classification_report'] = classification_report(
            y_test.values, y_pred, output_dict=True
        )
        
        # Save results
        results_file = self.results_path / 'evaluation_results.json'
        import json
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_numpy_types(evaluation_results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        
        return evaluation_results
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


def main():
    """Main function to run model evaluation."""
    # Configuration
    MODEL_PATH = "backend/ml_models/trained_models/random_forest_model.pkl"
    ARTIFACTS_PATH = "backend/ml_models/trained_models/training_artifacts.pkl"
    TEST_DATA_PATH = "data/processed/test_data.csv"
    RESULTS_PATH = "backend/ml_models/evaluation_results/"
    
    # Initialize evaluator
    evaluator = CVDModelEvaluator(MODEL_PATH, ARTIFACTS_PATH, RESULTS_PATH)
    
    # Load test data
    logger.info(f"Loading test data from {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Prepare test data (similar to training preprocessing)
    y_test = test_df['HeartDisease']  # Adjust column name as needed
    X_test = test_df.drop(columns=['HeartDisease'])
    
    # Define sensitive features for fairness analysis
    sensitive_features = ['Sex', 'AgeCategory', 'Race']  # Adjust as needed
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(X_test, y_test, sensitive_features)
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"Test Set Size: {results['dataset_size']}")
    print(f"Positive Class Ratio: {results['positive_class_ratio']:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"  Precision: {results['basic_metrics']['precision']:.4f}")
    print(f"  Recall: {results['basic_metrics']['recall']:.4f}")
    print(f"  F1 Score: {results['basic_metrics']['f1_score']:.4f}")
    print(f"  ROC AUC: {results['basic_metrics']['roc_auc']:.4f}")
    print(f"  Average Precision: {results['basic_metrics']['average_precision']:.4f}")
    print(f"\nOptimal Threshold: {results['threshold_analysis']['optimal_threshold']:.3f}")
    print(f"Brier Score: {results['calibration']['brier_score']:.4f}")


if __name__ == "__main__":
    main()