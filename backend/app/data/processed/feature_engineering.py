import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVDFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for cardiovascular disease prediction.
    
    This class handles feature creation, transformation, selection, and validation
    specifically designed for cardiovascular disease risk assessment data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config or {}
        
        # Initialize transformers
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        
        # Feature metadata
        self.feature_metadata = {}
        self.engineered_features = []
        self.selected_features = []
        self.transformation_pipeline = None
        
        # BMI categories mapping
        self.bmi_categories = {
            'underweight': (0, 18.5),
            'normal': (18.5, 24.9),
            'overweight': (25.0, 29.9),
            'obese_class1': (30.0, 34.9),
            'obese_class2': (35.0, 39.9),
            'obese_class3': (40.0, float('inf'))
        }
        
        # Age group mappings
        self.age_groups = {
            'young_adult': (18, 39),
            'middle_aged': (40, 64),
            'senior': (65, 79),
            'elderly': (80, float('inf'))
        }
        
        # Risk factor combinations
        self.risk_combinations = [
            ['Smoking', 'HighBP'],
            ['Diabetes', 'HighChol'],
            ['Smoking', 'Diabetes'],
            ['HighBP', 'HighChol'],
            ['PhysActivity', 'BMI'],
            ['Age', 'Sex'],
            ['Smoking', 'PhysActivity']
        ]
    
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load cardiovascular disease dataset.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Loaded DataFrame
        """
        data_path = Path(data_path)
        logger.info(f"Loading data from {data_path}")
        
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality and provide insights.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics
        """
        logger.info("Analyzing data quality...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'duplicates': df.duplicated().sum(),
            'outliers': {},
            'skewness': {},
            'correlations': {}
        }
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        quality_report['missing_data'] = {
            'counts': missing_data.to_dict(),
            'percentages': (missing_data / len(df) * 100).to_dict()
        }
        
        # Data types
        quality_report['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Analyze numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            # Outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            quality_report['outliers'][col] = outliers
            
            # Skewness
            quality_report['skewness'][col] = df[col].skew()
        
        # High correlations (>0.8)
        if len(numerical_columns) > 1:
            corr_matrix = df[numerical_columns].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            quality_report['high_correlations'] = high_corr_pairs
        
        return quality_report
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with basic engineered features
        """
        logger.info("Creating basic engineered features...")
        
        df_features = df.copy()
        
        # BMI-related features (if height and weight available or BMI directly)
        if 'BMI' in df.columns:
            # BMI categories
            df_features['BMI_Category'] = pd.cut(
                df['BMI'],
                bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']
            )
            
            # BMI risk level
            df_features['BMI_High_Risk'] = (df['BMI'] >= 30).astype(int)
            df_features['BMI_Very_High_Risk'] = (df['BMI'] >= 35).astype(int)
            
            # BMI squared (non-linear relationship)
            df_features['BMI_Squared'] = df['BMI'] ** 2
        
        # Age-related features
        if 'Age' in df.columns:
            # Age groups
            df_features['Age_Group'] = pd.cut(
                df['Age'],
                bins=[0, 39, 64, 79, float('inf')],
                labels=['Young_Adult', 'Middle_Aged', 'Senior', 'Elderly']
            )
            
            # Age risk levels
            df_features['Age_High_Risk'] = (df['Age'] >= 65).astype(int)
            df_features['Age_Very_High_Risk'] = (df['Age'] >= 75).astype(int)
            
            # Age squared
            df_features['Age_Squared'] = df['Age'] ** 2
        
        # Gender-Age interactions
        if 'Sex' in df.columns and 'Age' in df.columns:
            # Different risk profiles for men and women by age
            df_features['Male_Over_45'] = ((df['Sex'] == 'Male') & (df['Age'] > 45)).astype(int)
            df_features['Female_Over_55'] = ((df['Sex'] == 'Female') & (df['Age'] > 55)).astype(int)
        
        # Blood pressure features
        if 'HighBP' in df.columns:
            df_features['BP_Risk'] = df['HighBP'].astype(int)
        
        # Cholesterol features
        if 'HighChol' in df.columns:
            df_features['Cholesterol_Risk'] = df['HighChol'].astype(int)
        
        # Lifestyle risk score
        lifestyle_features = ['Smoking', 'PhysActivity', 'HvyAlcoholConsump']
        available_lifestyle = [f for f in lifestyle_features if f in df.columns]
        
        if available_lifestyle:
            # Create lifestyle risk score (higher = worse lifestyle)
            lifestyle_risk = 0
            if 'Smoking' in df.columns:
                lifestyle_risk += df['Smoking'].astype(int)
            if 'PhysActivity' in df.columns:
                lifestyle_risk += (1 - df['PhysActivity']).astype(int)  # No physical activity = risk
            if 'HvyAlcoholConsump' in df.columns:
                lifestyle_risk += df['HvyAlcoholConsump'].astype(int)
                
            df_features['Lifestyle_Risk_Score'] = lifestyle_risk
        
        # Medical conditions risk score
        medical_conditions = ['Diabetes', 'Stroke', 'HighBP', 'HighChol']
        available_medical = [f for f in medical_conditions if f in df.columns]
        
        if available_medical:
            medical_risk = sum(df[col].astype(int) for col in available_medical)
            df_features['Medical_Risk_Score'] = medical_risk
            df_features['Multiple_Conditions'] = (medical_risk >= 2).astype(int)
        
        # Physical health composite
        if 'PhysHlth' in df.columns:
            df_features['Poor_Physical_Health'] = (df['PhysHlth'] > 14).astype(int)  # Poor health >2 weeks/month
            df_features['Physical_Health_Category'] = pd.cut(
                df['PhysHlth'],
                bins=[-1, 0, 7, 14, float('inf')],
                labels=['Excellent', 'Good', 'Fair', 'Poor']
            )
        
        # Mental health composite
        if 'MentHlth' in df.columns:
            df_features['Poor_Mental_Health'] = (df['MentHlth'] > 14).astype(int)
            df_features['Mental_Health_Category'] = pd.cut(
                df['MentHlth'],
                bins=[-1, 0, 7, 14, float('inf')],
                labels=['Excellent', 'Good', 'Fair', 'Poor']
            )
        
        # Healthcare access
        if 'NoDocbcCost' in df.columns:
            df_features['Healthcare_Access_Risk'] = df['NoDocbcCost'].astype(int)
        
        self.engineered_features.extend([col for col in df_features.columns if col not in df.columns])
        logger.info(f"Created {len(self.engineered_features)} basic features")
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between risk factors.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_interactions = df.copy()
        
        # Risk factor combinations
        for combo in self.risk_combinations:
            available_features = [f for f in combo if f in df.columns]
            
            if len(available_features) == 2:
                feature_name = f"{available_features[0]}_X_{available_features[1]}"
                df_interactions[feature_name] = (
                    df[available_features[0]].astype(int) * 
                    df[available_features[1]].astype(int)
                )
        
        # Age interactions with various risk factors
        if 'Age' in df.columns:
            age_interaction_features = ['BMI', 'HighBP', 'HighChol', 'Diabetes', 'Smoking']
            for feature in age_interaction_features:
                if feature in df.columns:
                    if feature == 'BMI':
                        # Continuous interaction
                        df_interactions[f'Age_X_BMI'] = df['Age'] * df['BMI']
                    else:
                        # Binary interaction
                        df_interactions[f'Age_X_{feature}'] = df['Age'] * df[feature].astype(int)
        
        # BMI interactions
        if 'BMI' in df.columns:
            bmi_interaction_features = ['HighBP', 'Diabetes', 'PhysActivity']
            for feature in bmi_interaction_features:
                if feature in df.columns:
                    df_interactions[f'BMI_X_{feature}'] = df['BMI'] * df[feature].astype(int)
        
        # Triple interactions (high-risk combinations)
        triple_interactions = [
            ['Age', 'Smoking', 'HighBP'],
            ['BMI', 'Diabetes', 'HighChol'],
            ['Age', 'Sex', 'HighBP']
        ]
        
        for triple in triple_interactions:
            available_triple = [f for f in triple if f in df.columns]
            if len(available_triple) == 3:
                feature_name = f"{available_triple[0]}_X_{available_triple[1]}_X_{available_triple[2]}"
                if available_triple[0] in ['Age', 'BMI']:  # Continuous features
                    df_interactions[feature_name] = (
                        df[available_triple[0]] * 
                        df[available_triple[1]].astype(int) * 
                        df[available_triple[2]].astype(int)
                    )
                else:  # All binary
                    df_interactions[feature_name] = (
                        df[available_triple[0]].astype(int) * 
                        df[available_triple[1]].astype(int) * 
                        df[available_triple[2]].astype(int)
                    )
        
        new_interactions = [col for col in df_interactions.columns if col not in df.columns]
        logger.info(f"Created {len(new_interactions)} interaction features")
        
        return df_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for continuous variables.
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features (degree={degree})...")
        
        df_poly = df.copy()
        
        # Continuous features that might benefit from polynomial transformation
        continuous_features = []
        
        # Check for continuous features
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10:
                continuous_features.append(col)
        
        # Common continuous features in cardiovascular data
        target_features = ['Age', 'BMI', 'PhysHlth', 'MentHlth']
        continuous_features = [f for f in target_features if f in continuous_features]
        
        for feature in continuous_features:
            for deg in range(2, degree + 1):
                poly_feature_name = f"{feature}_Poly_{deg}"
                df_poly[poly_feature_name] = df[feature] ** deg
        
        # Square root transformations (for right-skewed data)
        for feature in continuous_features:
            if (df[feature] >= 0).all():  # Only for non-negative values
                df_poly[f"{feature}_Sqrt"] = np.sqrt(df[feature])
        
        # Log transformations (for highly skewed data)
        for feature in continuous_features:
            if (df[feature] > 0).all():  # Only for positive values
                df_poly[f"{feature}_Log"] = np.log(df[feature])
        
        poly_features = [col for col in df_poly.columns if col not in df.columns]
        logger.info(f"Created {len(poly_features)} polynomial features")
        
        return df_poly
    
    def create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned versions of continuous features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with binned features
        """
        logger.info("Creating binned features...")
        
        df_binned = df.copy()
        
        # Age binning (more granular)
        if 'Age' in df.columns:
            df_binned['Age_Decile'] = pd.qcut(df['Age'], q=10, labels=False, duplicates='drop')
            df_binned['Age_Quintile'] = pd.qcut(df['Age'], q=5, labels=False, duplicates='drop')
        
        # BMI binning
        if 'BMI' in df.columns:
            df_binned['BMI_Decile'] = pd.qcut(df['BMI'], q=10, labels=False, duplicates='drop')
            df_binned['BMI_Quintile'] = pd.qcut(df['BMI'], q=5, labels=False, duplicates='drop')
        
        # Health days binning
        health_features = ['PhysHlth', 'MentHlth']
        for feature in health_features:
            if feature in df.columns:
                # Custom bins for health days (0, 1-7, 8-14, 15-30)
                df_binned[f'{feature}_Binned'] = pd.cut(
                    df[feature],
                    bins=[-1, 0, 7, 14, 30],
                    labels=['None', 'Few', 'Some', 'Many']
                )
        
        binned_features = [col for col in df_binned.columns if col not in df.columns]
        logger.info(f"Created {len(binned_features)} binned features")
        
        return df_binned
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values using various strategies.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'smart')
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Handling missing values using {strategy} strategy...")
        
        df_imputed = df.copy()
        
        if strategy == 'smart':
            # Smart imputation based on data type and distribution
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        # For numerical features
                        if df[col].skew() > 1:  # Highly skewed - use median
                            imputer = SimpleImputer(strategy='median')
                        else:  # Normal distribution - use mean
                            imputer = SimpleImputer(strategy='mean')
                    else:
                        # For categorical features - use most frequent
                        imputer = SimpleImputer(strategy='most_frequent')
                    
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                    self.imputers[col] = imputer
        
        elif strategy == 'knn':
            # KNN imputation for all features
            numerical_features = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_features) > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                df_imputed[numerical_features] = knn_imputer.fit_transform(df_imputed[numerical_features])
                self.imputers['knn_numerical'] = knn_imputer
            
            # Handle categorical separately
            categorical_features = df.select_dtypes(include=['object']).columns
            for col in categorical_features:
                if df[col].isnull().sum() > 0:
                    mode_imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed[col] = mode_imputer.fit_transform(df_imputed[[col]]).ravel()
                    self.imputers[col] = mode_imputer
        
        else:
            # Simple strategy for all features
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        imputer = SimpleImputer(strategy=strategy if strategy in ['mean', 'median'] else 'median')
                    else:
                        imputer = SimpleImputer(strategy='most_frequent')
                    
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                    self.imputers[col] = imputer
        
        missing_after = df_imputed.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")
        
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using appropriate methods.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            unique_values = df[col].nunique()
            
            if unique_values == 2:
                # Binary encoding for binary categories
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
            
            elif unique_values <= 5:
                # One-hot encoding for low cardinality
                df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=False)
            
            else:
                # Label encoding for high cardinality (with frequency-based ordering)
                value_counts = df[col].value_counts()
                df_encoded[col] = df_encoded[col].map(value_counts)  # Map to frequencies first
                
                # Then normalize
                scaler = MinMaxScaler()
                df_encoded[col] = scaler.fit_transform(df_encoded[[col]])
                self.encoders[col] = scaler
        
        encoded_features = [col for col in df_encoded.columns if col not in df.columns]
        logger.info(f"Created {len(encoded_features)} encoded features")
        
        return df_encoded
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'rfe', k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features using various methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('univariate', 'rfe', 'model_based', 'combined')
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting features using {method} method...")
        
        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selectors['univariate'] = selector
        
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selectors['rfe'] = selector
        
        elif method == 'model_based':
            # Model-based feature selection
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator=estimator, max_features=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selectors['model_based'] = selector
        
        elif method == 'combined':
            # Combined approach: multiple methods with voting
            selectors = {
                'univariate': SelectKBest(score_func=f_classif, k=k),
                'rfe': RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=k),
                'model_based': SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=k)
            }
            
            feature_scores = {}
            
            for name, selector in selectors.items():
                selector.fit(X, y)
                selected_mask = selector.get_support()
                
                # Score features (1 if selected, 0 if not)
                for i, feature in enumerate(X.columns):
                    if feature not in feature_scores:
                        feature_scores[feature] = 0
                    feature_scores[feature] += int(selected_mask[i])
                
                self.feature_selectors[name] = selector
            
            # Select features that were chosen by at least 2 methods
            selected_features = [feature for feature, score in feature_scores.items() if score >= 2]
            
            # If we don't have enough features, add the highest scoring ones
            if len(selected_features) < k:
                remaining_features = [(feature, score) for feature, score in feature_scores.items() 
                                    if feature not in selected_features]
                remaining_features.sort(key=lambda x: x[1], reverse=True)
                
                needed_features = k - len(selected_features)
                selected_features.extend([feature for feature, _ in remaining_features[:needed_features]])
            
            # If we have too many, keep the top k by score
            elif len(selected_features) > k:
                feature_score_pairs = [(feature, feature_scores[feature]) for feature in selected_features]
                feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
                selected_features = [feature for feature, _ in feature_score_pairs[:k]]
            
            X_selected = X[selected_features]
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            X: Feature matrix
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile')
            
        Returns:
            Scaled feature matrix
        """
        logger.info(f"Scaling features using {method} method...")
        
        X_scaled = X.copy()
        numerical_features = X.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if len(numerical_features) > 0:
            X_scaled[numerical_features] = scaler.fit_transform(X_scaled[numerical_features])
            self.scalers[method] = scaler
        
        return X_scaled
    
    def create_feature_pipeline(self, df: pd.DataFrame, target_column: str) -> Pipeline:
        """
        Create a complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Complete preprocessing pipeline
        """
        logger.info("Creating complete feature engineering pipeline...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Identify column types
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing steps for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessing steps for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create complete pipeline
        complete_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_classif, k=50))
        ])
        
        self.transformation_pipeline = complete_pipeline
        
        return complete_pipeline
    
    def run_feature_engineering_pipeline(self, data_path: Union[str, Path], 
                                       target_column: str = 'HeartDisease',
                                       output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Execute the complete feature engineering pipeline.
        
        Args:
            data_path: Path to input data
            target_column: Name of target column
            output_path: Path to save processed data
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting complete feature engineering pipeline...")
        
        pipeline_results = {
            'start_time': datetime.utcnow(),
            'input_path': str(data_path),
            'target_column': target_column,
            'steps_completed': [],
            'feature_counts': {},
            'quality_metrics': {}
        }
        
        try:
            # Load data
            df = self.load_data(data_path)
            pipeline_results['original_shape'] = df.shape
            pipeline_results['steps_completed'].append('data_loading')
            
            # Analyze data quality
            quality_report = self.analyze_data_quality(df)
            pipeline_results['quality_metrics'] = quality_report
            pipeline_results['steps_completed'].append('quality_analysis')
            
            # Handle missing values
            df = self.handle_missing_values(df, strategy='smart')
            pipeline_results['feature_counts']['after_imputation'] = df.shape[1]
            pipeline_results['steps_completed'].append('missing_value_handling')
            
            # Create basic features
            df = self.create_basic_features(df)
            pipeline_results['feature_counts']['after_basic_features'] = df.shape[1]
            pipeline_results['steps_completed'].append('basic_feature_creation')
            
            # Create interaction features
            df = self.create_interaction_features(df)
            pipeline_results['feature_counts']['after_interactions'] = df.shape[1]
            pipeline_results['steps_completed'].append('interaction_features')
            
            # Create polynomial features
            df = self.create_polynomial_features(df, degree=2)
            pipeline_results['feature_counts']['after_polynomial'] = df.shape[1]
            pipeline_results['steps_completed'].append('polynomial_features')
            
            # Create binning features
            df = self.create_binning_features(df)
            pipeline_results['feature_counts']['after_binning'] = df.shape[1]
            pipeline_results['steps_completed'].append('binning_features')
            
            # Encode categorical features
            df = self.encode_categorical_features(df)
            pipeline_results['feature_counts']['after_encoding'] = df.shape[1]
            pipeline_results['steps_completed'].append('categorical_encoding')
            
            # Feature selection
            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                X_selected, selected_features = self.select_features(X, y, method='combined', k=50)
                
                # Combine selected features with target
                df_final = X_selected.copy()
                df_final[target_column] = y
                
                pipeline_results['feature_counts']['after_selection'] = len(selected_features)
                pipeline_results['selected_features'] = selected_features
                pipeline_results['steps_completed'].append('feature_selection')
                
                # Scale features
                X_scaled = self.scale_features(X_selected, method='standard')
                df_final[X_scaled.columns] = X_scaled
                
                pipeline_results['steps_completed'].append('feature_scaling')
            else:
                df_final = df
                logger.warning(f"Target column '{target_column}' not found. Skipping feature selection.")
            
            # Save processed data
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save as CSV
                df_final.to_csv(output_path, index=False)
                
                # Save as Parquet for better performance
                parquet_path = output_path.with_suffix('.parquet')
                df_final.to_parquet(parquet_path, index=False)
                
                # Save feature engineering artifacts
                artifacts = {
                    'scalers': self.scalers,
                    'encoders': self.encoders,
                    'imputers': self.imputers,
                    'feature_selectors': self.feature_selectors,
                    'selected_features': self.selected_features,
                    'engineered_features': self.engineered_features,
                    'transformation_pipeline': self.transformation_pipeline
                }
                
                artifacts_path = output_path.parent / 'feature_engineering_artifacts.pkl'
                with open(artifacts_path, 'wb') as f:
                    pickle.dump(artifacts, f)
                
                pipeline_results['output_path'] = str(output_path)
                pipeline_results['artifacts_path'] = str(artifacts_path)
            
            pipeline_results['final_shape'] = df_final.shape
            pipeline_results['success'] = True
            pipeline_results['end_time'] = datetime.utcnow()
            pipeline_results['processing_time'] = (pipeline_results['end_time'] - pipeline_results['start_time']).total_seconds()
            
            logger.info("Feature engineering pipeline completed successfully!")
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.utcnow()
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            
            return pipeline_results


def main():
    """Main function to run feature engineering pipeline."""
    # Configuration
    INPUT_DATA_PATH = "data/processed/cardiovascular_disease_data_latest.csv"
    OUTPUT_DATA_PATH = "data/processed/cardiovascular_features_engineered.csv"
    TARGET_COLUMN = "HeartDisease"
    
    # Initialize feature engineer
    feature_engineer = CVDFeatureEngineer()
    
    # Run pipeline
    results = feature_engineer.run_feature_engineering_pipeline(
        data_path=INPUT_DATA_PATH,
        target_column=TARGET_COLUMN,
        output_path=OUTPUT_DATA_PATH
    )
    
    # Print results summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE SUMMARY")
    print("="*60)
    print(f"Success: {results['success']}")
    print(f"Processing Time: {results.get('processing_time', 0):.2f} seconds")
    
    if results['success']:
        print(f"Original Shape: {results['original_shape']}")
        print(f"Final Shape: {results['final_shape']}")
        print(f"\nFeature Counts by Step:")
        for step, count in results['feature_counts'].items():
            print(f"  {step}: {count} features")
        
        print(f"\nSteps Completed: {len(results['steps_completed'])}")
        for step in results['steps_completed']:
            print(f"  ✓ {step}")
        
        if results.get('selected_features'):
            print(f"\nTop 10 Selected Features:")
            for i, feature in enumerate(results['selected_features'][:10]):
                print(f"  {i+1}. {feature}")
        
        print(f"\nOutput saved to: {results.get('output_path', 'Not saved')}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()