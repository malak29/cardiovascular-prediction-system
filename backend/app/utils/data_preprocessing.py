import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import IsolationForest
import structlog

from app.schemas.prediction import ValidationResultSchema
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# Suppress pandas warnings for cleaner logs
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class DataValidator:
    """Comprehensive data validation for cardiovascular prediction datasets"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Required columns for cardiovascular prediction
        self.required_columns = [
            "age",
            "gender", 
            "Data_Value"  # Target variable
        ]
        
        # Expected columns with their data types
        self.expected_schema = {
            "age": "numeric",
            "gender": "categorical",
            "Data_Value": "numeric",
            "has_hypertension": "boolean",
            "has_diabetes": "boolean", 
            "has_heart_disease": "boolean",
            "systolic_bp": "numeric",
            "diastolic_bp": "numeric",
            "total_cholesterol": "numeric",
            "hdl_cholesterol": "numeric",
            "bmi": "numeric",
            "smoking_status": "categorical"
        }
        
        # Valid value ranges
        self.value_ranges = {
            "age": (18, 120),
            "systolic_bp": (50, 300),
            "diastolic_bp": (30, 200),
            "total_cholesterol": (50, 500),
            "hdl_cholesterol": (10, 150),
            "bmi": (10, 100),
            "Data_Value": (0, 10000)  # Hospitalization rate per 100,000
        }
        
        # Valid categorical values
        self.valid_categories = {
            "gender": ["male", "female", "other", "unknown"],
            "smoking_status": ["never", "former", "current", "unknown"],
            "race_ethnicity": [
                "white_non_hispanic", "black_non_hispanic", "hispanic",
                "asian_non_hispanic", "other", "unknown"
            ]
        }
    
    async def validate_patient_data(self, patient_data: Dict[str, Any]) -> ValidationResultSchema:
        """Validate individual patient data for prediction"""
        try:
            logger.debug("Validating patient data")
            
            errors = []
            warnings = []
            field_validation = {}
            
            # Check required fields
            missing_required = []
            for field in ["age", "gender"]:  # Minimum required for prediction
                if field not in patient_data or patient_data[field] is None:
                    missing_required.append(field)
            
            if missing_required:
                errors.append({
                    "field": "required_fields",
                    "message": f"Missing required fields: {', '.join(missing_required)}"
                })
            
            # Validate individual fields
            for field, value in patient_data.items():
                field_result = await self._validate_field(field, value)
                field_validation[field] = field_result
                
                if field_result.get("errors"):
                    errors.extend([{"field": field, "message": err} for err in field_result["errors"]])
                
                if field_result.get("warnings"):
                    warnings.extend([{"field": field, "message": warn} for warn in field_result["warnings"]])
            
            # Calculate quality score
            total_fields = len(self.expected_schema)
            valid_fields = len([f for f in field_validation.values() if f.get("valid", False)])
            completeness = len([v for v in patient_data.values() if v is not None]) / len(patient_data)
            
            quality_score = (valid_fields / total_fields) * 0.7 + completeness * 0.3
            
            # Generate recommendations
            recommendations = await self._generate_data_recommendations(errors, warnings, patient_data)
            
            validation_result = ValidationResultSchema(
                is_valid=len(errors) == 0,
                quality_score=round(quality_score, 3),
                errors=errors,
                warnings=warnings,
                field_validation=field_validation,
                required_fields=self.required_columns,
                missing_fields=missing_required,
                total_records=1,
                valid_records=1 if len(errors) == 0 else 0,
                invalid_records=1 if len(errors) > 0 else 0,
                recommendations=recommendations
            )
            
            logger.debug("Patient data validation completed", is_valid=validation_result.is_valid)
            return validation_result
            
        except Exception as e:
            logger.error("Patient data validation failed", error=str(e), exc_info=True)
            raise
    
    async def validate_dataset_structure(
        self, 
        df: pd.DataFrame, 
        dataset_type: str = "training"
    ) -> ValidationResultSchema:
        """Validate dataset structure and content"""
        try:
            logger.info("Validating dataset structure", rows=len(df), columns=len(df.columns))
            
            errors = []
            warnings = []
            field_validation = {}
            
            # Check minimum requirements
            if len(df) < 100:
                errors.append({
                    "field": "dataset_size",
                    "message": f"Dataset too small: {len(df)} rows (minimum 100 required)"
                })
            
            # Check required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                errors.append({
                    "field": "required_columns",
                    "message": f"Missing required columns: {', '.join(missing_columns)}"
                })
            
            # Validate each column
            for column in df.columns:
                if column in self.expected_schema:
                    field_result = await self._validate_column(df, column)
                    field_validation[column] = field_result
                    
                    if field_result.get("errors"):
                        errors.extend([{"field": column, "message": err} for err in field_result["errors"]])
                    
                    if field_result.get("warnings"):
                        warnings.extend([{"field": column, "message": warn} for warn in field_result["warnings"]])
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                warnings.append({
                    "field": "duplicates",
                    "message": f"Found {duplicate_count} duplicate rows"
                })
            
            # Calculate overall quality metrics
            quality_metrics = await self._calculate_dataset_quality(df)
            
            # Generate recommendations
            recommendations = await self._generate_dataset_recommendations(
                df, errors, warnings, dataset_type
            )
            
            validation_result = ValidationResultSchema(
                is_valid=len(errors) == 0,
                quality_score=quality_metrics["overall_score"],
                errors=errors,
                warnings=warnings,
                field_validation=field_validation,
                required_fields=self.required_columns,
                missing_fields=missing_columns,
                total_records=len(df),
                valid_records=quality_metrics["valid_records"],
                invalid_records=quality_metrics["invalid_records"],
                recommendations=recommendations
            )
            
            logger.info(
                "Dataset validation completed",
                is_valid=validation_result.is_valid,
                quality_score=validation_result.quality_score
            )
            
            return validation_result
            
        except Exception as e:
            logger.error("Dataset validation failed", error=str(e), exc_info=True)
            raise
    
    async def quick_validate(self, df: pd.DataFrame) -> ValidationResultSchema:
        """Quick validation focusing on critical issues only"""
        try:
            errors = []
            warnings = []
            
            # Check basic requirements
            if len(df) == 0:
                errors.append({"field": "dataset", "message": "Dataset is empty"})
            
            # Check for required columns
            missing_required = [col for col in self.required_columns if col not in df.columns]
            if missing_required:
                errors.append({
                    "field": "columns",
                    "message": f"Missing required columns: {', '.join(missing_required)}"
                })
            
            # Quick data type check
            numeric_columns = ["age", "Data_Value", "systolic_bp", "diastolic_bp"]
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    warnings.append({
                        "field": col,
                        "message": f"Column should be numeric but found {df[col].dtype}"
                    })
            
            # Calculate basic quality score
            completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            quality_score = 0.8 if len(errors) == 0 else 0.3
            quality_score = min(quality_score, completeness)
            
            return ValidationResultSchema(
                is_valid=len(errors) == 0,
                quality_score=round(quality_score, 3),
                errors=errors,
                warnings=warnings,
                field_validation={},
                required_fields=self.required_columns,
                missing_fields=missing_required,
                total_records=len(df),
                valid_records=len(df) if len(errors) == 0 else 0,
                invalid_records=len(df) if len(errors) > 0 else 0,
                recommendations=["Run comprehensive validation for detailed analysis"]
            )
            
        except Exception as e:
            logger.error("Quick validation failed", error=str(e))
            raise
    
    async def validate_schema_only(self, df: pd.DataFrame) -> ValidationResultSchema:
        """Validate only the schema/structure of the dataset"""
        try:
            errors = []
            warnings = []
            field_validation = {}
            
            # Check column presence
            for expected_col, expected_type in self.expected_schema.items():
                if expected_col in df.columns:
                    actual_dtype = str(df[expected_col].dtype)
                    field_validation[expected_col] = {
                        "present": True,
                        "expected_type": expected_type,
                        "actual_type": actual_dtype,
                        "valid": True
                    }
                else:
                    field_validation[expected_col] = {
                        "present": False,
                        "expected_type": expected_type,
                        "valid": False
                    }
                    
                    if expected_col in self.required_columns:
                        errors.append({
                            "field": expected_col,
                            "message": f"Required column missing"
                        })
            
            # Check for unexpected columns
            unexpected_columns = [col for col in df.columns if col not in self.expected_schema]
            if unexpected_columns:
                warnings.append({
                    "field": "schema",
                    "message": f"Unexpected columns found: {', '.join(unexpected_columns[:5])}"
                })
            
            return ValidationResultSchema(
                is_valid=len(errors) == 0,
                quality_score=1.0 if len(errors) == 0 else 0.5,
                errors=errors,
                warnings=warnings,
                field_validation=field_validation,
                required_fields=self.required_columns,
                missing_fields=[col for col in self.required_columns if col not in df.columns],
                total_records=len(df),
                valid_records=len(df),
                invalid_records=0,
                recommendations=["Schema validation completed"]
            )
            
        except Exception as e:
            logger.error("Schema validation failed", error=str(e))
            raise
    
    async def comprehensive_validate(self, df: pd.DataFrame) -> ValidationResultSchema:
        """Comprehensive validation including data quality analysis"""
        try:
            logger.info("Starting comprehensive validation", rows=len(df))
            
            # Start with basic structure validation
            basic_validation = await self.validate_dataset_structure(df)
            
            if not basic_validation.is_valid:
                return basic_validation
            
            # Additional comprehensive checks
            errors = list(basic_validation.errors)
            warnings = list(basic_validation.warnings)
            field_validation = dict(basic_validation.field_validation)
            
            # Data quality analysis
            quality_issues = await self._analyze_data_quality(df)
            errors.extend(quality_issues["errors"])
            warnings.extend(quality_issues["warnings"])
            
            # Outlier detection
            outlier_analysis = await self._detect_outliers(df)
            warnings.extend(outlier_analysis["warnings"])
            
            # Consistency checks
            consistency_issues = await self._check_data_consistency(df)
            warnings.extend(consistency_issues)
            
            # Statistical validation
            statistical_issues = await self._validate_statistical_properties(df)
            warnings.extend(statistical_issues)
            
            # Update quality score based on comprehensive analysis
            quality_metrics = await self._calculate_comprehensive_quality_score(df, errors, warnings)
            
            # Generate detailed recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                df, errors, warnings, quality_metrics
            )
            
            comprehensive_result = ValidationResultSchema(
                is_valid=len(errors) == 0,
                quality_score=quality_metrics["overall_score"],
                errors=errors,
                warnings=warnings,
                field_validation=field_validation,
                required_fields=self.required_columns,
                missing_fields=basic_validation.missing_fields,
                total_records=len(df),
                valid_records=quality_metrics["valid_records"],
                invalid_records=quality_metrics["invalid_records"],
                recommendations=recommendations
            )
            
            logger.info(
                "Comprehensive validation completed",
                is_valid=comprehensive_result.is_valid,
                quality_score=comprehensive_result.quality_score,
                errors_count=len(errors),
                warnings_count=len(warnings)
            )
            
            return comprehensive_result
            
        except Exception as e:
            logger.error("Comprehensive validation failed", error=str(e), exc_info=True)
            raise
    
    async def validate_csv_structure(self, df: pd.DataFrame) -> ValidationResultSchema:
        """Validate CSV structure for upload predictions"""
        try:
            errors = []
            warnings = []
            
            # Check for minimum required columns for prediction
            prediction_required = ["age", "gender"]
            missing_prediction_cols = [col for col in prediction_required if col not in df.columns]
            
            if missing_prediction_cols:
                errors.append({
                    "field": "prediction_columns",
                    "message": f"Missing columns required for prediction: {', '.join(missing_prediction_cols)}"
                })
            
            # Check data types
            if "age" in df.columns:
                if not pd.api.types.is_numeric_dtype(df["age"]):
                    errors.append({"field": "age", "message": "Age column must be numeric"})
            
            # Check for reasonable data ranges
            if "age" in df.columns:
                age_issues = df[(df["age"] < 18) | (df["age"] > 120)]
                if len(age_issues) > 0:
                    warnings.append({
                        "field": "age",
                        "message": f"Found {len(age_issues)} age values outside reasonable range (18-120)"
                    })
            
            quality_score = 1.0 if len(errors) == 0 else 0.7 if len(errors) < 3 else 0.3
            
            return ValidationResultSchema(
                is_valid=len(errors) == 0,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                field_validation={},
                required_fields=prediction_required,
                missing_fields=missing_prediction_cols,
                total_records=len(df),
                valid_records=len(df) if len(errors) == 0 else 0,
                invalid_records=len(df) if len(errors) > 0 else 0,
                recommendations=["Upload validation completed"]
            )
            
        except Exception as e:
            logger.error("CSV structure validation failed", error=str(e))
            raise
    
    # Private validation methods
    async def _validate_field(self, field_name: str, value: Any) -> Dict[str, Any]:
        """Validate individual field value"""
        try:
            result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            if value is None:
                result["warnings"].append(f"Missing value for {field_name}")
                return result
            
            # Type-specific validation
            if field_name in self.value_ranges:
                if not isinstance(value, (int, float)):
                    result["errors"].append(f"Expected numeric value, got {type(value).__name__}")
                    result["valid"] = False
                else:
                    min_val, max_val = self.value_ranges[field_name]
                    if value < min_val or value > max_val:
                        result["warnings"].append(f"Value {value} outside expected range [{min_val}, {max_val}]")
            
            # Categorical validation
            if field_name in self.valid_categories:
                if str(value).lower() not in [cat.lower() for cat in self.valid_categories[field_name]]:
                    result["warnings"].append(f"Unexpected category value: {value}")
            
            # Special validations
            if field_name == "age" and isinstance(value, (int, float)):
                if value < 0:
                    result["errors"].append("Age cannot be negative")
                    result["valid"] = False
                elif value > 150:
                    result["warnings"].append("Age seems unusually high")
            
            return result
            
        except Exception as e:
            logger.error(f"Field validation failed for {field_name}", error=str(e))
            return {"valid": False, "errors": [str(e)], "warnings": []}
    
    async def _validate_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Validate entire column in dataset"""
        try:
            result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "completeness": 1 - (df[column].isnull().sum() / len(df)),
                "unique_values": df[column].nunique(),
                "data_type": str(df[column].dtype)
            }
            
            # Check completeness
            missing_pct = df[column].isnull().sum() / len(df)
            if missing_pct > 0.5:
                result["errors"].append(f"Column has {missing_pct:.1%} missing values")
                result["valid"] = False
            elif missing_pct > 0.2:
                result["warnings"].append(f"Column has {missing_pct:.1%} missing values")
            
            # Validate numeric columns
            if column in self.value_ranges and pd.api.types.is_numeric_dtype(df[column]):
                min_val, max_val = self.value_ranges[column]
                outliers = df[(df[column] < min_val) | (df[column] > max_val)]
                
                if len(outliers) > 0:
                    outlier_pct = len(outliers) / len(df)
                    if outlier_pct > 0.1:
                        result["errors"].append(f"Column has {outlier_pct:.1%} values outside valid range")
                        result["valid"] = False
                    else:
                        result["warnings"].append(f"Column has {len(outliers)} outlier values")
            
            # Validate categorical columns
            if column in self.valid_categories:
                invalid_categories = []
                valid_cats = [cat.lower() for cat in self.valid_categories[column]]
                
                for val in df[column].dropna().unique():
                    if str(val).lower() not in valid_cats:
                        invalid_categories.append(val)
                
                if invalid_categories:
                    result["warnings"].append(f"Invalid categories found: {invalid_categories[:5]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Column validation failed for {column}", error=str(e))
            return {"valid": False, "errors": [str(e)], "warnings": []}
    
    async def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
        """Analyze overall data quality issues"""
        try:
            errors = []
            warnings = []
            
            # Check for empty DataFrame
            if df.empty:
                errors.append({"field": "dataset", "message": "Dataset is empty"})
                return {"errors": errors, "warnings": warnings}
            
            # Check for all-null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                warnings.append({
                    "field": "null_columns",
                    "message": f"Columns with all null values: {', '.join(null_columns)}"
                })
            
            # Check for single-value columns (no variance)
            constant_columns = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                warnings.append({
                    "field": "constant_columns",
                    "message": f"Columns with no variance: {', '.join(constant_columns)}"
                })
            
            # Check data types consistency
            for col in df.columns:
                if col in self.expected_schema:
                    expected_type = self.expected_schema[col]
                    actual_dtype = df[col].dtype
                    
                    if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(actual_dtype):
                        errors.append({
                            "field": col,
                            "message": f"Expected numeric type, found {actual_dtype}"
                        })
            
            return {"errors": errors, "warnings": warnings}
            
        except Exception as e:
            logger.error("Data quality analysis failed", error=str(e))
            return {"errors": [{"field": "analysis", "message": str(e)}], "warnings": []}
    
    async def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
        """Detect outliers in numeric columns"""
        try:
            warnings = []
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in df.columns and not df[col].isnull().all():
                    # Use IQR method for outlier detection
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    if len(outliers) > 0:
                        outlier_pct = len(outliers) / len(df)
                        if outlier_pct > 0.05:  # More than 5% outliers
                            warnings.append({
                                "field": col,
                                "message": f"High number of outliers detected: {len(outliers)} ({outlier_pct:.1%})"
                            })
            
            return {"warnings": warnings}
            
        except Exception as e:
            logger.error("Outlier detection failed", error=str(e))
            return {"warnings": [{"field": "outliers", "message": f"Outlier detection failed: {str(e)}"}]}
    
    async def _check_data_consistency(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Check for data consistency issues"""
        try:
            warnings = []
            
            # Check age consistency with Medicare eligibility
            if "age" in df.columns:
                young_medicare = df[df["age"] < 65]
                if len(young_medicare) > 0:
                    warnings.append({
                        "field": "age_medicare",
                        "message": f"Found {len(young_medicare)} Medicare patients under 65 (disability cases?)"
                    })
            
            # Check blood pressure consistency
            if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
                bp_issues = df[df["systolic_bp"] <= df["diastolic_bp"]]
                if len(bp_issues) > 0:
                    warnings.append({
                        "field": "blood_pressure",
                        "message": f"Found {len(bp_issues)} records where systolic ≤ diastolic BP"
                    })
            
            # Check hospitalization consistency
            if "cardiovascular_hospitalizations_last_year" in df.columns and "total_hospitalizations_last_year" in df.columns:
                inconsistent = df[df["cardiovascular_hospitalizations_last_year"] > df["total_hospitalizations_last_year"]]
                if len(inconsistent) > 0:
                    warnings.append({
                        "field": "hospitalizations",
                        "message": f"Found {len(inconsistent)} records where CV hospitalizations > total hospitalizations"
                    })
            
            return warnings
            
        except Exception as e:
            logger.error("Consistency check failed", error=str(e))
            return [{"field": "consistency", "message": f"Consistency check failed: {str(e)}"}]
    
    async def _validate_statistical_properties(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Validate statistical properties of the data"""
        try:
            warnings = []
            
            # Check for reasonable distributions
            if "Data_Value" in df.columns:
                target_stats = df["Data_Value"].describe()
                
                # Check for extreme skewness
                skewness = df["Data_Value"].skew()
                if abs(skewness) > 3:
                    warnings.append({
                        "field": "Data_Value",
                        "message": f"Target variable is highly skewed (skew={skewness:.2f})"
                    })
                
                # Check for zero variance
                if target_stats["std"] == 0:
                    warnings.append({
                        "field": "Data_Value",
                        "message": "Target variable has zero variance"
                    })
            
            return warnings
            
        except Exception as e:
            logger.error("Statistical validation failed", error=str(e))
            return [{"field": "statistics", "message": f"Statistical validation failed: {str(e)}"}]
    
    async def _calculate_dataset_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive dataset quality metrics"""
        try:
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells)
            
            # Count valid records (records with all required fields)
            valid_records = 0
            for _, row in df.iterrows():
                is_valid = all(
                    col in row and pd.notna(row[col])
                    for col in self.required_columns
                )
                if is_valid:
                    valid_records += 1
            
            invalid_records = len(df) - valid_records
            
            # Calculate overall quality score
            completeness_weight = 0.4
            validity_weight = 0.6
            
            validity_score = valid_records / len(df) if len(df) > 0 else 0
            overall_score = (completeness * completeness_weight) + (validity_score * validity_weight)
            
            return {
                "overall_score": round(overall_score, 3),
                "completeness": round(completeness, 3),
                "validity_score": round(validity_score, 3),
                "valid_records": valid_records,
                "invalid_records": invalid_records,
                "missing_cells": missing_cells,
                "total_cells": total_cells
            }
            
        except Exception as e:
            logger.error("Quality calculation failed", error=str(e))
            return {
                "overall_score": 0.0,
                "valid_records": 0,
                "invalid_records": len(df)
            }
    
    async def _calculate_comprehensive_quality_score(
        self,
        df: pd.DataFrame,
        errors: List[Dict[str, str]],
        warnings: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality score considering all validation results"""
        try:
            base_quality = await self._calculate_dataset_quality(df)
            
            # Penalty for errors and warnings
            error_penalty = len(errors) * 0.1
            warning_penalty = len(warnings) * 0.02
            
            adjusted_score = max(0.0, base_quality["overall_score"] - error_penalty - warning_penalty)
            
            return {
                **base_quality,
                "overall_score": round(adjusted_score, 3),
                "error_penalty": round(error_penalty, 3),
                "warning_penalty": round(warning_penalty, 3)
            }
            
        except Exception as e:
            logger.error("Comprehensive quality calculation failed", error=str(e))
            return {"overall_score": 0.0, "valid_records": 0, "invalid_records": len(df)}
    
    async def _generate_data_recommendations(
        self,
        errors: List[Dict[str, str]],
        warnings: List[Dict[str, str]],
        patient_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving patient data quality"""
        recommendations = []
        
        try:
            if errors:
                recommendations.append("Address data validation errors before making predictions")
            
            if warnings:
                recommendations.append("Review data warnings to improve prediction accuracy")
            
            # Specific recommendations based on missing data
            if "systolic_bp" not in patient_data or patient_data.get("systolic_bp") is None:
                recommendations.append("Include blood pressure measurements for better prediction accuracy")
            
            if "total_cholesterol" not in patient_data or patient_data.get("total_cholesterol") is None:
                recommendations.append("Include cholesterol levels for comprehensive risk assessment")
            
            if not recommendations:
                recommendations.append("Data quality is good - proceed with prediction")
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to generate data recommendations", error=str(e))
            return ["Review data quality and consult documentation"]
    
    async def _generate_dataset_recommendations(
        self,
        df: pd.DataFrame,
        errors: List[Dict[str, str]],
        warnings: List[Dict[str, str]],
        dataset_type: str
    ) -> List[str]:
        """Generate recommendations for dataset improvement"""
        recommendations = []
        
        try:
            if errors:
                recommendations.append("Address critical data errors before using dataset for training")
            
            if len(df) < 1000:
                recommendations.append("Consider collecting more data - larger datasets typically improve model performance")
            
            # Missing data recommendations
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_pct > 0.1:
                recommendations.append("Consider imputation strategies for missing values")
            
            # Feature recommendations
            if "Data_Value" in df.columns:
                target_variance = df["Data_Value"].var()
                if target_variance == 0:
                    recommendations.append("Target variable has no variance - check data collection process")
            
            if not recommendations:
                recommendations.append("Dataset quality is good for cardiovascular disease prediction")
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to generate dataset recommendations", error=str(e))
            return ["Review dataset structure and quality"]
    
    async def _generate_comprehensive_recommendations(
        self,
        df: pd.DataFrame,
        errors: List[Dict[str, str]],
        warnings: List[Dict[str, str]],
        quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive recommendations for data improvement"""
        recommendations = []
        
        try:
            quality_score = quality_metrics.get("overall_score", 0)
            
            if quality_score < 0.5:
                recommendations.append("Data quality is poor - significant improvements needed before model training")
            elif quality_score < 0.8:
                recommendations.append("Data quality is moderate - some improvements recommended")
            else:
                recommendations.append("Data quality is good - suitable for model training")
            
            # Specific recommendations based on analysis
            if quality_metrics.get("completeness", 1) < 0.9:
                recommendations.append("Improve data completeness by addressing missing values")
            
            if len(errors) > 0:
                recommendations.append("Fix data validation errors to ensure model reliability")
            
            if len(warnings) > 10:
                recommendations.append("Review and address data quality warnings")
            
            # Size recommendations
            if len(df) < 5000:
                recommendations.append("Consider collecting more data for better model performance")
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            logger.error("Failed to generate comprehensive recommendations", error=str(e))
            return ["Review data quality report and address identified issues"]


class DataProcessor:
    """Data processing and transformation utilities"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Processing pipelines
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": "RobustScaler"  # Would import from sklearn
        }
        
        self.imputers = {
            "simple_mean": SimpleImputer(strategy="mean"),
            "simple_median": SimpleImputer(strategy="median"),
            "simple_mode": SimpleImputer(strategy="most_frequent"),
            "knn": KNNImputer(n_neighbors=5)
        }
    
    async def process_cdc_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw CDC cardiovascular disease data"""
        try:
            logger.info("Processing CDC data", rows=len(raw_data))
            
            # Start with a copy
            df = raw_data.copy()
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            
            # Convert data types
            df = await self._convert_data_types(df)
            
            # Handle missing values
            df = await self._handle_missing_values(df)
            
            # Create derived features
            df = await self._create_derived_features(df)
            
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed_duplicates = initial_rows - len(df)
            
            if removed_duplicates > 0:
                logger.info("Removed duplicate rows", count=removed_duplicates)
            
            logger.info("CDC data processing completed", final_rows=len(df))
            return df
            
        except Exception as e:
            logger.error("CDC data processing failed", error=str(e), exc_info=True)
            raise
    
    async def process_dataset(
        self,
        df: pd.DataFrame,
        processing_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process dataset with custom parameters"""
        try:
            logger.info("Processing dataset with custom parameters", rows=len(df))
            
            processed_df = df.copy()
            
            # Apply processing steps based on parameters
            if processing_params.get("remove_duplicates", True):
                processed_df = processed_df.drop_duplicates()
            
            if processing_params.get("handle_missing_values", True):
                processed_df = await self._handle_missing_values(processed_df)
            
            if processing_params.get("outlier_detection", True):
                processed_df = await self._handle_outliers(processed_df)
            
            if processing_params.get("feature_engineering", True):
                processed_df = await self._create_derived_features(processed_df)
            
            if processing_params.get("data_validation", True):
                # Validate processed data
                validator = DataValidator()
                validation_result = await validator.quick_validate(processed_df)
                if not validation_result.is_valid:
                    logger.warning("Processed data validation failed", errors=validation_result.errors)
            
            logger.info("Dataset processing completed", final_rows=len(processed_df))
            return processed_df
            
        except Exception as e:
            logger.error("Dataset processing failed", error=str(e), exc_info=True)
            raise
    
    # Private processing methods
    async def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types"""
        try:
            # Numeric columns
            numeric_columns = [
                "age", "Data_Value", "Data_Value_Alt", "Low_Confidence_Limit", 
                "High_Confidence_Limit", "systolic_bp", "diastolic_bp",
                "total_cholesterol", "hdl_cholesterol", "bmi"
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Categorical columns
            categorical_columns = [
                "gender", "race_ethnicity", "smoking_status", "LocationAbbr",
                "Break_Out", "Break_Out_Category"
            ]
            
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            # Date columns
            date_columns = ["YearStart"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error("Data type conversion failed", error=str(e))
            return df
    
    async def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""
        try:
            # Strategy 1: Remove columns with >90% missing values
            threshold = 0.9
            missing_ratio = df.isnull().sum() / len(df)
            cols_to_remove = missing_ratio[missing_ratio > threshold].index.tolist()
            
            if cols_to_remove:
                df = df.drop(columns=cols_to_remove)
                logger.info("Removed columns with >90% missing values", columns=cols_to_remove)
            
            # Strategy 2: Impute numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    # Use median for numeric columns
                    df[col] = df[col].fillna(df[col].median())
            
            # Strategy 3: Impute categorical columns
            categorical_columns = df.select_dtypes(include=['category', 'object']).columns
            for col in categorical_columns:
                if df[col].isnull().sum() > 0:
                    # Use mode for categorical columns
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
                    df[col] = df[col].fillna(mode_value)
            
            # Strategy 4: Boolean columns
            boolean_columns = ["has_hypertension", "has_diabetes", "has_heart_disease"]
            for col in boolean_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(False)  # Assume False if missing
            
            return df
            
        except Exception as e:
            logger.error("Missing value handling failed", error=str(e))
            return df
    
    async def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in df.columns and col != "Data_Value":  # Don't modify target
                    # Use IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            return df
            
        except Exception as e:
            logger.error("Outlier handling failed", error=str(e))
            return df
    
    async def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for improved prediction"""
        try:
            # BMI calculation if height and weight available
            if "height_cm" in df.columns and "weight_kg" in df.columns:
                df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
            
            # Blood pressure categories
            if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
                df["bp_category"] = df.apply(self._categorize_blood_pressure, axis=1)
            
            # Age groups
            if "age" in df.columns:
                df["age_group"] = pd.cut(
                    df["age"],
                    bins=[0, 65, 75, 85, 120],
                    labels=["under_65", "65_74", "75_84", "85_plus"]
                )
            
            # Comorbidity count
            condition_columns = [
                "has_hypertension", "has_diabetes", "has_heart_disease",
                "has_stroke_history", "has_heart_attack_history"
            ]
            
            available_conditions = [col for col in condition_columns if col in df.columns]
            if available_conditions:
                df["comorbidity_count"] = df[available_conditions].sum(axis=1)
            
            # Risk factor interactions
            if "has_diabetes" in df.columns and "has_hypertension" in df.columns:
                df["diabetes_hypertension_combo"] = (
                    df["has_diabetes"] & df["has_hypertension"]
                ).astype(int)
            
            # Cholesterol ratio
            if "total_cholesterol" in df.columns and "hdl_cholesterol" in df.columns:
                df["cholesterol_ratio"] = df["total_cholesterol"] / df["hdl_cholesterol"].replace(0, np.nan)
            
            return df
            
        except Exception as e:
            logger.error("Feature engineering failed", error=str(e))
            return df
    
    def _categorize_blood_pressure(self, row) -> str:
        """Categorize blood pressure based on AHA guidelines"""
        try:
            systolic = row.get("systolic_bp")
            diastolic = row.get("diastolic_bp")
            
            if pd.isna(systolic) or pd.isna(diastolic):
                return "unknown"
            
            if systolic < 120 and diastolic < 80:
                return "normal"
            elif systolic < 130 and diastolic < 80:
                return "elevated"
            elif (120 <= systolic <= 129) or (80 <= diastolic <= 89):
                return "stage1_hypertension"
            elif systolic >= 130 or diastolic >= 90:
                return "stage2_hypertension"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"


class FeatureEngineer:
    """Advanced feature engineering for cardiovascular prediction"""
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_selectors = {
            "univariate": SelectKBest(score_func=f_regression),
            "mutual_info": SelectKBest(score_func=mutual_info_regression)
        }
    
    async def engineer_features(
        self,
        df: pd.DataFrame,
        target_column: str = "Data_Value"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Perform comprehensive feature engineering"""
        try:
            logger.info("Starting feature engineering", original_features=len(df.columns))
            
            # Start with processed dataframe
            feature_df = df.copy()
            
            # Create interaction features
            feature_df = await self._create_interaction_features(feature_df)
            
            # Create polynomial features for key numeric variables
            feature_df = await self._create_polynomial_features(feature_df)
            
            # Create binned features
            feature_df = await self._create_binned_features(feature_df)
            
            # Create temporal features if date columns exist
            feature_df = await self._create_temporal_features(feature_df)
            
            # One-hot encode categorical variables
            feature_df = await self._encode_categorical_features(feature_df)
            
            # Feature selection
            if target_column in feature_df.columns:
                feature_df, selected_features = await self._select_features(
                    feature_df, target_column
                )
            else:
                selected_features = [col for col in feature_df.columns if col != target_column]
            
            logger.info(
                "Feature engineering completed",
                final_features=len(feature_df.columns),
                selected_features=len(selected_features)
            )
            
            return feature_df, selected_features
            
        except Exception as e:
            logger.error("Feature engineering failed", error=str(e), exc_info=True)
            # Return original dataframe if feature engineering fails
            return df, [col for col in df.columns if col != target_column]
    
    async def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        try:
            # Age interactions
            if "age" in df.columns:
                if "has_diabetes" in df.columns:
                    df["age_diabetes_interaction"] = df["age"] * df["has_diabetes"].astype(int)
                
                if "has_hypertension" in df.columns:
                    df["age_hypertension_interaction"] = df["age"] * df["has_hypertension"].astype(int)
            
            # BMI interactions
            if "bmi" in df.columns:
                if "has_diabetes" in df.columns:
                    df["bmi_diabetes_interaction"] = df["bmi"] * df["has_diabetes"].astype(int)
            
            # Blood pressure interactions
            if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
                df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
                df["mean_arterial_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3)
            
            return df
            
        except Exception as e:
            logger.error("Interaction feature creation failed", error=str(e))
            return df
    
    async def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for key numeric variables"""
        try:
            # Create squared terms for key continuous variables
            polynomial_columns = ["age", "bmi", "systolic_bp", "total_cholesterol"]
            
            for col in polynomial_columns:
                if col in df.columns:
                    df[f"{col}_squared"] = df[col] ** 2
            
            return df
            
        except Exception as e:
            logger.error("Polynomial feature creation failed", error=str(e))
            return df
    
    async def _create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned versions of continuous variables"""
        try:
            # Age bins
            if "age" in df.columns:
                df["age_bin"] = pd.cut(
                    df["age"],
                    bins=[0, 65, 75, 85, 120],
                    labels=["under_65", "65_74", "75_84", "85_plus"]
                )
            
            # BMI bins
            if "bmi" in df.columns:
                df["bmi_category"] = pd.cut(
                    df["bmi"],
                    bins=[0, 18.5, 25, 30, 50],
                    labels=["underweight", "normal", "overweight", "obese"]
                )
            
            # Blood pressure bins
            if "systolic_bp" in df.columns:
                df["bp_category"] = pd.cut(
                    df["systolic_bp"],
                    bins=[0, 120, 130, 140, 300],
                    labels=["normal", "elevated", "stage1", "stage2"]
                )
            
            return df
            
        except Exception as e:
            logger.error("Binned feature creation failed", error=str(e))
            return df
    
    async def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date columns"""
        try:
            date_columns = df.select_dtypes(include=['datetime64']).columns
            
            for col in date_columns:
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_quarter"] = df[col].dt.quarter
                df[f"{col}_day_of_year"] = df[col].dt.dayofyear
            
            return df
            
        except Exception as e:
            logger.error("Temporal feature creation failed", error=str(e))
            return df
    
    async def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables"""
        try:
            # Get categorical columns
            categorical_columns = df.select_dtypes(include=['category', 'object']).columns
            
            # Exclude high-cardinality columns
            high_cardinality_threshold = 50
            categorical_to_encode = []
            
            for col in categorical_columns:
                unique_count = df[col].nunique()
                if unique_count <= high_cardinality_threshold:
                    categorical_to_encode.append(col)
                else:
                    logger.warning(f"Skipping high-cardinality column: {col} ({unique_count} unique values)")
            
            # One-hot encode
            if categorical_to_encode:
                df_encoded = pd.get_dummies(
                    df,
                    columns=categorical_to_encode,
                    drop_first=True,  # Avoid multicollinearity
                    dummy_na=False
                )
                return df_encoded
            
            return df
            
        except Exception as e:
            logger.error("Categorical encoding failed", error=str(e))
            return df
    
    async def _select_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_features: int = 50
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features for model training"""
        try:
            if target_column not in df.columns:
                # Return all features if no target
                feature_columns = [col for col in df.columns if col != target_column]
                return df, feature_columns
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Remove non-numeric columns for feature selection
            numeric_features = X.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                logger.warning("No numeric features found for selection")
                return df, list(X.columns)
            
            # Handle missing values for feature selection
            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(numeric_features)
            
            # Select features using mutual information
            n_features = min(max_features, len(numeric_features.columns))
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            
            X_selected = selector.fit_transform(X_imputed, y)
            selected_feature_indices = selector.get_support(indices=True)
            selected_feature_names = numeric_features.columns[selected_feature_indices].tolist()
            
            # Keep target column and selected features
            final_columns = selected_feature_names + [target_column]
            final_df = df[final_columns]
            
            logger.info(
                "Feature selection completed",
                original_features=len(X.columns),
                selected_features=len(selected_feature_names)
            )
            
            return final_df, selected_feature_names
            
        except Exception as e:
            logger.error("Feature selection failed", error=str(e))
            # Return original data if selection fails
            feature_columns = [col for col in df.columns if col != target_column]
            return df, feature_columns


class DataCleaner:
    """Data cleaning utilities for cardiovascular datasets"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def clean_cdc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CDC cardiovascular disease dataset"""
        try:
            logger.info("Cleaning CDC dataset", initial_rows=len(df))
            
            cleaned_df = df.copy()
            
            # Remove system/metadata columns that aren't useful for prediction
            system_columns = [
                "RowId", "ClassId", "TopicId", "QuestionId", "Data_Value_TypeID",
                "BreakOutCategoryId", "BreakOutId", "LocationId"
            ]
            
            columns_to_remove = [col for col in system_columns if col in cleaned_df.columns]
            if columns_to_remove:
                cleaned_df = cleaned_df.drop(columns=columns_to_remove)
                logger.info("Removed system columns", columns=columns_to_remove)
            
            # Clean text columns
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
            for col in text_columns:
                if col in cleaned_df.columns:
                    # Strip whitespace and standardize
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    
                    # Replace common inconsistencies
                    cleaned_df[col] = cleaned_df[col].replace({
                        "Unknown": "unknown",
                        "UNKNOWN": "unknown",
                        "N/A": "unknown",
                        "": "unknown"
                    })
            
            # Standardize gender values
            if "Break_Out" in cleaned_df.columns:
                gender_mapping = {
                    "Male": "male",
                    "Female": "female",
                    "Unknown": "unknown"
                }
                
                # Create gender column if it's in Break_Out
                if cleaned_df["Break_Out"].isin(gender_mapping.keys()).any():
                    cleaned_df["gender"] = cleaned_df["Break_Out"].map(gender_mapping)
            
            # Clean numeric columns
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Replace infinite values with NaN
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Remove negative values for columns that shouldn't be negative
                non_negative_columns = ["age", "Data_Value", "systolic_bp", "diastolic_bp"]
                if col in non_negative_columns:
                    cleaned_df[col] = cleaned_df[col].where(cleaned_df[col] >= 0)
            
            # Remove rows with missing target variable (for training data)
            if "Data_Value" in cleaned_df.columns:
                initial_count = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=["Data_Value"])
                removed_count = initial_count - len(cleaned_df)
                
                if removed_count > 0:
                    logger.info("Removed rows with missing target", count=removed_count)
            
            logger.info("CDC data cleaning completed", final_rows=len(cleaned_df))
            return cleaned_df
            
        except Exception as e:
            logger.error("CDC data cleaning failed", error=str(e), exc_info=True)
            raise
    
    async def clean_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean individual patient data for prediction"""
        try:
            cleaned_data = patient_data.copy()
            
            # Standardize string values
            string_fields = ["gender", "smoking_status", "race_ethnicity"]
            for field in string_fields:
                if field in cleaned_data and isinstance(cleaned_data[field], str):
                    cleaned_data[field] = cleaned_data[field].strip().lower()
            
            # Validate and clean numeric fields
            numeric_fields = ["age", "bmi", "systolic_bp", "diastolic_bp", "total_cholesterol"]
            for field in numeric_fields:
                if field in cleaned_data:
                    value = cleaned_data[field]
                    if value is not None:
                        try:
                            cleaned_data[field] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert {field} to numeric: {value}")
                            cleaned_data[field] = None
            
            # Ensure boolean fields are properly formatted
            boolean_fields = [
                "has_hypertension", "has_diabetes", "has_heart_disease",
                "medicare_part_a", "medicare_part_b", "data_sharing_consent"
            ]
            
            for field in boolean_fields:
                if field in cleaned_data:
                    value = cleaned_data[field]
                    if isinstance(value, str):
                        cleaned_data[field] = value.lower() in ["true", "yes", "1"]
                    elif value is None:
                        cleaned_data[field] = False
            
            return cleaned_data
            
        except Exception as e:
            logger.error("Patient data cleaning failed", error=str(e))
            return patient_data


class DataQualityAnalyzer:
    """Analyze and report on data quality metrics"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def analyze_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        try:
            logger.info("Analyzing data quality", rows=len(df))
            
            quality_report = {
                "completeness": await self._analyze_completeness(df),
                "consistency": await self._analyze_consistency(df),
                "accuracy": await self._analyze_accuracy(df),
                "validity": await self._analyze_validity(df),
                "uniqueness": await self._analyze_uniqueness(df),
                "timeliness": await self._analyze_timeliness(df)
            }
            
            # Calculate overall quality score
            scores = [metrics.get("score", 0) for metrics in quality_report.values()]
            overall_score = np.mean(scores)
            
            quality_report["overall_score"] = round(overall_score, 3)
            quality_report["analysis_date"] = datetime.utcnow().isoformat()
            
            return quality_report
            
        except Exception as e:
            logger.error("Data quality analysis failed", error=str(e))
            return {"overall_score": 0.0, "error": str(e)}
    
    async def _analyze_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness"""
        try:
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            completeness_score = 1 - (missing_cells / total_cells)
            
            # Per-column completeness
            column_completeness = {}
            for col in df.columns:
                col_completeness = 1 - (df[col].isnull().sum() / len(df))
                column_completeness[col] = round(col_completeness, 3)
            
            return {
                "score": round(completeness_score, 3),
                "total_cells": total_cells,
                "missing_cells": missing_cells,
                "column_completeness": column_completeness,
                "columns_with_missing": len([col for col, comp in column_completeness.items() if comp < 1.0])
            }
            
        except Exception as e:
            logger.error("Completeness analysis failed", error=str(e))
            return {"score": 0.0, "error": str(e)}
    
    async def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency"""
        try:
            consistency_issues = []
            
            # Check for format consistency in string columns
            for col in df.select_dtypes(include=['object']).columns:
                if col in df.columns:
                    # Check for mixed case
                    values = df[col].dropna().astype(str)
                    if len(values) > 0:
                        mixed_case = len(set(values.str.lower())) != len(set(values))
                        if mixed_case:
                            consistency_issues.append(f"{col}: Mixed case values detected")
            
            # Check numeric consistency
            if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
                invalid_bp = df[df["systolic_bp"] <= df["diastolic_bp"]]
                if len(invalid_bp) > 0:
                    consistency_issues.append(f"Blood pressure: {len(invalid_bp)} records with systolic ≤ diastolic")
            
            consistency_score = max(0.0, 1.0 - (len(consistency_issues) / 10))  # Penalty for issues
            
            return {
                "score": round(consistency_score, 3),
                "issues": consistency_issues,
                "issues_count": len(consistency_issues)
            }
            
        except Exception as e:
            logger.error("Consistency analysis failed", error=str(e))
            return {"score": 0.0, "error": str(e)}
    
    async def _analyze_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data accuracy (based on reasonable ranges)"""
        try:
            accuracy_issues = []
            
            # Check for values outside reasonable ranges
            for col, (min_val, max_val) in DataValidator().value_ranges.items():
                if col in df.columns:
                    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                    if len(out_of_range) > 0:
                        accuracy_issues.append(f"{col}: {len(out_of_range)} values outside reasonable range")
            
            accuracy_score = max(0.0, 1.0 - (len(accuracy_issues) / 20))
            
            return {
                "score": round(accuracy_score, 3),
                "issues": accuracy_issues,
                "issues_count": len(accuracy_issues)
            }
            
        except Exception as e:
            logger.error("Accuracy analysis failed", error=str(e))
            return {"score": 0.0, "error": str(e)}
    
    async def _analyze_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data validity against business rules"""
        try:
            validity_issues = []
            
            # Medicare age validity
            if "age" in df.columns:
                under_65 = df[df["age"] < 65]
                if len(under_65) > len(df) * 0.1:  # More than 10% under 65
                    validity_issues.append(f"Age: {len(under_65)} Medicare patients under 65 (unusual for standard Medicare)")
            
            validity_score = max(0.0, 1.0 - (len(validity_issues) / 10))
            
            return {
                "score": round(validity_score, 3),
                "issues": validity_issues,
                "issues_count": len(validity_issues)
            }
            
        except Exception as e:
            logger.error("Validity analysis failed", error=str(e))
            return {"score": 0.0, "error": str(e)}
    
    async def _analyze_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data uniqueness and duplicates"""
        try:
            duplicate_count = df.duplicated().sum()
            uniqueness_score = 1.0 - (duplicate_count / len(df))
            
            # Check for unexpected duplicates in key columns
            key_columns = ["external_id", "medicare_id"] if any(col in df.columns for col in ["external_id", "medicare_id"]) else []
            
            key_duplicates = 0
            for col in key_columns:
                if col in df.columns:
                    col_duplicates = df[col].duplicated().sum()
                    key_duplicates += col_duplicates
            
            return {
                "score": round(uniqueness_score, 3),
                "total_duplicates": duplicate_count,
                "key_column_duplicates": key_duplicates,
                "uniqueness_percentage": round(uniqueness_score * 100, 1)
            }
            
        except Exception as e:
            logger.error("Uniqueness analysis failed", error=str(e))
            return {"score": 0.0, "error": str(e)}
    
    async def _analyze_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data timeliness and recency"""
        try:
            # Check for date columns
            date_columns = df.select_dtypes(include=['datetime64']).columns
            
            if len(date_columns) == 0:
                return {"score": 1.0, "message": "No date columns to analyze"}
            
            # Analyze most recent date column
            date_col = date_columns[0]
            
            if not df[date_col].empty:
                latest_date = df[date_col].max()
                oldest_date = df[date_col].min()
                
                # Calculate recency score (more recent data scores higher)
                days_since_latest = (datetime.now() - latest_date).days
                timeliness_score = max(0.0, 1.0 - (days_since_latest / 365))  # 1 year decay
                
                return {
                    "score": round(timeliness_score, 3),
                    "latest_date": latest_date.isoformat(),
                    "oldest_date": oldest_date.isoformat(),
                    "days_since_latest": days_since_latest,
                    "date_range_days": (latest_date - oldest_date).days
                }
            
            return {"score": 0.0, "message": "No valid dates found"}
            
        except Exception as e:
            logger.error("Timeliness analysis failed", error=str(e))
            return {"score": 0.0, "error": str(e)}


# Export all preprocessing utilities
__all__ = [
    "DataValidator",
    "DataProcessor", 
    "FeatureEngineer",
    "DataCleaner",
    "DataQualityAnalyzer"
]