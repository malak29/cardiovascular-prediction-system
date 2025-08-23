import asyncio
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Union
import uuid
import json
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
import structlog
import pandas as pd
import numpy as np

from app.core.database import db_manager
from app.core.config import get_settings
from app.models.prediction import (
    Prediction,
    PredictionBatch,
    PredictionFeedback,
    PredictionAuditLog,
    PredictionModel
)
from app.models.patient import Patient
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    PredictionHistoryResponse,
    PredictionHistoryItem,
    FeatureImportanceSchema
)
from app.services.ml_service import MLService

logger = structlog.get_logger(__name__)


class PredictionService:
    """Service for handling cardiovascular disease predictions"""
    
    def __init__(self):
        self.settings = get_settings()
        self.ml_service = MLService()
        
        # Cache for frequently accessed data
        self._prediction_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, List[FeatureImportanceSchema]] = {}
    
    async def predict_single(
        self,
        patient_data: Dict[str, Any],
        model_version: Optional[str] = None,
        include_confidence: bool = False,
        include_features: bool = False
    ) -> PredictionResponse:
        """Make a single cardiovascular disease risk prediction"""
        try:
            start_time = datetime.utcnow()
            
            logger.info("Processing single prediction request")
            
            # Use ML service to make prediction
            ml_result = await self.ml_service.predict_single(
                features=patient_data,
                model_version=model_version,
                include_confidence=include_confidence
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Generate prediction ID
            prediction_id = uuid.uuid4()
            
            # Get feature importance if requested
            feature_importance = None
            if include_features:
                feature_importance = ml_result.get("feature_importance", [])
            
            # Generate clinical recommendations
            recommendations = await self._generate_recommendations(
                ml_result["risk_score"],
                patient_data
            )
            
            # Create response
            response = PredictionResponse(
                risk_score=ml_result["risk_score"],
                risk_category=ml_result["risk_category"],
                confidence_score=ml_result.get("confidence_score"),
                confidence_lower=ml_result.get("confidence_lower"),
                confidence_upper=ml_result.get("confidence_upper"),
                confidence_level=ml_result.get("confidence_level", 0.95),
                feature_importance=feature_importance,
                model_version=ml_result["model_version"],
                model_type=ml_result["model_type"],
                prediction_id=prediction_id,
                prediction_time_ms=processing_time,
                timestamp=datetime.utcnow(),
                recommendations=recommendations,
                risk_factors=await self._identify_risk_factors(patient_data)
            )
            
            logger.info(
                "Single prediction completed",
                prediction_id=prediction_id,
                risk_score=ml_result["risk_score"],
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error("Single prediction failed", error=str(e), exc_info=True)
            raise
    
    async def predict_batch(
        self,
        patients_data: List[Dict[str, Any]],
        model_version: Optional[str] = None,
        include_confidence: bool = False,
        parallel_processing: bool = True
    ) -> BatchPredictionResponse:
        """Make batch cardiovascular disease risk predictions"""
        try:
            start_time = datetime.utcnow()
            batch_id = uuid.uuid4()
            
            logger.info(
                "Processing batch prediction request",
                batch_id=batch_id,
                batch_size=len(patients_data)
            )
            
            # Use ML service for batch prediction
            ml_result = await self.ml_service.predict_batch(
                features_list=patients_data,
                model_version=model_version,
                parallel_processing=parallel_processing
            )
            
            # Process results
            predictions = []
            failed_predictions = []
            
            for result in ml_result["predictions"]:
                try:
                    patient_index = result["index"]
                    
                    # Create individual prediction response
                    individual_prediction = PredictionResponse(
                        risk_score=result["risk_score"],
                        risk_category=result["risk_category"],
                        model_version=result["model_version"],
                        model_type=result["model_type"],
                        prediction_id=uuid.uuid4(),
                        prediction_time_ms=0,  # Part of batch
                        timestamp=datetime.utcnow()
                    )
                    
                    batch_item = BatchPredictionItem(
                        patient_index=patient_index,
                        prediction=individual_prediction
                    )
                    
                    predictions.append(batch_item)
                    
                except Exception as e:
                    failed_item = BatchPredictionItem(
                        patient_index=result.get("index", -1),
                        error=str(e)
                    )
                    failed_predictions.append(failed_item)
            
            # Calculate processing time
            total_processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Generate batch summary
            batch_summary = await self._calculate_batch_summary(predictions)
            
            # Create batch response
            response = BatchPredictionResponse(
                predictions=predictions,
                failed_predictions=failed_predictions,
                summary=batch_summary,
                batch_id=batch_id,
                total_requested=len(patients_data),
                successful_count=len(predictions),
                failed_count=len(failed_predictions),
                processing_time_ms=total_processing_time,
                model_version=ml_result.get("model_version", "unknown"),
                timestamp=datetime.utcnow()
            )
            
            logger.info(
                "Batch prediction completed",
                batch_id=batch_id,
                successful_predictions=len(predictions),
                failed_predictions=len(failed_predictions),
                processing_time_ms=total_processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error("Batch prediction failed", error=str(e), exc_info=True)
            raise
    
    async def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        model_version: Optional[str] = None,
        include_confidence: bool = False
    ) -> pd.DataFrame:
        """Make predictions from a pandas DataFrame"""
        try:
            logger.info("Processing DataFrame prediction", rows=len(df))
            
            # Convert DataFrame rows to list of dictionaries
            patients_data = df.to_dict(orient='records')
            
            # Make batch predictions
            batch_result = await self.predict_batch(
                patients_data=patients_data,
                model_version=model_version,
                include_confidence=include_confidence,
                parallel_processing=True
            )
            
            # Add prediction results to original DataFrame
            result_df = df.copy()
            
            # Initialize prediction columns
            result_df['risk_score'] = np.nan
            result_df['risk_category'] = ''
            result_df['prediction_id'] = ''
            
            if include_confidence:
                result_df['confidence_lower'] = np.nan
                result_df['confidence_upper'] = np.nan
                result_df['confidence_score'] = np.nan
            
            # Fill in successful predictions
            for prediction_item in batch_result.predictions:
                idx = prediction_item.patient_index
                if prediction_item.prediction:
                    pred = prediction_item.prediction
                    result_df.loc[idx, 'risk_score'] = pred.risk_score
                    result_df.loc[idx, 'risk_category'] = pred.risk_category
                    result_df.loc[idx, 'prediction_id'] = str(pred.prediction_id)
                    
                    if include_confidence and pred.confidence_lower is not None:
                        result_df.loc[idx, 'confidence_lower'] = pred.confidence_lower
                        result_df.loc[idx, 'confidence_upper'] = pred.confidence_upper
                        result_df.loc[idx, 'confidence_score'] = pred.confidence_score
            
            # Mark failed predictions
            result_df['prediction_error'] = ''
            for failed_item in batch_result.failed_predictions:
                idx = failed_item.patient_index
                result_df.loc[idx, 'prediction_error'] = failed_item.error
            
            logger.info("DataFrame prediction completed", output_rows=len(result_df))
            return result_df
            
        except Exception as e:
            logger.error("DataFrame prediction failed", error=str(e), exc_info=True)
            raise
    
    async def store_prediction(
        self,
        prediction_request: PredictionRequest,
        prediction_result: PredictionResponse,
        user_id: Optional[str],
        correlation_id: str,
        db: AsyncSession
    ) -> None:
        """Store prediction in database for audit trail"""
        try:
            logger.debug("Storing prediction", prediction_id=prediction_result.prediction_id)
            
            # Create prediction record
            prediction = Prediction(
                id=prediction_result.prediction_id,
                correlation_id=correlation_id,
                prediction_type="single",
                model_version=prediction_result.model_version,
                input_features=prediction_request.patient_data.dict(),
                risk_score=prediction_result.risk_score,
                risk_category=prediction_result.risk_category,
                confidence_score=prediction_result.confidence_score,
                confidence_lower=prediction_result.confidence_lower,
                confidence_upper=prediction_result.confidence_upper,
                confidence_level=prediction_result.confidence_level,
                feature_importance=[fi.dict() for fi in prediction_result.feature_importance] if prediction_result.feature_importance else None,
                prediction_time_ms=prediction_result.prediction_time_ms,
                user_id=uuid.UUID(user_id) if user_id else None
            )
            
            # Add to database
            db.add(prediction)
            
            # Create audit log entry
            audit_log = PredictionAuditLog(
                event_type="prediction_created",
                event_description=f"Single prediction made with risk score {prediction_result.risk_score}",
                prediction_id=prediction_result.prediction_id,
                user_id=uuid.UUID(user_id) if user_id else None,
                correlation_id=correlation_id,
                event_data={
                    "prediction_type": "single",
                    "model_version": prediction_result.model_version,
                    "risk_category": prediction_result.risk_category
                }
            )
            
            db.add(audit_log)
            await db.commit()
            
            logger.debug("Prediction stored successfully", prediction_id=prediction_result.prediction_id)
            
        except Exception as e:
            logger.error("Failed to store prediction", error=str(e), exc_info=True)
            # Don't raise - this is background task
    
    async def store_batch_predictions(
        self,
        batch_request: BatchPredictionRequest,
        batch_result: BatchPredictionResponse,
        user_id: Optional[str],
        correlation_id: str,
        db: AsyncSession
    ) -> None:
        """Store batch predictions in database"""
        try:
            logger.debug("Storing batch predictions", batch_id=batch_result.batch_id)
            
            # Create batch record
            batch_record = PredictionBatch(
                id=batch_result.batch_id,
                batch_name=batch_request.batch_name,
                correlation_id=correlation_id,
                job_status="completed",
                total_records=batch_result.total_requested,
                processed_records=batch_result.total_requested,
                successful_predictions=batch_result.successful_count,
                failed_predictions=batch_result.failed_count,
                model_version=batch_result.model_version,
                processing_parameters=batch_request.dict(),
                started_at=batch_result.timestamp,
                completed_at=batch_result.timestamp,
                processing_duration_seconds=batch_result.processing_time_ms / 1000,
                user_id=uuid.UUID(user_id) if user_id else None,
                results_summary=batch_result.summary
            )
            
            db.add(batch_record)
            
            # Store individual predictions
            for prediction_item in batch_result.predictions:
                if prediction_item.prediction:
                    pred = prediction_item.prediction
                    prediction_record = Prediction(
                        id=pred.prediction_id,
                        correlation_id=correlation_id,
                        prediction_type="batch",
                        model_version=pred.model_version,
                        input_features=batch_request.patients_data[prediction_item.patient_index].dict(),
                        risk_score=pred.risk_score,
                        risk_category=pred.risk_category,
                        confidence_score=pred.confidence_score,
                        confidence_lower=pred.confidence_lower,
                        confidence_upper=pred.confidence_upper,
                        prediction_time_ms=0,  # Part of batch
                        user_id=uuid.UUID(user_id) if user_id else None
                    )
                    
                    db.add(prediction_record)
            
            await db.commit()
            
            logger.info(
                "Batch predictions stored successfully",
                batch_id=batch_result.batch_id,
                stored_predictions=batch_result.successful_count
            )
            
        except Exception as e:
            logger.error("Failed to store batch predictions", error=str(e), exc_info=True)
    
    async def store_csv_prediction_job(
        self,
        original_filename: str,
        row_count: int,
        user_id: Optional[str],
        correlation_id: str,
        db: AsyncSession
    ) -> None:
        """Store CSV prediction job information"""
        try:
            logger.debug("Storing CSV prediction job", filename=original_filename)
            
            # Create batch record for CSV job
            batch_record = PredictionBatch(
                batch_name=f"CSV Upload: {original_filename}",
                correlation_id=correlation_id,
                job_status="completed",
                total_records=row_count,
                processed_records=row_count,
                successful_predictions=row_count,  # Assume all successful for CSV
                original_filename=original_filename,
                user_id=uuid.UUID(user_id) if user_id else None,
                completed_at=datetime.utcnow()
            )
            
            db.add(batch_record)
            await db.commit()
            
            logger.debug("CSV prediction job stored", correlation_id=correlation_id)
            
        except Exception as e:
            logger.error("Failed to store CSV prediction job", error=str(e), exc_info=True)
    
    async def get_prediction_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        risk_category: Optional[str] = None,
        db: AsyncSession = None
    ) -> PredictionHistoryResponse:
        """Get prediction history with filtering and pagination"""
        try:
            if not db:
                async with db_manager.get_async_session() as db:
                    return await self._get_prediction_history_impl(
                        user_id, limit, offset, start_date, end_date, risk_category, db
                    )
            else:
                return await self._get_prediction_history_impl(
                    user_id, limit, offset, start_date, end_date, risk_category, db
                )
                
        except Exception as e:
            logger.error("Failed to get prediction history", error=str(e), exc_info=True)
            raise
    
    async def _get_prediction_history_impl(
        self,
        user_id: Optional[str],
        limit: int,
        offset: int,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        risk_category: Optional[str],
        db: AsyncSession
    ) -> PredictionHistoryResponse:
        """Implementation of get_prediction_history"""
        try:
            # Build query
            query = select(Prediction).order_by(desc(Prediction.created_at))
            
            # Apply filters
            filters = []
            
            if user_id:
                filters.append(Prediction.user_id == uuid.UUID(user_id))
            
            if start_date:
                filters.append(Prediction.created_at >= start_date)
            
            if end_date:
                filters.append(Prediction.created_at <= end_date)
            
            if risk_category:
                filters.append(Prediction.risk_category == risk_category)
            
            if filters:
                query = query.where(and_(*filters))
            
            # Get total count for pagination
            count_query = select(func.count(Prediction.id))
            if filters:
                count_query = count_query.where(and_(*filters))
            
            count_result = await db.execute(count_query)
            total_count = count_result.scalar()
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Execute query
            result = await db.execute(query)
            predictions = result.scalars().all()
            
            # Convert to response format
            history_items = []
            for prediction in predictions:
                # Extract patient demographics from input features
                input_features = prediction.input_features or {}
                
                history_item = PredictionHistoryItem(
                    prediction_id=prediction.id,
                    risk_score=prediction.risk_score,
                    risk_category=prediction.risk_category,
                    model_version=prediction.model_version,
                    created_at=prediction.created_at,
                    patient_age=input_features.get("age", 0),
                    patient_gender=input_features.get("gender", "unknown"),
                    has_feedback=prediction.actual_outcome is not None,
                    actual_outcome=prediction.actual_outcome
                )
                
                history_items.append(history_item)
            
            # Calculate summary statistics
            summary = await self._calculate_history_summary(predictions)
            
            response = PredictionHistoryResponse(
                predictions=history_items,
                total_count=total_count,
                summary=summary,
                limit=limit,
                offset=offset,
                has_more=offset + limit < total_count
            )
            
            logger.info(
                "Prediction history retrieved",
                total_count=total_count,
                returned_count=len(history_items)
            )
            
            return response
            
        except Exception as e:
            logger.error("Prediction history implementation failed", error=str(e), exc_info=True)
            raise
    
    async def delete_prediction(
        self,
        prediction_id: str,
        user_id: str,
        db: AsyncSession
    ) -> bool:
        """Delete a prediction record"""
        try:
            logger.info("Deleting prediction", prediction_id=prediction_id, user_id=user_id)
            
            # Find prediction
            query = select(Prediction).where(Prediction.id == uuid.UUID(prediction_id))
            result = await db.execute(query)
            prediction = result.scalar_one_or_none()
            
            if not prediction:
                return False
            
            # Create audit log before deletion
            audit_log = PredictionAuditLog(
                event_type="prediction_deleted",
                event_description=f"Prediction {prediction_id} deleted by user {user_id}",
                prediction_id=prediction.id,
                user_id=uuid.UUID(user_id),
                event_data={
                    "deleted_prediction": {
                        "risk_score": prediction.risk_score,
                        "risk_category": prediction.risk_category,
                        "model_version": prediction.model_version
                    }
                }
            )
            
            db.add(audit_log)
            
            # Delete prediction
            await db.delete(prediction)
            await db.commit()
            
            logger.info("Prediction deleted successfully", prediction_id=prediction_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete prediction", prediction_id=prediction_id, error=str(e), exc_info=True)
            return False
    
    async def get_prediction_statistics(
        self,
        timeframe: str = "30d",
        user_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get comprehensive prediction statistics"""
        try:
            if not db:
                async with db_manager.get_async_session() as db:
                    return await self._get_prediction_statistics_impl(timeframe, user_id, db)
            else:
                return await self._get_prediction_statistics_impl(timeframe, user_id, db)
                
        except Exception as e:
            logger.error("Failed to get prediction statistics", error=str(e))
            return {}
    
    async def _get_prediction_statistics_impl(
        self,
        timeframe: str,
        user_id: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Implementation of get_prediction_statistics"""
        try:
            # Calculate date range based on timeframe
            end_date = datetime.utcnow()
            
            timeframe_days = {
                "1d": 1,
                "7d": 7,
                "30d": 30,
                "90d": 90,
                "1y": 365
            }
            
            if timeframe in timeframe_days:
                start_date = end_date - timedelta(days=timeframe_days[timeframe])
            else:  # "all"
                start_date = datetime(2020, 1, 1)  # Far back date
            
            # Build base query
            query = select(Prediction).where(Prediction.created_at >= start_date)
            
            if user_id:
                query = query.where(Prediction.user_id == uuid.UUID(user_id))
            
            # Execute query
            result = await db.execute(query)
            predictions = result.scalars().all()
            
            if not predictions:
                return {
                    "total_predictions": 0,
                    "timeframe": timeframe,
                    "message": "No predictions found for the specified timeframe"
                }
            
            # Calculate statistics
            total_predictions = len(predictions)
            risk_scores = [p.risk_score for p in predictions]
            
            # Risk distribution
            risk_distribution = {
                "low": len([p for p in predictions if p.risk_category == "low"]),
                "medium": len([p for p in predictions if p.risk_category == "medium"]),
                "high": len([p for p in predictions if p.risk_category == "high"])
            }
            
            # Model usage
            model_usage = {}
            for prediction in predictions:
                model_version = prediction.model_version
                if model_version not in model_usage:
                    model_usage[model_version] = 0
                model_usage[model_version] += 1
            
            # Trends analysis
            trends = await self._calculate_prediction_trends(predictions, timeframe)
            
            # Performance metrics (if feedback available)
            performance_metrics = await self._calculate_performance_metrics(predictions)
            
            statistics = {
                "total_predictions": total_predictions,
                "timeframe": timeframe,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "risk_distribution": risk_distribution,
                "risk_distribution_percentages": {
                    category: round((count / total_predictions) * 100, 1)
                    for category, count in risk_distribution.items()
                },
                "risk_statistics": {
                    "average_risk_score": round(np.mean(risk_scores), 4),
                    "median_risk_score": round(np.median(risk_scores), 4),
                    "std_risk_score": round(np.std(risk_scores), 4),
                    "min_risk_score": round(np.min(risk_scores), 4),
                    "max_risk_score": round(np.max(risk_scores), 4)
                },
                "model_usage": model_usage,
                "trends": trends,
                "performance_metrics": performance_metrics,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error("Prediction statistics implementation failed", error=str(e))
            return {}
    
    async def submit_prediction_feedback(
        self,
        prediction_id: str,
        feedback_data: Dict[str, Any],
        user_id: str,
        db: AsyncSession
    ) -> str:
        """Submit feedback for a prediction"""
        try:
            logger.info("Submitting prediction feedback", prediction_id=prediction_id)
            
            # Verify prediction exists
            query = select(Prediction).where(Prediction.id == uuid.UUID(prediction_id))
            result = await db.execute(query)
            prediction = result.scalar_one_or_none()
            
            if not prediction:
                raise ValueError(f"Prediction not found: {prediction_id}")
            
            # Create feedback record
            feedback = PredictionFeedback(
                prediction_id=uuid.UUID(prediction_id),
                feedback_type=feedback_data["feedback_type"],
                feedback_value=feedback_data,
                actual_outcome=feedback_data.get("actual_outcome"),
                outcome_date=feedback_data.get("outcome_date"),
                provider_rating=feedback_data.get("provider_rating"),
                provider_comments=feedback_data.get("provider_comments"),
                corrected_risk_score=feedback_data.get("corrected_risk_score"),
                corrected_risk_category=feedback_data.get("corrected_risk_category"),
                correction_reason=feedback_data.get("correction_reason"),
                provided_by=uuid.UUID(user_id)
            )
            
            db.add(feedback)
            
            # Update original prediction with feedback
            if feedback_data.get("actual_outcome") is not None:
                prediction.actual_outcome = feedback_data["actual_outcome"]
                prediction.outcome_date = feedback_data.get("outcome_date")
            
            if feedback_data.get("provider_rating"):
                prediction.feedback_score = feedback_data["provider_rating"]
            
            await db.commit()
            await db.refresh(feedback)
            
            logger.info("Prediction feedback submitted", feedback_id=feedback.id)
            return str(feedback.id)
            
        except Exception as e:
            logger.error("Failed to submit prediction feedback", error=str(e), exc_info=True)
            raise
    
    # Helper methods
    async def _generate_recommendations(
        self,
        risk_score: float,
        patient_data: Dict[str, Any]
    ) -> List[str]:
        """Generate clinical recommendations based on risk score and patient data"""
        recommendations = []
        
        try:
            # Risk-based recommendations
            if risk_score >= 0.7:  # High risk
                recommendations.extend([
                    "Immediate cardiovascular evaluation recommended",
                    "Consider cardiology referral",
                    "Intensive risk factor modification",
                    "Close monitoring and follow-up"
                ])
            elif risk_score >= 0.3:  # Medium risk
                recommendations.extend([
                    "Regular cardiovascular monitoring",
                    "Lifestyle modification counseling",
                    "Risk factor optimization"
                ])
            else:  # Low risk
                recommendations.extend([
                    "Continue current preventive care",
                    "Annual cardiovascular risk assessment"
                ])
            
            # Condition-specific recommendations
            if patient_data.get("has_hypertension"):
                recommendations.append("Blood pressure management and monitoring")
            
            if patient_data.get("has_diabetes"):
                recommendations.append("Diabetes management and glucose control")
            
            if patient_data.get("smoking_status") == "current":
                recommendations.append("Smoking cessation counseling and support")
            
            # BMI-based recommendations
            bmi = patient_data.get("bmi")
            if bmi and bmi >= 30:
                recommendations.append("Weight management and nutritional counseling")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error("Failed to generate recommendations", error=str(e))
            return ["Consult with healthcare provider for personalized recommendations"]
    
    async def _identify_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify key risk factors from patient data"""
        risk_factors = []
        
        try:
            # Age-based risk
            age = patient_data.get("age", 0)
            if age >= 75:
                risk_factors.append("Advanced age (≥75 years)")
            elif age >= 65:
                risk_factors.append("Age ≥65 years")
            
            # Medical conditions
            conditions = {
                "has_hypertension": "Hypertension",
                "has_diabetes": "Diabetes mellitus",
                "has_heart_disease": "Existing heart disease",
                "has_stroke_history": "Previous stroke",
                "has_heart_attack_history": "Previous myocardial infarction"
            }
            
            for condition_key, condition_name in conditions.items():
                if patient_data.get(condition_key):
                    risk_factors.append(condition_name)
            
            # Lifestyle factors
            if patient_data.get("smoking_status") == "current":
                risk_factors.append("Current smoking")
            
            # Clinical values
            systolic_bp = patient_data.get("systolic_bp")
            if systolic_bp and systolic_bp >= 140:
                risk_factors.append("Elevated blood pressure")
            
            cholesterol = patient_data.get("total_cholesterol")
            if cholesterol and cholesterol >= 240:
                risk_factors.append("High cholesterol")
            
            # BMI
            bmi = patient_data.get("bmi")
            if bmi and bmi >= 30:
                risk_factors.append("Obesity")
            
            return risk_factors
            
        except Exception as e:
            logger.error("Failed to identify risk factors", error=str(e))
            return []
    
    async def _calculate_batch_summary(self, predictions: List[BatchPredictionItem]) -> Dict[str, Any]:
        """Calculate summary statistics for batch predictions"""
        try:
            successful_predictions = [p for p in predictions if p.prediction is not None]
            
            if not successful_predictions:
                return {"message": "No successful predictions to summarize"}
            
            risk_scores = [p.prediction.risk_score for p in successful_predictions]
            risk_categories = [p.prediction.risk_category for p in successful_predictions]
            
            summary = {
                "risk_distribution": {
                    "low": risk_categories.count("low"),
                    "medium": risk_categories.count("medium"),
                    "high": risk_categories.count("high")
                },
                "risk_statistics": {
                    "average_risk_score": round(np.mean(risk_scores), 4),
                    "median_risk_score": round(np.median(risk_scores), 4),
                    "std_risk_score": round(np.std(risk_scores), 4),
                    "min_risk_score": round(np.min(risk_scores), 4),
                    "max_risk_score": round(np.max(risk_scores), 4)
                },
                "high_risk_percentage": round((risk_categories.count("high") / len(risk_categories)) * 100, 1),
                "successful_predictions": len(successful_predictions),
                "total_predictions": len(predictions)
            }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to calculate batch summary", error=str(e))
            return {}
    
    async def _calculate_history_summary(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """Calculate summary statistics for prediction history"""
        try:
            if not predictions:
                return {}
            
            risk_scores = [p.risk_score for p in predictions]
            risk_categories = [p.risk_category for p in predictions]
            
            # Model version usage
            model_versions = {}
            for prediction in predictions:
                version = prediction.model_version
                if version not in model_versions:
                    model_versions[version] = 0
                model_versions[version] += 1
            
            # Feedback statistics
            feedback_stats = {
                "predictions_with_feedback": len([p for p in predictions if p.actual_outcome is not None]),
                "positive_outcomes": len([p for p in predictions if p.actual_outcome is True]),
                "negative_outcomes": len([p for p in predictions if p.actual_outcome is False])
            }
            
            summary = {
                "total_predictions": len(predictions),
                "risk_distribution": {
                    "low": risk_categories.count("low"),
                    "medium": risk_categories.count("medium"),
                    "high": risk_categories.count("high")
                },
                "average_risk_score": round(np.mean(risk_scores), 4),
                "model_usage": model_versions,
                "feedback_statistics": feedback_stats,
                "time_range": {
                    "earliest": min(p.created_at for p in predictions).isoformat(),
                    "latest": max(p.created_at for p in predictions).isoformat()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to calculate history summary", error=str(e))
            return {}
    
    async def _calculate_prediction_trends(
        self,
        predictions: List[Prediction],
        timeframe: str
    ) -> Dict[str, Any]:
        """Calculate prediction trends over time"""
        try:
            if not predictions:
                return {}
            
            # Group predictions by date
            daily_stats = {}
            
            for prediction in predictions:
                date_key = prediction.created_at.date().isoformat()
                
                if date_key not in daily_stats:
                    daily_stats[date_key] = {
                        "total": 0,
                        "low_risk": 0,
                        "medium_risk": 0,
                        "high_risk": 0,
                        "avg_risk_score": []
                    }
                
                daily_stats[date_key]["total"] += 1
                daily_stats[date_key][f"{prediction.risk_category}_risk"] += 1
                daily_stats[date_key]["avg_risk_score"].append(prediction.risk_score)
            
            # Calculate averages
            for date_key in daily_stats:
                scores = daily_stats[date_key]["avg_risk_score"]
                daily_stats[date_key]["avg_risk_score"] = round(np.mean(scores), 4)
            
            return {
                "daily_statistics": daily_stats,
                "trend_analysis": "Trends calculated for {} timeframe".format(timeframe)
            }
            
        except Exception as e:
            logger.error("Failed to calculate prediction trends", error=str(e))
            return {}
    
    async def _calculate_performance_metrics(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, Any]:
        """Calculate performance metrics based on feedback"""
        try:
            predictions_with_outcomes = [
                p for p in predictions 
                if p.actual_outcome is not None
            ]
            
            if not predictions_with_outcomes:
                return {"message": "No outcome data available for performance calculation"}
            
            # Calculate accuracy metrics
            correct_predictions = 0
            for prediction in predictions_with_outcomes:
                # Simplified accuracy: high risk prediction + positive outcome = correct
                predicted_high_risk = prediction.risk_category == "high"
                actual_positive_outcome = prediction.actual_outcome
                
                if predicted_high_risk == actual_positive_outcome:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(predictions_with_outcomes)
            
            performance = {
                "predictions_with_outcomes": len(predictions_with_outcomes),
                "accuracy": round(accuracy, 4),
                "feedback_rate": round(len(predictions_with_outcomes) / len(predictions), 4),
                "positive_outcomes": len([p for p in predictions_with_outcomes if p.actual_outcome]),
                "negative_outcomes": len([p for p in predictions_with_outcomes if not p.actual_outcome])
            }
            
            return performance
            
        except Exception as e:
            logger.error("Failed to calculate performance metrics", error=str(e))
            return {}


# Export the prediction service
__all__ = ["PredictionService"]