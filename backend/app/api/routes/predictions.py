from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    BackgroundTasks,
    UploadFile,
    File,
    Query,
    Body
)
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import pandas as pd
from io import StringIO, BytesIO

from app.core.database import get_db
from app.core.security import get_current_user_from_token, get_current_active_user
from app.api.dependencies import get_rate_limiter, get_correlation_id, validate_request_size
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelPerformanceResponse,
    PredictionHistoryResponse
)
from app.services.prediction_service import PredictionService
from app.services.ml_service import MLService
from app.utils.data_preprocessing import DataValidator, DataProcessor

logger = structlog.get_logger(__name__)
router = APIRouter()

# Initialize services
prediction_service = PredictionService()
ml_service = MLService()
data_validator = DataValidator()
data_processor = DataProcessor()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Cardiovascular Disease Risk Prediction",
    description="Predict cardiovascular disease hospitalization risk for a single patient"
)
async def predict_cardiovascular_risk(
    prediction_request: PredictionRequest,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token),
    _: None = Depends(get_rate_limiter)
) -> PredictionResponse:
    """
    Predict cardiovascular disease hospitalization risk for a single patient.
    
    **Parameters:**
    - **patient_data**: Patient demographics and health indicators
    - **model_version**: Optional model version to use (defaults to latest)
    - **include_confidence**: Whether to include confidence intervals
    - **include_features**: Whether to include feature importance
    
    **Returns:**
    - **risk_score**: Predicted hospitalization risk (0-1)
    - **risk_category**: Risk category (Low/Medium/High)
    - **confidence_interval**: 95% confidence interval (if requested)
    - **feature_importance**: Top contributing factors (if requested)
    """
    
    try:
        logger.info(
            "Single prediction requested",
            correlation_id=correlation_id,
            user_id=current_user.get("user_id") if current_user else "anonymous",
            model_version=prediction_request.model_version
        )
        
        # Validate input data
        validation_result = await data_validator.validate_patient_data(
            prediction_request.patient_data
        )
        
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Invalid patient data",
                    "validation_errors": validation_result.errors,
                    "correlation_id": correlation_id
                }
            )
        
        # Make prediction
        prediction_result = await prediction_service.predict_single(
            patient_data=prediction_request.patient_data,
            model_version=prediction_request.model_version,
            include_confidence=prediction_request.include_confidence,
            include_features=prediction_request.include_features
        )
        
        # Store prediction in database for audit trail
        background_tasks.add_task(
            prediction_service.store_prediction,
            prediction_request=prediction_request,
            prediction_result=prediction_result,
            user_id=current_user.get("user_id") if current_user else None,
            correlation_id=correlation_id,
            db=db
        )
        
        logger.info(
            "Single prediction completed",
            correlation_id=correlation_id,
            risk_score=prediction_result.risk_score,
            risk_category=prediction_result.risk_category
        )
        
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Single prediction failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Prediction failed",
                "message": "An error occurred while processing the prediction",
                "correlation_id": correlation_id
            }
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Cardiovascular Disease Risk Predictions",
    description="Predict cardiovascular disease risk for multiple patients"
)
async def predict_batch_cardiovascular_risk(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token),
    _: None = Depends(get_rate_limiter)
) -> BatchPredictionResponse:
    """
    Predict cardiovascular disease risk for multiple patients in batch.
    
    **Parameters:**
    - **patients_data**: List of patient data objects
    - **model_version**: Optional model version to use
    - **include_confidence**: Whether to include confidence intervals
    - **parallel_processing**: Enable parallel processing for large batches
    
    **Returns:**
    - **predictions**: List of prediction results
    - **summary**: Batch processing summary statistics
    - **failed_predictions**: Any predictions that failed with reasons
    """
    
    try:
        logger.info(
            "Batch prediction requested",
            correlation_id=correlation_id,
            batch_size=len(batch_request.patients_data),
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        # Validate batch size
        max_batch_size = 1000  # Configurable limit
        if len(batch_request.patients_data) > max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "Batch size too large",
                    "max_batch_size": max_batch_size,
                    "provided_size": len(batch_request.patients_data),
                    "correlation_id": correlation_id
                }
            )
        
        # Process batch predictions
        batch_result = await prediction_service.predict_batch(
            patients_data=batch_request.patients_data,
            model_version=batch_request.model_version,
            include_confidence=batch_request.include_confidence,
            parallel_processing=batch_request.parallel_processing
        )
        
        # Store batch results asynchronously
        background_tasks.add_task(
            prediction_service.store_batch_predictions,
            batch_request=batch_request,
            batch_result=batch_result,
            user_id=current_user.get("user_id") if current_user else None,
            correlation_id=correlation_id,
            db=db
        )
        
        logger.info(
            "Batch prediction completed",
            correlation_id=correlation_id,
            successful_predictions=len(batch_result.predictions),
            failed_predictions=len(batch_result.failed_predictions)
        )
        
        return batch_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Batch prediction failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Batch prediction failed",
                "message": "An error occurred while processing the batch predictions",
                "correlation_id": correlation_id
            }
        )


@router.post(
    "/predict/upload",
    summary="Predict from Uploaded CSV File",
    description="Upload a CSV file and get predictions for all patients in the file"
)
async def predict_from_csv_upload(
    file: UploadFile = File(..., description="CSV file with patient data"),
    model_version: Optional[str] = Query(None, description="Model version to use"),
    include_confidence: bool = Query(False, description="Include confidence intervals"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token),
    _: None = Depends(validate_request_size)
) -> StreamingResponse:
    """
    Upload a CSV file and get cardiovascular disease risk predictions.
    
    **File Format Requirements:**
    - CSV format with headers
    - Required columns: age, gender, prior_conditions, etc.
    - Maximum file size: 50MB
    - Maximum rows: 10,000
    
    **Returns:**
    - **CSV file** with original data plus prediction columns
    - **risk_score**: Predicted risk score (0-1)
    - **risk_category**: Risk categorization
    - **confidence_lower/upper**: Confidence intervals (if requested)
    """
    
    try:
        logger.info(
            "CSV upload prediction requested",
            correlation_id=correlation_id,
            filename=file.filename,
            content_type=file.content_type,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        # Validate file
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid file format",
                    "message": "Only CSV files are supported",
                    "correlation_id": correlation_id
                }
            )
        
        # Read and validate CSV content
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "File too large",
                    "max_size": "50MB",
                    "correlation_id": correlation_id
                }
            )
        
        # Parse CSV
        try:
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid CSV format",
                    "message": f"Could not parse CSV file: {str(e)}",
                    "correlation_id": correlation_id
                }
            )
        
        # Validate row count
        if len(df) > 10000:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "Too many rows",
                    "max_rows": 10000,
                    "provided_rows": len(df),
                    "correlation_id": correlation_id
                }
            )
        
        # Validate required columns
        validation_result = await data_validator.validate_csv_structure(df)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Invalid CSV structure",
                    "validation_errors": validation_result.errors,
                    "required_columns": validation_result.required_columns,
                    "correlation_id": correlation_id
                }
            )
        
        # Process predictions
        predictions_df = await prediction_service.predict_from_dataframe(
            df=df,
            model_version=model_version,
            include_confidence=include_confidence
        )
        
        # Convert back to CSV for download
        output = StringIO()
        predictions_df.to_csv(output, index=False)
        output.seek(0)
        
        # Store batch operation record
        background_tasks.add_task(
            prediction_service.store_csv_prediction_job,
            original_filename=file.filename,
            row_count=len(df),
            user_id=current_user.get("user_id") if current_user else None,
            correlation_id=correlation_id,
            db=db
        )
        
        logger.info(
            "CSV prediction completed",
            correlation_id=correlation_id,
            processed_rows=len(predictions_df)
        )
        
        # Return CSV as streaming response
        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{file.filename}",
                "X-Correlation-ID": correlation_id
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "CSV prediction failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "CSV prediction failed",
                "message": "An error occurred while processing the CSV file",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/models",
    summary="List Available Models",
    description="Get information about available prediction models"
)
async def list_available_models(
    correlation_id: str = Depends(get_correlation_id),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token)
) -> Dict[str, Any]:
    """
    List all available prediction models with their metadata.
    
    **Returns:**
    - **models**: List of available models with versions and performance metrics
    - **default_model**: Current default model information
    - **model_types**: Supported model types
    """
    
    try:
        logger.info(
            "Model list requested",
            correlation_id=correlation_id,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        models_info = await ml_service.get_available_models()
        
        return {
            "models": models_info.models,
            "default_model": models_info.default_model,
            "model_types": models_info.supported_types,
            "total_models": len(models_info.models),
            "last_updated": models_info.last_updated.isoformat() if models_info.last_updated else None
        }
        
    except Exception as e:
        logger.error(
            "Failed to list models",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve model information",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/models/{model_version}/performance",
    response_model=ModelPerformanceResponse,
    summary="Get Model Performance Metrics",
    description="Retrieve detailed performance metrics for a specific model version"
)
async def get_model_performance(
    model_version: str,
    correlation_id: str = Depends(get_correlation_id),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token)
) -> ModelPerformanceResponse:
    """
    Get detailed performance metrics for a specific model version.
    
    **Parameters:**
    - **model_version**: Version of the model to get performance for
    
    **Returns:**
    - **metrics**: Comprehensive performance metrics (R², MSE, MAE, etc.)
    - **cross_validation**: Cross-validation results
    - **feature_importance**: Feature importance rankings
    - **model_info**: Model metadata and training information
    """
    
    try:
        logger.info(
            "Model performance requested",
            correlation_id=correlation_id,
            model_version=model_version,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        performance_data = await ml_service.get_model_performance(model_version)
        
        if not performance_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Model not found",
                    "model_version": model_version,
                    "correlation_id": correlation_id
                }
            )
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get model performance",
            correlation_id=correlation_id,
            model_version=model_version,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve model performance",
                "correlation_id": correlation_id
            }
        )


@router.post(
    "/models/retrain",
    summary="Trigger Model Retraining",
    description="Manually trigger model retraining with latest data"
)
async def trigger_model_retraining(
    background_tasks: BackgroundTasks,
    retrain_params: Dict[str, Any] = Body(
        default={},
        description="Optional retraining parameters"
    ),
    correlation_id: str = Depends(get_correlation_id),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Manually trigger model retraining process.
    
    **Note:** This endpoint requires authentication and appropriate permissions.
    
    **Parameters:**
    - **retrain_params**: Optional parameters for retraining process
    - **force_retrain**: Force retraining even if current model performance is good
    - **use_latest_data**: Whether to fetch latest data from CDC
    
    **Returns:**
    - **job_id**: Training job identifier for tracking
    - **estimated_duration**: Estimated training time
    - **status**: Current training status
    """
    
    try:
        # Check user permissions (implement role-based access)
        user_roles = current_user.get("roles", [])
        if "admin" not in user_roles and "ml_engineer" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": "Model retraining requires admin or ML engineer role",
                    "correlation_id": correlation_id
                }
            )
        
        logger.info(
            "Model retraining triggered",
            correlation_id=correlation_id,
            user_id=current_user["user_id"],
            retrain_params=retrain_params
        )
        
        # Start retraining job in background
        job_id = await ml_service.start_retraining_job(
            user_id=current_user["user_id"],
            parameters=retrain_params,
            correlation_id=correlation_id
        )
        
        # Log job start
        background_tasks.add_task(
            ml_service.log_retraining_job,
            job_id=job_id,
            user_id=current_user["user_id"],
            correlation_id=correlation_id,
            db=db
        )
        
        return {
            "message": "Model retraining started",
            "job_id": job_id,
            "estimated_duration": "15-30 minutes",
            "status": "queued",
            "correlation_id": correlation_id,
            "check_status_url": f"/api/v1/models/retrain/{job_id}/status"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to trigger retraining",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to start retraining",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/models/retrain/{job_id}/status",
    summary="Check Retraining Job Status",
    description="Check the status of a model retraining job"
)
async def get_retraining_job_status(
    job_id: str,
    correlation_id: str = Depends(get_correlation_id),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Check the status of a model retraining job.
    
    **Parameters:**
    - **job_id**: The retraining job identifier
    
    **Returns:**
    - **status**: Current job status (queued, running, completed, failed)
    - **progress**: Training progress percentage
    - **metrics**: Current training metrics (if available)
    - **estimated_completion**: Estimated completion time
    """
    
    try:
        logger.info(
            "Retraining status requested",
            correlation_id=correlation_id,
            job_id=job_id,
            user_id=current_user["user_id"]
        )
        
        job_status = await ml_service.get_retraining_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Job not found",
                    "job_id": job_id,
                    "correlation_id": correlation_id
                }
            )
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get retraining status",
            correlation_id=correlation_id,
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve job status",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/history",
    response_model=PredictionHistoryResponse,
    summary="Get Prediction History",
    description="Retrieve prediction history for the current user or system"
)
async def get_prediction_history(
    limit: int = Query(50, ge=1, le=1000, description="Number of predictions to retrieve"),
    offset: int = Query(0, ge=0, description="Number of predictions to skip"),
    start_date: Optional[datetime] = Query(None, description="Filter predictions from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter predictions until this date"),
    risk_category: Optional[str] = Query(None, regex="^(low|medium|high)$", description="Filter by risk category"),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token)
) -> PredictionHistoryResponse:
    """
    Retrieve prediction history with filtering and pagination.
    
    **Parameters:**
    - **limit**: Maximum number of predictions to return (1-1000)
    - **offset**: Number of predictions to skip for pagination
    - **start_date**: Filter predictions from this date onwards
    - **end_date**: Filter predictions until this date
    - **risk_category**: Filter by risk category (low/medium/high)
    
    **Returns:**
    - **predictions**: List of historical predictions
    - **total_count**: Total number of predictions matching filters
    - **summary**: Summary statistics of predictions
    """
    
    try:
        logger.info(
            "Prediction history requested",
            correlation_id=correlation_id,
            limit=limit,
            offset=offset,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        # Get prediction history
        history_result = await prediction_service.get_prediction_history(
            user_id=current_user.get("user_id") if current_user else None,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            risk_category=risk_category,
            db=db
        )
        
        return history_result
        
    except Exception as e:
        logger.error(
            "Failed to get prediction history",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve prediction history",
                "correlation_id": correlation_id
            }
        )


@router.delete(
    "/predictions/{prediction_id}",
    summary="Delete Prediction Record",
    description="Delete a specific prediction record (admin only)"
)
async def delete_prediction(
    prediction_id: str,
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, str]:
    """
    Delete a specific prediction record.
    
    **Note:** This endpoint requires admin permissions.
    
    **Parameters:**
    - **prediction_id**: The ID of the prediction to delete
    
    **Returns:**
    - **message**: Confirmation of deletion
    """
    
    try:
        # Check admin permissions
        user_roles = current_user.get("roles", [])
        if "admin" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": "Prediction deletion requires admin role",
                    "correlation_id": correlation_id
                }
            )
        
        logger.info(
            "Prediction deletion requested",
            correlation_id=correlation_id,
            prediction_id=prediction_id,
            user_id=current_user["user_id"]
        )
        
        # Delete prediction
        deleted = await prediction_service.delete_prediction(
            prediction_id=prediction_id,
            user_id=current_user["user_id"],
            db=db
        )
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Prediction not found",
                    "prediction_id": prediction_id,
                    "correlation_id": correlation_id
                }
            )
        
        logger.info(
            "Prediction deleted successfully",
            correlation_id=correlation_id,
            prediction_id=prediction_id
        )
        
        return {
            "message": "Prediction deleted successfully",
            "prediction_id": prediction_id,
            "correlation_id": correlation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete prediction",
            correlation_id=correlation_id,
            prediction_id=prediction_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to delete prediction",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/stats",
    summary="Get Prediction Statistics",
    description="Get overall prediction statistics and insights"
)
async def get_prediction_statistics(
    timeframe: str = Query("30d", regex="^(1d|7d|30d|90d|1y|all)$", description="Statistics timeframe"),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_token)
) -> Dict[str, Any]:
    """
    Get comprehensive prediction statistics and insights.
    
    **Parameters:**
    - **timeframe**: Time period for statistics (1d, 7d, 30d, 90d, 1y, all)
    
    **Returns:**
    - **total_predictions**: Total number of predictions made
    - **risk_distribution**: Distribution of risk categories
    - **model_usage**: Usage statistics by model version
    - **trends**: Prediction trends over time
    - **performance_metrics**: Aggregated model performance
    """
    
    try:
        logger.info(
            "Prediction statistics requested",
            correlation_id=correlation_id,
            timeframe=timeframe,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        stats = await prediction_service.get_prediction_statistics(
            timeframe=timeframe,
            user_id=current_user.get("user_id") if current_user else None,
            db=db
        )
        
        return stats
        
    except Exception as e:
        logger.error(
            "Failed to get prediction statistics",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve prediction statistics",
                "correlation_id": correlation_id
            }
        )


# Export the router
__all__ = ["router"]