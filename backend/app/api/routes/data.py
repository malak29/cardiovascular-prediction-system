from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    BackgroundTasks,
    UploadFile,
    File,
    Query,
    Body,
    Form
)
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import pandas as pd
from io import StringIO

from app.core.database import get_db
from app.core.security import get_current_active_user, generate_secure_filename
from app.api.dependencies import get_correlation_id, validate_file_upload, get_rate_limiter
from app.schemas.data import (
    DataValidationResponse,
    DataSyncResponse,
    DatasetInfoResponse,
    DataProcessingResponse
)
from app.services.data_service import DataService
from app.utils.data_preprocessing import DataValidator, DataProcessor
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()

# Initialize services
data_service = DataService()
data_validator = DataValidator()
data_processor = DataProcessor()


@router.post(
    "/upload",
    summary="Upload Dataset",
    description="Upload a new dataset for training or validation"
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file (CSV, JSON, or Parquet)"),
    dataset_name: str = Form(..., description="Name for the dataset"),
    dataset_type: str = Form("training", regex="^(training|validation|test)$"),
    description: Optional[str] = Form(None, description="Dataset description"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    _: None = Depends(validate_file_upload)
) -> Dict[str, Any]:
    """
    Upload a new dataset for the cardiovascular prediction system.
    
    **Parameters:**
    - **file**: Dataset file (supports CSV, JSON, Parquet formats)
    - **dataset_name**: Unique name for the dataset
    - **dataset_type**: Type of dataset (training, validation, test)
    - **description**: Optional description of the dataset
    
    **Returns:**
    - **dataset_id**: Unique identifier for the uploaded dataset
    - **validation_results**: Data validation results
    - **processing_status**: Initial processing status
    """
    
    try:
        logger.info(
            "Dataset upload started",
            correlation_id=correlation_id,
            filename=file.filename,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            user_id=current_user["user_id"]
        )
        
        # Validate file format
        allowed_extensions = get_settings().ALLOWED_EXTENSIONS
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Unsupported file format",
                    "allowed_formats": allowed_extensions,
                    "provided_format": file_extension,
                    "correlation_id": correlation_id
                }
            )
        
        # Generate secure filename
        secure_filename = generate_secure_filename(file.filename)
        
        # Read and validate file content
        content = await file.read()
        
        # Parse the data based on file type
        if file_extension == 'csv':
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        elif file_extension == 'json':
            df = pd.read_json(StringIO(content.decode('utf-8')))
        elif file_extension in ['parquet', 'pq']:
            df = pd.read_parquet(StringIO(content.decode('utf-8')))
        
        # Validate dataset structure
        validation_result = await data_validator.validate_dataset_structure(
            df=df,
            dataset_type=dataset_type
        )
        
        # Save dataset
        dataset_info = await data_service.save_dataset(
            df=df,
            filename=secure_filename,
            original_filename=file.filename,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            description=description,
            user_id=current_user["user_id"],
            validation_result=validation_result,
            db=db
        )
        
        # Process dataset in background if validation passed
        if validation_result.is_valid:
            background_tasks.add_task(
                data_service.process_dataset,
                dataset_id=dataset_info["dataset_id"],
                correlation_id=correlation_id,
                db=db
            )
        
        logger.info(
            "Dataset upload completed",
            correlation_id=correlation_id,
            dataset_id=dataset_info["dataset_id"],
            rows=len(df),
            columns=len(df.columns)
        )
        
        return {
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset_info["dataset_id"],
            "dataset_name": dataset_name,
            "filename": secure_filename,
            "rows": len(df),
            "columns": len(df.columns),
            "validation_results": validation_result.dict(),
            "processing_status": "queued" if validation_result.is_valid else "validation_failed",
            "correlation_id": correlation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Dataset upload failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Dataset upload failed",
                "message": "An error occurred while uploading the dataset",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/datasets",
    response_model=List[DatasetInfoResponse],
    summary="List Datasets",
    description="List all available datasets with metadata"
)
async def list_datasets(
    dataset_type: Optional[str] = Query(None, regex="^(training|validation|test)$"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_active_user)
) -> List[DatasetInfoResponse]:
    """
    List all available datasets with filtering and pagination.
    
    **Parameters:**
    - **dataset_type**: Filter by dataset type (training, validation, test)
    - **limit**: Maximum number of datasets to return
    - **offset**: Number of datasets to skip for pagination
    
    **Returns:**
    - **List of datasets** with metadata including size, validation status, processing status
    """
    
    try:
        logger.info(
            "Dataset list requested",
            correlation_id=correlation_id,
            dataset_type=dataset_type,
            limit=limit,
            offset=offset,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        datasets = await data_service.list_datasets(
            dataset_type=dataset_type,
            limit=limit,
            offset=offset,
            user_id=current_user.get("user_id") if current_user else None,
            db=db
        )
        
        return datasets
        
    except Exception as e:
        logger.error(
            "Failed to list datasets",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve datasets",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetInfoResponse,
    summary="Get Dataset Details",
    description="Get detailed information about a specific dataset"
)
async def get_dataset_details(
    dataset_id: str,
    include_sample: bool = Query(False, description="Include data sample in response"),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_active_user)
) -> DatasetInfoResponse:
    """
    Get detailed information about a specific dataset.
    
    **Parameters:**
    - **dataset_id**: Unique identifier of the dataset
    - **include_sample**: Whether to include a sample of the data
    
    **Returns:**
    - **Dataset metadata** including schema, statistics, validation results
    """
    
    try:
        logger.info(
            "Dataset details requested",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            include_sample=include_sample,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        dataset_info = await data_service.get_dataset_details(
            dataset_id=dataset_id,
            include_sample=include_sample,
            user_id=current_user.get("user_id") if current_user else None,
            db=db
        )
        
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Dataset not found",
                    "dataset_id": dataset_id,
                    "correlation_id": correlation_id
                }
            )
        
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get dataset details",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve dataset details",
                "correlation_id": correlation_id
            }
        )


@router.post(
    "/sync/cdc",
    response_model=DataSyncResponse,
    summary="Sync CDC Data",
    description="Synchronize data from CDC cardiovascular disease dataset"
)
async def sync_cdc_data(
    background_tasks: BackgroundTasks,
    force_update: bool = Body(False, description="Force update even if data is recent"),
    data_range: Optional[Dict[str, str]] = Body(None, description="Date range for data sync"),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    _: None = Depends(get_rate_limiter)
) -> DataSyncResponse:
    """
    Synchronize cardiovascular disease data from CDC.
    
    **Note:** This endpoint requires authentication and appropriate permissions.
    
    **Parameters:**
    - **force_update**: Force data sync even if recent data exists
    - **data_range**: Optional date range for data synchronization
    
    **Returns:**
    - **sync_job_id**: Job identifier for tracking sync progress
    - **estimated_duration**: Estimated sync duration
    - **data_sources**: List of data sources that will be synchronized
    """
    
    try:
        # Check user permissions
        user_roles = current_user.get("roles", [])
        if "admin" not in user_roles and "data_manager" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": "Data synchronization requires admin or data manager role",
                    "correlation_id": correlation_id
                }
            )
        
        logger.info(
            "CDC data sync requested",
            correlation_id=correlation_id,
            user_id=current_user["user_id"],
            force_update=force_update,
            data_range=data_range
        )
        
        # Check if recent sync exists and force_update is False
        if not force_update:
            last_sync = await data_service.get_last_sync_info("cdc", db)
            if last_sync and last_sync.get("completed_at"):
                last_sync_time = datetime.fromisoformat(last_sync["completed_at"])
                if datetime.utcnow() - last_sync_time < timedelta(hours=24):
                    return DataSyncResponse(
                        sync_job_id=last_sync["job_id"],
                        status="completed",
                        message="Recent data sync already exists. Use force_update=true to override.",
                        last_sync_time=last_sync_time,
                        correlation_id=correlation_id
                    )
        
        # Start data synchronization job
        sync_job = await data_service.start_cdc_sync_job(
            user_id=current_user["user_id"],
            force_update=force_update,
            data_range=data_range,
            correlation_id=correlation_id
        )
        
        # Process sync in background
        background_tasks.add_task(
            data_service.execute_cdc_sync,
            job_id=sync_job["job_id"],
            correlation_id=correlation_id,
            db=db
        )
        
        logger.info(
            "CDC data sync started",
            correlation_id=correlation_id,
            job_id=sync_job["job_id"]
        )
        
        return DataSyncResponse(
            sync_job_id=sync_job["job_id"],
            status="started",
            message="CDC data synchronization started",
            estimated_duration="10-20 minutes",
            data_sources=["CDC Cardiovascular Disease Dataset"],
            correlation_id=correlation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "CDC data sync failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Data synchronization failed",
                "message": "An error occurred while starting data synchronization",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/sync/{job_id}/status",
    summary="Get Sync Job Status",
    description="Check the status of a data synchronization job"
)
async def get_sync_job_status(
    job_id: str,
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Check the status of a data synchronization job.
    
    **Parameters:**
    - **job_id**: The synchronization job identifier
    
    **Returns:**
    - **status**: Current job status
    - **progress**: Synchronization progress
    - **records_processed**: Number of records processed
    - **estimated_completion**: Estimated completion time
    """
    
    try:
        logger.info(
            "Sync job status requested",
            correlation_id=correlation_id,
            job_id=job_id,
            user_id=current_user["user_id"]
        )
        
        job_status = await data_service.get_sync_job_status(job_id, db)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Sync job not found",
                    "job_id": job_id,
                    "correlation_id": correlation_id
                }
            )
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get sync job status",
            correlation_id=correlation_id,
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve sync job status",
                "correlation_id": correlation_id
            }
        )


@router.post(
    "/validate",
    response_model=DataValidationResponse,
    summary="Validate Dataset",
    description="Validate a dataset against cardiovascular prediction requirements"
)
async def validate_dataset(
    file: UploadFile = File(..., description="Dataset file to validate"),
    validation_type: str = Query("full", regex="^(quick|full|schema_only)$"),
    correlation_id: str = Depends(get_correlation_id),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_active_user),
    _: None = Depends(validate_file_upload)
) -> DataValidationResponse:
    """
    Validate a dataset against cardiovascular prediction system requirements.
    
    **Parameters:**
    - **file**: Dataset file to validate
    - **validation_type**: Type of validation (quick, full, schema_only)
    
    **Returns:**
    - **validation_results**: Comprehensive validation results
    - **data_quality_score**: Overall data quality score (0-100)
    - **recommendations**: Data improvement recommendations
    """
    
    try:
        logger.info(
            "Dataset validation requested",
            correlation_id=correlation_id,
            filename=file.filename,
            validation_type=validation_type,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        # Read file content
        content = await file.read()
        
        # Parse based on file extension
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        elif file_extension == 'json':
            df = pd.read_json(StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Unsupported file format for validation",
                    "supported_formats": ["csv", "json"],
                    "correlation_id": correlation_id
                }
            )
        
        # Perform validation based on type
        if validation_type == "quick":
            validation_result = await data_validator.quick_validate(df)
        elif validation_type == "schema_only":
            validation_result = await data_validator.validate_schema_only(df)
        else:  # full validation
            validation_result = await data_validator.comprehensive_validate(df)
        
        logger.info(
            "Dataset validation completed",
            correlation_id=correlation_id,
            is_valid=validation_result.is_valid,
            quality_score=validation_result.quality_score
        )
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Dataset validation failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Dataset validation failed",
                "message": "An error occurred while validating the dataset",
                "correlation_id": correlation_id
            }
        )


@router.post(
    "/process/{dataset_id}",
    response_model=DataProcessingResponse,
    summary="Process Dataset",
    description="Process a dataset for training or prediction use"
)
async def process_dataset(
    dataset_id: str,
    processing_params: Dict[str, Any] = Body(
        default={},
        description="Processing parameters and options"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> DataProcessingResponse:
    """
    Process a dataset for training or prediction use.
    
    **Parameters:**
    - **dataset_id**: Identifier of the dataset to process
    - **processing_params**: Custom processing parameters
    
    **Returns:**
    - **processing_job_id**: Job identifier for tracking
    - **processing_status**: Current processing status
    - **estimated_duration**: Estimated processing time
    """
    
    try:
        logger.info(
            "Dataset processing requested",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            user_id=current_user["user_id"],
            processing_params=processing_params
        )
        
        # Verify dataset exists and user has access
        dataset_info = await data_service.get_dataset_details(
            dataset_id=dataset_id,
            user_id=current_user["user_id"],
            db=db
        )
        
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Dataset not found",
                    "dataset_id": dataset_id,
                    "correlation_id": correlation_id
                }
            )
        
        # Start processing job
        processing_job = await data_service.start_processing_job(
            dataset_id=dataset_id,
            processing_params=processing_params,
            user_id=current_user["user_id"],
            correlation_id=correlation_id
        )
        
        # Execute processing in background
        background_tasks.add_task(
            data_service.execute_data_processing,
            job_id=processing_job["job_id"],
            dataset_id=dataset_id,
            processing_params=processing_params,
            correlation_id=correlation_id,
            db=db
        )
        
        logger.info(
            "Dataset processing started",
            correlation_id=correlation_id,
            processing_job_id=processing_job["job_id"]
        )
        
        return DataProcessingResponse(
            processing_job_id=processing_job["job_id"],
            dataset_id=dataset_id,
            status="started",
            message="Dataset processing started",
            estimated_duration="5-15 minutes",
            correlation_id=correlation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Dataset processing failed",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Dataset processing failed",
                "correlation_id": correlation_id
            }
        )


@router.delete(
    "/datasets/{dataset_id}",
    summary="Delete Dataset",
    description="Delete a dataset and all associated files"
)
async def delete_dataset(
    dataset_id: str,
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, str]:
    """
    Delete a dataset and all associated files.
    
    **Note:** This action is irreversible.
    
    **Parameters:**
    - **dataset_id**: Identifier of the dataset to delete
    
    **Returns:**
    - **message**: Confirmation of deletion
    """
    
    try:
        logger.info(
            "Dataset deletion requested",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            user_id=current_user["user_id"]
        )
        
        # Verify dataset exists and user has permission to delete
        dataset_info = await data_service.get_dataset_details(
            dataset_id=dataset_id,
            user_id=current_user["user_id"],
            db=db
        )
        
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Dataset not found",
                    "dataset_id": dataset_id,
                    "correlation_id": correlation_id
                }
            )
        
        # Check if user owns the dataset or is admin
        user_roles = current_user.get("roles", [])
        if dataset_info.owner_id != current_user["user_id"] and "admin" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": "You can only delete your own datasets",
                    "correlation_id": correlation_id
                }
            )
        
        # Delete dataset
        deleted = await data_service.delete_dataset(
            dataset_id=dataset_id,
            user_id=current_user["user_id"],
            db=db
        )
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to delete dataset",
                    "correlation_id": correlation_id
                }
            )
        
        logger.info(
            "Dataset deleted successfully",
            correlation_id=correlation_id,
            dataset_id=dataset_id
        )
        
        return {
            "message": "Dataset deleted successfully",
            "dataset_id": dataset_id,
            "correlation_id": correlation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Dataset deletion failed",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Dataset deletion failed",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/export/{dataset_id}",
    summary="Export Dataset",
    description="Export a processed dataset in specified format"
)
async def export_dataset(
    dataset_id: str,
    format: str = Query("csv", regex="^(csv|json|parquet)$"),
    include_metadata: bool = Query(True, description="Include dataset metadata"),
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> FileResponse:
    """
    Export a dataset in the specified format.
    
    **Parameters:**
    - **dataset_id**: Identifier of the dataset to export
    - **format**: Export format (csv, json, parquet)
    - **include_metadata**: Whether to include metadata in export
    
    **Returns:**
    - **File download** in the requested format
    """
    
    try:
        logger.info(
            "Dataset export requested",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            format=format,
            user_id=current_user["user_id"]
        )
        
        # Get dataset and check permissions
        dataset_info = await data_service.get_dataset_details(
            dataset_id=dataset_id,
            user_id=current_user["user_id"],
            db=db
        )
        
        if not dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Dataset not found",
                    "dataset_id": dataset_id,
                    "correlation_id": correlation_id
                }
            )
        
        # Generate export file
        export_path = await data_service.export_dataset(
            dataset_id=dataset_id,
            format=format,
            include_metadata=include_metadata,
            user_id=current_user["user_id"],
            db=db
        )
        
        if not export_path or not export_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Export generation failed",
                    "correlation_id": correlation_id
                }
            )
        
        # Determine media type
        media_type_map = {
            "csv": "text/csv",
            "json": "application/json",
            "parquet": "application/octet-stream"
        }
        
        filename = f"{dataset_info.name}_{dataset_id}.{format}"
        
        logger.info(
            "Dataset export completed",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            export_file=str(export_path)
        )
        
        return FileResponse(
            path=export_path,
            filename=filename,
            media_type=media_type_map.get(format, "application/octet-stream"),
            headers={"X-Correlation-ID": correlation_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Dataset export failed",
            correlation_id=correlation_id,
            dataset_id=dataset_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Dataset export failed",
                "correlation_id": correlation_id
            }
        )


@router.get(
    "/stats",
    summary="Get Data Statistics",
    description="Get overall data management statistics"
)
async def get_data_statistics(
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get comprehensive data management statistics.
    
    **Returns:**
    - **total_datasets**: Total number of datasets
    - **dataset_types**: Breakdown by dataset type
    - **storage_usage**: Storage usage statistics
    - **sync_history**: Recent synchronization history
    """
    
    try:
        logger.info(
            "Data statistics requested",
            correlation_id=correlation_id,
            user_id=current_user.get("user_id") if current_user else "anonymous"
        )
        
        stats = await data_service.get_data_statistics(
            user_id=current_user.get("user_id") if current_user else None,
            db=db
        )
        
        return stats
        
    except Exception as e:
        logger.error(
            "Failed to get data statistics",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve data statistics",
                "correlation_id": correlation_id
            }
        )


# Export the router
__all__ = ["router"]