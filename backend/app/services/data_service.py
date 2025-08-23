import asyncio
import aiofiles
import httpx
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import json
from io import StringIO, BytesIO

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
import structlog

from app.core.config import get_settings
from app.core.database import db_manager
from app.models.patient import Dataset, DataSyncJob, DataProcessingJob
from app.schemas.data import (
    DatasetInfoResponse,
    DataSyncResponse,
    DataProcessingResponse,
    DataValidationResponse
)
from app.utils.data_preprocessing import DataValidator, DataProcessor

logger = structlog.get_logger(__name__)


class DataService:
    """Main data management service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_validator = DataValidator()
        self.data_processor = DataProcessor()
        
        # Data storage paths
        self.raw_data_path = self.settings.DATA_STORAGE_PATH / "raw"
        self.processed_data_path = self.settings.DATA_STORAGE_PATH / "processed"
        self.temp_data_path = self.settings.DATA_STORAGE_PATH / "temp"
        
        # Ensure directories exist
        for path in [self.raw_data_path, self.processed_data_path, self.temp_data_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str,
        original_filename: str,
        dataset_name: str,
        dataset_type: str,
        description: Optional[str],
        user_id: str,
        validation_result: DataValidationResponse,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Save a new dataset to the system"""
        try:
            logger.info(
                "Saving dataset",
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                rows=len(df),
                columns=len(df.columns)
            )
            
            # Generate unique dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Determine file path based on type
            if dataset_type == "training":
                file_path = self.raw_data_path / filename
            else:
                file_path = self.processed_data_path / filename
            
            # Save DataFrame to file
            if filename.endswith('.csv'):
                df.to_csv(file_path, index=False)
                file_format = "csv"
            elif filename.endswith('.json'):
                df.to_json(file_path, orient='records', date_format='iso')
                file_format = "json"
            elif filename.endswith('.parquet'):
                df.to_parquet(file_path, index=False)
                file_format = "parquet"
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            # Calculate file size
            file_size = file_path.stat().st_size
            
            # Analyze data characteristics
            data_analysis = await self._analyze_dataset(df)
            
            # Create dataset record
            dataset = Dataset(
                id=uuid.UUID(dataset_id),
                name=dataset_name,
                description=description,
                dataset_type=dataset_type,
                source_type="upload",
                original_filename=original_filename,
                stored_filename=filename,
                file_path=str(file_path),
                file_size_bytes=file_size,
                file_format=file_format,
                total_records=len(df),
                total_columns=len(df.columns),
                column_names=df.columns.tolist(),
                data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
                completeness_score=validation_result.quality_score,
                validation_status="passed" if validation_result.is_valid else "failed",
                validation_results=validation_result.dict(),
                data_start_date=data_analysis.get("date_range", {}).get("start"),
                data_end_date=data_analysis.get("date_range", {}).get("end"),
                geographic_coverage=data_analysis.get("geographic_coverage"),
                metadata=data_analysis,
                created_by=uuid.UUID(user_id)
            )
            
            # Save to database
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)
            
            logger.info("Dataset saved successfully", dataset_id=dataset_id)
            
            return {
                "dataset_id": dataset_id,
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "analysis": data_analysis
            }
            
        except Exception as e:
            logger.error("Failed to save dataset", error=str(e), exc_info=True)
            raise
    
    async def get_dataset_details(
        self,
        dataset_id: str,
        include_sample: bool = False,
        user_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> Optional[DatasetInfoResponse]:
        """Get detailed information about a dataset"""
        try:
            if not db:
                async with db_manager.get_async_session() as db:
                    return await self._get_dataset_details_impl(dataset_id, include_sample, user_id, db)
            else:
                return await self._get_dataset_details_impl(dataset_id, include_sample, user_id, db)
            
        except Exception as e:
            logger.error("Failed to get dataset details", dataset_id=dataset_id, error=str(e))
            return None
    
    async def _get_dataset_details_impl(
        self,
        dataset_id: str,
        include_sample: bool,
        user_id: Optional[str],
        db: AsyncSession
    ) -> Optional[DatasetInfoResponse]:
        """Implementation of get_dataset_details"""
        try:
            # Query dataset
            query = select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
            
            # Add user filter if specified (privacy)
            if user_id and not self._is_admin_user(user_id):
                query = query.where(Dataset.created_by == uuid.UUID(user_id))
            
            result = await db.execute(query)
            dataset = result.scalar_one_or_none()
            
            if not dataset:
                return None
            
            # Load sample data if requested
            sample_data = None
            if include_sample:
                sample_data = await self._load_dataset_sample(dataset.file_path)
            
            # Create response
            response_data = {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "dataset_type": dataset.dataset_type,
                "total_records": dataset.total_records,
                "total_columns": dataset.total_columns,
                "file_size_mb": round(dataset.file_size_bytes / (1024 * 1024), 2),
                "completeness_score": dataset.completeness_score,
                "validation_status": dataset.validation_status,
                "processing_status": dataset.processing_status,
                "created_at": dataset.created_at,
                "last_used_date": dataset.last_used_date,
                "usage_count": dataset.usage_count,
                "owner_id": dataset.created_by,
                "sample_data": sample_data
            }
            
            return DatasetInfoResponse(**response_data)
            
        except Exception as e:
            logger.error("Dataset details implementation failed", error=str(e))
            return None
    
    async def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> List[DatasetInfoResponse]:
        """List datasets with filtering and pagination"""
        try:
            if not db:
                async with db_manager.get_async_session() as db:
                    return await self._list_datasets_impl(dataset_type, limit, offset, user_id, db)
            else:
                return await self._list_datasets_impl(dataset_type, limit, offset, user_id, db)
            
        except Exception as e:
            logger.error("Failed to list datasets", error=str(e))
            return []
    
    async def _list_datasets_impl(
        self,
        dataset_type: Optional[str],
        limit: int,
        offset: int,
        user_id: Optional[str],
        db: AsyncSession
    ) -> List[DatasetInfoResponse]:
        """Implementation of list_datasets"""
        try:
            # Build query
            query = select(Dataset).order_by(desc(Dataset.created_at))
            
            # Apply filters
            if dataset_type:
                query = query.where(Dataset.dataset_type == dataset_type)
            
            if user_id and not self._is_admin_user(user_id):
                query = query.where(Dataset.created_by == uuid.UUID(user_id))
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Execute query
            result = await db.execute(query)
            datasets = result.scalars().all()
            
            # Convert to response format
            dataset_responses = []
            for dataset in datasets:
                response_data = {
                    "id": dataset.id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "dataset_type": dataset.dataset_type,
                    "total_records": dataset.total_records,
                    "total_columns": dataset.total_columns,
                    "file_size_mb": round(dataset.file_size_bytes / (1024 * 1024), 2),
                    "completeness_score": dataset.completeness_score,
                    "validation_status": dataset.validation_status,
                    "processing_status": dataset.processing_status,
                    "created_at": dataset.created_at,
                    "last_used_date": dataset.last_used_date,
                    "usage_count": dataset.usage_count,
                    "owner_id": dataset.created_by
                }
                dataset_responses.append(DatasetInfoResponse(**response_data))
            
            return dataset_responses
            
        except Exception as e:
            logger.error("List datasets implementation failed", error=str(e))
            return []
    
    async def start_cdc_sync_job(
        self,
        user_id: str,
        force_update: bool = False,
        data_range: Optional[Dict[str, str]] = None,
        correlation_id: str = ""
    ) -> Dict[str, Any]:
        """Start CDC data synchronization job"""
        try:
            job_id = str(uuid.uuid4())
            
            logger.info(
                "Starting CDC sync job",
                job_id=job_id,
                user_id=user_id,
                force_update=force_update,
                correlation_id=correlation_id
            )
            
            # Determine data range
            if data_range:
                start_date = date.fromisoformat(data_range.get("start", "2016-01-01"))
                end_date = date.fromisoformat(data_range.get("end", date.today().isoformat()))
            else:
                # Default to last 2 years
                end_date = date.today()
                start_date = end_date - timedelta(days=730)
            
            # Create sync job record
            sync_job = DataSyncJob(
                id=uuid.UUID(job_id),
                job_name=f"CDC Sync {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                data_source="cdc",
                sync_type="full" if force_update else "incremental",
                status="queued",
                date_range_start=start_date,
                date_range_end=end_date,
                sync_parameters={
                    "force_update": force_update,
                    "api_endpoint": f"{self.settings.CDC_API_BASE_URL}/views/iw6q-r3ja/rows.csv",
                    "filters": data_range
                },
                triggered_by=uuid.UUID(user_id),
                correlation_id=correlation_id
            )
            
            # Save job to database (if db session available)
            # For now, return job info
            
            return {
                "job_id": job_id,
                "status": "queued",
                "data_source": "cdc",
                "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()}
            }
            
        except Exception as e:
            logger.error("Failed to start CDC sync job", error=str(e), exc_info=True)
            raise
    
    async def execute_cdc_sync(
        self,
        job_id: str,
        correlation_id: str,
        db: AsyncSession
    ) -> None:
        """Execute CDC data synchronization"""
        try:
            logger.info("Executing CDC sync", job_id=job_id, correlation_id=correlation_id)
            
            # Update job status to running
            # In real implementation, would update database record
            
            # Fetch data from CDC API
            cdc_data = await self._fetch_cdc_data()
            
            if cdc_data is None or cdc_data.empty:
                logger.error("No data received from CDC API")
                return
            
            # Process and validate data
            processed_data = await self.data_processor.process_cdc_data(cdc_data)
            validation_result = await self.data_validator.validate_dataset_structure(
                processed_data, "training"
            )
            
            # Save processed data
            output_filename = f"cdc_sync_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = self.processed_data_path / output_filename
            
            processed_data.to_csv(output_path, index=False)
            
            # Create dataset record
            dataset = Dataset(
                name=f"CDC Cardiovascular Data {datetime.utcnow().strftime('%Y-%m-%d')}",
                description="Synchronized cardiovascular disease data from CDC",
                dataset_type="training",
                source_type="cdc",
                stored_filename=output_filename,
                file_path=str(output_path),
                file_size_bytes=output_path.stat().st_size,
                file_format="csv",
                total_records=len(processed_data),
                total_columns=len(processed_data.columns),
                column_names=processed_data.columns.tolist(),
                validation_status="passed" if validation_result.is_valid else "failed",
                processing_status="processed",
                created_by=uuid.UUID("00000000-0000-0000-0000-000000000000")  # System user
            )
            
            db.add(dataset)
            await db.commit()
            
            logger.info(
                "CDC sync completed successfully",
                job_id=job_id,
                records_processed=len(processed_data)
            )
            
        except Exception as e:
            logger.error("CDC sync execution failed", job_id=job_id, error=str(e), exc_info=True)
            raise
    
    async def _fetch_cdc_data(self) -> Optional[pd.DataFrame]:
        """Fetch cardiovascular disease data from CDC API"""
        try:
            cdc_url = f"{self.settings.CDC_API_BASE_URL}/views/iw6q-r3ja/rows.csv?accessType=DOWNLOAD"
            
            logger.info("Fetching data from CDC", url=cdc_url)
            
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
                response = await client.get(cdc_url)
                
                if response.status_code != 200:
                    logger.error("CDC API request failed", status_code=response.status_code)
                    return None
                
                # Parse CSV data
                csv_content = response.text
                df = pd.read_csv(StringIO(csv_content))
                
                logger.info("CDC data fetched successfully", rows=len(df), columns=len(df.columns))
                return df
                
        except Exception as e:
            logger.error("Failed to fetch CDC data", error=str(e), exc_info=True)
            return None
    
    async def get_last_sync_info(self, data_source: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get information about the last data synchronization"""
        try:
            query = select(DataSyncJob).where(
                and_(
                    DataSyncJob.data_source == data_source,
                    DataSyncJob.status == "completed"
                )
            ).order_by(desc(DataSyncJob.completed_at)).limit(1)
            
            result = await db.execute(query)
            last_sync = result.scalar_one_or_none()
            
            if last_sync:
                return {
                    "job_id": str(last_sync.id),
                    "completed_at": last_sync.completed_at.isoformat(),
                    "records_processed": last_sync.records_processed,
                    "success_rate": last_sync.success_rate
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get last sync info", error=str(e))
            return None
    
    async def get_sync_job_status(self, job_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get status of a synchronization job"""
        try:
            query = select(DataSyncJob).where(DataSyncJob.id == uuid.UUID(job_id))
            result = await db.execute(query)
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            # Calculate progress and estimates
            estimated_remaining = job.estimated_remaining_minutes
            
            return {
                "job_id": job_id,
                "status": job.status,
                "progress_percentage": job.progress_percentage,
                "records_processed": job.records_processed,
                "records_total": job.records_requested,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "estimated_completion": job.estimated_completion.isoformat() if job.estimated_completion else None,
                "estimated_remaining_minutes": estimated_remaining,
                "success_rate": job.success_rate,
                "error_message": job.error_message
            }
            
        except Exception as e:
            logger.error("Failed to get sync job status", job_id=job_id, error=str(e))
            return None
    
    async def start_processing_job(
        self,
        dataset_id: str,
        processing_params: Dict[str, Any],
        user_id: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Start a data processing job"""
        try:
            job_id = str(uuid.uuid4())
            
            logger.info(
                "Starting data processing job",
                job_id=job_id,
                dataset_id=dataset_id,
                user_id=user_id,
                correlation_id=correlation_id
            )
            
            # Default processing configuration
            default_config = {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "outlier_detection": True,
                "feature_engineering": True,
                "data_validation": True
            }
            
            # Merge with user parameters
            processing_config = {**default_config, **processing_params}
            
            return {
                "job_id": job_id,
                "status": "queued",
                "processing_config": processing_config,
                "estimated_duration_minutes": 10
            }
            
        except Exception as e:
            logger.error("Failed to start processing job", error=str(e), exc_info=True)
            raise
    
    async def execute_data_processing(
        self,
        job_id: str,
        dataset_id: str,
        processing_params: Dict[str, Any],
        correlation_id: str,
        db: AsyncSession
    ) -> None:
        """Execute data processing job"""
        try:
            logger.info("Executing data processing", job_id=job_id, dataset_id=dataset_id)
            
            # Load dataset
            dataset = await self._load_dataset_by_id(dataset_id, db)
            if not dataset:
                raise ValueError(f"Dataset not found: {dataset_id}")
            
            # Load data file
            df = await self._load_dataframe_from_path(dataset.file_path)
            
            # Apply processing steps
            processed_df = await self.data_processor.process_dataset(df, processing_params)
            
            # Validate processed data
            validation_result = await self.data_validator.comprehensive_validate(processed_df)
            
            # Save processed dataset
            output_filename = f"processed_{dataset.stored_filename}"
            output_path = self.processed_data_path / output_filename
            processed_df.to_csv(output_path, index=False)
            
            # Update dataset record
            dataset.processing_status = "processed"
            dataset.validation_results = validation_result.dict()
            dataset.updated_at = datetime.utcnow()
            
            await db.commit()
            
            logger.info("Data processing completed", job_id=job_id)
            
        except Exception as e:
            logger.error("Data processing execution failed", job_id=job_id, error=str(e), exc_info=True)
            raise
    
    async def export_dataset(
        self,
        dataset_id: str,
        format: str,
        include_metadata: bool,
        user_id: str,
        db: AsyncSession
    ) -> Optional[Path]:
        """Export a dataset in specified format"""
        try:
            logger.info("Exporting dataset", dataset_id=dataset_id, format=format)
            
            # Get dataset
            dataset = await self._load_dataset_by_id(dataset_id, db)
            if not dataset:
                return None
            
            # Load data
            df = await self._load_dataframe_from_path(dataset.file_path)
            
            # Generate export filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            export_filename = f"{dataset.name}_{timestamp}.{format}"
            export_path = self.temp_data_path / export_filename
            
            # Export in requested format
            if format == "csv":
                df.to_csv(export_path, index=False)
            elif format == "json":
                df.to_json(export_path, orient='records', date_format='iso', indent=2)
            elif format == "parquet":
                df.to_parquet(export_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Add metadata file if requested
            if include_metadata:
                metadata = {
                    "dataset_info": {
                        "name": dataset.name,
                        "description": dataset.description,
                        "total_records": dataset.total_records,
                        "total_columns": dataset.total_columns,
                        "created_at": dataset.created_at.isoformat(),
                        "data_quality_score": dataset.data_quality_score
                    },
                    "export_info": {
                        "exported_at": datetime.utcnow().isoformat(),
                        "exported_by": user_id,
                        "export_format": format
                    }
                }
                
                metadata_path = export_path.with_suffix(f"{export_path.suffix}.metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info("Dataset export completed", export_path=str(export_path))
            return export_path
            
        except Exception as e:
            logger.error("Dataset export failed", dataset_id=dataset_id, error=str(e), exc_info=True)
            return None
    
    async def delete_dataset(
        self,
        dataset_id: str,
        user_id: str,
        db: AsyncSession
    ) -> bool:
        """Delete a dataset and its files"""
        try:
            logger.info("Deleting dataset", dataset_id=dataset_id, user_id=user_id)
            
            # Get dataset
            dataset = await self._load_dataset_by_id(dataset_id, db)
            if not dataset:
                return False
            
            # Check permissions
            if str(dataset.created_by) != user_id and not self._is_admin_user(user_id):
                raise PermissionError("User does not have permission to delete this dataset")
            
            # Delete physical files
            file_path = Path(dataset.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete related files (metadata, processed versions, etc.)
            file_stem = file_path.stem
            related_files = file_path.parent.glob(f"{file_stem}*")
            for related_file in related_files:
                if related_file.exists():
                    related_file.unlink()
            
            # Delete database record
            await db.delete(dataset)
            await db.commit()
            
            logger.info("Dataset deleted successfully", dataset_id=dataset_id)
            return True
            
        except Exception as e:
            logger.error("Dataset deletion failed", dataset_id=dataset_id, error=str(e), exc_info=True)
            return False
    
    async def get_data_statistics(
        self,
        user_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get comprehensive data statistics"""
        try:
            if not db:
                async with db_manager.get_async_session() as db:
                    return await self._get_data_statistics_impl(user_id, db)
            else:
                return await self._get_data_statistics_impl(user_id, db)
            
        except Exception as e:
            logger.error("Failed to get data statistics", error=str(e))
            return {}
    
    async def _get_data_statistics_impl(
        self,
        user_id: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Implementation of get_data_statistics"""
        try:
            # Build base query
            query = select(Dataset)
            if user_id and not self._is_admin_user(user_id):
                query = query.where(Dataset.created_by == uuid.UUID(user_id))
            
            result = await db.execute(query)
            datasets = result.scalars().all()
            
            # Calculate statistics
            total_datasets = len(datasets)
            total_size_bytes = sum(d.file_size_bytes for d in datasets if d.file_size_bytes)
            total_records = sum(d.total_records for d in datasets)
            
            # Dataset type breakdown
            type_breakdown = {}
            for dataset in datasets:
                dataset_type = dataset.dataset_type
                if dataset_type not in type_breakdown:
                    type_breakdown[dataset_type] = {"count": 0, "records": 0, "size_mb": 0}
                
                type_breakdown[dataset_type]["count"] += 1
                type_breakdown[dataset_type]["records"] += dataset.total_records
                type_breakdown[dataset_type]["size_mb"] += (dataset.file_size_bytes or 0) / (1024 * 1024)
            
            # Recent activity
            recent_datasets = [d for d in datasets if d.created_at > datetime.utcnow() - timedelta(days=30)]
            
            statistics = {
                "total_datasets": total_datasets,
                "total_records": total_records,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
                "dataset_types": type_breakdown,
                "recent_activity": {
                    "datasets_added_last_30_days": len(recent_datasets),
                    "average_dataset_size_mb": round(total_size_bytes / (1024 * 1024) / max(1, total_datasets), 2)
                },
                "data_quality": {
                    "average_completeness": sum(d.completeness_score for d in datasets if d.completeness_score) / max(1, len([d for d in datasets if d.completeness_score])),
                    "validated_datasets": len([d for d in datasets if d.validation_status == "passed"])
                },
                "storage_usage": {
                    "raw_data_path": str(self.raw_data_path),
                    "processed_data_path": str(self.processed_data_path),
                    "available_space_gb": self._get_available_disk_space()
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error("Data statistics implementation failed", error=str(e))
            return {}
    
    # Helper methods
    async def _analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics"""
        try:
            analysis = {
                "basic_stats": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                    "missing_values": df.isnull().sum().sum(),
                    "duplicate_rows": df.duplicated().sum()
                },
                "column_types": {
                    "numeric": len(df.select_dtypes(include=['int64', 'float64']).columns),
                    "categorical": len(df.select_dtypes(include=['object']).columns),
                    "datetime": len(df.select_dtypes(include=['datetime64']).columns)
                }
            }
            
            # Date range analysis
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0:
                date_col = date_columns[0]
                analysis["date_range"] = {
                    "start": df[date_col].min().date().isoformat(),
                    "end": df[date_col].max().date().isoformat()
                }
            
            # Geographic coverage
            if "LocationAbbr" in df.columns:
                states = df["LocationAbbr"].unique().tolist()
                analysis["geographic_coverage"] = {
                    "states": states,
                    "state_count": len(states)
                }
            
            return analysis
            
        except Exception as e:
            logger.error("Dataset analysis failed", error=str(e))
            return {}
    
    async def _load_dataset_sample(self, file_path: str, sample_size: int = 100) -> Optional[Dict[str, Any]]:
        """Load a sample of dataset for preview"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            if path.suffix == '.csv':
                df = pd.read_csv(path, nrows=sample_size)
            elif path.suffix == '.json':
                df = pd.read_json(path, lines=True, nrows=sample_size)
            elif path.suffix == '.parquet':
                df = pd.read_parquet(path)
                df = df.head(sample_size)
            else:
                return None
            
            return {
                "columns": df.columns.tolist(),
                "sample_rows": df.head(10).to_dict(orient='records'),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_size": len(df)
            }
            
        except Exception as e:
            logger.error("Failed to load dataset sample", file_path=file_path, error=str(e))
            return None
    
    async def _load_dataset_by_id(self, dataset_id: str, db: AsyncSession) -> Optional[Dataset]:
        """Load dataset by ID"""
        try:
            query = select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
            result = await db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to load dataset", dataset_id=dataset_id, error=str(e))
            return None
    
    async def _load_dataframe_from_path(self, file_path: str) -> pd.DataFrame:
        """Load DataFrame from file path"""
        try:
            path = Path(file_path)
            
            if path.suffix == '.csv':
                return pd.read_csv(path)
            elif path.suffix == '.json':
                return pd.read_json(path)
            elif path.suffix == '.parquet':
                return pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
                
        except Exception as e:
            logger.error("Failed to load dataframe", file_path=file_path, error=str(e))
            raise
    
    def _is_admin_user(self, user_id: str) -> bool:
        """Check if user is admin (simplified check)"""
        # In real implementation, this would check user roles in database
        return False  # Placeholder
    
    def _get_available_disk_space(self) -> float:
        """Get available disk space in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.settings.DATA_STORAGE_PATH)
            return round(free / (1024 * 1024 * 1024), 2)
        except Exception:
            return 0.0


# Export the data service
__all__ = ["DataService"]