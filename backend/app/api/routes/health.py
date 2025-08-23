from typing import Any, Dict, List
from datetime import datetime
import asyncio
import time
import psutil
import platform

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import redis.asyncio as redis
import httpx

from app.core.database import get_db, check_db_health, DatabaseHealthChecker
from app.core.config import get_settings, AppConstants
from app.api.dependencies import get_correlation_id
from app.services.ml_service import MLService
from app.utils.logger import get_log_level

logger = structlog.get_logger(__name__)
router = APIRouter()

# Initialize services
ml_service = MLService()


@router.get(
    "/health",
    summary="Basic Health Check",
    description="Basic health check endpoint for load balancers and monitoring"
)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint for quick status verification.
    
    **Returns:**
    - **status**: Overall health status
    - **timestamp**: Current server timestamp
    - **version**: API version
    """
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "cardiovascular-prediction-api"
    }


@router.get(
    "/health/detailed",
    summary="Detailed Health Check",
    description="Comprehensive health check with all system components"
)
async def detailed_health_check(
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Comprehensive health check covering all system components.
    
    **Returns:**
    - **overall_status**: Overall system health
    - **components**: Individual component health status
    - **system_info**: System resource and configuration information
    - **performance_metrics**: Basic performance indicators
    """
    
    start_time = time.time()
    settings = get_settings()
    
    try:
        logger.info("Detailed health check started", correlation_id=correlation_id)
        
        # Initialize health check results
        health_results = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "components": {},
            "system_info": {},
            "performance_metrics": {},
            "warnings": [],
            "errors": []
        }
        
        # Check all components concurrently
        tasks = {
            "database": _check_database_health(db),
            "redis": _check_redis_health(),
            "ml_models": _check_ml_models_health(),
            "external_apis": _check_external_apis_health(),
            "file_system": _check_file_system_health()
        }
        
        # Execute health checks
        component_results = {}
        for component, task in tasks.items():
            try:
                component_results[component] = await asyncio.wait_for(
                    task, 
                    timeout=AppConstants.HEALTH_CHECK_TIMEOUT
                )
            except asyncio.TimeoutError:
                component_results[component] = {
                    "status": "unhealthy",
                    "message": "Health check timed out",
                    "response_time": AppConstants.HEALTH_CHECK_TIMEOUT
                }
                health_results["errors"].append(f"{component} health check timed out")
            except Exception as e:
                component_results[component] = {
                    "status": "unhealthy",
                    "message": f"Health check failed: {str(e)}",
                    "error": str(e)
                }
                health_results["errors"].append(f"{component} health check failed: {str(e)}")
        
        health_results["components"] = component_results
        
        # Determine overall health status
        unhealthy_components = [
            name for name, result in component_results.items()
            if result.get("status") != "healthy"
        ]
        
        critical_unhealthy = [
            comp for comp in unhealthy_components 
            if comp in AppConstants.CRITICAL_SERVICES
        ]
        
        if critical_unhealthy:
            health_results["overall_status"] = "unhealthy"
            health_results["errors"].append(f"Critical services unhealthy: {critical_unhealthy}")
        elif unhealthy_components:
            health_results["overall_status"] = "degraded"
            health_results["warnings"].append(f"Non-critical services unhealthy: {unhealthy_components}")
        
        # Add system information
        health_results["system_info"] = await _get_system_info()
        
        # Add performance metrics
        health_check_duration = time.time() - start_time
        health_results["performance_metrics"] = {
            "health_check_duration": f"{health_check_duration:.4f}s",
            "uptime": _get_uptime(),
            "memory_usage": _get_memory_usage(),
            "cpu_usage": _get_cpu_usage()
        }
        
        # Log health check completion
        logger.info(
            "Detailed health check completed",
            correlation_id=correlation_id,
            overall_status=health_results["overall_status"],
            duration=health_check_duration,
            unhealthy_components=unhealthy_components
        )
        
        # Return appropriate HTTP status based on health
        if health_results["overall_status"] == "unhealthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_results
            )
        elif health_results["overall_status"] == "degraded":
            return JSONResponse(
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                content=health_results
            )
        
        return health_results
        
    except Exception as e:
        logger.error(
            "Health check failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        
        error_response = {
            "overall_status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "error": "Health check system failure",
            "message": str(e)
        }
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response
        )


@router.get(
    "/health/readiness",
    summary="Readiness Probe",
    description="Kubernetes readiness probe endpoint"
)
async def readiness_probe(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Kubernetes readiness probe to check if the service is ready to serve traffic.
    
    **Returns:**
    - **status**: Readiness status
    - **message**: Status description
    """
    
    try:
        # Check critical dependencies
        db_healthy = await DatabaseHealthChecker.ping_database()
        
        if not db_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "message": "Database not available"
                }
            )
        
        # Check if ML models are loaded
        models_loaded = await ml_service.check_models_loaded()
        if not models_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "message": "ML models not loaded"
                }
            )
        
        return {
            "status": "ready",
            "message": "Service is ready to serve traffic"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "message": f"Readiness check failed: {str(e)}"
            }
        )


@router.get(
    "/health/liveness",
    summary="Liveness Probe",
    description="Kubernetes liveness probe endpoint"
)
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe to check if the service is alive.
    
    **Returns:**
    - **status**: Liveness status
    - **timestamp**: Current timestamp
    """
    
    try:
        # Basic liveness check - just verify the application is responding
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "dead",
                "message": f"Liveness check failed: {str(e)}"
            }
        )


# Helper functions for health checks
async def _check_database_health() -> Dict[str, Any]:
    """Check database health"""
    start_time = time.time()
    
    try:
        db_health = await check_db_health()
        response_time = time.time() - start_time
        
        return {
            "status": db_health["status"],
            "message": db_health["message"],
            "response_time": f"{response_time:.4f}s",
            "details": db_health
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "unhealthy",
            "message": f"Database check failed: {str(e)}",
            "response_time": f"{response_time:.4f}s",
            "error": str(e)
        }


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis health"""
    start_time = time.time()
    settings = get_settings()
    
    try:
        redis_client = redis.from_url(str(settings.redis.REDIS_URL))
        
        # Test Redis connection
        pong = await redis_client.ping()
        
        # Get Redis info
        info = await redis_client.info()
        
        await redis_client.close()
        
        response_time = time.time() - start_time
        
        return {
            "status": "healthy" if pong else "unhealthy",
            "message": "Redis connection successful",
            "response_time": f"{response_time:.4f}s",
            "details": {
                "version": info.get("redis_version"),
                "memory_usage": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime": info.get("uptime_in_seconds")
            }
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "unhealthy",
            "message": f"Redis check failed: {str(e)}",
            "response_time": f"{response_time:.4f}s",
            "error": str(e)
        }


async def _check_ml_models_health() -> Dict[str, Any]:
    """Check ML models health"""
    start_time = time.time()
    
    try:
        models_status = await ml_service.health_check()
        response_time = time.time() - start_time
        
        return {
            "status": "healthy" if models_status.get("loaded") else "unhealthy",
            "message": "ML models are loaded and functional",
            "response_time": f"{response_time:.4f}s",
            "details": models_status
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "unhealthy",
            "message": f"ML models check failed: {str(e)}",
            "response_time": f"{response_time:.4f}s",
            "error": str(e)
        }


async def _check_external_apis_health() -> Dict[str, Any]:
    """Check external APIs health"""
    start_time = time.time()
    settings = get_settings()
    
    try:
        api_statuses = {}
        
        # Check CDC API
        if settings.CDC_API_BASE_URL:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{settings.CDC_API_BASE_URL}/views.json?limit=1")
                    api_statuses["cdc_api"] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds()
                    }
            except Exception as e:
                api_statuses["cdc_api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check Medicare API if configured
        if settings.MEDICARE_API_BASE_URL:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(settings.MEDICARE_API_BASE_URL)
                    api_statuses["medicare_api"] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds()
                    }
            except Exception as e:
                api_statuses["medicare_api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        response_time = time.time() - start_time
        overall_status = "healthy" if all(
            api.get("status") == "healthy" for api in api_statuses.values()
        ) else "degraded"
        
        return {
            "status": overall_status,
            "message": "External APIs check completed",
            "response_time": f"{response_time:.4f}s",
            "details": api_statuses
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "unhealthy",
            "message": f"External APIs check failed: {str(e)}",
            "response_time": f"{response_time:.4f}s",
            "error": str(e)
        }


async def _check_file_system_health() -> Dict[str, Any]:
    """Check file system health"""
    start_time = time.time()
    settings = get_settings()
    
    try:
        # Check data directory
        data_path = settings.DATA_STORAGE_PATH
        ml_path = settings.ml.MODEL_PATH
        
        directories_status = {}
        
        # Check if directories exist and are writable
        for name, path in [("data", data_path), ("models", ml_path)]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = path / "health_check_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
                
                # Get directory size and file count
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                file_count = len(list(path.rglob('*')))
                
                directories_status[name] = {
                    "status": "healthy",
                    "path": str(path),
                    "exists": True,
                    "writable": True,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": file_count
                }
                
            except Exception as e:
                directories_status[name] = {
                    "status": "unhealthy",
                    "path": str(path),
                    "error": str(e)
                }
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        disk_free_percent = (disk_usage.free / disk_usage.total) * 100
        
        disk_status = {
            "total_gb": round(disk_usage.total / (1024**3), 2),
            "used_gb": round(disk_usage.used / (1024**3), 2),
            "free_gb": round(disk_usage.free / (1024**3), 2),
            "free_percent": round(disk_free_percent, 2)
        }
        
        # Warn if disk space is low
        warnings = []
        if disk_free_percent < 10:
            warnings.append("Low disk space (< 10% free)")
        elif disk_free_percent < 20:
            warnings.append("Disk space running low (< 20% free)")
        
        response_time = time.time() - start_time
        overall_status = "healthy"
        
        if any(d.get("status") == "unhealthy" for d in directories_status.values()):
            overall_status = "unhealthy"
        elif warnings:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "message": "File system check completed",
            "response_time": f"{response_time:.4f}s",
            "details": {
                "directories": directories_status,
                "disk_usage": disk_status
            },
            "warnings": warnings
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "unhealthy",
            "message": f"File system check failed: {str(e)}",
            "response_time": f"{response_time:.4f}s",
            "error": str(e)
        }


async def _get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        settings = get_settings()
        
        return {
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "api_version": settings.API_VERSION,
            "log_level": get_log_level(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "configuration": {
                "cors_origins": settings.CORS_ORIGINS,
                "prometheus_enabled": settings.PROMETHEUS_ENABLED,
                "rate_limiting": settings.security.RATE_LIMIT_ENABLED,
                "auto_retrain": settings.ml.AUTO_RETRAIN
            }
        }
        
    except Exception as e:
        logger.error("Failed to get system info", error=str(e))
        return {"error": "Failed to retrieve system information"}


def _get_uptime() -> str:
    """Get system uptime"""
    try:
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        
        return f"{days}d {hours}h {minutes}m"
        
    except Exception:
        return "unknown"


def _get_memory_usage() -> Dict[str, Any]:
    """Get memory usage information"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent
        }
    except Exception:
        return {"error": "Unable to retrieve memory information"}


def _get_cpu_usage() -> Dict[str, Any]:
    """Get CPU usage information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "load_average": {
                "1min": load_avg[0] if load_avg else None,
                "5min": load_avg[1] if load_avg else None,
                "15min": load_avg[2] if load_avg else None
            } if load_avg else None
        }
    except Exception:
        return {"error": "Unable to retrieve CPU information"}


@router.get(
    "/health/startup",
    summary="Startup Health Check",
    description="Check if all startup procedures completed successfully"
)
async def startup_health_check(
    correlation_id: str = Depends(get_correlation_id),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Verify that all startup procedures completed successfully.
    
    **Returns:**
    - **startup_status**: Overall startup health
    - **initialized_components**: List of successfully initialized components
    - **startup_time**: Time taken for startup procedures
    """
    
    try:
        logger.info("Startup health check requested", correlation_id=correlation_id)
        
        startup_checks = {
            "database_tables": await DatabaseHealthChecker.check_tables_exist(),
            "ml_models_loaded": await ml_service.check_models_loaded(),
            "configuration_valid": _validate_startup_configuration(),
            "required_directories": _check_required_directories()
        }
        
        # Determine startup status
        all_checks_passed = all(
            isinstance(check, dict) and check.get("status", True) != False
            for check in startup_checks.values()
        )
        
        startup_status = "complete" if all_checks_passed else "incomplete"
        
        return {
            "startup_status": startup_status,
            "correlation_id": correlation_id,
            "checks": startup_checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Startup health check failed",
            correlation_id=correlation_id,
            error=str(e),
            exc_info=True
        )
        
        return {
            "startup_status": "failed",
            "correlation_id": correlation_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


def _validate_startup_configuration() -> Dict[str, Any]:
    """Validate that configuration is properly set for startup"""
    try:
        settings = get_settings()
        
        validation_results = {
            "database_url_set": bool(settings.database.DATABASE_URL),
            "secret_key_set": bool(settings.security.SECRET_KEY),
            "environment_valid": settings.ENVIRONMENT in ["development", "staging", "production"],
            "required_paths_exist": all([
                settings.DATA_STORAGE_PATH.exists(),
                settings.ml.MODEL_PATH.exists()
            ])
        }
        
        return {
            "status": all(validation_results.values()),
            "details": validation_results
        }
        
    except Exception as e:
        return {
            "status": False,
            "error": str(e)
        }


def _check_required_directories() -> Dict[str, Any]:
    """Check that all required directories exist"""
    try:
        settings = get_settings()
        
        required_dirs = {
            "data": settings.DATA_STORAGE_PATH,
            "models": settings.ml.MODEL_PATH,
            "data_raw": settings.DATA_STORAGE_PATH / "raw",
            "data_processed": settings.DATA_STORAGE_PATH / "processed"
        }
        
        dir_status = {}
        for name, path in required_dirs.items():
            dir_status[name] = {
                "exists": path.exists(),
                "is_directory": path.is_dir() if path.exists() else False,
                "path": str(path)
            }
        
        all_exist = all(
            status["exists"] and status["is_directory"] 
            for status in dir_status.values()
        )
        
        return {
            "status": all_exist,
            "directories": dir_status
        }
        
    except Exception as e:
        return {
            "status": False,
            "error": str(e)
        }


# Export the router
__all__ = ["router"]