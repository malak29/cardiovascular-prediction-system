from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import time

from app.core.config import get_settings
from app.core.database import create_db_and_tables, close_db_connection
from app.api.routes import predictions, health, data
from app.utils.logger import setup_logging

# Initialize structured logging
setup_logging()
logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)
        
        REQUEST_COUNT.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        # Generate correlation ID for request tracing
        correlation_id = request.headers.get("X-Correlation-ID", f"req-{int(time.time()*1000)}")
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            correlation_id=correlation_id,
            client_host=request.client.host if request.client else "unknown"
        )
        
        try:
            response = await call_next(request)
            
            # Log successful response
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration=f"{duration:.4f}s",
                correlation_id=correlation_id
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                duration=f"{duration:.4f}s",
                correlation_id=correlation_id,
                error=str(exc),
                exc_info=True
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    settings = get_settings()
    
    # Startup events
    logger.info("Starting Cardiovascular Prediction System API", version="1.0.0")
    
    try:
        # Initialize database
        await create_db_and_tables()
        logger.info("Database initialized successfully")
        
        # Load ML models
        # Note: This will be implemented in the ML service
        logger.info("ML models loaded successfully")
        
        # Startup health check
        logger.info(
            "API startup completed",
            environment=settings.ENVIRONMENT,
            debug=settings.DEBUG,
            database_url=settings.DATABASE_URL.replace(settings.POSTGRES_PASSWORD, "***")
        )
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e), exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown events
    logger.info("Shutting down Cardiovascular Prediction System API")
    
    try:
        # Close database connections
        await close_db_connection()
        logger.info("Database connections closed")
        
        # Cleanup resources
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)


def create_application() -> FastAPI:
    """Factory function to create FastAPI application"""
    settings = get_settings()
    
    # Create FastAPI instance
    app = FastAPI(
        title="Cardiovascular Disease Prediction API",
        description="Production-ready ML system for predicting cardiovascular disease hospitalization rates among Medicare beneficiaries",
        version="1.0.0",
        openapi_url=f"/api/{settings.API_VERSION}/openapi.json" if settings.DEBUG else None,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
        # Additional metadata
        contact={
            "name": "ML Engineering Team",
            "email": "ml-eng@yourorg.com",
            "url": "https://github.com/yourorg/cardiovascular-prediction-system"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    )
    
    # Add security middleware
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID"]
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    if settings.PROMETHEUS_ENABLED:
        app.add_middleware(PrometheusMiddleware)
    
    # Include API routes
    app.include_router(
        health.router,
        prefix=f"/api/{settings.API_VERSION}",
        tags=["Health"]
    )
    
    app.include_router(
        predictions.router,
        prefix=f"/api/{settings.API_VERSION}",
        tags=["Predictions"]
    )
    
    app.include_router(
        data.router,
        prefix=f"/api/{settings.API_VERSION}",
        tags=["Data Management"]
    )
    
    return app


# Create the FastAPI application
app = create_application()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions"""
    correlation_id = request.headers.get("X-Correlation-ID", "unknown")
    
    logger.error(
        "Unhandled exception",
        url=str(request.url),
        method=request.method,
        correlation_id=correlation_id,
        error=str(exc),
        exc_info=True
    )
    
    # Don't expose internal errors in production
    settings = get_settings()
    if settings.ENVIRONMENT == "production":
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "correlation_id": correlation_id,
                "message": "An unexpected error occurred. Please try again later."
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "correlation_id": correlation_id,
                "message": str(exc),
                "type": type(exc).__name__
            }
        )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    settings = get_settings()
    return {
        "message": "Cardiovascular Disease Prediction API",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs" if settings.DEBUG else "Contact administrator for API documentation",
        "health_check": f"/api/{settings.API_VERSION}/health",
        "prediction_endpoint": f"/api/{settings.API_VERSION}/predict"
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint"""
    settings = get_settings()
    if not settings.PROMETHEUS_ENABLED:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics endpoint is disabled"}
        )
    
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


def main():
    """Main function to run the application"""
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )


if __name__ == "__main__":
    main()