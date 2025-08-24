import os
import sys
import logging
import time
import json
from typing import Dict, Any, Optional, Union
from functools import wraps
from contextlib import contextmanager
from datetime import datetime
import uuid

# Import the centralized logging configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../monitoring'))
try:
    from logging_config import get_logger, CVDLogger, performance_logger
    CENTRALIZED_LOGGING_AVAILABLE = True
except ImportError:
    CENTRALIZED_LOGGING_AVAILABLE = False


class ApplicationLogger:
    """
    Application-specific logger wrapper for the CVD prediction system.
    
    This class provides a simplified interface for logging common events
    and patterns specific to the cardiovascular prediction application.
    """
    
    def __init__(self, name: str = "cvd-app"):
        """
        Initialize the application logger.
        
        Args:
            name: Logger name identifier
        """
        self.name = name
        
        if CENTRALIZED_LOGGING_AVAILABLE:
            self._logger = get_logger(name)
        else:
            # Fallback to standard logging
            self._setup_fallback_logging()
            
        self.correlation_id = None
        self.user_context = {}
        
    def _setup_fallback_logging(self):
        """Set up fallback logging configuration."""
        self._logger = logging.getLogger(self.name)
        
        if not self._logger.handlers:
            # Set up basic console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            
        # Set log level from environment
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self._logger.setLevel(getattr(logging, log_level, logging.INFO))
        
    def set_correlation_id(self, correlation_id: str):
        """
        Set the correlation ID for request tracking.
        
        Args:
            correlation_id: Unique identifier for request correlation
        """
        self.correlation_id = correlation_id
        
    def set_user_context(self, user_id: Optional[str] = None, 
                        session_id: Optional[str] = None,
                        additional_context: Optional[Dict[str, Any]] = None):
        """
        Set user context for logging.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            additional_context: Additional context information
        """
        self.user_context = {
            'user_id': user_id,
            'session_id': session_id
        }
        
        if additional_context:
            self.user_context.update(additional_context)
            
    def _log_with_context(self, level: int, message: str, 
                         extra_fields: Optional[Dict[str, Any]] = None,
                         exc_info: bool = False):
        """
        Log a message with context information.
        
        Args:
            level: Log level
            message: Log message
            extra_fields: Additional fields to include
            exc_info: Whether to include exception information
        """
        extra = {
            'correlation_id': self.correlation_id or str(uuid.uuid4()),
            'extra_fields': {}
        }
        
        # Add user context
        if self.user_context:
            extra['extra_fields'].update(self.user_context)
            
        # Add additional fields
        if extra_fields:
            extra['extra_fields'].update(extra_fields)
            
        if CENTRALIZED_LOGGING_AVAILABLE:
            self._logger.logger.log(level, message, extra=extra, exc_info=exc_info)
        else:
            # Fallback logging with simple format
            context_str = ""
            if extra['extra_fields']:
                context_str = f" | Context: {json.dumps(extra['extra_fields'])}"
            self._logger.log(level, f"{message}{context_str}", exc_info=exc_info)
            
    # Standard logging methods
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._log_with_context(logging.DEBUG, message, kwargs.get('extra'), kwargs.get('exc_info', False))
        
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self._log_with_context(logging.INFO, message, kwargs.get('extra'), kwargs.get('exc_info', False))
        
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._log_with_context(logging.WARNING, message, kwargs.get('extra'), kwargs.get('exc_info', False))
        
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self._log_with_context(logging.ERROR, message, kwargs.get('extra'), kwargs.get('exc_info', True))
        
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self._log_with_context(logging.CRITICAL, message, kwargs.get('extra'), kwargs.get('exc_info', True))
        
    # Application-specific logging methods
    def log_startup(self, component: str, version: str, config: Dict[str, Any]):
        """
        Log application startup information.
        
        Args:
            component: Component name (e.g., 'backend', 'frontend')
            version: Application version
            config: Startup configuration (sensitive data will be masked)
        """
        # Sanitize configuration
        safe_config = self._sanitize_config(config)
        
        self.info(f"{component} starting up", extra={
            'event_type': 'startup',
            'component': component,
            'version': version,
            'configuration': safe_config,
            'startup_time': datetime.utcnow().isoformat()
        })
        
    def log_shutdown(self, component: str, reason: str = "Normal shutdown"):
        """
        Log application shutdown information.
        
        Args:
            component: Component name
            reason: Shutdown reason
        """
        self.info(f"{component} shutting down", extra={
            'event_type': 'shutdown',
            'component': component,
            'reason': reason,
            'shutdown_time': datetime.utcnow().isoformat()
        })
        
    def log_prediction_request(self, patient_data: Dict[str, Any], 
                              model_version: str, request_id: Optional[str] = None):
        """
        Log a prediction request.
        
        Args:
            patient_data: Patient data for prediction (will be anonymized)
            model_version: Version of the ML model used
            request_id: Optional request identifier
        """
        # Anonymize patient data
        anonymized_data = self._anonymize_patient_data(patient_data)
        
        self.info("Prediction request received", extra={
            'event_type': 'prediction_request',
            'request_id': request_id,
            'patient_profile': anonymized_data,
            'model_version': model_version
        })
        
    def log_prediction_result(self, prediction: int, probability: float, 
                             risk_level: str, model_version: str,
                             processing_time_ms: float, request_id: Optional[str] = None):
        """
        Log prediction results.
        
        Args:
            prediction: Prediction result (0 or 1)
            probability: Prediction probability
            risk_level: Risk level classification
            model_version: Model version used
            processing_time_ms: Processing time in milliseconds
            request_id: Optional request identifier
        """
        self.info("Prediction completed", extra={
            'event_type': 'prediction_result',
            'request_id': request_id,
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'model_version': model_version,
            'processing_time_ms': processing_time_ms
        })
        
    def log_model_load(self, model_name: str, model_version: str, 
                      load_time_ms: float, model_size_mb: float):
        """
        Log model loading information.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            load_time_ms: Time taken to load the model in milliseconds
            model_size_mb: Model size in megabytes
        """
        self.info("Model loaded successfully", extra={
            'event_type': 'model_load',
            'model_name': model_name,
            'model_version': model_version,
            'load_time_ms': load_time_ms,
            'model_size_mb': model_size_mb
        })
        
    def log_model_error(self, model_name: str, error_type: str, 
                       error_message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log model-related errors.
        
        Args:
            model_name: Name of the model
            error_type: Type of error (e.g., 'load_error', 'prediction_error')
            error_message: Error message
            context: Additional error context
        """
        self.error("Model error occurred", extra={
            'event_type': 'model_error',
            'model_name': model_name,
            'error_type': error_type,
            'error_message': error_message,
            'error_context': context or {}
        })
        
    def log_database_operation(self, operation: str, table: str, 
                              execution_time_ms: float, affected_rows: int = 0):
        """
        Log database operations.
        
        Args:
            operation: Database operation (e.g., 'SELECT', 'INSERT', 'UPDATE')
            table: Table name
            execution_time_ms: Execution time in milliseconds
            affected_rows: Number of affected rows
        """
        log_level = logging.INFO
        if execution_time_ms > 1000:  # Log slow queries as warnings
            log_level = logging.WARNING
            
        self._log_with_context(log_level, f"Database operation: {operation} on {table}", {
            'event_type': 'database_operation',
            'operation': operation,
            'table': table,
            'execution_time_ms': execution_time_ms,
            'affected_rows': affected_rows,
            'is_slow_query': execution_time_ms > 1000
        })
        
    def log_api_request(self, method: str, path: str, status_code: int,
                       response_time_ms: float, request_size: Optional[int] = None,
                       response_size: Optional[int] = None):
        """
        Log HTTP API requests.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            request_size: Request size in bytes
            response_size: Response size in bytes
        """
        # Determine log level based on status code
        if status_code < 400:
            log_level = logging.INFO
        elif status_code < 500:
            log_level = logging.WARNING
        else:
            log_level = logging.ERROR
            
        self._log_with_context(log_level, f"{method} {path} - {status_code}", {
            'event_type': 'api_request',
            'http_method': method,
            'path': path,
            'status_code': status_code,
            'response_time_ms': response_time_ms,
            'request_size_bytes': request_size,
            'response_size_bytes': response_size,
            'success': status_code < 400
        })
        
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, source_ip: Optional[str] = None,
                          user_agent: Optional[str] = None):
        """
        Log security-related events.
        
        Args:
            event_type: Type of security event
            severity: Severity level (low, medium, high, critical)
            description: Event description
            source_ip: Source IP address
            user_agent: User agent string
        """
        # Map severity to log level
        severity_map = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        log_level = severity_map.get(severity.lower(), logging.WARNING)
        
        self._log_with_context(log_level, f"Security event: {event_type}", {
            'event_type': 'security_event',
            'security_event_type': event_type,
            'severity': severity,
            'description': description,
            'source_ip': source_ip,
            'user_agent': user_agent
        })
        
    def log_performance_metric(self, metric_name: str, value: float, 
                             unit: str, tags: Optional[Dict[str, str]] = None):
        """
        Log performance metrics.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags for the metric
        """
        self.info(f"Performance metric: {metric_name}", extra={
            'event_type': 'performance_metric',
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'metric_tags': tags or {}
        })
        
    def log_validation_error(self, field: str, value: Any, error_message: str,
                           validation_type: str = "input_validation"):
        """
        Log validation errors.
        
        Args:
            field: Field name that failed validation
            value: Field value (will be sanitized if sensitive)
            error_message: Validation error message
            validation_type: Type of validation
        """
        # Sanitize value if it might be sensitive
        safe_value = self._sanitize_value(field, value)
        
        self.warning("Validation error", extra={
            'event_type': 'validation_error',
            'validation_type': validation_type,
            'field': field,
            'field_value': safe_value,
            'error_message': error_message
        })
        
    @contextmanager
    def operation_timer(self, operation_name: str, **context):
        """
        Context manager for timing operations with automatic logging.
        
        Args:
            operation_name: Name of the operation
            **context: Additional context to include in logs
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        self.debug(f"Starting operation: {operation_name}", extra={
            'event_type': 'operation_start',
            'operation_name': operation_name,
            'operation_id': operation_id,
            **context
        })
        
        try:
            yield operation_id
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            self.error(f"Operation failed: {operation_name}", extra={
                'event_type': 'operation_error',
                'operation_name': operation_name,
                'operation_id': operation_id,
                'duration_ms': duration_ms,
                'error_type': type(e).__name__,
                'error_message': str(e),
                **context
            })
            raise
        else:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Log as warning if operation took too long
            log_level = logging.WARNING if duration_ms > 5000 else logging.INFO
            
            self._log_with_context(log_level, f"Operation completed: {operation_name}", {
                'event_type': 'operation_complete',
                'operation_name': operation_name,
                'operation_id': operation_id,
                'duration_ms': duration_ms,
                'is_slow_operation': duration_ms > 5000,
                **context
            })
            
    def _anonymize_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize patient data for logging.
        
        Args:
            patient_data: Original patient data
            
        Returns:
            Anonymized patient data
        """
        anonymized = {}
        
        # Age groups instead of exact age
        if 'Age' in patient_data:
            age = patient_data['Age']
            if age < 30:
                anonymized['age_group'] = '18-29'
            elif age < 40:
                anonymized['age_group'] = '30-39'
            elif age < 50:
                anonymized['age_group'] = '40-49'
            elif age < 60:
                anonymized['age_group'] = '50-59'
            elif age < 70:
                anonymized['age_group'] = '60-69'
            else:
                anonymized['age_group'] = '70+'
                
        # Keep non-sensitive categorical data
        safe_fields = ['Sex', 'Smoking', 'HighBP', 'HighChol', 'Diabetes', 'PhysActivity']
        for field in safe_fields:
            if field in patient_data:
                anonymized[field] = patient_data[field]
                
        # BMI ranges instead of exact values
        if 'BMI' in patient_data:
            bmi = patient_data['BMI']
            if bmi < 18.5:
                anonymized['bmi_category'] = 'underweight'
            elif bmi < 25:
                anonymized['bmi_category'] = 'normal'
            elif bmi < 30:
                anonymized['bmi_category'] = 'overweight'
            else:
                anonymized['bmi_category'] = 'obese'
                
        return anonymized
        
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize configuration data for logging.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sanitized configuration
        """
        sanitized = {}
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_config(value)
            else:
                sanitized[key] = value
                
        return sanitized
        
    def _sanitize_value(self, field_name: str, value: Any) -> Any:
        """
        Sanitize field values for logging.
        
        Args:
            field_name: Name of the field
            value: Field value
            
        Returns:
            Sanitized value
        """
        sensitive_fields = ['password', 'secret', 'key', 'token', 'ssn', 'credit_card']
        
        if any(sensitive in field_name.lower() for sensitive in sensitive_fields):
            return '[REDACTED]'
        
        # Truncate very long strings
        if isinstance(value, str) and len(value) > 100:
            return value[:97] + '...'
            
        return value


# Performance logging decorator
def log_performance(operation_name: str, logger: Optional[ApplicationLogger] = None):
    """
    Decorator to automatically log function performance.
    
    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance to use (creates new one if not provided)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_instance = logger or get_application_logger()
            with log_instance.operation_timer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global logger instances
_app_logger = None


def get_application_logger(name: str = "cvd-app") -> ApplicationLogger:
    """
    Get the global application logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        ApplicationLogger instance
    """
    global _app_logger
    if _app_logger is None:
        _app_logger = ApplicationLogger(name)
    return _app_logger


# Convenience functions for common logging patterns
def log_info(message: str, **kwargs):
    """Log an info message using the global logger."""
    get_application_logger().info(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log an error message using the global logger."""
    get_application_logger().error(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log a warning message using the global logger."""
    get_application_logger().warning(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log a debug message using the global logger."""
    get_application_logger().debug(message, **kwargs)


# Export main components
__all__ = [
    'ApplicationLogger',
    'get_application_logger', 
    'log_performance',
    'log_info',
    'log_error', 
    'log_warning',
    'log_debug'
]