import os
import sys
import json
import logging
import logging.config
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import uuid
import traceback
from contextlib import contextmanager
from functools import wraps
import time

# Third-party imports for enhanced logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSONLOGGER_AVAILABLE = True
except ImportError:
    JSONLOGGER_AVAILABLE = False

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    This formatter converts log records to JSON format for better parsing
    and integration with log aggregation systems.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the JSON formatter."""
        super().__init__(*args, **kwargs)
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        self.service_name = os.getenv('SERVICE_NAME', 'cvd-prediction')
        self.environment = os.getenv('APP_ENV', 'development')
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON formatted log string
        """
        # Create base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': self.service_name,
            'environment': self.environment,
            'hostname': self.hostname,
            'process_id': os.getpid(),
            'thread_id': record.thread,
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'file_path': record.pathname
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
            
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
            
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry['stack_trace'] = record.stack_info
            
        return json.dumps(log_entry, ensure_ascii=False)


class CorrelationFilter(logging.Filter):
    """
    Filter to add correlation IDs to log records.
    
    This filter ensures that all log messages include a correlation ID
    for request tracking across distributed services.
    """
    
    def __init__(self):
        """Initialize the correlation filter."""
        super().__init__()
        self._local = {}
        
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add correlation ID to log record.
        
        Args:
            record: The log record to filter
            
        Returns:
            True to include the record
        """
        # Try to get correlation ID from various sources
        correlation_id = getattr(record, 'correlation_id', None)
        
        if not correlation_id:
            # Try to get from context variables (if using contextvars)
            try:
                import contextvars
                correlation_var = contextvars.ContextVar('correlation_id', default=None)
                correlation_id = correlation_var.get()
            except (ImportError, LookupError):
                pass
                
        if not correlation_id:
            # Generate new correlation ID
            correlation_id = str(uuid.uuid4())
            
        record.correlation_id = correlation_id
        return True


class PerformanceFilter(logging.Filter):
    """
    Filter to add performance metrics to log records.
    
    This filter adds timing information and system metrics
    to help with performance monitoring and debugging.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add performance metrics to log record.
        
        Args:
            record: The log record to filter
            
        Returns:
            True to include the record
        """
        # Add performance metrics
        if hasattr(record, 'start_time') and hasattr(record, 'end_time'):
            record.duration_ms = (record.end_time - record.start_time) * 1000
            
        # Add memory usage (basic)
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except ImportError:
            pass
            
        return True


class SecurityFilter(logging.Filter):
    """
    Security filter to sanitize sensitive information from logs.
    
    This filter removes or masks sensitive data like passwords,
    API keys, and personal information from log messages.
    """
    
    SENSITIVE_PATTERNS = [
        r'password["\s]*[:=]["\s]*[^"\s,}]+',
        r'api[_-]?key["\s]*[:=]["\s]*[^"\s,}]+',
        r'secret["\s]*[:=]["\s]*[^"\s,}]+',
        r'token["\s]*[:=]["\s]*[^"\s,}]+',
        r'bearer\s+[a-zA-Z0-9\-_]+',
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    ]
    
    def __init__(self):
        """Initialize the security filter."""
        super().__init__()
        import re
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SENSITIVE_PATTERNS]
        
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Sanitize sensitive information from log record.
        
        Args:
            record: The log record to filter
            
        Returns:
            True to include the record
        """
        # Sanitize the main message
        message = record.getMessage()
        for pattern in self.patterns:
            message = pattern.sub('[REDACTED]', message)
        
        # Update the record's args to reflect the sanitized message
        record.args = ()
        record.msg = message
        
        # Sanitize extra fields if present
        if hasattr(record, 'extra_fields'):
            sanitized_fields = {}
            for key, value in record.extra_fields.items():
                if isinstance(value, str):
                    for pattern in self.patterns:
                        value = pattern.sub('[REDACTED]', value)
                sanitized_fields[key] = value
            record.extra_fields = sanitized_fields
            
        return True


class CVDLogger:
    """
    Main logger class for the CVD prediction system.
    
    This class provides a unified interface for logging across
    all components of the system with proper configuration
    and integration capabilities.
    """
    
    def __init__(self, name: str = 'cvd-prediction'):
        """
        Initialize the CVD logger.
        
        Args:
            name: Name of the logger
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self._configured = False
        
    def configure(self, config: Optional[Dict[str, Any]] = None):
        """
        Configure the logger with the provided configuration.
        
        Args:
            config: Logging configuration dictionary
        """
        if self._configured:
            return
            
        if config is None:
            config = self._get_default_config()
            
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Add custom filters
        self._add_filters()
        
        self._configured = True
        self.logger.info("Logger configured successfully", extra={
            'extra_fields': {
                'environment': os.getenv('APP_ENV', 'development'),
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'service_name': os.getenv('SERVICE_NAME', 'cvd-prediction')
            }
        })
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default logging configuration.
        
        Returns:
            Default logging configuration dictionary
        """
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        environment = os.getenv('APP_ENV', 'development')
        log_dir = Path(os.getenv('LOG_DIR', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Base configuration
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    '()': JSONFormatter,
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'filters': {
                'correlation': {
                    '()': CorrelationFilter,
                },
                'performance': {
                    '()': PerformanceFilter,
                },
                'security': {
                    '()': SecurityFilter,
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': log_level,
                    'formatter': 'json' if environment == 'production' else 'detailed',
                    'stream': sys.stdout,
                    'filters': ['correlation', 'security']
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'json',
                    'filename': str(log_dir / 'cvd-prediction.log'),
                    'maxBytes': 100 * 1024 * 1024,  # 100MB
                    'backupCount': 5,
                    'filters': ['correlation', 'performance', 'security']
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'json',
                    'filename': str(log_dir / 'cvd-prediction-errors.log'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 10,
                    'filters': ['correlation', 'performance', 'security']
                }
            },
            'loggers': {
                # Root logger
                '': {
                    'level': log_level,
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                # CVD application loggers
                'cvd': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'app': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                # Third-party library loggers
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'fastapi': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'sqlalchemy': {
                    'level': 'WARNING',
                    'handlers': ['file'],
                    'propagate': False
                },
                'sqlalchemy.engine': {
                    'level': 'INFO' if log_level == 'DEBUG' else 'WARNING',
                    'handlers': ['file'],
                    'propagate': False
                },
                'alembic': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'redis': {
                    'level': 'WARNING',
                    'handlers': ['file'],
                    'propagate': False
                }
            }
        }
        
        # Add colored logging for development
        if environment == 'development' and COLORLOG_AVAILABLE:
            config['formatters']['colored'] = {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s',
                'log_colors': {
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            }
            config['handlers']['console']['formatter'] = 'colored'
            
        # Add cloud logging handler if configured
        if os.getenv('CLOUD_LOGGING_ENABLED') == 'true':
            cloud_handler = self._get_cloud_logging_handler()
            if cloud_handler:
                config['handlers']['cloud'] = cloud_handler
                for logger_config in config['loggers'].values():
                    if 'handlers' in logger_config:
                        logger_config['handlers'].append('cloud')
                        
        return config
        
    def _get_cloud_logging_handler(self) -> Optional[Dict[str, Any]]:
        """
        Get cloud logging handler configuration.
        
        Returns:
            Cloud logging handler configuration or None
        """
        cloud_provider = os.getenv('CLOUD_PROVIDER', '').lower()
        
        if cloud_provider == 'aws':
            # AWS CloudWatch Logs
            return {
                'class': 'watchtower.CloudWatchLogsHandler',
                'log_group': os.getenv('AWS_LOG_GROUP', 'cvd-prediction'),
                'stream_name': os.getenv('AWS_LOG_STREAM', '{hostname}-{uuid}'),
                'formatter': 'json',
                'filters': ['correlation', 'security']
            }
        elif cloud_provider == 'gcp':
            # Google Cloud Logging
            return {
                'class': 'google.cloud.logging.handlers.CloudLoggingHandler',
                'formatter': 'json',
                'filters': ['correlation', 'security']
            }
        elif cloud_provider == 'azure':
            # Azure Monitor
            return {
                'class': 'azure.monitor.opentelemetry.exporter.AzureMonitorLogExporter',
                'formatter': 'json',
                'filters': ['correlation', 'security']
            }
        
        return None
        
    def _add_filters(self):
        """Add custom filters to loggers."""
        for handler in self.logger.handlers:
            if not any(isinstance(f, (CorrelationFilter, PerformanceFilter, SecurityFilter)) 
                      for f in handler.filters):
                handler.addFilter(CorrelationFilter())
                handler.addFilter(PerformanceFilter()) 
                handler.addFilter(SecurityFilter())
                
    @contextmanager
    def timer(self, operation: str, level: int = logging.INFO):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation being timed
            level: Log level for the timing information
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        self.logger.log(level, f"Starting operation: {operation}", extra={
            'correlation_id': correlation_id,
            'extra_fields': {
                'operation': operation,
                'phase': 'start'
            }
        })
        
        try:
            yield correlation_id
        except Exception as e:
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            self.logger.error(f"Operation failed: {operation}", extra={
                'correlation_id': correlation_id,
                'extra_fields': {
                    'operation': operation,
                    'phase': 'error',
                    'duration_ms': duration,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            }, exc_info=True)
            raise
        else:
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            self.logger.log(level, f"Completed operation: {operation}", extra={
                'correlation_id': correlation_id,
                'extra_fields': {
                    'operation': operation,
                    'phase': 'complete',
                    'duration_ms': duration
                }
            })
            
    def log_prediction(self, patient_data: Dict[str, Any], prediction_result: Dict[str, Any], 
                      correlation_id: Optional[str] = None):
        """
        Log a prediction event with structured data.
        
        Args:
            patient_data: Input patient data (will be sanitized)
            prediction_result: Prediction results
            correlation_id: Optional correlation ID
        """
        # Sanitize patient data (remove sensitive information)
        sanitized_patient_data = {
            'age_group': self._get_age_group(patient_data.get('Age', 0)),
            'sex': patient_data.get('Sex', 'Unknown'),
            'risk_factors_count': sum([
                patient_data.get('Smoking', False),
                patient_data.get('HighBP', False),
                patient_data.get('HighChol', False),
                patient_data.get('Diabetes', False)
            ])
        }
        
        self.logger.info("Prediction completed", extra={
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'extra_fields': {
                'event_type': 'prediction',
                'patient_data': sanitized_patient_data,
                'prediction': prediction_result.get('prediction'),
                'probability': prediction_result.get('probability'),
                'risk_level': prediction_result.get('risk_level'),
                'model_version': prediction_result.get('model_version'),
                'processing_time': prediction_result.get('processing_time')
            }
        })
        
    def log_api_request(self, method: str, path: str, status_code: int, 
                       response_time: float, correlation_id: Optional[str] = None,
                       user_id: Optional[str] = None):
        """
        Log an API request with timing and status information.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: HTTP status code
            response_time: Response time in milliseconds
            correlation_id: Request correlation ID
            user_id: User ID if available
        """
        log_level = logging.INFO
        if status_code >= 400:
            log_level = logging.WARNING
        if status_code >= 500:
            log_level = logging.ERROR
            
        self.logger.log(log_level, f"{method} {path} - {status_code}", extra={
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'user_id': user_id,
            'extra_fields': {
                'event_type': 'api_request',
                'http_method': method,
                'path': path,
                'status_code': status_code,
                'response_time_ms': response_time,
                'success': status_code < 400
            }
        })
        
    def log_model_performance(self, model_name: str, metrics: Dict[str, float],
                             correlation_id: Optional[str] = None):
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
            correlation_id: Optional correlation ID
        """
        self.logger.info("Model performance metrics", extra={
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'extra_fields': {
                'event_type': 'model_performance',
                'model_name': model_name,
                'metrics': metrics
            }
        })
        
    def log_system_health(self, component: str, status: str, metrics: Dict[str, Any],
                         correlation_id: Optional[str] = None):
        """
        Log system health information.
        
        Args:
            component: Component name (e.g., 'database', 'redis', 'ml_model')
            status: Health status ('healthy', 'unhealthy', 'degraded')
            metrics: Health metrics
            correlation_id: Optional correlation ID
        """
        log_level = logging.INFO if status == 'healthy' else logging.WARNING
        
        self.logger.log(log_level, f"Health check: {component} - {status}", extra={
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'extra_fields': {
                'event_type': 'health_check',
                'component': component,
                'status': status,
                'metrics': metrics
            }
        })
        
    @staticmethod
    def _get_age_group(age: int) -> str:
        """
        Get age group for privacy-preserving logging.
        
        Args:
            age: Age in years
            
        Returns:
            Age group string
        """
        if age < 30:
            return '18-29'
        elif age < 40:
            return '30-39'
        elif age < 50:
            return '40-49'
        elif age < 60:
            return '50-59'
        elif age < 70:
            return '60-69'
        else:
            return '70+'


def performance_logger(operation_name: str):
    """
    Decorator for logging function performance.
    
    Args:
        operation_name: Name of the operation being logged
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = CVDLogger()
            with logger.timer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_logger(name: str = 'cvd-prediction') -> CVDLogger:
    """
    Get a configured CVD logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured CVDLogger instance
    """
    logger = CVDLogger(name)
    if not logger._configured:
        logger.configure()
    return logger


# Global logger instance
logger = get_logger()


# Export main components
__all__ = [
    'CVDLogger',
    'JSONFormatter',
    'CorrelationFilter',
    'PerformanceFilter', 
    'SecurityFilter',
    'performance_logger',
    'get_logger',
    'logger'
]