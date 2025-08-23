from app.api.routes import predictions, health, data
from app.api.dependencies import (
    get_rate_limiter,
    validate_api_key,
    get_current_user_optional,
    get_correlation_id
)

__version__ = "1.0.0"

# API route modules
__all__ = [
    "predictions",
    "health", 
    "data",
    "get_rate_limiter",
    "validate_api_key",
    "get_current_user_optional",
    "get_correlation_id"
]