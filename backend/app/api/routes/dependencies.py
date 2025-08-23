from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import time
import secrets
from functools import wraps

from fastapi import (
    Depends,
    HTTPException,
    status,
    Request,
    UploadFile,
    Header
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
import structlog

from app.core.config import get_settings
from app.core.security import (
    get_current_user_from_token,
    APIKeyManager,
    SecurityError
)

logger = structlog.get_logger(__name__)

# Initialize HTTP Bearer for API key authentication
api_key_security = HTTPBearer(auto_error=False)


class RateLimiter:
    """Rate limiting implementation using Redis"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for rate limiting"""
        if not self.redis_client:
            settings = get_settings()
            self.redis_client = redis.from_url(
                str(settings.redis.REDIS_URL),
                decode_responses=True
            )
        return self.redis_client
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int = 60,
        burst: int = 10
    ) -> Dict[str, Any]:
        """
        Check if request is allowed based on rate limiting rules.
        
        **Parameters:**
        - **key**: Rate limiting key (usually IP or user ID)
        - **limit**: Number of requests allowed per window
        - **window**: Time window in seconds
        - **burst**: Additional burst capacity
        
        **Returns:**
        - **allowed**: Whether the request is allowed
        - **remaining**: Remaining requests in current window
        - **reset_time**: When the window resets
        """
        
        try:
            redis_client = await self._get_redis_client()
            current_time = int(time.time())
            window_start = current_time - (current_time % window)
            
            # Create rate limiting key
            rate_limit_key = f"rate_limit:{key}:{window_start}"
            
            # Get current count
            current_count = await redis_client.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0
            
            # Check burst capacity first
            burst_key = f"burst:{key}"
            burst_count = await redis_client.get(burst_key)
            burst_count = int(burst_count) if burst_count else 0
            
            # Allow burst if within burst limit
            if burst_count < burst:
                await redis_client.incr(burst_key)
                await redis_client.expire(burst_key, 10)  # 10 second burst window
                
                return {
                    "allowed": True,
                    "remaining": limit - current_count,
                    "reset_time": window_start + window,
                    "burst_used": True
                }
            
            # Check regular rate limit
            if current_count >= limit:
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": window_start + window,
                    "burst_used": False
                }
            
            # Increment counter
            await redis_client.incr(rate_limit_key)
            await redis_client.expire(rate_limit_key, window)
            
            return {
                "allowed": True,
                "remaining": limit - current_count - 1,
                "reset_time": window_start + window,
                "burst_used": False
            }
            
        except Exception as e:
            logger.error("Rate limiting check failed", error=str(e))
            # Fail open - allow request if rate limiting fails
            return {
                "allowed": True,
                "remaining": limit,
                "reset_time": int(time.time()) + window,
                "error": "Rate limiting unavailable"
            }


# Global rate limiter instance
rate_limiter = RateLimiter()


async def get_rate_limiter(request: Request) -> None:
    """
    FastAPI dependency for rate limiting.
    
    **Raises:**
    - **HTTPException**: If rate limit is exceeded
    """
    
    settings = get_settings()
    
    if not settings.security.RATE_LIMIT_ENABLED:
        return
    
    try:
        # Get client identifier (IP or authenticated user)
        client_ip = request.client.host if request.client else "unknown"
        
        # Try to get user ID from token for authenticated users
        auth_header = request.headers.get("Authorization")
        rate_limit_key = client_ip
        
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from app.core.security import JWTManager
                token = auth_header.split(" ")[1]
                payload = JWTManager.decode_token(token)
                user_id = payload.get("sub")
                if user_id:
                    rate_limit_key = f"user:{user_id}"
            except Exception:
                # Continue with IP-based limiting if token is invalid
                pass
        
        # Check rate limit
        result = await rate_limiter.is_allowed(
            key=rate_limit_key,
            limit=settings.security.RATE_LIMIT_REQUESTS_PER_MINUTE,
            window=60,
            burst=settings.security.RATE_LIMIT_BURST
        )
        
        if not result["allowed"]:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                rate_limit_key=rate_limit_key,
                reset_time=result["reset_time"]
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": result["reset_time"] - int(time.time()),
                    "remaining": result["remaining"]
                },
                headers={
                    "Retry-After": str(result["reset_time"] - int(time.time())),
                    "X-RateLimit-Limit": str(settings.security.RATE_LIMIT_REQUESTS_PER_MINUTE),
                    "X-RateLimit-Remaining": str(result["remaining"]),
                    "X-RateLimit-Reset": str(result["reset_time"])
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rate limiting error", error=str(e))
        # Fail open - don't block requests if rate limiting fails
        return


async def validate_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(api_key_security)
) -> Optional[Dict[str, Any]]:
    """
    Validate API key for external integrations.
    
    **Returns:**
    - **api_key_info**: API key metadata if valid, None if no key provided
    
    **Raises:**
    - **HTTPException**: If API key is invalid
    """
    
    if not credentials:
        return None
    
    try:
        # Extract API key from Bearer token
        api_key = credentials.credentials
        
        # Validate API key format
        if not api_key.startswith("cvd_api_"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "Invalid API key format",
                    "message": "API key must start with 'cvd_api_'"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Hash and verify API key
        # Note: In real implementation, you'd check against database
        # For now, we'll implement a basic validation
        hashed_key = APIKeyManager.hash_api_key(api_key)
        
        # Here you would typically query the database for the API key
        # For this example, we'll create a mock validation
        api_key_info = {
            "api_key_id": "mock_key_id",
            "name": "External Integration",
            "permissions": ["read", "predict"],
            "rate_limit": 1000,  # requests per hour
            "last_used": datetime.utcnow(),
            "is_active": True
        }
        
        logger.info(
            "API key validated",
            api_key_id=api_key_info["api_key_id"],
            permissions=api_key_info["permissions"]
        )
        
        return api_key_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("API key validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "API key validation failed",
                "message": "Unable to validate API key"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user_optional(
    request: Request
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, but don't require authentication.
    
    **Returns:**
    - **user_info**: User information if authenticated, None otherwise
    """
    
    try:
        # Try to get user from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth_header.split(" ")[1]
            )
            user_info = await get_current_user_from_token(credentials)
            return user_info
        
        # Try to get user from API key
        api_key_info = await validate_api_key(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth_header.split(" ")[1] if auth_header else ""
            ) if auth_header else None
        )
        
        if api_key_info:
            return {
                "user_id": api_key_info["api_key_id"],
                "username": api_key_info["name"],
                "auth_type": "api_key",
                "permissions": api_key_info["permissions"]
            }
        
        return None
        
    except Exception:
        # Return None if any authentication method fails
        return None


def get_correlation_id(
    x_correlation_id: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
    request_id: Optional[str] = Header(None)
) -> str:
    """
    Get or generate correlation ID for request tracing.
    
    **Returns:**
    - **correlation_id**: Unique identifier for request tracing
    """
    
    # Try different header names for correlation ID
    correlation_id = x_correlation_id or x_request_id or request_id
    
    if not correlation_id:
        # Generate new correlation ID
        correlation_id = f"cvd-{secrets.token_hex(8)}"
    
    return correlation_id


async def validate_request_size(request: Request) -> None:
    """
    Validate request content size.
    
    **Raises:**
    - **HTTPException**: If request is too large
    """
    
    settings = get_settings()
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        max_size = settings.UPLOAD_MAX_FILE_SIZE
        
        if content_length > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "Request too large",
                    "max_size_mb": round(max_size / (1024 * 1024), 2),
                    "provided_size_mb": round(content_length / (1024 * 1024), 2)
                }
            )


async def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file constraints.
    
    **Parameters:**
    - **file**: Uploaded file to validate
    
    **Raises:**
    - **HTTPException**: If file validation fails
    """
    
    settings = get_settings()
    
    # Check file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid file",
                "message": "Filename is required"
            }
        )
    
    file_extension = file.filename.split('.')[-1].lower()
    allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Unsupported file format",
                "allowed_formats": allowed_extensions,
                "provided_format": file_extension
            }
        )
    
    # Check content type
    allowed_content_types = {
        "csv": ["text/csv", "application/csv"],
        "json": ["application/json", "text/json"],
        "parquet": ["application/octet-stream", "application/parquet"]
    }
    
    expected_types = allowed_content_types.get(file_extension, [])
    if expected_types and file.content_type not in expected_types:
        logger.warning(
            "Content type mismatch",
            filename=file.filename,
            expected_types=expected_types,
            actual_type=file.content_type
        )
        # Don't fail hard on content type mismatch, just log warning


def require_permissions(*required_permissions: str) -> Callable:
    """
    Dependency factory for permission-based access control.
    
    **Parameters:**
    - **required_permissions**: List of required permissions
    
    **Returns:**
    - **Dependency function** that validates user permissions
    """
    
    def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user_from_token)
    ) -> Dict[str, Any]:
        """Check if user has required permissions"""
        
        user_permissions = current_user.get("permissions", [])
        user_roles = current_user.get("roles", [])
        
        # Check if user has any of the required permissions
        has_permission = any(
            perm in user_permissions or perm in user_roles
            for perm in required_permissions
        )
        
        # Admin role bypasses all permission checks
        if "admin" in user_roles:
            has_permission = True
        
        if not has_permission:
            logger.warning(
                "Access denied - insufficient permissions",
                user_id=current_user.get("user_id"),
                required_permissions=required_permissions,
                user_permissions=user_permissions,
                user_roles=user_roles
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "required_permissions": required_permissions,
                    "message": "You don't have permission to access this resource"
                }
            )
        
        return current_user
    
    return permission_dependency


def require_roles(*required_roles: str) -> Callable:
    """
    Dependency factory for role-based access control.
    
    **Parameters:**
    - **required_roles**: List of required roles
    
    **Returns:**
    - **Dependency function** that validates user roles
    """
    
    def role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user_from_token)
    ) -> Dict[str, Any]:
        """Check if user has required roles"""
        
        user_roles = current_user.get("roles", [])
        
        # Check if user has any of the required roles
        has_role = any(role in user_roles for role in required_roles)
        
        if not has_role:
            logger.warning(
                "Access denied - insufficient roles",
                user_id=current_user.get("user_id"),
                required_roles=required_roles,
                user_roles=user_roles
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient roles",
                    "required_roles": required_roles,
                    "message": "You don't have the required role to access this resource"
                }
            )
        
        return current_user
    
    return role_dependency


async def validate_json_payload(request: Request) -> None:
    """
    Validate JSON payload size and structure.
    
    **Raises:**
    - **HTTPException**: If JSON validation fails
    """
    
    try:
        # Check content type
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return  # Skip validation for non-JSON requests
        
        # Validate JSON structure (basic check)
        try:
            body = await request.json()
            
            # Check for common JSON injection patterns
            if isinstance(body, str):
                suspicious_patterns = ["<script", "javascript:", "eval(", "function("]
                if any(pattern in body.lower() for pattern in suspicious_patterns):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "Invalid JSON content",
                            "message": "JSON contains potentially malicious content"
                        }
                    )
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid JSON",
                    "message": f"JSON parsing failed: {str(e)}"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("JSON validation failed", error=str(e))
        # Don't fail hard on validation errors
        return


class RequestValidator:
    """Request validation utilities"""
    
    @staticmethod
    async def validate_pagination_params(
        limit: int,
        offset: int,
        max_limit: int = 1000
    ) -> None:
        """Validate pagination parameters"""
        
        if limit <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid limit",
                    "message": "Limit must be greater than 0"
                }
            )
        
        if limit > max_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Limit too large",
                    "message": f"Limit cannot exceed {max_limit}",
                    "max_limit": max_limit
                }
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid offset",
                    "message": "Offset must be non-negative"
                }
            )
    
    @staticmethod
    async def validate_date_range(
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> None:
        """Validate date range parameters"""
        
        if start_date and end_date:
            if start_date >= end_date:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Invalid date range",
                        "message": "Start date must be before end date"
                    }
                )
            
            # Check if date range is reasonable (not too large)
            max_range = timedelta(days=365 * 5)  # 5 years max
            if end_date - start_date > max_range:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Date range too large",
                        "message": "Date range cannot exceed 5 years",
                        "max_range_days": max_range.days
                    }
                )


def log_api_usage(endpoint: str):
    """
    Decorator to log API endpoint usage for analytics.
    
    **Parameters:**
    - **endpoint**: Endpoint name for logging
    """
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract correlation ID and user info from kwargs
            correlation_id = kwargs.get("correlation_id", "unknown")
            current_user = kwargs.get("current_user")
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(
                    "API endpoint accessed",
                    endpoint=endpoint,
                    correlation_id=correlation_id,
                    user_id=current_user.get("user_id") if current_user else "anonymous",
                    duration=f"{duration:.4f}s",
                    status="success"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "API endpoint error",
                    endpoint=endpoint,
                    correlation_id=correlation_id,
                    user_id=current_user.get("user_id") if current_user else "anonymous",
                    duration=f"{duration:.4f}s",
                    status="error",
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator


class CacheManager:
    """Response caching utilities"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for caching"""
        if not self.redis_client:
            settings = get_settings()
            self.redis_client = redis.from_url(
                str(settings.redis.REDIS_URL),
                decode_responses=True
            )
        return self.redis_client
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        try:
            redis_client = await self._get_redis_client()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                import json
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error("Cache retrieval failed", cache_key=cache_key, error=str(e))
            return None
    
    async def cache_response(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl: int = 3600
    ) -> None:
        """Cache response data"""
        try:
            redis_client = await self._get_redis_client()
            import json
            
            await redis_client.setex(
                cache_key,
                ttl,
                json.dumps(data, default=str)
            )
            
            logger.debug("Response cached", cache_key=cache_key, ttl=ttl)
            
        except Exception as e:
            logger.error("Cache storage failed", cache_key=cache_key, error=str(e))


# Global cache manager instance
cache_manager = CacheManager()


def cached_response(cache_key_template: str, ttl: int = 3600):
    """
    Decorator for caching API responses.
    
    **Parameters:**
    - **cache_key_template**: Template for cache key (supports formatting)
    - **ttl**: Time to live in seconds
    """
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_template.format(**kwargs)
            
            # Try to get cached response
            cached_data = await cache_manager.get_cached_response(cache_key)
            if cached_data:
                logger.debug("Cache hit", cache_key=cache_key)
                return cached_data
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            if isinstance(result, dict):
                await cache_manager.cache_response(cache_key, result, ttl)
                logger.debug("Response cached", cache_key=cache_key)
            
            return result
        
        return wrapper
    return decorator


# Export all dependencies
__all__ = [
    "RateLimiter",
    "rate_limiter",
    "get_rate_limiter",
    "validate_api_key",
    "get_current_user_optional",
    "get_correlation_id",
    "validate_request_size",
    "validate_file_upload",
    "validate_json_payload",
    "require_permissions",
    "require_roles",
    "RequestValidator",
    "log_api_usage",
    "CacheManager",
    "cache_manager",
    "cached_response"
]