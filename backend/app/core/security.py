from datetime import datetime, timedelta
from typing import Any, Optional, Union, Dict
import secrets
import hashlib

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize HTTP Bearer for JWT authentication
security = HTTPBearer(auto_error=False)


class SecurityError(Exception):
    """Base exception for security-related errors"""
    pass


class TokenGenerator:
    """Utility class for generating secure tokens and keys"""
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a correlation ID for request tracing"""
        return f"cvd-{secrets.token_hex(8)}"
    
    @staticmethod
    def hash_string(value: str, salt: Optional[str] = None) -> str:
        """Hash a string value with optional salt"""
        if salt:
            value = f"{value}{salt}"
        return hashlib.sha256(value.encode()).hexdigest()


class PasswordManager:
    """Password hashing and verification utilities"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        settings = get_settings()
        
        # Configure bcrypt rounds based on settings
        pwd_context.update(bcrypt__rounds=settings.security.BCRYPT_ROUNDS)
        
        try:
            hashed = pwd_context.hash(password)
            logger.debug("Password hashed successfully")
            return hashed
        except Exception as e:
            logger.error("Failed to hash password", error=str(e))
            raise SecurityError("Password hashing failed")
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            is_valid = pwd_context.verify(plain_password, hashed_password)
            logger.debug("Password verification completed", is_valid=is_valid)
            return is_valid
        except Exception as e:
            logger.error("Failed to verify password", error=str(e))
            return False
    
    @staticmethod
    def check_password_strength(password: str) -> Dict[str, Any]:
        """Check password strength and return validation results"""
        checks = {
            "length": len(password) >= 8,
            "uppercase": any(c.isupper() for c in password),
            "lowercase": any(c.islower() for c in password),
            "digit": any(c.isdigit() for c in password),
            "special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        }
        
        score = sum(checks.values())
        strength = "weak"
        if score >= 4:
            strength = "strong"
        elif score >= 3:
            strength = "medium"
        
        return {
            "strength": strength,
            "score": score,
            "checks": checks,
            "is_valid": score >= 3
        }


class JWTManager:
    """JWT token generation and validation"""
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        settings = get_settings()
        
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_hex(16)  # JWT ID for tracking
        })
        
        try:
            encoded_jwt = jwt.encode(
                to_encode,
                settings.security.JWT_SECRET_KEY,
                algorithm=settings.security.JWT_ALGORITHM
            )
            
            logger.debug("Access token created", expires_at=expire.isoformat())
            return encoded_jwt
            
        except Exception as e:
            logger.error("Failed to create access token", error=str(e))
            raise SecurityError("Token creation failed")
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create a JWT refresh token"""
        settings = get_settings()
        
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(
            days=settings.security.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_hex(16)
        })
        
        try:
            encoded_jwt = jwt.encode(
                to_encode,
                settings.security.JWT_SECRET_KEY,
                algorithm=settings.security.JWT_ALGORITHM
            )
            
            logger.debug("Refresh token created", expires_at=expire.isoformat())
            return encoded_jwt
            
        except Exception as e:
            logger.error("Failed to create refresh token", error=str(e))
            raise SecurityError("Refresh token creation failed")
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token"""
        settings = get_settings()
        
        try:
            payload = jwt.decode(
                token,
                settings.security.JWT_SECRET_KEY,
                algorithms=[settings.security.JWT_ALGORITHM]
            )
            
            # Validate token type
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                raise SecurityError("Invalid token type")
            
            logger.debug("Token decoded successfully", token_type=token_type)
            return payload
            
        except JWTError as e:
            logger.warning("JWT validation failed", error=str(e))
            raise SecurityError("Invalid token")
        except Exception as e:
            logger.error("Token decoding failed", error=str(e))
            raise SecurityError("Token processing failed")


class APIKeyManager:
    """API key generation and validation"""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key"""
        return f"cvd_api_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash an API key for storage"""
        settings = get_settings()
        salt = settings.security.SECRET_KEY
        return hashlib.sha256(f"{api_key}{salt}".encode()).hexdigest()
    
    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash"""
        try:
            computed_hash = APIKeyManager.hash_api_key(api_key)
            return secrets.compare_digest(computed_hash, hashed_key)
        except Exception as e:
            logger.error("API key verification failed", error=str(e))
            return False


# Dependency functions for FastAPI
async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Extract current user from JWT token"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not credentials:
        raise credentials_exception
    
    try:
        payload = JWTManager.decode_token(credentials.credentials)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
        
        # Validate token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "user_id": user_id,
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "token_jti": payload.get("jti")
        }
        
    except SecurityError:
        raise credentials_exception
    except Exception as e:
        logger.error("Token validation failed", error=str(e))
        raise credentials_exception


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user_from_token)
) -> Dict[str, Any]:
    """Get current active user with additional validation"""
    
    # Here you would typically check if the user is active in the database
    # For now, we'll assume all users are active
    
    if not current_user.get("user_id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user data"
        )
    
    return current_user


# Security utility functions
def create_password_reset_token(email: str) -> str:
    """Create a password reset token"""
    settings = get_settings()
    
    data = {
        "sub": email,
        "type": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    }
    
    return jwt.encode(
        data,
        settings.security.SECRET_KEY,
        algorithm=settings.security.JWT_ALGORITHM
    )


def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email"""
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.security.SECRET_KEY,
            algorithms=[settings.security.JWT_ALGORITHM]
        )
        
        if payload.get("type") != "password_reset":
            return None
        
        return payload.get("sub")
        
    except JWTError:
        return None


def generate_secure_filename(original_filename: str) -> str:
    """Generate a secure filename for uploaded files"""
    # Extract file extension
    if "." in original_filename:
        name, ext = original_filename.rsplit(".", 1)
        ext = ext.lower()
    else:
        name, ext = original_filename, ""
    
    # Generate secure name
    secure_name = f"{secrets.token_hex(16)}"
    
    return f"{secure_name}.{ext}" if ext else secure_name


# Export all security utilities
__all__ = [
    "SecurityError",
    "TokenGenerator",
    "PasswordManager", 
    "JWTManager",
    "APIKeyManager",
    "get_current_user_from_token",
    "get_current_active_user",
    "create_password_reset_token",
    "verify_password_reset_token",
    "generate_secure_filename"
]