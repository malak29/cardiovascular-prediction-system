from typing import AsyncGenerator, Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, MetaData, event, pool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text
import structlog

from app.core.config import get_settings, get_database_url

logger = structlog.get_logger(__name__)

# SQLAlchemy Base class for models
Base = declarative_base()

# Global variables for database connections
async_engine: Optional[AsyncEngine] = None
sync_engine = None
AsyncSessionLocal: Optional[async_sessionmaker] = None
SessionLocal = None

# Metadata for table creation
metadata = MetaData()


class DatabaseManager:
    """Database connection and session manager"""
    
    def __init__(self):
        self.async_engine: Optional[AsyncEngine] = None
        self.sync_engine = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        self.sync_session_factory = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connections"""
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        settings = get_settings()
        database_url = str(settings.database.DATABASE_URL)
        
        try:
            # Create async engine
            self.async_engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=settings.database.DB_POOL_SIZE,
                max_overflow=settings.database.DB_MAX_OVERFLOW,
                pool_timeout=settings.database.DB_POOL_TIMEOUT,
                pool_recycle=settings.database.DB_POOL_RECYCLE,
                pool_pre_ping=True,  # Validate connections before use
                echo=settings.DEBUG,  # Log SQL queries in debug mode
                future=True
            )
            
            # Create sync engine for migrations and other sync operations
            sync_database_url = database_url.replace(
                "postgresql+asyncpg://", "postgresql+psycopg2://"
            )
            self.sync_engine = create_engine(
                sync_database_url,
                poolclass=QueuePool,
                pool_size=settings.database.DB_POOL_SIZE,
                max_overflow=settings.database.DB_MAX_OVERFLOW,
                pool_timeout=settings.database.DB_POOL_TIMEOUT,
                pool_recycle=settings.database.DB_POOL_RECYCLE,
                pool_pre_ping=True,
                echo=settings.DEBUG,
                future=True
            )
            
            # Create session factories
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )
            
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autocommit=False,
                autoflush=False
            )
            
            # Test connections
            await self._test_async_connection()
            self._test_sync_connection()
            
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e), exc_info=True)
            raise
    
    async def _test_async_connection(self) -> None:
        """Test async database connection"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                row = result.fetchone()
                if row[0] != 1:
                    raise Exception("Database connection test failed")
            logger.debug("Async database connection test passed")
        except Exception as e:
            logger.error("Async database connection test failed", error=str(e))
            raise
    
    def _test_sync_connection(self) -> None:
        """Test sync database connection"""
        try:
            with self.sync_engine.begin() as conn:
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                if row[0] != 1:
                    raise Exception("Sync database connection test failed")
            logger.debug("Sync database connection test passed")
        except Exception as e:
            logger.error("Sync database connection test failed", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close database connections"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                logger.debug("Async engine disposed")
            
            if self.sync_engine:
                self.sync_engine.dispose()
                logger.debug("Sync engine disposed")
            
            self._initialized = False
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error("Database session error", error=str(e))
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_sync_session(self) -> Session:
        """Get sync database session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()
        
        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Sync database session error", error=str(e))
            raise
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()


# Database dependency functions for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function to get database session"""
    async with db_manager.get_async_session() as session:
        yield session


async def get_sync_db() -> Session:
    """Dependency function to get sync database session"""
    async with db_manager.get_sync_session() as session:
        yield session


# Database utility functions
async def create_db_and_tables() -> None:
    """Create database and all tables"""
    try:
        await db_manager.initialize()
        
        # Create all tables
        async with db_manager.async_engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models import prediction, patient  # noqa: F401
            
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e), exc_info=True)
        raise


async def close_db_connection() -> None:
    """Close database connections"""
    await db_manager.close()


async def check_db_health() -> Dict[str, Any]:
    """Check database health and return status"""
    try:
        if not db_manager._initialized:
            return {
                "status": "unhealthy",
                "message": "Database not initialized"
            }
        
        # Test async connection
        async with db_manager.async_engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            db_version = result.fetchone()[0]
        
        # Get connection pool status
        pool = db_manager.async_engine.pool
        pool_status = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
        
        return {
            "status": "healthy",
            "database_version": db_version,
            "pool_status": pool_status,
            "message": "Database connection is healthy"
        }
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "message": f"Database health check failed: {str(e)}"
        }


class DatabaseHealthChecker:
    """Utility class for database health monitoring"""
    
    @staticmethod
    async def ping_database() -> bool:
        """Simple database ping"""
        try:
            async with db_manager.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
        except Exception:
            return False
    
    @staticmethod
    async def check_tables_exist() -> Dict[str, bool]:
        """Check if required tables exist"""
        try:
            async with db_manager.async_engine.begin() as conn:
                # Check for main application tables
                tables_to_check = [
                    "patients",
                    "predictions", 
                    "prediction_models",
                    "api_keys",
                    "users"
                ]
                
                table_status = {}
                for table in tables_to_check:
                    result = await conn.execute(
                        text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = '{table}'
                        );
                        """)
                    )
                    table_status[table] = result.fetchone()[0]
                
                return table_status
                
        except Exception as e:
            logger.error("Failed to check table existence", error=str(e))
            return {}


# Database event listeners for logging
@event.listens_for(pool.Pool, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set database connection parameters"""
    if "postgresql" in str(dbapi_connection):
        # Set PostgreSQL specific settings
        cursor = dbapi_connection.cursor()
        cursor.execute("SET timezone TO 'UTC'")
        cursor.close()


@event.listens_for(pool.Pool, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log database connection checkout"""
    logger.debug("Database connection checked out")


@event.listens_for(pool.Pool, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log database connection checkin"""
    logger.debug("Database connection checked in")


# Backward compatibility - maintain global variables
async def init_db():
    """Initialize global database connections (backward compatibility)"""
    global async_engine, sync_engine, AsyncSessionLocal, SessionLocal
    
    await db_manager.initialize()
    
    async_engine = db_manager.async_engine
    sync_engine = db_manager.sync_engine
    AsyncSessionLocal = db_manager.async_session_factory
    SessionLocal = db_manager.sync_session_factory


# Export all database utilities
__all__ = [
    "Base",
    "metadata",
    "DatabaseManager", 
    "db_manager",
    "get_db",
    "get_sync_db",
    "create_db_and_tables",
    "close_db_connection", 
    "check_db_health",
    "DatabaseHealthChecker",
    "init_db",
    # Backward compatibility exports
    "async_engine",
    "sync_engine", 
    "AsyncSessionLocal",
    "SessionLocal"
]