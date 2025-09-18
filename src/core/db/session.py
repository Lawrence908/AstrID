"""Database session management.

This module handles SQLAlchemy async session management and database connection
configuration. It provides:
1. Async engine configuration with connection pooling
2. Session factory for creating database sessions
3. FastAPI dependencies for session management
4. Connection testing utilities

The configuration uses environment variables and supports SSL connections.

Next Steps:
1. Implement session lifecycle hooks for metrics/logging
2. Add connection retry logic for improved resilience
"""

import logging
import os
import ssl
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.constants import (
    DATABASE_TEST_URL,
    DB_CONFIG,
    LOG_LEVEL,
    get_database_url,
    get_db_config,
)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)

# Create SSL context with proper verification
ssl_context = None
try:
    cert_path = DB_CONFIG.get("ssl_cert_path")
    if cert_path and isinstance(cert_path, str) and os.path.exists(cert_path):
        logger.debug(f"Using SSL certificate at {cert_path}")
        ssl_context = ssl.create_default_context(cafile=cert_path)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
        logger.info("SSL context created successfully with certificate verification")
    else:
        logger.info("No SSL certificate path provided, using default SSL context")
        ssl_context = ssl.create_default_context()
        logger.info("Using default SSL context with system certificate store")
except Exception as e:
    logger.error(f"Failed to create SSL context: {str(e)}", exc_info=True)
    # Fallback to default SSL context
    ssl_context = ssl.create_default_context()
    logger.warning("Using fallback SSL context due to error")

# Create async engine with SSL context and connection pooling
try:
    db_url = get_database_url()
    # Mask password in logs
    password = DB_CONFIG.get("password")
    masked_url = (
        db_url.replace(password, "****")
        if password and isinstance(password, str)
        else db_url
    )
    logger.info(f"Creating database engine with URL: {masked_url}")

    db_config = get_db_config()
    engine = create_async_engine(
        db_url,
        echo=db_config["echo"],
        pool_pre_ping=db_config["pool_pre_ping"],
        pool_recycle=db_config["pool_recycle"],
        pool_size=db_config["pool_size"],
        max_overflow=db_config["max_overflow"],
        pool_timeout=db_config["pool_timeout"],
        future=True,  # Enable SQLAlchemy 2.0 style
        connect_args={
            "ssl": ssl_context,
            "command_timeout": db_config.get("command_timeout", 60),
            "server_settings": {
                "application_name": "astrid_app",
                "client_encoding": "utf8",
                "timezone": "UTC",
            },
        },
    )
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}", exc_info=True)
    raise

# Create async session factory with optimized settings
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autocommit=False,  # Explicitly set autocommit to False
    autoflush=False,  # Disable autoflush to prevent unexpected queries
    future=True,  # Enable SQLAlchemy 2.0 style
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides an async database session.

    Includes connection pool monitoring and proper error handling for Supabase limits.
    """
    session = None
    try:
        session = AsyncSessionLocal()
        # Test connection before yielding
        await session.execute(text("SELECT 1"))
        yield session
    except Exception as e:
        logger.error(f"Database session error: {str(e)}", exc_info=True)
        if session:
            await session.rollback()
        # Check if it's a connection pool exhaustion error
        if "MaxClientsInSessionMode" in str(e) or "max clients reached" in str(e):
            logger.warning(
                "Supabase connection pool exhausted - consider reducing pool sizes"
            )
        raise
    finally:
        if session:
            await session.close()


async def get_test_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get test database session."""
    if not DATABASE_TEST_URL:
        raise ValueError("Test database URL not configured")

    logger.info("Creating test database engine")
    db_config = get_db_config()
    test_engine = create_async_engine(
        DATABASE_TEST_URL,
        echo=False,
        pool_pre_ping=db_config["pool_pre_ping"],
        pool_size=db_config["pool_size"],
        max_overflow=db_config["max_overflow"],
        future=True,
        connect_args={
            "command_timeout": db_config.get("command_timeout", 60),
            "server_settings": {
                "application_name": "astrid_test",
                "client_encoding": "utf8",
                "timezone": "UTC",
            },
        },
    )

    TestSessionLocal = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
        future=True,
    )

    async with TestSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Test database session error: {str(e)}", exc_info=True)
            await session.rollback()
            raise
        finally:
            await session.close()
            await test_engine.dispose()


async def test_connection() -> bool:
    """Test the database connection by executing a simple query.

    Returns:
        bool: True if connection successful, False otherwise

    Logs detailed error information if connection fails.
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to database")
            if result.scalar() == 1:
                logger.info("Connection test successful")
            else:
                logger.error("Connection test failed: unexpected result")
                return False
        logger.debug("Connection test executed successfully")
        return True

    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}", exc_info=True)
        logger.error("Detailed error information:")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__}")
        return False


async def check_pool_health() -> dict[str, Any]:
    """Check the health of the connection pool.

    Returns:
        dict: Health check results including:
            - pool_size: Current size of the connection pool
            - overflow: Number of overflow connections in use
            - checkedout: Number of connections currently checked out
            - response_time: Time taken to get and return a connection
    """
    try:
        start_time = datetime.now()

        # Get pool statistics
        pool = engine.pool
        stats = {
            "pool_size": getattr(pool, "size", lambda: 0)(),
            "overflow": getattr(pool, "overflow", lambda: 0)(),
            "checkedout": getattr(pool, "checkedout", lambda: 0)(),
        }

        # Test getting a connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        stats["response_time"] = response_time

        logger.info(f"Pool health check successful: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Pool health check failed: {str(e)}")
        # Check for Supabase-specific connection errors
        if "MaxClientsInSessionMode" in str(e) or "max clients reached" in str(e):
            logger.warning(
                "Supabase connection pool exhausted - consider reducing pool sizes"
            )
            raise HTTPException(
                status_code=503,
                detail="Database connection pool exhausted. Please try again later.",
            ) from e
        raise HTTPException(
            status_code=503, detail=f"Database pool health check failed: {str(e)}"
        ) from e


async def get_connection_pool_status() -> dict[str, Any]:
    """Get current connection pool status for monitoring.

    Returns:
        dict: Pool status including size, checked out connections, and overflow
    """
    try:
        pool = engine.pool
        pool_size = getattr(pool, "size", lambda: 0)()
        overflow = getattr(pool, "overflow", lambda: 0)()
        checked_out = getattr(pool, "checkedout", lambda: 0)()
        checked_in = getattr(pool, "checkedin", lambda: 0)()

        return {
            "pool_size": pool_size,
            "checked_out": checked_out,
            "overflow": overflow,
            "checked_in": checked_in,
            "total_connections": pool_size + overflow,
            "status": "healthy"
            if checked_out < (pool_size + overflow)
            else "exhausted",
        }
    except Exception as e:
        logger.error(f"Failed to get pool status: {str(e)}")
        return {"error": str(e), "status": "error"}
