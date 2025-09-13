"""Industry-standard logging utilities and decorators for AstrID."""

import functools
import time
import traceback
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

from core.logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def log_function_call(
    logger_name: str | None = None,
    log_args: bool = True,
    log_result: bool = True,
    log_duration: bool = True,
    log_level: str = "info",
) -> Callable[[F], F]:
    """
    Decorator to log function calls with arguments, results, and duration.

    Args:
        logger_name: Name of the logger (defaults to module name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log execution duration
        log_level: Log level for the function call log
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(logger_name or func.__module__)
            start_time = time.time()

            # Log function entry
            log_msg = f"Calling {func.__name__}"
            if log_args and (args or kwargs):
                # Filter out sensitive data
                safe_args = _sanitize_args(args, kwargs)
                log_msg += f" with args: {safe_args}"

            getattr(logger, log_level)(log_msg)

            try:
                result = await func(*args, **kwargs)

                # Log function exit
                duration = time.time() - start_time
                exit_msg = f"Completed {func.__name__}"
                if log_duration:
                    exit_msg += f" in {duration:.3f}s"
                if log_result and result is not None:
                    exit_msg += f" -> {_sanitize_result(result)}"

                getattr(logger, log_level)(exit_msg)
                return result

            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Failed {func.__name__}"
                if log_duration:
                    error_msg += f" after {duration:.3f}s"
                error_msg += f" with error: {str(e)}"

                logger.error(error_msg)
                logger.debug(f"Exception traceback: {traceback.format_exc()}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(logger_name or func.__module__)
            start_time = time.time()

            # Log function entry
            log_msg = f"Calling {func.__name__}"
            if log_args and (args or kwargs):
                safe_args = _sanitize_args(args, kwargs)
                log_msg += f" with args: {safe_args}"

            getattr(logger, log_level)(log_msg)

            try:
                result = func(*args, **kwargs)

                # Log function exit
                duration = time.time() - start_time
                exit_msg = f"Completed {func.__name__}"
                if log_duration:
                    exit_msg += f" in {duration:.3f}s"
                if log_result and result is not None:
                    exit_msg += f" -> {_sanitize_result(result)}"

                getattr(logger, log_level)(exit_msg)
                return result

            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Failed {func.__name__}"
                if log_duration:
                    error_msg += f" after {duration:.3f}s"
                error_msg += f" with error: {str(e)}"

                logger.error(error_msg)
                logger.debug(f"Exception traceback: {traceback.format_exc()}")
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        else:
            return sync_wrapper  # type: ignore[return-value]

    return decorator


def log_database_operation(operation: str, entity_type: str) -> Callable[[F], F]:
    """
    Decorator specifically for database operations.

    Args:
        operation: Type of operation (create, read, update, delete)
        entity_type: Type of entity being operated on
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start_time = time.time()

            # Extract entity ID if available
            entity_id = _extract_entity_id(args, kwargs)
            entity_info = f"{entity_type}"
            if entity_id:
                entity_info += f" (id={entity_id})"

            logger.info(f"Starting {operation} operation on {entity_info}")

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                if result:
                    logger.info(
                        f"Successfully completed {operation} operation on {entity_info} in {duration:.3f}s"
                    )
                else:
                    logger.warning(
                        f"{operation.capitalize()} operation on {entity_info} returned no result"
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {operation} operation on {entity_info} after {duration:.3f}s: {str(e)}"
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_api_request(method: str, endpoint: str) -> Callable[[F], F]:
    """
    Decorator for API request logging.

    Args:
        method: HTTP method
        endpoint: API endpoint
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start_time = time.time()

            # Extract request info
            request_id = _extract_request_id(args, kwargs)
            user_id = _extract_user_id(args, kwargs)

            request_info = f"{method} {endpoint}"
            if request_id:
                request_info += f" (request_id={request_id})"
            if user_id:
                request_info += f" (user_id={user_id})"

            logger.info(f"API request started: {request_info}")

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                status_code = _extract_status_code(result)
                logger.info(
                    f"API request completed: {request_info} -> {status_code} in {duration:.3f}s"
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"API request failed: {request_info} after {duration:.3f}s: {str(e)}"
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_business_operation(operation: str, domain: str) -> Callable[[F], F]:
    """
    Decorator for business logic operations.

    Args:
        operation: Business operation name
        domain: Domain name
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(f"{domain}.{func.__module__.split('.')[-1]}")
            start_time = time.time()

            logger.info(f"Starting business operation: {operation}")

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"Completed business operation: {operation} in {duration:.3f}s"
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed business operation: {operation} after {duration:.3f}s: {str(e)}"
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_performance_metrics(threshold_seconds: float = 1.0) -> Callable[[F], F]:
    """
    Decorator to log performance metrics for slow operations.

    Args:
        threshold_seconds: Log warning if operation takes longer than this
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                if duration > threshold_seconds:
                    logger.warning(
                        f"Slow operation detected: {func.__name__} took {duration:.3f}s (threshold: {threshold_seconds}s)"
                    )
                else:
                    logger.debug(
                        f"Operation completed: {func.__name__} in {duration:.3f}s"
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation failed: {func.__name__} after {duration:.3f}s: {str(e)}"
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_data_processing(stage: str, data_type: str) -> Callable[[F], F]:
    """
    Decorator for data processing operations.

    Args:
        stage: Processing stage (ingestion, preprocessing, analysis, etc.)
        data_type: Type of data being processed
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start_time = time.time()

            # Extract data identifiers
            data_id = _extract_data_id(args, kwargs)
            data_info = f"{data_type}"
            if data_id:
                data_info += f" (id={data_id})"

            logger.info(f"Starting {stage} processing: {data_info}")

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Log processing results
                if isinstance(result, dict) and "count" in result:
                    logger.info(
                        f"Completed {stage} processing: {data_info} -> {result['count']} items in {duration:.3f}s"
                    )
                else:
                    logger.info(
                        f"Completed {stage} processing: {data_info} in {duration:.3f}s"
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {stage} processing: {data_info} after {duration:.3f}s: {str(e)}"
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# Helper functions for data sanitization and extraction


def _sanitize_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Sanitize function arguments to remove sensitive data."""
    sensitive_keys = {"password", "token", "secret", "key", "auth", "credential"}

    sanitized = {}

    # Handle positional arguments
    for i, arg in enumerate(args):
        if isinstance(arg, str | int | float | bool | UUID):
            sanitized[f"arg_{i}"] = arg
        elif isinstance(arg, dict):
            sanitized[f"arg_{i}"] = {
                k: "***" if any(sk in k.lower() for sk in sensitive_keys) else v
                for k, v in arg.items()
            }  # type: ignore[assignment]
        else:
            sanitized[f"arg_{i}"] = str(type(arg).__name__)

    # Handle keyword arguments
    for key, value in kwargs.items():
        if any(sk in key.lower() for sk in sensitive_keys):
            sanitized[key] = "***"
        elif isinstance(value, str | int | float | bool | UUID):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = {
                k: "***" if any(sk in k.lower() for sk in sensitive_keys) else v
                for k, v in value.items()
            }  # type: ignore[assignment]
        else:
            sanitized[key] = str(type(value).__name__)

    return sanitized


def _sanitize_result(result: Any) -> str:
    """Sanitize function result for logging."""
    if isinstance(result, str | int | float | bool):
        return str(result)
    elif isinstance(result, UUID):
        return str(result)
    elif isinstance(result, dict):
        if "id" in result:
            return f"{{id: {result['id']}, ...}}"
        elif len(result) <= 3:
            return str(result)
        else:
            return f"{{keys: {list(result.keys())}}}"
    elif isinstance(result, list):
        return f"[{len(result)} items]"
    else:
        return str(type(result).__name__)


def _extract_entity_id(args: tuple, kwargs: dict) -> str | None:
    """Extract entity ID from function arguments."""
    # Look for common ID patterns
    for arg in args:
        if isinstance(arg, str | UUID):
            return str(arg)
        elif isinstance(arg, dict) and "id" in arg:
            return str(arg["id"])

    for key, value in kwargs.items():
        if "id" in key.lower() and isinstance(value, str | UUID):
            return str(value)

    return None


def _extract_request_id(args: tuple, kwargs: dict) -> str | None:
    """Extract request ID from function arguments."""
    for arg in args:
        if hasattr(arg, "headers") and "x-request-id" in arg.headers:
            return arg.headers["x-request-id"]  # type: ignore[no-any-return]

    return None


def _extract_user_id(args: tuple, kwargs: dict) -> str | None:
    """Extract user ID from function arguments."""
    for arg in args:
        if hasattr(arg, "user") and hasattr(arg.user, "id"):
            return str(arg.user.id)
        elif hasattr(arg, "current_user") and hasattr(arg.current_user, "id"):
            return str(arg.current_user.id)

    return None


def _extract_status_code(result: Any) -> str:
    """Extract HTTP status code from result."""
    if hasattr(result, "status_code"):
        return str(result.status_code)
    elif isinstance(result, dict) and "status_code" in result:
        return str(result["status_code"])
    else:
        return "200"  # Default assumption


def _extract_data_id(args: tuple, kwargs: dict) -> str | None:
    """Extract data ID from function arguments."""
    # Look for common data ID patterns
    for arg in args:
        if isinstance(arg, str | UUID):
            return str(arg)
        elif isinstance(arg, dict):
            for key in ["id", "observation_id", "survey_id", "detection_id"]:
                if key in arg:
                    return str(arg[key])

    for key, value in kwargs.items():
        if any(
            id_key in key.lower()
            for id_key in ["id", "observation_id", "survey_id", "detection_id"]
        ):
            return str(value)

    return None


# Convenience functions for common logging patterns


def log_audit_event(
    action: str, resource_type: str, resource_id: str, user_id: str | None = None
) -> None:
    """Log an audit event."""
    logger = get_logger("audit")
    event_info = (
        f"action={action}, resource_type={resource_type}, resource_id={resource_id}"
    )
    if user_id:
        event_info += f", user_id={user_id}"
    logger.info(f"AUDIT: {event_info}")


def log_security_event(
    event_type: str, details: str, severity: str = "warning"
) -> None:
    """Log a security event."""
    logger = get_logger("security")
    getattr(logger, severity)(f"SECURITY: {event_type} - {details}")


def log_system_health(component: str, status: str, details: str | None = None) -> None:
    """Log system health status."""
    logger = get_logger("system.health")
    health_info = f"component={component}, status={status}"
    if details:
        health_info += f", details={details}"
    logger.info(f"HEALTH: {health_info}")


def log_metrics(
    metric_name: str, value: float, unit: str = "", tags: dict[str, str] | None = None
) -> None:
    """Log a metric value."""
    logger = get_logger("metrics")
    metric_info = f"metric={metric_name}, value={value}"
    if unit:
        metric_info += f", unit={unit}"
    if tags:
        tag_str = ", ".join(f"{k}={v}" for k, v in tags.items())
        metric_info += f", tags=[{tag_str}]"
    logger.info(f"METRIC: {metric_info}")
