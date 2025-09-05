"""Examples of how to use the AstrID logging system."""

from uuid import UUID

from core.logging import configure_domain_logger
from core.logging_utils import (
    log_api_request,
    log_audit_event,
    log_business_operation,
    log_data_processing,
    log_database_operation,
    log_function_call,
    log_metrics,
    log_performance_metrics,
    log_security_event,
    log_system_health,
)
from sqlalchemy.ext.asyncio import AsyncSession


# Example 1: Basic domain logging
class ExampleService:
    """Example service showing basic logging patterns."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = configure_domain_logger("example.service")

    async def create_entity(self, data: dict) -> dict:
        """Create an entity with comprehensive logging."""
        self.logger.info(f"Creating entity with data: {data}")

        try:
            # Simulate database operation
            entity_id = UUID("12345678-1234-5678-9012-123456789012")
            self.logger.info(f"Successfully created entity: id={entity_id}")

            # Log audit event
            log_audit_event("CREATE", "entity", str(entity_id))

            return {"id": entity_id, "status": "created"}

        except Exception as e:
            self.logger.error(f"Failed to create entity: error={str(e)}")
            raise


# Example 2: Using decorators for automatic logging
class DecoratedService:
    """Example service using logging decorators."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = configure_domain_logger("example.decorated")

    @log_function_call(log_args=True, log_result=True, log_duration=True)  # type: ignore[misc]
    async def process_data(self, data: list[dict], algorithm: str = "standard") -> dict:
        """Process data with automatic function call logging."""
        self.logger.info(f"Processing {len(data)} items with algorithm: {algorithm}")

        # Simulate processing
        processed_count = len(data)

        # Log metrics
        log_metrics(
            "data_processed", processed_count, "items", {"algorithm": algorithm}
        )

        return {"processed_count": processed_count, "algorithm": algorithm}

    @log_database_operation("create", "observation")  # type: ignore[misc]
    async def create_observation(self, observation_data: dict) -> dict:
        """Create observation with database operation logging."""
        # Simulate database creation
        observation_id = UUID("87654321-4321-8765-2109-876543210987")
        return {"id": observation_id, "data": observation_data}

    @log_api_request("POST", "/api/observations")  # type: ignore[misc]
    async def api_create_observation(self, request_data: dict) -> dict:
        """API endpoint with request logging."""
        # Simulate API processing
        return {"status": "created", "data": request_data}

    @log_business_operation("data_validation", "observations")  # type: ignore[misc]
    async def validate_observation_data(self, data: dict) -> dict:
        """Business logic with operation logging."""
        # Simulate validation
        is_valid = len(data) > 0
        if not is_valid:
            self.logger.warning("Observation data validation failed: empty data")

        return {"is_valid": is_valid, "errors": [] if is_valid else ["Empty data"]}

    @log_performance_metrics(threshold_seconds=0.5)  # type: ignore[misc]
    async def slow_operation(self, items: list[str]) -> dict:
        """Operation that might be slow."""
        import asyncio

        # Simulate slow operation
        await asyncio.sleep(0.6)  # This will trigger the performance warning
        return {"processed": len(items)}

    @log_data_processing("preprocessing", "astronomical_image")  # type: ignore[misc]
    async def preprocess_image(self, image_id: str, algorithm: str) -> dict:
        """Data processing with stage logging."""
        # Simulate image preprocessing
        self.logger.info(f"Applying {algorithm} preprocessing to image {image_id}")
        return {"image_id": image_id, "algorithm": algorithm, "status": "processed"}


# Example 3: Security and audit logging
class SecurityExample:
    """Example showing security and audit logging."""

    def __init__(self) -> None:
        self.logger = configure_domain_logger("security.example")

    async def authenticate_user(self, username: str, password: str) -> dict:
        """Authenticate user with security logging."""
        self.logger.info(f"Authentication attempt for user: {username}")

        # Simulate authentication
        is_valid = username == "valid_user" and password == "valid_password"

        if is_valid:
            self.logger.info(f"Successful authentication for user: {username}")
            log_audit_event("LOGIN", "user", username)
        else:
            self.logger.warning(f"Failed authentication attempt for user: {username}")
            log_security_event(
                "AUTHENTICATION_FAILURE", f"Invalid credentials for user: {username}"
            )

        return {"authenticated": is_valid, "user": username if is_valid else None}

    async def check_permissions(self, user_id: str, resource: str, action: str) -> dict:
        """Check user permissions with audit logging."""
        self.logger.debug(
            f"Checking permissions: user={user_id}, resource={resource}, action={action}"
        )

        # Simulate permission check
        has_permission = True  # Simplified for example

        if has_permission:
            log_audit_event("PERMISSION_GRANTED", resource, user_id)
        else:
            log_security_event(
                "PERMISSION_DENIED", f"User {user_id} denied access to {resource}"
            )

        return {"has_permission": has_permission}


# Example 4: System health and metrics
class SystemHealthExample:
    """Example showing system health and metrics logging."""

    def __init__(self) -> None:
        self.logger = configure_domain_logger("system.health")

    async def check_database_health(self) -> dict:
        """Check database health status."""
        try:
            # Simulate database health check
            is_healthy = True
            response_time = 0.05  # seconds

            if is_healthy:
                log_system_health(
                    "database", "healthy", f"response_time={response_time}s"
                )
                log_metrics("database_response_time", response_time, "seconds")
            else:
                log_system_health("database", "unhealthy", "connection failed")

            return {"healthy": is_healthy, "response_time": response_time}

        except Exception as e:
            log_system_health("database", "error", str(e))
            raise

    async def check_storage_health(self) -> dict:
        """Check storage system health."""
        # Simulate storage check
        free_space_gb = 100.5
        total_space_gb = 1000.0

        usage_percent = ((total_space_gb - free_space_gb) / total_space_gb) * 100

        if usage_percent > 90:
            log_system_health("storage", "warning", f"usage={usage_percent:.1f}%")
        else:
            log_system_health("storage", "healthy", f"usage={usage_percent:.1f}%")

        log_metrics("storage_usage_percent", usage_percent, "percent")
        log_metrics("storage_free_space", free_space_gb, "GB")

        return {"free_space_gb": free_space_gb, "usage_percent": usage_percent}


# Example 5: Error handling and exception logging
class ErrorHandlingExample:
    """Example showing proper error handling and logging."""

    def __init__(self) -> None:
        self.logger = configure_domain_logger("error.example")

    async def risky_operation(self, data: dict) -> dict:
        """Operation that might fail with proper error logging."""
        self.logger.info(f"Starting risky operation with data: {data}")

        try:
            # Simulate risky operation
            if "error" in data:
                raise ValueError("Simulated error for demonstration")

            result = {"processed": True, "data": data}
            self.logger.info("Risky operation completed successfully")
            return result

        except ValueError as e:
            self.logger.error(f"Validation error in risky operation: {str(e)}")
            log_security_event("VALIDATION_ERROR", f"Invalid data provided: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in risky operation: {str(e)}")
            self.logger.debug(f"Exception details: {e.__class__.__name__}: {str(e)}")
            raise

    async def retry_operation(self, max_retries: int = 3) -> dict:
        """Operation with retry logic and logging."""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Retry attempt {attempt + 1}/{max_retries}")

                # Simulate operation that might fail
                if attempt < max_retries - 1:
                    raise ConnectionError("Simulated connection error")

                self.logger.info("Operation succeeded after retries")
                return {"success": True, "attempts": attempt + 1}

            except ConnectionError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise

        # This should never be reached, but mypy needs it
        raise ConnectionError("All retry attempts failed")


# Example usage function
async def demonstrate_logging() -> None:
    """Demonstrate all logging features."""
    from sqlalchemy.ext.asyncio import AsyncSession

    # Initialize services
    db = AsyncSession()  # This would be a real database session
    example_service = ExampleService(db)
    decorated_service = DecoratedService(db)
    security_example = SecurityExample()
    health_example = SystemHealthExample()
    error_example = ErrorHandlingExample()

    # Demonstrate basic logging
    await example_service.create_entity({"name": "test", "value": 42})

    # Demonstrate decorated logging
    await decorated_service.process_data([{"id": 1}, {"id": 2}], "advanced")
    await decorated_service.create_observation({"ra": 180.0, "dec": 30.0})
    await decorated_service.validate_observation_data({"data": "valid"})
    await decorated_service.slow_operation(["item1", "item2"])
    await decorated_service.preprocess_image("img_123", "bias_correction")

    # Demonstrate security logging
    await security_example.authenticate_user("valid_user", "valid_password")
    await security_example.authenticate_user("invalid_user", "wrong_password")
    await security_example.check_permissions("user_123", "observations", "read")

    # Demonstrate health and metrics
    await health_example.check_database_health()
    await health_example.check_storage_health()

    # Demonstrate error handling
    await error_example.risky_operation({"data": "valid"})
    try:
        await error_example.risky_operation({"error": "trigger"})
    except ValueError:
        pass  # Expected error

    await error_example.retry_operation()


if __name__ == "__main__":
    import asyncio

    asyncio.run(demonstrate_logging())
