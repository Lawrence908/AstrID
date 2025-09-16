"""Survey integration specific exceptions."""

from typing import Any


class SurveyIntegrationError(Exception):
    """Base exception for survey integration errors."""

    def __init__(
        self,
        message: str,
        survey_name: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize survey integration error.

        Args:
            message: Error message
            survey_name: Name of the survey that caused the error
            error_code: Specific error code
            details: Additional error details
        """
        super().__init__(message)
        self.survey_name = survey_name
        self.error_code = error_code
        self.details = details or {}


class SurveyAPIError(SurveyIntegrationError):
    """Error when communicating with external survey APIs."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        api_endpoint: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        """Initialize survey API error.

        Args:
            message: Error message
            survey_name: Name of the survey
            api_endpoint: API endpoint that failed
            status_code: HTTP status code
            response_body: Response body from the API
        """
        details = {
            "api_endpoint": api_endpoint,
            "status_code": status_code,
            "response_body": response_body,
        }
        super().__init__(message, survey_name, "API_ERROR", details)


class SurveyRateLimitError(SurveyAPIError):
    """Error when hitting rate limits on survey APIs."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        retry_after: int | None = None,
        request_count: int | None = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            survey_name: Name of the survey
            retry_after: Suggested retry delay in seconds
            request_count: Number of requests made
        """
        super().__init__(message, survey_name)
        self.error_code = "RATE_LIMIT_EXCEEDED"
        self.details.update(
            {
                "retry_after": retry_after,
                "request_count": request_count,
            }
        )


class SurveyTimeoutError(SurveyAPIError):
    """Error when survey API requests timeout."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        timeout_duration: float,
        operation: str | None = None,
    ):
        """Initialize timeout error.

        Args:
            message: Error message
            survey_name: Name of the survey
            timeout_duration: Timeout duration in seconds
            operation: Operation that timed out
        """
        super().__init__(message, survey_name)
        self.error_code = "REQUEST_TIMEOUT"
        self.details.update(
            {
                "timeout_duration": timeout_duration,
                "operation": operation,
            }
        )


class SurveyDataError(SurveyIntegrationError):
    """Error when survey data is invalid or corrupted."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        data_type: str | None = None,
        validation_errors: list[str] | None = None,
    ):
        """Initialize data error.

        Args:
            message: Error message
            survey_name: Name of the survey
            data_type: Type of data that failed validation
            validation_errors: List of specific validation errors
        """
        super().__init__(message, survey_name, "DATA_VALIDATION_ERROR")
        self.details.update(
            {
                "data_type": data_type,
                "validation_errors": validation_errors or [],
            }
        )


class SurveyConfigurationError(SurveyIntegrationError):
    """Error in survey configuration."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        config_key: str | None = None,
        config_value: Any = None,
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            survey_name: Name of the survey
            config_key: Configuration key that failed
            config_value: Configuration value that failed
        """
        super().__init__(message, survey_name, "CONFIGURATION_ERROR")
        self.details.update(
            {
                "config_key": config_key,
                "config_value": config_value,
            }
        )


class SurveyAuthenticationError(SurveyAPIError):
    """Error in survey API authentication."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        auth_method: str | None = None,
    ):
        """Initialize authentication error.

        Args:
            message: Error message
            survey_name: Name of the survey
            auth_method: Authentication method used
        """
        super().__init__(message, survey_name)
        self.error_code = "AUTHENTICATION_ERROR"
        self.details.update(
            {
                "auth_method": auth_method,
            }
        )


class SurveyDataNotFoundError(SurveyAPIError):
    """Error when requested survey data is not found."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        observation_id: str | None = None,
        coordinates: tuple[float, float] | None = None,
        search_radius: float | None = None,
    ):
        """Initialize data not found error.

        Args:
            message: Error message
            survey_name: Name of the survey
            observation_id: Observation ID that was not found
            coordinates: Search coordinates
            search_radius: Search radius
        """
        super().__init__(message, survey_name)
        self.error_code = "DATA_NOT_FOUND"
        self.details.update(
            {
                "observation_id": observation_id,
                "coordinates": coordinates,
                "search_radius": search_radius,
            }
        )


class AdapterError(SurveyIntegrationError):
    """Error in survey data adapters."""

    def __init__(
        self,
        message: str,
        adapter_name: str,
        operation: str | None = None,
        raw_data: dict[str, Any] | None = None,
    ):
        """Initialize adapter error.

        Args:
            message: Error message
            adapter_name: Name of the adapter
            operation: Operation that failed
            raw_data: Raw data that failed processing
        """
        super().__init__(message, adapter_name, "ADAPTER_ERROR")
        self.details.update(
            {
                "operation": operation,
                "raw_data_keys": list(raw_data.keys()) if raw_data else None,
            }
        )


class MetadataExtractionError(SurveyIntegrationError):
    """Error in metadata extraction."""

    def __init__(
        self,
        message: str,
        file_type: str | None = None,
        extraction_stage: str | None = None,
        file_size: int | None = None,
    ):
        """Initialize metadata extraction error.

        Args:
            message: Error message
            file_type: Type of file being processed
            extraction_stage: Stage of extraction that failed
            file_size: Size of file in bytes
        """
        super().__init__(message, None, "METADATA_EXTRACTION_ERROR")
        self.details.update(
            {
                "file_type": file_type,
                "extraction_stage": extraction_stage,
                "file_size": file_size,
            }
        )


class SurveyServiceUnavailableError(SurveyAPIError):
    """Error when survey service is temporarily unavailable."""

    def __init__(
        self,
        message: str,
        survey_name: str,
        estimated_downtime: int | None = None,
        service_status_url: str | None = None,
    ):
        """Initialize service unavailable error.

        Args:
            message: Error message
            survey_name: Name of the survey
            estimated_downtime: Estimated downtime in seconds
            service_status_url: URL for service status information
        """
        super().__init__(message, survey_name)
        self.error_code = "SERVICE_UNAVAILABLE"
        self.details.update(
            {
                "estimated_downtime": estimated_downtime,
                "service_status_url": service_status_url,
            }
        )


def handle_survey_api_exception(
    exception: Exception,
    survey_name: str,
    operation: str,
    context: dict[str, Any] | None = None,
) -> SurveyIntegrationError:
    """Convert generic exceptions to survey-specific exceptions.

    Args:
        exception: Original exception
        survey_name: Name of the survey
        operation: Operation that was being performed
        context: Additional context information

    Returns:
        Survey-specific exception
    """
    context = context or {}

    if "timeout" in str(exception).lower():
        return SurveyTimeoutError(
            message=f"Timeout during {operation}: {str(exception)}",
            survey_name=survey_name,
            timeout_duration=context.get("timeout", 30.0),
            operation=operation,
        )

    if "rate limit" in str(exception).lower() or "429" in str(exception):
        return SurveyRateLimitError(
            message=f"Rate limit exceeded during {operation}: {str(exception)}",
            survey_name=survey_name,
            retry_after=context.get("retry_after"),
            request_count=context.get("request_count"),
        )

    if "authentication" in str(exception).lower() or "401" in str(exception):
        return SurveyAuthenticationError(
            message=f"Authentication failed during {operation}: {str(exception)}",
            survey_name=survey_name,
            auth_method=context.get("auth_method"),
        )

    if "not found" in str(exception).lower() or "404" in str(exception):
        return SurveyDataNotFoundError(
            message=f"Data not found during {operation}: {str(exception)}",
            survey_name=survey_name,
            observation_id=context.get("observation_id"),
            coordinates=context.get("coordinates"),
            search_radius=context.get("search_radius"),
        )

    if "503" in str(exception) or "service unavailable" in str(exception).lower():
        return SurveyServiceUnavailableError(
            message=f"Service unavailable during {operation}: {str(exception)}",
            survey_name=survey_name,
            estimated_downtime=context.get("estimated_downtime"),
            service_status_url=context.get("service_status_url"),
        )

    # Generic survey API error
    return SurveyAPIError(
        message=f"API error during {operation}: {str(exception)}",
        survey_name=survey_name,
        api_endpoint=context.get("api_endpoint"),
        status_code=context.get("status_code"),
        response_body=context.get("response_body"),
    )
