"""Custom exceptions for AstrID application."""

from typing import Any


class AstrIDException(Exception):
    """Base exception for all AstrID-specific errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(message)

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# Authentication & Authorization
class AuthenticationError(AstrIDException):
    """Raised when authentication fails."""

    pass


class AuthorizationError(AstrIDException):
    """Raised when authorization fails."""

    pass


class EmailNotConfirmedError(AuthenticationError):
    """Raised when a user tries to log in but their email is not confirmed."""

    pass


class PasswordUpdateError(AstrIDException):
    """Raised when there's an issue updating a password."""

    pass


class PermissionDeniedError(AstrIDException):
    """Raised when user doesn't have permission for operation."""

    pass


# Resource Management
class ResourceNotFoundError(AstrIDException):
    """Raised when a resource cannot be found."""

    pass


class ResourceConflictError(AstrIDException):
    """Raised when there's a conflict with existing resource."""

    pass


class ResourceValidationError(AstrIDException):
    """Raised when resource validation fails."""

    pass


# Data & Processing
class DatabaseError(AstrIDException):
    """Raised when there's an issue with the database."""

    pass


class ValidationError(AstrIDException):
    """Raised when validation fails."""

    pass


class ProcessingError(AstrIDException):
    """Raised when there's an issue with processing."""

    pass


class DataIntegrityError(AstrIDException):
    """Raised when data integrity constraints are violated."""

    pass


# External Services & Integrations
class IntegrationError(AstrIDException):
    """Raised when there's an issue with an external integration."""

    pass


class ExternalServiceError(AstrIDException):
    """Raised when an external service fails."""

    pass


class ConfigurationError(AstrIDException):
    """Raised when there's an issue with configuration."""

    pass


# Domain-Specific Base Classes
class ObservationError(AstrIDException):
    """Base exception for observation-related errors."""

    pass


class DetectionError(AstrIDException):
    """Base exception for detection-related errors."""

    pass


class DifferencingError(AstrIDException):
    """Base exception for differencing-related errors."""

    pass


class PreprocessingError(AstrIDException):
    """Base exception for preprocessing-related errors."""

    pass


class CurationError(AstrIDException):
    """Base exception for curation-related errors."""

    pass


class CatalogError(AstrIDException):
    """Base exception for catalog-related errors."""

    pass


# Observations Domain Specific Exceptions
class InvalidCoordinatesError(ObservationError):
    """Raised when observation coordinates are invalid."""

    pass


class SurveyNotFoundError(ObservationError):
    """Raised when referenced survey doesn't exist."""

    pass


class FITSFileError(ObservationError):
    """Raised when there's an issue with FITS file processing."""

    pass


class ObservationAlreadyExistsError(ObservationError):
    """Raised when trying to create a duplicate observation."""

    pass


class InvalidExposureTimeError(ObservationError):
    """Raised when exposure time is invalid."""

    pass


class InvalidFilterBandError(ObservationError):
    """Raised when filter band is invalid."""

    pass


# Detection Domain Specific Exceptions
class ModelNotFoundError(DetectionError):
    """Raised when ML model is not found."""

    pass


class ModelNotActiveError(DetectionError):
    """Raised when trying to use an inactive model."""

    pass


class ModelTrainingError(DetectionError):
    """Raised when model training fails."""

    pass


class ModelInferenceError(DetectionError):
    """Raised when model inference fails."""

    pass


class InvalidConfidenceScoreError(DetectionError):
    """Raised when confidence score is invalid."""

    pass


class DetectionNotFoundError(DetectionError):
    """Raised when detection is not found."""

    pass


class InvalidDetectionTypeError(DetectionError):
    """Raised when detection type is invalid."""

    pass


# Differencing Domain Specific Exceptions
class DifferenceRunNotFoundError(DifferencingError):
    """Raised when difference run is not found."""

    pass


class InvalidAlgorithmError(DifferencingError):
    """Raised when differencing algorithm is invalid."""

    pass


class ReferenceObservationNotFoundError(DifferencingError):
    """Raised when reference observation is not found."""

    pass


class CandidateNotFoundError(DifferencingError):
    """Raised when candidate is not found."""

    pass


class InvalidSignificanceError(DifferencingError):
    """Raised when significance value is invalid."""

    pass


class DifferencingFailedError(DifferencingError):
    """Raised when differencing algorithm fails."""

    pass


# Preprocessing Domain Specific Exceptions
class PreprocessRunNotFoundError(PreprocessingError):
    """Raised when preprocessing run is not found."""

    pass


class CalibrationFailedError(PreprocessingError):
    """Raised when calibration fails."""

    pass


class WCSAlignmentError(PreprocessingError):
    """Raised when WCS alignment fails."""

    pass


class RegistrationFailedError(PreprocessingError):
    """Raised when image registration fails."""

    pass


class InvalidImageFormatError(PreprocessingError):
    """Raised when image format is invalid."""

    pass


# Curation Domain Specific Exceptions
class ValidationEventNotFoundError(CurationError):
    """Raised when validation event is not found."""

    pass


class AlertNotFoundError(CurationError):
    """Raised when alert is not found."""

    pass


class InvalidConfidenceLevelError(CurationError):
    """Raised when confidence level is invalid."""

    pass


class InvalidQualityLevelError(CurationError):
    """Raised when quality level is invalid."""

    pass


class AlertDeliveryFailedError(CurationError):
    """Raised when alert delivery fails."""

    pass


class ValidationAlreadyExistsError(CurationError):
    """Raised when validation already exists for detection."""

    pass


# Catalog Domain Specific Exceptions
class SystemConfigNotFoundError(CatalogError):
    """Raised when system configuration is not found."""

    pass


class ProcessingJobNotFoundError(CatalogError):
    """Raised when processing job is not found."""

    pass


class AuditLogNotFoundError(CatalogError):
    """Raised when audit log entry is not found."""

    pass


class InvalidJobStatusError(CatalogError):
    """Raised when job status transition is invalid."""

    pass


class JobRetryLimitExceededError(CatalogError):
    """Raised when job retry limit is exceeded."""

    pass


class InvalidConfigurationError(CatalogError):
    """Raised when system configuration is invalid."""

    pass


# System & Infrastructure
class MonitoringError(AstrIDException):
    """Raised when monitoring operations fail."""

    pass


class SchedulerError(AstrIDException):
    """Raised when scheduler operations fail."""

    pass


class FileSystemError(AstrIDException):
    """Raised when there's an issue with file system operations."""

    pass


class NetworkError(AstrIDException):
    """Raised when there's a network-related error."""

    pass


# Business Logic
class BusinessLogicError(AstrIDException):
    """Raised when business logic constraints are violated."""

    pass


class WorkflowError(AstrIDException):
    """Raised when workflow operations fail."""

    pass


class StateTransitionError(AstrIDException):
    """Raised when an invalid state transition is attempted."""

    pass
