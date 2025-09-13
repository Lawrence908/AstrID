"""Database-specific exceptions."""

from typing import Any

from psycopg.errors import (
    CheckViolation,
    ForeignKeyViolation,
    InsufficientPrivilege,
    InvalidDatetimeFormat,
    InvalidTextRepresentation,
    NotNullViolation,
    UndefinedColumn,
    UndefinedTable,
    UniqueViolation,
)
from sqlalchemy.exc import (
    DataError,
    DisconnectionError,
    IntegrityError,
    OperationalError,
    ProgrammingError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLTimeoutError,
)

from src.core.exceptions import DatabaseError as CoreDatabaseError


class DBError(CoreDatabaseError):
    """Base exception for all database operations."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, details)
        self.original_error = original_error


class DatabaseConnectionError(DBError):
    """Raised when there's an issue connecting to the database."""

    pass


class DatabaseTimeoutError(DBError):
    """Raised when a database operation times out."""

    pass


class DatabasePermissionError(DBError):
    """Raised when the operation is not permitted."""

    pass


class ForeignKeyViolationError(DBError):
    """Raised when there's a foreign key constraint violation."""

    pass


class UniqueConstraintViolationError(DBError):
    """Raised when there's a unique constraint violation."""

    pass


class NotNullViolationError(DBError):
    """Raised when a NOT NULL constraint is violated."""

    pass


class CheckConstraintViolationError(DBError):
    """Raised when a CHECK constraint is violated."""

    pass


class InvalidDataFormatError(DBError):
    """Raised when data format is invalid for the database."""

    pass


class TableNotFoundError(DBError):
    """Raised when a referenced table doesn't exist."""

    pass


class ColumnNotFoundError(DBError):
    """Raised when a referenced column doesn't exist."""

    pass


class DatabaseTransactionError(DBError):
    """Raised when there's an issue with database transactions."""

    pass


class DatabaseQueryError(DBError):
    """Raised when there's an issue with database queries."""

    pass


def create_db_error(
    message: str, original_error: Exception, details: dict[str, Any] | None = None
) -> DBError:
    """
    Create a DBError with proper context and chaining.

    Args:
        message: Human-readable error message
        original_error: The original SQLAlchemy/database exception
        details: Optional additional details

    Returns:
        DBError: Properly formatted database error
    """
    return DBError(
        message=message, details=details or {}, original_error=original_error
    )


def handle_database_exception(exc: Exception) -> DBError:
    """
    Convert SQLAlchemy and database-specific exceptions to AstrID database exceptions.

    Args:
        exc: The original database exception

    Returns:
        DBError: Appropriate AstrID database exception
    """
    # SQLAlchemy exceptions
    if isinstance(exc, IntegrityError):
        # Check for specific constraint violations
        if hasattr(exc.orig, "pgcode"):
            pgcode = exc.orig.pgcode

            if pgcode == UniqueViolation.sqlstate:
                return UniqueConstraintViolationError(
                    message="Unique constraint violation",
                    details={"constraint": str(exc.orig)},
                    original_error=exc,
                )
            elif pgcode == ForeignKeyViolation.sqlstate:
                return ForeignKeyViolationError(
                    message="Foreign key constraint violation",
                    details={"constraint": str(exc.orig)},
                    original_error=exc,
                )
            elif pgcode == NotNullViolation.sqlstate:
                return NotNullViolationError(
                    message="NOT NULL constraint violation",
                    details={"constraint": str(exc.orig)},
                    original_error=exc,
                )
            elif pgcode == CheckViolation.sqlstate:
                return CheckConstraintViolationError(
                    message="CHECK constraint violation",
                    details={"constraint": str(exc.orig)},
                    original_error=exc,
                )

        # Generic integrity error
        return DBError(
            message="Database integrity constraint violation",
            details={"error": str(exc)},
            original_error=exc,
        )

    elif isinstance(exc, OperationalError):
        if isinstance(exc, DisconnectionError):
            return DatabaseConnectionError(
                message="Database connection lost",
                details={"error": str(exc)},
                original_error=exc,
            )
        elif isinstance(exc, SQLTimeoutError):
            return DatabaseTimeoutError(
                message="Database operation timed out",
                details={"error": str(exc)},
                original_error=exc,
            )
        else:
            return DatabaseConnectionError(
                message="Database operational error",
                details={"error": str(exc)},
                original_error=exc,
            )

    elif isinstance(exc, ProgrammingError):
        if hasattr(exc.orig, "pgcode"):
            pgcode = exc.orig.pgcode

            if pgcode == UndefinedTable.sqlstate:
                return TableNotFoundError(
                    message="Referenced table does not exist",
                    details={"table": str(exc.orig)},
                    original_error=exc,
                )
            elif pgcode == UndefinedColumn.sqlstate:
                return ColumnNotFoundError(
                    message="Referenced column does not exist",
                    details={"column": str(exc.orig)},
                    original_error=exc,
                )
            elif pgcode == InsufficientPrivilege.sqlstate:
                return DatabasePermissionError(
                    message="Insufficient database privileges",
                    details={"error": str(exc.orig)},
                    original_error=exc,
                )

        return DatabaseQueryError(
            message="Database programming error",
            details={"error": str(exc)},
            original_error=exc,
        )

    elif isinstance(exc, DataError):
        if hasattr(exc.orig, "pgcode"):
            pgcode = exc.orig.pgcode

            if pgcode in [
                InvalidTextRepresentation.sqlstate,
                InvalidDatetimeFormat.sqlstate,
            ]:
                return InvalidDataFormatError(
                    message="Invalid data format for database",
                    details={"error": str(exc.orig)},
                    original_error=exc,
                )

        return InvalidDataFormatError(
            message="Invalid data format",
            details={"error": str(exc)},
            original_error=exc,
        )

    # Fallback for any other database-related exception
    return DBError(
        message="Database error occurred",
        details={"error": str(exc), "type": type(exc).__name__},
        original_error=exc,
    )
