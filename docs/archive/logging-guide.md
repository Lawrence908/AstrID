# AstrID Logging System Guide

## Overview

The AstrID logging system provides comprehensive, industry-standard logging capabilities across all domains. It includes structured logging, performance monitoring, security auditing, and system health tracking.

## Core Components

### 1. Basic Logging Setup (`src/core/logging.py`)

The core logging module provides:
- **Automatic initialization** with one-time setup
- **File and console logging** with rotation
- **Sentry integration** for error tracking
- **Domain-specific loggers** for organized logging

#### Usage

```python
from core.logging import get_logger, configure_domain_logger

# General logger
logger = get_logger(__name__)

# Domain-specific logger
domain_logger = configure_domain_logger("observations.survey")
```

### 2. Advanced Logging Utilities (`src/core/logging_utils.py`)

Industry-standard decorators and utilities for:
- **Function call logging** with arguments, results, and timing
- **Database operation logging** with entity tracking
- **API request logging** with request/response details
- **Business operation logging** with domain context
- **Performance monitoring** with threshold alerts
- **Data processing logging** with stage tracking
- **Security and audit logging** for compliance
- **System health monitoring** with metrics

## Domain Logging Implementation

All domains now include comprehensive logging:

### Catalog Domain
- **SystemConfigService**: Configuration management logging
- **ProcessingJobService**: Job lifecycle tracking
- **AuditLogService**: Audit trail logging

### Curation Domain
- **ValidationEventService**: Validation process logging
- **AlertService**: Alert creation and management

### Detection Domain
- **DetectionService**: ML inference and validation logging

### Observations Domain
- **SurveyService**: Survey management operations
- **ObservationService**: Observation lifecycle tracking

### Differencing Domain
- **DifferenceRunService**: Image differencing operations
- **CandidateService**: Candidate detection logging

### Preprocessing Domain
- **PreprocessRunService**: Image preprocessing operations

## Logging Patterns

### 1. Basic Service Logging

```python
class MyService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = configure_domain_logger("my_domain.service")
    
    async def create_entity(self, data: EntityCreate):
        self.logger.info(f"Creating entity: {data.name}")
        try:
            result = await self.repository.create(data)
            self.logger.info(f"Successfully created entity: id={result.id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to create entity: error={str(e)}")
            raise
```

### 2. Decorator-Based Logging

```python
from core.logging_utils import log_function_call, log_database_operation

@log_function_call(log_args=True, log_result=True, log_duration=True)
async def process_data(self, data: List[dict]):
    # Function automatically logged with args, result, and timing
    return processed_data

@log_database_operation("create", "observation")
async def create_observation(self, data: dict):
    # Database operation automatically logged
    return await self.repository.create(data)
```

### 3. Security and Audit Logging

```python
from core.logging_utils import log_audit_event, log_security_event

# Audit trail
log_audit_event("CREATE", "observation", str(obs_id), user_id)

# Security events
log_security_event("AUTHENTICATION_FAILURE", f"Invalid credentials for {username}")
```

### 4. Performance Monitoring

```python
from core.logging_utils import log_performance_metrics, log_metrics

@log_performance_metrics(threshold_seconds=1.0)
async def slow_operation(self, data):
    # Warns if operation takes longer than 1 second
    return result

# Custom metrics
log_metrics("observations_processed", count, "items", {"algorithm": "ml"})
```

## Log Levels and Usage

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about operations
- **WARNING**: Something unexpected happened but the system is still working
- **ERROR**: A serious problem occurred
- **CRITICAL**: A very serious error occurred

### When to Use Each Level

- **DEBUG**: Variable values, function entry/exit, detailed flow
- **INFO**: Successful operations, business events, user actions
- **WARNING**: Recoverable errors, deprecated usage, performance issues
- **ERROR**: Failed operations, exceptions, system errors
- **CRITICAL**: System failures, data corruption, security breaches

## Log Format

All logs follow a consistent format:
```
2025-09-04 22:27:22,149 - astrid.domains.observations.survey - INFO - Creating survey: name=ZTF, description=Zwicky Transient Facility
```

Components:
- **Timestamp**: ISO format with milliseconds
- **Logger Name**: Hierarchical logger name (domain.subdomain.component)
- **Level**: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Message**: Structured log message with context

## Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_DIR`: Directory for log files (default: "logs")
- `SENTRY_DSN`: Sentry DSN for error tracking (optional)

### Log File Rotation

- **Max file size**: 10MB
- **Backup count**: 5 files
- **Encoding**: UTF-8
- **Location**: `logs/app.log` (configurable via LOG_DIR)

## Best Practices

### 1. Use Structured Logging

```python
# Good
logger.info(f"Processing observation: id={obs_id}, survey={survey_name}, status={status}")

# Avoid
logger.info("Processing observation")
```

### 2. Include Context

```python
# Good
logger.error(f"Failed to create detection: observation_id={obs_id}, model_version={model}, error={str(e)}")

# Avoid
logger.error("Failed to create detection")
```

### 3. Use Appropriate Log Levels

```python
# DEBUG: Detailed flow information
logger.debug(f"Retrieved {len(results)} observations from database")

# INFO: Successful operations
logger.info(f"Successfully created survey: id={survey.id}, name={survey.name}")

# WARNING: Recoverable issues
logger.warning(f"Slow query detected: {query_time:.3f}s for {count} records")

# ERROR: Failed operations
logger.error(f"Database connection failed: {str(e)}")
```

### 4. Log Security Events

```python
# Always log authentication attempts
logger.info(f"Authentication attempt: user={username}")
logger.warning(f"Failed authentication: user={username}, reason=invalid_password")

# Log permission checks
log_audit_event("PERMISSION_CHECK", "observations", user_id)
```

### 5. Use Decorators for Common Patterns

```python
# Instead of manual logging
@log_function_call()
async def my_function(self, data):
    # Automatic logging of entry, exit, duration, and errors
    return result

# Instead of manual database logging
@log_database_operation("create", "observation")
async def create_observation(self, data):
    # Automatic logging of database operations
    return await self.repository.create(data)
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Error Rates**: Track ERROR and CRITICAL log counts
2. **Performance**: Monitor slow operations via performance decorators
3. **Security**: Alert on security events and failed authentications
4. **System Health**: Track component health status
5. **Business Metrics**: Monitor key business operations

### Log Analysis

Use tools like:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Grafana** with Loki
- **Splunk**
- **CloudWatch** (AWS)
- **Azure Monitor** (Azure)

### Sentry Integration

The system automatically sends errors to Sentry when configured:
- **Environment**: Automatically set based on APP_ENV
- **Release**: Uses APP_VERSION
- **Traces**: Performance monitoring enabled
- **Profiles**: Code profiling in development

## Examples

See `src/core/logging_examples.py` for comprehensive examples of:
- Basic service logging
- Decorator usage
- Security and audit logging
- System health monitoring
- Error handling patterns
- Performance monitoring

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check LOG_LEVEL environment variable
2. **File permissions**: Ensure LOG_DIR is writable
3. **Sentry not working**: Verify SENTRY_DSN is set correctly
4. **Performance issues**: Check if too much logging is enabled

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

This will show:
- All function calls with arguments
- Database queries
- Detailed error traces
- Performance metrics

## Migration Guide

If you have existing logging code:

1. **Replace print statements** with logger calls
2. **Add domain loggers** to service classes
3. **Use decorators** for common patterns
4. **Add structured context** to log messages
5. **Implement security logging** for sensitive operations

## Support

For questions or issues with the logging system:
1. Check this guide and examples
2. Review existing domain implementations
3. Consult the logging utilities documentation
4. Create an issue in the project repository
