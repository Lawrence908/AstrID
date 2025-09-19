# AstrID Test Framework

A comprehensive testing infrastructure for the AstrID project with async support, mock services, and coverage reporting.

## Features

- **Async Testing**: Full async/await support with pytest-asyncio
- **Database Testing**: Transaction-based test isolation with rollback
- **Mock Services**: Complete mock implementations for external dependencies
- **Coverage Reporting**: Comprehensive coverage analysis with configurable thresholds
- **Performance Testing**: Built-in performance measurement utilities
- **Parallel Execution**: Multi-core test execution support

## Quick Start

### Install Dependencies

```bash
uv pip install --dev .
```

### Run All Tests

```bash
./scripts/run-tests.sh
```

### Run Specific Test Types

```bash
# Unit tests only
./scripts/run-tests.sh -t unit

# Integration tests with parallel execution
./scripts/run-tests.sh -t integration --parallel

# Performance tests with verbose output
./scripts/run-tests.sh -t performance -v

# Exclude slow tests
./scripts/run-tests.sh -m "not slow"
```

## Test Structure

```
tests/
├── unit/                          # Unit tests
│   ├── domains/                   # Domain-specific tests
│   │   ├── observations/
│   │   ├── detections/
│   │   ├── preprocessing/
│   │   ├── differencing/
│   │   └── ml/
│   ├── infrastructure/            # Infrastructure tests
│   │   ├── storage/
│   │   ├── mlflow/
│   │   └── workflow/
│   └── api/                       # API tests
├── integration/                   # Integration tests
│   ├── api/
│   ├── database/
│   └── external_services/
├── e2e/                          # End-to-end tests
│   ├── workflows/
│   └── user_journeys/
├── fixtures/                     # Test fixtures
│   ├── database.py               # Database fixtures
│   ├── data.py                   # Sample data fixtures
│   └── services.py               # Service fixtures
├── mocks/                        # Mock implementations
│   ├── storage.py               # Mock storage client
│   ├── mlflow.py                # Mock MLflow client
│   ├── prefect.py               # Mock Prefect client
│   ├── dramatiq.py              # Mock Dramatiq broker
│   ├── supabase.py              # Mock Supabase client
│   └── external_apis.py         # Mock external APIs
├── conftest.py                   # Global test configuration
├── utils.py                      # Test utilities
└── test_framework_validation.py  # Framework validation tests
```

## Test Configuration

Tests are configured via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "--strict-config", 
    "--verbose",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=90",
]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "slow: Slow running tests",
    "requires_gpu: Tests requiring GPU",
    "requires_external: Tests requiring external services",
]
```

## Available Fixtures

### Database Fixtures

- `db_session`: Database session with automatic rollback
- `db_transaction`: Explicit transaction management
- `test_db_url`: Test database URL
- `db_cleanup`: Database cleanup utilities

### Sample Data Fixtures

- `sample_observation_data`: Sample observation data
- `sample_survey_data`: Sample survey data  
- `sample_detection_data`: Sample detection data
- `sample_model_data`: Sample ML model data

### Mock Service Fixtures

- `mock_mlflow_client`: Mock MLflow tracking client
- `mock_prefect_client`: Mock Prefect workflow client
- `mock_dramatiq_broker`: Mock Dramatiq message broker
- `mock_supabase_client`: Mock Supabase auth client
- `mock_storage_service`: Mock cloud storage service

### Utility Fixtures

- `performance_timer`: Performance timing utility
- `temp_dir`: Temporary directory for test files
- `test_config`: Test configuration settings

## Writing Tests

### Basic Unit Test

```python
import pytest
from src.domains.observations.models import Observation

class TestObservation:
    def test_create_observation(self, sample_observation_data):
        """Test observation creation."""
        obs = Observation(**sample_observation_data)
        assert obs.id == sample_observation_data["id"]
        assert obs.ra == sample_observation_data["ra"]
```

### Async Database Test

```python
import pytest
from src.domains.observations.repository import ObservationRepository

class TestObservationRepository:
    @pytest.mark.asyncio
    async def test_create_observation(self, db_session, sample_observation_data):
        """Test creating an observation in the database."""
        repo = ObservationRepository(db_session)
        
        obs = await repo.create(sample_observation_data)
        assert obs.id is not None
        
        # Verify it was saved
        retrieved = await repo.get_by_id(obs.id)
        assert retrieved.id == obs.id
```

### Mock Service Test

```python
import pytest

class TestMLService:
    @pytest.mark.asyncio
    async def test_model_prediction(self, mock_mlflow_client):
        """Test ML model prediction with mocked MLflow."""
        from src.domains.ml.service import MLService
        
        service = MLService(mlflow_client=mock_mlflow_client)
        
        # Mock return value
        mock_mlflow_client.predict.return_value = [0.95, 0.12, 0.88]
        
        result = await service.predict_anomalies(image_data)
        assert len(result) == 3
        assert result[0] > 0.9
```

### Performance Test

```python
import pytest
from tests.utils import PerformanceTestUtils

class TestPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_detection_throughput(self):
        """Test detection processing throughput."""
        async def process_detection():
            # Your detection logic here
            pass
        
        metrics = await PerformanceTestUtils.measure_throughput(
            process_detection,
            duration=10.0
        )
        
        assert metrics["throughput"] > 100  # ops/sec
        assert metrics["avg_time"] < 0.01   # seconds
```

## Mock Services

### MockStorageClient

```python
# Test file upload/download
client = MockStorageClient()
url = await client.upload_file("local.txt", "remote.txt")
content = await client.download_file("remote.txt")

# Simulate errors
client.simulate_upload_error(True)
with pytest.raises(Exception):
    await client.upload_file("test.txt", "remote.txt")
```

### MockMLflowClient

```python
# Test experiment tracking
client = MockMLflowClient()
exp_id = client.create_experiment("test_exp")
run = client.create_run(exp_id)
client.log_metric("accuracy", 0.95)
client.log_param("lr", 0.01)
```

### MockPrefectClient

```python
# Test workflow orchestration
client = MockPrefectClient()
flow = await client.create_flow("test_flow")
run = await client.create_flow_run(flow["id"])
await client.set_flow_run_state(run["id"], "RUNNING")
```

## Coverage Reporting

Coverage is automatically generated when running tests:

```bash
# Generate HTML coverage report
./scripts/run-tests.sh

# View coverage report
open htmlcov/index.html
```

Coverage threshold is set to 90% in `pyproject.toml`. Tests will fail if coverage drops below this threshold.

## Performance Testing

Performance tests are marked with `@pytest.mark.performance`:

```bash
# Run only performance tests
./scripts/run-tests.sh -t performance

# Or use markers directly
pytest -m performance
```

## Continuous Integration

The test framework integrates with CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    ./scripts/run-tests.sh --no-coverage
    
- name: Generate coverage
  run: |
    ./scripts/run-tests.sh -t unit
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Use Fixtures**: Leverage fixtures for common setup/teardown
3. **Mock External Services**: Always mock external dependencies
4. **Clear Test Names**: Use descriptive test names that explain what is being tested
5. **Test Edge Cases**: Include tests for error conditions and edge cases
6. **Performance Awareness**: Mark slow tests appropriately
7. **Documentation**: Document complex test scenarios

## Troubleshooting

### Common Issues

1. **Async Test Errors**: Ensure `@pytest.mark.asyncio` decorator is used
2. **Database Conflicts**: Use `db_session` fixture for automatic rollback
3. **Import Errors**: Check that test paths are correctly configured
4. **Coverage Issues**: Ensure all source code is in the `src/` directory

### Debug Mode

Run tests with extra debugging:

```bash
./scripts/run-tests.sh -v --tb=long
```

### Performance Issues

Run tests in parallel to speed up execution:

```bash
./scripts/run-tests.sh --parallel
```
