# AstrID Development Guide

## Development Environment Setup

### Prerequisites

- **Python 3.11+**: Latest stable Python version
- **Docker & Docker Compose**: Containerization tools
- **Git**: Version control system
- **VS Code** (recommended): IDE with Python extensions

### Initial Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/chrislawrence/AstrID.git
   cd AstrID
   ```

2. **Install uv Package Manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or restart terminal
   ```

3. **Install Dependencies**
   ```bash
   uv pip install --dev .
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Development Services

Start the development environment:

```bash
# Start all services
docker-compose -f docker/compose.yml up -d

# Check service status
docker-compose -f docker/compose.yml ps

# View logs
docker-compose -f docker/compose.yml logs -f api
```

## Project Structure

### Directory Organization

```
astrid/
├── src/                          # Source code
│   ├── domains/                  # Business logic
│   │   ├── observations/         # Observation management
│   │   ├── preprocessing/        # Image preprocessing
│   │   ├── differencing/         # Image differencing
│   │   ├── detection/            # Anomaly detection
│   │   ├── curation/             # Human validation
│   │   └── catalog/              # Data cataloging
│   ├── adapters/                 # Framework integrations
│   │   ├── api/                  # FastAPI application
│   │   ├── db/                   # Database layer
│   │   ├── storage/              # Cloud storage
│   │   ├── ml/                   # ML model interfaces
│   │   ├── imaging/              # Astronomical image processing
│   │   └── scheduler/            # Workflow orchestration
│   └── utils/                    # Shared utilities
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                     # End-to-end tests
├── notebooks/                    # Jupyter notebooks
├── docker/                       # Docker configuration
├── docs/                         # Documentation
└── .github/                      # GitHub configuration
```

### Import Rules

- **Domains**: May import only other domains and utils
- **Adapters**: May import domains, utils, and external libraries
- **Utils**: May import only standard library and third-party packages
- **Composition**: Wire interfaces to implementations in adapters

## Coding Standards

### Python Style Guide

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Comprehensive docstrings for all public functions
- **Line Length**: 88 characters (Black formatter default)

### Code Quality Tools

1. **Ruff**: Fast Python linter and formatter
   ```bash
   uv run ruff check src/          # Lint code
   uv run ruff format src/         # Format code
   ```

2. **Black**: Code formatter
   ```bash
   uv run black src/               # Format code
   uv run black --check src/       # Check formatting
   ```

3. **MyPy**: Static type checker
   ```bash
   uv run mypy src/                # Type checking
   ```

4. **Pre-commit**: Automated quality checks
   ```bash
   pre-commit run --all-files      # Run all hooks
   ```

### Naming Conventions

- **Classes**: PascalCase (e.g., `ObservationService`)
- **Functions/Methods**: snake_case (e.g., `process_observation`)
- **Variables**: snake_case (e.g., `observation_id`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- **Files**: snake_case (e.g., `observation_service.py`)

## Testing Strategy

### Test Categories

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test system performance under load

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m e2e

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_observations.py

# Run with verbose output
uv run pytest -v
```

### Test Structure

```python
# tests/unit/test_observations.py
import pytest
from src.domains.observations.models import Observation

class TestObservation:
    def test_observation_creation(self):
        """Test observation creation with valid data."""
        obs = Observation(
            id="test_001",
            survey="DSS",
            observation_id="dss_123",
            ra=180.0,
            dec=45.0,
            observation_time="2024-01-01T00:00:00Z",
            filter_band="R",
            exposure_time=30.0,
            fits_url="https://example.com/image.fits"
        )
        
        assert obs.id == "test_001"
        assert obs.survey == "DSS"
        assert obs.ra == 180.0

    def test_invalid_coordinates(self):
        """Test validation of coordinate ranges."""
        with pytest.raises(ValueError):
            Observation(
                id="test_002",
                survey="DSS",
                observation_id="dss_124",
                ra=400.0,  # Invalid RA > 360
                dec=45.0,
                observation_time="2024-01-01T00:00:00Z",
                filter_band="R",
                exposure_time=30.0,
                fits_url="https://example.com/image.fits"
            )
```

### Test Data Management

- **Fixtures**: Use pytest fixtures for test data
- **Factories**: Create test data factories for complex objects
- **Mocking**: Mock external dependencies and services
- **Database**: Use test database for integration tests

## Database Development

### Model Development

1. **Define Domain Models**: Pure business logic models
2. **Create Database Models**: SQLAlchemy ORM models
3. **Write Migrations**: Alembic migration scripts
4. **Update Tests**: Ensure all tests pass

### Migration Workflow

```bash
# Create new migration
alembic revision --autogenerate -m "Add observation table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Check migration status
alembic current
```

### Database Testing

```python
# tests/conftest.py
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from src.core.db.session import Base

@pytest.fixture
async def db_session():
    """Create test database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()
```

## API Development

### Endpoint Development

1. **Define Models**: Pydantic request/response models
2. **Implement Logic**: Business logic in domain services
3. **Create Routes**: FastAPI route handlers
4. **Add Validation**: Input validation and error handling
5. **Write Tests**: Comprehensive endpoint testing

### API Testing

```python
# tests/integration/test_api.py
from fastapi.testclient import TestClient
from src.adapters.api.main import app

client = TestClient(app)

def test_create_observation():
    """Test observation creation endpoint."""
    response = client.post(
        "/observations/",
        json={
            "survey": "DSS",
            "observation_id": "dss_123",
            "ra": 180.0,
            "dec": 45.0,
            "observation_time": "2024-01-01T00:00:00Z",
            "filter_band": "R",
            "exposure_time": 30.0,
            "fits_url": "https://example.com/image.fits"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["survey"] == "DSS"
    assert data["observation_id"] == "dss_123"
```

## Machine Learning Development

### Model Development

1. **Data Preparation**: Clean and preprocess training data
2. **Model Architecture**: Design and implement model
3. **Training Pipeline**: Implement training workflow
4. **Evaluation**: Comprehensive model evaluation
5. **Integration**: Integrate with production system

### MLflow Integration

```python
import mlflow
import mlflow.keras

def train_model():
    """Train U-Net model with MLflow tracking."""
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 100)
        
        # Train model
        model = train_unet_model()
        
        # Log metrics
        mlflow.log_metric("train_loss", 0.1)
        mlflow.log_metric("val_loss", 0.15)
        mlflow.log_metric("train_accuracy", 0.95)
        mlflow.log_metric("val_accuracy", 0.92)
        
        # Save model
        mlflow.keras.log_model(model, "unet_model")
        
        return model
```

## Debugging and Troubleshooting

### Common Issues

1. **Import Errors**: Check import paths and virtual environment
2. **Database Connection**: Verify database URL and credentials
3. **Docker Issues**: Check container logs and health status
4. **ML Model Issues**: Verify model file paths and dependencies

### Debug Tools

1. **VS Code Debugger**: Set breakpoints and inspect variables
2. **Python Debugger**: Use `pdb` or `ipdb` for debugging
3. **Logging**: Add structured logging for debugging
4. **Docker Logs**: Check container logs for errors

### Performance Profiling

```python
import cProfile
import pstats

def profile_function():
    """Profile function performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run function to profile
    result = expensive_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

## Deployment and CI/CD

### Local Testing

```bash
# Run all quality checks
pre-commit run --all-files

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Build Docker images
docker build -f docker/api.Dockerfile -t astrid-api .
docker build -f docker/worker.Dockerfile -t astrid-worker .
```

### CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs linting and type checking
2. Executes test suite
3. Performs security checks
4. Builds and pushes Docker images

### Production Deployment

1. **Environment Setup**: Configure production environment variables
2. **Database Migration**: Apply latest database migrations
3. **Service Deployment**: Deploy updated services
4. **Health Monitoring**: Verify service health and performance

## Contributing Guidelines

### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
2. **Make Changes**: Implement feature with tests
3. **Quality Checks**: Ensure all pre-commit hooks pass
4. **Test Coverage**: Maintain or improve test coverage
5. **Documentation**: Update relevant documentation
6. **Submit PR**: Create pull request with clear description

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed
- [ ] Error handling is appropriate

### Release Process

1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog**: Update `CHANGELOG.md`
3. **Tag Release**: Create git tag for version
4. **Deploy**: Trigger production deployment
5. **Announce**: Notify stakeholders of release

## Resources and References

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prefect Documentation](https://docs.prefect.io/)

### Tools and Libraries

- [uv Package Manager](https://docs.astral.sh/uv/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)
- [Pytest Testing Framework](https://docs.pytest.org/)

### Best Practices

- [Python Best Practices](https://docs.python-guide.org/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/best-practices/)
- [SQLAlchemy Best Practices](https://docs.sqlalchemy.org/en/20/orm/best_practices.html)
- [Testing Best Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
