Test framework
==============

Run tests
---------

```bash
uv pip install --dev .
uv run pytest -m "not slow" --cov=src --cov-report=term-missing --cov-report=html
```

Structure
---------

- `tests/conftest.py`: shared asyncio loop, async DB session, and adapter mocks
- `tests/unit/...`: unit tests
- `tests/integration/...`: integration tests


