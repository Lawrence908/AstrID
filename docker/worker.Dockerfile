# ----- Builder Stage -----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed for package installation
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache-dir --system .

# ----- Runtime Stage -----
FROM python:3.11-slim-bookworm

# Create non-root user
RUN adduser --disabled-password --gecos '' astrid

# Set up directory structure and environment
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    LOG_DIR=/app/logs

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && \
    chown astrid:astrid /app/logs

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and application code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Set ownership
RUN chown -R astrid:astrid /app

# Switch to non-root user
USER astrid

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import redis; redis.Redis.from_url('$REDIS_URL').ping()" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["dramatiq", "-w", "4", "-p", "1", "src.adapters.workers.tasks"]
