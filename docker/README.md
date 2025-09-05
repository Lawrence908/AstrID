# AstrID Docker Development Setup

This directory contains the Docker configuration for AstrID development environment, optimized with best practices from production deployments.

## Quick Start

```bash
# Start the development environment
./docker/start-dev.sh

# Start with rebuild
./docker/start-dev.sh --build

# Stop the environment
./docker/start-dev.sh --down
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI application with hot reload |
| MLflow | 5000 | ML experiment tracking |
| Prefect | 4200 | Workflow orchestration |
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Message queue and caching |

## Key Improvements Applied

### ✅ **From Boss's Best Practices:**

1. **Container Names**: All services have explicit `container_name` for easier debugging
2. **Environment Files**: Uses `env_file` to load from `.env` files (cleaner than inline env vars)
3. **Volume Mounting**: Strategic volume mounts for development (source code, logs)
4. **Network Isolation**: Custom network with explicit naming (`astrid-dev`)
5. **Health Checks**: Proper health checks with start periods
6. **Redis Security**: Password-protected Redis with proper entrypoint
7. **Development Commands**: Hot reload commands for development
8. **Project Naming**: Uses `-p astrid-dev` for project isolation

### ✅ **Environment Variable Handling:**

- **YES, you need environment variables in compose** even with `constants.py`
- Your `constants.py` uses `load_dotenv()` which loads from `.env` files
- Docker containers don't automatically have access to your host's `.env` file
- The compose environment variables override/feed into the container's environment

### ✅ **Database URL Configuration:**

- **Your app uses `get_database_url()` from `constants.py`** which builds URLs from Supabase components
- **NOT using `DATABASE_URL` environment variable** - instead using individual Supabase components
- **Driver distinction**: Main app uses `asyncpg` (async), MLflow uses `psycopg` (sync)
- **Compose provides**: `SUPABASE_HOST`, `SUPABASE_PASSWORD`, `SUPABASE_PROJECT_REF` for local development

### ✅ **Fixed Inconsistencies:**

- Fixed `APP_ENV` vs `ENVIRONMENT` mismatch in `env.example`
- Consistent database URLs between compose and constants
- Proper Redis password handling
- Added missing `PYTHONPATH` and `DEBUG` environment variables

## Development Features

### Hot Reload
- API service automatically reloads on code changes
- Source code is mounted as volumes for instant updates
- Logs are persisted to `./logs/` directory

### Health Monitoring
- All services have health checks with proper start periods
- Startup script waits for services to be healthy before completing
- Easy debugging with named containers

### Security
- Redis is password-protected
- Non-root users in containers
- Network isolation between services

## Environment Configuration

1. Copy `env.example` to `.env`:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` with your actual values:
   - Supabase credentials
   - Cloudflare R2 credentials
   - Redis password
   - Other service configurations

3. Start the environment:
   ```bash
   ./docker/start-dev.sh
   ```

## Troubleshooting

### View Logs
```bash
# All services
docker-compose -p astrid-dev -f docker/compose.yml logs -f

# Specific service
docker-compose -p astrid-dev -f docker/compose.yml logs -f api
```

### Rebuild Services
```bash
./docker/start-dev.sh --build
```

### Reset Everything
```bash
./docker/start-dev.sh --down
docker-compose -p astrid-dev -f docker/compose.yml down -v
```

## Production Considerations

This setup is optimized for development. For production:

1. Remove volume mounts for source code
2. Use production Dockerfile targets
3. Set proper secrets management
4. Configure proper logging
5. Set up monitoring and alerting
