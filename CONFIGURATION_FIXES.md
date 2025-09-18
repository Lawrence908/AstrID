# AstrID Configuration Fixes

## Issues Addressed

### 1. Database Connection Pool Exhaustion
**Problem**: Supabase was rejecting connections due to hitting max clients limit in Session mode.

**Solution**: 
- Reduced connection pool sizes from `pool_size=2, max_overflow=1` to `pool_size=1, max_overflow=0`
- This limits each service to exactly 1 connection, with 3 services = 3 total connections
- Added better error handling and logging for connection pool issues

**Files Modified**:
- `src/core/constants.py`: Updated `POOL_CONFIGS` for both development and production

### 2. MLflow Worker Timeouts
**Problem**: MLflow workers were timing out and being killed due to resource constraints.

**Solution**:
- Reduced Gunicorn timeout from 120s to 60s
- Limited workers to 1 (from default 4)
- Added worker recycling with `--max-requests 100 --max-requests-jitter 10`
- Used sync worker class for better stability

**Files Modified**:
- `docker-compose.yaml`: Updated MLflow command with optimized Gunicorn settings

### 3. Dramatiq Worker Resource Exhaustion
**Problem**: Dramatiq workers were using too many resources with 4 processes.

**Solution**:
- Reduced Dramatiq processes from 4 to 1 (`-p 1`)
- Kept single thread per process (`-t 1`)

**Files Modified**:
- `docker-compose.yaml`: Updated worker command

### 4. Enhanced Monitoring
**Problem**: Limited visibility into connection pool status and errors.

**Solution**:
- Added connection pool health check function
- Enhanced error logging for Supabase connection issues
- Added debug logging for database session creation

**Files Modified**:
- `src/core/db/session.py`: Added `get_pool_health()` function and enhanced error handling

## Configuration Summary

### Database Connection Pools
- **Development**: `pool_size=1, max_overflow=0` (1 connection per service)
- **Production**: `pool_size=1, max_overflow=0` (1 connection per service)
- **Total Connections**: 3 (API + MLflow + Prefect)

### MLflow Configuration
- **Workers**: 1 (reduced from 4)
- **Timeout**: 60s (reduced from 120s)
- **Worker Class**: sync
- **Max Requests**: 100 with jitter

### Dramatiq Workers
- **Processes**: 1 (reduced from 4)
- **Threads per Process**: 1

## Expected Benefits

1. **Stability**: Reduced resource contention and connection pool exhaustion
2. **Reliability**: Better error handling and monitoring
3. **Performance**: More predictable resource usage
4. **Debugging**: Enhanced logging for troubleshooting

## Next Steps

1. Restart the services to apply the new configuration
2. Monitor logs for connection pool issues
3. Test worker functionality with the reduced process count
4. Consider scaling up gradually if needed

## Commands to Apply Changes

```bash
# Stop current services
docker-compose down

# Rebuild and start with new configuration
docker-compose up --build -d

# Monitor logs
docker-compose logs -f
```
