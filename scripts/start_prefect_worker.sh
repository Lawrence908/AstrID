#!/bin/bash

# Start Prefect worker with retry mechanism
# This script waits for the Prefect server to be available before starting the worker

set -e

# Configuration
PREFECT_API_URL=${PREFECT_API_URL:-"http://prefect:4200/api"}
PREFECT_WORK_POOL_NAME=${PREFECT_WORK_POOL_NAME:-"astrid-pool"}
PREFECT_WAIT_MAX_RETRIES=${PREFECT_WAIT_MAX_RETRIES:-30}
PREFECT_WAIT_DELAY=${PREFECT_WAIT_DELAY:-2.0}

echo "Starting Prefect worker..."
echo "Prefect API URL: $PREFECT_API_URL"
echo "Work Pool: $PREFECT_WORK_POOL_NAME"

# Function to check if Prefect server is available
check_prefect_server() {
    echo "Checking Prefect server availability..."
    python -c "
import requests
import sys
try:
    response = requests.get('$PREFECT_API_URL/health', timeout=5)
    if response.status_code == 200:
        print('Prefect server is available')
        sys.exit(0)
    else:
        print(f'Prefect server returned status {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'Prefect server not available: {e}')
    sys.exit(1)
"
}

# Wait for Prefect server to be available
echo "Waiting for Prefect server to be available..."
for i in $(seq 1 $PREFECT_WAIT_MAX_RETRIES); do
    if check_prefect_server; then
        echo "Prefect server is ready!"
        break
    else
        echo "Attempt $i/$PREFECT_WAIT_MAX_RETRIES: Prefect server not ready, waiting ${PREFECT_WAIT_DELAY}s..."
        sleep $PREFECT_WAIT_DELAY
    fi

    if [ $i -eq $PREFECT_WAIT_MAX_RETRIES ]; then
        echo "ERROR: Prefect server not available after $PREFECT_WAIT_MAX_RETRIES attempts"
        exit 1
    fi
done

# Set Prefect API URL
export PREFECT_API_URL

# Start Prefect worker
echo "Starting Prefect worker with work pool: $PREFECT_WORK_POOL_NAME"
exec prefect worker start --pool "$PREFECT_WORK_POOL_NAME" --limit 4
