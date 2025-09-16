#!/bin/bash
set -e

echo "Starting Prefect worker with connection retry..."

# Wait for Prefect server to be ready
python /app/scripts/wait_for_prefect.py

if [ $? -eq 0 ]; then
    echo "Prefect server is ready, starting worker..."
    # Start the Prefect worker with the specified pool
    exec python -m prefect worker start --pool "${PREFECT_WORK_POOL_NAME:-astrid-pool}"
else
    echo "Failed to connect to Prefect server, exiting..."
    exit 1
fi
