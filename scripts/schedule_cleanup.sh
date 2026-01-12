#!/bin/bash
# Schedule cleanup to run after specified hours
# Usage: ./schedule_cleanup.sh [hours] [log_file]

HOURS=${1:-5}
LOG_FILE=${2:-"output/cleanup_$(date +%Y%m%d_%H%M%S).log"}

echo "Scheduling cleanup in $HOURS hours..."
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "Will run at: $(date -d "+$HOURS hours")"
echo ""

# Create log directory if needed
mkdir -p "$(dirname "$LOG_FILE")"

# Sleep for specified hours (convert to seconds)
SLEEP_SECONDS=$((HOURS * 3600))

echo "Sleeping for $HOURS hours ($SLEEP_SECONDS seconds)..."
sleep $SLEEP_SECONDS

echo ""
echo "=========================================="
echo "Starting cleanup at $(date)"
echo "=========================================="

# Activate venv and run cleanup
cd "$(dirname "$0")/.."
source .venv/bin/activate

python3 scripts/cleanup_auxiliary_files.py \
    --input-dir output/fits_downloads \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Cleanup completed at $(date)"
echo "=========================================="
