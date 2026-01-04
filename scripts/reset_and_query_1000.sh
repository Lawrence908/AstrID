#!/bin/bash
# Reset checkpoint and query 1000 supernovae fresh

echo "============================================================"
echo "Resetting checkpoint and querying 1000 supernovae"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Clear the checkpoint file"
echo "  2. Remove empty chunk files"
echo "  3. Start fresh query for 1000 supernovae"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run with reset
./scripts/query_chunked_1000.sh
