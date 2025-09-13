#!/bin/bash

# Script to set up Prefect for AstrID workflow orchestration

set -e

echo "🚀 Setting up Prefect for AstrID workflow orchestration..."

# Check if Prefect is available
if ! command -v prefect &> /dev/null; then
    echo "❌ Prefect CLI not found. Please install prefect with: pip install prefect>=2.14.0"
    exit 1
fi

# Set Prefect API URL (should point to the server in docker-compose)
export PREFECT_API_URL="http://localhost:4200/api"
echo "📡 Setting Prefect API URL to: $PREFECT_API_URL"

# Wait for Prefect server to be ready
echo "⏳ Waiting for Prefect server to be ready..."
timeout=60
elapsed=0
while ! curl -f "$PREFECT_API_URL/health" &> /dev/null; do
    if [ $elapsed -ge $timeout ]; then
        echo "❌ Timeout waiting for Prefect server. Please ensure it's running with docker-compose."
        echo "   Run: docker-compose up prefect"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "   Still waiting... (${elapsed}s/${timeout}s)"
done

echo "✅ Prefect server is ready!"

# Create work pool for AstrID flows
echo "🏊 Creating work pool 'astrid-pool'..."
prefect work-pool create --type process astrid-pool 2>/dev/null || echo "   Work pool may already exist"

# Create work queue
echo "📋 Creating work queue 'astrid-queue'..."
prefect work-queue create --pool astrid-pool astrid-queue 2>/dev/null || echo "   Work queue may already exist"

# Deploy flows
echo "🚀 Deploying AstrID flows..."
cd "$(dirname "$0")/.."  # Go to project root

# Deploy flows using the deployment script
python -m src.adapters.scheduler.deploy &
DEPLOY_PID=$!

# Give it time to deploy and then check
sleep 10

# Check if deployment is still running
if kill -0 $DEPLOY_PID 2>/dev/null; then
    echo "⏳ Flow deployment is running in background (PID: $DEPLOY_PID)"
    echo "   You can check the status in the Prefect UI at http://localhost:4200"
else
    echo "✅ Flow deployment completed"
fi

# Show deployed flows
echo ""
echo "📊 Deployed flows:"
prefect deployment ls 2>/dev/null || echo "   (Use 'prefect deployment ls' to see deployed flows)"

echo ""
echo "🎉 Prefect setup completed!"
echo ""
echo "📝 Next steps:"
echo "   1. Check the Prefect UI at http://localhost:4200"
echo "   2. Start a Prefect worker with: docker-compose up prefect-worker"
echo "   3. Monitor flows and deployments in the UI"
echo "   4. Trigger manual runs or wait for scheduled executions"
echo ""
echo "🔧 Useful commands:"
echo "   - List deployments: prefect deployment ls"
echo "   - Trigger a flow: prefect deployment run 'observation-processing'"
echo "   - View flow runs: prefect flow-run ls"
echo "   - Check worker status: prefect work-queue ls"
