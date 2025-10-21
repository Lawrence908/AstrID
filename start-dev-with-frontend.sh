#!/bin/bash

# AstrID Development Environment with Frontend
# This script starts the full development environment including the frontend

echo "🚀 Starting AstrID Full Development Environment..."
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yaml" ]; then
    echo "❌ Error: docker-compose.yaml not found. Please run this script from the AstrID root directory."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your environment variables."
    echo "   You can copy from .env.example if it exists."
    echo ""
fi

# Parse command line arguments
BUILD=false
DOWN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --down)
            DOWN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--build] [--down]"
            echo "  --build    Rebuild all containers"
            echo "  --down     Stop and remove all containers"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [ "$DOWN" = true ]; then
    echo "🛑 Stopping AstrID development environment..."
    docker-compose -p astrid-dev -f docker-compose.yaml down
    echo "✅ Environment stopped"
    exit 0
fi

# Copy latest diagrams and docs to frontend
echo "📋 Updating frontend static files..."
mkdir -p frontend/public/docs/diagrams
cp docs/diagrams/*.svg frontend/public/docs/diagrams/ 2>/dev/null || echo "No SVG files to copy"
cp docs/*.md frontend/public/docs/ 2>/dev/null || echo "No markdown files to copy"
cp docs/consolidated-models.py frontend/public/docs/ 2>/dev/null || echo "No Python files to copy"
echo "✅ Frontend static files updated"
echo ""

# Start the development environment
if [ "$BUILD" = true ]; then
    echo "🔨 Building and starting AstrID development environment..."
    docker-compose -p astrid-dev -f docker-compose.yaml up --build -d
else
    echo "🚀 Starting AstrID development environment..."
    docker-compose -p astrid-dev -f docker-compose.yaml up -d
fi

echo ""
echo "⏳ Waiting for services to be healthy..."

# Wait for services to be healthy
timeout=300  # 5 minutes
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker-compose -p astrid-dev -f docker-compose.yaml ps | grep -q "unhealthy"; then
        echo "⚠️  Some services are still starting up..."
        sleep 10
        elapsed=$((elapsed + 10))
    else
        echo "✅ All services are healthy!"
        break
    fi
done

echo ""
echo "🎉 AstrID Development Environment is ready!"
echo ""
echo "📊 Services:"
echo "  🌐 Frontend (Planning Dashboard): http://localhost:3000"
echo "  🔧 API: http://127.0.0.1:8000"
echo "  📚 API Docs: http://127.0.0.1:8000/docs"
echo "  🔄 MLflow: http://localhost:5000"
echo "  ⚡ Prefect: http://localhost:4200"
echo ""
echo "📋 Useful Commands:"
echo "  View logs: docker-compose -p astrid-dev -f docker-compose.yaml logs -f"
echo "  Stop: ./start-dev-with-frontend.sh --down"
echo "  Rebuild: ./start-dev-with-frontend.sh --build"
echo ""
echo "🎯 Quick Access:"
echo "  Planning Dashboard: http://localhost:3000"
echo "  API Health: http://127.0.0.1:8000/health"
echo ""
