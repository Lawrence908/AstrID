#!/bin/bash

# AstrID Development Startup Script
# Usage: ./start-dev.sh [--build] [--down]

set -e

PROJECT_NAME="astrid-dev"
COMPOSE_FILE="docker-compose.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting AstrID Development Environment${NC}"

# Handle command line arguments
BUILD_FLAG=""
DOWN_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --down)
            DOWN_FLAG="--down"
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--build] [--down]"
            exit 1
            ;;
    esac
done

# Stop and remove containers if --down flag is provided
if [ "$DOWN_FLAG" = "--down" ]; then
    echo -e "${YELLOW}🛑 Stopping AstrID development environment...${NC}"
    docker-compose -p $PROJECT_NAME -f $COMPOSE_FILE down
    echo -e "${GREEN}✅ AstrID development environment stopped${NC}"
    exit 0
fi

# Start the services
echo -e "${BLUE}🐳 Starting Docker services...${NC}"

if [ "$BUILD_FLAG" = "--build" ]; then
    echo -e "${YELLOW}🔨 Building images...${NC}"
    docker-compose -p $PROJECT_NAME -f $COMPOSE_FILE up $BUILD_FLAG -d
else
    docker-compose -p $PROJECT_NAME -f $COMPOSE_FILE up -d
fi

# Wait for services to be healthy
echo -e "${BLUE}⏳ Waiting for services to be healthy...${NC}"

# Function to check if a service is healthy
check_service_health() {
    local service_name=$1
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose -p $PROJECT_NAME -f $COMPOSE_FILE ps $service_name | grep -q "healthy"; then
            echo -e "${GREEN}✅ $service_name is healthy${NC}"
            return 0
        fi

        echo -e "${YELLOW}⏳ Waiting for $service_name... (attempt $attempt/$max_attempts)${NC}"
        sleep 2
        ((attempt++))
    done

    echo -e "${RED}❌ $service_name failed to become healthy${NC}"
    return 1
}

# Check critical services
check_service_health "redis"
check_service_health "mlflow"

echo -e "${GREEN}🎉 AstrID development environment is ready!${NC}"
echo ""
echo -e "${BLUE}📋 Service URLs:${NC}"
echo -e "   API:          http://localhost:8000"
echo -e "   MLflow:       http://localhost:5000"
echo -e "   Prefect:      http://localhost:4200"
echo -e "   PostgreSQL:   Supabase (remote)"
echo -e "   Redis:        localhost:6379"
echo ""
echo -e "${BLUE}🔧 Useful commands:${NC}"
echo -e "   View logs:    docker-compose -p $PROJECT_NAME -f $COMPOSE_FILE logs -f [service]"
echo -e "   Stop all:     ./start-dev.sh --down"
echo -e "   Rebuild:      ./start-dev.sh --build"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
