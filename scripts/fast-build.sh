#!/bin/bash

# Fast build script with Docker BuildKit and caching
set -e

echo "🚀 Starting optimized Docker build..."

# Enable BuildKit for parallel builds and better caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build base image first (with heavy dependencies)
echo "📦 Building base image with heavy dependencies..."
docker build -f Dockerfile.base -t astrid-base:latest .

# Build services using the base image
echo "🔨 Building services..."
docker compose -f docker-compose.yml -f docker-compose.override.yml build

echo "✅ Build complete! Starting services..."
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

echo "🎉 All services started! Check status with: docker compose ps"
