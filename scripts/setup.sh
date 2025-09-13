#!/bin/bash

# AstrID Project Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up AstrID development environment..."

# Check if Python 3.12.3+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ $(echo "$python_version >= 3.12.3" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.12.3+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
else
    echo "✅ uv is already installed"
fi

# Install project dependencies
echo "📦 Installing project dependencies..."
uv pip install --dev .

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Check Docker availability
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✅ Docker and Docker Compose are available"
    echo "🐳 You can start the development environment with:"
    echo "   docker-compose -f docker/compose.yml up -d"
else
    echo "⚠️  Docker or Docker Compose not found"
    echo "   Install Docker to use the containerized development environment"
fi

echo ""
echo "🎉 AstrID development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Start development services: docker-compose -f docker/compose.yml up -d"
echo "3. Run tests: uv run pytest"
echo "4. Start development: uv run uvicorn src.adapters.api.main:app --reload"
echo ""
echo "Happy coding! 🚀"
