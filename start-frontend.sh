#!/bin/bash

# AstrID Frontend Startup Script
# This script starts the Next.js frontend development server

echo "🚀 Starting AstrID Frontend Development Server..."
echo ""

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: frontend/package.json not found. Please run this script from the AstrID root directory."
    exit 1
fi

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✅ Dependencies installed"
    echo ""
fi

# Copy latest diagrams and docs
echo "📋 Updating static files..."
cp docs/diagrams/*.svg frontend/public/docs/diagrams/ 2>/dev/null || echo "No SVG files to copy"
cp docs/*.md frontend/public/docs/ 2>/dev/null || echo "No markdown files to copy"
cp docs/consolidated-models.py frontend/public/docs/ 2>/dev/null || echo "No Python files to copy"
echo "✅ Static files updated"
echo ""

# Start the development server
echo "🌐 Starting development server on http://localhost:3000"
echo "📊 Frontend will be available at: http://localhost:3000"
echo "🔧 Backend API should be running on: http://127.0.0.1:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd frontend
npm run dev
