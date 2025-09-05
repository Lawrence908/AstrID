#!/bin/bash
# Lint script for AstrID project
# uv run pre-commit run ruff --all-files
# uv run pre-commit run black --all-files
# uv run pre-commit run mypy --all-files


set -e

echo "🔍 Running pre-commit hooks on all files..."
echo "📝 Note: Markdown files are excluded from formatting to preserve structure"
uv run pre-commit run --all-files

echo "✅ All pre-commit hooks passed!"
