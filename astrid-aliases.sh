#!/bin/bash

# AstrID Docker Compose Aliases
# Source this file in your shell: source astrid-aliases.sh

echo "  AstrID Docker Aliases               - astrid-"
# Start everything (including Organizr)
# Access the unified dashboard
# See all service URLs
echo "   astrid-up      astrid-organizr       astrid-urls        astrid-status"
echo "   astrid-logs    astrid-restart        astrid-build        astrid-prune"



# Core service management
alias astrid-up='docker-compose -p astrid-dev -f docker-compose.yaml up -d'
alias astrid-down='docker-compose -p astrid-dev -f docker-compose.yaml down'
alias astrid-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f'
alias astrid-restart='docker-compose -p astrid-dev -f docker-compose.yaml restart'
alias astrid-build='docker-compose -p astrid-dev -f docker-compose.yaml up --build -d'
alias astrid-prune='docker-compose -p astrid-dev -f docker-compose.yaml down -v --remove-orphans && sudo docker system prune -a --volumes'

# Service-specific logs
alias astrid-api-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f api'
alias astrid-worker-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f worker'
alias astrid-prefect-worker-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f prefect-worker'
alias astrid-frontend-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f frontend'
alias astrid-redis-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f redis'
alias astrid-mlflow-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f mlflow'
alias astrid-prefect-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f prefect'
alias astrid-organizr-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f organizr'

# Service-specific shells
alias astrid-shell='docker-compose -p astrid-dev -f docker-compose.yaml exec api bash'
alias astrid-worker-shell='docker-compose -p astrid-dev -f docker-compose.yaml exec worker bash'
alias astrid-prefect-worker-shell='docker-compose -p astrid-dev -f docker-compose.yaml exec prefect-worker bash'
alias astrid-frontend-shell='docker-compose -p astrid-dev -f docker-compose.yaml exec frontend sh'

# Service-specific restarts
alias astrid-api-restart='docker-compose -p astrid-dev -f docker-compose.yaml restart api'
alias astrid-worker-restart='docker-compose -p astrid-dev -f docker-compose.yaml restart worker'
alias astrid-prefect-worker-restart='docker-compose -p astrid-dev -f docker-compose.yaml restart prefect-worker'
alias astrid-frontend-restart='docker-compose -p astrid-dev -f docker-compose.yaml restart frontend'

# Development helpers
alias astrid-status='docker-compose -p astrid-dev -f docker-compose.yaml ps'
alias astrid-health='curl -s http://127.0.0.1:8000/health | jq .'
alias astrid-frontend='open http://localhost:3010'
alias astrid-api-docs='open http://127.0.0.1:8000/docs'
alias astrid-mlflow='open http://localhost:5000'
alias astrid-prefect='open http://localhost:4200'
alias astrid-organizr='open http://localhost:8080'

# Database troubleshooting
alias astrid-db-test='docker-compose -p astrid-dev -f docker-compose.yaml exec api python -c "import asyncio; from src.core.db.session import test_connection; print(asyncio.run(test_connection()))"'
alias astrid-db-shell='docker-compose -p astrid-dev -f docker-compose.yaml exec api python -c "import asyncio; from src.core.db.session import get_db; async def test(): async for db in get_db(): print(\"DB connected\"); break; asyncio.run(test())"'

# Clean restart (for timeout issues)
alias astrid-clean-restart='docker-compose -p astrid-dev -f docker-compose.yaml down && docker system prune -f && docker-compose -p astrid-dev -f docker-compose.yaml up -d'

# Service-specific builds (when you change code)
alias astrid-api-build='docker-compose -p astrid-dev -f docker-compose.yaml build api && docker-compose -p astrid-dev -f docker-compose.yaml up -d api'
alias astrid-prefect-build='docker-compose -p astrid-dev -f docker-compose.yaml build prefect && docker-compose -p astrid-dev -f docker-compose.yaml up -d prefect'

# Monitor specific problematic services
alias astrid-db-logs='docker-compose -p astrid-dev -f docker-compose.yaml logs -f api prefect | grep -E "(database|timeout|connection|asyncpg)"'
alias astrid-errors='docker-compose -p astrid-dev -f docker-compose.yaml logs --tail=50 | grep -i error'

# Quick access URLs
alias astrid-urls='echo "Organizr Dashboard: http://localhost:8080" && echo "Frontend: http://localhost:3010" && echo "API: http://127.0.0.1:8000" && echo "API Docs: http://127.0.0.1:8000/docs" && echo "MLflow: http://localhost:5000" && echo "Prefect: http://localhost:4200"'

# echo "  Available aliases:"
# echo "    astrid-up, astrid-down, astrid-logs, astrid-restart, astrid-build, astrid-prune"
# echo "    astrid-api-logs, astrid-worker-logs, astrid-prefect-worker-logs, astrid-frontend-logs"
# echo "    astrid-redis-logs, astrid-mlflow-logs, astrid-prefect-logs"
# echo "    astrid-shell, astrid-worker-shell, astrid-prefect-worker-shell, astrid-frontend-shell"
# echo "    astrid-api-restart, astrid-worker-restart, astrid-prefect-worker-restart, astrid-frontend-restart"
# echo "    astrid-status, astrid-health, astrid-frontend, astrid-api-docs, astrid-prefect"
# echo "    astrid-db-test, astrid-db-shell, astrid-clean-restart, astrid-db-logs, astrid-errors"
# echo "    astrid-api-build, astrid-prefect-build"
# echo "    astrid-urls (shows all service URLs)"
