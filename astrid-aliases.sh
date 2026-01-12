#!/bin/bash

# AstrID Docker Compose Aliases
# Source this file in your shell: source astrid-aliases.sh

echo "  AstrID Docker Aliases               - astrid-"
# Start everything (including Organizr)
# Access the unified dashboard
# See all service URLs
echo "   astrid-up      astrid-organizr       astrid-urls        astrid-status"
echo "   astrid-logs    astrid-restart        astrid-build        astrid-prune"
echo "   astrid-health  astrid-api-health     astrid-prefect-health   astrid-mlflow-health"



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
alias astrid-health='curl -s http://localhost:9001/health | jq .'
alias astrid-api-health='docker-compose -p astrid-dev -f docker-compose.yaml exec api curl -sf http://localhost:8000/health || echo "API unhealthy"'
alias astrid-prefect-health='docker-compose -p astrid-dev -f docker-compose.yaml exec prefect curl -sf http://localhost:4200/health || echo "Prefect unhealthy"'
alias astrid-mlflow-health='docker-compose -p astrid-dev -f docker-compose.yaml exec mlflow python -c "import requests; print(requests.get('http://localhost:5000/health').status_code)" || echo "MLflow unhealthy"'

alias astrid-restart-api='docker-compose -p astrid-dev -f docker-compose.yaml restart api && docker-compose -p astrid-dev -f docker-compose.yaml logs -f api'
alias astrid-restart-prefect='docker-compose -p astrid-dev -f docker-compose.yaml restart prefect && docker-compose -p astrid-dev -f docker-compose.yaml logs -f prefect'
alias astrid-restart-mlflow='docker-compose -p astrid-dev -f docker-compose.yaml restart mlflow && docker-compose -p astrid-dev -f docker-compose.yaml logs -f mlflow'
alias astrid-frontend='open http://localhost:9002'
alias astrid-api-docs='open http://localhost:9001/docs'
alias astrid-mlflow='open http://localhost:9003'
alias astrid-prefect='open http://localhost:9004'
alias astrid-organizr='open http://localhost:9005'

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
alias astrid-urls='echo "Organizr Dashboard: http://localhost:9005" && echo "Frontend: http://localhost:9002" && echo "API: http://localhost:9001" && echo "API Docs: http://localhost:9001/docs" && echo "MLflow: http://localhost:9003" && echo "Prefect: http://localhost:9004"'

# ============================================================================
# Supernova Data Pipeline Aliases
# ============================================================================
# Usage order: 1) Query → 2) Dry-run → 3) Download → 4) Status → 5) Ingest
# ============================================================================

# Query MAST for observations
alias astrid-sn-query='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --missions GALEX PS1 SWIFT TESS --min-year 2010 --days-before 1095 --days-after 1825 --limit 100 --output output/sn_queries.json'
alias astrid-sn-query-all='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --no-mission-filter --min-year 2010 --days-before 1095 --days-after 1825 --limit 100 --output output/sn_queries_all.json'
alias astrid-sn-query-recent='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --missions GALEX PS1 SWIFT TESS --min-year 2015 --days-before 1095 --days-after 1825 --limit 100 --output output/sn_queries_recent.json'
alias astrid-sn-query-wide='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --missions GALEX PS1 SWIFT TESS --min-year 2010 --days-before 1825 --days-after 2555 --limit 100 --output output/sn_queries_wide.json'
alias astrid-sn-query-hst='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --missions HST JWST --min-year 2010 --days-before 1095 --days-after 1825 --limit 100 --output output/sn_queries_hst.json'

# Preview downloads (ALWAYS run before downloading!)
alias astrid-sn-download-dry='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --dry-run'
alias astrid-sn-download-dry-all='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries_all.json --output-dir data/fits --dry-run'
alias astrid-sn-download-dry-recent='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries_recent.json --output-dir data/fits --dry-run'

# Download FITS files
alias astrid-sn-download='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --require-both --verify-fits'
alias astrid-sn-download-test='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --require-both --verify-fits --limit 10'
alias astrid-sn-download-partial='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --allow-partial --verify-fits'
alias astrid-sn-download-fast='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --require-both --no-verify'
alias astrid-sn-download-more='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --require-both --verify-fits --max-obs 10'

# Check download status
alias astrid-sn-status='cd ~/github/AstrID && if [ -f data/fits/download_results.json ]; then jq -r ".[] | \"\(.sn_name): \(.reference_files | length) ref, \(.science_files | length) sci, \(.verified_files | map(select(.valid)) | length) valid, \(.errors | length) errors\"" data/fits/download_results.json; else echo "No download_results.json found. Run astrid-sn-download first."; fi'

# Ingest into pipeline (requires survey UUID)
astrid-sn-ingest-with() {
    if [ -z "$1" ]; then
        echo "Usage: astrid-sn-ingest-with <survey-uuid>"
        echo "Get your survey UUID: curl http://localhost:9001/api/v1/surveys | jq '.[] | {id, name}'"
        return 1
    fi
    cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/ingest_sn_fits_to_pipeline.py --input-dir data/fits --survey-id "$1" --verify-wcs
}

# Complete pipeline (query → dry-run → download → ingest)
astrid-sn-pipeline() {
    if [ -z "$1" ]; then
        echo "Usage: astrid-sn-pipeline <survey-uuid> [limit]"
        echo "Example: astrid-sn-pipeline 12345678-1234-1234-1234-123456789abc 10"
        return 1
    fi
    LIMIT=${2:-50}
    cd ~/github/AstrID && source .venv/bin/activate && \
        echo "Step 1: Querying MAST..." && \
        python3 scripts/query_sn_fits_from_catalog.py --catalog resources/sncat_latest_view.txt --missions GALEX PS1 SWIFT TESS --min-year 2010 --days-before 1095 --days-after 1825 --limit 100 --output output/sn_queries.json && \
        echo -e "\nStep 2: Preview (dry-run)..." && \
        python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --dry-run && \
        echo -e "\n⚠️  Ready to download. Continue? (y/N)" && \
        read -r confirm && \
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "Step 3: Downloading..." && \
            python3 scripts/download_sn_fits.py --query-results output/sn_queries.json --output-dir data/fits --require-both --verify-fits --limit "$LIMIT" && \
            echo -e "\nStep 4: Ingesting to pipeline..." && \
            python3 scripts/ingest_sn_fits_to_pipeline.py --input-dir data/fits --survey-id "$1" --verify-wcs
        else
            echo "Cancelled."
        fi
}

# Check ingestion status
alias astrid-sn-ingest-status='cd ~/github/AstrID && if [ -f data/fits/ingestion_results.json ]; then jq -r ".[] | \"\(.sn_name): \(.reference_observations | length) ref, \(.science_observations | length) sci, \(.errors | length) errors\"" data/fits/ingestion_results.json; else echo "No ingestion_results.json found. Run astrid-sn-ingest-with first."; fi'

# Clean downloaded files (with confirmation)
alias astrid-sn-clean='cd ~/github/AstrID && echo "⚠️  This will delete all files in data/fits/*/" && echo "Press Ctrl+C to cancel, or Enter to continue..." && read && rm -rf data/fits/*/ && echo "✅ Cleaned data/fits directory"'

# Audit and organize training data
alias astrid-sn-audit='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/audit_sn_downloads.py --input-dir output/fits_downloads'
alias astrid-sn-audit-validate='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/audit_sn_downloads.py --input-dir output/fits_downloads --validate-fits'
alias astrid-sn-organize='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/organize_training_pairs.py --input-dir output/fits_downloads --output-dir output/fits_training'
alias astrid-sn-organize-symlink='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/organize_training_pairs.py --input-dir output/fits_downloads --output-dir output/fits_training --symlink'
alias astrid-sn-organize-decompress='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/organize_training_pairs.py --input-dir output/fits_downloads --output-dir output/fits_training --decompress'

# Generate difference images for training
alias astrid-sn-diff='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/generate_difference_images.py --input-dir output/fits_training --output-dir output/difference_images'
alias astrid-sn-diff-viz='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/generate_difference_images.py --input-dir output/fits_training --output-dir output/difference_images --visualize'
alias astrid-sn-diff-test='cd ~/github/AstrID && source .venv/bin/activate && python3 scripts/generate_difference_images.py --input-dir output/fits_training --output-dir output/difference_images --sn 2014J --visualize'

# Show help
alias astrid-sn-help='echo "Supernova Pipeline Aliases:" && echo "" && echo "QUERY (Step 1):" && echo "  astrid-sn-query          - Default query (GALEX/PS1/SWIFT/TESS, 2010+, 3yr/5yr)" && echo "  astrid-sn-query-all      - All missions" && echo "  astrid-sn-query-recent   - Recent SNe (2015+)" && echo "  astrid-sn-query-wide     - Wider time windows" && echo "  astrid-sn-query-hst      - HST/JWST only" && echo "" && echo "PREVIEW (Step 2):" && echo "  astrid-sn-download-dry    - Preview what would be downloaded" && echo "" && echo "DOWNLOAD (Step 3):" && echo "  astrid-sn-download       - Download complete pairs (default)" && echo "  astrid-sn-download-test - Download 10 SNe for testing" && echo "  astrid-sn-download-partial - Allow partial data" && echo "  astrid-sn-download-fast  - Skip verification" && echo "  astrid-sn-download-more  - More obs per SN (10 instead of 3)" && echo "" && echo "STATUS (Step 4):" && echo "  astrid-sn-status         - Check download results" && echo "" && echo "AUDIT & ORGANIZE (Step 5):" && echo "  astrid-sn-audit          - Audit downloaded files" && echo "  astrid-sn-audit-validate - Audit with FITS validation" && echo "  astrid-sn-organize       - Organize pairs for training" && echo "  astrid-sn-organize-symlink - Organize using symlinks" && echo "  astrid-sn-organize-decompress - Decompress .fits.gz" && echo "" && echo "DIFFERENCING (Step 6):" && echo "  astrid-sn-diff           - Generate difference images" && echo "  astrid-sn-diff-viz       - Generate with visualizations" && echo "  astrid-sn-diff-test      - Test on SN 2014J only" && echo "" && echo "INGEST (Step 7):" && echo "  astrid-sn-ingest-with <uuid> - Ingest to pipeline" && echo "  astrid-sn-ingest-status  - Check ingestion results" && echo "" && echo "COMPLETE PIPELINE:" && echo "  astrid-sn-pipeline <uuid> [limit] - Run all steps" && echo "" && echo "UTILITIES:" && echo "  astrid-sn-clean          - Clean downloaded files" && echo "  astrid-sn-help           - Show this help"'

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
# echo "    astrid-sn-* (supernova pipeline commands - see astrid-sn-help)"
