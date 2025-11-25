#!/bin/bash
# Query compiled catalog with missions that actually have data

# Activate venv if needed
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Query with missions that have good coverage
python3 scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_compiled.txt \
    --output output/sn_queries_compiled.json \
    --min-year 2010 \
    --limit 200 \
    --missions TESS GALEX PS1 SWIFT \
    --days-before 1095 \
    --days-after 730

echo ""
echo "Query complete! Check results:"
echo "  python3 -c \"import json; data = json.load(open('output/sn_queries_compiled.json')); viable = sum(1 for sn in data if sn.get('reference_observations') and sn.get('science_observations')); print(f'Viable supernovae: {viable}/{len(data)}')\""
