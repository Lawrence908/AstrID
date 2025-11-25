#!/bin/bash
# Query more supernovae from compiled catalog to get more viable pairs

LIMIT="${1:-1000}"  # Default to 1000, can override

echo "Querying $LIMIT supernovae from compiled catalog..."
python3 scripts/query_sn_fits_from_catalog.py \
    --catalog resources/sncat_compiled.txt \
    --output output/sn_queries_compiled_${LIMIT}.json \
    --min-year 2010 \
    --limit "$LIMIT" \
    --missions TESS GALEX PS1 SWIFT \
    --days-before 1095 \
    --days-after 730

echo ""
echo "Checking results..."
python3 -c "
import json
data = json.load(open('output/sn_queries_compiled_${LIMIT}.json'))
viable = sum(1 for sn in data if sn.get('reference_observations') and sn.get('science_observations'))
print(f'Total: {len(data)}')
print(f'✅ Viable (both ref & sci): {viable} ({viable/len(data)*100:.1f}%)')
"
