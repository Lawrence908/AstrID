# Supernova Data Pipeline - Quick Start

## 🚀 Get Started in 5 Minutes

### Step 1: Source the Aliases

```bash
cd ~/github/AstrID
source astrid-aliases.sh
```

### Step 2: Query for Supernovae

```bash
astrid-sn-query
```

This queries MAST for observations from **all missions** of supernovae from 2010 onwards, searching:
- **3 years before** discovery (for reference images)
- **5 years after** discovery (for science images)

This wider window maximizes the chance of finding complete before/after pairs for training.

### Step 3: Preview Available Data

```bash
astrid-sn-download-dry
```

This shows you how many supernovae have downloadable before/after image pairs.

### Step 4: Download Test Data

```bash
astrid-sn-download-test
```

Downloads 10 supernovae for testing.

### Step 5: Check Results

```bash
astrid-sn-status
```

Shows what was downloaded and verified.

### Step 6: Get Your Survey UUID

Get your survey ID from the API:

```bash
curl http://localhost:9001/api/v1/surveys | jq '.[] | {id, name}'
```

Or create a new survey:

```bash
curl -X POST http://localhost:9001/api/v1/surveys \
    -H "Content-Type: application/json" \
    -d '{"name": "SN Training Data", "description": "Supernova FITS from MAST"}'
```

### Step 7: Ingest to Pipeline

Replace `<your-survey-uuid>` with the UUID from step 6:

```bash
astrid-sn-ingest-with <your-survey-uuid>
```

Done! Your observations are now in the pipeline.

---

## 🔄 Complete Pipeline (One Command)

After testing, you can run everything in one command:

```bash
astrid-sn-pipeline <your-survey-uuid> 50
```

This will:
1. Query MAST
2. Show preview
3. Ask for confirmation
4. Download FITS files
5. Ingest to pipeline

---

## 📊 What Gets Downloaded

For each supernova with viable data:

```
data/fits/
└── 2010O/
    ├── reference/          # Images BEFORE supernova
    │   ├── obs_001.fits
    │   └── obs_002.fits
    └── science/            # Images AFTER supernova
        ├── obs_003.fits
        └── obs_004.fits
```

These before/after pairs are used for:
- **Image differencing** (ZOGY algorithm)
- **ML training** (CNN detection & classification)

---

## 🎯 Key Features

✅ **Smart filtering** - Only downloads viable before/after pairs  
✅ **FITS validation** - Ensures files have required metadata  
✅ **Dry-run mode** - Preview before downloading  
✅ **Progress tracking** - See what's working, what's not  
✅ **Auto-ingestion** - Triggers full ML pipeline  

---

## 📚 More Information

- **Full alias reference**: See `scripts/ALIASES_REFERENCE.md`
- **Feature documentation**: See `scripts/README_DOWNLOAD_ENHANCEMENTS.md`
- **Complete workflow**: See `scripts/WORKFLOW_GUIDE.md`
- **Implementation details**: See `scripts/IMPLEMENTATION_SUMMARY.md`

---

## 🔧 Troubleshooting

### No viable supernovae?

Try recent SNe (better data availability):
```bash
astrid-sn-query-recent
astrid-sn-download-dry
```

### All observations have filesize: 0?

This means MAST doesn't have the actual files. The default query already filters for HST/JWST which have the best availability.

Try:
1. More recent supernovae: `astrid-sn-query-recent`
2. Wider time windows: `astrid-sn-query-wide`
3. More SNe in query: Edit the `--limit` in aliases

### Need help?

```bash
astrid-sn-help
```

---

## 💡 Pro Tips

1. **Always run dry-run first** - Saves time and bandwidth
2. **Start small** - Use `--limit 10` for initial testing
3. **Check status** - Use `astrid-sn-status` to see what worked
4. **Recent is better** - SNe from 2015+ have better data coverage
5. **Add to bashrc** - Auto-load aliases: `echo "source ~/github/AstrID/astrid-aliases.sh" >> ~/.bashrc`

---

## 🎓 Example Session

```bash
# Source aliases
source astrid-aliases.sh

# Query recent supernovae
astrid-sn-query-recent

# Preview what's available
astrid-sn-download-dry
# Shows: "✅ With both ref & sci (viable): 15"

# Download test batch
astrid-sn-download-test

# Check results
astrid-sn-status
# Shows: Downloaded files and verification status

# Get survey UUID
curl http://localhost:9001/api/v1/surveys | jq '.[0].id'
# Returns: "12345678-1234-1234-1234-123456789abc"

# Ingest to pipeline
astrid-sn-ingest-with 12345678-1234-1234-1234-123456789abc

# Monitor in Prefect
astrid-prefect  # Opens http://localhost:9004
```

Your ML pipeline is now running! 🎉

---

## Next Steps

1. **Monitor pipeline** - Check Prefect (http://localhost:9004) and MLflow (http://localhost:9003)
2. **Download more data** - Use `astrid-sn-download` (no test limit)
3. **Visualize results** - Check notebooks in `notebooks/training/`
4. **Train models** - Use the ingested data for ML training

