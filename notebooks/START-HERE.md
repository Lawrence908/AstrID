### Start Here: Real Data → Training (ASTR-113 + ASTR-106)

This note is the quick launcher for continuing the real-data training work. It captures the decisions, next steps, and the minimum commands/API calls you’ll need.

### TL;DR decisions
- **Prefect = orchestration/scheduling**, wiring together multi-step flows and tracking state/lineage.
- **Dramatiq = workers that do the heavy work** (ingestion, preprocessing, differencing, detection, sample creation).
- Automate data collection now (ASTR-113). Keep training (ASTR-106) in the notebook until stable, then extract to a headless module and orchestrate with Prefect.

### 10-minute bootstrap checklist (dev/staging)
- Services up: api, worker, prefect, mlflow, redis
  - Health: `http://localhost:8000/health`, MLflow UI: `http://localhost:5000`, Prefect: `http://localhost:4200`
- DB migrated (includes training tables from docs)
- R2 and DB env vars set
- Create/refresh a small dataset window via the API and capture `dataset_id`

Example request to create a training dataset (ASTR-113):
```bash
curl -X POST http://localhost:8000/training/datasets/collect \
  -H 'Content-Type: application/json' \
  -d '{
    "survey_ids": ["hst"],
    "start": "2024-01-01T00:00:00",
    "end": "2024-12-31T23:59:59",
    "confidence_threshold": 0.7,
    "max_samples": 200,
    "name": "hst_2024_training"
  }'
```
List and inspect datasets:
```bash
curl http://localhost:8000/training/datasets | jq
curl http://localhost:8000/training/datasets/<dataset_id> | jq
```

### Where things live (quick map)
- Tickets/docs
  - `docs/tickets/113.md` (Real Data Loading Integration)
  - `docs/tickets/106.md` (Training Notebook + MLflow)
  - `docs/training-data-pipeline.md` (runbook + API examples)
- API and services (likely touch points)
  - `src/domains/ml/training_data/` (models/services/api/workers/flows)
  - `src/adapters/api/docs.py` and `docs/api/openapi.yaml` for routes
- Notebooks
  - `notebooks/ml_training_data/real_data_loading.ipynb` (smoke test: list/inspect dataset)
  - `notebooks/training/model_training.ipynb` (replace synthetic data with real samples by `dataset_id`)

### What to do next (sequence)
1) ASTR-113: Data pipeline
- Verify you can create a dataset via `/training/datasets/collect` and see it in GET endpoints
- Ensure basic MLflow logging for dataset creation appears (run name `training_data_<dataset_id>`)
- If `total_samples == 0`, widen date range or lower `confidence_threshold`

2) ASTR-106: Training notebook
- Add a helper cell to fetch samples for `dataset_id` via API and build a `Dataset`
- Train with real images/masks; monitor MLflow metrics and artifacts

3) Orchestration
- Keep heavy steps as Dramatiq actors (ingest, preprocess, differencing, detection, sample/mask creation)
- Use a Prefect flow to chain: collect detections → prepare samples → create dataset → (optionally) trigger training

### Rules of thumb
- **Dramatiq**: per-observation/per-detection work; CPU/GPU-bound; idempotent; parallelizable
- **Prefect**: cross-service orchestration; retries; alerts; scheduling; lineage
- Automate data collection early; automate training after the notebook loop and metrics are stable

### Minimal API surface to support the notebook
- POST `/training/datasets/collect` → returns `dataset_id`
- GET `/training/datasets` → list
- GET `/training/datasets/{dataset_id}` → counts/metadata
- Optional: GET `/training/datasets/{dataset_id}/samples` → list (paths + labels) for direct notebook loading

### Common pitfalls
- No detections in range → expand time window or lower threshold
- Preprocessing/differencing not run → trigger those flows first
- R2 path mismatches → check that stored paths are accessible to the notebook
- MLflow not logging → verify server URL/env and run names

### Milestones checklist
- [ ] Create dataset via API and see non-zero samples
- [ ] Visualize a few images/masks in notebook from returned paths
- [ ] Successful training epoch with real data; metrics in MLflow
- [ ] Prefect flow chaining data collection end-to-end in staging
- [ ] Extract notebook training into a callable module for future scheduling

### References
- See `docs/training-data-pipeline.md` for fuller runbook and schema examples
- Implementation scaffolds for `TrainingDataCollector`, `RealDataLoader`, `TrainingDatasetManager` are in `docs/tickets/113.md`
