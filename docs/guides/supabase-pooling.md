## Supabase Connection Pooling Hardening

This guide explains how to keep AstrID under Supabase Session-mode client limits and where Transaction pooling is safe to use.

### Why this matters
- Each process (API, MLflow, Prefect, worker processes) creates its own DB pool.
- Supabase Session mode has strict client caps; exceeding them causes `MaxClientsInSessionMode` errors.

### Target configuration
- pool_size=1, max_overflow=0 for all services
- Minimal process counts (Gunicorn workers=1, Dramatiq processes=1)
- Short-lived DB sessions (open late, close early)

### Docker Compose changes

1) MLflow backend-store-uri (add pool params):

```yaml
mlflow:
  command: >
    bash -c "pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org psycopg2-binary &&
    mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri postgresql+psycopg2://postgres.$$MLFLOW_SUPABASE_PROJECT_REF:$$MLFLOW_SUPABASE_PASSWORD@$$MLFLOW_SUPABASE_HOST/postgres?pool_size=1&max_overflow=0&connect_timeout=10 \
    --default-artifact-root s3://astrid-models \
    --gunicorn-opts '--timeout 60 --workers 1 --worker-class sync --max-requests 100 --max-requests-jitter 10'"
```

2) Prefect server DB URL (add pool params):

```yaml
prefect:
  environment:
    - PREFECT_SERVER_DATABASE_CONNECTION_URL=postgresql+asyncpg://postgres.${PREFECT_SUPABASE_PROJECT_REF}:${PREFECT_SUPABASE_PASSWORD}@${PREFECT_SUPABASE_HOST}/postgres?pool_size=1&max_overflow=0&timeout=10
```

3) Consider Transaction pooling
- Compatible with: API requests, MLflow tracking, Prefect server, and short worker tasks
- Avoid if you rely on: temp tables, session-level state across requests, long-held sessions, server-side cursors held across transactions, advisory locks spanning long durations
- If Supabase provides a transaction-pooler host, point `${SUPABASE_HOST}`, `${MLFLOW_SUPABASE_HOST}`, `${PREFECT_SUPABASE_HOST}` to that endpoint

### Code-level practices
- Keep SQLAlchemy pools minimal (already set in `src/core/constants.py`): `pool_size=1`, `max_overflow=0`
- Use short transactions; no long-lived sessions in workers
- Add `connect_timeout`/statement timeouts to fail fast on bad connections

### Verification checklist
- [ ] `docker-compose.yaml` updated for MLflow and Prefect URLs
- [ ] Process counts: MLflow workers=1; Dramatiq `-p 1 -t 1`
- [ ] Logs show no `MaxClientsInSessionMode`
- [ ] API `/health` prints pool health OK
- [ ] Workers complete tasks without DB connection errors

### Rollout steps
1. Bring the stack down: `docker-compose down`
2. Start clean with rebuild: `docker-compose up --build -d`
3. Watch logs: `docker-compose logs -f`

### Notes
- If you need more concurrency, consider upgrading plan limits or isolating MLflow to a separate Postgres instance.


