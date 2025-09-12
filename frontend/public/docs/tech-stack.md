# **AstrID:** *Tech stack*

## Language, tooling, quality

* Python 3.11
* **uv** for env/deps
* **Ruff** (lint/format) + **MyPy** (types) + **pre-commit**
* Pytest (+ pytest-asyncio)

**Quick init**

```bash
uv init astrid && cd astrid
uv add fastapi uvicorn[standard] pydantic-settings python-multipart
uv add "sqlalchemy>=2" alembic asyncpg psycopg[binary]
uv add "mlflow>=2.14" dvc[ssh,s3] boto3
uv add astropy astroquery photutils sep opencv-python scikit-image
uv add numpy pandas scipy scikit-learn
uv add tensorflow==2.13.0 keras==2.13.1
uv add structlog loguru
uv add redis dramatiq prefect "httpx[http2]" python-dotenv
# dev
uv add --dev ruff mypy black pre-commit pytest pytest-asyncio httpx
```

**.pre-commit-config.yaml**

```yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.5.7
  hooks: [{id: ruff}, {id: ruff-format}]
- repo: https://github.com/psf/black
  rev: 24.4.2
  hooks: [{id: black}]
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.10.0
  hooks: [{id: mypy, additional_dependencies: ["pydantic==2.*","sqlalchemy==2.*"]}]
```

## Data & storage

* **Postgres (Supabase)** for relational data (observations, candidates, detections, runs, users).
* **Cloudflare R2** for objects (FITS, PNGs/masks/diff frames, DVC/MLflow artifacts). R2 is **S3-API compatible**.
* **SQLAlchemy 2 + Alembic** for ORM & migrations.
* **DVC** to version datasets & model artifacts.

**R2 env** (works for DVC & MLflow):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=auto
AWS_S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
MLFLOW_S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
```

**DVC remote (R2)**

```bash
dvc init
dvc remote add -d r2 s3://astrid-data
dvc remote modify r2 endpointurl $AWS_S3_ENDPOINT_URL
dvc remote modify r2 access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify r2 secret_access_key $AWS_SECRET_ACCESS_KEY
```

**MLflow**

```
MLFLOW_TRACKING_URI=postgresql+psycopg://USER:PWD@HOST:PORT/DB  # Supabase
MLFLOW_S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
```

## Services & orchestration

* **FastAPI** (control-plane API + auth check with Supabase JWTs)
* **Prefect** for orchestration/scheduling (chosen for simplicity); **Dramatiq** workers (Redis broker)
* **Redis** for queues and lightweight pub/sub (e.g., `observation.ingested`, `detection.scored`)

## ML / CV

* **Keras/TensorFlow** U-Net (segmentation)
* **photutils + SEP** (source extraction) + **scikit-image/OpenCV** (preprocess, registration, differencing/ZOGY)
* Optional later: ConvLSTM/3D-CNN or VAE anomaly head, but keep interfaces thin so you can swap frameworks

## Experiment tracking

* **MLflow** runs, metrics, params, registry (artifacts in R2)
* DVC for dataset/model versioning

## Observability (light)

* **structlog** JSON logs; optional **Sentry** DSN for error reporting
* Add Prometheus/Grafana later if needed

## Frontend

* **Next.js (TS) + Tailwind**
* REST to FastAPI; **SSE** endpoint for live detections

## Packaging & deploy

* Docker for all services; **docker compose** for dev
* Single GCP VM is fine to start; later harden with systemd or k8s if scale warrants
* **GitHub Actions** (lint, types, tests, image build/push, optional DVC/MLflow artifact steps)

---

# Architecture (DDD + evented flow)

## Bounded contexts

1. **Ingestion** → Astroquery, FITS download, metadata; emits `observation.ingested`
2. **Calibration & Registration** → bias/dark/flat (if applicable), WCS/align; emits `observation.preprocessed`
3. **Difference & Candidates** → ZOGY/classic differencing + SEP/photutils; emits `candidate.found`
4. **Inference** → U-Net / anomaly head; emits `detection.scored`
5. **Validation & Curation** → human review, labels; emits `detection.validated`
6. **Catalog & Analytics** → durable store + dashboard queries
7. **Notification** → rules → email/webhook/Slack
8. **API Gateway** → auth, REST, SSE, admin triggers

## Data model (high level)

* `surveys`, `observations`, `preprocess_runs`, `candidates`,
  `detections`, `models`, `validation_events`, `alerts`

## End-to-end

1. discover/fetch → 2) preprocess → 3) difference/extract → 4) infer/score
   → 5) persist (R2 + Postgres) + track (MLflow) → 6) notify/stream → 7) validate/curate

---

# Repo layout (no nested package)

```
astrid/
  src/
    domains/                         # pure logic (no FastAPI/SQL/TF imports)
      observations/ {models.py,services.py,repositories.py,events.py}
      preprocessing/ {...}
      differencing/ {...}
      detection/ {...}
      curation/ {...}
      catalog/ {...}
    adapters/                        # ports/adapters (frameworks & IO)
      api/           # FastAPI app
        main.py
        routes/{observations.py,detections.py,stream.py}
        deps.py
      db/            # SQLAlchemy + Alembic
        session.py
        repositories.py
        alembic/
      storage/       # Cloudflare R2, DVC, MLflow
        r2_client.py
        dvc_client.py
        mlflow_client.py
      messaging/     # Redis pub/sub
        redis_client.py
      imaging/       # astropy/photutils/opencv wrappers
        fits_io.py
        registration.py
        differencing_zogy.py
      ml/            # Keras/TensorFlow behind thin ports
        unet.py
        anomaly_heads.py
      scheduler/     # Prefect flows
        flows/{nightly_ingest.py,process_new.py,retrain.py}
      workers/       # Dramatiq tasks
        ingest.py preprocess.py difference.py infer.py notify.py
      observability/
        logging.py sentry.py
      auth/
        supabase.py
    utils/
      config.py ids.py
  tests/ (unit/integration/e2e mirroring src/)
  notebooks/
  dvc.yaml
  docker/
    api.Dockerfile worker.Dockerfile prefect.Dockerfile compose.yml
  .github/workflows/ci.yml
  README.md
```

**Import rules**

* `domains/*` → may import `domains/*` + `utils/*` only
* `adapters/*` → may import `domains/*`, `utils/*`, external libs
* **Composition** (wiring interfaces→impls) happens in `adapters/api/deps.py`, workers, and flows

---

# Minimal stubs to paste

**src/adapters/api/main.py**

```python
from fastapi import FastAPI
from .routes import observations, detections, stream

app = FastAPI(title="AstrID API")
app.include_router(observations.router, prefix="/observations", tags=["observations"])
app.include_router(detections.router, prefix="/detections", tags=["detections"])
app.include_router(stream.router, prefix="/stream", tags=["stream"])
```

**src/adapters/scheduler/flows/process\_new\.py**

```python
from prefect import flow, task

@task(retries=3) def ingest_window(): ...
@task(retries=3) def preprocess(obs_id: str): ...
@task(retries=3) def difference(obs_id: str): ...
@task(retries=3) def infer(obs_id: str): ...
@task           def persist_and_notify(result): ...

@flow(name="process-new-observations")
def run():
    obs_ids = ingest_window()
    for oid in obs_ids:
        pre = preprocess.submit(oid)
        dif = difference.submit(oid, wait_for=[pre])
        res = infer.submit(oid, wait_for=[dif])
        persist_and_notify.submit(res)
```

**src/utils/config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_s3_endpoint_url: str = "https://<accountid>.r2.cloudflarestorage.com"
    mlflow_tracking_uri: str
    mlflow_s3_endpoint_url: str | None = None
    redis_url: str = "redis://localhost:6379/0"
    supabase_jwt_secret: str | None = None

settings = Settings()  # auto-loads from env/.env
```

---

# API surface (first pass)

* `POST /observations/sync?s=<survey>&from=&to=` → enqueue ingest
* `GET /observations?...`
* `GET /candidates?obs_id=...`
* `POST /detections/infer?obs_id=...`
* `GET /detections?status=&min_score=&since=...`
* `POST /detections/{id}/validate`
* `GET /stream/detections` (SSE)

---

# Scheduler & workers

* **Prefect flows**: `nightly_ingest`, `process_new`, `weekly_retrain`
* **Dramatiq workers**: idempotent steps; payloads carry Postgres IDs + R2 URIs; artifacts are content-addressed (hash in path) to make retries safe

---

# Modeling strategy (near-term)

* U-Net for segmentation/localization
* Difference-first pipeline (ZOGY/classic) → reduce FPs; cutouts into optional ConvLSTM/3D-CNN or VAE
* Track PR/ROC and calibration in MLflow; promote models when AUCPR beats baseline by δ