# Breast Cancer Triage Service

A containerised scikit-learn screening service for Meridian Diagnostics Lab. It classifies
cell-nucleus measurements from fine needle aspirate samples as likely **malignant** or likely
**benign**, so pathologists can prioritise their review queue. *Educational exercise, not a
medical device.*

- **Dataset:** `sklearn.datasets.load_breast_cancer`, exported to `data/raw/breast_cancer.csv`
  and versioned with DVC (v1 = 569 rows, v2 = 539 rows; v2 is used from Milestone 2 onward).
- **Model:** two candidates (Logistic Regression and Random Forest), each a `Pipeline` with a
  `StandardScaler`. The candidate with the best mean 5-fold CV malignant recall is selected
  (seed 42 everywhere). Malignant is the positive class.
- **Serving:** FastAPI + Uvicorn, with a Redis request counter, packaged as a multi-stage,
  non-root Docker image and run with Docker Compose.
- **Docker Hub:** <https://hub.docker.com/r/sharaz123/breast-cancer-triage-api>
- **Report:** [`docs/REPORT.md`](docs/REPORT.md) (screenshots in `docs/screenshots/`, raw
  command logs in `docs/logs/`).

## Repository layout

```
app/                    FastAPI service (main.py, config.py, schemas.py, counter.py)
training/               prepare_data.py (CSV export) and train.py (training + selection)
data/raw/*.csv.dvc      DVC pointer to the dataset (the CSV itself is not in Git)
dvc-remote/             local DVC remote, committed so that `dvc pull` works after a clone
dvc.yaml, dvc.lock      DVC pipeline (train stage) and its lock file
params.yaml             seed, split, CV folds, recall gate and hyperparameters
Dockerfile              production multi-stage image (train stage + slim serving stage)
Dockerfile.naive        baseline single-stage Ubuntu image, used for comparison
docker-compose.yml      api (built locally) + redis
docker-compose.hub.yml  api (pulled from Docker Hub) + redis, no build section
.env.example            documents every Compose variable (copy it to .env)
```

## 1. Set up and restore the data

```bash
git clone https://github.com/Sharazsony/breast-cancer-triage-service.git
cd breast-cancer-triage-service
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
dvc pull                # restores data/raw/breast_cancer.csv from dvc-remote/
```

## 2. Run the training pipeline

```bash
dvc repro               # runs the train stage -> artifacts/model.joblib, model_card.json, metrics.json
dvc metrics show
```

To run the training script directly:

```bash
python -m training.train --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
```

The script exits with a non-zero status if the test-set malignant recall is below 0.95.

## 3. Run the API locally (without Docker)

```bash
MODEL_VERSION=1.0.0 GIT_SHA=$(git rev-parse --short HEAD) uvicorn app.main:app --port 8000
# interactive docs: http://localhost:8000/docs
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample.json
```

## 4. Build and run the production image

```bash
dvc pull                # the training stage needs the CSV in the build context
docker build -t sharaz123/breast-cancer-triage-api:1.0.0 \
  --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
docker run -d --name triage -p 8000:8000 sharaz123/breast-cancer-triage-api:1.0.0
curl localhost:8000/model/info
```

Change the configuration at run time without rebuilding:

```bash
docker run -d -p 8001:8000 -e DECISION_THRESHOLD=0.9 -e LOG_LEVEL=DEBUG sharaz123/breast-cancer-triage-api:1.0.0
```

Baseline image, for comparison: `docker build -f Dockerfile.naive -t breast-cancer-triage-api:naive .`

## 5. Start the Compose stack

```bash
cp .env.example .env    # then edit DOCKERHUB_USERNAME / API_PORT if needed
GIT_SHA=$(git rev-parse --short HEAD) docker compose up -d --build
docker compose ps       # api and redis should both show (healthy)
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample.json
curl localhost:8000/stats
docker compose down     # keeps the redis-data volume (the counter survives)
docker compose down -v  # also removes the volume (the counter resets)
```

## 6. Push to Docker Hub

```bash
docker login -u sharaz123
docker compose build    # image name comes from DOCKERHUB_USERNAME and IMAGE_TAG in .env
docker tag sharaz123/breast-cancer-triage-api:1.0.0 sharaz123/breast-cancer-triage-api:latest
docker push sharaz123/breast-cancer-triage-api:1.0.0
docker push sharaz123/breast-cancer-triage-api:latest
```

Run from the registry only, in an empty directory holding `docker-compose.hub.yml` and `.env`:

```bash
docker compose -f docker-compose.hub.yml up -d
```

## API contract

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok"}`; does not depend on Redis |
| POST | `/predict` | all 30 features (numbers ≥ 0); returns label, malignant_probability, threshold, model_version; 422 on invalid input |
| GET | `/model/info` | model version and family, scikit-learn version, git commit, threshold, feature count, test metrics |
| GET | `/stats` | number of predictions served, counted in Redis |

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/app/artifacts/model.joblib` | model artifact location |
| `DECISION_THRESHOLD` | `0.5` | label is malignant when malignant_probability ≥ threshold |
| `LOG_LEVEL` | `INFO` | logging verbosity (logs go to stdout only) |
| `REDIS_HOST` | unset | request counting is enabled only when this is set |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL_VERSION`, `GIT_SHA` | build args | baked in at build time and reported by `/model/info` |
