# breast-cancer-triage-api

A FastAPI screening-support service that classifies fine needle aspirate cell-nucleus
measurements (the 30 features of the Wisconsin breast cancer dataset) as likely
**malignant** or likely **benign**, so pathologists can prioritise their review queue.
Malignant is the positive class. *Educational exercise, not a medical device.*

Source code: https://github.com/Sharazsony/breast-cancer-triage-service

## Tags

| Tag | Description |
|---|---|
| `1.0.0` | First release: Logistic Regression pipeline (StandardScaler + LR), scikit-learn 1.7.2 |
| `latest` | Points to the same image as `1.0.0` |

The image is a multi-stage build on `python:3.12.12-slim-bookworm`. It runs as a non-root
user (`app`, uid 10001), has a built-in `HEALTHCHECK` on `/health`, and contains no
training data.

## Run with docker run

```bash
docker run -d --name triage -p 8000:8000 sharaz123/breast-cancer-triage-api:1.0.0
curl localhost:8000/health
curl localhost:8000/model/info
# interactive API docs: http://localhost:8000/docs
```

## Run with Docker Compose (with the Redis request counter)

```yaml
services:
  api:
    image: sharaz123/breast-cancer-triage-api:1.0.0
    ports: ["8000:8000"]
    environment:
      REDIS_HOST: redis
    depends_on:
      redis: { condition: service_healthy }
  redis:
    image: redis:7.4.6-alpine
    command: ["redis-server", "--appendonly", "yes"]
    healthcheck: { test: ["CMD", "redis-cli", "ping"], interval: 5s, retries: 5 }
    volumes: [redis-data:/data]
volumes:
  redis-data:
```

```bash
docker compose up -d
curl localhost:8000/stats
```

## Endpoints

- `GET /health`: returns `{"status": "ok"}` and does not depend on Redis
- `POST /predict`: JSON with all 30 features (`mean_radius` … `worst_fractal_dimension`), each a number ≥ 0; invalid input returns 422
- `GET /model/info`: model version and family, scikit-learn version, git commit, threshold, feature count, test metrics
- `GET /stats`: predictions served, counted in Redis

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DECISION_THRESHOLD` | `0.5` | label is malignant when malignant_probability ≥ threshold |
| `LOG_LEVEL` | `INFO` | logging verbosity (stdout only) |
| `REDIS_HOST` | unset | enables the Redis prediction counter when set |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL_PATH` | `/app/artifacts/model.joblib` | model artifact inside the image |
| `MODEL_VERSION`, `GIT_SHA` | set at build time | reported by `/model/info` |

## Exposed port

`8000/tcp`: Uvicorn listens on `0.0.0.0:8000` inside the container.
