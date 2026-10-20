"""FastAPI service for breast-cancer triage predictions.

Malignant is the positive class: label = "malignant" when
malignant_probability >= DECISION_THRESHOLD.
"""
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, Request

from app.config import get_settings
from app.counter import PredictionCounter
from app.schemas import FEATURE_NAMES, PredictRequest, PredictResponse

settings = get_settings()


def configure_logging(level: str) -> None:
    """Send all logs (ours and uvicorn's) to standard output only."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.setLevel(level)
        lg.propagate = False


configure_logging(settings.log_level)
logger = logging.getLogger("triage")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(settings.model_path)
    if not model_path.is_file():
        logger.critical("Model artifact not found at MODEL_PATH=%s - refusing to start. "
                        "Run `dvc repro` (or build the Docker image) first.", model_path)
        raise RuntimeError(f"Model artifact not found: {model_path}")

    app.state.model = joblib.load(model_path)  # loaded once, not per request
    card_path = model_path.with_name("model_card.json")
    app.state.card = json.loads(card_path.read_text()) if card_path.is_file() else {}
    if app.state.card.get("feature_names", FEATURE_NAMES) != FEATURE_NAMES:
        raise RuntimeError("Model feature order does not match the API schema")
    app.state.counter = PredictionCounter(settings.redis_host, settings.redis_port)
    logger.info("Model ready: %s loaded from %s (threshold=%s)",
                app.state.card.get("model_family"), model_path, settings.decision_threshold)
    yield


app = FastAPI(title="Breast Cancer Triage API", version=settings.model_version, lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness probe used by the Docker HEALTHCHECK; never touches Redis."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    row = np.array([[getattr(payload, f) for f in FEATURE_NAMES]], dtype=float)
    proba = float(request.app.state.model.predict_proba(row)[0, 1])
    label = "malignant" if proba >= settings.decision_threshold else "benign"
    request.app.state.counter.increment()
    logger.info("prediction label=%s malignant_probability=%.4f", label, proba)
    return PredictResponse(label=label, malignant_probability=round(proba, 4),
                           threshold=settings.decision_threshold,
                           model_version=settings.model_version)


@app.get("/model/info")
def model_info(request: Request):
    card = request.app.state.card
    return {
        "model_version": settings.model_version,
        "model_family": card.get("model_family"),
        "sklearn_version": sklearn.__version__,
        "trained_with_sklearn_version": card.get("sklearn_version"),
        "git_commit": settings.git_sha,
        "decision_threshold": settings.decision_threshold,
        "feature_count": len(FEATURE_NAMES),
        "test_metrics": card.get("test_metrics"),
    }


@app.get("/stats")
def stats(request: Request):
    counter = request.app.state.counter
    return {"predictions_served": counter.value(), "redis_enabled": counter.enabled}
