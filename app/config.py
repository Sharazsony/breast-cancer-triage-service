"""Service configuration, read only from environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str
    decision_threshold: float
    log_level: str
    redis_host: str | None
    redis_port: int
    model_version: str
    git_sha: str


def get_settings() -> Settings:
    return Settings(
        model_path=os.getenv("MODEL_PATH", "artifacts/model.joblib"),
        decision_threshold=float(os.getenv("DECISION_THRESHOLD", "0.5")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        redis_host=os.getenv("REDIS_HOST") or None,
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        model_version=os.getenv("MODEL_VERSION", "dev"),
        git_sha=os.getenv("GIT_SHA", "unknown"),
    )
