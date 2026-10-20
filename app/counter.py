"""Prediction counter backed by Redis.

Counting is enabled only when REDIS_HOST is set. Redis failures are logged and
never break /predict.
"""
import logging

import redis

logger = logging.getLogger("triage.counter")
KEY = "predictions_served"


class PredictionCounter:
    def __init__(self, host: str | None, port: int):
        self.enabled = bool(host)
        self._client = (
            redis.Redis(host=host, port=port, socket_connect_timeout=0.5, socket_timeout=0.5)
            if self.enabled else None
        )
        if self.enabled:
            logger.info("Request counting enabled (redis %s:%s)", host, port)
        else:
            logger.info("REDIS_HOST not set - request counting disabled")

    def increment(self) -> int | None:
        if not self.enabled:
            return None
        try:
            return int(self._client.incr(KEY))
        except redis.RedisError as exc:
            logger.warning("Redis unavailable, prediction not counted: %s", exc)
            return None

    def value(self) -> int | None:
        if not self.enabled:
            return None
        try:
            return int(self._client.get(KEY) or 0)
        except redis.RedisError as exc:
            logger.warning("Redis unavailable, cannot read counter: %s", exc)
            return None
