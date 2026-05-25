"""Session state helpers and configuration hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date
from typing import Any

SESSION_CONFIG = "mvo.config"
SESSION_LAST_HASH = "mvo.last_hash"
SESSION_RESULTS = "mvo.results"
SESSION_PRICES = "mvo.prices"
SESSION_RETURNS = "mvo.returns"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def config_hash(cfg: Any) -> str:
    """Return a stable 24-character BLAKE2b hex digest of the config."""
    if hasattr(cfg, "__dataclass_fields__"):
        payload = asdict(cfg)
    elif isinstance(cfg, dict):
        payload = cfg
    else:
        payload = {"value": str(cfg)}
    blob = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=12).hexdigest()
