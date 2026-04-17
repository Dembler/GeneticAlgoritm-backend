from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from app.domain.models import RouteResponse


class RouteCacheRepository(Protocol):
    def get(self, key: str) -> RouteResponse | None:
        raise NotImplementedError

    def set(self, key: str, value: RouteResponse) -> None:
        raise NotImplementedError


@dataclass
class _CacheEntry:
    expires_at: float
    value: RouteResponse


class InMemoryRouteCacheRepository(RouteCacheRepository):
    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl_seconds = ttl_seconds
        self._storage: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> RouteResponse | None:
        entry = self._storage.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.time():
            self._storage.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: RouteResponse) -> None:
        self._storage[key] = _CacheEntry(expires_at=time.time() + self._ttl_seconds, value=value)