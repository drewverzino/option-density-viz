# src/data/cache.py
from __future__ import annotations

import asyncio
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


@dataclass
class CacheConfig:
    path: Path = Path(".cache/sqlite_cache.db")
    ensure_dirs: bool = True


class KVCache:
    """
    Tiny TTL cache with:
      - fast in-memory dict
      - persistent SQLite backing store
    Values are pickled; use for API responses or normalized dicts.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        if self.config.ensure_dirs:
            self.config.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._mem: dict[str, tuple[float, bytes]] = {}  # key -> (expires_at, pickled)
        self._init_db()

    # ------------ public API ------------

    async def get_cached(
        self,
        key: str,
        fetcher: Callable[[], Awaitable[Any]],
        ttl_seconds: float,
    ) -> Any:
        """
        Get value from cache; if expired/missing, await fetcher(), store and return.
        """
        now = time.time()
        # Fast path: memory
        hit = self._mem.get(key)
        if hit and hit[0] > now:
            return pickle.loads(hit[1])

        async with self._lock:
            # Re-check after acquiring lock
            hit = self._mem.get(key)
            if hit and hit[0] > now:
                return pickle.loads(hit[1])

            # Try disk
            obj = self._get_disk(key, now)
            if obj is not None:
                # Promote to memory
                self._mem[key] = (now + ttl_seconds, pickle.dumps(obj))
                return obj

            # Miss → fetch
            obj = await fetcher()
            self.set(key, obj, ttl_seconds)
            return obj

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        expires_at = time.time() + ttl_seconds
        p = pickle.dumps(value)
        self._mem[key] = (expires_at, p)
        self._set_disk(key, p, expires_at)

    def clear_memory(self) -> None:
        self._mem.clear()

    def vacuum_disk(self) -> None:
        with sqlite3.connect(self.config.path) as con:
            con.execute("DELETE FROM kv WHERE expires_at < ?;", (time.time(),))
            con.commit()

    # ------------ internals ------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.config.path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                  k TEXT PRIMARY KEY,
                  v BLOB NOT NULL,
                  expires_at REAL NOT NULL
                );
                """
            )
            con.commit()

    def _get_disk(self, key: str, now: float) -> Optional[Any]:
        with sqlite3.connect(self.config.path) as con:
            cur = con.execute("SELECT v, expires_at FROM kv WHERE k = ?;", (key,))
            row = cur.fetchone()
            if not row:
                return None
            v_blob, expires_at = row
            if expires_at <= now:
                # expired on disk: best-effort delete
                con.execute("DELETE FROM kv WHERE k = ?;", (key,))
                con.commit()
                return None
            return pickle.loads(v_blob)

    def _set_disk(self, key: str, pickled_value: bytes, expires_at: float) -> None:
        with sqlite3.connect(self.config.path) as con:
            con.execute(
                "REPLACE INTO kv (k, v, expires_at) VALUES (?, ?, ?);",
                (key, pickled_value, expires_at),
            )
            con.commit()
