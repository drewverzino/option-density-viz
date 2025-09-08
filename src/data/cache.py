"""
A tiny TTL cache with:
- fast in-memory dictionary for hot items, and
- SQLite persistence for warm restarts.

Use cases:
- Caching 'public/instruments' lists (change slowly) or ticker payloads (short TTL).
- Avoids hammering APIs; keeps the app responsive for repeated queries.

Security note:
- Values are pickled for flexibility. Only cache **trusted** objects created by
  your own process; never unpickle untrusted data.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("data.cache")


@dataclass
class CacheConfig:
    path: Path = Path(".cache/sqlite_cache.db")  # single-file sqlite DB
    ensure_dirs: bool = True  # create parent folder if missing


class KVCache:
    """
    Key-value cache with TTL semantics.

    Strategy:
    - Check memory first (O(1)).
    - If miss/expired, check disk (SQLite).
    - If miss, await the provided 'fetcher()' coroutine to produce a fresh value,
      then write-through to memory and disk.

    Concurrency:
    - A single asyncio.Lock protects disk I/O and write-through to keep invariants sane.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        if self.config.ensure_dirs:
            self.config.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = asyncio.Lock()
        # In-memory store: key -> (expires_at, pickled_value)
        self._mem: dict[str, tuple[float, bytes]] = {}

        logger.debug(
            f"Initializing KVCache with SQLite at: {self.config.path}"
        )
        self._init_db()

    # ------------ public API ------------

    async def get_cached(
        self,
        key: str,
        fetcher: Callable[[], Awaitable[Any]],
        ttl_seconds: float,
    ) -> Any:
        """
        Return a cached value if present and fresh; otherwise await fetcher() and cache it.

        This function is safe to call concurrently: only one coroutine will fetch+write,
        others will see the updated value.
        """
        now = time.time()
        logger.debug(
            f"Cache lookup for key: {key[:50]}{'...' if len(key) > 50 else ''}"
        )

        # Fast path: in-memory hit
        hit = self._mem.get(key)
        if hit and hit[0] > now:
            logger.debug(
                f"Cache HIT (memory): {key[:50]}{'...' if len(key) > 50 else ''}"
            )
            return pickle.loads(hit[1])

        async with self._lock:
            # Double-check after acquiring the lock (another task may have filled it)
            hit = self._mem.get(key)
            if hit and hit[0] > now:
                logger.debug(
                    f"Cache HIT (memory, after lock): {key[:50]}{'...' if len(key) > 50 else ''}"
                )
                return pickle.loads(hit[1])

            # Try disk
            obj = self._get_disk(key, now)
            if obj is not None:
                logger.debug(
                    f"Cache HIT (disk): {key[:50]}{'...' if len(key) > 50 else ''}"
                )
                # Promote to memory for the next fast read
                self._mem[key] = (now + ttl_seconds, pickle.dumps(obj))
                return obj

            # Miss → fetch fresh data
            logger.debug(
                f"Cache MISS, fetching: {key[:50]}{'...' if len(key) > 50 else ''}"
            )
            start_time = time.time()
            obj = await fetcher()
            fetch_time = time.time() - start_time
            logger.debug(
                f"Fetched in {fetch_time:.3f}s, caching with TTL {ttl_seconds}s"
            )

            self.set(key, obj, ttl_seconds)
            return obj

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Write-through to memory and disk with a fresh TTL."""
        logger.debug(
            f"Caching key: {key[:50]}{'...' if len(key) > 50 else ''} "
            f"(TTL: {ttl_seconds}s)"
        )
        expires_at = time.time() + ttl_seconds
        p = pickle.dumps(value)
        self._mem[key] = (expires_at, p)
        self._set_disk(key, p, expires_at)

    def clear_memory(self) -> None:
        """Drop the in-memory layer; useful in long-lived processes before big tasks."""
        mem_count = len(self._mem)
        self._mem.clear()
        logger.info(f"Cleared {mem_count} items from memory cache")

    def vacuum_disk(self) -> None:
        """Remove expired rows from the SQLite file."""
        logger.debug("Vacuuming expired entries from disk cache")
        with sqlite3.connect(self.config.path) as con:
            cursor = con.execute(
                "SELECT COUNT(*) FROM kv WHERE expires_at < ?;", (time.time(),)
            )
            expired_count = cursor.fetchone()[0]
            con.execute("DELETE FROM kv WHERE expires_at < ?;", (time.time(),))
            con.commit()
        logger.debug(
            f"Removed {expired_count} expired entries from disk cache"
        )

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
        logger.debug("SQLite cache database initialized")

    def _get_disk(self, key: str, now: float) -> Optional[Any]:
        """Return a value from disk if fresh; otherwise delete and return None."""
        with sqlite3.connect(self.config.path) as con:
            cur = con.execute(
                "SELECT v, expires_at FROM kv WHERE k = ?;", (key,)
            )
            row = cur.fetchone()
            if not row:
                return None
            v_blob, expires_at = row
            if expires_at <= now:
                # Expired on disk: best-effort cleanup
                logger.debug(
                    f"Removing expired key from disk: {key[:50]}{'...' if len(key) > 50 else ''}"
                )
                con.execute("DELETE FROM kv WHERE k = ?;", (key,))
                con.commit()
                return None
            return pickle.loads(v_blob)

    def _set_disk(
        self, key: str, pickled_value: bytes, expires_at: float
    ) -> None:
        with sqlite3.connect(self.config.path) as con:
            con.execute(
                "REPLACE INTO kv (k, v, expires_at) VALUES (?, ?, ?);",
                (key, pickled_value, expires_at),
            )
            con.commit()
