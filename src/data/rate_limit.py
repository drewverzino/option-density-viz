# src/data/rate_limit.py
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable, Optional


class AsyncRateLimiter:
    """
    Simple concurrency gate for polite API usage.
    Use one limiter per resource (e.g., OKX tickers).
    """

    def __init__(self, max_concurrent: int = 5):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def run(self, coro_func: Callable[[], Awaitable[Any]]) -> Any:
        async with self._sem:
            return await coro_func()


async def retry_with_backoff(
    func: Callable[[], Awaitable[Any]],
    *,
    retries: int = 5,
    base_delay: float = 0.25,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.25,
) -> Any:
    """
    Retry an async operation with exponential backoff and jitter.
    """
    attempt = 0
    delay = base_delay
    while True:
        try:
            return await func()
        except retry_on as e:
            attempt += 1
            if attempt > retries:
                raise
            # jitter in [ -jitter, +jitter ] * delay
            jitter_factor = 1.0 + (random.random() * 2 - 1) * jitter
            await asyncio.sleep(min(max_delay, delay * jitter_factor))
            delay = min(max_delay, delay * 2)
