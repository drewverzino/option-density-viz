"""
Polite rate-limiting and retry utilities for async code.

Why this file:
- Public APIs will throttle or rate-limit. We want to be a "good citizen" and
  also have predictable behavior when the network flaps.

Two primitives:
1) AsyncRateLimiter: a simple semaphore-based concurrency gate.
2) retry_with_backoff: exponential backoff with jitter for transient failures.

Usage examples:
    limiter = AsyncRateLimiter(max_concurrent=6)
    result = await limiter.run(lambda: client.get(...))

    resp = await retry_with_backoff(lambda: client.get(url), retries=4)
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable


class AsyncRateLimiter:
    """
    Enforce a maximum number of in-flight operations.

    This is intentionally minimal; for more complex needs you can layer buckets or
    separate limiters per API path.
    """

    def __init__(self, max_concurrent: int = 5):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def run(self, coro_func: Callable[[], Awaitable[Any]]) -> Any:
        """Run 'coro_func' under the concurrency gate."""
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
    Retry an async operation with exponential backoff and multiplicative jitter.

    Parameters:
      retries: max number of *retries* (total attempts = retries + 1)
      base_delay: initial delay before the first retry
      max_delay: clamp for delay growth
      retry_on: tuple of exception types that should trigger a retry
      jitter: random factor in [-jitter, +jitter] applied to the delay

    Returns:
      The result of func() if it eventually succeeds.

    Raises:
      The last exception if we exhaust retries.
    """
    attempt = 0
    delay = base_delay
    while True:
        try:
            return await func()
        except retry_on:
            attempt += 1
            if attempt > retries:
                raise
            # Jitter factor in [1 - jitter, 1 + jitter]
            jitter_factor = 1.0 + (random.random() * 2 - 1) * jitter
            await asyncio.sleep(min(max_delay, delay * jitter_factor))
            delay = min(max_delay, delay * 2)
