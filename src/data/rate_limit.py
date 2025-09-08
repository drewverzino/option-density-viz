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
import logging
import random
from typing import Any, Awaitable, Callable

logger = logging.getLogger("data.rate_limit")


class AsyncRateLimiter:
    """
    Enforce a maximum number of in-flight operations.

    This is intentionally minimal; for more complex needs you can layer buckets or
    separate limiters per API path.
    """

    def __init__(self, max_concurrent: int = 5):
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        logger.debug(
            f"AsyncRateLimiter initialized with max_concurrent={max_concurrent}"
        )

    async def run(self, coro_func: Callable[[], Awaitable[Any]]) -> Any:
        """Run 'coro_func' under the concurrency gate."""
        logger.debug(
            f"Acquiring semaphore ({self._sem._value}/{self.max_concurrent} available)"
        )
        async with self._sem:
            logger.debug("Semaphore acquired, executing function")
            try:
                result = await coro_func()
                logger.debug("Function completed successfully")
                return result
            except Exception as e:
                logger.debug(f"Function failed: {e}")
                raise


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
    logger.debug(
        f"Starting retry_with_backoff: max_retries={retries}, "
        f"base_delay={base_delay}, max_delay={max_delay}"
    )

    attempt = 0
    delay = base_delay
    last_exception = None

    while True:
        try:
            if attempt == 0:
                logger.debug("Initial attempt")
            else:
                logger.debug(f"Retry attempt {attempt}/{retries}")

            result = await func()
            if attempt > 0:
                logger.info(f"Function succeeded on attempt {attempt + 1}")
            return result

        except retry_on as e:
            last_exception = e
            attempt += 1

            if attempt > retries:
                logger.error(f"All {retries + 1} attempts failed, giving up")
                raise e

            # Jitter factor in [1 - jitter, 1 + jitter]
            jitter_factor = 1.0 + (random.random() * 2 - 1) * jitter
            sleep_time = min(max_delay, delay * jitter_factor)

            logger.warning(
                f"Attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s..."
            )
            await asyncio.sleep(sleep_time)
            delay = min(max_delay, delay * 2)
