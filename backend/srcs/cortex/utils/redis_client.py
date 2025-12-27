"""
Redis client utility.
"""
from __future__ import annotations

from functools import lru_cache

import redis
from cortex.config.loader import get_config


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """
    Returns a Redis client instance.

    The client is cached using lru_cache to ensure only one instance
    and connection pool is created per process.
    """
    config = get_config()
    # from_url automatically manages a connection pool
    return redis.from_url(config.redis.url)
