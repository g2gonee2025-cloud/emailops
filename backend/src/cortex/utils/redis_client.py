"""
Redis client utility.
"""
from __future__ import annotations

import redis
from cortex.config.loader import get_config


def get_redis_client() -> redis.Redis:
    """
    Returns a Redis client instance.
    """
    config = get_config()
    return redis.from_url(config.redis.url)
