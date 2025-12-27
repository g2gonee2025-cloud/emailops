
import redis.asyncio as redis
from fastapi import Request

async def get_redis(request: Request) -> redis.Redis:
    """
    FastAPI dependency to get the Redis client from application state.
    The client is managed by the application's lifespan manager.
    """
    return request.app.state.redis
