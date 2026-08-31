"""Caching utilities for MCP servers."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from personalos.config import settings

logger = logging.getLogger(__name__)

# Try to import Redis for caching
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheKey:
    """Generate cache keys for tool results."""

    @staticmethod
    def from_params(tool_name: str, **kwargs) -> str:
        """Generate cache key from tool name and parameters."""
        param_str = json.dumps(kwargs, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"mcp:{tool_name}:{param_hash}"


class Cache:
    """Cache interface for MCP results."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        raise NotImplementedError

    async def delete(self, key: str):
        """Delete value from cache."""
        raise NotImplementedError


class RedisCache(Cache):
    """Redis-backed cache."""

    def __init__(self, url: str = None):
        """Initialize Redis cache."""
        url = url or settings.redis_url
        self.redis_client = redis.from_url(url)
        logger.info(f"Initialized Redis cache: {url}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get failed for '{key}': {str(e)}")
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Cache set failed for '{key}': {str(e)}")

    async def delete(self, key: str):
        """Delete value from cache."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete failed for '{key}': {str(e)}")


class InMemoryCache(Cache):
    """Simple in-memory cache."""

    def __init__(self):
        """Initialize in-memory cache."""
        self._cache: dict[str, tuple[Any, datetime]] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                del self._cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = (value, expiry)

    async def delete(self, key: str):
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]


def get_cache() -> Cache:
    """Get cache instance based on configuration."""
    if REDIS_AVAILABLE:
        try:
            return RedisCache(settings.redis_url)
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}, falling back to in-memory")
    return InMemoryCache()


async def cached(cache: Optional[Cache] = None, ttl: int = 3600) -> Callable:
    """Decorator for caching tool results."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            c = cache or get_cache()
            cache_key = CacheKey.from_params(func.__name__, **kwargs)

            # Try to get from cache
            cached_value = await c.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for '{func.__name__}'")
                return cached_value

            # Execute function
            result = await func(*args, **kwargs) if hasattr(func, "__await__") else func(
                *args, **kwargs
            )

            # Store in cache
            await c.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator
