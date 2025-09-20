"""Rate limiting middleware and utilities."""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from fastapi import Request, HTTPException
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import redis.asyncio as redis
from src.core.constants import REDIS_URL


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window_seconds: int
    algorithm: str = "sliding_window"  # "token_bucket" or "sliding_window"


class RateLimiter:
    """Rate limiter implementation with multiple algorithms."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_cache: Dict[str, deque] = defaultdict(deque)
        self.token_buckets: Dict[str, Tuple[float, float]] = {}  # (tokens, last_refill)
    
    async def is_allowed(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed and return rate limit info."""
        
        if self.redis_client:
            return await self._redis_check(key, limit, user_id)
        else:
            return await self._local_check(key, limit, user_id)
    
    async def _redis_check(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit using Redis."""
        
        if limit.algorithm == "sliding_window":
            return await self._redis_sliding_window(key, limit, user_id)
        elif limit.algorithm == "token_bucket":
            return await self._redis_token_bucket(key, limit, user_id)
        else:
            raise ValueError(f"Unsupported algorithm: {limit.algorithm}")
    
    async def _redis_sliding_window(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Sliding window rate limiting using Redis."""
        
        now = time.time()
        window_start = now - limit.window_seconds
        
        # Use Redis sorted set for sliding window
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiration
            pipe.expire(key, limit.window_seconds)
            
            results = await pipe.execute()
        else:
            results = [0, 0]
        current_requests = results[1]
        
        allowed = current_requests < limit.requests
        remaining = max(0, limit.requests - current_requests - 1)
        
        return allowed, {
            "limit": limit.requests,
            "remaining": remaining,
            "reset": int(now + limit.window_seconds)
        }
    
    async def _redis_token_bucket(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Token bucket rate limiting using Redis."""
        
        now = time.time()
        bucket_key = f"{key}:bucket"
        
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            
            # Get current bucket state
            pipe.hmget(bucket_key, ["tokens", "last_refill"])
            
            # Refill tokens
            pipe.eval("""
                local key = KEYS[1]
                local now = tonumber(ARGV[1])
                local window = tonumber(ARGV[2])
                local limit = tonumber(ARGV[3])
                
                local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
                local tokens = tonumber(bucket[1]) or limit
                local last_refill = tonumber(bucket[2]) or now
                
                local elapsed = now - last_refill
                local tokens_to_add = elapsed * limit / window
                tokens = math.min(limit, tokens + tokens_to_add)
                
                if tokens >= 1 then
                    tokens = tokens - 1
                    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                    redis.call('EXPIRE', key, window)
                    return {1, tokens}
                else
                    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                    redis.call('EXPIRE', key, window)
                    return {0, tokens}
                end
            """, 1, bucket_key, str(now), str(limit.window_seconds), str(limit.requests))
            
            results = await pipe.execute()
        else:
            results = [[0, 0], [0, 0]]
        bucket_state = results[0]
        bucket_result = results[1]
        
        allowed = bucket_result[0] == 1
        remaining = int(bucket_result[1])
        
        return allowed, {
            "limit": limit.requests,
            "remaining": remaining,
            "reset": int(now + limit.window_seconds)
        }
    
    async def _local_check(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit using local memory."""
        
        if limit.algorithm == "sliding_window":
            return await self._local_sliding_window(key, limit, user_id)
        elif limit.algorithm == "token_bucket":
            return await self._local_token_bucket(key, limit, user_id)
        else:
            raise ValueError(f"Unsupported algorithm: {limit.algorithm}")
    
    async def _local_sliding_window(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Local sliding window rate limiting."""
        
        now = time.time()
        window_start = now - limit.window_seconds
        
        # Clean old entries
        requests = self.local_cache[key]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        allowed = len(requests) < limit.requests
        
        if allowed:
            requests.append(now)
        
        remaining = max(0, limit.requests - len(requests))
        
        return allowed, {
            "limit": limit.requests,
            "remaining": remaining,
            "reset": int(now + limit.window_seconds)
        }
    
    async def _local_token_bucket(
        self, 
        key: str, 
        limit: RateLimit, 
        user_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """Local token bucket rate limiting."""
        
        now = time.time()
        
        if key not in self.token_buckets:
            self.token_buckets[key] = (limit.requests, now)
        
        tokens, last_refill = self.token_buckets[key]
        
        # Refill tokens
        elapsed = now - last_refill
        tokens_to_add = elapsed * limit.requests / limit.window_seconds
        tokens = min(limit.requests, tokens + tokens_to_add)
        
        # Check if tokens available
        allowed = tokens >= 1
        
        if allowed:
            tokens -= 1
        
        self.token_buckets[key] = (tokens, now)
        
        return allowed, {
            "limit": limit.requests,
            "remaining": int(tokens),
            "reset": int(now + limit.window_seconds)
        }


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI."""
    
    def __init__(
        self, 
        app, 
        redis_client: Optional[redis.Redis] = None,
        default_limits: Optional[Dict[str, RateLimit]] = None
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(redis_client)
        self.default_limits = default_limits or {
            "default": RateLimit(requests=1000, window_seconds=3600),  # 1000/hour
            "observations": RateLimit(requests=100, window_seconds=3600),  # 100/hour
            "detections": RateLimit(requests=200, window_seconds=3600),  # 200/hour
            "workflows": RateLimit(requests=50, window_seconds=3600),  # 50/hour
            "admin": RateLimit(requests=10000, window_seconds=3600),  # 10000/hour
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Determine rate limit key
        key = self._get_rate_limit_key(request)
        
        # Get rate limit configuration
        limit = self._get_rate_limit(request, key)
        
        # Check rate limit
        allowed, rate_info = await self.rate_limiter.is_allowed(key, limit)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["reset"] - int(time.time()))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        
        return response
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key for request."""
        
        # Get user ID from request (if authenticated)
        user_id = getattr(request.state, 'user_id', None)
        
        # Get endpoint category
        path = request.url.path
        if path.startswith('/observations'):
            category = 'observations'
        elif path.startswith('/detections'):
            category = 'detections'
        elif path.startswith('/workflows'):
            category = 'workflows'
        elif path.startswith('/admin'):
            category = 'admin'
        else:
            category = 'default'
        
        # Create key
        if user_id:
            return f"rate_limit:{category}:user:{user_id}"
        else:
            # Use IP address for unauthenticated requests
            client_ip = request.client.host if request.client else "unknown"
            return f"rate_limit:{category}:ip:{client_ip}"
    
    def _get_rate_limit(self, request: Request, key: str) -> RateLimit:
        """Get rate limit configuration for request."""
        
        # Check for user-specific limits (admin users get higher limits)
        user_role = getattr(request.state, 'user_role', None)
        if user_role == 'admin':
            return self.default_limits['admin']
        
        # Check for endpoint-specific limits
        path = request.url.path
        for endpoint, limit in self.default_limits.items():
            if endpoint != 'default' and path.startswith(f'/{endpoint}'):
                return limit
        
        # Default limit
        return self.default_limits['default']


# Rate limit configurations
RATE_LIMITS = {
    "default": RateLimit(requests=1000, window_seconds=3600),  # 1000/hour
    "observations": RateLimit(requests=100, window_seconds=3600),  # 100/hour
    "detections": RateLimit(requests=200, window_seconds=3600),  # 200/hour
    "workflows": RateLimit(requests=50, window_seconds=3600),  # 50/hour
    "admin": RateLimit(requests=10000, window_seconds=3600),  # 10000/hour
}

RATE_LIMIT_HEADERS = {
    "X-RateLimit-Limit": "Request limit per window",
    "X-RateLimit-Remaining": "Remaining requests in window",
    "X-RateLimit-Reset": "Window reset timestamp"
}
