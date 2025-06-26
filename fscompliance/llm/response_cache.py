"""LLM response caching system for cost optimization and performance improvement."""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, OrderedDict

from pydantic import BaseModel, Field

from .providers import LLMRequest, LLMResponse, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategy types."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns
    SEMANTIC = "semantic"          # Semantic similarity-based


class CacheLevel(str, Enum):
    """Cache levels for different granularity."""
    EXACT = "exact"                # Exact request match
    SEMANTIC = "semantic"          # Semantically similar requests
    PARAMETRIC = "parametric"      # Similar parameters, different content
    RESPONSE_FRAGMENTS = "response_fragments"  # Partial response caching


@dataclass
class CacheKey:
    """Cache key with multiple components for flexible matching."""
    
    # Request fingerprint
    request_hash: str
    
    # Semantic components
    task_type: str
    complexity: str
    intent_hash: str
    
    # Model components
    model_id: str
    provider: str
    
    # Parameters
    temperature: float
    max_tokens: int
    
    def to_string(self, level: CacheLevel = CacheLevel.EXACT) -> str:
        """Convert to string representation for given cache level."""
        
        if level == CacheLevel.EXACT:
            return f"{self.request_hash}:{self.model_id}:{self.provider}"
        
        elif level == CacheLevel.SEMANTIC:
            return f"{self.intent_hash}:{self.task_type}:{self.model_id}"
        
        elif level == CacheLevel.PARAMETRIC:
            return f"{self.task_type}:{self.complexity}:{self.model_id}:{self.temperature}:{self.max_tokens}"
        
        else:  # RESPONSE_FRAGMENTS
            return f"{self.task_type}:{self.model_id}"


@dataclass
class CacheEntry:
    """Cache entry with metadata and content."""
    
    # Cache key and content
    key: CacheKey
    request: Union[LLMRequest, EmbeddingRequest]
    response: Union[LLMResponse, EmbeddingResponse]
    
    # Metadata
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    
    # Size and cost
    size_bytes: int
    original_cost: float
    cost_saved: float
    
    # Quality metrics
    semantic_similarity: float = 0.0
    response_quality_score: float = 0.0
    user_feedback_score: float = 0.0
    
    # Cache level
    cache_level: CacheLevel = CacheLevel.EXACT
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # Never expires
        
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def calculate_value_score(self) -> float:
        """Calculate value score for cache eviction decisions."""
        
        # Factors: access frequency, recency, cost savings, quality
        frequency_score = min(self.access_count / 10.0, 1.0)
        recency_score = max(0.0, 1.0 - (self.get_age_seconds() / (7 * 24 * 3600)))  # 7 days
        cost_score = min(self.cost_saved / 1.0, 1.0)  # Normalize to $1
        quality_score = (self.response_quality_score + self.user_feedback_score) / 2.0
        
        return (frequency_score * 0.3 + 
                recency_score * 0.25 + 
                cost_score * 0.25 + 
                quality_score * 0.2)


class CacheStatistics(BaseModel):
    """Cache performance statistics."""
    
    # Hit/miss statistics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # By cache level
    exact_hits: int = 0
    semantic_hits: int = 0
    parametric_hits: int = 0
    fragment_hits: int = 0
    
    # Cost and performance
    total_cost_saved: float = 0.0
    total_time_saved_ms: float = 0.0
    average_response_time_ms: float = 0.0
    
    # Storage
    entries_count: int = 0
    total_size_bytes: int = 0
    evictions_count: int = 0
    
    # Quality
    average_semantic_similarity: float = 0.0
    average_response_quality: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "hit_rate": self.hit_rate,
            "total_requests": self.total_requests,
            "cost_saved": self.total_cost_saved,
            "time_saved_hours": self.total_time_saved_ms / (1000 * 3600),
            "entries_count": self.entries_count,
            "storage_mb": self.total_size_bytes / (1024 * 1024)
        }


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get number of cache entries."""
        pass


class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                self.cache[key] = entry
                return entry
        return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        async with self._lock:
            # Remove oldest entries if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        async with self._lock:
            self.cache.clear()
            return True
    
    async def keys(self) -> List[str]:
        async with self._lock:
            return list(self.cache.keys())
    
    async def size(self) -> int:
        return len(self.cache)


class LLMResponseCache:
    """Main LLM response cache with multiple cache levels and strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Cache configuration
        self.enabled_levels = set(config.get("enabled_levels", [
            CacheLevel.EXACT,
            CacheLevel.SEMANTIC,
            CacheLevel.PARAMETRIC
        ]))
        
        self.cache_strategy = CacheStrategy(config.get("strategy", CacheStrategy.ADAPTIVE))
        self.max_cache_size = config.get("max_cache_size", 1000)
        self.default_ttl_seconds = config.get("default_ttl_seconds", 3600)  # 1 hour
        
        # Semantic similarity threshold for cache hits
        self.semantic_threshold = config.get("semantic_threshold", 0.85)
        
        # Cost thresholds for caching decisions
        self.min_cost_to_cache = config.get("min_cost_to_cache", 0.001)  # $0.001
        self.max_cache_value = config.get("max_cache_value", 10.0)  # $10
        
        # Cache backend
        backend_type = config.get("backend", "memory")
        if backend_type == "memory":
            self.backend = MemoryCacheBackend(self.max_cache_size)
        else:
            raise ValueError(f"Unsupported cache backend: {backend_type}")
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Semantic similarity function (placeholder)
        self.similarity_func = self._default_similarity_function
    
    async def get_response(
        self,
        request: Union[LLMRequest, EmbeddingRequest],
        model_id: str,
        provider: str
    ) -> Optional[Tuple[Union[LLMResponse, EmbeddingResponse], CacheLevel]]:
        """Get cached response if available."""
        
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(request, model_id, provider)
            
            # Try different cache levels
            for level in self.enabled_levels:
                key_str = cache_key.to_string(level)
                
                if level == CacheLevel.EXACT:
                    entry = await self._get_exact_match(key_str)
                elif level == CacheLevel.SEMANTIC:
                    entry = await self._get_semantic_match(cache_key, request)
                elif level == CacheLevel.PARAMETRIC:
                    entry = await self._get_parametric_match(cache_key, request)
                else:  # RESPONSE_FRAGMENTS
                    entry = await self._get_fragment_match(cache_key, request)
                
                if entry:
                    # Update access statistics
                    entry.update_access()
                    await self.backend.set(key_str, entry)
                    
                    # Update statistics
                    self.stats.cache_hits += 1
                    self.stats.total_cost_saved += entry.original_cost
                    
                    response_time = (time.time() - start_time) * 1000
                    self.stats.total_time_saved_ms += max(0, entry.response.latency_ms - response_time)
                    
                    # Update level-specific stats
                    if level == CacheLevel.EXACT:
                        self.stats.exact_hits += 1
                    elif level == CacheLevel.SEMANTIC:
                        self.stats.semantic_hits += 1
                    elif level == CacheLevel.PARAMETRIC:
                        self.stats.parametric_hits += 1
                    else:
                        self.stats.fragment_hits += 1
                    
                    logger.info(f"Cache hit at {level.value} level for {model_id}")
                    
                    return entry.response, level
            
            # No cache hit
            self.stats.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            self.stats.cache_misses += 1
            return None
    
    async def store_response(
        self,
        request: Union[LLMRequest, EmbeddingRequest],
        response: Union[LLMResponse, EmbeddingResponse],
        model_id: str,
        provider: str
    ) -> bool:
        """Store response in cache."""
        
        try:
            # Check if worth caching
            if not self._should_cache_response(request, response):
                return False
            
            # Generate cache key
            cache_key = self._generate_cache_key(request, model_id, provider)
            
            # Calculate cache entry details
            size_bytes = self._estimate_entry_size(request, response)
            ttl_seconds = self._calculate_ttl(request, response)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                request=request,
                response=response,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                original_cost=response.estimated_cost,
                cost_saved=0.0,
                cache_level=CacheLevel.EXACT
            )
            
            # Store with exact key
            exact_key = cache_key.to_string(CacheLevel.EXACT)
            success = await self.backend.set(exact_key, entry)
            
            if success:
                # Update statistics
                self.stats.entries_count += 1
                self.stats.total_size_bytes += size_bytes
                
                # Trigger eviction if needed
                await self._evict_if_needed()
                
                logger.debug(f"Cached response for {model_id} (key: {exact_key[:16]}...)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    async def invalidate_cache(
        self,
        request: Optional[Union[LLMRequest, EmbeddingRequest]] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None
    ):
        """Invalidate cache entries matching criteria."""
        
        if request is None and model_id is None and provider is None:
            # Clear all cache
            await self.backend.clear()
            self.stats = CacheStatistics()
            return
        
        # Get all keys and filter
        all_keys = await self.backend.keys()
        keys_to_delete = []
        
        for key in all_keys:
            entry = await self.backend.get(key)
            if entry:
                should_delete = False
                
                if model_id and entry.key.model_id == model_id:
                    should_delete = True
                
                if provider and entry.key.provider == provider:
                    should_delete = True
                
                if request and self._requests_match(request, entry.request):
                    should_delete = True
                
                if should_delete:
                    keys_to_delete.append(key)
        
        # Delete matching entries
        for key in keys_to_delete:
            await self.backend.delete(key)
            self.stats.entries_count = max(0, self.stats.entries_count - 1)
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        return self.stats
    
    async def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        
        all_keys = await self.backend.keys()
        expired_keys = []
        
        for key in all_keys:
            entry = await self.backend.get(key)
            if entry and entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.backend.delete(key)
            self.stats.entries_count = max(0, self.stats.entries_count - 1)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def optimize_cache(self):
        """Optimize cache by removing low-value entries."""
        
        if await self.backend.size() <= self.max_cache_size * 0.8:
            return  # No optimization needed
        
        all_keys = await self.backend.keys()
        entries_with_scores = []
        
        # Calculate value scores for all entries
        for key in all_keys:
            entry = await self.backend.get(key)
            if entry:
                value_score = entry.calculate_value_score()
                entries_with_scores.append((key, entry, value_score))
        
        # Sort by value score (lowest first for removal)
        entries_with_scores.sort(key=lambda x: x[2])
        
        # Remove lowest value entries
        target_size = int(self.max_cache_size * 0.7)
        entries_to_remove = len(entries_with_scores) - target_size
        
        for i in range(entries_to_remove):
            key, entry, _ = entries_with_scores[i]
            await self.backend.delete(key)
            self.stats.entries_count = max(0, self.stats.entries_count - 1)
            self.stats.evictions_count += 1
        
        if entries_to_remove > 0:
            logger.info(f"Optimized cache by removing {entries_to_remove} low-value entries")
    
    def _generate_cache_key(
        self,
        request: Union[LLMRequest, EmbeddingRequest],
        model_id: str,
        provider: str
    ) -> CacheKey:
        """Generate cache key for request."""
        
        # Request hash (exact content)
        request_content = self._serialize_request(request)
        request_hash = hashlib.sha256(request_content.encode()).hexdigest()
        
        # Intent hash (semantic content)
        if isinstance(request, LLMRequest):
            intent_content = f"{request.prompt}|{request.system_prompt or ''}"
            task_type = request.task_type
            complexity = request.complexity.value
            temperature = request.temperature
            max_tokens = request.max_tokens or 512
        else:  # EmbeddingRequest
            intent_content = "|".join(request.texts)
            task_type = request.task_type
            complexity = "simple"
            temperature = 0.0
            max_tokens = 0
        
        intent_hash = hashlib.md5(intent_content.encode()).hexdigest()
        
        return CacheKey(
            request_hash=request_hash,
            task_type=task_type,
            complexity=complexity,
            intent_hash=intent_hash,
            model_id=model_id,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _serialize_request(self, request: Union[LLMRequest, EmbeddingRequest]) -> str:
        """Serialize request for exact matching."""
        
        if isinstance(request, LLMRequest):
            return json.dumps({
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop_sequences": request.stop_sequences,
                "task_type": request.task_type,
                "complexity": request.complexity.value
            }, sort_keys=True)
        else:  # EmbeddingRequest
            return json.dumps({
                "texts": request.texts,
                "model_preference": request.model_preference,
                "task_type": request.task_type
            }, sort_keys=True)
    
    async def _get_exact_match(self, key: str) -> Optional[CacheEntry]:
        """Get exact cache match."""
        
        entry = await self.backend.get(key)
        if entry and not entry.is_expired():
            return entry
        
        if entry and entry.is_expired():
            await self.backend.delete(key)
            self.stats.entries_count = max(0, self.stats.entries_count - 1)
        
        return None
    
    async def _get_semantic_match(
        self,
        cache_key: CacheKey,
        request: Union[LLMRequest, EmbeddingRequest]
    ) -> Optional[CacheEntry]:
        """Get semantic cache match."""
        
        # Get all entries with same task type and model
        all_keys = await self.backend.keys()
        best_match = None
        best_similarity = 0.0
        
        for key in all_keys:
            entry = await self.backend.get(key)
            if not entry or entry.is_expired():
                continue
            
            # Check if same task type and model
            if (entry.key.task_type == cache_key.task_type and
                entry.key.model_id == cache_key.model_id):
                
                # Calculate semantic similarity
                similarity = self._calculate_semantic_similarity(request, entry.request)
                
                if similarity > best_similarity and similarity >= self.semantic_threshold:
                    best_similarity = similarity
                    best_match = entry
        
        if best_match:
            best_match.semantic_similarity = best_similarity
        
        return best_match
    
    async def _get_parametric_match(
        self,
        cache_key: CacheKey,
        request: Union[LLMRequest, EmbeddingRequest]
    ) -> Optional[CacheEntry]:
        """Get parametric cache match (similar parameters, different content)."""
        
        # Look for entries with same parameters but different content
        param_key = cache_key.to_string(CacheLevel.PARAMETRIC)
        entry = await self.backend.get(param_key)
        
        if entry and not entry.is_expired():
            # Additional validation for parametric match
            if self._parameters_compatible(request, entry.request):
                return entry
        
        return None
    
    async def _get_fragment_match(
        self,
        cache_key: CacheKey,
        request: Union[LLMRequest, EmbeddingRequest]
    ) -> Optional[CacheEntry]:
        """Get response fragment match."""
        
        # For now, return None (fragment caching is complex)
        # Future implementation could cache common response patterns
        return None
    
    def _calculate_semantic_similarity(
        self,
        request1: Union[LLMRequest, EmbeddingRequest],
        request2: Union[LLMRequest, EmbeddingRequest]
    ) -> float:
        """Calculate semantic similarity between requests."""
        
        if type(request1) != type(request2):
            return 0.0
        
        if isinstance(request1, LLMRequest):
            text1 = f"{request1.prompt} {request1.system_prompt or ''}"
            text2 = f"{request2.prompt} {request2.system_prompt or ''}"
        else:
            text1 = " ".join(request1.texts)
            text2 = " ".join(request2.texts)
        
        return self.similarity_func(text1, text2)
    
    def _default_similarity_function(self, text1: str, text2: str) -> float:
        """Default semantic similarity function (simple token overlap)."""
        
        # Simple token-based similarity (would be replaced with embedding similarity)
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _should_cache_response(
        self,
        request: Union[LLMRequest, EmbeddingRequest],
        response: Union[LLMResponse, EmbeddingResponse]
    ) -> bool:
        """Determine if response should be cached."""
        
        # Don't cache if response has errors
        if hasattr(response, 'errors') and response.errors:
            return False
        
        # Don't cache very cheap responses (not worth the overhead)
        if response.estimated_cost < self.min_cost_to_cache:
            return False
        
        # Don't cache extremely expensive responses (may be one-off)
        if response.estimated_cost > self.max_cache_value:
            return False
        
        # Don't cache very long responses (memory concern)
        if hasattr(response, 'text') and len(response.text) > 10000:
            return False
        
        return True
    
    def _calculate_ttl(
        self,
        request: Union[LLMRequest, EmbeddingRequest],
        response: Union[LLMResponse, EmbeddingResponse]
    ) -> int:
        """Calculate TTL for cache entry."""
        
        base_ttl = self.default_ttl_seconds
        
        # Longer TTL for more expensive responses
        if response.estimated_cost > 0.1:
            base_ttl *= 2
        elif response.estimated_cost > 0.01:
            base_ttl *= 1.5
        
        # Longer TTL for high-quality responses
        if hasattr(response, 'confidence_score') and response.confidence_score:
            if response.confidence_score > 0.9:
                base_ttl *= 1.5
        
        # Task-specific TTL
        if isinstance(request, LLMRequest):
            if request.task_type in ["regulatory_interpretation", "requirement_extraction"]:
                base_ttl *= 2  # Regulatory content is stable
        
        return int(base_ttl)
    
    def _estimate_entry_size(
        self,
        request: Union[LLMRequest, EmbeddingRequest],
        response: Union[LLMResponse, EmbeddingResponse]
    ) -> int:
        """Estimate cache entry size in bytes."""
        
        try:
            # Serialize entry to estimate size
            entry_data = {
                "request": request.dict() if hasattr(request, 'dict') else str(request),
                "response": response.dict() if hasattr(response, 'dict') else str(response)
            }
            
            serialized = json.dumps(entry_data)
            return len(serialized.encode('utf-8'))
            
        except Exception:
            # Fallback estimation
            if hasattr(response, 'text'):
                return len(response.text) * 2  # Rough estimate
            else:
                return 1024  # Default 1KB
    
    def _requests_match(
        self,
        request1: Union[LLMRequest, EmbeddingRequest],
        request2: Union[LLMRequest, EmbeddingRequest]
    ) -> bool:
        """Check if two requests match exactly."""
        
        return self._serialize_request(request1) == self._serialize_request(request2)
    
    def _parameters_compatible(
        self,
        request1: Union[LLMRequest, EmbeddingRequest],
        request2: Union[LLMRequest, EmbeddingRequest]
    ) -> bool:
        """Check if request parameters are compatible."""
        
        if type(request1) != type(request2):
            return False
        
        if isinstance(request1, LLMRequest):
            return (
                abs(request1.temperature - request2.temperature) < 0.1 and
                abs((request1.max_tokens or 512) - (request2.max_tokens or 512)) < 100 and
                request1.task_type == request2.task_type
            )
        else:
            return request1.task_type == request2.task_type
    
    async def _evict_if_needed(self):
        """Evict entries if cache is too full."""
        
        current_size = await self.backend.size()
        
        if current_size > self.max_cache_size:
            await self.optimize_cache()


# Factory functions
def create_response_cache(config: Optional[Dict[str, Any]] = None) -> LLMResponseCache:
    """Create LLM response cache."""
    if config is None:
        config = {}
    
    return LLMResponseCache(config)


def create_financial_compliance_cache() -> LLMResponseCache:
    """Create cache optimized for financial compliance workloads."""
    
    config = {
        "enabled_levels": [CacheLevel.EXACT, CacheLevel.SEMANTIC],
        "strategy": CacheStrategy.ADAPTIVE,
        "max_cache_size": 2000,
        "default_ttl_seconds": 7200,  # 2 hours
        "semantic_threshold": 0.90,   # High threshold for financial content
        "min_cost_to_cache": 0.005,   # Cache responses costing > $0.005
        "max_cache_value": 5.0,       # Cache up to $5 responses
        "backend": "memory"
    }
    
    return LLMResponseCache(config)


# Cache decorators and utilities
class CacheAwareLLMProvider:
    """Wrapper that adds caching to any LLM provider."""
    
    def __init__(self, provider, cache: LLMResponseCache):
        self.provider = provider
        self.cache = cache
    
    async def generate_text_with_cache(self, request: LLMRequest) -> LLMResponse:
        """Generate text with caching."""
        
        # Try to get from cache
        cached_result = await self.cache.get_response(
            request,
            self.provider.get_available_models()[0].model_id,
            self.provider.get_provider_name()
        )
        
        if cached_result:
            response, cache_level = cached_result
            logger.info(f"Cache hit at {cache_level.value} level")
            return response
        
        # Generate new response
        response = await self.provider.generate_text(request)
        
        # Store in cache
        await self.cache.store_response(
            request,
            response,
            self.provider.get_available_models()[0].model_id,
            self.provider.get_provider_name()
        )
        
        return response
    
    async def generate_embeddings_with_cache(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings with caching."""
        
        # Try to get from cache
        cached_result = await self.cache.get_response(
            request,
            "embedding_model",
            self.provider.get_provider_name()
        )
        
        if cached_result:
            response, cache_level = cached_result
            logger.info(f"Cache hit at {cache_level.value} level")
            return response
        
        # Generate new response
        response = await self.provider.generate_embeddings(request)
        
        # Store in cache
        await self.cache.store_response(
            request,
            response,
            "embedding_model",
            self.provider.get_provider_name()
        )
        
        return response


def cache_aware_provider(provider, cache_config: Optional[Dict[str, Any]] = None):
    """Decorator to make any provider cache-aware."""
    cache = create_response_cache(cache_config)
    return CacheAwareLLMProvider(provider, cache)