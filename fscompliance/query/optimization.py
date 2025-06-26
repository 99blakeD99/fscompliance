"""Query performance optimization for compliance query processing."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import hashlib

from pydantic import BaseModel, Field

from .processing import ProcessedQuery
from .routing import ClassifiedQuery
from .response_generation import GeneratedResponse
from .ranking import RankedSearchResults

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Types of performance optimization strategies."""
    CACHING = "caching"                   # Cache frequently accessed results
    QUERY_BATCHING = "query_batching"    # Batch similar queries
    PARALLEL_PROCESSING = "parallel_processing"  # Process components in parallel
    RESULT_PREFETCHING = "result_prefetching"   # Prefetch likely needed results
    QUERY_REWRITING = "query_rewriting"   # Rewrite queries for better performance
    INDEX_OPTIMIZATION = "index_optimization"   # Optimize search indices
    LOAD_BALANCING = "load_balancing"     # Balance processing load
    RESOURCE_POOLING = "resource_pooling"  # Pool expensive resources


class CacheType(str, Enum):
    """Types of caches used in the system."""
    QUERY_RESULTS = "query_results"       # Cache complete query results
    SEARCH_RESULTS = "search_results"     # Cache search results
    PROCESSED_QUERIES = "processed_queries"  # Cache processed query objects
    RESPONSE_FRAGMENTS = "response_fragments"  # Cache response components
    RANKING_SCORES = "ranking_scores"     # Cache ranking computations
    CLASSIFICATION_RESULTS = "classification_results"  # Cache query classifications


class PerformanceMetric(str, Enum):
    """Performance metrics tracked by the system."""
    QUERY_PROCESSING_TIME = "query_processing_time"
    SEARCH_EXECUTION_TIME = "search_execution_time"
    RANKING_TIME = "ranking_time"
    RESPONSE_GENERATION_TIME = "response_generation_time"
    TOTAL_RESPONSE_TIME = "total_response_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    THROUGHPUT = "throughput"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_RATE = "error_rate"


@dataclass
class CacheEntry:
    """Entry in the performance cache."""
    
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def is_fresh(self, max_age_seconds: int) -> bool:
        """Check if cache entry is still fresh."""
        return datetime.utcnow() < self.created_at + timedelta(seconds=max_age_seconds)


class PerformanceCache:
    """High-performance cache for query processing components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caches: Dict[CacheType, Dict[str, CacheEntry]] = {
            cache_type: {} for cache_type in CacheType
        }
        self.cache_stats = {
            "hits": defaultdict(int),
            "misses": defaultdict(int),
            "evictions": defaultdict(int),
            "total_size_bytes": defaultdict(int)
        }
        self.max_cache_size = config.get("max_cache_size_mb", 100) * 1024 * 1024  # MB to bytes
        self.default_ttl = config.get("default_ttl_seconds", 3600)  # 1 hour
    
    async def get(self, cache_type: CacheType, key: str) -> Optional[Any]:
        """Get value from cache."""
        
        cache = self.caches[cache_type]
        
        if key in cache:
            entry = cache[key]
            
            # Check if expired
            if entry.is_expired():
                await self._evict_entry(cache_type, key)
                self.cache_stats["misses"][cache_type] += 1
                return None
            
            # Update access statistics
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            self.cache_stats["hits"][cache_type] += 1
            
            return entry.value
        
        self.cache_stats["misses"][cache_type] += 1
        return None
    
    async def put(
        self, 
        cache_type: CacheType, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ):
        """Put value in cache."""
        
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # Calculate approximate size
        size_bytes = self._estimate_size(value)
        
        # Check if we need to evict entries to make space
        await self._ensure_cache_space(cache_type, size_bytes)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=0,
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes
        )
        
        # Store in cache
        self.caches[cache_type][key] = entry
        self.cache_stats["total_size_bytes"][cache_type] += size_bytes
    
    async def invalidate(self, cache_type: CacheType, key: str):
        """Invalidate specific cache entry."""
        
        if key in self.caches[cache_type]:
            await self._evict_entry(cache_type, key)
    
    async def clear(self, cache_type: Optional[CacheType] = None):
        """Clear cache entries."""
        
        if cache_type:
            self.caches[cache_type].clear()
            self.cache_stats["total_size_bytes"][cache_type] = 0
        else:
            for ct in CacheType:
                self.caches[ct].clear()
                self.cache_stats["total_size_bytes"][ct] = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        
        stats = {}
        
        for cache_type in CacheType:
            hits = self.cache_stats["hits"][cache_type]
            misses = self.cache_stats["misses"][cache_type]
            total_requests = hits + misses
            
            hit_rate = hits / total_requests if total_requests > 0 else 0.0
            
            stats[cache_type.value] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "entries": len(self.caches[cache_type]),
                "size_bytes": self.cache_stats["total_size_bytes"][cache_type],
                "evictions": self.cache_stats["evictions"][cache_type]
            }
        
        return stats
    
    async def _ensure_cache_space(self, cache_type: CacheType, required_bytes: int):
        """Ensure sufficient cache space by evicting entries if needed."""
        
        cache = self.caches[cache_type]
        current_size = self.cache_stats["total_size_bytes"][cache_type]
        
        # If adding this entry would exceed cache size, evict entries
        while current_size + required_bytes > self.max_cache_size and cache:
            # Evict least recently used entry
            lru_key = min(cache.keys(), key=lambda k: cache[k].last_accessed)
            await self._evict_entry(cache_type, lru_key)
            current_size = self.cache_stats["total_size_bytes"][cache_type]
    
    async def _evict_entry(self, cache_type: CacheType, key: str):
        """Evict specific cache entry."""
        
        cache = self.caches[cache_type]
        
        if key in cache:
            entry = cache[key]
            del cache[key]
            self.cache_stats["total_size_bytes"][cache_type] -= entry.size_bytes
            self.cache_stats["evictions"][cache_type] += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (dict, list)):
            return len(str(value).encode('utf-8'))
        elif hasattr(value, '__dict__'):
            return len(str(value.__dict__).encode('utf-8'))
        else:
            return len(str(value).encode('utf-8'))


class QueryOptimizer:
    """Query performance optimizer with multiple optimization strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = PerformanceCache(config.get("cache_config", {}))
        self.optimization_stats = {
            "queries_optimized": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "optimization_time_saved_ms": 0.0,
            "parallel_operations": 0,
            "query_rewrites": 0
        }
        self.enabled_strategies = set(config.get("enabled_strategies", [
            OptimizationStrategy.CACHING,
            OptimizationStrategy.PARALLEL_PROCESSING,
            OptimizationStrategy.QUERY_REWRITING
        ]))
    
    async def optimize_query_processing(
        self, 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[ClassifiedQuery], bool]:
        """Optimize query classification and processing."""
        
        start_time = time.time()
        
        # Strategy 1: Check cache for classified query
        if OptimizationStrategy.CACHING in self.enabled_strategies:
            cached_result = await self._get_cached_classification(query, context)
            if cached_result:
                self.optimization_stats["cache_hits"] += 1
                time_saved = (time.time() - start_time) * 1000
                self.optimization_stats["optimization_time_saved_ms"] += time_saved
                return cached_result, True
            
            self.optimization_stats["cache_misses"] += 1
        
        # Strategy 2: Query rewriting for better performance
        if OptimizationStrategy.QUERY_REWRITING in self.enabled_strategies:
            optimized_query = await self._rewrite_query_for_performance(query)
            if optimized_query != query:
                self.optimization_stats["query_rewrites"] += 1
                query = optimized_query
        
        return None, False
    
    async def optimize_search_execution(
        self, 
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[RankedSearchResults]:
        """Optimize search execution and result ranking."""
        
        # Strategy 1: Check cache for search results
        if OptimizationStrategy.CACHING in self.enabled_strategies:
            cached_results = await self._get_cached_search_results(processed_query)
            if cached_results:
                self.optimization_stats["cache_hits"] += 1
                return cached_results
            
            self.optimization_stats["cache_misses"] += 1
        
        # Strategy 2: Parallel processing optimization
        if OptimizationStrategy.PARALLEL_PROCESSING in self.enabled_strategies:
            # This would be implemented with actual search execution
            # For now, we return None to indicate no cached result
            pass
        
        return None
    
    async def optimize_response_generation(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[RankedSearchResults] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[GeneratedResponse]:
        """Optimize response generation."""
        
        # Strategy 1: Check cache for similar responses
        if OptimizationStrategy.CACHING in self.enabled_strategies:
            cached_response = await self._get_cached_response(processed_query, search_results)
            if cached_response:
                self.optimization_stats["cache_hits"] += 1
                return cached_response
            
            self.optimization_stats["cache_misses"] += 1
        
        return None
    
    async def cache_results(
        self,
        query: str,
        classified_query: Optional[ClassifiedQuery] = None,
        processed_query: Optional[ProcessedQuery] = None,
        search_results: Optional[RankedSearchResults] = None,
        response: Optional[GeneratedResponse] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Cache various query processing results for future optimization."""
        
        try:
            # Cache classified query
            if classified_query:
                cache_key = self._generate_classification_cache_key(query, context)
                await self.cache.put(CacheType.CLASSIFICATION_RESULTS, cache_key, classified_query)
            
            # Cache processed query
            if processed_query:
                cache_key = self._generate_processing_cache_key(processed_query.original_query)
                await self.cache.put(CacheType.PROCESSED_QUERIES, cache_key, processed_query)
            
            # Cache search results
            if search_results:
                cache_key = self._generate_search_cache_key(processed_query)
                await self.cache.put(
                    CacheType.SEARCH_RESULTS, 
                    cache_key, 
                    search_results,
                    ttl_seconds=1800  # 30 minutes for search results
                )
            
            # Cache response
            if response:
                cache_key = self._generate_response_cache_key(processed_query, search_results)
                await self.cache.put(
                    CacheType.QUERY_RESULTS, 
                    cache_key, 
                    response,
                    ttl_seconds=3600  # 1 hour for complete responses
                )
        
        except Exception as e:
            logger.warning(f"Error caching results: {e}")
    
    async def _get_cached_classification(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[ClassifiedQuery]:
        """Get cached query classification."""
        
        cache_key = self._generate_classification_cache_key(query, context)
        return await self.cache.get(CacheType.CLASSIFICATION_RESULTS, cache_key)
    
    async def _get_cached_search_results(
        self, 
        processed_query: ProcessedQuery
    ) -> Optional[RankedSearchResults]:
        """Get cached search results."""
        
        cache_key = self._generate_search_cache_key(processed_query)
        return await self.cache.get(CacheType.SEARCH_RESULTS, cache_key)
    
    async def _get_cached_response(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[RankedSearchResults]
    ) -> Optional[GeneratedResponse]:
        """Get cached response."""
        
        cache_key = self._generate_response_cache_key(processed_query, search_results)
        return await self.cache.get(CacheType.QUERY_RESULTS, cache_key)
    
    async def _rewrite_query_for_performance(self, query: str) -> str:
        """Rewrite query for better performance."""
        
        # Simple query optimizations
        optimized = query.strip()
        
        # Remove redundant words
        redundant_words = ["please", "could you", "can you", "would you"]
        for word in redundant_words:
            optimized = optimized.replace(word, "").strip()
        
        # Normalize whitespace
        optimized = " ".join(optimized.split())
        
        # Expand common abbreviations for better matching
        abbreviations = {
            "reqs": "requirements",
            "docs": "documents",
            "info": "information",
            "reg": "regulation",
            "mgmt": "management"
        }
        
        for abbrev, full_form in abbreviations.items():
            optimized = optimized.replace(abbrev, full_form)
        
        return optimized
    
    def _generate_classification_cache_key(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for query classification."""
        
        # Include context elements that affect classification
        context_elements = []
        if context:
            context_elements.extend([
                context.get("user_role", ""),
                context.get("firm_type", ""),
                context.get("business_function", "")
            ])
        
        key_parts = [query.lower().strip()] + context_elements
        key_string = "|".join(str(part) for part in key_parts)
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_processing_cache_key(self, query: str) -> str:
        """Generate cache key for processed query."""
        
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _generate_search_cache_key(self, processed_query: ProcessedQuery) -> str:
        """Generate cache key for search results."""
        
        # Include elements that affect search results
        key_parts = [
            processed_query.normalized_query,
            processed_query.classified_query.query_type.value,
            processed_query.classified_query.query_domain.value,
            "|".join(processed_query.search_terms),
            str(processed_query.search_filters)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_response_cache_key(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[RankedSearchResults]
    ) -> str:
        """Generate cache key for response."""
        
        # Include elements that affect response generation
        key_parts = [
            processed_query.normalized_query,
            processed_query.classified_query.query_type.value,
            processed_query.classified_query.query_intent.value
        ]
        
        # Include search results signature
        if search_results:
            results_signature = f"{len(search_results.ranked_results)}_{search_results.average_relevance_score:.2f}"
            key_parts.append(results_signature)
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        
        cache_stats = self.cache.get_cache_stats()
        
        # Calculate overall cache hit rate
        total_hits = sum(self.optimization_stats.get(f"cache_hits", 0) for _ in range(1))
        total_requests = total_hits + sum(self.optimization_stats.get(f"cache_misses", 0) for _ in range(1))
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "optimization_statistics": self.optimization_stats,
            "cache_statistics": cache_stats,
            "overall_cache_hit_rate": overall_hit_rate,
            "enabled_strategies": [strategy.value for strategy in self.enabled_strategies]
        }
    
    async def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        
        for cache_type in CacheType:
            cache = self.cache.caches[cache_type]
            expired_keys = [
                key for key, entry in cache.items() 
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                await self.cache._evict_entry(cache_type, key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries from {cache_type.value} cache")


class PerformanceMonitor:
    """Monitor and track performance metrics for query processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = defaultdict(list)
        self.current_metrics = defaultdict(float)
        self.monitoring_window_seconds = config.get("monitoring_window_seconds", 300)  # 5 minutes
        self.max_history_entries = config.get("max_history_entries", 1000)
    
    def record_metric(self, metric: PerformanceMetric, value: float, timestamp: Optional[datetime] = None):
        """Record a performance metric."""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Store in history
        self.metrics_history[metric].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Limit history size
        if len(self.metrics_history[metric]) > self.max_history_entries:
            self.metrics_history[metric] = self.metrics_history[metric][-self.max_history_entries:]
        
        # Update current metric (rolling average over window)
        self._update_current_metric(metric)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return dict(self.current_metrics)
    
    def get_metric_history(self, metric: PerformanceMetric, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metric history for specified time period."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            entry for entry in self.metrics_history[metric]
            if entry["timestamp"] > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        summary = {
            "current_metrics": self.get_current_metrics(),
            "metric_trends": {},
            "performance_indicators": {}
        }
        
        # Calculate trends for each metric
        for metric in PerformanceMetric:
            recent_values = self._get_recent_values(metric, hours=1)
            if len(recent_values) >= 2:
                trend = self._calculate_trend(recent_values)
                summary["metric_trends"][metric.value] = trend
        
        # Performance indicators
        summary["performance_indicators"] = {
            "system_healthy": self._is_system_healthy(),
            "cache_efficiency": self._calculate_cache_efficiency(),
            "response_time_trend": self._get_response_time_trend(),
            "throughput_trend": self._get_throughput_trend()
        }
        
        return summary
    
    def _update_current_metric(self, metric: PerformanceMetric):
        """Update current metric value based on recent history."""
        
        recent_values = self._get_recent_values(metric, minutes=5)
        
        if recent_values:
            # Use rolling average for current value
            self.current_metrics[metric] = sum(recent_values) / len(recent_values)
    
    def _get_recent_values(
        self, 
        metric: PerformanceMetric, 
        hours: int = 0, 
        minutes: int = 0
    ) -> List[float]:
        """Get recent metric values within time window."""
        
        if hours == 0 and minutes == 0:
            minutes = 5  # Default to 5 minutes
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours, minutes=minutes)
        
        return [
            entry["value"] for entry in self.metrics_history[metric]
            if entry["timestamp"] > cutoff_time
        ]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values."""
        
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _is_system_healthy(self) -> bool:
        """Determine if system performance is healthy."""
        
        # Check key metrics
        response_time = self.current_metrics.get(PerformanceMetric.TOTAL_RESPONSE_TIME, 0)
        error_rate = self.current_metrics.get(PerformanceMetric.ERROR_RATE, 0)
        cache_hit_rate = self.current_metrics.get(PerformanceMetric.CACHE_HIT_RATE, 0)
        
        # Health thresholds
        return (
            response_time < 5000 and  # Less than 5 seconds
            error_rate < 0.05 and     # Less than 5% error rate
            cache_hit_rate > 0.3      # At least 30% cache hit rate
        )
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate overall cache efficiency."""
        
        hit_rate = self.current_metrics.get(PerformanceMetric.CACHE_HIT_RATE, 0)
        return hit_rate
    
    def _get_response_time_trend(self) -> str:
        """Get response time trend."""
        
        recent_values = self._get_recent_values(PerformanceMetric.TOTAL_RESPONSE_TIME, hours=1)
        return self._calculate_trend(recent_values)
    
    def _get_throughput_trend(self) -> str:
        """Get throughput trend."""
        
        recent_values = self._get_recent_values(PerformanceMetric.THROUGHPUT, hours=1)
        return self._calculate_trend(recent_values)


class QueryPerformanceManager:
    """Main manager for query performance optimization and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer = QueryOptimizer(config.get("optimizer_config", {}))
        self.monitor = PerformanceMonitor(config.get("monitor_config", {}))
        self.optimization_enabled = config.get("optimization_enabled", True)
        self.monitoring_enabled = config.get("monitoring_enabled", True)
    
    async def optimize_query_pipeline(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize entire query processing pipeline."""
        
        start_time = time.time()
        optimization_results = {
            "optimizations_applied": [],
            "time_saved_ms": 0.0,
            "cache_hits": 0,
            "performance_improved": False
        }
        
        if not self.optimization_enabled:
            return optimization_results
        
        try:
            # Optimize query processing
            cached_classification, was_cached = await self.optimizer.optimize_query_processing(query, context)
            if was_cached:
                optimization_results["optimizations_applied"].append("cached_classification")
                optimization_results["cache_hits"] += 1
            
            # Record performance metric
            if self.monitoring_enabled:
                processing_time = (time.time() - start_time) * 1000
                self.monitor.record_metric(PerformanceMetric.QUERY_PROCESSING_TIME, processing_time)
        
        except Exception as e:
            logger.error(f"Error in query pipeline optimization: {e}")
        
        return optimization_results
    
    async def track_query_performance(
        self,
        query_id: str,
        stage: str,
        processing_time_ms: float,
        success: bool = True
    ):
        """Track performance metrics for query processing stages."""
        
        if not self.monitoring_enabled:
            return
        
        try:
            # Map stage to metric
            stage_metrics = {
                "classification": PerformanceMetric.QUERY_PROCESSING_TIME,
                "processing": PerformanceMetric.QUERY_PROCESSING_TIME,
                "search": PerformanceMetric.SEARCH_EXECUTION_TIME,
                "ranking": PerformanceMetric.RANKING_TIME,
                "response_generation": PerformanceMetric.RESPONSE_GENERATION_TIME,
                "total": PerformanceMetric.TOTAL_RESPONSE_TIME
            }
            
            metric = stage_metrics.get(stage, PerformanceMetric.QUERY_PROCESSING_TIME)
            self.monitor.record_metric(metric, processing_time_ms)
            
            # Track error rate
            if not success:
                current_error_rate = self.monitor.current_metrics.get(PerformanceMetric.ERROR_RATE, 0)
                self.monitor.record_metric(PerformanceMetric.ERROR_RATE, current_error_rate + 1)
        
        except Exception as e:
            logger.error(f"Error tracking query performance: {e}")
    
    async def cache_query_results(self, **kwargs):
        """Cache query processing results."""
        
        if self.optimization_enabled:
            await self.optimizer.cache_results(**kwargs)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        
        dashboard = {
            "optimization_stats": {},
            "performance_summary": {},
            "cache_efficiency": {},
            "system_health": {}
        }
        
        try:
            if self.optimization_enabled:
                dashboard["optimization_stats"] = self.optimizer.get_optimization_stats()
            
            if self.monitoring_enabled:
                dashboard["performance_summary"] = self.monitor.get_performance_summary()
                dashboard["cache_efficiency"] = {
                    "hit_rate": self.monitor.current_metrics.get(PerformanceMetric.CACHE_HIT_RATE, 0),
                    "trend": self.monitor._calculate_trend(
                        self.monitor._get_recent_values(PerformanceMetric.CACHE_HIT_RATE, hours=1)
                    )
                }
                dashboard["system_health"] = {
                    "healthy": self.monitor._is_system_healthy(),
                    "response_time": self.monitor.current_metrics.get(PerformanceMetric.TOTAL_RESPONSE_TIME, 0),
                    "throughput": self.monitor.current_metrics.get(PerformanceMetric.THROUGHPUT, 0),
                    "error_rate": self.monitor.current_metrics.get(PerformanceMetric.ERROR_RATE, 0)
                }
        
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            dashboard["error"] = str(e)
        
        return dashboard
    
    async def cleanup_and_optimize(self):
        """Perform cleanup and optimization maintenance."""
        
        try:
            if self.optimization_enabled:
                await self.optimizer.cleanup_expired_cache()
            
            logger.info("Performance cleanup and optimization completed")
        
        except Exception as e:
            logger.error(f"Error in cleanup and optimization: {e}")


# Factory functions
def create_query_optimizer(config: Optional[Dict[str, Any]] = None) -> QueryOptimizer:
    """Create query optimizer."""
    if config is None:
        config = {}
    
    return QueryOptimizer(config)


def create_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Create performance monitor."""
    if config is None:
        config = {}
    
    return PerformanceMonitor(config)


def create_performance_manager(config: Optional[Dict[str, Any]] = None) -> QueryPerformanceManager:
    """Create query performance manager."""
    if config is None:
        config = {}
    
    return QueryPerformanceManager(config)