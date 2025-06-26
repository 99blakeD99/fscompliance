"""LLM Abstraction Layer - Multi-model support (LLaMA 3, Falcon, Mistral Medium)."""

from .providers import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ModelMetrics,
    ModelCapability,
    ModelSize,
    TaskComplexity,
    LLMProviderError,
    RateLimitError,
    ModelNotAvailableError,
    InsufficientBudgetError,
    ValidationError,
    COMPLIANCE_TASK_REQUIREMENTS,
    get_task_requirements,
    validate_provider_for_task
)

from .llamacpp_provider import (
    LLaMACppProvider,
    LLaMAProviderFactory,
    get_model_requirements,
    validate_model_path,
    estimate_model_memory_usage
)

from .huggingface_provider import (
    HuggingFaceProvider,
    HuggingFaceProviderFactory,
    get_recommended_model_for_task,
    estimate_model_resources,
    validate_model_compatibility
)

from .model_selector import (
    ModelSelectionCriteria,
    ModelSelection,
    ModelScore,
    SelectionStrategy,
    ProviderType,
    BaseModelSelector,
    CostOptimizedSelector,
    ModelSelectionManager,
    create_cost_optimized_selector,
    create_model_selection_manager,
    create_financial_compliance_criteria
)

from .response_cache import (
    CacheStrategy,
    CacheLevel,
    CacheKey,
    CacheEntry,
    CacheStatistics,
    BaseCacheBackend,
    MemoryCacheBackend,
    LLMResponseCache,
    CacheAwareLLMProvider,
    create_response_cache,
    create_financial_compliance_cache,
    cache_aware_provider
)

__all__ = [
    # Base provider interface
    "BaseLLMProvider",
    "LLMRequest",
    "LLMResponse", 
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ModelInfo",
    "ModelMetrics",
    
    # Enums and types
    "ModelCapability",
    "ModelSize", 
    "TaskComplexity",
    
    # Exceptions
    "LLMProviderError",
    "RateLimitError",
    "ModelNotAvailableError", 
    "InsufficientBudgetError",
    "ValidationError",
    
    # Provider utilities
    "COMPLIANCE_TASK_REQUIREMENTS",
    "get_task_requirements",
    "validate_provider_for_task",
    
    # LLaMA provider
    "LLaMACppProvider",
    "LLaMAProviderFactory",
    "get_model_requirements",
    "validate_model_path",
    "estimate_model_memory_usage",
    
    # Hugging Face provider
    "HuggingFaceProvider",
    "HuggingFaceProviderFactory",
    "get_recommended_model_for_task",
    "estimate_model_resources",
    "validate_model_compatibility",
    
    # Model selection
    "ModelSelectionCriteria",
    "ModelSelection",
    "ModelScore",
    "SelectionStrategy",
    "ProviderType",
    "BaseModelSelector",
    "CostOptimizedSelector",
    "ModelSelectionManager",
    "create_cost_optimized_selector",
    "create_model_selection_manager",
    "create_financial_compliance_criteria",
    
    # Response caching
    "CacheStrategy",
    "CacheLevel",
    "CacheKey",
    "CacheEntry", 
    "CacheStatistics",
    "BaseCacheBackend",
    "MemoryCacheBackend",
    "LLMResponseCache",
    "CacheAwareLLMProvider",
    "create_response_cache",
    "create_financial_compliance_cache",
    "cache_aware_provider"
]