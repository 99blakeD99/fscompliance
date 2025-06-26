"""LLM provider abstraction interface and implementations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import json
import time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Capabilities that LLM models can support."""
    TEXT_GENERATION = "text_generation"
    TEXT_EMBEDDING = "text_embedding"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"


class ModelSize(str, Enum):
    """Model size categories affecting cost and capability."""
    SMALL = "small"      # <7B parameters (fast, cheap)
    MEDIUM = "medium"    # 7B-30B parameters (balanced)
    LARGE = "large"      # 30B-70B parameters (high quality)
    XLARGE = "xlarge"    # 70B+ parameters (best quality, expensive)


class TaskComplexity(str, Enum):
    """Complexity levels for different compliance tasks."""
    SIMPLE = "simple"        # Basic extraction, classification
    MODERATE = "moderate"    # Analysis, summarization
    COMPLEX = "complex"      # Reasoning, interpretation
    EXPERT = "expert"        # Complex regulatory analysis


@dataclass
class ModelMetrics:
    """Performance and cost metrics for a model."""
    
    # Performance metrics
    accuracy_score: float = 0.0       # 0.0-1.0
    latency_ms: float = 0.0          # Average response time
    throughput_rpm: float = 0.0       # Requests per minute
    reliability_score: float = 0.0    # Uptime/success rate
    
    # Cost metrics  
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    cost_per_request: float = 0.0
    
    # Context limits
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    
    # Quality metrics
    instruction_following: float = 0.0  # How well it follows instructions
    factual_accuracy: float = 0.0       # Accuracy for factual queries
    consistency: float = 0.0            # Consistency across similar inputs


class ModelInfo(BaseModel):
    """Information about an LLM model."""
    
    # Basic information
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    provider: str = Field(..., description="Model provider (OpenAI, Anthropic, etc.)")
    model_version: str = Field(..., description="Model version")
    
    # Model characteristics
    model_size: ModelSize = Field(..., description="Model size category")
    capabilities: List[ModelCapability] = Field(..., description="Supported capabilities")
    supported_languages: List[str] = Field(default_factory=lambda: ["en"], description="Supported languages")
    
    # Performance and cost
    metrics: ModelMetrics = Field(..., description="Performance and cost metrics")
    
    # Availability
    is_available: bool = Field(True, description="Whether model is currently available")
    rate_limit_rpm: int = Field(60, description="Rate limit in requests per minute")
    
    # Compliance and security
    data_retention_days: int = Field(0, description="Data retention period (0 = no retention)")
    is_gdpr_compliant: bool = Field(True, description="GDPR compliance status")
    is_financial_approved: bool = Field(False, description="Approved for financial services")
    
    def get_cost_per_request(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a specific request."""
        input_cost = (input_tokens / 1000) * self.metrics.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.metrics.cost_per_1k_output_tokens
        return input_cost + output_cost + self.metrics.cost_per_request
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if model supports a specific capability."""
        return capability in self.capabilities
    
    def is_suitable_for_complexity(self, complexity: TaskComplexity) -> bool:
        """Check if model is suitable for task complexity."""
        complexity_requirements = {
            TaskComplexity.SIMPLE: [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE],
            TaskComplexity.MODERATE: [ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE],
            TaskComplexity.COMPLEX: [ModelSize.LARGE, ModelSize.XLARGE],
            TaskComplexity.EXPERT: [ModelSize.XLARGE]
        }
        return self.model_size in complexity_requirements[complexity]


class LLMRequest(BaseModel):
    """Request to an LLM provider."""
    
    # Core request
    prompt: str = Field(..., description="Input prompt for the model")
    system_prompt: Optional[str] = Field(None, description="System prompt for context")
    
    # Generation parameters
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop generation at these sequences")
    
    # Compliance parameters
    task_type: str = Field("general", description="Type of compliance task")
    complexity: TaskComplexity = Field(TaskComplexity.MODERATE, description="Task complexity level")
    required_capabilities: List[ModelCapability] = Field(
        default_factory=lambda: [ModelCapability.TEXT_GENERATION], 
        description="Required model capabilities"
    )
    
    # Context and metadata
    user_id: Optional[str] = Field(None, description="User identifier for tracking")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    # Budget and performance constraints
    max_cost: Optional[float] = Field(None, description="Maximum acceptable cost")
    max_latency_ms: Optional[int] = Field(None, description="Maximum acceptable latency")
    require_financial_approved: bool = Field(False, description="Require financially approved models")


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    
    # Core response
    text: str = Field(..., description="Generated text response")
    finish_reason: str = Field(..., description="Reason generation finished")
    
    # Usage metrics
    input_tokens: int = Field(..., description="Number of input tokens consumed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    total_tokens: int = Field(..., description="Total tokens consumed")
    
    # Performance metrics
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    processing_time_ms: float = Field(..., description="Actual processing time")
    
    # Cost information
    estimated_cost: float = Field(..., description="Estimated cost for this request")
    
    # Model information
    model_used: str = Field(..., description="Model that generated the response")
    provider_used: str = Field(..., description="Provider that handled the request")
    
    # Quality indicators
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence in response")
    safety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Safety/appropriateness score")
    
    # Metadata
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")


class EmbeddingRequest(BaseModel):
    """Request for text embeddings."""
    
    # Core request
    texts: List[str] = Field(..., description="Texts to embed")
    
    # Parameters
    model_preference: Optional[str] = Field(None, description="Preferred embedding model")
    embedding_dimensions: Optional[int] = Field(None, description="Desired embedding dimensions")
    
    # Context
    task_type: str = Field("semantic_search", description="Intended use for embeddings")
    user_id: Optional[str] = Field(None, description="User identifier")
    request_id: Optional[str] = Field(None, description="Request identifier")


class EmbeddingResponse(BaseModel):
    """Response containing text embeddings."""
    
    # Core response
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    
    # Metadata
    model_used: str = Field(..., description="Embedding model used")
    dimensions: int = Field(..., description="Embedding dimensions")
    
    # Usage metrics
    total_tokens: int = Field(..., description="Total tokens processed")
    latency_ms: float = Field(..., description="Processing latency")
    estimated_cost: float = Field(..., description="Estimated cost")
    
    # Request context
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "average_latency_ms": 0.0
        }
    
    @abstractmethod
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate text using the provider's model."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using the provider's embedding model."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from this provider."""
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and available."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass
    
    async def stream_text(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream text generation (optional capability)."""
        # Default implementation: generate full response and yield it
        response = await self.generate_text(request)
        yield response.text
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (approximate)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def _update_stats(self, response: LLMResponse, success: bool):
        """Update provider statistics."""
        self.provider_stats["requests_made"] += 1
        
        if success:
            self.provider_stats["successful_requests"] += 1
            self.provider_stats["total_cost"] += response.estimated_cost
            self.provider_stats["total_tokens"] += response.total_tokens
            
            # Update average latency
            total_requests = self.provider_stats["successful_requests"]
            old_avg = self.provider_stats["average_latency_ms"]
            new_latency = response.latency_ms
            self.provider_stats["average_latency_ms"] = (
                (old_avg * (total_requests - 1) + new_latency) / total_requests
            )
        else:
            self.provider_stats["failed_requests"] += 1
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider performance statistics."""
        stats = self.provider_stats.copy()
        
        # Calculate derived metrics
        total_requests = stats["requests_made"]
        if total_requests > 0:
            stats["success_rate"] = stats["successful_requests"] / total_requests
            stats["failure_rate"] = stats["failed_requests"] / total_requests
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        if stats["total_tokens"] > 0:
            stats["cost_per_token"] = stats["total_cost"] / stats["total_tokens"]
        else:
            stats["cost_per_token"] = 0.0
        
        return stats


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    
    def __init__(self, message: str, provider: str = None, model: str = None, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.error_code = error_code


class RateLimitError(LLMProviderError):
    """Raised when rate limits are exceeded."""
    pass


class ModelNotAvailableError(LLMProviderError):
    """Raised when requested model is not available."""
    pass


class InsufficientBudgetError(LLMProviderError):
    """Raised when request exceeds budget constraints."""
    pass


class ValidationError(LLMProviderError):
    """Raised when request validation fails."""
    pass


# Provider capability matrix for compliance tasks
COMPLIANCE_TASK_REQUIREMENTS = {
    "requirement_extraction": {
        "required_capabilities": [ModelCapability.TEXT_GENERATION, ModelCapability.ANALYSIS],
        "min_complexity": TaskComplexity.MODERATE,
        "preferred_size": ModelSize.MEDIUM
    },
    "gap_detection": {
        "required_capabilities": [ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
        "min_complexity": TaskComplexity.COMPLEX,
        "preferred_size": ModelSize.LARGE
    },
    "regulatory_interpretation": {
        "required_capabilities": [ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
        "min_complexity": TaskComplexity.EXPERT,
        "preferred_size": ModelSize.XLARGE
    },
    "document_summarization": {
        "required_capabilities": [ModelCapability.SUMMARIZATION],
        "min_complexity": TaskComplexity.SIMPLE,
        "preferred_size": ModelSize.SMALL
    },
    "response_generation": {
        "required_capabilities": [ModelCapability.TEXT_GENERATION],
        "min_complexity": TaskComplexity.MODERATE,
        "preferred_size": ModelSize.MEDIUM
    },
    "entity_extraction": {
        "required_capabilities": [ModelCapability.TEXT_GENERATION, ModelCapability.CLASSIFICATION],
        "min_complexity": TaskComplexity.SIMPLE,
        "preferred_size": ModelSize.SMALL
    },
    "knowledge_graph_processing": {
        "required_capabilities": [ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
        "min_complexity": TaskComplexity.COMPLEX,
        "preferred_size": ModelSize.LARGE
    }
}


def get_task_requirements(task_type: str) -> Dict[str, Any]:
    """Get capability requirements for a specific compliance task."""
    return COMPLIANCE_TASK_REQUIREMENTS.get(task_type, {
        "required_capabilities": [ModelCapability.TEXT_GENERATION],
        "min_complexity": TaskComplexity.MODERATE,
        "preferred_size": ModelSize.MEDIUM
    })


def validate_provider_for_task(provider: BaseLLMProvider, task_type: str) -> bool:
    """Validate if a provider can handle a specific compliance task."""
    requirements = get_task_requirements(task_type)
    available_models = provider.get_available_models()
    
    # Check if any model meets requirements
    for model in available_models:
        if not model.is_available:
            continue
        
        # Check capabilities
        required_caps = requirements.get("required_capabilities", [])
        if not all(model.supports_capability(cap) for cap in required_caps):
            continue
        
        # Check complexity
        min_complexity = requirements.get("min_complexity", TaskComplexity.SIMPLE)
        if not model.is_suitable_for_complexity(min_complexity):
            continue
        
        # Model is suitable
        return True
    
    return False