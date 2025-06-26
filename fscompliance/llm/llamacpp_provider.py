"""LLaMA 3 provider implementation using llama-cpp-python."""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio
import json

from pydantic import ValidationError

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
    ValidationError as LLMValidationError
)

logger = logging.getLogger(__name__)

# Optional dependency handling
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. LLaMA provider will not work.")


class LLaMACppProvider(BaseLLMProvider):
    """LLaMA 3 provider using llama-cpp-python for local inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not LLAMA_CPP_AVAILABLE:
            raise LLMProviderError(
                "llama-cpp-python is not installed. Please install with: pip install llama-cpp-python",
                provider="llama_cpp"
            )
        
        self.model_path = config.get("model_path")
        self.model_size = config.get("model_size", "7b")
        self.context_length = config.get("context_length", 4096)
        self.n_gpu_layers = config.get("n_gpu_layers", 0)
        self.verbose = config.get("verbose", False)
        
        # Initialize model
        self.llm = None
        self._initialize_model()
        
        # Model information
        self.model_info = self._create_model_info()
        
        # Rate limiting
        self.requests_per_minute = config.get("requests_per_minute", 60)
        self.request_timestamps = []
    
    def _initialize_model(self):
        """Initialize the LLaMA model."""
        
        if not self.model_path:
            raise LLMProviderError(
                "model_path is required for LLaMA provider",
                provider="llama_cpp"
            )
        
        try:
            logger.info(f"Loading LLaMA model from {self.model_path}")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                n_threads=self.config.get("n_threads", 4),
                n_batch=self.config.get("n_batch", 512),
                use_mlock=self.config.get("use_mlock", False),
                use_mmap=self.config.get("use_mmap", True)
            )
            
            logger.info("LLaMA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLaMA model: {e}")
            raise LLMProviderError(f"Failed to initialize LLaMA model: {e}", provider="llama_cpp")
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information based on configuration."""
        
        # Model size mapping
        size_mapping = {
            "7b": ModelSize.SMALL,
            "13b": ModelSize.MEDIUM,
            "30b": ModelSize.LARGE,
            "65b": ModelSize.XLARGE,
            "70b": ModelSize.XLARGE
        }
        
        model_size = size_mapping.get(self.model_size.lower(), ModelSize.MEDIUM)
        
        # Metrics based on model size
        metrics_mapping = {
            ModelSize.SMALL: ModelMetrics(
                accuracy_score=0.82,
                latency_ms=500.0,
                throughput_rpm=30.0,
                reliability_score=0.95,
                cost_per_1k_input_tokens=0.0,  # Local inference
                cost_per_1k_output_tokens=0.0,
                cost_per_request=0.0,
                max_input_tokens=3500,
                max_output_tokens=512,
                instruction_following=0.85,
                factual_accuracy=0.80,
                consistency=0.88
            ),
            ModelSize.MEDIUM: ModelMetrics(
                accuracy_score=0.87,
                latency_ms=1000.0,
                throughput_rpm=20.0,
                reliability_score=0.95,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                cost_per_request=0.0,
                max_input_tokens=3500,
                max_output_tokens=512,
                instruction_following=0.90,
                factual_accuracy=0.85,
                consistency=0.90
            ),
            ModelSize.LARGE: ModelMetrics(
                accuracy_score=0.90,
                latency_ms=2000.0,
                throughput_rpm=10.0,
                reliability_score=0.95,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                cost_per_request=0.0,
                max_input_tokens=3500,
                max_output_tokens=512,
                instruction_following=0.92,
                factual_accuracy=0.88,
                consistency=0.92
            ),
            ModelSize.XLARGE: ModelMetrics(
                accuracy_score=0.93,
                latency_ms=4000.0,
                throughput_rpm=5.0,
                reliability_score=0.95,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                cost_per_request=0.0,
                max_input_tokens=3500,
                max_output_tokens=512,
                instruction_following=0.95,
                factual_accuracy=0.92,
                consistency=0.94
            )
        }
        
        return ModelInfo(
            model_id=f"llama3-{self.model_size}",
            model_name=f"LLaMA 3 {self.model_size.upper()}",
            provider="llama_cpp",
            model_version="3.0",
            model_size=model_size,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CLASSIFICATION
            ],
            supported_languages=["en"],
            metrics=metrics_mapping[model_size],
            is_available=True,
            rate_limit_rpm=self.requests_per_minute,
            data_retention_days=0,  # Local processing
            is_gdpr_compliant=True,
            is_financial_approved=True  # Local processing suitable for financial data
        )
    
    def get_provider_name(self) -> str:
        return "llama_cpp"
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        return [self.model_info]
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information for specific model."""
        if model_id == self.model_info.model_id:
            return self.model_info
        return None
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            if self.llm is None:
                return False
            
            # Simple test generation
            test_response = self.llm(
                "Test", 
                max_tokens=1, 
                echo=False, 
                stop=[".", "\n"]
            )
            
            return test_response is not None
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate text using LLaMA model."""
        
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Prepare prompt
            prompt = self._prepare_prompt(request)
            
            # Estimate input tokens
            input_tokens = self.estimate_tokens(prompt)
            
            # Check token limits
            if input_tokens > self.model_info.metrics.max_input_tokens:
                raise LLMValidationError(
                    f"Input too long: {input_tokens} tokens (max: {self.model_info.metrics.max_input_tokens})",
                    provider=self.get_provider_name()
                )
            
            # Generate response
            generation_start = time.time()
            
            response = self.llm(
                prompt,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences or ["\n\n"],
                echo=False,
                stream=False
            )
            
            generation_end = time.time()
            
            # Extract response text
            if isinstance(response, dict):
                response_text = response.get("choices", [{}])[0].get("text", "")
                finish_reason = response.get("choices", [{}])[0].get("finish_reason", "stop")
            else:
                response_text = str(response)
                finish_reason = "stop"
            
            # Calculate metrics
            output_tokens = self.estimate_tokens(response_text)
            total_tokens = input_tokens + output_tokens
            
            processing_time = (generation_end - generation_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            
            # Create response
            llm_response = LLMResponse(
                text=response_text.strip(),
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=total_latency,
                processing_time_ms=processing_time,
                estimated_cost=0.0,  # Local inference
                model_used=self.model_info.model_id,
                provider_used=self.get_provider_name(),
                request_id=request_id,
                timestamp=datetime.utcnow()
            )
            
            # Update provider statistics
            self._update_stats(llm_response, True)
            
            logger.info(f"Generated {output_tokens} tokens in {processing_time:.1f}ms")
            
            return llm_response
            
        except Exception as e:
            # Create error response
            error_response = LLMResponse(
                text="",
                finish_reason="error",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                processing_time_ms=0.0,
                estimated_cost=0.0,
                model_used=self.model_info.model_id,
                provider_used=self.get_provider_name(),
                request_id=request_id,
                timestamp=datetime.utcnow(),
                errors=[str(e)]
            )
            
            self._update_stats(error_response, False)
            
            if isinstance(e, (LLMProviderError, ValidationError)):
                raise
            else:
                raise LLMProviderError(f"Text generation failed: {e}", provider=self.get_provider_name())
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings (not supported by standard llama-cpp-python)."""
        
        # Basic implementation - would need specialized embedding model
        raise LLMProviderError(
            "Embedding generation not supported by standard LLaMA models. Use a specialized embedding model.",
            provider=self.get_provider_name(),
            error_code="EMBEDDINGS_NOT_SUPPORTED"
        )
    
    async def stream_text(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream text generation."""
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Prepare prompt
            prompt = self._prepare_prompt(request)
            
            # Stream generation
            stream = self.llm(
                prompt,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences or ["\n\n"],
                echo=False,
                stream=True
            )
            
            for chunk in stream:
                if isinstance(chunk, dict):
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield text
                        # Small delay to prevent overwhelming the consumer
                        await asyncio.sleep(0.01)
                        
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise LLMProviderError(f"Text streaming failed: {e}", provider=self.get_provider_name())
    
    def _validate_request(self, request: LLMRequest):
        """Validate the request parameters."""
        
        if not request.prompt:
            raise LLMValidationError("Prompt is required", provider=self.get_provider_name())
        
        if request.max_tokens and request.max_tokens > self.model_info.metrics.max_output_tokens:
            raise LLMValidationError(
                f"max_tokens ({request.max_tokens}) exceeds model limit ({self.model_info.metrics.max_output_tokens})",
                provider=self.get_provider_name()
            )
        
        if not (0.0 <= request.temperature <= 2.0):
            raise LLMValidationError(
                f"temperature must be between 0.0 and 2.0, got {request.temperature}",
                provider=self.get_provider_name()
            )
        
        if not (0.0 <= request.top_p <= 1.0):
            raise LLMValidationError(
                f"top_p must be between 0.0 and 1.0, got {request.top_p}",
                provider=self.get_provider_name()
            )
    
    async def _check_rate_limits(self):
        """Check and enforce rate limits."""
        
        current_time = time.time()
        
        # Remove old timestamps (older than 1 minute)
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if current_time - ts < 60
        ]
        
        # Check if we're at the limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                provider=self.get_provider_name()
            )
        
        # Add current timestamp
        self.request_timestamps.append(current_time)
    
    def _prepare_prompt(self, request: LLMRequest) -> str:
        """Prepare the prompt for the model."""
        
        # LLaMA 3 Chat format
        if request.system_prompt:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{request.system_prompt}<|eot_id|>\n"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|>\n"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|>\n"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt


class LLaMAProviderFactory:
    """Factory for creating LLaMA providers with different configurations."""
    
    @staticmethod
    def create_provider(
        model_path: str,
        model_size: str = "7b",
        context_length: int = 4096,
        n_gpu_layers: int = 0,
        **kwargs
    ) -> LLaMACppProvider:
        """Create a LLaMA provider with specified configuration."""
        
        config = {
            "model_path": model_path,
            "model_size": model_size,
            "context_length": context_length,
            "n_gpu_layers": n_gpu_layers,
            **kwargs
        }
        
        return LLaMACppProvider(config)
    
    @staticmethod
    def create_financial_optimized_provider(
        model_path: str,
        model_size: str = "13b"
    ) -> LLaMACppProvider:
        """Create a provider optimized for financial compliance tasks."""
        
        config = {
            "model_path": model_path,
            "model_size": model_size,
            "context_length": 4096,
            "n_gpu_layers": 20,  # Assume GPU acceleration
            "requests_per_minute": 30,  # Conservative for complex tasks
            "verbose": False,
            "n_threads": 6,
            "n_batch": 512,
            "use_mlock": True,
            "use_mmap": True
        }
        
        return LLaMACppProvider(config)
    
    @staticmethod
    def create_development_provider(
        model_path: str,
        model_size: str = "7b"
    ) -> LLaMACppProvider:
        """Create a provider optimized for development and testing."""
        
        config = {
            "model_path": model_path,
            "model_size": model_size,
            "context_length": 2048,
            "n_gpu_layers": 0,  # CPU only for development
            "requests_per_minute": 60,
            "verbose": True,
            "n_threads": 4,
            "n_batch": 256
        }
        
        return LLaMACppProvider(config)


# Utility functions
def get_model_requirements(task_complexity: TaskComplexity) -> Dict[str, Any]:
    """Get recommended model requirements for task complexity."""
    
    requirements = {
        TaskComplexity.SIMPLE: {
            "min_model_size": "7b",
            "recommended_context": 2048,
            "max_tokens": 256
        },
        TaskComplexity.MODERATE: {
            "min_model_size": "13b",
            "recommended_context": 4096,
            "max_tokens": 512
        },
        TaskComplexity.COMPLEX: {
            "min_model_size": "30b",
            "recommended_context": 4096,
            "max_tokens": 1024
        },
        TaskComplexity.EXPERT: {
            "min_model_size": "70b",
            "recommended_context": 4096,
            "max_tokens": 2048
        }
    }
    
    return requirements.get(task_complexity, requirements[TaskComplexity.MODERATE])


def validate_model_path(model_path: str) -> bool:
    """Validate that the model path exists and is accessible."""
    
    import os
    
    if not model_path:
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False
    
    if not os.path.isfile(model_path):
        logger.error(f"Model path is not a file: {model_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.gguf', '.ggml', '.bin']
    file_extension = os.path.splitext(model_path)[1].lower()
    
    if file_extension not in valid_extensions:
        logger.warning(f"Model file extension '{file_extension}' may not be supported")
    
    return True


def estimate_model_memory_usage(model_size: str, n_gpu_layers: int = 0) -> Dict[str, float]:
    """Estimate memory usage for different model sizes."""
    
    # Memory estimates in GB
    memory_estimates = {
        "7b": {"cpu": 4.0, "gpu_per_layer": 0.15},
        "13b": {"cpu": 8.0, "gpu_per_layer": 0.25},
        "30b": {"cpu": 16.0, "gpu_per_layer": 0.5},
        "65b": {"cpu": 32.0, "gpu_per_layer": 1.0},
        "70b": {"cpu": 35.0, "gpu_per_layer": 1.1}
    }
    
    if model_size not in memory_estimates:
        model_size = "13b"  # Default fallback
    
    estimates = memory_estimates[model_size]
    
    cpu_memory = estimates["cpu"]
    gpu_memory = n_gpu_layers * estimates["gpu_per_layer"]
    
    # Reduce CPU memory if using GPU layers
    if n_gpu_layers > 0:
        cpu_memory = cpu_memory * (1 - min(n_gpu_layers / 40, 0.8))  # Assume ~40 layers max
    
    return {
        "cpu_memory_gb": cpu_memory,
        "gpu_memory_gb": gpu_memory,
        "total_memory_gb": cpu_memory + gpu_memory
    }