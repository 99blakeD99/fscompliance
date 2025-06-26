"""Hugging Face provider implementation for Falcon and Mistral Medium models."""

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
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("transformers not available. Hugging Face provider will not work.")


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face provider for Falcon and Mistral Medium models."""
    
    SUPPORTED_MODELS = {
        "falcon-7b": {
            "model_name": "tiiuae/falcon-7b-instruct",
            "model_size": ModelSize.SMALL,
            "capabilities": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.SUMMARIZATION
            ]
        },
        "falcon-40b": {
            "model_name": "tiiuae/falcon-40b-instruct", 
            "model_size": ModelSize.LARGE,
            "capabilities": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CODE_GENERATION
            ]
        },
        "mistral-7b": {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "model_size": ModelSize.SMALL,
            "capabilities": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CLASSIFICATION
            ]
        },
        "mistral-medium": {
            "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "model_size": ModelSize.LARGE,
            "capabilities": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.CLASSIFICATION
            ]
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HUGGINGFACE_AVAILABLE:
            raise LLMProviderError(
                "transformers library is not installed. Please install with: pip install transformers torch",
                provider="huggingface"
            )
        
        self.model_id = config.get("model_id", "mistral-7b")
        self.device = config.get("device", "auto")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.max_memory = config.get("max_memory", None)
        
        # Validate model
        if self.model_id not in self.SUPPORTED_MODELS:
            raise LLMProviderError(
                f"Unsupported model: {self.model_id}. Supported models: {list(self.SUPPORTED_MODELS.keys())}",
                provider="huggingface"
            )
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
        
        # Model information
        self.model_info = self._create_model_info()
        
        # Rate limiting
        self.requests_per_minute = config.get("requests_per_minute", 20)
        self.request_timestamps = []
    
    def _initialize_model(self):
        """Initialize the Hugging Face model and tokenizer."""
        
        model_config = self.SUPPORTED_MODELS[self.model_id]
        model_name = model_config["model_name"]
        
        try:
            logger.info(f"Loading {self.model_id} model from {model_name}")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model loading configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "device_map": self.device if self.device != "auto" else "auto"
            }
            
            # Quantization options
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device if self.device != "auto" else None
            )
            
            logger.info(f"{self.model_id} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_id} model: {e}")
            raise LLMProviderError(f"Failed to initialize {self.model_id} model: {e}", provider="huggingface")
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information based on configuration."""
        
        model_config = self.SUPPORTED_MODELS[self.model_id]
        model_size = model_config["model_size"]
        
        # Metrics based on model type and size
        if "falcon" in self.model_id:
            if model_size == ModelSize.SMALL:
                metrics = ModelMetrics(
                    accuracy_score=0.84,
                    latency_ms=800.0,
                    throughput_rpm=25.0,
                    reliability_score=0.92,
                    cost_per_1k_input_tokens=0.0015,
                    cost_per_1k_output_tokens=0.002,
                    cost_per_request=0.0,
                    max_input_tokens=2048,
                    max_output_tokens=512,
                    instruction_following=0.82,
                    factual_accuracy=0.79,
                    consistency=0.85
                )
            else:  # Large
                metrics = ModelMetrics(
                    accuracy_score=0.89,
                    latency_ms=2500.0,
                    throughput_rpm=8.0,
                    reliability_score=0.94,
                    cost_per_1k_input_tokens=0.006,
                    cost_per_1k_output_tokens=0.008,
                    cost_per_request=0.0,
                    max_input_tokens=2048,
                    max_output_tokens=512,
                    instruction_following=0.88,
                    factual_accuracy=0.86,
                    consistency=0.90
                )
        else:  # Mistral
            if model_size == ModelSize.SMALL:
                metrics = ModelMetrics(
                    accuracy_score=0.87,
                    latency_ms=600.0,
                    throughput_rpm=30.0,
                    reliability_score=0.95,
                    cost_per_1k_input_tokens=0.0025,
                    cost_per_1k_output_tokens=0.0025,
                    cost_per_request=0.0,
                    max_input_tokens=4096,
                    max_output_tokens=1024,
                    instruction_following=0.90,
                    factual_accuracy=0.85,
                    consistency=0.89
                )
            else:  # Large (Mixtral)
                metrics = ModelMetrics(
                    accuracy_score=0.92,
                    latency_ms=1500.0,
                    throughput_rpm=15.0,
                    reliability_score=0.96,
                    cost_per_1k_input_tokens=0.007,
                    cost_per_1k_output_tokens=0.007,
                    cost_per_request=0.0,
                    max_input_tokens=4096,
                    max_output_tokens=1024,
                    instruction_following=0.93,
                    factual_accuracy=0.90,
                    consistency=0.92
                )
        
        return ModelInfo(
            model_id=self.model_id,
            model_name=f"{self.model_id.replace('-', ' ').title()}",
            provider="huggingface",
            model_version="1.0",
            model_size=model_size,
            capabilities=model_config["capabilities"],
            supported_languages=["en", "fr", "de", "es", "it"] if "mistral" in self.model_id else ["en"],
            metrics=metrics,
            is_available=True,
            rate_limit_rpm=self.requests_per_minute,
            data_retention_days=0,  # Self-hosted
            is_gdpr_compliant=True,
            is_financial_approved=True  # Self-hosted models suitable for financial data
        )
    
    def get_provider_name(self) -> str:
        return "huggingface"
    
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
            if self.pipeline is None:
                return False
            
            # Simple test generation
            test_response = self.pipeline(
                "Test", 
                max_new_tokens=1, 
                do_sample=False,
                return_full_text=False
            )
            
            return test_response is not None and len(test_response) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate text using Hugging Face model."""
        
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Prepare prompt
            prompt = self._prepare_prompt(request)
            
            # Tokenize and check length
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_tokens = inputs["input_ids"].shape[1]
            
            if input_tokens > self.model_info.metrics.max_input_tokens:
                raise LLMValidationError(
                    f"Input too long: {input_tokens} tokens (max: {self.model_info.metrics.max_input_tokens})",
                    provider=self.get_provider_name()
                )
            
            # Generate response
            generation_start = time.time()
            
            # Pipeline generation
            max_new_tokens = request.max_tokens or 512
            
            response = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True if request.temperature > 0 else False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            
            generation_end = time.time()
            
            # Extract response text
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get("generated_text", "")
            else:
                response_text = ""
            
            # Handle stop sequences
            if request.stop_sequences:
                for stop_seq in request.stop_sequences:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq)[0]
                        break
            
            # Calculate output tokens
            output_tokens = len(self.tokenizer.encode(response_text))
            total_tokens = input_tokens + output_tokens
            
            # Calculate costs
            estimated_cost = self.model_info.get_cost_per_request(input_tokens, output_tokens)
            
            processing_time = (generation_end - generation_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            
            # Create response
            llm_response = LLMResponse(
                text=response_text.strip(),
                finish_reason="stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=total_latency,
                processing_time_ms=processing_time,
                estimated_cost=estimated_cost,
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
        """Generate embeddings (requires separate embedding model)."""
        
        raise LLMProviderError(
            "Embedding generation not supported by text generation models. Use a dedicated embedding model.",
            provider=self.get_provider_name(),
            error_code="EMBEDDINGS_NOT_SUPPORTED"
        )
    
    async def stream_text(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream text generation (basic implementation)."""
        
        try:
            # For simplicity, generate full text and yield in chunks
            # Real streaming would require lower-level model access
            response = await self.generate_text(request)
            
            # Yield text in chunks
            text = response.text
            chunk_size = 10  # Characters per chunk
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.05)  # Simulate streaming delay
                
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
        """Prepare the prompt for the specific model."""
        
        if "falcon" in self.model_id:
            # Falcon format
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"
            else:
                prompt = f"User: {request.prompt}\n\nAssistant:"
        
        elif "mistral" in self.model_id:
            # Mistral/Mixtral format
            if request.system_prompt:
                prompt = f"<s>[INST] {request.system_prompt}\n\n{request.prompt} [/INST]"
            else:
                prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        else:
            # Generic format
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{request.prompt}"
            else:
                prompt = request.prompt
        
        return prompt


class HuggingFaceProviderFactory:
    """Factory for creating Hugging Face providers with different models."""
    
    @staticmethod
    def create_falcon_7b_provider(
        device: str = "auto",
        load_in_8bit: bool = False,
        **kwargs
    ) -> HuggingFaceProvider:
        """Create Falcon 7B provider."""
        
        config = {
            "model_id": "falcon-7b",
            "device": device,
            "load_in_8bit": load_in_8bit,
            "requests_per_minute": 25,
            **kwargs
        }
        
        return HuggingFaceProvider(config)
    
    @staticmethod
    def create_falcon_40b_provider(
        device: str = "auto",
        load_in_8bit: bool = True,
        **kwargs
    ) -> HuggingFaceProvider:
        """Create Falcon 40B provider."""
        
        config = {
            "model_id": "falcon-40b",
            "device": device,
            "load_in_8bit": load_in_8bit,
            "requests_per_minute": 8,
            **kwargs
        }
        
        return HuggingFaceProvider(config)
    
    @staticmethod
    def create_mistral_7b_provider(
        device: str = "auto",
        load_in_4bit: bool = False,
        **kwargs
    ) -> HuggingFaceProvider:
        """Create Mistral 7B provider."""
        
        config = {
            "model_id": "mistral-7b",
            "device": device,
            "load_in_4bit": load_in_4bit,
            "requests_per_minute": 30,
            **kwargs
        }
        
        return HuggingFaceProvider(config)
    
    @staticmethod
    def create_mistral_medium_provider(
        device: str = "auto",
        load_in_8bit: bool = True,
        **kwargs
    ) -> HuggingFaceProvider:
        """Create Mistral Medium (Mixtral 8x7B) provider."""
        
        config = {
            "model_id": "mistral-medium",
            "device": device,
            "load_in_8bit": load_in_8bit,
            "requests_per_minute": 15,
            **kwargs
        }
        
        return HuggingFaceProvider(config)
    
    @staticmethod
    def create_financial_optimized_provider(
        model_type: str = "mistral-7b",
        **kwargs
    ) -> HuggingFaceProvider:
        """Create provider optimized for financial compliance tasks."""
        
        if model_type not in ["falcon-7b", "falcon-40b", "mistral-7b", "mistral-medium"]:
            model_type = "mistral-7b"
        
        config = {
            "model_id": model_type,
            "device": "auto",
            "load_in_8bit": True if model_type in ["falcon-40b", "mistral-medium"] else False,
            "load_in_4bit": True if model_type in ["mistral-7b"] else False,
            "requests_per_minute": 20,
            **kwargs
        }
        
        return HuggingFaceProvider(config)


# Utility functions
def get_recommended_model_for_task(task_complexity: TaskComplexity) -> str:
    """Get recommended model for specific task complexity."""
    
    recommendations = {
        TaskComplexity.SIMPLE: "mistral-7b",
        TaskComplexity.MODERATE: "mistral-7b", 
        TaskComplexity.COMPLEX: "mistral-medium",
        TaskComplexity.EXPERT: "mistral-medium"
    }
    
    return recommendations.get(task_complexity, "mistral-7b")


def estimate_model_resources(model_id: str, quantization: str = "none") -> Dict[str, Any]:
    """Estimate resource requirements for different models."""
    
    base_requirements = {
        "falcon-7b": {"memory_gb": 14, "vram_gb": 14},
        "falcon-40b": {"memory_gb": 80, "vram_gb": 80},
        "mistral-7b": {"memory_gb": 14, "vram_gb": 14},
        "mistral-medium": {"memory_gb": 90, "vram_gb": 90}
    }
    
    if model_id not in base_requirements:
        model_id = "mistral-7b"
    
    requirements = base_requirements[model_id].copy()
    
    # Apply quantization reduction
    if quantization == "8bit":
        requirements["memory_gb"] *= 0.5
        requirements["vram_gb"] *= 0.5
    elif quantization == "4bit":
        requirements["memory_gb"] *= 0.25
        requirements["vram_gb"] *= 0.25
    
    # Add CPU-only fallback estimate
    requirements["cpu_memory_gb"] = requirements["memory_gb"] * 1.2
    
    return requirements


def validate_model_compatibility(model_id: str, available_memory_gb: float) -> bool:
    """Validate if model can run with available memory."""
    
    requirements = estimate_model_resources(model_id, "8bit")  # Conservative estimate
    required_memory = min(requirements["memory_gb"], requirements["cpu_memory_gb"])
    
    return available_memory_gb >= required_memory