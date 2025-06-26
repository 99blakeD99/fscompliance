"""Cost-based model selection logic for optimal LLM provider routing."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from pydantic import BaseModel, Field

from .providers import (
    BaseLLMProvider,
    LLMRequest,
    ModelInfo,
    ModelCapability,
    ModelSize,
    TaskComplexity,
    LLMProviderError,
    get_task_requirements
)

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    """Model selection strategies."""
    COST_OPTIMIZED = "cost_optimized"           # Minimize cost
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Maximize performance
    BALANCED = "balanced"                       # Balance cost and performance
    QUALITY_FIRST = "quality_first"            # Prioritize quality over cost
    LATENCY_OPTIMIZED = "latency_optimized"    # Minimize response time
    CUSTOM_WEIGHTED = "custom_weighted"        # Custom weighted scoring


class ProviderType(str, Enum):
    """Types of LLM providers."""
    LOCAL = "local"           # Local/self-hosted models
    CLOUD_API = "cloud_api"   # Cloud API services
    HYBRID = "hybrid"         # Hybrid deployment


@dataclass
class ModelSelectionCriteria:
    """Criteria for model selection."""
    
    # Task requirements
    required_capabilities: List[ModelCapability]
    min_complexity: TaskComplexity
    task_type: str
    
    # Budget constraints
    max_cost_per_request: Optional[float] = None
    max_cost_per_1k_tokens: Optional[float] = None
    monthly_budget: Optional[float] = None
    
    # Performance requirements
    max_latency_ms: Optional[int] = None
    min_throughput_rpm: Optional[float] = None
    min_accuracy_score: Optional[float] = None
    
    # Quality requirements
    min_instruction_following: Optional[float] = None
    min_factual_accuracy: Optional[float] = None
    min_consistency: Optional[float] = None
    
    # Compliance requirements
    require_financial_approved: bool = False
    require_gdpr_compliant: bool = True
    max_data_retention_days: Optional[int] = None
    
    # Selection preferences
    selection_strategy: SelectionStrategy = SelectionStrategy.BALANCED
    prefer_local: bool = False
    custom_weights: Optional[Dict[str, float]] = None


@dataclass
class ModelScore:
    """Scoring details for a model candidate."""
    
    model_info: ModelInfo
    provider: BaseLLMProvider
    
    # Individual scores (0.0 to 1.0)
    cost_score: float
    performance_score: float
    quality_score: float
    compliance_score: float
    capability_score: float
    
    # Weighted final score
    final_score: float
    
    # Detailed breakdown
    scoring_details: Dict[str, Any]
    
    # Estimated metrics for request
    estimated_cost: float
    estimated_latency_ms: float
    meets_requirements: bool
    
    def get_summary(self) -> Dict[str, Any]:
        """Get scoring summary."""
        return {
            "model_id": self.model_info.model_id,
            "provider": self.provider.get_provider_name(),
            "final_score": self.final_score,
            "cost_score": self.cost_score,
            "performance_score": self.performance_score,
            "quality_score": self.quality_score,
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency_ms,
            "meets_requirements": self.meets_requirements
        }


class ModelSelection(BaseModel):
    """Result of model selection process."""
    
    # Selected model
    selected_provider: str = Field(..., description="Selected provider name")
    selected_model: ModelInfo = Field(..., description="Selected model information")
    selection_score: float = Field(..., description="Selection score")
    
    # Alternative options
    alternative_models: List[ModelScore] = Field(default_factory=list, description="Alternative model options")
    
    # Selection context
    selection_criteria: ModelSelectionCriteria = Field(..., description="Selection criteria used")
    selection_reasoning: List[str] = Field(default_factory=list, description="Reasoning for selection")
    
    # Performance predictions
    estimated_cost: float = Field(..., description="Estimated cost for request")
    estimated_latency_ms: float = Field(..., description="Estimated latency")
    confidence_level: float = Field(..., description="Confidence in selection")
    
    # Metadata
    selection_timestamp: datetime = Field(default_factory=datetime.utcnow)
    selection_duration_ms: float = Field(..., description="Time taken for selection")
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get selection summary."""
        return {
            "selected_model": self.selected_model.model_id,
            "selected_provider": self.selected_provider,
            "score": self.selection_score,
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency_ms,
            "confidence": self.confidence_level,
            "alternatives_count": len(self.alternative_models),
            "strategy": self.selection_criteria.selection_strategy.value
        }


class BaseModelSelector(ABC):
    """Abstract base class for model selection algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selection_stats = {
            "selections_made": 0,
            "cost_savings": 0.0,
            "performance_improvements": 0.0,
            "selection_time_ms": 0.0
        }
    
    @abstractmethod
    async def select_model(
        self,
        providers: List[BaseLLMProvider],
        request: LLMRequest,
        criteria: ModelSelectionCriteria
    ) -> ModelSelection:
        """Select optimal model based on criteria."""
        pass
    
    @abstractmethod
    def get_selector_name(self) -> str:
        """Get selector algorithm name."""
        pass


class CostOptimizedSelector(BaseModelSelector):
    """Cost-optimized model selector with performance constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cost_weight = config.get("cost_weight", 0.6)
        self.performance_weight = config.get("performance_weight", 0.4)
        self.quality_threshold = config.get("quality_threshold", 0.7)
    
    def get_selector_name(self) -> str:
        return "cost_optimized_selector"
    
    async def select_model(
        self,
        providers: List[BaseLLMProvider],
        request: LLMRequest,
        criteria: ModelSelectionCriteria
    ) -> ModelSelection:
        """Select model optimizing for cost while meeting performance requirements."""
        
        start_time = datetime.utcnow()
        
        try:
            # Get all available models
            candidate_models = []
            for provider in providers:
                try:
                    models = provider.get_available_models()
                    for model in models:
                        if model.is_available:
                            candidate_models.append((provider, model))
                except Exception as e:
                    logger.warning(f"Error getting models from {provider.get_provider_name()}: {e}")
            
            if not candidate_models:
                raise LLMProviderError("No available models found")
            
            # Score all candidates
            scored_models = []
            for provider, model in candidate_models:
                score = await self._score_model(provider, model, request, criteria)
                if score.meets_requirements:
                    scored_models.append(score)
            
            if not scored_models:
                raise LLMProviderError("No models meet the specified criteria")
            
            # Select best model based on strategy
            best_model = self._select_best_model(scored_models, criteria)
            
            # Calculate alternatives
            alternatives = [m for m in scored_models if m != best_model]
            alternatives.sort(key=lambda x: x.final_score, reverse=True)
            
            # Build selection reasoning
            reasoning = self._build_reasoning(best_model, alternatives, criteria)
            
            # Calculate confidence
            confidence = self._calculate_confidence(best_model, alternatives)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            
            selection = ModelSelection(
                selected_provider=best_model.provider.get_provider_name(),
                selected_model=best_model.model_info,
                selection_score=best_model.final_score,
                alternative_models=alternatives[:5],  # Top 5 alternatives
                selection_criteria=criteria,
                selection_reasoning=reasoning,
                estimated_cost=best_model.estimated_cost,
                estimated_latency_ms=best_model.estimated_latency_ms,
                confidence_level=confidence,
                selection_duration_ms=duration
            )
            
            # Update statistics
            self.selection_stats["selections_made"] += 1
            self.selection_stats["selection_time_ms"] += duration
            
            return selection
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            raise LLMProviderError(f"Model selection failed: {e}")
    
    async def _score_model(
        self,
        provider: BaseLLMProvider,
        model: ModelInfo,
        request: LLMRequest,
        criteria: ModelSelectionCriteria
    ) -> ModelScore:
        """Score a model against selection criteria."""
        
        # Estimate request tokens
        estimated_input_tokens = provider.estimate_tokens(request.prompt)
        if request.system_prompt:
            estimated_input_tokens += provider.estimate_tokens(request.system_prompt)
        
        estimated_output_tokens = request.max_tokens or 512
        
        # Calculate cost
        estimated_cost = model.get_cost_per_request(estimated_input_tokens, estimated_output_tokens)
        
        # Score components
        cost_score = self._calculate_cost_score(estimated_cost, criteria)
        performance_score = self._calculate_performance_score(model, criteria)
        quality_score = self._calculate_quality_score(model, criteria)
        compliance_score = self._calculate_compliance_score(model, criteria)
        capability_score = self._calculate_capability_score(model, criteria)
        
        # Check if model meets requirements
        meets_requirements = self._check_requirements(
            model, estimated_cost, criteria, 
            cost_score, performance_score, quality_score, compliance_score, capability_score
        )
        
        # Calculate final score based on strategy
        final_score = self._calculate_final_score(
            cost_score, performance_score, quality_score,
            compliance_score, capability_score, criteria
        )
        
        scoring_details = {
            "cost_breakdown": {
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "cost_per_1k_input": model.metrics.cost_per_1k_input_tokens,
                "cost_per_1k_output": model.metrics.cost_per_1k_output_tokens
            },
            "performance_metrics": {
                "latency_ms": model.metrics.latency_ms,
                "throughput_rpm": model.metrics.throughput_rpm,
                "accuracy_score": model.metrics.accuracy_score
            },
            "quality_metrics": {
                "instruction_following": model.metrics.instruction_following,
                "factual_accuracy": model.metrics.factual_accuracy,
                "consistency": model.metrics.consistency
            }
        }
        
        return ModelScore(
            model_info=model,
            provider=provider,
            cost_score=cost_score,
            performance_score=performance_score,
            quality_score=quality_score,
            compliance_score=compliance_score,
            capability_score=capability_score,
            final_score=final_score,
            scoring_details=scoring_details,
            estimated_cost=estimated_cost,
            estimated_latency_ms=model.metrics.latency_ms,
            meets_requirements=meets_requirements
        )
    
    def _calculate_cost_score(self, estimated_cost: float, criteria: ModelSelectionCriteria) -> float:
        """Calculate cost score (higher is better = lower cost)."""
        
        if criteria.max_cost_per_request and estimated_cost > criteria.max_cost_per_request:
            return 0.0
        
        # Normalize cost score (inverse relationship)
        # Score based on cost relative to max budget or typical range
        max_reasonable_cost = criteria.max_cost_per_request or 1.0
        
        if estimated_cost <= 0:
            return 1.0  # Free/local models get max score
        
        # Inverse exponential scoring
        cost_ratio = estimated_cost / max_reasonable_cost
        return max(0.0, 1.0 - cost_ratio)
    
    def _calculate_performance_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Calculate performance score."""
        
        score = 0.0
        factors = 0
        
        # Latency score
        if criteria.max_latency_ms:
            if model.metrics.latency_ms <= criteria.max_latency_ms:
                latency_score = 1.0 - (model.metrics.latency_ms / criteria.max_latency_ms)
                score += max(0.0, latency_score)
            else:
                score += 0.0
            factors += 1
        else:
            # General latency scoring (lower is better)
            max_reasonable_latency = 10000  # 10 seconds
            latency_score = 1.0 - min(model.metrics.latency_ms / max_reasonable_latency, 1.0)
            score += latency_score
            factors += 1
        
        # Throughput score
        if criteria.min_throughput_rpm:
            if model.metrics.throughput_rpm >= criteria.min_throughput_rpm:
                score += 1.0
            else:
                score += model.metrics.throughput_rpm / criteria.min_throughput_rpm
            factors += 1
        else:
            # General throughput scoring (normalize to typical range)
            max_throughput = 100  # requests per minute
            throughput_score = min(model.metrics.throughput_rpm / max_throughput, 1.0)
            score += throughput_score
            factors += 1
        
        # Accuracy score
        if criteria.min_accuracy_score:
            if model.metrics.accuracy_score >= criteria.min_accuracy_score:
                score += 1.0
            else:
                score += model.metrics.accuracy_score / criteria.min_accuracy_score
            factors += 1
        else:
            score += model.metrics.accuracy_score
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_quality_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Calculate quality score."""
        
        score = 0.0
        factors = 0
        
        # Instruction following
        if criteria.min_instruction_following:
            if model.metrics.instruction_following >= criteria.min_instruction_following:
                score += 1.0
            else:
                score += model.metrics.instruction_following / criteria.min_instruction_following
        else:
            score += model.metrics.instruction_following
        factors += 1
        
        # Factual accuracy
        if criteria.min_factual_accuracy:
            if model.metrics.factual_accuracy >= criteria.min_factual_accuracy:
                score += 1.0
            else:
                score += model.metrics.factual_accuracy / criteria.min_factual_accuracy
        else:
            score += model.metrics.factual_accuracy
        factors += 1
        
        # Consistency
        if criteria.min_consistency:
            if model.metrics.consistency >= criteria.min_consistency:
                score += 1.0
            else:
                score += model.metrics.consistency / criteria.min_consistency
        else:
            score += model.metrics.consistency
        factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_compliance_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Calculate compliance score."""
        
        score = 1.0
        
        # Financial approval requirement
        if criteria.require_financial_approved and not model.is_financial_approved:
            return 0.0
        
        # GDPR compliance requirement
        if criteria.require_gdpr_compliant and not model.is_gdpr_compliant:
            return 0.0
        
        # Data retention requirements
        if criteria.max_data_retention_days is not None:
            if model.data_retention_days > criteria.max_data_retention_days:
                return 0.0
        
        # Bonus for better compliance
        if model.is_financial_approved:
            score += 0.1
        
        if model.data_retention_days == 0:  # No data retention
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_capability_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Calculate capability score."""
        
        required_caps = set(criteria.required_capabilities)
        available_caps = set(model.capabilities)
        
        if not required_caps.issubset(available_caps):
            return 0.0  # Missing required capabilities
        
        # Bonus for additional capabilities
        additional_caps = available_caps - required_caps
        bonus = min(len(additional_caps) * 0.1, 0.3)
        
        # Task complexity suitability
        complexity_suitable = model.is_suitable_for_complexity(criteria.min_complexity)
        complexity_score = 1.0 if complexity_suitable else 0.5
        
        return min(complexity_score + bonus, 1.0)
    
    def _check_requirements(
        self,
        model: ModelInfo,
        estimated_cost: float,
        criteria: ModelSelectionCriteria,
        cost_score: float,
        performance_score: float,
        quality_score: float,
        compliance_score: float,
        capability_score: float
    ) -> bool:
        """Check if model meets all hard requirements."""
        
        # Capability requirements
        if capability_score == 0.0:
            return False
        
        # Compliance requirements
        if compliance_score == 0.0:
            return False
        
        # Cost requirements
        if criteria.max_cost_per_request and estimated_cost > criteria.max_cost_per_request:
            return False
        
        # Performance requirements
        if criteria.max_latency_ms and model.metrics.latency_ms > criteria.max_latency_ms:
            return False
        
        if criteria.min_throughput_rpm and model.metrics.throughput_rpm < criteria.min_throughput_rpm:
            return False
        
        # Quality requirements
        if criteria.min_accuracy_score and model.metrics.accuracy_score < criteria.min_accuracy_score:
            return False
        
        return True
    
    def _calculate_final_score(
        self,
        cost_score: float,
        performance_score: float,
        quality_score: float,
        compliance_score: float,
        capability_score: float,
        criteria: ModelSelectionCriteria
    ) -> float:
        """Calculate weighted final score."""
        
        if criteria.selection_strategy == SelectionStrategy.COST_OPTIMIZED:
            weights = {"cost": 0.5, "performance": 0.2, "quality": 0.15, "compliance": 0.1, "capability": 0.05}
        
        elif criteria.selection_strategy == SelectionStrategy.PERFORMANCE_OPTIMIZED:
            weights = {"cost": 0.1, "performance": 0.5, "quality": 0.25, "compliance": 0.1, "capability": 0.05}
        
        elif criteria.selection_strategy == SelectionStrategy.QUALITY_FIRST:
            weights = {"cost": 0.15, "performance": 0.2, "quality": 0.5, "compliance": 0.1, "capability": 0.05}
        
        elif criteria.selection_strategy == SelectionStrategy.LATENCY_OPTIMIZED:
            weights = {"cost": 0.2, "performance": 0.5, "quality": 0.15, "compliance": 0.1, "capability": 0.05}
        
        elif criteria.selection_strategy == SelectionStrategy.CUSTOM_WEIGHTED:
            weights = criteria.custom_weights or {"cost": 0.3, "performance": 0.3, "quality": 0.2, "compliance": 0.1, "capability": 0.1}
        
        else:  # BALANCED
            weights = {"cost": 0.3, "performance": 0.25, "quality": 0.25, "compliance": 0.15, "capability": 0.05}
        
        final_score = (
            cost_score * weights["cost"] +
            performance_score * weights["performance"] +
            quality_score * weights["quality"] +
            compliance_score * weights["compliance"] +
            capability_score * weights["capability"]
        )
        
        return final_score
    
    def _select_best_model(self, scored_models: List[ModelScore], criteria: ModelSelectionCriteria) -> ModelScore:
        """Select the best model from scored candidates."""
        
        # Sort by final score
        scored_models.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply additional preferences
        if criteria.prefer_local:
            # Prefer local models if scores are close
            local_models = [m for m in scored_models if m.estimated_cost == 0.0]
            if local_models and len(scored_models) > 1:
                best_score = scored_models[0].final_score
                best_local = local_models[0]
                
                # If local model is within 10% of best score, prefer it
                if best_local.final_score >= best_score * 0.9:
                    return best_local
        
        return scored_models[0]
    
    def _build_reasoning(
        self,
        selected: ModelScore,
        alternatives: List[ModelScore],
        criteria: ModelSelectionCriteria
    ) -> List[str]:
        """Build human-readable reasoning for the selection."""
        
        reasoning = []
        
        reasoning.append(f"Selected {selected.model_info.model_id} with score {selected.final_score:.3f}")
        reasoning.append(f"Strategy: {criteria.selection_strategy.value}")
        
        # Cost reasoning
        if selected.estimated_cost == 0.0:
            reasoning.append("Selected model has no usage costs (local/free)")
        else:
            reasoning.append(f"Estimated cost: ${selected.estimated_cost:.4f} per request")
        
        # Performance reasoning
        reasoning.append(f"Expected latency: {selected.estimated_latency_ms:.0f}ms")
        
        # Quality reasoning
        if selected.model_info.metrics.accuracy_score > 0.9:
            reasoning.append("High accuracy model selected")
        
        # Compare to best alternative
        if alternatives:
            best_alt = alternatives[0]
            score_diff = selected.final_score - best_alt.final_score
            reasoning.append(f"Selected model outperformed best alternative by {score_diff:.3f} points")
        
        return reasoning
    
    def _calculate_confidence(self, selected: ModelScore, alternatives: List[ModelScore]) -> float:
        """Calculate confidence in the selection."""
        
        if not alternatives:
            return 0.95  # High confidence if only one option
        
        best_alt = alternatives[0]
        score_gap = selected.final_score - best_alt.final_score
        
        # Confidence based on score gap
        confidence = 0.5 + min(score_gap * 2, 0.45)  # 0.5 to 0.95 range
        
        # Boost confidence if selected model clearly better
        if selected.meets_requirements and not best_alt.meets_requirements:
            confidence += 0.1
        
        if selected.estimated_cost < best_alt.estimated_cost * 0.8:  # 20% cheaper
            confidence += 0.05
        
        return min(confidence, 0.99)


class ModelSelectionManager:
    """Main manager for model selection across multiple providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selector = CostOptimizedSelector(config.get("selector_config", {}))
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.selection_cache = {}
        self.cache_ttl_seconds = config.get("cache_ttl_seconds", 300)  # 5 minutes
    
    def add_provider(self, provider: BaseLLMProvider):
        """Add a provider to the selection pool."""
        self.providers[provider.get_provider_name()] = provider
    
    def remove_provider(self, provider_name: str):
        """Remove a provider from the selection pool."""
        if provider_name in self.providers:
            del self.providers[provider_name]
    
    async def select_optimal_model(
        self,
        request: LLMRequest,
        criteria: Optional[ModelSelectionCriteria] = None
    ) -> ModelSelection:
        """Select optimal model for the given request."""
        
        if not self.providers:
            raise LLMProviderError("No providers available for selection")
        
        # Use default criteria if none provided
        if criteria is None:
            criteria = self._create_default_criteria(request)
        
        # Check cache
        cache_key = self._generate_cache_key(request, criteria)
        cached_selection = self._get_cached_selection(cache_key)
        
        if cached_selection:
            logger.info("Using cached model selection")
            return cached_selection
        
        # Perform selection
        try:
            selection = await self.selector.select_model(
                list(self.providers.values()),
                request,
                criteria
            )
            
            # Cache the selection
            self._cache_selection(cache_key, selection)
            
            logger.info(f"Selected {selection.selected_model.model_id} from {selection.selected_provider}")
            
            return selection
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            raise
    
    def _create_default_criteria(self, request: LLMRequest) -> ModelSelectionCriteria:
        """Create default selection criteria based on request."""
        
        # Get task requirements
        task_requirements = get_task_requirements(request.task_type)
        
        return ModelSelectionCriteria(
            required_capabilities=task_requirements.get("required_capabilities", [ModelCapability.TEXT_GENERATION]),
            min_complexity=task_requirements.get("min_complexity", TaskComplexity.MODERATE),
            task_type=request.task_type,
            max_cost_per_request=request.max_cost,
            max_latency_ms=request.max_latency_ms,
            require_financial_approved=request.require_financial_approved,
            selection_strategy=SelectionStrategy.BALANCED
        )
    
    def _generate_cache_key(self, request: LLMRequest, criteria: ModelSelectionCriteria) -> str:
        """Generate cache key for selection."""
        
        import hashlib
        
        key_components = [
            request.task_type,
            str(request.complexity.value),
            str(criteria.selection_strategy.value),
            str(criteria.max_cost_per_request),
            str(criteria.max_latency_ms),
            str(criteria.require_financial_approved)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_selection(self, cache_key: str) -> Optional[ModelSelection]:
        """Get cached selection if still valid."""
        
        if cache_key not in self.selection_cache:
            return None
        
        cached_entry = self.selection_cache[cache_key]
        
        # Check if cache entry has expired
        age = datetime.utcnow() - cached_entry["timestamp"]
        if age.total_seconds() > self.cache_ttl_seconds:
            del self.selection_cache[cache_key]
            return None
        
        return cached_entry["selection"]
    
    def _cache_selection(self, cache_key: str, selection: ModelSelection):
        """Cache a selection result."""
        
        self.selection_cache[cache_key] = {
            "selection": selection,
            "timestamp": datetime.utcnow()
        }
        
        # Limit cache size
        if len(self.selection_cache) > 100:
            # Remove oldest entries
            sorted_entries = sorted(
                self.selection_cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            for key, _ in sorted_entries[:20]:  # Remove oldest 20
                del self.selection_cache[key]
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection performance statistics."""
        
        return {
            "selector_stats": self.selector.selection_stats,
            "providers_count": len(self.providers),
            "cache_entries": len(self.selection_cache),
            "available_providers": list(self.providers.keys())
        }
    
    async def validate_providers(self) -> Dict[str, bool]:
        """Validate health of all providers."""
        
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                health_status[name] = is_healthy
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                health_status[name] = False
        
        return health_status


# Factory functions
def create_cost_optimized_selector(config: Optional[Dict[str, Any]] = None) -> CostOptimizedSelector:
    """Create cost-optimized model selector."""
    if config is None:
        config = {}
    
    return CostOptimizedSelector(config)


def create_model_selection_manager(config: Optional[Dict[str, Any]] = None) -> ModelSelectionManager:
    """Create model selection manager."""
    if config is None:
        config = {}
    
    return ModelSelectionManager(config)


def create_financial_compliance_criteria(
    max_cost_per_request: float = 0.10,
    max_latency_ms: int = 5000
) -> ModelSelectionCriteria:
    """Create criteria optimized for financial compliance tasks."""
    
    return ModelSelectionCriteria(
        required_capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.ANALYSIS
        ],
        min_complexity=TaskComplexity.COMPLEX,
        task_type="regulatory_analysis",
        max_cost_per_request=max_cost_per_request,
        max_latency_ms=max_latency_ms,
        min_accuracy_score=0.85,
        min_instruction_following=0.90,
        require_financial_approved=True,
        require_gdpr_compliant=True,
        max_data_retention_days=0,
        selection_strategy=SelectionStrategy.QUALITY_FIRST,
        prefer_local=True
    )