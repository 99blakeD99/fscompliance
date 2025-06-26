"""Query result ranking and filtering for compliance search results."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from .processing import ProcessedQuery
from .routing import QueryType, QueryDomain, QueryComplexity

logger = logging.getLogger(__name__)


class RankingAlgorithm(str, Enum):
    """Types of ranking algorithms."""
    RELEVANCE_SCORE = "relevance_score"       # Pure relevance scoring
    AUTHORITY_WEIGHTED = "authority_weighted" # Authority-weighted relevance
    RECENCY_BOOSTED = "recency_boosted"      # Recency-boosted scoring
    CONTEXT_AWARE = "context_aware"          # Context-aware ranking
    MULTI_FACTOR = "multi_factor"            # Multi-factor composite ranking
    USER_PERSONALIZED = "user_personalized"  # User preference based


class FilterType(str, Enum):
    """Types of filters that can be applied."""
    CONTENT_TYPE = "content_type"           # Filter by content type
    SOURCE_AUTHORITY = "source_authority"   # Filter by source authority
    RECENCY = "recency"                    # Filter by date/recency
    RELEVANCE_THRESHOLD = "relevance_threshold"  # Minimum relevance score
    DOMAIN_SCOPE = "domain_scope"          # Domain/topic filtering
    FIRM_TYPE = "firm_type"               # Firm type applicability
    CONFIDENCE_LEVEL = "confidence_level"  # Confidence threshold
    LANGUAGE = "language"                  # Language filtering
    COMPLEXITY = "complexity"              # Complexity level filtering


class ContentType(str, Enum):
    """Types of content in search results."""
    REGULATION = "regulation"              # Primary regulations
    GUIDANCE = "guidance"                 # Regulatory guidance
    INTERPRETATION = "interpretation"      # Regulatory interpretations
    POLICY = "policy"                     # Internal policies
    PROCEDURE = "procedure"               # Procedures and processes
    CASE_STUDY = "case_study"            # Practical examples
    FAQ = "faq"                          # Frequently asked questions
    NEWS = "news"                        # Regulatory news/updates
    TEMPLATE = "template"                # Document templates
    CHECKLIST = "checklist"              # Compliance checklists


class SourceAuthority(str, Enum):
    """Authority levels for different sources."""
    PRIMARY_REGULATOR = "primary_regulator"    # FCA, PRA direct sources
    OFFICIAL_GUIDANCE = "official_guidance"    # Official regulatory guidance
    INDUSTRY_BODY = "industry_body"           # Trade association guidance
    PROFESSIONAL_FIRM = "professional_firm"   # Law/consulting firm guidance
    INTERNAL_POLICY = "internal_policy"       # Internal firm policies
    ACADEMIC = "academic"                     # Academic sources
    NEWS_MEDIA = "news_media"                # News and media
    COMMUNITY = "community"                   # Community contributions


@dataclass
class RankingFactor:
    """Individual factor contributing to ranking score."""
    
    factor_name: str
    raw_score: float       # 0.0 to 1.0
    weight: float          # 0.0 to 1.0
    normalized_score: float  # 0.0 to 1.0
    explanation: str
    
    @property
    def weighted_contribution(self) -> float:
        """Calculate weighted contribution to total score."""
        return self.normalized_score * self.weight


class SearchResult(BaseModel):
    """Enhanced search result with ranking and filtering metadata."""
    
    # Core result data
    result_id: str = Field(..., description="Unique result identifier")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content/summary")
    full_content: Optional[str] = Field(None, description="Full content if available")
    
    # Source information
    source_url: Optional[str] = Field(None, description="Source URL")
    source_title: str = Field(..., description="Source title/name")
    source_authority: SourceAuthority = Field(..., description="Authority level of source")
    content_type: ContentType = Field(..., description="Type of content")
    
    # Metadata
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    last_updated: Optional[datetime] = Field(None, description="Last update date")
    regulatory_source: Optional[str] = Field(None, description="Regulatory source reference")
    
    # Applicability
    applicable_firm_types: List[str] = Field(default_factory=list, description="Applicable firm types")
    regulatory_domains: List[str] = Field(default_factory=list, description="Regulatory domains covered")
    complexity_level: Optional[str] = Field(None, description="Content complexity level")
    
    # Ranking scores
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Base relevance score")
    authority_score: float = Field(0.0, ge=0.0, le=1.0, description="Source authority score")
    recency_score: float = Field(0.0, ge=0.0, le=1.0, description="Recency score")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Content quality score")
    
    # Computed ranking
    final_ranking_score: float = Field(0.0, ge=0.0, le=1.0, description="Final computed ranking score")
    ranking_factors: List[RankingFactor] = Field(default_factory=list, description="Factors used in ranking")
    ranking_algorithm: str = Field(..., description="Algorithm used for ranking")
    
    # Filtering metadata
    passes_filters: bool = Field(True, description="Whether result passes applied filters")
    filter_reasons: List[str] = Field(default_factory=list, description="Reasons if filtered out")
    
    # User interaction
    click_score: float = Field(0.0, ge=0.0, le=1.0, description="Historical click score")
    user_feedback_score: float = Field(0.0, ge=0.0, le=1.0, description="User feedback score")
    
    def get_ranking_summary(self) -> Dict[str, Any]:
        """Get summary of ranking information."""
        return {
            "final_score": self.final_ranking_score,
            "relevance": self.relevance_score,
            "authority": self.authority_score,
            "recency": self.recency_score,
            "quality": self.quality_score,
            "source_authority": self.source_authority.value,
            "content_type": self.content_type.value,
            "passes_filters": self.passes_filters
        }


class RankedSearchResults(BaseModel):
    """Collection of ranked and filtered search results."""
    
    # Query context
    query_id: str = Field(..., description="Original query ID")
    processed_query: ProcessedQuery = Field(..., description="Processed query details")
    
    # Results
    total_results_found: int = Field(..., description="Total results before filtering")
    results_after_filtering: int = Field(..., description="Results after filtering")
    results_returned: int = Field(..., description="Number of results returned")
    
    ranked_results: List[SearchResult] = Field(..., description="Ranked and filtered results")
    
    # Ranking configuration
    ranking_algorithm: RankingAlgorithm = Field(..., description="Ranking algorithm used")
    ranking_weights: Dict[str, float] = Field(..., description="Weights used in ranking")
    
    # Filtering configuration
    filters_applied: List[Dict[str, Any]] = Field(default_factory=list, description="Filters applied")
    filter_summary: Dict[str, int] = Field(default_factory=dict, description="Filter impact summary")
    
    # Performance metrics
    ranking_time_ms: float = Field(..., description="Time taken for ranking")
    filtering_time_ms: float = Field(..., description="Time taken for filtering")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    
    # Quality metrics
    average_relevance_score: float = Field(0.0, description="Average relevance of returned results")
    score_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of scores")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get high-level summary of results."""
        return {
            "total_found": self.total_results_found,
            "after_filtering": self.results_after_filtering,
            "returned": self.results_returned,
            "average_relevance": self.average_relevance_score,
            "top_score": max((r.final_ranking_score for r in self.ranked_results), default=0.0),
            "ranking_algorithm": self.ranking_algorithm.value,
            "filters_count": len(self.filters_applied)
        }
    
    def get_top_results(self, count: int = 5) -> List[SearchResult]:
        """Get top N ranked results."""
        return self.ranked_results[:count]


class BaseResultRanker(ABC):
    """Abstract base class for result ranking algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ranking_stats = {
            "results_ranked": 0,
            "ranking_sessions": 0,
            "average_ranking_time_ms": 0.0,
            "score_improvements": 0
        }
    
    @abstractmethod
    async def rank_results(
        self,
        results: List[SearchResult],
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Rank search results based on the algorithm."""
        pass
    
    @abstractmethod
    def get_ranker_name(self) -> str:
        """Get the name of the ranking algorithm."""
        pass
    
    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize score to 0-1 range."""
        if max_val == min_val:
            return 0.5  # Default if no variance
        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


class MultiFactorRanker(BaseResultRanker):
    """Multi-factor ranking algorithm considering multiple relevance signals."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ranking_weights = self._initialize_ranking_weights()
        self.authority_scores = self._initialize_authority_scores()
    
    def get_ranker_name(self) -> str:
        return "multi_factor_ranker"
    
    def _initialize_ranking_weights(self) -> Dict[str, float]:
        """Initialize weights for different ranking factors."""
        
        return {
            "relevance": 0.35,      # Content relevance to query
            "authority": 0.25,      # Source authority/credibility
            "recency": 0.15,        # How recent the content is
            "quality": 0.15,        # Content quality indicators
            "user_feedback": 0.05,  # Historical user feedback
            "completeness": 0.05    # Content completeness
        }
    
    def _initialize_authority_scores(self) -> Dict[SourceAuthority, float]:
        """Initialize authority scores for different source types."""
        
        return {
            SourceAuthority.PRIMARY_REGULATOR: 1.0,
            SourceAuthority.OFFICIAL_GUIDANCE: 0.9,
            SourceAuthority.INDUSTRY_BODY: 0.7,
            SourceAuthority.PROFESSIONAL_FIRM: 0.6,
            SourceAuthority.INTERNAL_POLICY: 0.5,
            SourceAuthority.ACADEMIC: 0.4,
            SourceAuthority.NEWS_MEDIA: 0.3,
            SourceAuthority.COMMUNITY: 0.2
        }
    
    async def rank_results(
        self,
        results: List[SearchResult],
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Rank results using multi-factor algorithm."""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Ranking {len(results)} results using multi-factor algorithm")
            
            if not results:
                return results
            
            # Calculate individual factor scores for all results
            for result in results:
                await self._calculate_factor_scores(result, processed_query, context)
            
            # Calculate final ranking scores
            for result in results:
                result.final_ranking_score = self._calculate_final_score(result)
                result.ranking_algorithm = self.get_ranker_name()
            
            # Sort by final ranking score
            ranked_results = sorted(results, key=lambda r: r.final_ranking_score, reverse=True)
            
            # Update statistics
            end_time = datetime.utcnow()
            ranking_time = (end_time - start_time).total_seconds() * 1000
            
            self.ranking_stats["results_ranked"] += len(results)
            self.ranking_stats["ranking_sessions"] += 1
            
            # Update average ranking time
            total_time = (self.ranking_stats["average_ranking_time_ms"] * 
                         (self.ranking_stats["ranking_sessions"] - 1) + ranking_time)
            self.ranking_stats["average_ranking_time_ms"] = total_time / self.ranking_stats["ranking_sessions"]
            
            logger.info(f"Ranking completed: top score {ranked_results[0].final_ranking_score:.3f}")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            raise
    
    async def _calculate_factor_scores(
        self,
        result: SearchResult,
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]]
    ):
        """Calculate individual factor scores for a result."""
        
        factors = []
        
        # 1. Relevance score
        relevance_score = self._calculate_relevance_score(result, processed_query)
        factors.append(RankingFactor(
            factor_name="relevance",
            raw_score=relevance_score,
            weight=self.ranking_weights["relevance"],
            normalized_score=relevance_score,
            explanation="Content relevance to query terms and intent"
        ))
        
        # 2. Authority score
        authority_score = self._calculate_authority_score(result)
        factors.append(RankingFactor(
            factor_name="authority",
            raw_score=authority_score,
            weight=self.ranking_weights["authority"],
            normalized_score=authority_score,
            explanation="Source credibility and authority level"
        ))
        
        # 3. Recency score
        recency_score = self._calculate_recency_score(result)
        factors.append(RankingFactor(
            factor_name="recency",
            raw_score=recency_score,
            weight=self.ranking_weights["recency"],
            normalized_score=recency_score,
            explanation="How recent the content is"
        ))
        
        # 4. Quality score
        quality_score = self._calculate_quality_score(result, processed_query)
        factors.append(RankingFactor(
            factor_name="quality",
            raw_score=quality_score,
            weight=self.ranking_weights["quality"],
            normalized_score=quality_score,
            explanation="Content quality and completeness indicators"
        ))
        
        # 5. User feedback score
        user_feedback_score = result.user_feedback_score
        factors.append(RankingFactor(
            factor_name="user_feedback",
            raw_score=user_feedback_score,
            weight=self.ranking_weights["user_feedback"],
            normalized_score=user_feedback_score,
            explanation="Historical user feedback and ratings"
        ))
        
        # 6. Completeness score
        completeness_score = self._calculate_completeness_score(result, processed_query)
        factors.append(RankingFactor(
            factor_name="completeness",
            raw_score=completeness_score,
            weight=self.ranking_weights["completeness"],
            normalized_score=completeness_score,
            explanation="Content completeness relative to query requirements"
        ))
        
        # Store factor scores in result
        result.relevance_score = relevance_score
        result.authority_score = authority_score
        result.recency_score = recency_score
        result.quality_score = quality_score
        result.ranking_factors = factors
    
    def _calculate_relevance_score(self, result: SearchResult, processed_query: ProcessedQuery) -> float:
        """Calculate relevance score based on query match."""
        
        score = 0.0
        
        # Text matching score
        query_terms = processed_query.search_terms
        result_text = f"{result.title} {result.content}".lower()
        
        # Exact term matches
        exact_matches = sum(1 for term in query_terms if term.lower() in result_text)
        if query_terms:
            exact_match_score = exact_matches / len(query_terms)
            score += exact_match_score * 0.4
        
        # Domain relevance
        query_domain = processed_query.classified_query.query_domain.value
        if query_domain in result.regulatory_domains:
            score += 0.3
        elif any(domain in result.regulatory_domains for domain in [query_domain]):
            score += 0.15
        
        # Content type relevance
        query_type = processed_query.classified_query.query_type
        content_type_relevance = self._get_content_type_relevance(query_type, result.content_type)
        score += content_type_relevance * 0.3
        
        return min(score, 1.0)
    
    def _calculate_authority_score(self, result: SearchResult) -> float:
        """Calculate authority score based on source credibility."""
        
        base_authority = self.authority_scores.get(result.source_authority, 0.5)
        
        # Boost for regulatory sources
        if result.regulatory_source:
            base_authority = min(base_authority + 0.1, 1.0)
        
        # Boost for official content types
        if result.content_type in [ContentType.REGULATION, ContentType.GUIDANCE]:
            base_authority = min(base_authority + 0.05, 1.0)
        
        return base_authority
    
    def _calculate_recency_score(self, result: SearchResult) -> float:
        """Calculate recency score based on publication/update date."""
        
        if not result.last_updated and not result.publication_date:
            return 0.5  # Default for unknown dates
        
        reference_date = result.last_updated or result.publication_date
        now = datetime.utcnow()
        
        if reference_date.tzinfo:
            reference_date = reference_date.replace(tzinfo=None)
        
        days_old = (now - reference_date).days
        
        # Scoring curve: 1.0 for content < 30 days, declining over time
        if days_old <= 30:
            return 1.0
        elif days_old <= 90:
            return 0.9
        elif days_old <= 180:
            return 0.8
        elif days_old <= 365:
            return 0.6
        elif days_old <= 730:  # 2 years
            return 0.4
        else:
            return 0.2
    
    def _calculate_quality_score(self, result: SearchResult, processed_query: ProcessedQuery) -> float:
        """Calculate quality score based on content indicators."""
        
        score = 0.5  # Base quality score
        
        # Content length indicator
        content_length = len(result.content)
        if content_length > 200:
            score += 0.2
        elif content_length > 100:
            score += 0.1
        
        # Title quality
        title_length = len(result.title)
        if 10 <= title_length <= 100:
            score += 0.1
        
        # Full content availability
        if result.full_content:
            score += 0.1
        
        # Regulatory source citation
        if result.regulatory_source:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_completeness_score(self, result: SearchResult, processed_query: ProcessedQuery) -> float:
        """Calculate completeness score relative to query requirements."""
        
        score = 0.5  # Base completeness
        
        # Query element coverage
        query_elements = len(processed_query.query_elements)
        if query_elements > 0:
            # Simple heuristic: more content suggests better coverage
            content_indicators = [
                bool(result.full_content),
                bool(result.regulatory_source),
                len(result.content) > 300,
                len(result.applicable_firm_types) > 0
            ]
            coverage_ratio = sum(content_indicators) / 4
            score += coverage_ratio * 0.5
        
        return min(score, 1.0)
    
    def _calculate_final_score(self, result: SearchResult) -> float:
        """Calculate final weighted ranking score."""
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor in result.ranking_factors:
            total_score += factor.weighted_contribution
            total_weight += factor.weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _get_content_type_relevance(self, query_type: QueryType, content_type: ContentType) -> float:
        """Get relevance score for content type given query type."""
        
        relevance_matrix = {
            QueryType.REQUIREMENT_LOOKUP: {
                ContentType.REGULATION: 1.0,
                ContentType.GUIDANCE: 0.8,
                ContentType.POLICY: 0.6,
                ContentType.FAQ: 0.5
            },
            QueryType.GUIDANCE_REQUEST: {
                ContentType.GUIDANCE: 1.0,
                ContentType.INTERPRETATION: 0.9,
                ContentType.CASE_STUDY: 0.7,
                ContentType.TEMPLATE: 0.6
            },
            QueryType.GAP_ANALYSIS: {
                ContentType.CHECKLIST: 1.0,
                ContentType.PROCEDURE: 0.8,
                ContentType.TEMPLATE: 0.7,
                ContentType.CASE_STUDY: 0.6
            },
            QueryType.INTERPRETATION: {
                ContentType.INTERPRETATION: 1.0,
                ContentType.GUIDANCE: 0.8,
                ContentType.CASE_STUDY: 0.7,
                ContentType.FAQ: 0.6
            }
        }
        
        return relevance_matrix.get(query_type, {}).get(content_type, 0.3)


class BaseResultFilter(ABC):
    """Abstract base class for result filtering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def filter_results(
        self,
        results: List[SearchResult],
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Filter search results based on criteria."""
        pass
    
    @abstractmethod
    def get_filter_name(self) -> str:
        """Get the name of the filter."""
        pass


class ComprehensiveResultFilter(BaseResultFilter):
    """Comprehensive result filtering with multiple criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.filter_thresholds = self._initialize_filter_thresholds()
    
    def get_filter_name(self) -> str:
        return "comprehensive_filter"
    
    def _initialize_filter_thresholds(self) -> Dict[str, Any]:
        """Initialize filtering thresholds and criteria."""
        
        return {
            "min_relevance_score": 0.1,
            "min_authority_score": 0.0,
            "max_age_days": 1825,  # 5 years
            "min_content_length": 20,
            "exclude_content_types": [],
            "exclude_source_authorities": [SourceAuthority.COMMUNITY],
            "require_regulatory_scope": False
        }
    
    async def filter_results(
        self,
        results: List[SearchResult],
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Apply comprehensive filtering to results."""
        
        if not results:
            return results
        
        filtered_results = []
        
        for result in results:
            passes_filters = True
            filter_reasons = []
            
            # 1. Relevance threshold filter
            if result.relevance_score < self.filter_thresholds["min_relevance_score"]:
                passes_filters = False
                filter_reasons.append(f"Relevance score too low: {result.relevance_score:.2f}")
            
            # 2. Authority threshold filter
            if result.authority_score < self.filter_thresholds["min_authority_score"]:
                passes_filters = False
                filter_reasons.append(f"Authority score too low: {result.authority_score:.2f}")
            
            # 3. Content age filter
            if result.publication_date or result.last_updated:
                ref_date = result.last_updated or result.publication_date
                if ref_date.tzinfo:
                    ref_date = ref_date.replace(tzinfo=None)
                
                age_days = (datetime.utcnow() - ref_date).days
                if age_days > self.filter_thresholds["max_age_days"]:
                    passes_filters = False
                    filter_reasons.append(f"Content too old: {age_days} days")
            
            # 4. Content length filter
            if len(result.content) < self.filter_thresholds["min_content_length"]:
                passes_filters = False
                filter_reasons.append(f"Content too short: {len(result.content)} chars")
            
            # 5. Content type exclusions
            if result.content_type in self.filter_thresholds["exclude_content_types"]:
                passes_filters = False
                filter_reasons.append(f"Excluded content type: {result.content_type.value}")
            
            # 6. Source authority exclusions
            if result.source_authority in self.filter_thresholds["exclude_source_authorities"]:
                passes_filters = False
                filter_reasons.append(f"Excluded source authority: {result.source_authority.value}")
            
            # 7. Regulatory scope requirement
            if self.filter_thresholds["require_regulatory_scope"]:
                query_domain = processed_query.classified_query.query_domain.value
                if query_domain not in result.regulatory_domains:
                    passes_filters = False
                    filter_reasons.append(f"Missing required regulatory scope: {query_domain}")
            
            # 8. Context-based filtering
            if context:
                context_passes, context_reasons = self._apply_context_filters(result, context)
                if not context_passes:
                    passes_filters = False
                    filter_reasons.extend(context_reasons)
            
            # Update result metadata
            result.passes_filters = passes_filters
            result.filter_reasons = filter_reasons
            
            # Add to filtered results if it passes
            if passes_filters:
                filtered_results.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)}")
        
        return filtered_results
    
    def _apply_context_filters(
        self, 
        result: SearchResult, 
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Apply context-specific filters."""
        
        passes = True
        reasons = []
        
        # Firm type filtering
        if context.get("firm_type") and result.applicable_firm_types:
            user_firm_type = context["firm_type"].lower()
            applicable_types = [ft.lower() for ft in result.applicable_firm_types]
            
            if not any(user_firm_type in ft for ft in applicable_types):
                passes = False
                reasons.append(f"Not applicable to firm type: {user_firm_type}")
        
        # User role filtering
        if context.get("user_role"):
            user_role = context["user_role"].lower()
            
            # Filter out overly technical content for non-technical roles
            if "technical" not in user_role and result.complexity_level == "expert":
                passes = False
                reasons.append("Content too technical for user role")
        
        # Language filtering
        if context.get("preferred_language", "english") != "english":
            # For now, assume all content is English
            # Future implementation could check content language
            pass
        
        return passes, reasons


class RankingAndFilteringPipeline:
    """Pipeline for orchestrating result ranking and filtering."""
    
    def __init__(
        self,
        rankers: List[BaseResultRanker],
        filters: List[BaseResultFilter]
    ):
        self.rankers = rankers
        self.filters = filters
        self.pipeline_stats = {
            "total_results_processed": 0,
            "total_sessions": 0,
            "average_filtering_ratio": 0.0,
            "average_processing_time_ms": 0.0
        }
    
    async def process_results(
        self,
        results: List[SearchResult],
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 20
    ) -> RankedSearchResults:
        """Process results through ranking and filtering pipeline."""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing {len(results)} results through ranking and filtering pipeline")
            
            total_results = len(results)
            
            # Step 1: Apply filters
            filtering_start = datetime.utcnow()
            filtered_results = results.copy()
            
            filters_applied = []
            for filter_instance in self.filters:
                filter_config = {
                    "filter_name": filter_instance.get_filter_name(),
                    "results_before": len(filtered_results)
                }
                
                filtered_results = await filter_instance.filter_results(
                    filtered_results, processed_query, context
                )
                
                filter_config["results_after"] = len(filtered_results)
                filter_config["filtered_out"] = filter_config["results_before"] - filter_config["results_after"]
                filters_applied.append(filter_config)
            
            filtering_end = datetime.utcnow()
            filtering_time = (filtering_end - filtering_start).total_seconds() * 1000
            
            # Step 2: Apply ranking
            ranking_start = datetime.utcnow()
            ranked_results = filtered_results.copy()
            
            ranking_weights = {}
            for ranker in self.rankers:
                ranked_results = await ranker.rank_results(ranked_results, processed_query, context)
                if hasattr(ranker, 'ranking_weights'):
                    ranking_weights.update(ranker.ranking_weights)
            
            ranking_end = datetime.utcnow()
            ranking_time = (ranking_end - ranking_start).total_seconds() * 1000
            
            # Step 3: Limit results
            final_results = ranked_results[:max_results]
            
            # Calculate metrics
            avg_relevance = sum(r.relevance_score for r in final_results) / len(final_results) if final_results else 0.0
            
            score_distribution = self._calculate_score_distribution(final_results)
            filter_summary = self._calculate_filter_summary(filters_applied)
            
            # Create result object
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds() * 1000
            
            ranked_search_results = RankedSearchResults(
                query_id=processed_query.query_id,
                processed_query=processed_query,
                total_results_found=total_results,
                results_after_filtering=len(filtered_results),
                results_returned=len(final_results),
                ranked_results=final_results,
                ranking_algorithm=RankingAlgorithm.MULTI_FACTOR if self.rankers else RankingAlgorithm.RELEVANCE_SCORE,
                ranking_weights=ranking_weights,
                filters_applied=filters_applied,
                filter_summary=filter_summary,
                ranking_time_ms=ranking_time,
                filtering_time_ms=filtering_time,
                total_processing_time_ms=total_time,
                average_relevance_score=avg_relevance,
                score_distribution=score_distribution
            )
            
            # Update statistics
            self.pipeline_stats["total_results_processed"] += total_results
            self.pipeline_stats["total_sessions"] += 1
            
            filtering_ratio = len(filtered_results) / total_results if total_results > 0 else 1.0
            total_filtering_ratio = (
                self.pipeline_stats["average_filtering_ratio"] * (self.pipeline_stats["total_sessions"] - 1) +
                filtering_ratio
            )
            self.pipeline_stats["average_filtering_ratio"] = total_filtering_ratio / self.pipeline_stats["total_sessions"]
            
            total_proc_time = (
                self.pipeline_stats["average_processing_time_ms"] * (self.pipeline_stats["total_sessions"] - 1) +
                total_time
            )
            self.pipeline_stats["average_processing_time_ms"] = total_proc_time / self.pipeline_stats["total_sessions"]
            
            logger.info(f"Pipeline completed: {len(final_results)} results returned with avg relevance {avg_relevance:.2f}")
            
            return ranked_search_results
            
        except Exception as e:
            logger.error(f"Error in ranking and filtering pipeline: {e}")
            raise
    
    def _calculate_score_distribution(self, results: List[SearchResult]) -> Dict[str, int]:
        """Calculate distribution of ranking scores."""
        
        distribution = {
            "0.9-1.0": 0,
            "0.8-0.9": 0,
            "0.7-0.8": 0,
            "0.6-0.7": 0,
            "0.5-0.6": 0,
            "0.0-0.5": 0
        }
        
        for result in results:
            score = result.final_ranking_score
            
            if score >= 0.9:
                distribution["0.9-1.0"] += 1
            elif score >= 0.8:
                distribution["0.8-0.9"] += 1
            elif score >= 0.7:
                distribution["0.7-0.8"] += 1
            elif score >= 0.6:
                distribution["0.6-0.7"] += 1
            elif score >= 0.5:
                distribution["0.5-0.6"] += 1
            else:
                distribution["0.0-0.5"] += 1
        
        return distribution
    
    def _calculate_filter_summary(self, filters_applied: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate summary of filtering impact."""
        
        summary = {}
        
        for filter_config in filters_applied:
            filter_name = filter_config["filter_name"]
            filtered_out = filter_config["filtered_out"]
            summary[filter_name] = filtered_out
        
        return summary
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        ranker_stats = {}
        for ranker in self.rankers:
            ranker_stats[ranker.get_ranker_name()] = ranker.ranking_stats
        
        return {
            **self.pipeline_stats,
            "ranker_statistics": ranker_stats,
            "ranker_count": len(self.rankers),
            "filter_count": len(self.filters)
        }


# Factory functions
def create_multi_factor_ranker(config: Optional[Dict[str, Any]] = None) -> MultiFactorRanker:
    """Create multi-factor ranking algorithm."""
    if config is None:
        config = {}
    
    return MultiFactorRanker(config)


def create_comprehensive_filter(config: Optional[Dict[str, Any]] = None) -> ComprehensiveResultFilter:
    """Create comprehensive result filter."""
    if config is None:
        config = {}
    
    return ComprehensiveResultFilter(config)


def create_ranking_filtering_pipeline(
    rankers: Optional[List[str]] = None,
    filters: Optional[List[str]] = None
) -> RankingAndFilteringPipeline:
    """Create ranking and filtering pipeline with specified components."""
    
    if rankers is None:
        rankers = ["multi_factor"]
    
    if filters is None:
        filters = ["comprehensive"]
    
    ranker_instances = []
    for ranker_name in rankers:
        if ranker_name == "multi_factor":
            ranker_instances.append(create_multi_factor_ranker())
        else:
            logger.warning(f"Ranker '{ranker_name}' not implemented, skipping")
    
    filter_instances = []
    for filter_name in filters:
        if filter_name == "comprehensive":
            filter_instances.append(create_comprehensive_filter())
        else:
            logger.warning(f"Filter '{filter_name}' not implemented, skipping")
    
    return RankingAndFilteringPipeline(ranker_instances, filter_instances)