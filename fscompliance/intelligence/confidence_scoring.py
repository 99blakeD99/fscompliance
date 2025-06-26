"""Confidence scoring for compliance recommendations and analysis results."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from .requirement_extraction import ExtractedRequirementEnhanced
from .gap_detection import ComplianceGap
from .requirement_categorization import CategorizedRequirement
from .compliance_scoring import ComplianceScore

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"      # 90-100%
    HIGH = "high"               # 75-89%
    MEDIUM = "medium"           # 60-74%
    LOW = "low"                # 40-59%
    VERY_LOW = "very_low"       # 0-39%


class ConfidenceFactorType(str, Enum):
    """Types of factors affecting confidence."""
    DATA_QUALITY = "data_quality"          # Quality of input data
    ALGORITHM_RELIABILITY = "algorithm_reliability"  # Algorithm performance
    DOMAIN_COVERAGE = "domain_coverage"    # Coverage of regulatory domain
    VALIDATION_STATUS = "validation_status"  # External validation
    TEMPORAL_STABILITY = "temporal_stability"  # Stability over time
    CROSS_VALIDATION = "cross_validation"   # Cross-validation results
    EXPERT_REVIEW = "expert_review"        # Expert validation
    SAMPLE_SIZE = "sample_size"            # Size of data sample
    COMPLEXITY_FACTOR = "complexity_factor"  # Complexity of analysis


class RecommendationType(str, Enum):
    """Types of recommendations that can be scored."""
    GAP_REMEDIATION = "gap_remediation"    # Gap remediation actions
    COMPLIANCE_IMPROVEMENT = "compliance_improvement"  # General improvements
    RISK_MITIGATION = "risk_mitigation"   # Risk reduction actions
    PROCESS_ENHANCEMENT = "process_enhancement"  # Process improvements
    POLICY_UPDATE = "policy_update"       # Policy changes
    TRAINING_REQUIREMENT = "training_requirement"  # Training needs
    MONITORING_ENHANCEMENT = "monitoring_enhancement"  # Monitoring improvements
    RESOURCE_ALLOCATION = "resource_allocation"  # Resource decisions


@dataclass
class ConfidenceFactor:
    """Individual factor contributing to confidence scoring."""
    
    factor_type: ConfidenceFactorType
    factor_name: str
    score: float  # 0.0 to 1.0
    weight: float  # 0.0 to 1.0
    evidence: List[str]
    explanation: str
    
    @property
    def weighted_contribution(self) -> float:
        """Calculate weighted contribution to overall confidence."""
        return self.score * self.weight


class ConfidenceAssessment(BaseModel):
    """Comprehensive confidence assessment for recommendations."""
    
    # Core assessment data
    assessment_id: str = Field(..., description="Unique assessment identifier")
    target_id: str = Field(..., description="ID of target being assessed")
    target_type: str = Field(..., description="Type of target (gap, score, recommendation)")
    recommendation_type: Optional[RecommendationType] = Field(None, description="Type of recommendation")
    
    # Confidence metrics
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level category")
    
    # Factor breakdown
    confidence_factors: List[ConfidenceFactor] = Field(
        default_factory=list, description="Individual confidence factors"
    )
    
    # Assessment metadata
    assessment_algorithm: str = Field(..., description="Algorithm used for assessment")
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Reliability indicators
    data_completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness of underlying data")
    method_validation: float = Field(..., ge=0.0, le=1.0, description="Validation status of methods used")
    domain_expertise: float = Field(..., ge=0.0, le=1.0, description="Domain expertise coverage")
    
    # Uncertainty measures
    confidence_interval_lower: float = Field(..., ge=0.0, le=1.0, description="Lower bound of confidence interval")
    confidence_interval_upper: float = Field(..., ge=0.0, le=1.0, description="Upper bound of confidence interval")
    uncertainty_factors: List[str] = Field(default_factory=list, description="Key uncertainty factors")
    
    # Actionability assessment
    actionability_score: float = Field(..., ge=0.0, le=1.0, description="How actionable the recommendation is")
    implementation_feasibility: float = Field(..., ge=0.0, le=1.0, description="Implementation feasibility")
    resource_availability: float = Field(..., ge=0.0, le=1.0, description="Resource availability for implementation")
    
    # Quality indicators
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Consistency with other findings")
    logical_coherence: float = Field(..., ge=0.0, le=1.0, description="Logical coherence of recommendation")
    evidence_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of supporting evidence")
    
    # Risk assessment
    confidence_risk_level: str = Field(..., description="Risk level of acting on this confidence")
    false_positive_risk: float = Field(..., ge=0.0, le=1.0, description="Risk of false positive")
    false_negative_risk: float = Field(..., ge=0.0, le=1.0, description="Risk of false negative")
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get high-level confidence summary."""
        return {
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level.value,
            "data_completeness": self.data_completeness,
            "actionability_score": self.actionability_score,
            "evidence_strength": self.evidence_strength,
            "confidence_interval": [self.confidence_interval_lower, self.confidence_interval_upper],
            "key_risk_factors": self.uncertainty_factors[:3]
        }
    
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence assessment."""
        return self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
    
    def get_recommendation_quality(self) -> float:
        """Calculate overall recommendation quality score."""
        return (
            self.overall_confidence * 0.4 +
            self.actionability_score * 0.3 +
            self.evidence_strength * 0.3
        )


class BaseConfidenceScorer(ABC):
    """Abstract base class for confidence scoring algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scoring_stats = {
            "assessments_generated": 0,
            "high_confidence_assessments": 0,
            "processing_errors": 0,
            "average_confidence": 0.0
        }
    
    @abstractmethod
    async def assess_gap_confidence(self, gap: ComplianceGap) -> ConfidenceAssessment:
        """Assess confidence in a compliance gap identification."""
        pass
    
    @abstractmethod
    async def assess_score_confidence(self, score: ComplianceScore) -> ConfidenceAssessment:
        """Assess confidence in a compliance score."""
        pass
    
    @abstractmethod
    async def assess_recommendation_confidence(
        self, 
        recommendation: str,
        recommendation_type: RecommendationType,
        supporting_evidence: List[str]
    ) -> ConfidenceAssessment:
        """Assess confidence in a specific recommendation."""
        pass
    
    @abstractmethod
    def get_scorer_name(self) -> str:
        """Get the name of the confidence scoring algorithm."""
        pass
    
    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from numeric score."""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_confidence_interval(
        self, 
        point_estimate: float, 
        uncertainty: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval around point estimate."""
        
        # Simple approach using uncertainty as standard error
        margin_of_error = 1.96 * uncertainty  # 95% confidence interval
        
        lower_bound = max(0.0, point_estimate - margin_of_error)
        upper_bound = min(1.0, point_estimate + margin_of_error)
        
        return lower_bound, upper_bound
    
    def _generate_assessment_id(self, target_id: str) -> str:
        """Generate unique assessment identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"CONF_{target_id}_{timestamp}"


class ComprehensiveConfidenceScorer(BaseConfidenceScorer):
    """Comprehensive confidence scoring using multiple factors and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.factor_weights = self._initialize_factor_weights()
    
    def get_scorer_name(self) -> str:
        return "comprehensive_confidence_scorer"
    
    def _initialize_factor_weights(self) -> Dict[ConfidenceFactorType, float]:
        """Initialize weights for different confidence factors."""
        
        return {
            ConfidenceFactorType.DATA_QUALITY: 0.20,
            ConfidenceFactorType.ALGORITHM_RELIABILITY: 0.18,
            ConfidenceFactorType.DOMAIN_COVERAGE: 0.15,
            ConfidenceFactorType.VALIDATION_STATUS: 0.12,
            ConfidenceFactorType.TEMPORAL_STABILITY: 0.10,
            ConfidenceFactorType.CROSS_VALIDATION: 0.08,
            ConfidenceFactorType.EXPERT_REVIEW: 0.07,
            ConfidenceFactorType.SAMPLE_SIZE: 0.06,
            ConfidenceFactorType.COMPLEXITY_FACTOR: 0.04
        }
    
    async def assess_gap_confidence(self, gap: ComplianceGap) -> ConfidenceAssessment:
        """Assess confidence in a compliance gap identification."""
        
        logger.debug(f"Assessing confidence for gap {gap.gap_id}")
        
        try:
            # Calculate confidence factors for gap
            factors = []
            
            # Data quality factor
            data_quality_score = gap.detection_confidence
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.DATA_QUALITY,
                factor_name="Gap Detection Data Quality",
                score=data_quality_score,
                weight=self.factor_weights[ConfidenceFactorType.DATA_QUALITY],
                evidence=[f"Detection confidence: {data_quality_score:.2f}"],
                explanation=f"Gap detection algorithm confidence of {data_quality_score:.1%}"
            ))
            
            # Algorithm reliability factor
            algorithm_reliability = self._assess_gap_algorithm_reliability(gap)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.ALGORITHM_RELIABILITY,
                factor_name="Gap Detection Algorithm Reliability",
                score=algorithm_reliability,
                weight=self.factor_weights[ConfidenceFactorType.ALGORITHM_RELIABILITY],
                evidence=[f"Algorithm: {gap.detection_algorithm}"],
                explanation="Reliability based on algorithm maturity and validation"
            ))
            
            # Domain coverage factor
            domain_coverage = self._assess_gap_domain_coverage(gap)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.DOMAIN_COVERAGE,
                factor_name="Regulatory Domain Coverage",
                score=domain_coverage,
                weight=self.factor_weights[ConfidenceFactorType.DOMAIN_COVERAGE],
                evidence=[f"Source: {gap.regulatory_source}"],
                explanation="Coverage of the specific regulatory domain"
            ))
            
            # Evidence strength factor
            evidence_strength = self._assess_gap_evidence_strength(gap)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.VALIDATION_STATUS,
                factor_name="Evidence Strength",
                score=evidence_strength,
                weight=self.factor_weights[ConfidenceFactorType.VALIDATION_STATUS],
                evidence=[gap.evidence],
                explanation="Strength and quality of supporting evidence"
            ))
            
            # Calculate overall confidence
            overall_confidence = sum(factor.weighted_contribution for factor in factors)
            
            # Calculate uncertainty measures
            uncertainty = self._calculate_gap_uncertainty(gap, factors)
            conf_lower, conf_upper = self._calculate_confidence_interval(overall_confidence, uncertainty)
            
            # Assess actionability
            actionability = self._assess_gap_actionability(gap)
            
            # Create confidence assessment
            assessment = ConfidenceAssessment(
                assessment_id=self._generate_assessment_id(gap.gap_id),
                target_id=gap.gap_id,
                target_type="compliance_gap",
                recommendation_type=RecommendationType.GAP_REMEDIATION,
                overall_confidence=overall_confidence,
                confidence_level=self._determine_confidence_level(overall_confidence),
                confidence_factors=factors,
                assessment_algorithm=self.get_scorer_name(),
                data_completeness=data_quality_score,
                method_validation=algorithm_reliability,
                domain_expertise=domain_coverage,
                confidence_interval_lower=conf_lower,
                confidence_interval_upper=conf_upper,
                uncertainty_factors=self._identify_gap_uncertainty_factors(gap),
                actionability_score=actionability,
                implementation_feasibility=self._assess_gap_implementation_feasibility(gap),
                resource_availability=0.7,  # Default assumption
                consistency_score=self._assess_gap_consistency(gap),
                logical_coherence=self._assess_gap_logical_coherence(gap),
                evidence_strength=evidence_strength,
                confidence_risk_level=self._assess_confidence_risk_level(overall_confidence),
                false_positive_risk=self._assess_false_positive_risk(gap),
                false_negative_risk=self._assess_false_negative_risk(gap)
            )
            
            # Update statistics
            self.scoring_stats["assessments_generated"] += 1
            if assessment.is_high_confidence():
                self.scoring_stats["high_confidence_assessments"] += 1
            
            return assessment
            
        except Exception as e:
            self.scoring_stats["processing_errors"] += 1
            logger.error(f"Error assessing gap confidence: {e}")
            raise
    
    async def assess_score_confidence(self, score: ComplianceScore) -> ConfidenceAssessment:
        """Assess confidence in a compliance score."""
        
        logger.debug(f"Assessing confidence for score {score.score_id}")
        
        try:
            # Calculate confidence factors for score
            factors = []
            
            # Data quality factor
            data_quality = score.data_completeness
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.DATA_QUALITY,
                factor_name="Score Data Completeness",
                score=data_quality,
                weight=self.factor_weights[ConfidenceFactorType.DATA_QUALITY],
                evidence=[f"Data completeness: {data_quality:.2f}"],
                explanation=f"Completeness of data used in scoring: {data_quality:.1%}"
            ))
            
            # Algorithm reliability factor
            algorithm_reliability = self._assess_score_algorithm_reliability(score)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.ALGORITHM_RELIABILITY,
                factor_name="Scoring Algorithm Reliability",
                score=algorithm_reliability,
                weight=self.factor_weights[ConfidenceFactorType.ALGORITHM_RELIABILITY],
                evidence=[f"Method: {score.scoring_method.value}"],
                explanation="Reliability of the scoring methodology"
            ))
            
            # Sample size factor
            sample_size_score = self._assess_score_sample_size(score)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.SAMPLE_SIZE,
                factor_name="Requirements Sample Size",
                score=sample_size_score,
                weight=self.factor_weights[ConfidenceFactorType.SAMPLE_SIZE],
                evidence=[f"Requirements evaluated: {score.total_requirements}"],
                explanation="Adequacy of the sample size for reliable scoring"
            ))
            
            # Validation factor
            validation_score = score.scoring_confidence
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.VALIDATION_STATUS,
                factor_name="Scoring Validation",
                score=validation_score,
                weight=self.factor_weights[ConfidenceFactorType.VALIDATION_STATUS],
                evidence=[f"Scoring confidence: {validation_score:.2f}"],
                explanation="Internal validation confidence of the scoring process"
            ))
            
            # Calculate overall confidence
            overall_confidence = sum(factor.weighted_contribution for factor in factors)
            
            # Calculate uncertainty measures
            uncertainty = self._calculate_score_uncertainty(score, factors)
            conf_lower, conf_upper = self._calculate_confidence_interval(overall_confidence, uncertainty)
            
            # Create confidence assessment
            assessment = ConfidenceAssessment(
                assessment_id=self._generate_assessment_id(score.score_id),
                target_id=score.score_id,
                target_type="compliance_score",
                recommendation_type=RecommendationType.COMPLIANCE_IMPROVEMENT,
                overall_confidence=overall_confidence,
                confidence_level=self._determine_confidence_level(overall_confidence),
                confidence_factors=factors,
                assessment_algorithm=self.get_scorer_name(),
                data_completeness=data_quality,
                method_validation=algorithm_reliability,
                domain_expertise=0.8,  # Assume good domain coverage for scoring
                confidence_interval_lower=conf_lower,
                confidence_interval_upper=conf_upper,
                uncertainty_factors=self._identify_score_uncertainty_factors(score),
                actionability_score=self._assess_score_actionability(score),
                implementation_feasibility=0.8,  # Generally feasible to act on scores
                resource_availability=0.7,
                consistency_score=0.8,  # Assume reasonable consistency
                logical_coherence=0.9,  # Scoring methodology is logically coherent
                evidence_strength=validation_score,
                confidence_risk_level=self._assess_confidence_risk_level(overall_confidence),
                false_positive_risk=0.1,  # Low risk for scoring
                false_negative_risk=0.15   # Slightly higher risk of missing issues
            )
            
            # Update statistics
            self.scoring_stats["assessments_generated"] += 1
            if assessment.is_high_confidence():
                self.scoring_stats["high_confidence_assessments"] += 1
            
            return assessment
            
        except Exception as e:
            self.scoring_stats["processing_errors"] += 1
            logger.error(f"Error assessing score confidence: {e}")
            raise
    
    async def assess_recommendation_confidence(
        self, 
        recommendation: str,
        recommendation_type: RecommendationType,
        supporting_evidence: List[str]
    ) -> ConfidenceAssessment:
        """Assess confidence in a specific recommendation."""
        
        rec_id = f"REC_{recommendation_type.value}_{datetime.utcnow().strftime('%H%M%S')}"
        
        logger.debug(f"Assessing confidence for recommendation {rec_id}")
        
        try:
            # Calculate confidence factors for recommendation
            factors = []
            
            # Evidence strength factor
            evidence_strength = self._assess_recommendation_evidence_strength(supporting_evidence)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.VALIDATION_STATUS,
                factor_name="Supporting Evidence Strength",
                score=evidence_strength,
                weight=self.factor_weights[ConfidenceFactorType.VALIDATION_STATUS],
                evidence=supporting_evidence,
                explanation="Quality and quantity of supporting evidence"
            ))
            
            # Domain coverage factor
            domain_coverage = self._assess_recommendation_domain_coverage(recommendation, recommendation_type)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.DOMAIN_COVERAGE,
                factor_name="Domain Coverage",
                score=domain_coverage,
                weight=self.factor_weights[ConfidenceFactorType.DOMAIN_COVERAGE],
                evidence=[f"Recommendation type: {recommendation_type.value}"],
                explanation="Coverage of the relevant domain area"
            ))
            
            # Complexity factor
            complexity_score = self._assess_recommendation_complexity(recommendation, recommendation_type)
            factors.append(ConfidenceFactor(
                factor_type=ConfidenceFactorType.COMPLEXITY_FACTOR,
                factor_name="Implementation Complexity",
                score=1.0 - complexity_score,  # Lower complexity = higher confidence
                weight=self.factor_weights[ConfidenceFactorType.COMPLEXITY_FACTOR],
                evidence=[f"Text length: {len(recommendation)} chars"],
                explanation="Inverse of implementation complexity"
            ))
            
            # Calculate overall confidence
            overall_confidence = sum(factor.weighted_contribution for factor in factors)
            
            # Add base confidence for recommendation types
            base_confidence = self._get_recommendation_type_base_confidence(recommendation_type)
            overall_confidence = (overall_confidence + base_confidence) / 2
            
            # Calculate uncertainty measures
            uncertainty = 0.15  # Default uncertainty for recommendations
            conf_lower, conf_upper = self._calculate_confidence_interval(overall_confidence, uncertainty)
            
            # Assess actionability
            actionability = self._assess_recommendation_actionability(recommendation, recommendation_type)
            
            # Create confidence assessment
            assessment = ConfidenceAssessment(
                assessment_id=self._generate_assessment_id(rec_id),
                target_id=rec_id,
                target_type="recommendation",
                recommendation_type=recommendation_type,
                overall_confidence=overall_confidence,
                confidence_level=self._determine_confidence_level(overall_confidence),
                confidence_factors=factors,
                assessment_algorithm=self.get_scorer_name(),
                data_completeness=evidence_strength,
                method_validation=0.8,  # Assume good method validation
                domain_expertise=domain_coverage,
                confidence_interval_lower=conf_lower,
                confidence_interval_upper=conf_upper,
                uncertainty_factors=self._identify_recommendation_uncertainty_factors(recommendation, recommendation_type),
                actionability_score=actionability,
                implementation_feasibility=self._assess_recommendation_implementation_feasibility(recommendation_type),
                resource_availability=0.7,
                consistency_score=0.8,
                logical_coherence=0.9,
                evidence_strength=evidence_strength,
                confidence_risk_level=self._assess_confidence_risk_level(overall_confidence),
                false_positive_risk=self._assess_recommendation_false_positive_risk(recommendation_type),
                false_negative_risk=self._assess_recommendation_false_negative_risk(recommendation_type)
            )
            
            # Update statistics
            self.scoring_stats["assessments_generated"] += 1
            if assessment.is_high_confidence():
                self.scoring_stats["high_confidence_assessments"] += 1
            
            return assessment
            
        except Exception as e:
            self.scoring_stats["processing_errors"] += 1
            logger.error(f"Error assessing recommendation confidence: {e}")
            raise
    
    # Helper methods for gap confidence assessment
    def _assess_gap_algorithm_reliability(self, gap: ComplianceGap) -> float:
        """Assess reliability of gap detection algorithm."""
        
        # Algorithm reliability scores based on method maturity
        algorithm_scores = {
            "keyword_based_detector": 0.75,
            "nlp_pattern_detector": 0.80,
            "ml_detector": 0.85,
            "hybrid_detector": 0.90
        }
        
        return algorithm_scores.get(gap.detection_algorithm, 0.70)
    
    def _assess_gap_domain_coverage(self, gap: ComplianceGap) -> float:
        """Assess domain coverage for the gap."""
        
        # Higher confidence for well-covered regulatory areas
        source_coverage = {
            "SYSC": 0.90,  # Well-covered systems and controls
            "COBS": 0.85,  # Well-covered conduct of business
            "PRIN": 0.80,  # Well-covered principles
            "DEPP": 0.75,  # Decision procedures
        }
        
        source_prefix = gap.regulatory_source.split('.')[0] if '.' in gap.regulatory_source else gap.regulatory_source
        return source_coverage.get(source_prefix, 0.70)
    
    def _assess_gap_evidence_strength(self, gap: ComplianceGap) -> float:
        """Assess strength of evidence for the gap."""
        
        evidence_length = len(gap.evidence)
        implementation_gaps = len(gap.implementation_gaps)
        
        # Base score from evidence quality
        base_score = min(evidence_length / 100, 0.8)  # Normalize evidence length
        
        # Boost for specific implementation gaps identified
        gap_boost = min(implementation_gaps * 0.1, 0.2)
        
        return min(base_score + gap_boost, 1.0)
    
    def _assess_gap_actionability(self, gap: ComplianceGap) -> float:
        """Assess how actionable the gap recommendations are."""
        
        # More specific recommendations are more actionable
        num_actions = len(gap.recommended_actions)
        specificity_score = min(num_actions / 5, 1.0)
        
        # Factor in urgency (more urgent = more actionable pressure)
        urgency_scores = {
            "immediate": 1.0,
            "urgent": 0.9,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        urgency_score = urgency_scores.get(gap.remediation_urgency.value, 0.6)
        
        return (specificity_score + urgency_score) / 2
    
    def _calculate_gap_uncertainty(self, gap: ComplianceGap, factors: List[ConfidenceFactor]) -> float:
        """Calculate uncertainty in gap assessment."""
        
        # Base uncertainty from detection confidence
        base_uncertainty = 1.0 - gap.detection_confidence
        
        # Add uncertainty from factor variance
        factor_scores = [factor.score for factor in factors]
        if factor_scores:
            factor_variance = sum((score - sum(factor_scores)/len(factor_scores))**2 for score in factor_scores) / len(factor_scores)
            variance_uncertainty = math.sqrt(factor_variance)
        else:
            variance_uncertainty = 0.0
        
        return min((base_uncertainty + variance_uncertainty) / 2, 0.3)
    
    # Additional helper methods (simplified for brevity)
    def _assess_gap_implementation_feasibility(self, gap: ComplianceGap) -> float:
        """Assess implementation feasibility."""
        return 0.8  # Default assumption
    
    def _assess_gap_consistency(self, gap: ComplianceGap) -> float:
        """Assess consistency with other findings."""
        return 0.8  # Default assumption
    
    def _assess_gap_logical_coherence(self, gap: ComplianceGap) -> float:
        """Assess logical coherence of gap identification."""
        return 0.9  # Gap identification is generally logically coherent
    
    def _identify_gap_uncertainty_factors(self, gap: ComplianceGap) -> List[str]:
        """Identify key uncertainty factors for gap."""
        factors = []
        
        if gap.detection_confidence < 0.8:
            factors.append("Low detection algorithm confidence")
        
        if len(gap.evidence) < 50:
            factors.append("Limited supporting evidence")
        
        if not gap.implementation_gaps:
            factors.append("No specific implementation gaps identified")
        
        return factors
    
    def _assess_confidence_risk_level(self, confidence: float) -> str:
        """Assess risk level of acting on this confidence."""
        if confidence >= 0.9:
            return "Very Low Risk"
        elif confidence >= 0.75:
            return "Low Risk"
        elif confidence >= 0.6:
            return "Medium Risk"
        elif confidence >= 0.4:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _assess_false_positive_risk(self, gap: ComplianceGap) -> float:
        """Assess risk of false positive in gap detection."""
        # Higher confidence = lower false positive risk
        return 1.0 - gap.detection_confidence
    
    def _assess_false_negative_risk(self, gap: ComplianceGap) -> float:
        """Assess risk of false negative (missing actual gaps)."""
        # Assume moderate risk based on detection method
        base_risk = 0.2
        
        # Adjust based on evidence strength
        if len(gap.evidence) < 30:
            base_risk += 0.1
        
        return min(base_risk, 0.4)
    
    # Score confidence helper methods (simplified)
    def _assess_score_algorithm_reliability(self, score: ComplianceScore) -> float:
        """Assess reliability of scoring algorithm."""
        method_scores = {
            "weighted_average": 0.85,
            "gap_penalty": 0.80,
            "risk_adjusted": 0.90,
            "hybrid": 0.95
        }
        return method_scores.get(score.scoring_method.value, 0.75)
    
    def _assess_score_sample_size(self, score: ComplianceScore) -> float:
        """Assess adequacy of sample size for scoring."""
        # Score based on number of requirements
        if score.total_requirements >= 50:
            return 1.0
        elif score.total_requirements >= 25:
            return 0.8
        elif score.total_requirements >= 10:
            return 0.6
        else:
            return 0.4
    
    def _calculate_score_uncertainty(self, score: ComplianceScore, factors: List[ConfidenceFactor]) -> float:
        """Calculate uncertainty in score assessment."""
        # Base uncertainty from scoring confidence
        return (1.0 - score.scoring_confidence) * 0.3
    
    def _identify_score_uncertainty_factors(self, score: ComplianceScore) -> List[str]:
        """Identify uncertainty factors for score."""
        factors = []
        
        if score.total_requirements < 20:
            factors.append("Small sample size of requirements")
        
        if score.data_completeness < 0.8:
            factors.append("Incomplete underlying data")
        
        if score.scoring_confidence < 0.8:
            factors.append("Low scoring algorithm confidence")
        
        return factors
    
    def _assess_score_actionability(self, score: ComplianceScore) -> float:
        """Assess actionability of score results."""
        # Score-based recommendations are generally actionable
        base_actionability = 0.8
        
        # Boost if specific improvement priorities provided
        if score.improvement_priority:
            base_actionability += 0.1
        
        return min(base_actionability, 1.0)
    
    # Recommendation confidence helper methods (simplified)
    def _assess_recommendation_evidence_strength(self, evidence: List[str]) -> float:
        """Assess strength of recommendation evidence."""
        if not evidence:
            return 0.3
        
        # Score based on quantity and quality of evidence
        evidence_score = min(len(evidence) / 5, 0.8)
        
        # Boost for detailed evidence
        avg_length = sum(len(e) for e in evidence) / len(evidence)
        detail_boost = min(avg_length / 100, 0.2)
        
        return min(evidence_score + detail_boost, 1.0)
    
    def _assess_recommendation_domain_coverage(self, recommendation: str, rec_type: RecommendationType) -> float:
        """Assess domain coverage for recommendation."""
        # Domain coverage scores by recommendation type
        type_scores = {
            RecommendationType.GAP_REMEDIATION: 0.90,
            RecommendationType.COMPLIANCE_IMPROVEMENT: 0.85,
            RecommendationType.RISK_MITIGATION: 0.80,
            RecommendationType.PROCESS_ENHANCEMENT: 0.75,
            RecommendationType.POLICY_UPDATE: 0.85,
            RecommendationType.TRAINING_REQUIREMENT: 0.80
        }
        return type_scores.get(rec_type, 0.70)
    
    def _assess_recommendation_complexity(self, recommendation: str, rec_type: RecommendationType) -> float:
        """Assess implementation complexity."""
        # Complexity scores by recommendation type (higher = more complex)
        type_complexity = {
            RecommendationType.GAP_REMEDIATION: 0.6,
            RecommendationType.COMPLIANCE_IMPROVEMENT: 0.7,
            RecommendationType.RISK_MITIGATION: 0.5,
            RecommendationType.PROCESS_ENHANCEMENT: 0.8,
            RecommendationType.POLICY_UPDATE: 0.4,
            RecommendationType.TRAINING_REQUIREMENT: 0.3
        }
        return type_complexity.get(rec_type, 0.6)
    
    def _get_recommendation_type_base_confidence(self, rec_type: RecommendationType) -> float:
        """Get base confidence by recommendation type."""
        base_confidences = {
            RecommendationType.GAP_REMEDIATION: 0.85,
            RecommendationType.COMPLIANCE_IMPROVEMENT: 0.80,
            RecommendationType.RISK_MITIGATION: 0.75,
            RecommendationType.PROCESS_ENHANCEMENT: 0.70,
            RecommendationType.POLICY_UPDATE: 0.80,
            RecommendationType.TRAINING_REQUIREMENT: 0.85
        }
        return base_confidences.get(rec_type, 0.75)
    
    def _assess_recommendation_actionability(self, recommendation: str, rec_type: RecommendationType) -> float:
        """Assess actionability of recommendation."""
        # Actionability scores by type
        type_actionability = {
            RecommendationType.GAP_REMEDIATION: 0.90,
            RecommendationType.POLICY_UPDATE: 0.85,
            RecommendationType.TRAINING_REQUIREMENT: 0.95,
            RecommendationType.PROCESS_ENHANCEMENT: 0.75,
            RecommendationType.COMPLIANCE_IMPROVEMENT: 0.80
        }
        return type_actionability.get(rec_type, 0.75)
    
    def _assess_recommendation_implementation_feasibility(self, rec_type: RecommendationType) -> float:
        """Assess implementation feasibility by recommendation type."""
        feasibility_scores = {
            RecommendationType.TRAINING_REQUIREMENT: 0.95,
            RecommendationType.POLICY_UPDATE: 0.85,
            RecommendationType.GAP_REMEDIATION: 0.80,
            RecommendationType.PROCESS_ENHANCEMENT: 0.70,
            RecommendationType.COMPLIANCE_IMPROVEMENT: 0.75
        }
        return feasibility_scores.get(rec_type, 0.75)
    
    def _identify_recommendation_uncertainty_factors(self, recommendation: str, rec_type: RecommendationType) -> List[str]:
        """Identify uncertainty factors for recommendation."""
        factors = []
        
        if len(recommendation) < 50:
            factors.append("Brief recommendation lacks detail")
        
        if rec_type in [RecommendationType.PROCESS_ENHANCEMENT, RecommendationType.COMPLIANCE_IMPROVEMENT]:
            factors.append("Complex implementation requirements")
        
        return factors
    
    def _assess_recommendation_false_positive_risk(self, rec_type: RecommendationType) -> float:
        """Assess false positive risk by recommendation type."""
        # Risk of recommending unnecessary actions
        risk_scores = {
            RecommendationType.GAP_REMEDIATION: 0.10,
            RecommendationType.TRAINING_REQUIREMENT: 0.05,
            RecommendationType.POLICY_UPDATE: 0.15,
            RecommendationType.PROCESS_ENHANCEMENT: 0.20
        }
        return risk_scores.get(rec_type, 0.15)
    
    def _assess_recommendation_false_negative_risk(self, rec_type: RecommendationType) -> float:
        """Assess false negative risk by recommendation type."""
        # Risk of missing necessary actions
        return 0.15  # Default moderate risk


class ConfidenceScoringPipeline:
    """Pipeline for orchestrating confidence scoring across different targets."""
    
    def __init__(self, scorers: List[BaseConfidenceScorer]):
        self.scorers = scorers
        self.pipeline_stats = {
            "total_assessments": 0,
            "high_confidence_rate": 0.0,
            "average_confidence": 0.0,
            "scoring_sessions": 0
        }
    
    async def assess_gaps_confidence(self, gaps: List[ComplianceGap]) -> List[ConfidenceAssessment]:
        """Assess confidence for multiple gaps."""
        
        assessments = []
        
        logger.info(f"Assessing confidence for {len(gaps)} gaps using {len(self.scorers)} scorers")
        
        for gap in gaps:
            for scorer in self.scorers:
                try:
                    assessment = await scorer.assess_gap_confidence(gap)
                    assessments.append(assessment)
                    
                except Exception as e:
                    logger.error(f"Error assessing gap {gap.gap_id} with scorer {scorer.get_scorer_name()}: {e}")
                    continue
        
        self._update_pipeline_stats(assessments)
        return assessments
    
    async def assess_scores_confidence(self, scores: List[ComplianceScore]) -> List[ConfidenceAssessment]:
        """Assess confidence for multiple scores."""
        
        assessments = []
        
        logger.info(f"Assessing confidence for {len(scores)} scores using {len(self.scorers)} scorers")
        
        for score in scores:
            for scorer in self.scorers:
                try:
                    assessment = await scorer.assess_score_confidence(score)
                    assessments.append(assessment)
                    
                except Exception as e:
                    logger.error(f"Error assessing score {score.score_id} with scorer {scorer.get_scorer_name()}: {e}")
                    continue
        
        self._update_pipeline_stats(assessments)
        return assessments
    
    def _update_pipeline_stats(self, assessments: List[ConfidenceAssessment]):
        """Update pipeline statistics."""
        
        if not assessments:
            return
        
        self.pipeline_stats["total_assessments"] += len(assessments)
        self.pipeline_stats["scoring_sessions"] += 1
        
        high_confidence_count = sum(1 for a in assessments if a.is_high_confidence())
        self.pipeline_stats["high_confidence_rate"] = high_confidence_count / len(assessments)
        
        avg_confidence = sum(a.overall_confidence for a in assessments) / len(assessments)
        self.pipeline_stats["average_confidence"] = avg_confidence
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        scorer_stats = {}
        for scorer in self.scorers:
            scorer_stats[scorer.get_scorer_name()] = scorer.scoring_stats
        
        return {
            **self.pipeline_stats,
            "scorer_statistics": scorer_stats
        }


# Factory functions
def create_comprehensive_confidence_scorer(config: Optional[Dict[str, Any]] = None) -> ComprehensiveConfidenceScorer:
    """Create comprehensive confidence scorer."""
    if config is None:
        config = {}
    
    return ComprehensiveConfidenceScorer(config)


def create_confidence_scoring_pipeline(
    scorers: Optional[List[str]] = None
) -> ConfidenceScoringPipeline:
    """Create confidence scoring pipeline with specified scorers."""
    
    if scorers is None:
        scorers = ["comprehensive"]
    
    scorer_instances = []
    
    for scorer_name in scorers:
        if scorer_name == "comprehensive":
            scorer_instances.append(create_comprehensive_confidence_scorer())
        else:
            logger.warning(f"Confidence scorer '{scorer_name}' not implemented, skipping")
    
    return ConfidenceScoringPipeline(scorer_instances)