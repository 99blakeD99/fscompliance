"""Compliance scoring mechanisms for evaluating regulatory compliance levels."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from ..models import RiskLevel, SeverityLevel
from .requirement_extraction import ExtractedRequirementEnhanced, RequirementScope
from .gap_detection import ComplianceGap, GapSeverity, GapType
from .requirement_categorization import (
    CategorizedRequirement, 
    FunctionalCategory,
    RiskBasedCategory,
    TemporalCategory,
    ComplexityCategory
)

logger = logging.getLogger(__name__)


class ScoreType(str, Enum):
    """Types of compliance scores."""
    OVERALL = "overall"                    # Overall compliance score
    FUNCTIONAL = "functional"              # By business function
    RISK_BASED = "risk_based"             # By risk category
    REQUIREMENT_TYPE = "requirement_type"  # By requirement type
    TEMPORAL = "temporal"                 # By urgency/timing
    FIRM_SPECIFIC = "firm_specific"       # Customized for firm


class ComplianceLevel(str, Enum):
    """Compliance level categories."""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-89%
    ADEQUATE = "adequate"       # 60-74%
    POOR = "poor"              # 40-59%
    CRITICAL = "critical"       # 0-39%


class ScoringMethod(str, Enum):
    """Scoring calculation methods."""
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by importance
    GAP_PENALTY = "gap_penalty"           # Penalty-based scoring
    RISK_ADJUSTED = "risk_adjusted"       # Risk-weighted scoring
    CATEGORICAL = "categorical"           # Category-based scoring
    HYBRID = "hybrid"                     # Combined approach


@dataclass
class ScoreComponent:
    """Individual component contributing to a compliance score."""
    
    component_id: str
    component_name: str
    score: float  # 0.0 to 100.0
    weight: float  # 0.0 to 1.0
    max_possible_score: float = 100.0
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted contribution to total score."""
        return self.score * self.weight
    
    @property
    def score_percentage(self) -> float:
        """Score as percentage of maximum possible."""
        return (self.score / self.max_possible_score) * 100 if self.max_possible_score > 0 else 0


class ComplianceScore(BaseModel):
    """Comprehensive compliance score with detailed breakdown."""
    
    # Core score data
    score_id: str = Field(..., description="Unique score identifier")
    score_type: ScoreType = Field(..., description="Type of compliance score")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall compliance score (0-100)")
    compliance_level: ComplianceLevel = Field(..., description="Compliance level category")
    
    # Score breakdown
    score_components: List[ScoreComponent] = Field(
        default_factory=list, description="Individual score components"
    )
    
    # Metadata
    scoring_method: ScoringMethod = Field(..., description="Method used for scoring")
    scoring_algorithm: str = Field(..., description="Algorithm used for calculation")
    scoring_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context information
    scope_description: str = Field(..., description="What this score covers")
    total_requirements: int = Field(..., description="Total requirements evaluated")
    requirements_met: int = Field(..., description="Requirements fully met")
    requirements_partial: int = Field(..., description="Requirements partially met")
    requirements_failed: int = Field(..., description="Requirements failed")
    
    # Gap analysis summary
    total_gaps: int = Field(default=0, description="Total compliance gaps identified")
    critical_gaps: int = Field(default=0, description="Critical severity gaps")
    high_gaps: int = Field(default=0, description="High severity gaps")
    medium_gaps: int = Field(default=0, description="Medium severity gaps")
    low_gaps: int = Field(default=0, description="Low severity gaps")
    
    # Risk assessment
    overall_risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_factors: List[str] = Field(default_factory=list, description="Key risk factors")
    
    # Recommendations
    improvement_priority: List[str] = Field(
        default_factory=list, description="Priority areas for improvement"
    )
    quick_wins: List[str] = Field(
        default_factory=list, description="Quick improvement opportunities"
    )
    
    # Confidence and reliability
    scoring_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in scoring")
    data_completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness of data used")
    
    def get_score_summary(self) -> Dict[str, Any]:
        """Get high-level score summary."""
        return {
            "overall_score": self.overall_score,
            "compliance_level": self.compliance_level.value,
            "total_requirements": self.total_requirements,
            "compliance_rate": (self.requirements_met / self.total_requirements * 100) if self.total_requirements > 0 else 0,
            "total_gaps": self.total_gaps,
            "risk_level": self.overall_risk_level.value,
            "scoring_confidence": self.scoring_confidence
        }
    
    def get_improvement_priorities(self) -> List[Tuple[str, int]]:
        """Get improvement priorities with urgency scores."""
        priorities = []
        
        # Add critical gaps as highest priority
        if self.critical_gaps > 0:
            priorities.append(("Address critical compliance gaps", 10))
        
        # Add high severity gaps
        if self.high_gaps > 0:
            priorities.append(("Resolve high severity gaps", 8))
        
        # Add specific improvement areas
        for i, priority in enumerate(self.improvement_priority[:3]):
            urgency = 7 - i  # Decreasing urgency
            priorities.append((priority, urgency))
        
        return sorted(priorities, key=lambda x: x[1], reverse=True)


class BaseComplianceScorer(ABC):
    """Abstract base class for compliance scoring algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scoring_stats = {
            "scores_calculated": 0,
            "high_confidence_scores": 0,
            "processing_errors": 0,
            "total_requirements_evaluated": 0
        }
    
    @abstractmethod
    async def calculate_compliance_score(
        self,
        requirements: List[CategorizedRequirement],
        gaps: List[ComplianceGap],
        scope: str = "overall"
    ) -> ComplianceScore:
        """Calculate compliance score using the specific algorithm."""
        pass
    
    @abstractmethod
    def get_scorer_name(self) -> str:
        """Get the name of the scoring algorithm."""
        pass
    
    def _determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level from numeric score."""
        if score >= 90:
            return ComplianceLevel.EXCELLENT
        elif score >= 75:
            return ComplianceLevel.GOOD
        elif score >= 60:
            return ComplianceLevel.ADEQUATE
        elif score >= 40:
            return ComplianceLevel.POOR
        else:
            return ComplianceLevel.CRITICAL
    
    def _assess_overall_risk_level(
        self, 
        gaps: List[ComplianceGap], 
        score: float
    ) -> RiskLevel:
        """Assess overall risk level based on gaps and score."""
        
        # Count gaps by severity
        critical_count = sum(1 for gap in gaps if gap.gap_severity == GapSeverity.CRITICAL)
        high_count = sum(1 for gap in gaps if gap.gap_severity == GapSeverity.HIGH)
        
        # Risk level based on critical gaps
        if critical_count > 0:
            return RiskLevel.CRITICAL
        elif high_count > 2:
            return RiskLevel.HIGH
        elif high_count > 0 or score < 70:
            return RiskLevel.MEDIUM
        elif score < 85:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE
    
    def _calculate_data_completeness(
        self, 
        requirements: List[CategorizedRequirement]
    ) -> float:
        """Calculate completeness of data used in scoring."""
        
        if not requirements:
            return 0.0
        
        total_confidence = sum(req.requirement.extraction_confidence for req in requirements)
        avg_confidence = total_confidence / len(requirements)
        
        # Factor in categorization confidence
        total_cat_confidence = sum(req.categorization_confidence for req in requirements)
        avg_cat_confidence = total_cat_confidence / len(requirements)
        
        # Combined completeness score
        return (avg_confidence + avg_cat_confidence) / 2
    
    def _generate_score_id(self, scope: str) -> str:
        """Generate unique score identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"SCORE_{scope.upper()}_{timestamp}"


class WeightedAverageScorer(BaseComplianceScorer):
    """Weighted average compliance scoring based on requirement importance."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weight_config = self._initialize_weights()
    
    def get_scorer_name(self) -> str:
        return "weighted_average_scorer"
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights for different requirement aspects."""
        
        return {
            # Severity weights
            "severity": {
                SeverityLevel.HIGH: 1.0,
                SeverityLevel.MEDIUM: 0.7,
                SeverityLevel.LOW: 0.4
            },
            
            # Scope weights
            "scope": {
                RequirementScope.MANDATORY: 1.0,
                RequirementScope.CONDITIONAL: 0.8,
                RequirementScope.GUIDANCE: 0.5,
                RequirementScope.EVIDENTIAL: 0.3,
                RequirementScope.DEFINITIONAL: 0.2
            },
            
            # Risk category weights
            "risk_category": {
                RiskBasedCategory.PRUDENTIAL_HIGH: 1.0,
                RiskBasedCategory.CONDUCT_HIGH: 1.0,
                RiskBasedCategory.OPERATIONAL_HIGH: 0.9,
                RiskBasedCategory.MARKET_HIGH: 0.9,
                RiskBasedCategory.PRUDENTIAL_MEDIUM: 0.7,
                RiskBasedCategory.CONDUCT_MEDIUM: 0.7,
                RiskBasedCategory.OPERATIONAL_MEDIUM: 0.6,
                RiskBasedCategory.MARKET_MEDIUM: 0.6,
                RiskBasedCategory.LOW_RISK: 0.3
            },
            
            # Temporal urgency weights
            "temporal": {
                TemporalCategory.IMMEDIATE: 1.0,
                TemporalCategory.SHORT_TERM: 0.9,
                TemporalCategory.MEDIUM_TERM: 0.7,
                TemporalCategory.LONG_TERM: 0.5,
                TemporalCategory.ONGOING: 0.8,
                TemporalCategory.PERIODIC: 0.6,
                TemporalCategory.EVENT_DRIVEN: 0.7
            }
        }
    
    async def calculate_compliance_score(
        self,
        requirements: List[CategorizedRequirement],
        gaps: List[ComplianceGap],
        scope: str = "overall"
    ) -> ComplianceScore:
        """Calculate weighted average compliance score."""
        
        logger.info(f"Calculating weighted average compliance score for {len(requirements)} requirements")
        
        try:
            # Create gap lookup for faster access
            gap_lookup = {gap.requirement_id: gap for gap in gaps}
            
            # Calculate individual requirement scores
            score_components = []
            total_weighted_score = 0.0
            total_weight = 0.0
            
            requirements_met = 0
            requirements_partial = 0
            requirements_failed = 0
            
            for req in requirements:
                # Calculate requirement weight
                weight = self._calculate_requirement_weight(req)
                
                # Calculate requirement score based on gaps
                req_score = self._calculate_requirement_score(req, gap_lookup.get(req.requirement.requirement_id))
                
                # Determine compliance status
                if req_score >= 90:
                    requirements_met += 1
                elif req_score >= 50:
                    requirements_partial += 1
                else:
                    requirements_failed += 1
                
                # Create score component
                component = ScoreComponent(
                    component_id=req.requirement.requirement_id,
                    component_name=f"{req.requirement.source_section} - {req.requirement.requirement_type.value}",
                    score=req_score,
                    weight=weight,
                    evidence=[f"Requirement: {req.requirement.requirement_text[:100]}..."]
                )
                score_components.append(component)
                
                # Add to weighted total
                total_weighted_score += req_score * weight
                total_weight += weight
            
            # Calculate overall score
            overall_score = (total_weighted_score / total_weight) if total_weight > 0 else 0
            
            # Analyze gaps by severity
            gap_counts = self._analyze_gap_severity(gaps)
            
            # Generate improvement recommendations
            improvement_priority, quick_wins = self._generate_recommendations(
                requirements, gaps, overall_score
            )
            
            # Calculate confidence metrics
            scoring_confidence = self._calculate_scoring_confidence(requirements, gaps)
            data_completeness = self._calculate_data_completeness(requirements)
            
            # Create compliance score
            compliance_score = ComplianceScore(
                score_id=self._generate_score_id(scope),
                score_type=ScoreType.OVERALL,
                overall_score=round(overall_score, 2),
                compliance_level=self._determine_compliance_level(overall_score),
                score_components=score_components,
                scoring_method=ScoringMethod.WEIGHTED_AVERAGE,
                scoring_algorithm=self.get_scorer_name(),
                scope_description=f"Weighted average compliance score for {scope}",
                total_requirements=len(requirements),
                requirements_met=requirements_met,
                requirements_partial=requirements_partial,
                requirements_failed=requirements_failed,
                total_gaps=len(gaps),
                critical_gaps=gap_counts["critical"],
                high_gaps=gap_counts["high"],
                medium_gaps=gap_counts["medium"],
                low_gaps=gap_counts["low"],
                overall_risk_level=self._assess_overall_risk_level(gaps, overall_score),
                risk_factors=self._identify_risk_factors(gaps),
                improvement_priority=improvement_priority,
                quick_wins=quick_wins,
                scoring_confidence=scoring_confidence,
                data_completeness=data_completeness
            )
            
            # Update statistics
            self.scoring_stats["scores_calculated"] += 1
            self.scoring_stats["total_requirements_evaluated"] += len(requirements)
            if scoring_confidence >= 0.8:
                self.scoring_stats["high_confidence_scores"] += 1
            
            logger.info(f"Compliance score calculated: {overall_score:.1f}% ({compliance_score.compliance_level.value})")
            
            return compliance_score
            
        except Exception as e:
            self.scoring_stats["processing_errors"] += 1
            logger.error(f"Error calculating compliance score: {e}")
            raise
    
    def _calculate_requirement_weight(self, req: CategorizedRequirement) -> float:
        """Calculate weight for individual requirement."""
        
        base_weight = 1.0
        
        # Weight by severity
        severity_weight = self.weight_config["severity"].get(req.requirement.severity, 0.5)
        
        # Weight by scope
        scope_weight = self.weight_config["scope"].get(req.requirement.requirement_scope, 0.5)
        
        # Weight by risk category (use primary risk category)
        risk_weight = 0.7  # default
        if req.primary_risk:
            risk_weight = self.weight_config["risk_category"].get(req.primary_risk, 0.7)
        
        # Weight by temporal urgency
        temporal_weight = 0.6  # default
        if req.primary_temporal:
            temporal_weight = self.weight_config["temporal"].get(req.primary_temporal, 0.6)
        
        # Combined weight (geometric mean to avoid extreme weights)
        combined_weight = math.pow(
            severity_weight * scope_weight * risk_weight * temporal_weight, 0.25
        )
        
        return min(combined_weight, 2.0)  # Cap maximum weight
    
    def _calculate_requirement_score(
        self, 
        req: CategorizedRequirement, 
        gap: Optional[ComplianceGap]
    ) -> float:
        """Calculate score for individual requirement."""
        
        if gap is None:
            # No gap found - assume full compliance
            return 100.0
        
        # Base score depends on gap type and severity
        base_score = 100.0
        
        # Penalty based on gap type
        gap_type_penalties = {
            GapType.MISSING_REQUIREMENT: 80,
            GapType.INCOMPLETE_IMPLEMENTATION: 40,
            GapType.OUTDATED_IMPLEMENTATION: 30,
            GapType.INSUFFICIENT_EVIDENCE: 20,
            GapType.PROCESS_GAP: 35,
            GapType.POLICY_GAP: 45,
            GapType.TRAINING_GAP: 25,
            GapType.MONITORING_GAP: 30
        }
        
        type_penalty = gap_type_penalties.get(gap.gap_type, 50)
        
        # Severity multiplier
        severity_multipliers = {
            GapSeverity.CRITICAL: 1.0,
            GapSeverity.HIGH: 0.8,
            GapSeverity.MEDIUM: 0.6,
            GapSeverity.LOW: 0.4,
            GapSeverity.NEGLIGIBLE: 0.2
        }
        
        severity_multiplier = severity_multipliers.get(gap.gap_severity, 0.7)
        
        # Calculate final score
        final_penalty = type_penalty * severity_multiplier
        final_score = max(base_score - final_penalty, 0)
        
        # Adjust based on detection confidence
        confidence_adjustment = gap.detection_confidence * 10  # Up to 10 point adjustment
        final_score = max(final_score - confidence_adjustment, 0)
        
        return final_score
    
    def _analyze_gap_severity(self, gaps: List[ComplianceGap]) -> Dict[str, int]:
        """Analyze gaps by severity level."""
        
        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "negligible": 0
        }
        
        for gap in gaps:
            if gap.gap_severity == GapSeverity.CRITICAL:
                counts["critical"] += 1
            elif gap.gap_severity == GapSeverity.HIGH:
                counts["high"] += 1
            elif gap.gap_severity == GapSeverity.MEDIUM:
                counts["medium"] += 1
            elif gap.gap_severity == GapSeverity.LOW:
                counts["low"] += 1
            else:
                counts["negligible"] += 1
        
        return counts
    
    def _generate_recommendations(
        self, 
        requirements: List[CategorizedRequirement],
        gaps: List[ComplianceGap], 
        overall_score: float
    ) -> Tuple[List[str], List[str]]:
        """Generate improvement recommendations."""
        
        improvement_priority = []
        quick_wins = []
        
        # Priority recommendations based on critical/high gaps
        critical_gaps = [gap for gap in gaps if gap.gap_severity == GapSeverity.CRITICAL]
        high_gaps = [gap for gap in gaps if gap.gap_severity == GapSeverity.HIGH]
        
        if critical_gaps:
            improvement_priority.append(f"Address {len(critical_gaps)} critical compliance gaps immediately")
            
        if high_gaps:
            improvement_priority.append(f"Resolve {len(high_gaps)} high severity gaps within 30 days")
        
        # Functional area priorities
        functional_gaps = {}
        for gap in gaps:
            # Find corresponding requirement to get functional category
            req = next((r for r in requirements if r.requirement.requirement_id == gap.requirement_id), None)
            if req and req.primary_functional:
                func = req.primary_functional.value
                if func not in functional_gaps:
                    functional_gaps[func] = []
                functional_gaps[func].append(gap)
        
        # Add functional area recommendations
        for func, func_gaps in functional_gaps.items():
            if len(func_gaps) >= 3:
                improvement_priority.append(f"Strengthen {func.replace('_', ' ')} framework")
        
        # Quick wins - low complexity, high impact
        evidence_gaps = [gap for gap in gaps if gap.gap_type == GapType.INSUFFICIENT_EVIDENCE]
        if evidence_gaps:
            quick_wins.append(f"Document existing processes to address {len(evidence_gaps)} evidence gaps")
        
        training_gaps = [gap for gap in gaps if gap.gap_type == GapType.TRAINING_GAP]
        if training_gaps:
            quick_wins.append(f"Deliver targeted training to address {len(training_gaps)} competence gaps")
        
        return improvement_priority[:5], quick_wins[:3]
    
    def _identify_risk_factors(self, gaps: List[ComplianceGap]) -> List[str]:
        """Identify key risk factors from gaps."""
        
        risk_factors = []
        
        # Risk factors by gap type
        gap_type_counts = {}
        for gap in gaps:
            gap_type = gap.gap_type
            gap_type_counts[gap_type] = gap_type_counts.get(gap_type, 0) + 1
        
        # Identify significant risk patterns
        if gap_type_counts.get(GapType.MISSING_REQUIREMENT, 0) >= 3:
            risk_factors.append("Multiple missing regulatory requirements")
        
        if gap_type_counts.get(GapType.PROCESS_GAP, 0) >= 2:
            risk_factors.append("Process deficiencies in key areas")
        
        if gap_type_counts.get(GapType.POLICY_GAP, 0) >= 2:
            risk_factors.append("Policy framework inadequacies")
        
        # Risk factors by severity
        critical_count = sum(1 for gap in gaps if gap.gap_severity == GapSeverity.CRITICAL)
        high_count = sum(1 for gap in gaps if gap.gap_severity == GapSeverity.HIGH)
        
        if critical_count > 0:
            risk_factors.append(f"{critical_count} critical compliance failures")
        
        if high_count > 3:
            risk_factors.append(f"{high_count} high-risk compliance gaps")
        
        return risk_factors[:5]
    
    def _calculate_scoring_confidence(
        self, 
        requirements: List[CategorizedRequirement], 
        gaps: List[ComplianceGap]
    ) -> float:
        """Calculate confidence in the scoring results."""
        
        if not requirements:
            return 0.0
        
        # Base confidence from requirement extraction
        extraction_confidence = sum(req.requirement.extraction_confidence for req in requirements) / len(requirements)
        
        # Categorization confidence
        categorization_confidence = sum(req.categorization_confidence for req in requirements) / len(requirements)
        
        # Gap detection confidence
        gap_confidence = 0.8  # default
        if gaps:
            gap_confidence = sum(gap.detection_confidence for gap in gaps) / len(gaps)
        
        # Data coverage factor
        coverage_factor = min(len(requirements) / 50, 1.0)  # Assume 50 requirements for full coverage
        
        # Combined confidence
        combined_confidence = (
            extraction_confidence * 0.3 +
            categorization_confidence * 0.3 +
            gap_confidence * 0.3 +
            coverage_factor * 0.1
        )
        
        return min(combined_confidence, 1.0)


class ComplianceScoringPipeline:
    """Pipeline for orchestrating compliance scoring."""
    
    def __init__(self, scorers: List[BaseComplianceScorer]):
        self.scorers = scorers
        self.pipeline_stats = {
            "scores_generated": 0,
            "scoring_sessions": 0,
            "average_score": 0.0
        }
    
    async def calculate_compliance_scores(
        self,
        requirements: List[CategorizedRequirement],
        gaps: List[ComplianceGap],
        scopes: List[str] = None
    ) -> List[ComplianceScore]:
        """Calculate compliance scores using all configured scorers."""
        
        if scopes is None:
            scopes = ["overall"]
        
        all_scores = []
        
        logger.info(f"Starting scoring pipeline for {len(scopes)} scopes using {len(self.scorers)} scorers")
        
        for scope in scopes:
            for scorer in self.scorers:
                try:
                    score = await scorer.calculate_compliance_score(requirements, gaps, scope)
                    all_scores.append(score)
                    
                    logger.info(f"Score calculated by '{scorer.get_scorer_name()}' for '{scope}': {score.overall_score:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Error in scorer '{scorer.get_scorer_name()}' for scope '{scope}': {e}")
                    continue
        
        # Update statistics
        self.pipeline_stats["scores_generated"] += len(all_scores)
        self.pipeline_stats["scoring_sessions"] += 1
        
        if all_scores:
            avg_score = sum(score.overall_score for score in all_scores) / len(all_scores)
            self.pipeline_stats["average_score"] = avg_score
        
        logger.info(f"Scoring pipeline completed: {len(all_scores)} scores generated")
        
        return all_scores
    
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
def create_weighted_average_scorer(config: Optional[Dict[str, Any]] = None) -> WeightedAverageScorer:
    """Create weighted average compliance scorer."""
    if config is None:
        config = {}
    
    return WeightedAverageScorer(config)


def create_compliance_scoring_pipeline(
    scorers: Optional[List[str]] = None
) -> ComplianceScoringPipeline:
    """Create compliance scoring pipeline with specified scorers."""
    
    if scorers is None:
        scorers = ["weighted_average"]
    
    scorer_instances = []
    
    for scorer_name in scorers:
        if scorer_name == "weighted_average":
            scorer_instances.append(create_weighted_average_scorer())
        else:
            logger.warning(f"Scorer '{scorer_name}' not implemented, skipping")
    
    return ComplianceScoringPipeline(scorer_instances)