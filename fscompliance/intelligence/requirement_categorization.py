"""Requirement categorization system for organizing and classifying regulatory requirements."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..models import RequirementType, SeverityLevel
from .requirement_extraction import ExtractedRequirementEnhanced, RequirementScope

logger = logging.getLogger(__name__)


class CategoryType(str, Enum):
    """Types of requirement categories."""
    FUNCTIONAL = "functional"          # By business function
    RISK_BASED = "risk_based"         # By risk profile
    TEMPORAL = "temporal"             # By timing/urgency
    COMPLEXITY = "complexity"         # By implementation complexity
    APPLICABILITY = "applicability"   # By firm type/size
    ENFORCEMENT = "enforcement"       # By enforcement priority
    THEMATIC = "thematic"            # By regulatory theme


class FunctionalCategory(str, Enum):
    """Functional categorization of requirements."""
    GOVERNANCE_OVERSIGHT = "governance_oversight"
    RISK_MANAGEMENT = "risk_management" 
    CONDUCT_CUSTOMER = "conduct_customer"
    OPERATIONAL_RESILIENCE = "operational_resilience"
    FINANCIAL_RESOURCES = "financial_resources"
    MARKET_CONDUCT = "market_conduct"
    REPORTING_DISCLOSURE = "reporting_disclosure"
    RECORD_KEEPING = "record_keeping"
    TRAINING_COMPETENCE = "training_competence"
    COMPLAINTS_HANDLING = "complaints_handling"


class RiskBasedCategory(str, Enum):
    """Risk-based categorization of requirements."""
    PRUDENTIAL_HIGH = "prudential_high"        # High prudential risk
    CONDUCT_HIGH = "conduct_high"              # High conduct risk
    OPERATIONAL_HIGH = "operational_high"      # High operational risk
    MARKET_HIGH = "market_high"               # High market risk
    PRUDENTIAL_MEDIUM = "prudential_medium"    # Medium prudential risk
    CONDUCT_MEDIUM = "conduct_medium"          # Medium conduct risk
    OPERATIONAL_MEDIUM = "operational_medium"  # Medium operational risk
    MARKET_MEDIUM = "market_medium"           # Medium market risk
    LOW_RISK = "low_risk"                     # Low risk across categories


class TemporalCategory(str, Enum):
    """Temporal categorization based on timing requirements."""
    IMMEDIATE = "immediate"              # Immediate implementation
    SHORT_TERM = "short_term"           # Within 3 months  
    MEDIUM_TERM = "medium_term"         # 3-12 months
    LONG_TERM = "long_term"             # 12+ months
    ONGOING = "ongoing"                 # Continuous requirement
    PERIODIC = "periodic"               # Regular intervals
    EVENT_DRIVEN = "event_driven"       # Triggered by events


class ComplexityCategory(str, Enum):
    """Complexity categorization for implementation effort."""
    ADMINISTRATIVE = "administrative"    # Simple admin changes
    PROCEDURAL = "procedural"           # Process changes required
    SYSTEMATIC = "systematic"           # System/technology changes
    STRUCTURAL = "structural"           # Organizational changes
    TRANSFORMATIONAL = "transformational"  # Fundamental business changes


class ApplicabilityCategory(str, Enum):
    """Applicability categorization by firm characteristics."""
    ALL_FIRMS = "all_firms"                    # All authorized firms
    LARGE_FIRMS = "large_firms"                # Large/complex firms
    SMALL_FIRMS = "small_firms"                # Small/simple firms
    INVESTMENT_FIRMS = "investment_firms"       # Investment firms only
    BANKS = "banks"                           # Banks/credit institutions
    INSURERS = "insurers"                     # Insurance firms
    PAYMENT_FIRMS = "payment_firms"            # Payment institutions
    ASSET_MANAGERS = "asset_managers"          # Asset management firms


class EnforcementCategory(str, Enum):
    """Enforcement priority categorization."""
    SUPERVISORY_PRIORITY = "supervisory_priority"  # High supervisor focus
    THEMATIC_REVIEW = "thematic_review"           # Subject to thematic reviews
    SKILLED_PERSON = "skilled_person"             # Skilled person reports
    ENFORCEMENT_ACTION = "enforcement_action"      # Enforcement cases
    INDUSTRY_GUIDANCE = "industry_guidance"        # Industry guidance priority
    STANDARD_MONITORING = "standard_monitoring"    # Standard supervision


class ThematicCategory(str, Enum):
    """Thematic categorization by regulatory themes."""
    CONSUMER_DUTY = "consumer_duty"              # Consumer Duty theme
    ESG_SUSTAINABILITY = "esg_sustainability"    # ESG/sustainability
    OPERATIONAL_RESILIENCE = "operational_resilience"  # Op resilience
    FINANCIAL_CRIME = "financial_crime"          # Financial crime prevention
    DATA_GOVERNANCE = "data_governance"          # Data and technology
    CULTURE_GOVERNANCE = "culture_governance"    # Culture and governance
    MARKET_INTEGRITY = "market_integrity"        # Market integrity
    PRUDENTIAL_STANDARDS = "prudential_standards"  # Prudential regulation


@dataclass
class CategoryWeight:
    """Weight assigned to a category for a requirement."""
    
    category: str
    weight: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]


class CategorizedRequirement(BaseModel):
    """A requirement with its assigned categories."""
    
    # Base requirement
    requirement: ExtractedRequirementEnhanced = Field(..., description="The base requirement")
    
    # Category assignments
    functional_categories: List[Tuple[FunctionalCategory, float]] = Field(
        default_factory=list, description="Functional categories with weights"
    )
    risk_categories: List[Tuple[RiskBasedCategory, float]] = Field(
        default_factory=list, description="Risk-based categories with weights"
    )
    temporal_categories: List[Tuple[TemporalCategory, float]] = Field(
        default_factory=list, description="Temporal categories with weights"
    )
    complexity_categories: List[Tuple[ComplexityCategory, float]] = Field(
        default_factory=list, description="Complexity categories with weights"
    )
    applicability_categories: List[Tuple[ApplicabilityCategory, float]] = Field(
        default_factory=list, description="Applicability categories with weights"
    )
    enforcement_categories: List[Tuple[EnforcementCategory, float]] = Field(
        default_factory=list, description="Enforcement categories with weights"
    )
    thematic_categories: List[Tuple[ThematicCategory, float]] = Field(
        default_factory=list, description="Thematic categories with weights"
    )
    
    # Categorization metadata
    categorization_algorithm: str = Field(..., description="Algorithm used for categorization")
    categorization_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    categorization_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Primary categories (highest weighted in each type)
    primary_functional: Optional[FunctionalCategory] = Field(None, description="Primary functional category")
    primary_risk: Optional[RiskBasedCategory] = Field(None, description="Primary risk category")
    primary_temporal: Optional[TemporalCategory] = Field(None, description="Primary temporal category")
    primary_complexity: Optional[ComplexityCategory] = Field(None, description="Primary complexity category")
    
    def get_all_categories(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get all categories as a unified dictionary."""
        return {
            "functional": [(cat.value, weight) for cat, weight in self.functional_categories],
            "risk_based": [(cat.value, weight) for cat, weight in self.risk_categories],
            "temporal": [(cat.value, weight) for cat, weight in self.temporal_categories],
            "complexity": [(cat.value, weight) for cat, weight in self.complexity_categories],
            "applicability": [(cat.value, weight) for cat, weight in self.applicability_categories],
            "enforcement": [(cat.value, weight) for cat, weight in self.enforcement_categories],
            "thematic": [(cat.value, weight) for cat, weight in self.thematic_categories]
        }


class BaseRequirementCategorizer(ABC):
    """Abstract base class for requirement categorization algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categorization_stats = {
            "requirements_categorized": 0,
            "categories_assigned": 0,
            "high_confidence_categorizations": 0,
            "processing_errors": 0
        }
    
    @abstractmethod
    async def categorize_requirements(
        self, 
        requirements: List[ExtractedRequirementEnhanced]
    ) -> List[CategorizedRequirement]:
        """Categorize requirements using the specific algorithm."""
        pass
    
    @abstractmethod
    def get_categorizer_name(self) -> str:
        """Get the name of the categorization algorithm."""
        pass
    
    def _determine_primary_categories(self, categorized_req: CategorizedRequirement):
        """Determine primary categories from weighted assignments."""
        
        # Find highest weighted category in each type
        if categorized_req.functional_categories:
            categorized_req.primary_functional = max(
                categorized_req.functional_categories, key=lambda x: x[1]
            )[0]
        
        if categorized_req.risk_categories:
            categorized_req.primary_risk = max(
                categorized_req.risk_categories, key=lambda x: x[1]
            )[0]
        
        if categorized_req.temporal_categories:
            categorized_req.primary_temporal = max(
                categorized_req.temporal_categories, key=lambda x: x[1]
            )[0]
        
        if categorized_req.complexity_categories:
            categorized_req.primary_complexity = max(
                categorized_req.complexity_categories, key=lambda x: x[1]
            )[0]


class RuleBasedCategorizer(BaseRequirementCategorizer):
    """Rule-based requirement categorization using patterns and heuristics."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.categorization_rules = self._initialize_categorization_rules()
    
    def get_categorizer_name(self) -> str:
        return "rule_based_categorizer"
    
    def _initialize_categorization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize categorization rules and patterns."""
        
        return {
            "functional": {
                FunctionalCategory.GOVERNANCE_OVERSIGHT: {
                    "keywords": ["governance", "oversight", "board", "senior management", "accountability"],
                    "requirement_types": [RequirementType.GOVERNANCE],
                    "weight": 0.9
                },
                FunctionalCategory.RISK_MANAGEMENT: {
                    "keywords": ["risk", "control", "assessment", "mitigation", "monitoring"],
                    "requirement_types": [RequirementType.RISK_MANAGEMENT],
                    "weight": 0.9
                },
                FunctionalCategory.CONDUCT_CUSTOMER: {
                    "keywords": ["customer", "client", "consumer", "fair treatment", "outcome"],
                    "requirement_types": [RequirementType.CONDUCT, RequirementType.CLIENT_PROTECTION],
                    "weight": 0.9
                },
                FunctionalCategory.REPORTING_DISCLOSURE: {
                    "keywords": ["report", "disclose", "notify", "submission", "publication"],
                    "requirement_types": [RequirementType.REPORTING],
                    "weight": 0.9
                },
                FunctionalCategory.RECORD_KEEPING: {
                    "keywords": ["record", "document", "maintain", "store", "archive"],
                    "requirement_types": [RequirementType.RECORD_KEEPING],
                    "weight": 0.9
                }
            },
            
            "risk_based": {
                RiskBasedCategory.PRUDENTIAL_HIGH: {
                    "keywords": ["capital", "liquidity", "solvency", "prudential"],
                    "severity": [SeverityLevel.HIGH],
                    "weight": 0.8
                },
                RiskBasedCategory.CONDUCT_HIGH: {
                    "keywords": ["consumer duty", "fair treatment", "suitable", "appropriate"],
                    "requirement_types": [RequirementType.CONDUCT],
                    "severity": [SeverityLevel.HIGH],
                    "weight": 0.8
                },
                RiskBasedCategory.OPERATIONAL_HIGH: {
                    "keywords": ["operational", "system", "process", "procedure", "resilience"],
                    "severity": [SeverityLevel.HIGH],
                    "weight": 0.7
                }
            },
            
            "temporal": {
                TemporalCategory.IMMEDIATE: {
                    "keywords": ["immediate", "without delay", "forthwith"],
                    "deadlines": ["immediate"],
                    "weight": 0.9
                },
                TemporalCategory.SHORT_TERM: {
                    "keywords": ["days", "weeks", "month"],
                    "deadlines": ["within.*days", "within.*weeks", "within.*month"],
                    "weight": 0.8
                },
                TemporalCategory.ONGOING: {
                    "keywords": ["ongoing", "continuous", "maintain", "ensure"],
                    "modal_verbs": ["must", "shall"],
                    "weight": 0.7
                },
                TemporalCategory.PERIODIC: {
                    "keywords": ["annual", "quarterly", "monthly", "regular"],
                    "deadlines": ["annually", "quarterly", "monthly"],
                    "weight": 0.8
                }
            },
            
            "complexity": {
                ComplexityCategory.ADMINISTRATIVE: {
                    "keywords": ["notify", "inform", "disclose", "report"],
                    "action_verbs": ["notify", "inform", "report"],
                    "weight": 0.7
                },
                ComplexityCategory.PROCEDURAL: {
                    "keywords": ["procedure", "process", "method", "approach"],
                    "action_verbs": ["establish", "implement", "develop"],
                    "weight": 0.7
                },
                ComplexityCategory.SYSTEMATIC: {
                    "keywords": ["system", "technology", "automated", "electronic"],
                    "action_verbs": ["implement", "deploy", "integrate"],
                    "weight": 0.8
                },
                ComplexityCategory.STRUCTURAL: {
                    "keywords": ["organization", "structure", "arrangement", "framework"],
                    "requirement_types": [RequirementType.GOVERNANCE],
                    "weight": 0.8
                }
            },
            
            "applicability": {
                ApplicabilityCategory.ALL_FIRMS: {
                    "firm_types": [],  # Empty means applies to all
                    "keywords": ["firm", "person"],
                    "weight": 0.5
                },
                ApplicabilityCategory.INVESTMENT_FIRMS: {
                    "firm_types": ["investment_firm"],
                    "keywords": ["investment firm", "investment business"],
                    "weight": 0.9
                },
                ApplicabilityCategory.BANKS: {
                    "firm_types": ["bank", "credit_institution"],
                    "keywords": ["bank", "credit institution", "deposit"],
                    "weight": 0.9
                }
            },
            
            "enforcement": {
                EnforcementCategory.SUPERVISORY_PRIORITY: {
                    "keywords": ["consumer duty", "operational resilience", "ESG"],
                    "severity": [SeverityLevel.HIGH],
                    "weight": 0.8
                },
                EnforcementCategory.STANDARD_MONITORING: {
                    "severity": [SeverityLevel.LOW, SeverityLevel.MEDIUM],
                    "weight": 0.6
                }
            },
            
            "thematic": {
                ThematicCategory.CONSUMER_DUTY: {
                    "keywords": ["consumer duty", "consumer outcome", "fair value", "consumer understanding"],
                    "weight": 0.9
                },
                ThematicCategory.OPERATIONAL_RESILIENCE: {
                    "keywords": ["operational resilience", "business continuity", "system resilience"],
                    "weight": 0.9
                },
                ThematicCategory.FINANCIAL_CRIME: {
                    "keywords": ["money laundering", "financial crime", "sanctions", "terrorist financing"],
                    "weight": 0.9
                },
                ThematicCategory.DATA_GOVERNANCE: {
                    "keywords": ["data", "technology", "cyber", "information security"],
                    "weight": 0.8
                }
            }
        }
    
    async def categorize_requirements(
        self, 
        requirements: List[ExtractedRequirementEnhanced]
    ) -> List[CategorizedRequirement]:
        """Categorize requirements using rule-based patterns."""
        
        categorized_requirements = []
        
        logger.info(f"Starting rule-based categorization for {len(requirements)} requirements")
        
        for requirement in requirements:
            try:
                # Categorize requirement across all category types
                categorized_req = CategorizedRequirement(
                    requirement=requirement,
                    categorization_algorithm=self.get_categorizer_name(),
                    categorization_confidence=0.0  # Will be calculated
                )
                
                # Apply functional categorization
                categorized_req.functional_categories = self._categorize_functional(requirement)
                
                # Apply risk-based categorization
                categorized_req.risk_categories = self._categorize_risk_based(requirement)
                
                # Apply temporal categorization
                categorized_req.temporal_categories = self._categorize_temporal(requirement)
                
                # Apply complexity categorization
                categorized_req.complexity_categories = self._categorize_complexity(requirement)
                
                # Apply applicability categorization
                categorized_req.applicability_categories = self._categorize_applicability(requirement)
                
                # Apply enforcement categorization
                categorized_req.enforcement_categories = self._categorize_enforcement(requirement)
                
                # Apply thematic categorization
                categorized_req.thematic_categories = self._categorize_thematic(requirement)
                
                # Determine primary categories
                self._determine_primary_categories(categorized_req)
                
                # Calculate overall confidence
                categorized_req.categorization_confidence = self._calculate_categorization_confidence(categorized_req)
                
                categorized_requirements.append(categorized_req)
                
                # Update statistics
                self.categorization_stats["requirements_categorized"] += 1
                total_categories = (
                    len(categorized_req.functional_categories) +
                    len(categorized_req.risk_categories) +
                    len(categorized_req.temporal_categories) +
                    len(categorized_req.complexity_categories) +
                    len(categorized_req.applicability_categories) +
                    len(categorized_req.enforcement_categories) +
                    len(categorized_req.thematic_categories)
                )
                self.categorization_stats["categories_assigned"] += total_categories
                
                if categorized_req.categorization_confidence >= 0.8:
                    self.categorization_stats["high_confidence_categorizations"] += 1
                
            except Exception as e:
                logger.error(f"Error categorizing requirement {requirement.requirement_id}: {e}")
                self.categorization_stats["processing_errors"] += 1
                continue
        
        logger.info(f"Rule-based categorization completed: {len(categorized_requirements)} requirements categorized")
        
        return categorized_requirements
    
    def _categorize_functional(self, requirement: ExtractedRequirementEnhanced) -> List[Tuple[FunctionalCategory, float]]:
        """Apply functional categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["functional"].items():
            score = 0.0
            
            # Check requirement type match
            if "requirement_types" in rules:
                if requirement.requirement_type in rules["requirement_types"]:
                    score += rules["weight"]
            
            # Check keyword matches
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * 0.6
            
            # Add category if score is significant
            if score >= 0.3:
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:3]  # Top 3 functional categories
    
    def _categorize_risk_based(self, requirement: ExtractedRequirement Enhanced) -> List[Tuple[RiskBasedCategory, float]]:
        """Apply risk-based categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["risk_based"].items():
            score = 0.0
            
            # Check severity level
            if "severity" in rules:
                if requirement.severity in rules["severity"]:
                    score += rules["weight"]
            
            # Check requirement type match
            if "requirement_types" in rules:
                if requirement.requirement_type in rules["requirement_types"]:
                    score += 0.5
            
            # Check keyword matches
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * 0.4
            
            # Add category if score is significant
            if score >= 0.3:
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:2]  # Top 2 risk categories
    
    def _categorize_temporal(self, requirement: ExtractedRequirementEnhanced) -> List[Tuple[TemporalCategory, float]]:
        """Apply temporal categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["temporal"].items():
            score = 0.0
            
            # Check deadlines
            if "deadlines" in rules and requirement.deadlines:
                for deadline in requirement.deadlines:
                    for pattern in rules["deadlines"]:
                        if pattern in deadline.lower():
                            score += rules["weight"]
                            break
            
            # Check modal verbs
            if "modal_verbs" in rules:
                modal_matches = sum(1 for modal in rules["modal_verbs"] if modal in requirement.modal_verbs)
                if modal_matches > 0:
                    score += 0.3
            
            # Check keywords
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * 0.4
            
            # Add category if score is significant
            if score >= 0.3:
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:2]  # Top 2 temporal categories
    
    def _categorize_complexity(self, requirement: ExtractedRequirementEnhanced) -> List[Tuple[ComplexityCategory, float]]:
        """Apply complexity categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["complexity"].items():
            score = 0.0
            
            # Check action verbs
            if "action_verbs" in rules:
                verb_matches = sum(1 for verb in rules["action_verbs"] if verb in requirement.action_verbs)
                if verb_matches > 0:
                    score += rules["weight"]
            
            # Check requirement type
            if "requirement_types" in rules:
                if requirement.requirement_type in rules["requirement_types"]:
                    score += 0.5
            
            # Check keywords
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * 0.4
            
            # Add category if score is significant
            if score >= 0.3:
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:2]  # Top 2 complexity categories
    
    def _categorize_applicability(self, requirement: ExtractedRequirementEnhanced) -> List[Tuple[ApplicabilityCategory, float]]:
        """Apply applicability categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["applicability"].items():
            score = 0.0
            
            # Check firm types
            if "firm_types" in rules:
                if not rules["firm_types"]:  # Empty list means all firms
                    score += 0.5
                else:
                    firm_matches = sum(1 for firm in rules["firm_types"] if firm in requirement.applicable_firms)
                    if firm_matches > 0:
                        score += rules["weight"]
            
            # Check keywords
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * 0.4
            
            # Add category if score is significant
            if score >= 0.3:
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:3]  # Top 3 applicability categories
    
    def _categorize_enforcement(self, requirement: ExtractedRequirementEnhanced) -> List[Tuple[EnforcementCategory, float]]:
        """Apply enforcement categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["enforcement"].items():
            score = 0.0
            
            # Check severity
            if "severity" in rules:
                if requirement.severity in rules["severity"]:
                    score += rules["weight"]
            
            # Check keywords
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * 0.6
            
            # Add category if score is significant
            if score >= 0.3:
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:2]  # Top 2 enforcement categories
    
    def _categorize_thematic(self, requirement: ExtractedRequirementEnhanced) -> List[Tuple[ThematicCategory, float]]:
        """Apply thematic categorization to a requirement."""
        
        categories = []
        
        for category, rules in self.categorization_rules["thematic"].items():
            score = 0.0
            
            # Check keywords
            if "keywords" in rules:
                req_text_lower = requirement.requirement_text.lower()
                keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in req_text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches / len(rules["keywords"])) * rules["weight"]
            
            # Add category if score is significant
            if score >= 0.4:  # Higher threshold for thematic categories
                categories.append((category, min(score, 1.0)))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:2]  # Top 2 thematic categories
    
    def _calculate_categorization_confidence(self, categorized_req: CategorizedRequirement) -> float:
        """Calculate overall confidence in the categorization."""
        
        # Base confidence from requirement extraction
        base_confidence = categorized_req.requirement.extraction_confidence
        
        # Count categories assigned
        total_categories = (
            len(categorized_req.functional_categories) +
            len(categorized_req.risk_categories) +
            len(categorized_req.temporal_categories) +
            len(categorized_req.complexity_categories) +
            len(categorized_req.applicability_categories) +
            len(categorized_req.enforcement_categories) +
            len(categorized_req.thematic_categories)
        )
        
        # Higher confidence if more categories assigned
        category_bonus = min(total_categories * 0.05, 0.2)
        
        # Higher confidence if high-weight categories assigned
        max_weights = []
        if categorized_req.functional_categories:
            max_weights.append(max(weight for _, weight in categorized_req.functional_categories))
        if categorized_req.risk_categories:
            max_weights.append(max(weight for _, weight in categorized_req.risk_categories))
        
        weight_bonus = max(max_weights) * 0.1 if max_weights else 0.0
        
        final_confidence = base_confidence + category_bonus + weight_bonus
        
        return min(final_confidence, 1.0)


class RequirementCategorizationPipeline:
    """Pipeline for orchestrating requirement categorization."""
    
    def __init__(self, categorizers: List[BaseRequirementCategorizer]):
        self.categorizers = categorizers
        self.pipeline_stats = {
            "requirements_processed": 0,
            "total_categories_assigned": 0,
            "categorization_sessions": 0
        }
    
    async def categorize_requirements(
        self, 
        requirements: List[ExtractedRequirementEnhanced]
    ) -> List[CategorizedRequirement]:
        """Categorize requirements using all configured categorizers."""
        
        all_categorized = []
        
        logger.info(f"Starting categorization pipeline for {len(requirements)} requirements using {len(self.categorizers)} categorizers")
        
        for categorizer in self.categorizers:
            try:
                categorized_reqs = await categorizer.categorize_requirements(requirements)
                all_categorized.extend(categorized_reqs)
                
                logger.info(f"Categorizer '{categorizer.get_categorizer_name()}' processed {len(categorized_reqs)} requirements")
                
            except Exception as e:
                logger.error(f"Error in categorizer '{categorizer.get_categorizer_name()}': {e}")
                continue
        
        # Merge results from multiple categorizers (if applicable)
        merged_categorized = self._merge_categorizations(all_categorized)
        
        # Update statistics
        self.pipeline_stats["requirements_processed"] += len(merged_categorized)
        self.pipeline_stats["categorization_sessions"] += 1
        
        for categorized_req in merged_categorized:
            total_categories = sum(len(cats) for cats in categorized_req.get_all_categories().values())
            self.pipeline_stats["total_categories_assigned"] += total_categories
        
        logger.info(f"Categorization pipeline completed: {len(merged_categorized)} requirements categorized")
        
        return merged_categorized
    
    def _merge_categorizations(
        self, 
        categorized_requirements: List[CategorizedRequirement]
    ) -> List[CategorizedRequirement]:
        """Merge categorizations from multiple categorizers."""
        
        # For now, just return the categorizations as-is
        # In future, could implement logic to merge results from multiple categorizers
        return categorized_requirements
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        categorizer_stats = {}
        for categorizer in self.categorizers:
            categorizer_stats[categorizer.get_categorizer_name()] = categorizer.categorization_stats
        
        return {
            **self.pipeline_stats,
            "categorizer_statistics": categorizer_stats
        }


# Factory functions
def create_rule_based_categorizer(config: Optional[Dict[str, Any]] = None) -> RuleBasedCategorizer:
    """Create rule-based requirement categorizer."""
    if config is None:
        config = {}
    
    return RuleBasedCategorizer(config)


def create_categorization_pipeline(
    categorizers: Optional[List[str]] = None
) -> RequirementCategorizationPipeline:
    """Create requirement categorization pipeline with specified categorizers."""
    
    if categorizers is None:
        categorizers = ["rule_based"]
    
    categorizer_instances = []
    
    for categorizer_name in categorizers:
        if categorizer_name == "rule_based":
            categorizer_instances.append(create_rule_based_categorizer())
        else:
            logger.warning(f"Categorizer '{categorizer_name}' not implemented, skipping")
    
    return RequirementCategorizationPipeline(categorizer_instances)