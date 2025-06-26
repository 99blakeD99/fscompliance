"""Requirement categorization system for FCA regulatory content."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..models import ConductRequirement, FCAFirmType, RequirementType, SeverityLevel
from .parser import ExtractedRequirement

logger = logging.getLogger(__name__)


class CategoryConfidence(str, Enum):
    """Confidence levels for categorization."""
    HIGH = "high"      # 0.8+
    MEDIUM = "medium"  # 0.6-0.8
    LOW = "low"        # 0.4-0.6
    UNCERTAIN = "uncertain"  # <0.4


class ApplicabilityScope(str, Enum):
    """Scope of requirement applicability."""
    ALL_FIRMS = "all_firms"
    INVESTMENT_FIRMS = "investment_firms"
    BANKS = "banks"
    INSURERS = "insurers"
    PAYMENT_INSTITUTIONS = "payment_institutions"
    CONSUMER_CREDIT = "consumer_credit"
    MORTGAGE_PROVIDERS = "mortgage_providers"
    SPECIFIC_ACTIVITIES = "specific_activities"


class BusinessFunction(str, Enum):
    """Business functions affected by requirements."""
    GOVERNANCE = "governance"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"
    CLIENT_FACING = "client_facing"
    TRADING = "trading"
    LENDING = "lending"
    INVESTMENT_ADVICE = "investment_advice"
    CUSTODY = "custody"
    PAYMENTS = "payments"
    REPORTING = "reporting"
    AUDIT = "audit"


class RegulatoryImpact(str, Enum):
    """Impact level of regulatory requirements."""
    STRUCTURAL = "structural"      # Changes to organizational structure
    PROCEDURAL = "procedural"      # Changes to processes/procedures
    DOCUMENTATION = "documentation"  # Documentation requirements
    REPORTING = "reporting"        # Reporting obligations
    TRAINING = "training"          # Staff training requirements
    TECHNOLOGY = "technology"      # IT system changes
    MONITORING = "monitoring"      # Ongoing monitoring requirements


class CategoryResult(BaseModel):
    """Result of requirement categorization."""
    
    requirement_id: str = Field(..., description="Unique requirement identifier")
    primary_category: RequirementType = Field(..., description="Primary requirement category")
    secondary_categories: List[RequirementType] = Field(default_factory=list, description="Additional categories")
    severity: SeverityLevel = Field(..., description="Assessed severity level")
    applicability_scope: ApplicabilityScope = Field(..., description="Firm type applicability")
    applicable_firm_types: List[FCAFirmType] = Field(default_factory=list, description="Specific firm types")
    business_functions: List[BusinessFunction] = Field(default_factory=list, description="Affected business functions")
    regulatory_impact: List[RegulatoryImpact] = Field(default_factory=list, description="Types of impact")
    confidence: CategoryConfidence = Field(..., description="Categorization confidence")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numerical confidence score")
    reasoning: str = Field(..., description="Explanation of categorization logic")
    keywords_matched: List[str] = Field(default_factory=list, description="Keywords that influenced categorization")
    
    class Config:
        schema_extra = {
            "example": {
                "requirement_id": "SYSC.4.1.1",
                "primary_category": "governance",
                "secondary_categories": ["risk_management"],
                "severity": "high",
                "applicability_scope": "all_firms",
                "applicable_firm_types": ["investment_firm", "bank"],
                "business_functions": ["governance", "risk_management"],
                "regulatory_impact": ["structural", "procedural"],
                "confidence": "high",
                "confidence_score": 0.85,
                "reasoning": "Contains governance requirements with risk management elements",
                "keywords_matched": ["governance", "arrangements", "robust"]
            }
        }


@dataclass
class CategoryRules:
    """Rules for categorizing regulatory requirements."""
    
    # Keywords for different requirement types
    governance_keywords: Set[str] = None
    conduct_keywords: Set[str] = None
    reporting_keywords: Set[str] = None
    record_keeping_keywords: Set[str] = None
    risk_management_keywords: Set[str] = None
    client_protection_keywords: Set[str] = None
    
    # Severity indicators
    high_severity_phrases: Set[str] = None
    medium_severity_phrases: Set[str] = None
    low_severity_phrases: Set[str] = None
    
    # Firm type indicators
    firm_type_indicators: Dict[str, List[FCAFirmType]] = None
    
    def __post_init__(self):
        """Initialize default rule sets if not provided."""
        
        if self.governance_keywords is None:
            self.governance_keywords = {
                'governance', 'board', 'director', 'management', 'oversight', 'structure',
                'arrangements', 'responsibility', 'accountability', 'delegation', 'authority',
                'senior', 'executive', 'committee', 'framework', 'policy', 'strategy'
            }
        
        if self.conduct_keywords is None:
            self.conduct_keywords = {
                'customer', 'client', 'consumer', 'conduct', 'fair', 'treating', 'outcome',
                'advice', 'recommendation', 'suitability', 'appropriateness', 'best', 'interest',
                'disclosure', 'transparency', 'communication', 'marketing', 'promotion'
            }
        
        if self.reporting_keywords is None:
            self.reporting_keywords = {
                'report', 'reporting', 'notification', 'notify', 'inform', 'return',
                'submission', 'filing', 'disclosure', 'publication', 'data', 'information',
                'regular', 'periodic', 'annual', 'quarterly', 'monthly'
            }
        
        if self.record_keeping_keywords is None:
            self.record_keeping_keywords = {
                'record', 'records', 'maintain', 'retention', 'preserve', 'documentation',
                'document', 'evidence', 'audit', 'trail', 'register', 'log', 'file',
                'archive', 'storage', 'backup', 'retrieval'
            }
        
        if self.risk_management_keywords is None:
            self.risk_management_keywords = {
                'risk', 'risks', 'control', 'controls', 'mitigation', 'assessment',
                'management', 'monitoring', 'identification', 'measurement', 'exposure',
                'limit', 'threshold', 'tolerance', 'appetite', 'framework', 'system'
            }
        
        if self.client_protection_keywords is None:
            self.client_protection_keywords = {
                'protection', 'safeguarding', 'segregation', 'security', 'safe', 'custody',
                'asset', 'money', 'fund', 'client', 'customer', 'compensation', 'insurance',
                'guarantee', 'cover', 'protection'
            }
        
        if self.high_severity_phrases is None:
            self.high_severity_phrases = {
                'must not', 'prohibited', 'shall not', 'forbidden', 'breach', 'violation',
                'criminal', 'offence', 'penalty', 'enforcement', 'immediate', 'urgent',
                'critical', 'essential', 'mandatory', 'required'
            }
        
        if self.medium_severity_phrases is None:
            self.medium_severity_phrases = {
                'should', 'ought', 'expected', 'appropriate', 'reasonable', 'adequate',
                'sufficient', 'proper', 'suitable', 'effective', 'robust', 'sound'
            }
        
        if self.low_severity_phrases is None:
            self.low_severity_phrases = {
                'may', 'might', 'could', 'consider', 'guidance', 'example', 'illustration',
                'suggest', 'recommend', 'advisable', 'beneficial', 'helpful'
            }
        
        if self.firm_type_indicators is None:
            self.firm_type_indicators = {
                'investment': [FCAFirmType.INVESTMENT_FIRM],
                'bank': [FCAFirmType.BANK],
                'insurer': [FCAFirmType.INSURER],
                'insurance': [FCAFirmType.INSURER],
                'payment': [FCAFirmType.PAYMENT_INSTITUTION],
                'credit': [FCAFirmType.CONSUMER_CREDIT_FIRM],
                'mortgage': [FCAFirmType.MORTGAGE_BROKER],
                'lending': [FCAFirmType.CONSUMER_CREDIT_FIRM, FCAFirmType.BANK],
                'custody': [FCAFirmType.INVESTMENT_FIRM, FCAFirmType.BANK],
                'advice': [FCAFirmType.INVESTMENT_FIRM, FCAFirmType.MORTGAGE_BROKER],
            }


class BaseCategorizer(ABC):
    """Abstract base class for requirement categorizers."""
    
    def __init__(self, rules: Optional[CategoryRules] = None):
        self.rules = rules or CategoryRules()
    
    @abstractmethod
    async def categorize_requirement(self, requirement: ExtractedRequirement) -> CategoryResult:
        """Categorize a single requirement."""
        pass
    
    async def categorize_batch(self, requirements: List[ExtractedRequirement]) -> List[CategoryResult]:
        """Categorize multiple requirements."""
        results = []
        for requirement in requirements:
            try:
                result = await self.categorize_requirement(requirement)
                results.append(result)
            except Exception as e:
                logger.error(f"Error categorizing requirement {requirement.section_id}: {e}")
                continue
        return results


class KeywordBasedCategorizer(BaseCategorizer):
    """Categorizer using keyword matching and rule-based logic."""
    
    async def categorize_requirement(self, requirement: ExtractedRequirement) -> CategoryResult:
        """Categorize requirement using keyword matching."""
        
        text = requirement.requirement_text.lower()
        words = set(text.split())
        
        # Primary category classification
        category_scores = {}
        keywords_matched = []
        
        # Score each category based on keyword matches
        category_scores[RequirementType.GOVERNANCE] = self._calculate_keyword_score(
            words, self.rules.governance_keywords, keywords_matched
        )
        category_scores[RequirementType.CONDUCT] = self._calculate_keyword_score(
            words, self.rules.conduct_keywords, keywords_matched
        )
        category_scores[RequirementType.REPORTING] = self._calculate_keyword_score(
            words, self.rules.reporting_keywords, keywords_matched
        )
        category_scores[RequirementType.RECORD_KEEPING] = self._calculate_keyword_score(
            words, self.rules.record_keeping_keywords, keywords_matched
        )
        category_scores[RequirementType.RISK_MANAGEMENT] = self._calculate_keyword_score(
            words, self.rules.risk_management_keywords, keywords_matched
        )
        category_scores[RequirementType.CLIENT_PROTECTION] = self._calculate_keyword_score(
            words, self.rules.client_protection_keywords, keywords_matched
        )
        
        # Determine primary and secondary categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        primary_category = sorted_categories[0][0]
        secondary_categories = [cat for cat, score in sorted_categories[1:] if score > 0.3]
        
        # Assess severity
        severity = self._assess_severity(text)
        
        # Determine applicability scope
        applicability_scope, firm_types = self._determine_applicability(text)
        
        # Identify business functions
        business_functions = self._identify_business_functions(text, primary_category)
        
        # Assess regulatory impact
        regulatory_impact = self._assess_regulatory_impact(text, primary_category)
        
        # Calculate confidence
        max_score = sorted_categories[0][1]
        confidence_score = min(max_score, 1.0)
        confidence = self._determine_confidence_level(confidence_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_category, secondary_categories, severity, keywords_matched
        )
        
        return CategoryResult(
            requirement_id=requirement.section_id,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            severity=severity,
            applicability_scope=applicability_scope,
            applicable_firm_types=firm_types,
            business_functions=business_functions,
            regulatory_impact=regulatory_impact,
            confidence=confidence,
            confidence_score=confidence_score,
            reasoning=reasoning,
            keywords_matched=keywords_matched
        )
    
    def _calculate_keyword_score(self, words: Set[str], category_keywords: Set[str], 
                                matched_keywords: List[str]) -> float:
        """Calculate score for category based on keyword matches."""
        matches = words.intersection(category_keywords)
        matched_keywords.extend(matches)
        
        if not matches:
            return 0.0
        
        # Score based on number of matches and keyword strength
        base_score = len(matches) / len(category_keywords)
        
        # Boost score for multiple matches
        if len(matches) > 1:
            base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def _assess_severity(self, text: str) -> SeverityLevel:
        """Assess severity level based on language used."""
        
        # Check for high severity indicators
        for phrase in self.rules.high_severity_phrases:
            if phrase in text:
                return SeverityLevel.HIGH
        
        # Check for low severity indicators
        for phrase in self.rules.low_severity_phrases:
            if phrase in text:
                return SeverityLevel.LOW
        
        # Check for medium severity indicators or default
        for phrase in self.rules.medium_severity_phrases:
            if phrase in text:
                return SeverityLevel.MEDIUM
        
        # Default to medium if no clear indicators
        return SeverityLevel.MEDIUM
    
    def _determine_applicability(self, text: str) -> Tuple[ApplicabilityScope, List[FCAFirmType]]:
        """Determine which firm types the requirement applies to."""
        
        applicable_firms = []
        
        # Check for specific firm type mentions
        for indicator, firm_types in self.rules.firm_type_indicators.items():
            if indicator in text:
                applicable_firms.extend(firm_types)
        
        # Remove duplicates
        applicable_firms = list(set(applicable_firms))
        
        # Determine scope
        if not applicable_firms:
            scope = ApplicabilityScope.ALL_FIRMS
        elif len(applicable_firms) == 1:
            if applicable_firms[0] == FCAFirmType.INVESTMENT_FIRM:
                scope = ApplicabilityScope.INVESTMENT_FIRMS
            elif applicable_firms[0] == FCAFirmType.BANK:
                scope = ApplicabilityScope.BANKS
            elif applicable_firms[0] == FCAFirmType.INSURER:
                scope = ApplicabilityScope.INSURERS
            else:
                scope = ApplicabilityScope.SPECIFIC_ACTIVITIES
        else:
            scope = ApplicabilityScope.SPECIFIC_ACTIVITIES
        
        return scope, applicable_firms
    
    def _identify_business_functions(self, text: str, primary_category: RequirementType) -> List[BusinessFunction]:
        """Identify business functions affected by the requirement."""
        
        functions = []
        
        # Map requirement types to likely business functions
        type_function_map = {
            RequirementType.GOVERNANCE: [BusinessFunction.GOVERNANCE, BusinessFunction.COMPLIANCE],
            RequirementType.CONDUCT: [BusinessFunction.CLIENT_FACING, BusinessFunction.COMPLIANCE],
            RequirementType.REPORTING: [BusinessFunction.REPORTING, BusinessFunction.COMPLIANCE],
            RequirementType.RECORD_KEEPING: [BusinessFunction.OPERATIONS, BusinessFunction.COMPLIANCE],
            RequirementType.RISK_MANAGEMENT: [BusinessFunction.RISK_MANAGEMENT, BusinessFunction.GOVERNANCE],
            RequirementType.CLIENT_PROTECTION: [BusinessFunction.CLIENT_FACING, BusinessFunction.CUSTODY],
        }
        
        # Add functions based on primary category
        if primary_category in type_function_map:
            functions.extend(type_function_map[primary_category])
        
        # Check for specific function keywords
        function_keywords = {
            'trading': BusinessFunction.TRADING,
            'lending': BusinessFunction.LENDING,
            'advice': BusinessFunction.INVESTMENT_ADVICE,
            'custody': BusinessFunction.CUSTODY,
            'payment': BusinessFunction.PAYMENTS,
            'audit': BusinessFunction.AUDIT,
        }
        
        for keyword, function in function_keywords.items():
            if keyword in text:
                functions.append(function)
        
        return list(set(functions))  # Remove duplicates
    
    def _assess_regulatory_impact(self, text: str, primary_category: RequirementType) -> List[RegulatoryImpact]:
        """Assess the type of regulatory impact."""
        
        impacts = []
        
        # Impact keywords
        impact_keywords = {
            'structure': RegulatoryImpact.STRUCTURAL,
            'process': RegulatoryImpact.PROCEDURAL,
            'procedure': RegulatoryImpact.PROCEDURAL,
            'document': RegulatoryImpact.DOCUMENTATION,
            'record': RegulatoryImpact.DOCUMENTATION,
            'report': RegulatoryImpact.REPORTING,
            'training': RegulatoryImpact.TRAINING,
            'system': RegulatoryImpact.TECHNOLOGY,
            'technology': RegulatoryImpact.TECHNOLOGY,
            'monitor': RegulatoryImpact.MONITORING,
            'ongoing': RegulatoryImpact.MONITORING,
        }
        
        # Check for impact keywords
        for keyword, impact in impact_keywords.items():
            if keyword in text:
                impacts.append(impact)
        
        # Default impacts based on requirement type
        if not impacts:
            if primary_category == RequirementType.GOVERNANCE:
                impacts.append(RegulatoryImpact.STRUCTURAL)
            elif primary_category == RequirementType.REPORTING:
                impacts.append(RegulatoryImpact.REPORTING)
            elif primary_category == RequirementType.RECORD_KEEPING:
                impacts.append(RegulatoryImpact.DOCUMENTATION)
            else:
                impacts.append(RegulatoryImpact.PROCEDURAL)
        
        return list(set(impacts))  # Remove duplicates
    
    def _determine_confidence_level(self, score: float) -> CategoryConfidence:
        """Convert numerical score to confidence level."""
        if score >= 0.8:
            return CategoryConfidence.HIGH
        elif score >= 0.6:
            return CategoryConfidence.MEDIUM
        elif score >= 0.4:
            return CategoryConfidence.LOW
        else:
            return CategoryConfidence.UNCERTAIN
    
    def _generate_reasoning(self, primary_category: RequirementType, 
                           secondary_categories: List[RequirementType],
                           severity: SeverityLevel, keywords_matched: List[str]) -> str:
        """Generate human-readable reasoning for categorization."""
        
        reasoning_parts = []
        
        # Primary category
        reasoning_parts.append(f"Classified as {primary_category.value}")
        
        # Secondary categories
        if secondary_categories:
            secondary_names = [cat.value for cat in secondary_categories]
            reasoning_parts.append(f"with secondary elements of {', '.join(secondary_names)}")
        
        # Severity
        reasoning_parts.append(f"assessed as {severity.value} severity")
        
        # Keywords
        if keywords_matched:
            top_keywords = list(set(keywords_matched))[:5]  # Top 5 unique keywords
            reasoning_parts.append(f"based on keywords: {', '.join(top_keywords)}")
        
        return "; ".join(reasoning_parts)


class CategorizationPipeline:
    """Main pipeline for requirement categorization."""
    
    def __init__(self, categorizer: Optional[BaseCategorizer] = None):
        self.categorizer = categorizer or KeywordBasedCategorizer()
    
    async def categorize_requirements(self, requirements: List[ExtractedRequirement]) -> List[CategoryResult]:
        """Categorize a list of requirements."""
        
        logger.info(f"Starting categorization of {len(requirements)} requirements")
        
        results = await self.categorizer.categorize_batch(requirements)
        
        # Generate summary statistics
        self._log_categorization_stats(results)
        
        return results
    
    def _log_categorization_stats(self, results: List[CategoryResult]):
        """Log categorization statistics."""
        
        if not results:
            logger.warning("No categorization results to analyze")
            return
        
        # Count by category
        category_counts = {}
        confidence_counts = {}
        
        for result in results:
            category = result.primary_category.value
            confidence = result.confidence.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        logger.info(f"Categorization complete: {len(results)} requirements processed")
        logger.info(f"Category distribution: {category_counts}")
        logger.info(f"Confidence distribution: {confidence_counts}")
        
        # Average confidence score
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        logger.info(f"Average confidence score: {avg_confidence:.2f}")


# Utility functions
async def create_default_categorization_pipeline() -> CategorizationPipeline:
    """Create default categorization pipeline."""
    rules = CategoryRules()
    categorizer = KeywordBasedCategorizer(rules)
    return CategorizationPipeline(categorizer)


def export_categorization_results(results: List[CategoryResult], output_file: str):
    """Export categorization results to JSON file."""
    import json
    
    # Convert to dict for JSON serialization
    results_dict = [result.dict() for result in results]
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    logger.info(f"Exported {len(results)} categorization results to {output_file}")