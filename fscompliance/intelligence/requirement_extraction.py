"""Requirement extraction algorithms for regulatory compliance analysis."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..models import ConductRequirement, FCASourcebook, RequirementType, SeverityLevel
from ..knowledge import ExtractedRequirement, ParsedSection

logger = logging.getLogger(__name__)


class ExtractionAlgorithm(str, Enum):
    """Types of requirement extraction algorithms."""
    RULE_BASED = "rule_based"
    NLP_PATTERN = "nlp_pattern"
    ML_CLASSIFICATION = "ml_classification"
    LLM_EXTRACTION = "llm_extraction"
    HYBRID = "hybrid"


class RequirementScope(str, Enum):
    """Scope of requirement applicability."""
    MANDATORY = "mandatory"          # Must comply
    CONDITIONAL = "conditional"      # Applies under conditions
    GUIDANCE = "guidance"           # Recommended practice
    EVIDENTIAL = "evidential"       # Supporting evidence
    DEFINITIONAL = "definitional"   # Definitions and interpretations


@dataclass
class ExtractionContext:
    """Context for requirement extraction."""
    
    sourcebook: FCASourcebook
    chapter: str
    section_hierarchy: List[str]
    document_type: str = "regulation"
    firm_types: List[str] = None
    business_functions: List[str] = None
    
    def __post_init__(self):
        if self.firm_types is None:
            self.firm_types = []
        if self.business_functions is None:
            self.business_functions = []


class ExtractedRequirementEnhanced(BaseModel):
    """Enhanced extracted requirement with detailed analysis."""
    
    # Core requirement data
    requirement_id: str = Field(..., description="Unique requirement identifier")
    source_section: str = Field(..., description="Source section reference")
    requirement_text: str = Field(..., description="Full requirement text")
    
    # Classification
    requirement_type: RequirementType = Field(..., description="Type of requirement")
    requirement_scope: RequirementScope = Field(..., description="Scope of applicability")
    severity: SeverityLevel = Field(..., description="Severity level")
    
    # Extraction metadata
    extraction_algorithm: ExtractionAlgorithm = Field(..., description="Algorithm used for extraction")
    extraction_confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Linguistic analysis
    modal_verbs: List[str] = Field(default_factory=list, description="Modal verbs found (must, shall, should)")
    action_verbs: List[str] = Field(default_factory=list, description="Action verbs required")
    regulatory_entities: List[str] = Field(default_factory=list, description="Regulatory entities mentioned")
    
    # Applicability analysis
    applicable_firms: List[str] = Field(default_factory=list, description="Applicable firm types")
    applicable_functions: List[str] = Field(default_factory=list, description="Applicable business functions")
    conditions: List[str] = Field(default_factory=list, description="Conditions for applicability")
    
    # Compliance implications
    deadlines: List[str] = Field(default_factory=list, description="Compliance deadlines")
    documentation_required: List[str] = Field(default_factory=list, description="Required documentation")
    reporting_obligations: List[str] = Field(default_factory=list, description="Reporting requirements")
    
    # Cross-references
    cross_references: List[str] = Field(default_factory=list, description="Referenced sections")
    related_requirements: List[str] = Field(default_factory=list, description="Related requirement IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "requirement_id": "SYSC.4.1.1.001",
                "source_section": "SYSC.4.1.1",
                "requirement_text": "A firm must have robust governance arrangements...",
                "requirement_type": "governance",
                "requirement_scope": "mandatory",
                "severity": "high",
                "extraction_confidence": 0.92,
                "modal_verbs": ["must"],
                "action_verbs": ["have", "establish", "maintain"],
                "applicable_firms": ["investment_firm", "bank"],
                "deadlines": ["immediate"],
                "cross_references": ["SYSC.4.1.2", "SYSC.5.1.1"]
            }
        }


class BaseRequirementExtractor(ABC):
    """Abstract base class for requirement extraction algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extraction_stats = {
            "requirements_extracted": 0,
            "high_confidence_extractions": 0,
            "processing_errors": 0,
            "processing_time_seconds": 0.0
        }
    
    @abstractmethod
    async def extract_requirements(
        self, 
        content: str, 
        context: ExtractionContext
    ) -> List[ExtractedRequirementEnhanced]:
        """Extract requirements from content using the specific algorithm."""
        pass
    
    @abstractmethod
    def get_algorithm_type(self) -> ExtractionAlgorithm:
        """Get the type of extraction algorithm."""
        pass
    
    async def extract_from_section(
        self, 
        section: ParsedSection
    ) -> List[ExtractedRequirementEnhanced]:
        """Extract requirements from a parsed section."""
        
        context = ExtractionContext(
            sourcebook=section.sourcebook,
            chapter=section.chapter,
            section_hierarchy=[section.section_id],
            document_type="regulation"
        )
        
        return await self.extract_requirements(section.content, context)
    
    def _generate_requirement_id(self, section_id: str, index: int) -> str:
        """Generate unique requirement ID."""
        return f"{section_id}.{index:03d}"
    
    def _calculate_confidence_score(
        self, 
        features: Dict[str, Any], 
        context: ExtractionContext
    ) -> float:
        """Calculate confidence score based on extraction features."""
        
        base_confidence = 0.5
        
        # Boost for strong modal verbs
        strong_modals = {"must", "shall", "required"}
        if any(modal in features.get("modal_verbs", []) for modal in strong_modals):
            base_confidence += 0.2
        
        # Boost for clear action verbs
        if len(features.get("action_verbs", [])) > 0:
            base_confidence += 0.1
        
        # Boost for specific firm type mentions
        if len(features.get("applicable_firms", [])) > 0:
            base_confidence += 0.1
        
        # Boost for cross-references
        if len(features.get("cross_references", [])) > 0:
            base_confidence += 0.1
        
        # Context-based adjustments
        if context.sourcebook in [FCASourcebook.SYSC, FCASourcebook.COBS]:
            base_confidence += 0.05  # Well-structured sourcebooks
        
        return min(base_confidence, 1.0)


class RuleBasedRequirementExtractor(BaseRequirementExtractor):
    """Rule-based requirement extraction using linguistic patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.patterns = self._initialize_patterns()
    
    def get_algorithm_type(self) -> ExtractionAlgorithm:
        return ExtractionAlgorithm.RULE_BASED
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize extraction patterns."""
        
        return {
            # Modal verb patterns for requirement identification
            "modal_verbs": {
                "mandatory": [r"\bmust\b", r"\bshall\b", r"\brequired to\b", r"\bobliged to\b"],
                "conditional": [r"\bshould\b", r"\bought to\b", r"\bexpected to\b"],
                "guidance": [r"\bmay\b", r"\bcould\b", r"\bmight\b", r"\bconsider\b"]
            },
            
            # Action verb patterns
            "action_verbs": [
                r"\b(establish|maintain|implement|ensure|provide|create|develop)\b",
                r"\b(monitor|review|assess|evaluate|measure|test)\b",
                r"\b(document|record|report|notify|disclose|publish)\b",
                r"\b(train|educate|inform|communicate|demonstrate)\b"
            ],
            
            # Firm type patterns
            "firm_types": {
                "investment_firm": [r"investment firm[s]?", r"investment compan(?:y|ies)"],
                "bank": [r"bank[s]?", r"credit institution[s]?", r"deposit taker[s]?"],
                "insurer": [r"insurer[s]?", r"insurance compan(?:y|ies)", r"insurance undertaking[s]?"],
                "payment_institution": [r"payment institution[s]?", r"payment service provider[s]?"],
                "electronic_money": [r"electronic money institution[s]?", r"e-money issuer[s]?"]
            },
            
            # Business function patterns
            "business_functions": {
                "governance": [r"governance", r"oversight", r"management arrangement[s]?"],
                "risk_management": [r"risk management", r"risk control[s]?", r"risk assessment"],
                "compliance": [r"compliance", r"regulatory", r"supervisory"],
                "audit": [r"audit", r"internal audit", r"assurance"],
                "trading": [r"trading", r"dealing", r"market making"],
                "client_facing": [r"client", r"customer", r"retail", r"advisory"]
            },
            
            # Deadline patterns
            "deadlines": [
                r"within (\d+) (?:business )?days?",
                r"within (\d+) weeks?",
                r"within (\d+) months?",
                r"immediately",
                r"without delay",
                r"as soon as (?:reasonably )?practicable",
                r"annually",
                r"quarterly",
                r"monthly"
            ],
            
            # Documentation patterns
            "documentation": [
                r"document[s]?",
                r"record[s]?",
                r"register[s]?",
                r"policy",
                r"procedure[s]?",
                r"manual[s]?",
                r"evidence"
            ],
            
            # Cross-reference patterns
            "cross_references": [
                r"[A-Z]{3,5}\.?\s*\d+(?:\.?\d+)*(?:\.?\d+)*",  # e.g., SYSC.4.1.1
                r"(?:section|rule|paragraph|chapter)\s+\d+",
                r"see\s+[A-Z]{3,5}",
                r"in accordance with\s+[A-Z]{3,5}"
            ]
        }
    
    async def extract_requirements(
        self, 
        content: str, 
        context: ExtractionContext
    ) -> List[ExtractedRequirementEnhanced]:
        """Extract requirements using rule-based patterns."""
        
        start_time = datetime.utcnow()
        requirements = []
        
        try:
            # Split content into sentences for requirement-level analysis
            sentences = self._split_into_sentences(content)
            
            requirement_index = 0
            
            for sentence in sentences:
                # Check if sentence contains requirement indicators
                if self._is_requirement_sentence(sentence):
                    
                    # Extract features from sentence
                    features = self._extract_features(sentence, context)
                    
                    # Determine requirement scope and type
                    scope = self._determine_requirement_scope(features)
                    req_type = self._classify_requirement_type(features, context)
                    severity = self._assess_severity(features)
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence_score(features, context)
                    
                    # Create enhanced requirement
                    requirement = ExtractedRequirementEnhanced(
                        requirement_id=self._generate_requirement_id(
                            context.section_hierarchy[0], 
                            requirement_index
                        ),
                        source_section=context.section_hierarchy[0],
                        requirement_text=sentence.strip(),
                        requirement_type=req_type,
                        requirement_scope=scope,
                        severity=severity,
                        extraction_algorithm=self.get_algorithm_type(),
                        extraction_confidence=confidence,
                        modal_verbs=features.get("modal_verbs", []),
                        action_verbs=features.get("action_verbs", []),
                        regulatory_entities=features.get("regulatory_entities", []),
                        applicable_firms=features.get("applicable_firms", []),
                        applicable_functions=features.get("applicable_functions", []),
                        conditions=features.get("conditions", []),
                        deadlines=features.get("deadlines", []),
                        documentation_required=features.get("documentation", []),
                        reporting_obligations=features.get("reporting", []),
                        cross_references=features.get("cross_references", [])
                    )
                    
                    requirements.append(requirement)
                    requirement_index += 1
            
            # Update statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            self.extraction_stats["requirements_extracted"] += len(requirements)
            self.extraction_stats["high_confidence_extractions"] += sum(
                1 for req in requirements if req.extraction_confidence >= 0.8
            )
            self.extraction_stats["processing_time_seconds"] += processing_time
            
            logger.info(f"Extracted {len(requirements)} requirements using rule-based algorithm")
            
            return requirements
            
        except Exception as e:
            self.extraction_stats["processing_errors"] += 1
            logger.error(f"Error in rule-based requirement extraction: {e}")
            return []
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences for analysis."""
        
        # Simple sentence splitting - could be enhanced with NLP libraries
        sentences = re.split(r'[.!?]+', content)
        
        # Filter out very short sentences and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _is_requirement_sentence(self, sentence: str) -> bool:
        """Check if sentence contains requirement indicators."""
        
        sentence_lower = sentence.lower()
        
        # Check for modal verbs
        mandatory_modals = ["must", "shall", "required to", "obliged to"]
        conditional_modals = ["should", "ought to", "expected to"]
        guidance_modals = ["may", "could", "might", "consider"]
        
        all_modals = mandatory_modals + conditional_modals + guidance_modals
        
        return any(modal in sentence_lower for modal in all_modals)
    
    def _extract_features(self, sentence: str, context: ExtractionContext) -> Dict[str, List[str]]:
        """Extract linguistic and regulatory features from sentence."""
        
        features = {
            "modal_verbs": [],
            "action_verbs": [],
            "regulatory_entities": [],
            "applicable_firms": [],
            "applicable_functions": [],
            "conditions": [],
            "deadlines": [],
            "documentation": [],
            "reporting": [],
            "cross_references": []
        }
        
        sentence_lower = sentence.lower()
        
        # Extract modal verbs
        for scope, modals in self.patterns["modal_verbs"].items():
            for modal_pattern in modals:
                if re.search(modal_pattern, sentence_lower):
                    features["modal_verbs"].append(modal_pattern.strip(r'\b'))
        
        # Extract action verbs
        for action_pattern in self.patterns["action_verbs"]:
            matches = re.findall(action_pattern, sentence_lower)
            features["action_verbs"].extend(matches)
        
        # Extract firm types
        for firm_type, patterns in self.patterns["firm_types"].items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    features["applicable_firms"].append(firm_type)
        
        # Extract business functions
        for function, patterns in self.patterns["business_functions"].items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    features["applicable_functions"].append(function)
        
        # Extract deadlines
        for deadline_pattern in self.patterns["deadlines"]:
            matches = re.findall(deadline_pattern, sentence_lower)
            if matches:
                features["deadlines"].extend(matches)
        
        # Extract documentation requirements
        for doc_pattern in self.patterns["documentation"]:
            if re.search(doc_pattern, sentence_lower):
                features["documentation"].append(doc_pattern)
        
        # Extract cross-references
        for ref_pattern in self.patterns["cross_references"]:
            matches = re.findall(ref_pattern, sentence, re.IGNORECASE)
            features["cross_references"].extend(matches)
        
        # Remove duplicates
        for key in features:
            features[key] = list(set(features[key]))
        
        return features
    
    def _determine_requirement_scope(self, features: Dict[str, List[str]]) -> RequirementScope:
        """Determine the scope of the requirement based on features."""
        
        modal_verbs = [verb.lower() for verb in features.get("modal_verbs", [])]
        
        # Mandatory requirements
        if any(modal in modal_verbs for modal in ["must", "shall", "required", "obliged"]):
            return RequirementScope.MANDATORY
        
        # Conditional requirements
        elif any(modal in modal_verbs for modal in ["should", "ought", "expected"]):
            return RequirementScope.CONDITIONAL
        
        # Guidance
        elif any(modal in modal_verbs for modal in ["may", "could", "might", "consider"]):
            return RequirementScope.GUIDANCE
        
        # Default to guidance for unclear cases
        return RequirementScope.GUIDANCE
    
    def _classify_requirement_type(
        self, 
        features: Dict[str, List[str]], 
        context: ExtractionContext
    ) -> RequirementType:
        """Classify the type of requirement based on features and context."""
        
        action_verbs = [verb.lower() for verb in features.get("action_verbs", [])]
        functions = features.get("applicable_functions", [])
        
        # Governance requirements
        if any(func in functions for func in ["governance", "oversight"]) or \
           any(verb in action_verbs for verb in ["establish", "maintain", "ensure"]):
            return RequirementType.GOVERNANCE
        
        # Risk management requirements
        if "risk_management" in functions or \
           any(verb in action_verbs for verb in ["assess", "monitor", "control"]):
            return RequirementType.RISK_MANAGEMENT
        
        # Conduct requirements
        if any(func in functions for func in ["client_facing", "advisory"]) or \
           context.sourcebook == FCASourcebook.COBS:
            return RequirementType.CONDUCT
        
        # Reporting requirements
        if any(verb in action_verbs for verb in ["report", "notify", "disclose", "publish"]) or \
           len(features.get("reporting", [])) > 0:
            return RequirementType.REPORTING
        
        # Record keeping requirements
        if len(features.get("documentation", [])) > 0 or \
           any(verb in action_verbs for verb in ["document", "record"]):
            return RequirementType.RECORD_KEEPING
        
        # Client protection requirements
        if any(word in features.get("regulatory_entities", []) for word in ["client", "customer"]):
            return RequirementType.CLIENT_PROTECTION
        
        # Default based on sourcebook
        if context.sourcebook == FCASourcebook.SYSC:
            return RequirementType.GOVERNANCE
        elif context.sourcebook == FCASourcebook.COBS:
            return RequirementType.CONDUCT
        else:
            return RequirementType.GOVERNANCE
    
    def _assess_severity(self, features: Dict[str, List[str]]) -> SeverityLevel:
        """Assess the severity level of the requirement."""
        
        modal_verbs = [verb.lower() for verb in features.get("modal_verbs", [])]
        
        # High severity for mandatory requirements
        if any(modal in modal_verbs for modal in ["must", "shall"]):
            return SeverityLevel.HIGH
        
        # Medium severity for conditional requirements
        elif any(modal in modal_verbs for modal in ["should", "ought", "expected"]):
            return SeverityLevel.MEDIUM
        
        # Low severity for guidance
        else:
            return SeverityLevel.LOW


class RequirementExtractionPipeline:
    """Pipeline for orchestrating requirement extraction algorithms."""
    
    def __init__(self, extractors: List[BaseRequirementExtractor]):
        self.extractors = extractors
        self.pipeline_stats = {
            "total_requirements_extracted": 0,
            "algorithms_used": len(extractors),
            "processing_sessions": 0
        }
    
    async def extract_from_sections(
        self, 
        sections: List[ParsedSection]
    ) -> List[ExtractedRequirementEnhanced]:
        """Extract requirements from multiple sections using all configured extractors."""
        
        all_requirements = []
        
        logger.info(f"Starting requirement extraction for {len(sections)} sections using {len(self.extractors)} algorithms")
        
        for section in sections:
            for extractor in self.extractors:
                try:
                    requirements = await extractor.extract_from_section(section)
                    all_requirements.extend(requirements)
                    
                    logger.debug(f"Extracted {len(requirements)} requirements from {section.section_id} using {extractor.get_algorithm_type().value}")
                    
                except Exception as e:
                    logger.error(f"Error extracting from {section.section_id} with {extractor.get_algorithm_type().value}: {e}")
                    continue
        
        # Deduplicate requirements
        deduplicated_requirements = self._deduplicate_requirements(all_requirements)
        
        # Update statistics
        self.pipeline_stats["total_requirements_extracted"] += len(deduplicated_requirements)
        self.pipeline_stats["processing_sessions"] += 1
        
        logger.info(f"Extraction pipeline completed: {len(deduplicated_requirements)} unique requirements extracted")
        
        return deduplicated_requirements
    
    def _deduplicate_requirements(
        self, 
        requirements: List[ExtractedRequirementEnhanced]
    ) -> List[ExtractedRequirementEnhanced]:
        """Remove duplicate requirements based on text similarity."""
        
        deduplicated = []
        seen_texts = set()
        
        for requirement in requirements:
            # Use normalized text for deduplication
            normalized_text = requirement.requirement_text.lower().strip()
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                deduplicated.append(requirement)
            else:
                # Keep the requirement with higher confidence
                for i, existing in enumerate(deduplicated):
                    if existing.requirement_text.lower().strip() == normalized_text:
                        if requirement.extraction_confidence > existing.extraction_confidence:
                            deduplicated[i] = requirement
                        break
        
        return deduplicated
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        extractor_stats = {}
        for extractor in self.extractors:
            extractor_stats[extractor.get_algorithm_type().value] = extractor.extraction_stats
        
        return {
            **self.pipeline_stats,
            "extractor_statistics": extractor_stats
        }


# Factory functions
def create_rule_based_extractor(config: Optional[Dict[str, Any]] = None) -> RuleBasedRequirementExtractor:
    """Create rule-based requirement extractor."""
    if config is None:
        config = {}
    
    return RuleBasedRequirementExtractor(config)


def create_requirement_extraction_pipeline(
    algorithms: List[ExtractionAlgorithm] = None
) -> RequirementExtractionPipeline:
    """Create requirement extraction pipeline with specified algorithms."""
    
    if algorithms is None:
        algorithms = [ExtractionAlgorithm.RULE_BASED]
    
    extractors = []
    
    for algorithm in algorithms:
        if algorithm == ExtractionAlgorithm.RULE_BASED:
            extractors.append(create_rule_based_extractor())
        else:
            logger.warning(f"Algorithm {algorithm} not yet implemented, skipping")
    
    return RequirementExtractionPipeline(extractors)