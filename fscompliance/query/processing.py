"""Natural language query processing for compliance queries."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from .routing import ClassifiedQuery, QueryContext, QueryType, QueryIntent, QueryDomain

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Stages of query processing."""
    PREPROCESSING = "preprocessing"       # Text normalization, cleaning
    PARSING = "parsing"                  # Syntactic parsing
    UNDERSTANDING = "understanding"      # Semantic understanding
    INTENT_RESOLUTION = "intent_resolution"  # Resolve specific intent
    CONTEXT_ENRICHMENT = "context_enrichment"  # Add contextual information
    QUERY_EXPANSION = "query_expansion"  # Expand query with synonyms/related terms
    VALIDATION = "validation"            # Validate query completeness


class QueryElement(str, Enum):
    """Elements that can be extracted from queries."""
    SUBJECT = "subject"                 # What the query is about
    PREDICATE = "predicate"            # What action/relation is requested
    OBJECT = "object"                  # Target of the query
    QUALIFIER = "qualifier"            # Qualifiers (time, scope, conditions)
    CONSTRAINT = "constraint"          # Constraints or filters
    CONTEXT = "context"               # Contextual information


@dataclass
class ProcessedQueryElement:
    """A processed element of a query."""
    
    element_type: QueryElement
    original_text: str
    normalized_text: str
    confidence: float
    synonyms: List[str] = None
    related_terms: List[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.related_terms is None:
            self.related_terms = []


class ProcessedQuery(BaseModel):
    """A fully processed natural language query."""
    
    # Core query information
    query_id: str = Field(..., description="Unique query identifier")
    original_query: str = Field(..., description="Original query text")
    normalized_query: str = Field(..., description="Normalized query text")
    
    # Classification (from routing)
    classified_query: ClassifiedQuery = Field(..., description="Classification results")
    
    # Processed elements
    query_elements: List[ProcessedQueryElement] = Field(
        default_factory=list, description="Extracted query elements"
    )
    
    # Semantic understanding
    semantic_representation: Dict[str, Any] = Field(
        default_factory=dict, description="Semantic representation of query"
    )
    
    # Query expansion
    expanded_terms: Dict[str, List[str]] = Field(
        default_factory=dict, description="Expanded terms and synonyms"
    )
    
    # Processing metadata
    processing_algorithm: str = Field(..., description="Algorithm used for processing")
    processing_stages: List[str] = Field(default_factory=list, description="Processing stages completed")
    processing_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall processing confidence")
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Validation results
    is_valid: bool = Field(..., description="Whether query is valid and processable")
    validation_issues: List[str] = Field(default_factory=list, description="Validation issues found")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Query completeness score")
    
    # Context enrichment
    enriched_context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context information"
    )
    
    # Search preparation
    search_terms: List[str] = Field(default_factory=list, description="Terms prepared for search")
    search_filters: Dict[str, Any] = Field(default_factory=dict, description="Filters for search")
    search_scope: List[str] = Field(default_factory=list, description="Search scope limitations")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get high-level processing summary."""
        return {
            "query_type": self.classified_query.query_type.value,
            "intent": self.classified_query.query_intent.value,
            "domain": self.classified_query.query_domain.value,
            "elements_extracted": len(self.query_elements),
            "processing_confidence": self.processing_confidence,
            "is_valid": self.is_valid,
            "completeness_score": self.completeness_score,
            "search_terms_count": len(self.search_terms)
        }
    
    def get_search_ready_query(self) -> Dict[str, Any]:
        """Get query prepared for search execution."""
        return {
            "primary_terms": self.search_terms,
            "expanded_terms": self.expanded_terms,
            "filters": self.search_filters,
            "scope": self.search_scope,
            "semantic_context": self.semantic_representation,
            "query_type": self.classified_query.query_type.value,
            "domain": self.classified_query.query_domain.value
        }


class BaseQueryProcessor(ABC):
    """Abstract base class for natural language query processors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_stats = {
            "queries_processed": 0,
            "successful_processing": 0,
            "processing_errors": 0,
            "average_confidence": 0.0,
            "average_processing_time_ms": 0.0
        }
    
    @abstractmethod
    async def process_query(self, classified_query: ClassifiedQuery) -> ProcessedQuery:
        """Process a classified query into a structured format."""
        pass
    
    @abstractmethod
    def get_processor_name(self) -> str:
        """Get the name of the processing algorithm."""
        pass
    
    def _normalize_text(self, text: str) -> str:
        """Normalize query text."""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Handle contractions
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Normalize punctuation
        normalized = re.sub(r'[^\w\s\.]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized


class ComprehensiveQueryProcessor(BaseQueryProcessor):
    """Comprehensive natural language query processor."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_vocabulary = self._initialize_domain_vocabulary()
        self.processing_pipeline = self._initialize_processing_pipeline()
    
    def get_processor_name(self) -> str:
        return "comprehensive_nlp_processor"
    
    def _initialize_domain_vocabulary(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize domain-specific vocabulary and synonyms."""
        
        return {
            "governance": {
                "synonyms": {
                    "governance": ["oversight", "management", "control", "supervision"],
                    "board": ["directors", "governing body", "management board"],
                    "responsibility": ["accountability", "obligation", "duty"],
                    "framework": ["structure", "system", "arrangement", "setup"]
                },
                "related_terms": {
                    "governance": ["policy", "procedure", "committee", "delegation"],
                    "oversight": ["monitoring", "review", "supervision", "control"],
                    "management": ["leadership", "direction", "coordination"]
                }
            },
            "risk": {
                "synonyms": {
                    "risk": ["threat", "exposure", "hazard", "danger"],
                    "assessment": ["evaluation", "analysis", "review", "appraisal"],
                    "mitigation": ["reduction", "control", "management", "treatment"],
                    "monitoring": ["tracking", "surveillance", "observation", "review"]
                },
                "related_terms": {
                    "risk": ["control", "appetite", "tolerance", "profile"],
                    "assessment": ["identification", "measurement", "quantification"],
                    "control": ["procedure", "process", "mechanism", "safeguard"]
                }
            },
            "conduct": {
                "synonyms": {
                    "customer": ["client", "consumer", "retail client"],
                    "fair": ["equitable", "reasonable", "appropriate"],
                    "treatment": ["handling", "service", "dealing"],
                    "outcome": ["result", "consequence", "effect"]
                },
                "related_terms": {
                    "customer": ["protection", "service", "experience", "journey"],
                    "conduct": ["behavior", "practice", "standard", "ethics"],
                    "suitability": ["appropriateness", "matching", "fit"]
                }
            },
            "compliance": {
                "synonyms": {
                    "compliance": ["adherence", "conformity", "observance"],
                    "requirement": ["rule", "regulation", "obligation", "standard"],
                    "breach": ["violation", "non-compliance", "infringement"],
                    "monitoring": ["surveillance", "oversight", "tracking"]
                },
                "related_terms": {
                    "compliance": ["framework", "program", "culture", "testing"],
                    "requirement": ["standard", "guideline", "expectation"],
                    "reporting": ["disclosure", "notification", "submission"]
                }
            }
        }
    
    def _initialize_processing_pipeline(self) -> List[str]:
        """Initialize the processing pipeline stages."""
        
        return [
            "preprocessing",
            "entity_extraction",
            "element_parsing",
            "semantic_analysis",
            "context_enrichment",
            "query_expansion",
            "search_preparation",
            "validation"
        ]
    
    async def process_query(self, classified_query: ClassifiedQuery) -> ProcessedQuery:
        """Process a classified query comprehensively."""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing query {classified_query.query_id} with {len(self.processing_pipeline)} stages")
            
            # Initialize processed query
            processed_query = ProcessedQuery(
                query_id=classified_query.query_id,
                original_query=classified_query.original_query,
                normalized_query=self._normalize_text(classified_query.original_query),
                classified_query=classified_query,
                processing_algorithm=self.get_processor_name(),
                is_valid=True,
                completeness_score=0.0,
                processing_confidence=0.0
            )
            
            # Stage 1: Preprocessing
            await self._preprocess_query(processed_query)
            processed_query.processing_stages.append("preprocessing")
            
            # Stage 2: Entity extraction
            await self._extract_entities(processed_query)
            processed_query.processing_stages.append("entity_extraction")
            
            # Stage 3: Element parsing
            await self._parse_query_elements(processed_query)
            processed_query.processing_stages.append("element_parsing")
            
            # Stage 4: Semantic analysis
            await self._analyze_semantics(processed_query)
            processed_query.processing_stages.append("semantic_analysis")
            
            # Stage 5: Context enrichment
            await self._enrich_context(processed_query)
            processed_query.processing_stages.append("context_enrichment")
            
            # Stage 6: Query expansion
            await self._expand_query(processed_query)
            processed_query.processing_stages.append("query_expansion")
            
            # Stage 7: Search preparation
            await self._prepare_for_search(processed_query)
            processed_query.processing_stages.append("search_preparation")
            
            # Stage 8: Validation
            await self._validate_query(processed_query)
            processed_query.processing_stages.append("validation")
            
            # Calculate overall processing confidence
            processed_query.processing_confidence = self._calculate_processing_confidence(processed_query)
            
            # Update statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["successful_processing"] += 1
            
            # Update average confidence
            total_confidence = (self.processing_stats["average_confidence"] * 
                              (self.processing_stats["successful_processing"] - 1) + 
                              processed_query.processing_confidence)
            self.processing_stats["average_confidence"] = total_confidence / self.processing_stats["successful_processing"]
            
            # Update average processing time
            total_time = (self.processing_stats["average_processing_time_ms"] * 
                         (self.processing_stats["successful_processing"] - 1) + 
                         processing_time)
            self.processing_stats["average_processing_time_ms"] = total_time / self.processing_stats["successful_processing"]
            
            logger.info(f"Query processing completed: {processed_query.processing_confidence:.2f} confidence")
            
            return processed_query
            
        except Exception as e:
            self.processing_stats["processing_errors"] += 1
            logger.error(f"Error processing query: {e}")
            raise
    
    async def _preprocess_query(self, processed_query: ProcessedQuery):
        """Preprocess the query text."""
        
        # Additional normalization beyond basic text normalization
        normalized = processed_query.normalized_query
        
        # Handle regulatory reference formatting
        normalized = re.sub(r'([A-Z]+)\.?\s*(\d+(?:\.\d+)*)', r'\1.\2', normalized)
        
        # Normalize firm type references
        firm_type_mappings = {
            "investment firms": "investment firm",
            "banks": "bank",
            "insurers": "insurer",
            "payment institutions": "payment institution"
        }
        
        for plural, singular in firm_type_mappings.items():
            normalized = normalized.replace(plural, singular)
        
        processed_query.normalized_query = normalized
    
    async def _extract_entities(self, processed_query: ProcessedQuery):
        """Extract additional entities from the processed query."""
        
        # Use entities from classification and add more detailed extraction
        entities = processed_query.classified_query.entities.copy()
        
        # Extract additional regulatory concepts
        regulatory_concepts = self._extract_regulatory_concepts(processed_query.normalized_query)
        entities["regulatory_concepts"] = regulatory_concepts
        
        # Extract temporal references
        temporal_refs = self._extract_temporal_references(processed_query.normalized_query)
        entities["temporal_references"] = temporal_refs
        
        # Extract process/action references
        action_refs = self._extract_action_references(processed_query.normalized_query)
        entities["actions"] = action_refs
        
        # Update entities in semantic representation
        processed_query.semantic_representation["entities"] = entities
    
    async def _parse_query_elements(self, processed_query: ProcessedQuery):
        """Parse query into structured elements."""
        
        query_text = processed_query.normalized_query
        query_type = processed_query.classified_query.query_type
        
        elements = []
        
        # Extract subject (what the query is about)
        subject = self._extract_subject(query_text, query_type)
        if subject:
            elements.append(ProcessedQueryElement(
                element_type=QueryElement.SUBJECT,
                original_text=subject,
                normalized_text=subject.lower(),
                confidence=0.8
            ))
        
        # Extract predicate (what action/relation is requested)
        predicate = self._extract_predicate(query_text, query_type)
        if predicate:
            elements.append(ProcessedQueryElement(
                element_type=QueryElement.PREDICATE,
                original_text=predicate,
                normalized_text=predicate.lower(),
                confidence=0.8
            ))
        
        # Extract object (target of the query)
        obj = self._extract_object(query_text, query_type)
        if obj:
            elements.append(ProcessedQueryElement(
                element_type=QueryElement.OBJECT,
                original_text=obj,
                normalized_text=obj.lower(),
                confidence=0.7
            ))
        
        # Extract qualifiers and constraints
        qualifiers = self._extract_qualifiers(query_text)
        for qualifier in qualifiers:
            elements.append(ProcessedQueryElement(
                element_type=QueryElement.QUALIFIER,
                original_text=qualifier,
                normalized_text=qualifier.lower(),
                confidence=0.6
            ))
        
        processed_query.query_elements = elements
    
    async def _analyze_semantics(self, processed_query: ProcessedQuery):
        """Analyze semantic meaning of the query."""
        
        semantic_rep = processed_query.semantic_representation
        
        # Determine primary focus
        semantic_rep["primary_focus"] = self._determine_primary_focus(processed_query)
        
        # Identify relationships
        semantic_rep["relationships"] = self._identify_relationships(processed_query)
        
        # Determine scope
        semantic_rep["scope"] = self._determine_scope(processed_query)
        
        # Identify constraints
        semantic_rep["constraints"] = self._identify_constraints(processed_query)
    
    async def _enrich_context(self, processed_query: ProcessedQuery):
        """Enrich query with contextual information."""
        
        context = processed_query.enriched_context
        
        # Add domain-specific context
        domain = processed_query.classified_query.query_domain
        context["domain_context"] = self._get_domain_context(domain)
        
        # Add regulatory context
        if processed_query.semantic_representation.get("entities", {}).get("regulatory_references"):
            context["regulatory_context"] = self._get_regulatory_context(
                processed_query.semantic_representation["entities"]["regulatory_references"]
            )
        
        # Add temporal context
        if processed_query.semantic_representation.get("entities", {}).get("temporal_references"):
            context["temporal_context"] = self._get_temporal_context(
                processed_query.semantic_representation["entities"]["temporal_references"]
            )
    
    async def _expand_query(self, processed_query: ProcessedQuery):
        """Expand query with synonyms and related terms."""
        
        domain = processed_query.classified_query.query_domain.value
        domain_vocab = self.domain_vocabulary.get(domain, {})
        
        expanded_terms = {}
        
        # Expand based on query elements
        for element in processed_query.query_elements:
            element_text = element.normalized_text
            
            # Find synonyms
            synonyms = []
            for term, term_synonyms in domain_vocab.get("synonyms", {}).items():
                if term in element_text:
                    synonyms.extend(term_synonyms)
                    element.synonyms = term_synonyms
            
            # Find related terms
            related = []
            for term, term_related in domain_vocab.get("related_terms", {}).items():
                if term in element_text:
                    related.extend(term_related)
                    element.related_terms = term_related
            
            if synonyms or related:
                expanded_terms[element_text] = {
                    "synonyms": synonyms,
                    "related": related
                }
        
        processed_query.expanded_terms = expanded_terms
    
    async def _prepare_for_search(self, processed_query: ProcessedQuery):
        """Prepare query for search execution."""
        
        # Extract primary search terms
        search_terms = []
        
        # Add terms from query elements
        for element in processed_query.query_elements:
            if element.element_type in [QueryElement.SUBJECT, QueryElement.OBJECT]:
                search_terms.append(element.normalized_text)
                # Add synonyms
                search_terms.extend(element.synonyms[:2])  # Limit to top 2 synonyms
        
        # Add terms from entities
        entities = processed_query.semantic_representation.get("entities", {})
        for entity_type, entity_list in entities.items():
            if entity_type in ["regulatory_references", "regulatory_concepts"]:
                search_terms.extend(entity_list[:3])  # Limit to top 3
        
        processed_query.search_terms = list(set(search_terms))  # Remove duplicates
        
        # Prepare search filters
        filters = {}
        
        # Domain filter
        filters["domain"] = processed_query.classified_query.query_domain.value
        
        # Firm type filter
        if entities.get("firm_types"):
            filters["firm_types"] = entities["firm_types"]
        
        # Regulatory source filter
        if entities.get("regulatory_references"):
            filters["regulatory_sources"] = entities["regulatory_references"]
        
        processed_query.search_filters = filters
        
        # Determine search scope
        scope = []
        query_type = processed_query.classified_query.query_type
        
        if query_type in ["requirement_lookup", "guidance_request"]:
            scope.extend(["requirements", "guidance"])
        elif query_type == "gap_analysis":
            scope.extend(["requirements", "controls", "policies"])
        elif query_type == "interpretation":
            scope.extend(["guidance", "interpretations", "examples"])
        
        processed_query.search_scope = scope
    
    async def _validate_query(self, processed_query: ProcessedQuery):
        """Validate the processed query."""
        
        issues = []
        
        # Check if we have sufficient information
        if not processed_query.query_elements:
            issues.append("No query elements extracted")
        
        if not processed_query.search_terms:
            issues.append("No search terms identified")
        
        # Check for ambiguity
        if len(processed_query.query_elements) > 10:
            issues.append("Query may be too complex or ambiguous")
        
        # Check for completeness
        has_subject = any(e.element_type == QueryElement.SUBJECT for e in processed_query.query_elements)
        has_predicate = any(e.element_type == QueryElement.PREDICATE for e in processed_query.query_elements)
        
        if not has_subject:
            issues.append("Query subject unclear")
        
        if not has_predicate:
            issues.append("Query action/intent unclear")
        
        # Calculate completeness score
        completeness_factors = [
            has_subject,
            has_predicate,
            bool(processed_query.search_terms),
            bool(processed_query.search_filters),
            len(processed_query.query_elements) >= 2,
            processed_query.classified_query.classification_confidence >= 0.7
        ]
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        processed_query.validation_issues = issues
        processed_query.is_valid = len(issues) <= 1  # Allow one minor issue
        processed_query.completeness_score = completeness_score
    
    def _calculate_processing_confidence(self, processed_query: ProcessedQuery) -> float:
        """Calculate overall processing confidence."""
        
        # Base confidence from classification
        base_confidence = processed_query.classified_query.classification_confidence
        
        # Factor in completeness
        completeness_factor = processed_query.completeness_score * 0.3
        
        # Factor in element extraction success
        element_factor = min(len(processed_query.query_elements) / 4, 1.0) * 0.2
        
        # Factor in search term extraction
        search_factor = min(len(processed_query.search_terms) / 5, 1.0) * 0.2
        
        # Factor in validation success
        validation_factor = (1.0 if processed_query.is_valid else 0.5) * 0.3
        
        overall_confidence = (
            base_confidence * 0.4 +
            completeness_factor +
            element_factor +
            search_factor +
            validation_factor
        )
        
        return min(overall_confidence, 1.0)
    
    # Helper methods for element extraction
    def _extract_subject(self, query_text: str, query_type: QueryType) -> Optional[str]:
        """Extract the subject of the query."""
        
        # Simple patterns for subject extraction
        subject_patterns = [
            r"requirements?\s+for\s+(.+?)(?:\s+(?:in|under|from)|\?|$)",
            r"what\s+(?:are\s+)?(.+?)\s+requirements?",
            r"(.+?)\s+compliance",
            r"(.+?)\s+(?:rules?|regulations?)"
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_predicate(self, query_text: str, query_type: QueryType) -> Optional[str]:
        """Extract the predicate/action of the query."""
        
        # Map query types to likely predicates
        type_predicates = {
            "requirement_lookup": "find requirements",
            "gap_analysis": "identify gaps",
            "compliance_check": "check compliance",
            "guidance_request": "provide guidance",
            "interpretation": "explain meaning",
            "risk_assessment": "assess risk"
        }
        
        # Check for explicit action words
        action_patterns = [
            r"(identify|find|search|locate|discover)",
            r"(explain|describe|clarify|interpret)",
            r"(check|verify|confirm|validate)",
            r"(assess|evaluate|analyze|review)",
            r"(recommend|suggest|advise)"
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fall back to type-based predicate
        return type_predicates.get(query_type.value, "process")
    
    def _extract_object(self, query_text: str, query_type: QueryType) -> Optional[str]:
        """Extract the object/target of the query."""
        
        # Look for common objects after prepositions
        object_patterns = [
            r"(?:for|of|about|regarding)\s+(.+?)(?:\s+(?:in|under|from)|\?|$)",
            r"compliance\s+with\s+(.+?)(?:\s+(?:in|under|from)|\?|$)"
        ]
        
        for pattern in object_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_qualifiers(self, query_text: str) -> List[str]:
        """Extract qualifiers and constraints."""
        
        qualifiers = []
        
        # Temporal qualifiers
        temporal_patterns = [
            r"(?:for|in|during)\s+(\d+\s+(?:days?|weeks?|months?|years?))",
            r"(immediately|urgently|soon|quickly)",
            r"(annually|quarterly|monthly|weekly)"
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            qualifiers.extend(matches)
        
        # Scope qualifiers
        scope_patterns = [
            r"(?:for|applicable to)\s+(all\s+firms?|investment\s+firms?|banks?)",
            r"(?:in|under)\s+([A-Z]{3,5}(?:\.\d+)*)"
        ]
        
        for pattern in scope_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            qualifiers.extend(matches)
        
        return qualifiers
    
    def _extract_regulatory_concepts(self, query_text: str) -> List[str]:
        """Extract regulatory concepts from query."""
        
        concepts = []
        
        # Common regulatory concepts
        regulatory_patterns = [
            r"(capital adequacy|liquidity|solvency)",
            r"(consumer duty|fair treatment|customer outcome)",
            r"(operational resilience|business continuity)",
            r"(governance|risk management|compliance)",
            r"(market conduct|best execution)",
            r"(financial crime|money laundering|sanctions)"
        ]
        
        for pattern in regulatory_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            concepts.extend(matches)
        
        return concepts
    
    def _extract_temporal_references(self, query_text: str) -> List[str]:
        """Extract temporal references."""
        
        temporal_refs = []
        
        patterns = [
            r"\d{4}",  # Years
            r"(?:january|february|march|april|may|june|july|august|september|october|november|december)",
            r"(?:q1|q2|q3|q4|quarter)",
            r"(?:annual|yearly|quarterly|monthly|weekly|daily)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            temporal_refs.extend(matches)
        
        return temporal_refs
    
    def _extract_action_references(self, query_text: str) -> List[str]:
        """Extract action/process references."""
        
        actions = []
        
        action_patterns = [
            r"(implement|establish|maintain|monitor|review)",
            r"(assess|evaluate|measure|test|validate)",
            r"(report|disclose|notify|submit|publish)",
            r"(train|educate|communicate|inform)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            actions.extend(matches)
        
        return actions
    
    # Helper methods for semantic analysis
    def _determine_primary_focus(self, processed_query: ProcessedQuery) -> str:
        """Determine the primary focus of the query."""
        
        # Analyze query elements to determine focus
        subjects = [e for e in processed_query.query_elements if e.element_type == QueryElement.SUBJECT]
        
        if subjects:
            return subjects[0].normalized_text
        
        # Fall back to domain
        return processed_query.classified_query.query_domain.value
    
    def _identify_relationships(self, processed_query: ProcessedQuery) -> List[str]:
        """Identify relationships in the query."""
        
        relationships = []
        
        query_text = processed_query.normalized_query
        
        # Common relationship indicators
        if "applies to" in query_text or "applicable to" in query_text:
            relationships.append("applicability")
        
        if "requires" in query_text or "requirement" in query_text:
            relationships.append("requirement")
        
        if "related to" in query_text or "concerning" in query_text:
            relationships.append("relation")
        
        return relationships
    
    def _determine_scope(self, processed_query: ProcessedQuery) -> Dict[str, Any]:
        """Determine the scope of the query."""
        
        scope = {
            "domain": processed_query.classified_query.query_domain.value,
            "complexity": processed_query.classified_query.query_complexity.value,
            "firm_types": [],
            "regulatory_sources": []
        }
        
        entities = processed_query.semantic_representation.get("entities", {})
        
        if entities.get("firm_types"):
            scope["firm_types"] = entities["firm_types"]
        
        if entities.get("regulatory_references"):
            scope["regulatory_sources"] = entities["regulatory_references"]
        
        return scope
    
    def _identify_constraints(self, processed_query: ProcessedQuery) -> List[Dict[str, Any]]:
        """Identify constraints in the query."""
        
        constraints = []
        
        # Temporal constraints
        temporal_refs = processed_query.semantic_representation.get("entities", {}).get("temporal_references", [])
        for ref in temporal_refs:
            constraints.append({
                "type": "temporal",
                "value": ref,
                "description": f"Time constraint: {ref}"
            })
        
        # Scope constraints from qualifiers
        qualifiers = [e for e in processed_query.query_elements if e.element_type == QueryElement.QUALIFIER]
        for qualifier in qualifiers:
            constraints.append({
                "type": "scope",
                "value": qualifier.normalized_text,
                "description": f"Scope constraint: {qualifier.normalized_text}"
            })
        
        return constraints
    
    # Helper methods for context enrichment
    def _get_domain_context(self, domain: str) -> Dict[str, Any]:
        """Get contextual information for a domain."""
        
        domain_contexts = {
            "governance": {
                "key_concepts": ["oversight", "accountability", "structure", "delegation"],
                "typical_requirements": ["board composition", "management arrangements", "committees"],
                "common_issues": ["lack of clarity", "insufficient oversight", "poor delegation"]
            },
            "risk_management": {
                "key_concepts": ["identification", "assessment", "mitigation", "monitoring"],
                "typical_requirements": ["risk appetite", "risk controls", "stress testing"],
                "common_issues": ["inadequate controls", "poor risk culture", "insufficient monitoring"]
            },
            "conduct": {
                "key_concepts": ["customer outcomes", "fair treatment", "suitability"],
                "typical_requirements": ["customer journey", "product governance", "advice suitability"],
                "common_issues": ["poor outcomes", "unsuitable advice", "inadequate disclosure"]
            }
        }
        
        return domain_contexts.get(domain, {})
    
    def _get_regulatory_context(self, references: List[str]) -> Dict[str, Any]:
        """Get contextual information for regulatory references."""
        
        context = {
            "sources": [],
            "themes": [],
            "related_areas": []
        }
        
        for ref in references:
            if ref.startswith("SYSC"):
                context["sources"].append("Systems and Controls")
                context["themes"].append("governance")
            elif ref.startswith("COBS"):
                context["sources"].append("Conduct of Business")
                context["themes"].append("conduct")
            elif ref.startswith("PRIN"):
                context["sources"].append("Principles for Businesses")
                context["themes"].append("high_level_principles")
        
        return context
    
    def _get_temporal_context(self, temporal_refs: List[str]) -> Dict[str, Any]:
        """Get contextual information for temporal references."""
        
        context = {
            "time_horizon": "unknown",
            "urgency": "medium",
            "periodicity": None
        }
        
        for ref in temporal_refs:
            ref_lower = ref.lower()
            
            if any(word in ref_lower for word in ["immediate", "urgent", "asap"]):
                context["urgency"] = "high"
            elif any(word in ref_lower for word in ["annual", "quarterly", "monthly"]):
                context["periodicity"] = ref_lower
            elif any(word in ref_lower for word in ["day", "week"]):
                context["time_horizon"] = "short_term"
            elif any(word in ref_lower for word in ["month", "quarter"]):
                context["time_horizon"] = "medium_term"
            elif any(word in ref_lower for word in ["year", "annual"]):
                context["time_horizon"] = "long_term"
        
        return context


# Factory functions
def create_comprehensive_processor(config: Optional[Dict[str, Any]] = None) -> ComprehensiveQueryProcessor:
    """Create comprehensive query processor."""
    if config is None:
        config = {}
    
    return ComprehensiveQueryProcessor(config)