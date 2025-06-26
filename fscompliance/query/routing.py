"""Query routing and classification for compliance queries."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from ..models import ComplianceQuery, RequirementType

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of compliance queries."""
    REQUIREMENT_LOOKUP = "requirement_lookup"      # Find specific requirements
    GAP_ANALYSIS = "gap_analysis"                 # Identify compliance gaps
    COMPLIANCE_CHECK = "compliance_check"         # Check compliance status
    GUIDANCE_REQUEST = "guidance_request"         # Request regulatory guidance
    INTERPRETATION = "interpretation"             # Interpret regulations
    COMPARISON = "comparison"                     # Compare requirements
    REPORTING = "reporting"                       # Generate reports
    RISK_ASSESSMENT = "risk_assessment"          # Assess compliance risks
    REMEDIATION = "remediation"                  # Get remediation advice
    MONITORING = "monitoring"                    # Set up monitoring
    TRAINING = "training"                        # Training requirements
    GENERAL_INQUIRY = "general_inquiry"          # General questions


class QueryIntent(str, Enum):
    """Intent classification for queries."""
    FIND = "find"                    # Find information
    EXPLAIN = "explain"              # Explain concepts
    COMPARE = "compare"              # Compare items
    ASSESS = "assess"                # Assess situation
    RECOMMEND = "recommend"          # Get recommendations
    CHECK = "check"                  # Check compliance
    ANALYZE = "analyze"              # Analyze data
    GENERATE = "generate"            # Generate outputs
    MONITOR = "monitor"              # Set up monitoring
    VALIDATE = "validate"            # Validate approach


class QueryDomain(str, Enum):
    """Domain areas for queries."""
    GOVERNANCE = "governance"                 # Governance and oversight
    RISK_MANAGEMENT = "risk_management"      # Risk management
    CONDUCT = "conduct"                      # Conduct of business
    PRUDENTIAL = "prudential"               # Prudential regulation
    OPERATIONAL = "operational"             # Operational requirements
    REPORTING = "reporting"                 # Regulatory reporting
    MARKET_CONDUCT = "market_conduct"       # Market conduct
    CONSUMER_PROTECTION = "consumer_protection"  # Consumer protection
    FINANCIAL_CRIME = "financial_crime"     # Financial crime
    DATA_GOVERNANCE = "data_governance"     # Data and technology
    CROSS_CUTTING = "cross_cutting"         # Multiple domains


class QueryComplexity(str, Enum):
    """Complexity levels for queries."""
    SIMPLE = "simple"           # Single fact lookup
    MODERATE = "moderate"       # Multi-step analysis
    COMPLEX = "complex"         # Cross-domain analysis
    EXPERT = "expert"          # Requires expert interpretation


class QueryUrgency(str, Enum):
    """Urgency levels for queries."""
    LOW = "low"               # General information
    MEDIUM = "medium"         # Business planning
    HIGH = "high"            # Active compliance issue
    CRITICAL = "critical"     # Regulatory deadline/breach


@dataclass
class QueryContext:
    """Context information for query processing."""
    
    user_role: Optional[str] = None
    firm_type: Optional[str] = None
    business_function: Optional[str] = None
    regulatory_scope: Optional[List[str]] = None
    time_horizon: Optional[str] = None
    risk_appetite: Optional[str] = None
    
    def __post_init__(self):
        if self.regulatory_scope is None:
            self.regulatory_scope = []


class ClassifiedQuery(BaseModel):
    """A query with classification and routing information."""
    
    # Original query
    query_id: str = Field(..., description="Unique query identifier")
    original_query: str = Field(..., description="Original query text")
    
    # Classification results
    query_type: QueryType = Field(..., description="Type of query")
    query_intent: QueryIntent = Field(..., description="Intent of query")
    query_domain: QueryDomain = Field(..., description="Domain area")
    query_complexity: QueryComplexity = Field(..., description="Complexity level")
    query_urgency: QueryUrgency = Field(..., description="Urgency level")
    
    # Extracted entities
    entities: Dict[str, List[str]] = Field(
        default_factory=dict, description="Extracted entities by type"
    )
    
    # Processing metadata
    classification_algorithm: str = Field(..., description="Algorithm used for classification")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    classification_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Routing information
    recommended_handlers: List[str] = Field(
        default_factory=list, description="Recommended processing handlers"
    )
    processing_priority: int = Field(..., ge=1, le=10, description="Processing priority (1=highest)")
    estimated_complexity_score: float = Field(..., ge=0.0, le=1.0, description="Estimated processing complexity")
    
    # Context
    query_context: Optional[QueryContext] = Field(None, description="Query context information")
    
    # Parsed query structure
    parsed_components: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed query components"
    )
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get summary for routing decisions."""
        return {
            "query_type": self.query_type.value,
            "intent": self.query_intent.value,
            "domain": self.query_domain.value,
            "complexity": self.query_complexity.value,
            "urgency": self.query_urgency.value,
            "priority": self.processing_priority,
            "handlers": self.recommended_handlers,
            "confidence": self.classification_confidence
        }


class BaseQueryClassifier(ABC):
    """Abstract base class for query classification algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classification_stats = {
            "queries_classified": 0,
            "high_confidence_classifications": 0,
            "processing_errors": 0,
            "classification_time_ms": 0
        }
    
    @abstractmethod
    async def classify_query(
        self, 
        query: str,
        context: Optional[QueryContext] = None
    ) -> ClassifiedQuery:
        """Classify a query and determine routing."""
        pass
    
    @abstractmethod
    def get_classifier_name(self) -> str:
        """Get the name of the classification algorithm."""
        pass
    
    def _generate_query_id(self) -> str:
        """Generate unique query identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return f"QUERY_{timestamp}"


class RuleBasedQueryClassifier(BaseQueryClassifier):
    """Rule-based query classification using patterns and keywords."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.classification_patterns = self._initialize_patterns()
        self.entity_extractors = self._initialize_entity_extractors()
    
    def get_classifier_name(self) -> str:
        return "rule_based_classifier"
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize classification patterns."""
        
        return {
            "query_type": {
                QueryType.REQUIREMENT_LOOKUP: {
                    "patterns": [
                        r"what are the requirements for",
                        r"what does .+ require",
                        r"requirements relating to",
                        r"regulatory requirements for",
                        r"what are the rules"
                    ],
                    "keywords": ["requirement", "rule", "regulation", "must", "shall"]
                },
                QueryType.GAP_ANALYSIS: {
                    "patterns": [
                        r"identify gaps",
                        r"compliance gaps",
                        r"what are we missing",
                        r"gap analysis",
                        r"areas of non-compliance"
                    ],
                    "keywords": ["gap", "missing", "non-compliance", "deficiency"]
                },
                QueryType.COMPLIANCE_CHECK: {
                    "patterns": [
                        r"are we compliant",
                        r"compliance status",
                        r"do we meet",
                        r"compliance with",
                        r"check compliance"
                    ],
                    "keywords": ["compliant", "status", "meet", "satisfy"]
                },
                QueryType.GUIDANCE_REQUEST: {
                    "patterns": [
                        r"guidance on",
                        r"how to comply",
                        r"best practice",
                        r"recommended approach",
                        r"what should we do"
                    ],
                    "keywords": ["guidance", "practice", "approach", "recommend", "should"]
                },
                QueryType.INTERPRETATION: {
                    "patterns": [
                        r"what does .+ mean",
                        r"interpretation of",
                        r"explain .+",
                        r"clarify .+",
                        r"understanding of"
                    ],
                    "keywords": ["mean", "interpret", "explain", "clarify", "understand"]
                },
                QueryType.RISK_ASSESSMENT: {
                    "patterns": [
                        r"risk assessment",
                        r"compliance risk",
                        r"what are the risks",
                        r"risk of non-compliance",
                        r"regulatory risk"
                    ],
                    "keywords": ["risk", "exposure", "threat", "likelihood"]
                },
                QueryType.REMEDIATION: {
                    "patterns": [
                        r"how to fix",
                        r"remediation plan",
                        r"corrective action",
                        r"how to address",
                        r"resolve the issue"
                    ],
                    "keywords": ["fix", "remediate", "correct", "address", "resolve"]
                }
            },
            
            "query_intent": {
                QueryIntent.FIND: {
                    "patterns": [r"find", r"search", r"look for", r"identify", r"locate"],
                    "keywords": ["find", "search", "identify", "locate", "discover"]
                },
                QueryIntent.EXPLAIN: {
                    "patterns": [r"explain", r"what is", r"what does", r"describe", r"clarify"],
                    "keywords": ["explain", "describe", "clarify", "define", "mean"]
                },
                QueryIntent.COMPARE: {
                    "patterns": [r"compare", r"difference", r"versus", r"vs", r"contrast"],
                    "keywords": ["compare", "difference", "versus", "contrast", "similar"]
                },
                QueryIntent.ASSESS: {
                    "patterns": [r"assess", r"evaluate", r"analyze", r"review", r"examine"],
                    "keywords": ["assess", "evaluate", "analyze", "review", "examine"]
                },
                QueryIntent.RECOMMEND: {
                    "patterns": [r"recommend", r"suggest", r"advise", r"what should", r"best"],
                    "keywords": ["recommend", "suggest", "advise", "should", "best"]
                },
                QueryIntent.CHECK: {
                    "patterns": [r"check", r"verify", r"confirm", r"validate", r"ensure"],
                    "keywords": ["check", "verify", "confirm", "validate", "ensure"]
                }
            },
            
            "query_domain": {
                QueryDomain.GOVERNANCE: {
                    "keywords": ["governance", "board", "oversight", "management", "accountability", "structure"]
                },
                QueryDomain.RISK_MANAGEMENT: {
                    "keywords": ["risk", "control", "assessment", "mitigation", "monitoring", "appetite"]
                },
                QueryDomain.CONDUCT: {
                    "keywords": ["conduct", "customer", "client", "treating", "fair", "outcome", "suitability"]
                },
                QueryDomain.PRUDENTIAL: {
                    "keywords": ["capital", "liquidity", "solvency", "prudential", "financial resources"]
                },
                QueryDomain.REPORTING: {
                    "keywords": ["report", "submission", "return", "disclosure", "publication", "filing"]
                },
                QueryDomain.OPERATIONAL: {
                    "keywords": ["operational", "resilience", "continuity", "system", "process", "procedure"]
                },
                QueryDomain.CONSUMER_PROTECTION: {
                    "keywords": ["consumer", "protection", "duty", "vulnerable", "fair treatment"]
                },
                QueryDomain.FINANCIAL_CRIME: {
                    "keywords": ["money laundering", "financial crime", "sanctions", "terrorist", "aml"]
                }
            },
            
            "complexity_indicators": {
                QueryComplexity.SIMPLE: {
                    "patterns": [r"^what is", r"^define", r"^when is"],
                    "max_words": 10
                },
                QueryComplexity.MODERATE: {
                    "patterns": [r"how to", r"what are the steps", r"process for"],
                    "max_words": 20
                },
                QueryComplexity.COMPLEX: {
                    "patterns": [r"analyze", r"comprehensive", r"across", r"multiple"],
                    "keywords": ["comprehensive", "across", "multiple", "complex", "various"]
                },
                QueryComplexity.EXPERT: {
                    "keywords": ["interpretation", "legal", "regulatory position", "expert view"]
                }
            },
            
            "urgency_indicators": {
                QueryUrgency.CRITICAL: {
                    "keywords": ["urgent", "immediate", "critical", "breach", "deadline", "emergency"]
                },
                QueryUrgency.HIGH: {
                    "keywords": ["important", "priority", "soon", "quickly", "asap"]
                },
                QueryUrgency.MEDIUM: {
                    "keywords": ["planning", "upcoming", "prepare", "next month"]
                },
                QueryUrgency.LOW: {
                    "keywords": ["general", "information", "background", "understanding"]
                }
            }
        }
    
    def _initialize_entity_extractors(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns."""
        
        return {
            "regulatory_references": [
                r"[A-Z]{3,5}\.?\s*\d+(?:\.\d+)*",  # e.g., SYSC.4.1.1
                r"(?:section|rule|paragraph|chapter)\s+\d+",
                r"(?:SYSC|COBS|PRIN|DEPP|EG|SUP|COND)"
            ],
            "firm_types": [
                r"investment firm[s]?",
                r"bank[s]?",
                r"credit institution[s]?",
                r"insurer[s]?",
                r"payment institution[s]?",
                r"electronic money institution[s]?"
            ],
            "business_functions": [
                r"governance",
                r"risk management",
                r"compliance",
                r"audit",
                r"trading",
                r"client facing",
                r"operations"
            ],
            "time_expressions": [
                r"\d+\s+(?:days?|weeks?|months?|years?)",
                r"immediately",
                r"annually",
                r"quarterly",
                r"monthly"
            ],
            "amounts": [
                r"£[\d,]+(?:\.\d{2})?",
                r"\d+(?:\.\d+)?\s*%",
                r"\d+(?:\.\d+)?\s*(?:million|billion|thousand)"
            ]
        }
    
    async def classify_query(
        self, 
        query: str,
        context: Optional[QueryContext] = None
    ) -> ClassifiedQuery:
        """Classify a query using rule-based patterns."""
        
        start_time = datetime.utcnow()
        
        try:
            query_lower = query.lower()
            
            # Extract entities
            entities = self._extract_entities(query)
            
            # Classify query type
            query_type = self._classify_query_type(query_lower)
            
            # Classify intent
            query_intent = self._classify_intent(query_lower)
            
            # Classify domain
            query_domain = self._classify_domain(query_lower, entities)
            
            # Assess complexity
            query_complexity = self._assess_complexity(query_lower, query)
            
            # Assess urgency
            query_urgency = self._assess_urgency(query_lower, context)
            
            # Calculate classification confidence
            confidence = self._calculate_classification_confidence(
                query_type, query_intent, query_domain, query_complexity, entities
            )
            
            # Determine routing
            handlers, priority = self._determine_routing(
                query_type, query_complexity, query_urgency
            )
            
            # Parse query components
            parsed_components = self._parse_query_components(query, entities)
            
            # Create classified query
            classified_query = ClassifiedQuery(
                query_id=self._generate_query_id(),
                original_query=query,
                query_type=query_type,
                query_intent=query_intent,
                query_domain=query_domain,
                query_complexity=query_complexity,
                query_urgency=query_urgency,
                entities=entities,
                classification_algorithm=self.get_classifier_name(),
                classification_confidence=confidence,
                recommended_handlers=handlers,
                processing_priority=priority,
                estimated_complexity_score=self._estimate_processing_complexity(query_complexity),
                query_context=context,
                parsed_components=parsed_components
            )
            
            # Update statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            self.classification_stats["queries_classified"] += 1
            self.classification_stats["classification_time_ms"] += processing_time
            
            if confidence >= 0.8:
                self.classification_stats["high_confidence_classifications"] += 1
            
            logger.info(f"Query classified: {query_type.value} ({confidence:.2f} confidence)")
            
            return classified_query
            
        except Exception as e:
            self.classification_stats["processing_errors"] += 1
            logger.error(f"Error classifying query: {e}")
            raise
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query text."""
        
        entities = {}
        
        for entity_type, patterns in self.entity_extractors.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    entities[entity_type].extend(matches)
            
            # Remove duplicates
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Classify the type of query."""
        
        best_type = QueryType.GENERAL_INQUIRY
        best_score = 0
        
        for query_type, config in self.classification_patterns["query_type"].items():
            score = 0
            
            # Check patterns
            if "patterns" in config:
                for pattern in config["patterns"]:
                    if re.search(pattern, query_lower):
                        score += 2
            
            # Check keywords
            if "keywords" in config:
                for keyword in config["keywords"]:
                    if keyword in query_lower:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_type = query_type
        
        return best_type
    
    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify the intent of the query."""
        
        best_intent = QueryIntent.FIND
        best_score = 0
        
        for intent, config in self.classification_patterns["query_intent"].items():
            score = 0
            
            # Check patterns
            if "patterns" in config:
                for pattern in config["patterns"]:
                    if re.search(pattern, query_lower):
                        score += 2
            
            # Check keywords
            if "keywords" in config:
                for keyword in config["keywords"]:
                    if keyword in query_lower:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent
    
    def _classify_domain(self, query_lower: str, entities: Dict[str, List[str]]) -> QueryDomain:
        """Classify the domain area of the query."""
        
        best_domain = QueryDomain.CROSS_CUTTING
        best_score = 0
        
        for domain, config in self.classification_patterns["query_domain"].items():
            score = 0
            
            # Check keywords
            if "keywords" in config:
                for keyword in config["keywords"]:
                    if keyword in query_lower:
                        score += 1
            
            # Boost score for regulatory references in relevant domains
            if entities.get("regulatory_references"):
                for ref in entities["regulatory_references"]:
                    if domain == QueryDomain.GOVERNANCE and "SYSC" in ref:
                        score += 2
                    elif domain == QueryDomain.CONDUCT and "COBS" in ref:
                        score += 2
                    elif domain == QueryDomain.GOVERNANCE and "PRIN" in ref:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_domain = domain
        
        return best_domain
    
    def _assess_complexity(self, query_lower: str, original_query: str) -> QueryComplexity:
        """Assess the complexity of the query."""
        
        word_count = len(original_query.split())
        
        # Check for explicit complexity indicators
        for complexity, config in self.classification_patterns["complexity_indicators"].items():
            # Check patterns
            if "patterns" in config:
                for pattern in config["patterns"]:
                    if re.search(pattern, query_lower):
                        return complexity
            
            # Check keywords
            if "keywords" in config:
                for keyword in config["keywords"]:
                    if keyword in query_lower:
                        return complexity
            
            # Check word count thresholds
            if "max_words" in config and word_count <= config["max_words"]:
                return complexity
        
        # Default based on word count
        if word_count <= 8:
            return QueryComplexity.SIMPLE
        elif word_count <= 15:
            return QueryComplexity.MODERATE
        elif word_count <= 25:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT
    
    def _assess_urgency(self, query_lower: str, context: Optional[QueryContext]) -> QueryUrgency:
        """Assess the urgency of the query."""
        
        # Check for explicit urgency indicators
        for urgency, config in self.classification_patterns["urgency_indicators"].items():
            if "keywords" in config:
                for keyword in config["keywords"]:
                    if keyword in query_lower:
                        return urgency
        
        # Default urgency
        return QueryUrgency.MEDIUM
    
    def _calculate_classification_confidence(
        self,
        query_type: QueryType,
        query_intent: QueryIntent,
        query_domain: QueryDomain,
        query_complexity: QueryComplexity,
        entities: Dict[str, List[str]]
    ) -> float:
        """Calculate confidence in the classification."""
        
        base_confidence = 0.6
        
        # Boost confidence if specific entities found
        if entities.get("regulatory_references"):
            base_confidence += 0.2
        
        if entities.get("firm_types"):
            base_confidence += 0.1
        
        if entities.get("business_functions"):
            base_confidence += 0.1
        
        # Reduce confidence for general classifications
        if query_type == QueryType.GENERAL_INQUIRY:
            base_confidence -= 0.2
        
        if query_domain == QueryDomain.CROSS_CUTTING:
            base_confidence -= 0.1
        
        return min(base_confidence, 1.0)
    
    def _determine_routing(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        urgency: QueryUrgency
    ) -> Tuple[List[str], int]:
        """Determine routing handlers and priority."""
        
        handlers = []
        
        # Determine handlers based on query type
        type_handlers = {
            QueryType.REQUIREMENT_LOOKUP: ["knowledge_retrieval", "requirement_search"],
            QueryType.GAP_ANALYSIS: ["gap_detector", "compliance_analyzer"],
            QueryType.COMPLIANCE_CHECK: ["compliance_scorer", "status_checker"],
            QueryType.GUIDANCE_REQUEST: ["guidance_generator", "best_practice_finder"],
            QueryType.INTERPRETATION: ["interpretation_engine", "expert_system"],
            QueryType.RISK_ASSESSMENT: ["risk_analyzer", "threat_assessor"],
            QueryType.REMEDIATION: ["remediation_planner", "action_generator"],
            QueryType.COMPARISON: ["comparison_engine", "requirement_matcher"],
            QueryType.REPORTING: ["report_generator", "template_engine"]
        }
        
        handlers = type_handlers.get(query_type, ["general_processor"])
        
        # Add complexity-specific handlers
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            handlers.append("expert_review")
        
        # Determine priority based on urgency and complexity
        urgency_priority = {
            QueryUrgency.CRITICAL: 1,
            QueryUrgency.HIGH: 3,
            QueryUrgency.MEDIUM: 5,
            QueryUrgency.LOW: 7
        }
        
        complexity_adjustment = {
            QueryComplexity.SIMPLE: 0,
            QueryComplexity.MODERATE: 1,
            QueryComplexity.COMPLEX: 2,
            QueryComplexity.EXPERT: 3
        }
        
        priority = urgency_priority[urgency] + complexity_adjustment[complexity]
        priority = max(1, min(priority, 10))  # Clamp between 1-10
        
        return handlers, priority
    
    def _estimate_processing_complexity(self, complexity: QueryComplexity) -> float:
        """Estimate processing complexity score."""
        
        complexity_scores = {
            QueryComplexity.SIMPLE: 0.2,
            QueryComplexity.MODERATE: 0.5,
            QueryComplexity.COMPLEX: 0.8,
            QueryComplexity.EXPERT: 1.0
        }
        
        return complexity_scores[complexity]
    
    def _parse_query_components(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Parse query into structured components."""
        
        components = {
            "word_count": len(query.split()),
            "sentence_count": len([s for s in query.split('.') if s.strip()]),
            "has_question_mark": "?" in query,
            "entities_found": {k: len(v) for k, v in entities.items()},
            "regulatory_context": bool(entities.get("regulatory_references")),
            "firm_specific": bool(entities.get("firm_types")),
            "function_specific": bool(entities.get("business_functions")),
            "time_sensitive": bool(entities.get("time_expressions"))
        }
        
        return components


class QueryRoutingPipeline:
    """Pipeline for orchestrating query classification and routing."""
    
    def __init__(self, classifiers: List[BaseQueryClassifier]):
        self.classifiers = classifiers
        self.pipeline_stats = {
            "queries_processed": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "average_confidence": 0.0,
            "processing_sessions": 0
        }
    
    async def process_query(
        self, 
        query: str,
        context: Optional[QueryContext] = None
    ) -> ClassifiedQuery:
        """Process a query through the classification pipeline."""
        
        logger.info(f"Processing query through {len(self.classifiers)} classifiers")
        
        best_classification = None
        best_confidence = 0.0
        
        for classifier in self.classifiers:
            try:
                classification = await classifier.classify_query(query, context)
                
                if classification.classification_confidence > best_confidence:
                    best_confidence = classification.classification_confidence
                    best_classification = classification
                
                logger.debug(f"Classifier '{classifier.get_classifier_name()}' confidence: {classification.classification_confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error in classifier '{classifier.get_classifier_name()}': {e}")
                continue
        
        if best_classification is None:
            raise RuntimeError("No classifier successfully processed the query")
        
        # Update statistics
        self.pipeline_stats["queries_processed"] += 1
        self.pipeline_stats["processing_sessions"] += 1
        
        if best_classification:
            self.pipeline_stats["successful_classifications"] += 1
            
            # Update average confidence
            total_confidence = (self.pipeline_stats["average_confidence"] * 
                              (self.pipeline_stats["successful_classifications"] - 1) + 
                              best_classification.classification_confidence)
            self.pipeline_stats["average_confidence"] = total_confidence / self.pipeline_stats["successful_classifications"]
        else:
            self.pipeline_stats["failed_classifications"] += 1
        
        logger.info(f"Query classified as {best_classification.query_type.value} with {best_confidence:.2f} confidence")
        
        return best_classification
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        classifier_stats = {}
        for classifier in self.classifiers:
            classifier_stats[classifier.get_classifier_name()] = classifier.classification_stats
        
        return {
            **self.pipeline_stats,
            "classifier_statistics": classifier_stats
        }


# Factory functions
def create_rule_based_classifier(config: Optional[Dict[str, Any]] = None) -> RuleBasedQueryClassifier:
    """Create rule-based query classifier."""
    if config is None:
        config = {}
    
    return RuleBasedQueryClassifier(config)


def create_query_routing_pipeline(
    classifiers: Optional[List[str]] = None
) -> QueryRoutingPipeline:
    """Create query routing pipeline with specified classifiers."""
    
    if classifiers is None:
        classifiers = ["rule_based"]
    
    classifier_instances = []
    
    for classifier_name in classifiers:
        if classifier_name == "rule_based":
            classifier_instances.append(create_rule_based_classifier())
        else:
            logger.warning(f"Classifier '{classifier_name}' not implemented, skipping")
    
    return QueryRoutingPipeline(classifier_instances)