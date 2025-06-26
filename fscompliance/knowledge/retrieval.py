"""Dual-level retrieval system for FSCompliance knowledge base."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ..models import ComplianceQuery, ComplianceResponse, ComplianceStatus, FCASourcebook, RiskLevel
from .entity_extraction import ExtractedEntity, ExtractedRelationship
from .knowledge_base import BaseKnowledgeBase, SearchQuery, SearchResult
from .lightrag_integration import FSComplianceLightRAG

logger = logging.getLogger(__name__)


class RetrievalLevel(str, Enum):
    """Levels of retrieval in the dual-level system."""
    LOW_LEVEL = "low_level"      # Specific entities and facts
    HIGH_LEVEL = "high_level"    # Broader concepts and themes
    HYBRID = "hybrid"            # Combination of both levels


class RetrievalMode(str, Enum):
    """Modes of retrieval operation."""
    EXACT_MATCH = "exact_match"           # Exact keyword/phrase matching
    SEMANTIC_SEARCH = "semantic_search"   # Vector-based semantic similarity
    GRAPH_TRAVERSAL = "graph_traversal"   # Knowledge graph navigation
    CONTEXTUAL = "contextual"             # Context-aware retrieval
    HYBRID = "hybrid"                     # Multiple modes combined


class QueryComplexity(str, Enum):
    """Complexity levels of user queries."""
    SIMPLE = "simple"           # Single concept/entity
    MODERATE = "moderate"       # Multiple related concepts
    COMPLEX = "complex"         # Multi-faceted with relationships
    ANALYTICAL = "analytical"   # Requires inference and analysis


@dataclass
class RetrievalContext:
    """Context for retrieval operations."""
    
    user_role: str
    firm_type: Optional[str] = None
    business_functions: List[str] = None
    regulatory_scope: List[str] = None
    time_constraint: Optional[str] = None
    priority_level: str = "normal"
    
    def __post_init__(self):
        if self.business_functions is None:
            self.business_functions = []
        if self.regulatory_scope is None:
            self.regulatory_scope = []


class RetrievalResult(BaseModel):
    """Result from retrieval operation."""
    
    query_id: str = Field(..., description="Original query identifier")
    retrieval_level: RetrievalLevel = Field(..., description="Level of retrieval used")
    retrieval_mode: RetrievalMode = Field(..., description="Mode of retrieval used")
    
    # Low-level results (specific facts/entities)
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Retrieved entities")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="Retrieved relationships")
    specific_requirements: List[SearchResult] = Field(default_factory=list, description="Specific requirements")
    
    # High-level results (concepts/themes)
    conceptual_matches: List[SearchResult] = Field(default_factory=list, description="Conceptual matches")
    thematic_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Thematic patterns")
    contextual_information: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    
    # Metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    processing_time_seconds: float = Field(..., description="Processing time")
    retrieval_stats: Dict[str, int] = Field(default_factory=dict, description="Retrieval statistics")
    
    class Config:
        arbitrary_types_allowed = True


class BaseDualLevelRetriever(ABC):
    """Abstract base class for dual-level retrieval systems."""
    
    def __init__(self, knowledge_base: BaseKnowledgeBase, lightrag: Optional[FSComplianceLightRAG] = None):
        self.knowledge_base = knowledge_base
        self.lightrag = lightrag
        self.retrieval_stats = {
            "queries_processed": 0,
            "low_level_retrievals": 0,
            "high_level_retrievals": 0,
            "hybrid_retrievals": 0,
            "average_processing_time": 0.0
        }
    
    @abstractmethod
    async def retrieve_low_level(
        self, 
        query: ComplianceQuery, 
        context: RetrievalContext
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship], List[SearchResult]]:
        """Perform low-level retrieval for specific entities and facts."""
        pass
    
    @abstractmethod
    async def retrieve_high_level(
        self, 
        query: ComplianceQuery, 
        context: RetrievalContext
    ) -> Tuple[List[SearchResult], List[Dict[str, Any]], Dict[str, Any]]:
        """Perform high-level retrieval for concepts and themes."""
        pass
    
    async def retrieve(
        self, 
        query: ComplianceQuery, 
        retrieval_level: RetrievalLevel = RetrievalLevel.HYBRID,
        context: Optional[RetrievalContext] = None
    ) -> RetrievalResult:
        """Perform dual-level retrieval based on query and context."""
        
        start_time = datetime.utcnow()
        
        if context is None:
            context = RetrievalContext(user_role=query.user_role.value)
        
        # Determine query complexity
        complexity = self._assess_query_complexity(query)
        
        # Adjust retrieval strategy based on complexity
        if complexity == QueryComplexity.SIMPLE and retrieval_level == RetrievalLevel.HYBRID:
            retrieval_level = RetrievalLevel.LOW_LEVEL
        elif complexity == QueryComplexity.ANALYTICAL and retrieval_level == RetrievalLevel.HYBRID:
            retrieval_level = RetrievalLevel.HIGH_LEVEL
        
        # Initialize result structure
        result = RetrievalResult(
            query_id=query.id,
            retrieval_level=retrieval_level,
            retrieval_mode=RetrievalMode.HYBRID,  # Will be updated based on actual retrieval
            confidence_score=0.0,
            processing_time_seconds=0.0
        )
        
        try:
            if retrieval_level == RetrievalLevel.LOW_LEVEL:
                # Low-level retrieval only
                entities, relationships, requirements = await self.retrieve_low_level(query, context)
                result.entities = entities
                result.relationships = relationships
                result.specific_requirements = requirements
                result.retrieval_mode = RetrievalMode.EXACT_MATCH
                
                self.retrieval_stats["low_level_retrievals"] += 1
                
            elif retrieval_level == RetrievalLevel.HIGH_LEVEL:
                # High-level retrieval only
                concepts, patterns, contextual_info = await self.retrieve_high_level(query, context)
                result.conceptual_matches = concepts
                result.thematic_patterns = patterns
                result.contextual_information = contextual_info
                result.retrieval_mode = RetrievalMode.SEMANTIC_SEARCH
                
                self.retrieval_stats["high_level_retrievals"] += 1
                
            else:  # HYBRID
                # Both levels of retrieval
                entities, relationships, requirements = await self.retrieve_low_level(query, context)
                concepts, patterns, contextual_info = await self.retrieve_high_level(query, context)
                
                result.entities = entities
                result.relationships = relationships
                result.specific_requirements = requirements
                result.conceptual_matches = concepts
                result.thematic_patterns = patterns
                result.contextual_information = contextual_info
                result.retrieval_mode = RetrievalMode.HYBRID
                
                self.retrieval_stats["hybrid_retrievals"] += 1
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence_score(result, query, complexity)
            
            # Calculate processing time
            end_time = datetime.utcnow()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
            
            # Update statistics
            result.retrieval_stats = {
                "entities_found": len(result.entities),
                "relationships_found": len(result.relationships),
                "specific_requirements_found": len(result.specific_requirements),
                "conceptual_matches_found": len(result.conceptual_matches),
                "thematic_patterns_found": len(result.thematic_patterns)
            }
            
            self.retrieval_stats["queries_processed"] += 1
            self._update_average_processing_time(result.processing_time_seconds)
            
            logger.info(f"Retrieval completed: {retrieval_level.value} level, {result.confidence_score:.2f} confidence")
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval failed for query {query.id}: {e}")
            
            # Return empty result with error information
            end_time = datetime.utcnow()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
            result.contextual_information = {"error": str(e)}
            
            return result
    
    def _assess_query_complexity(self, query: ComplianceQuery) -> QueryComplexity:
        """Assess the complexity of the user query."""
        
        content = query.content.lower()
        word_count = len(content.split())
        
        # Simple heuristics for complexity assessment
        complex_indicators = [
            "compare", "contrast", "analyze", "evaluate", "assess", "relationship",
            "impact", "implications", "consequences", "differences", "similarities"
        ]
        
        moderate_indicators = [
            "and", "or", "but", "however", "although", "because", "since", "when", "where"
        ]
        
        # Count indicators
        complex_count = sum(1 for indicator in complex_indicators if indicator in content)
        moderate_count = sum(1 for indicator in moderate_indicators if indicator in content)
        
        # Determine complexity
        if complex_count >= 2 or "how" in content and word_count > 15:
            return QueryComplexity.ANALYTICAL
        elif complex_count >= 1 or moderate_count >= 2 or word_count > 10:
            return QueryComplexity.COMPLEX
        elif moderate_count >= 1 or word_count > 5:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _calculate_confidence_score(
        self, 
        result: RetrievalResult, 
        query: ComplianceQuery, 
        complexity: QueryComplexity
    ) -> float:
        """Calculate overall confidence score for retrieval result."""
        
        base_confidence = 0.5
        
        # Boost based on number of results found
        total_results = (
            len(result.entities) + 
            len(result.relationships) + 
            len(result.specific_requirements) + 
            len(result.conceptual_matches)
        )
        
        if total_results > 0:
            base_confidence += 0.2
        if total_results > 5:
            base_confidence += 0.1
        if total_results > 10:
            base_confidence += 0.1
        
        # Adjust based on query complexity match
        if complexity == QueryComplexity.SIMPLE and len(result.specific_requirements) > 0:
            base_confidence += 0.1
        elif complexity == QueryComplexity.ANALYTICAL and len(result.thematic_patterns) > 0:
            base_confidence += 0.1
        
        # Boost if both low-level and high-level results are present
        if (len(result.entities) > 0 or len(result.specific_requirements) > 0) and \
           (len(result.conceptual_matches) > 0 or len(result.thematic_patterns) > 0):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _update_average_processing_time(self, new_time: float):
        """Update average processing time statistics."""
        queries_count = self.retrieval_stats["queries_processed"]
        current_avg = self.retrieval_stats["average_processing_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (queries_count - 1)) + new_time) / queries_count
        self.retrieval_stats["average_processing_time"] = new_avg


class StandardDualLevelRetriever(BaseDualLevelRetriever):
    """Standard implementation of dual-level retrieval system."""
    
    async def retrieve_low_level(
        self, 
        query: ComplianceQuery, 
        context: RetrievalContext
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship], List[SearchResult]]:
        """Perform low-level retrieval for specific entities and facts."""
        
        entities = []
        relationships = []
        requirements = []
        
        try:
            # 1. Exact keyword matching in knowledge base
            search_query = SearchQuery(
                query_text=query.content,
                query_type="keyword_search",
                filters={
                    "entity_type": "requirement"
                },
                limit=20
            )
            
            kb_results = await self.knowledge_base.search(search_query)
            requirements.extend(kb_results)
            
            # 2. Entity-specific searches
            # Extract potential entities from query text
            entity_keywords = self._extract_entity_keywords(query.content)
            
            for keyword in entity_keywords:
                entity_search = SearchQuery(
                    query_text=keyword,
                    query_type="exact_match",
                    filters={"entity_type": "requirement"},
                    limit=10
                )
                
                entity_results = await self.knowledge_base.search(entity_search)
                requirements.extend(entity_results)
            
            # 3. Section-specific lookups if section references found
            section_refs = self._extract_section_references(query.content)
            
            for section_ref in section_refs:
                section_search = SearchQuery(
                    query_text=section_ref,
                    query_type="exact_match",
                    filters={"sourcebook": context.regulatory_scope[0] if context.regulatory_scope else None},
                    limit=5
                )
                
                section_results = await self.knowledge_base.search(section_search)
                requirements.extend(section_results)
            
            # Remove duplicates
            requirements = self._deduplicate_search_results(requirements)
            
            logger.debug(f"Low-level retrieval found {len(requirements)} specific requirements")
            
        except Exception as e:
            logger.error(f"Low-level retrieval failed: {e}")
        
        return entities, relationships, requirements
    
    async def retrieve_high_level(
        self, 
        query: ComplianceQuery, 
        context: RetrievalContext
    ) -> Tuple[List[SearchResult], List[Dict[str, Any]], Dict[str, Any]]:
        """Perform high-level retrieval for concepts and themes."""
        
        conceptual_matches = []
        thematic_patterns = []
        contextual_info = {}
        
        try:
            # 1. Semantic search using LightRAG if available
            if self.lightrag:
                try:
                    lightrag_result = await self.lightrag.query_knowledge_graph(
                        query.content,
                        mode="hybrid",
                        only_need_context=False
                    )
                    
                    # Parse LightRAG result into structured format
                    contextual_info["lightrag_response"] = lightrag_result
                    
                except Exception as e:
                    logger.warning(f"LightRAG query failed: {e}")
            
            # 2. Conceptual matching in knowledge base
            concept_keywords = self._extract_conceptual_keywords(query.content)
            
            for concept in concept_keywords:
                concept_search = SearchQuery(
                    query_text=concept,
                    query_type="semantic_search",
                    filters={},
                    limit=10
                )
                
                concept_results = await self.knowledge_base.search(concept_search)
                conceptual_matches.extend(concept_results)
            
            # 3. Thematic pattern analysis
            themes = self._identify_themes(query.content, context)
            
            for theme in themes:
                pattern = {
                    "theme": theme,
                    "relevance_score": 0.8,  # Placeholder
                    "related_concepts": self._get_related_concepts(theme),
                    "regulatory_implications": self._get_regulatory_implications(theme, context)
                }
                thematic_patterns.append(pattern)
            
            # 4. Contextual information enrichment
            contextual_info.update({
                "user_role_context": self._get_role_context(context.user_role),
                "regulatory_context": self._get_regulatory_context(context.regulatory_scope),
                "business_context": self._get_business_context(context.business_functions)
            })
            
            # Remove duplicates from conceptual matches
            conceptual_matches = self._deduplicate_search_results(conceptual_matches)
            
            logger.debug(f"High-level retrieval found {len(conceptual_matches)} conceptual matches, {len(thematic_patterns)} patterns")
            
        except Exception as e:
            logger.error(f"High-level retrieval failed: {e}")
        
        return conceptual_matches, thematic_patterns, contextual_info
    
    def _extract_entity_keywords(self, text: str) -> List[str]:
        """Extract potential entity keywords from query text."""
        
        # Common regulatory entity keywords
        entity_keywords = [
            "firm", "bank", "insurer", "investment", "payment", "credit",
            "client", "customer", "consumer", "retail", "professional",
            "governance", "risk", "compliance", "audit", "control",
            "report", "notification", "disclosure", "record", "documentation"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in entity_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_section_references(self, text: str) -> List[str]:
        """Extract section references from query text."""
        
        import re
        
        # Pattern for FCA section references (e.g., SYSC.4.1.1, COBS 9.2)
        section_pattern = r'[A-Z]{3,5}[\.\s]*\d+(?:[\.\s]*\d+)*(?:[\.\s]*\d+)*'
        
        matches = re.findall(section_pattern, text.upper())
        
        # Normalize section references
        normalized = []
        for match in matches:
            normalized_ref = re.sub(r'\s+', '.', match.strip())
            normalized.append(normalized_ref)
        
        return normalized
    
    def _extract_conceptual_keywords(self, text: str) -> List[str]:
        """Extract conceptual keywords for high-level search."""
        
        conceptual_keywords = [
            "conduct", "governance", "risk management", "compliance",
            "customer protection", "market integrity", "financial crime",
            "prudential", "operational resilience", "data protection",
            "anti-money laundering", "know your customer", "best execution",
            "suitability", "appropriateness", "treating customers fairly"
        ]
        
        found_concepts = []
        text_lower = text.lower()
        
        for concept in conceptual_keywords:
            if concept in text_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _identify_themes(self, text: str, context: RetrievalContext) -> List[str]:
        """Identify thematic patterns in query."""
        
        themes = []
        text_lower = text.lower()
        
        # Regulatory themes
        if any(word in text_lower for word in ["customer", "client", "consumer"]):
            themes.append("customer_protection")
        
        if any(word in text_lower for word in ["risk", "control", "management"]):
            themes.append("risk_management")
        
        if any(word in text_lower for word in ["report", "notification", "disclosure"]):
            themes.append("regulatory_reporting")
        
        if any(word in text_lower for word in ["governance", "oversight", "responsibility"]):
            themes.append("governance_oversight")
        
        if any(word in text_lower for word in ["record", "documentation", "maintain"]):
            themes.append("record_keeping")
        
        return themes
    
    def _get_related_concepts(self, theme: str) -> List[str]:
        """Get concepts related to a theme."""
        
        related_concepts = {
            "customer_protection": ["treating customers fairly", "suitability", "appropriateness", "disclosure"],
            "risk_management": ["operational risk", "credit risk", "market risk", "liquidity risk"],
            "regulatory_reporting": ["returns", "notifications", "disclosures", "prudential reports"],
            "governance_oversight": ["board responsibilities", "senior management", "delegation", "accountability"],
            "record_keeping": ["audit trail", "retention periods", "documentation standards", "data integrity"]
        }
        
        return related_concepts.get(theme, [])
    
    def _get_regulatory_implications(self, theme: str, context: RetrievalContext) -> List[str]:
        """Get regulatory implications of a theme."""
        
        implications = {
            "customer_protection": ["Enhanced due diligence", "Regular suitability reviews", "Clear disclosures"],
            "risk_management": ["Risk appetite framework", "Control testing", "Stress testing"],
            "regulatory_reporting": ["Timely submissions", "Data accuracy", "Regulatory liaison"],
            "governance_oversight": ["Clear responsibilities", "Regular reviews", "Escalation procedures"],
            "record_keeping": ["Secure storage", "Regular backups", "Access controls"]
        }
        
        return implications.get(theme, [])
    
    def _get_role_context(self, user_role: str) -> Dict[str, Any]:
        """Get context specific to user role."""
        
        role_contexts = {
            "compliance_officer": {
                "focus_areas": ["regulatory requirements", "policy implementation", "monitoring"],
                "typical_concerns": ["compliance gaps", "regulatory changes", "reporting obligations"]
            },
            "risk_manager": {
                "focus_areas": ["risk assessment", "control effectiveness", "risk appetite"],
                "typical_concerns": ["risk exposure", "control failures", "risk monitoring"]
            },
            "regulatory_inspector": {
                "focus_areas": ["compliance verification", "enforcement", "regulatory standards"],
                "typical_concerns": ["compliance breaches", "regulatory effectiveness", "industry practices"]
            }
        }
        
        return role_contexts.get(user_role, {})
    
    def _get_regulatory_context(self, regulatory_scope: List[str]) -> Dict[str, Any]:
        """Get context for regulatory scope."""
        
        if not regulatory_scope:
            return {}
        
        scope_contexts = {
            "sysc": {"focus": "governance and systems", "key_areas": ["senior management", "risk management", "compliance"]},
            "cobs": {"focus": "conduct of business", "key_areas": ["customer treatment", "advice", "disclosure"]},
            "conc": {"focus": "consumer credit", "key_areas": ["affordability", "forbearance", "fair treatment"]},
        }
        
        context = {}
        for scope in regulatory_scope:
            if scope.lower() in scope_contexts:
                context[scope] = scope_contexts[scope.lower()]
        
        return context
    
    def _get_business_context(self, business_functions: List[str]) -> Dict[str, Any]:
        """Get context for business functions."""
        
        if not business_functions:
            return {}
        
        function_contexts = {
            "governance": {"responsibilities": ["oversight", "accountability", "delegation"]},
            "risk_management": {"responsibilities": ["identification", "assessment", "mitigation", "monitoring"]},
            "compliance": {"responsibilities": ["monitoring", "reporting", "training", "advice"]},
            "client_facing": {"responsibilities": ["advice", "execution", "disclosure", "complaints"]}
        }
        
        context = {}
        for function in business_functions:
            if function in function_contexts:
                context[function] = function_contexts[function]
        
        return context
    
    def _deduplicate_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results."""
        
        seen_entities = set()
        deduplicated = []
        
        for result in results:
            entity_id = result.entity_id
            
            if entity_id not in seen_entities:
                seen_entities.add(entity_id)
                deduplicated.append(result)
        
        return deduplicated


class DualLevelRetrievalSystem:
    """Complete dual-level retrieval system for FSCompliance."""
    
    def __init__(
        self, 
        knowledge_base: BaseKnowledgeBase,
        lightrag: Optional[FSComplianceLightRAG] = None,
        retriever: Optional[BaseDualLevelRetriever] = None
    ):
        self.knowledge_base = knowledge_base
        self.lightrag = lightrag
        self.retriever = retriever or StandardDualLevelRetriever(knowledge_base, lightrag)
        
        self.system_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_confidence": 0.0
        }
    
    async def process_compliance_query(
        self, 
        query: ComplianceQuery,
        retrieval_level: RetrievalLevel = RetrievalLevel.HYBRID
    ) -> ComplianceResponse:
        """Process compliance query and generate comprehensive response."""
        
        try:
            # Create retrieval context from query
            context = RetrievalContext(
                user_role=query.user_role.value,
                firm_type=query.fca_firm_type.value if query.fca_firm_type else None,
                business_functions=[],  # Could be extracted from query content
                regulatory_scope=[sb.value for sb in query.sourcebooks_in_scope],
                priority_level="normal"
            )
            
            # Perform retrieval
            retrieval_result = await self.retriever.retrieve(query, retrieval_level, context)
            
            # Generate compliance response
            response = await self._generate_compliance_response(query, retrieval_result, context)
            
            # Update system statistics
            self._update_system_stats(retrieval_result, success=True)
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Update system statistics
            self._update_system_stats(None, success=False)
            
            # Return error response
            return ComplianceResponse(
                query_id=query.id,
                compliance_status=ComplianceStatus.UNCLEAR,
                confidence_score=0.0,
                risk_level=RiskLevel.HIGH,
                action_required=True,
                recommendations=["Query processing failed - please contact support"],
                regulatory_impact="Unable to assess due to processing error"
            )
    
    async def _generate_compliance_response(
        self, 
        query: ComplianceQuery, 
        retrieval_result: RetrievalResult,
        context: RetrievalContext
    ) -> ComplianceResponse:
        """Generate compliance response from retrieval results."""
        
        # Analyze retrieval results to determine compliance status
        compliance_status = self._assess_compliance_status(retrieval_result)
        
        # Assess risk level
        risk_level = self._assess_risk_level(retrieval_result, context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(retrieval_result, context)
        
        # Determine if action is required
        action_required = self._determine_action_required(compliance_status, risk_level, retrieval_result)
        
        # Extract relevant sourcebooks
        sourcebooks_referenced = self._extract_sourcebooks_referenced(retrieval_result)
        
        # Generate regulatory impact assessment
        regulatory_impact = self._assess_regulatory_impact(retrieval_result, context)
        
        # Determine if escalation is needed
        escalation_required = self._determine_escalation_required(risk_level, compliance_status)
        
        return ComplianceResponse(
            query_id=query.id,
            compliance_status=compliance_status,
            confidence_score=retrieval_result.confidence_score,
            risk_level=risk_level,
            action_required=action_required,
            fca_sourcebooks_referenced=sourcebooks_referenced,
            recommendations=recommendations,
            regulatory_impact=regulatory_impact,
            escalation_required=escalation_required
        )
    
    def _assess_compliance_status(self, retrieval_result: RetrievalResult) -> ComplianceStatus:
        """Assess compliance status from retrieval results."""
        
        # Simple heuristic - would be more sophisticated in production
        if len(retrieval_result.specific_requirements) == 0:
            return ComplianceStatus.UNCLEAR
        
        # Check confidence score
        if retrieval_result.confidence_score >= 0.8:
            return ComplianceStatus.COMPLIANT
        elif retrieval_result.confidence_score >= 0.6:
            return ComplianceStatus.PARTIAL_COMPLIANCE
        else:
            return ComplianceStatus.REQUIRES_REVIEW
    
    def _assess_risk_level(self, retrieval_result: RetrievalResult, context: RetrievalContext) -> RiskLevel:
        """Assess risk level from results and context."""
        
        # Risk assessment heuristics
        high_risk_indicators = len([req for req in retrieval_result.specific_requirements 
                                   if req.relevance_score < 0.5])
        
        if high_risk_indicators > 3:
            return RiskLevel.HIGH
        elif high_risk_indicators > 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(
        self, 
        retrieval_result: RetrievalResult, 
        context: RetrievalContext
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Based on retrieval results
        if len(retrieval_result.specific_requirements) > 0:
            recommendations.append("Review specific regulatory requirements identified")
        
        if len(retrieval_result.thematic_patterns) > 0:
            recommendations.append("Consider broader regulatory themes and their implications")
        
        # Based on context
        if context.user_role == "compliance_officer":
            recommendations.append("Update compliance monitoring procedures")
        elif context.user_role == "risk_manager":
            recommendations.append("Review risk assessment frameworks")
        
        if not recommendations:
            recommendations.append("Seek additional regulatory guidance")
        
        return recommendations
    
    def _determine_action_required(
        self, 
        compliance_status: ComplianceStatus, 
        risk_level: RiskLevel, 
        retrieval_result: RetrievalResult
    ) -> bool:
        """Determine if immediate action is required."""
        
        return (
            compliance_status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.REQUIRES_REVIEW] or
            risk_level == RiskLevel.HIGH or
            retrieval_result.confidence_score < 0.6
        )
    
    def _extract_sourcebooks_referenced(self, retrieval_result: RetrievalResult) -> List[FCASourcebook]:
        """Extract FCA sourcebooks referenced in results."""
        
        sourcebooks = set()
        
        for req in retrieval_result.specific_requirements:
            if hasattr(req.entity, 'sourcebook'):
                sourcebooks.add(req.entity.sourcebook)
        
        return list(sourcebooks)
    
    def _assess_regulatory_impact(
        self, 
        retrieval_result: RetrievalResult, 
        context: RetrievalContext
    ) -> str:
        """Assess regulatory impact based on results."""
        
        if retrieval_result.confidence_score >= 0.8:
            return "Clear regulatory requirements identified with high confidence"
        elif retrieval_result.confidence_score >= 0.6:
            return "Moderate regulatory implications requiring further review"
        else:
            return "Uncertain regulatory impact - additional analysis required"
    
    def _determine_escalation_required(
        self, 
        risk_level: RiskLevel, 
        compliance_status: ComplianceStatus
    ) -> bool:
        """Determine if escalation to senior management is required."""
        
        return (
            risk_level == RiskLevel.HIGH or
            compliance_status == ComplianceStatus.NON_COMPLIANT
        )
    
    def _update_system_stats(self, retrieval_result: Optional[RetrievalResult], success: bool):
        """Update system statistics."""
        
        self.system_stats["total_queries"] += 1
        
        if success:
            self.system_stats["successful_retrievals"] += 1
            
            if retrieval_result:
                # Update average confidence
                total_successful = self.system_stats["successful_retrievals"]
                current_avg = self.system_stats["average_confidence"]
                new_avg = ((current_avg * (total_successful - 1)) + retrieval_result.confidence_score) / total_successful
                self.system_stats["average_confidence"] = new_avg
        else:
            self.system_stats["failed_retrievals"] += 1
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        
        return {
            **self.system_stats,
            "retriever_stats": self.retriever.retrieval_stats,
            "success_rate": self.system_stats["successful_retrievals"] / max(self.system_stats["total_queries"], 1)
        }


# Factory functions
async def create_dual_level_retrieval_system(
    knowledge_base: BaseKnowledgeBase,
    lightrag: Optional[FSComplianceLightRAG] = None
) -> DualLevelRetrievalSystem:
    """Create complete dual-level retrieval system."""
    
    retriever = StandardDualLevelRetriever(knowledge_base, lightrag)
    
    return DualLevelRetrievalSystem(knowledge_base, lightrag, retriever)