"""Context-aware response generation for compliance queries."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from ..models import ComplianceResponse
from .processing import ProcessedQuery
from .routing import QueryType, QueryIntent

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Types of responses that can be generated."""
    DIRECT_ANSWER = "direct_answer"           # Direct factual answer
    GUIDED_EXPLANATION = "guided_explanation"  # Step-by-step explanation
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # Comparison of options
    RECOMMENDATION_LIST = "recommendation_list"  # List of recommendations
    PROCESS_GUIDE = "process_guide"          # How-to guide
    RISK_ASSESSMENT = "risk_assessment"      # Risk analysis
    COMPLIANCE_STATUS = "compliance_status"  # Status report
    REFERENCE_COLLECTION = "reference_collection"  # Collection of references
    INTERPRETIVE_GUIDANCE = "interpretive_guidance"  # Regulatory interpretation
    CONTEXTUAL_SUMMARY = "contextual_summary"  # Summary with context


class ResponseFormat(str, Enum):
    """Formats for response presentation."""
    NARRATIVE = "narrative"           # Flowing narrative text
    STRUCTURED = "structured"        # Structured sections
    BULLET_POINTS = "bullet_points"  # Bullet point list
    NUMBERED_STEPS = "numbered_steps"  # Numbered steps
    TABLE = "table"                  # Tabular format
    FLOWCHART = "flowchart"         # Process flow
    CHECKLIST = "checklist"         # Actionable checklist
    FAQ = "faq"                     # Question and answer format


class ResponseTone(str, Enum):
    """Tone of response delivery."""
    FORMAL = "formal"               # Formal regulatory tone
    CONVERSATIONAL = "conversational"  # Friendly conversational
    TECHNICAL = "technical"         # Technical and precise
    EDUCATIONAL = "educational"     # Teaching/explanatory
    ADVISORY = "advisory"          # Professional advisory
    CAUTIONARY = "cautionary"      # Warning/cautious tone


class ConfidenceIndicator(str, Enum):
    """Confidence indicators for response content."""
    HIGH_CERTAINTY = "high_certainty"      # Very confident in answer
    MEDIUM_CERTAINTY = "medium_certainty"  # Moderately confident
    LOW_CERTAINTY = "low_certainty"        # Less confident
    REQUIRES_VALIDATION = "requires_validation"  # Needs expert validation
    INTERPRETATION_NEEDED = "interpretation_needed"  # Regulatory interpretation needed


@dataclass
class ResponseSection:
    """A section of a structured response."""
    
    section_id: str
    title: str
    content: str
    section_type: str = "content"  # content, summary, recommendation, warning
    confidence: float = 0.8
    sources: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class ResponseMetadata:
    """Metadata about response generation."""
    
    generation_algorithm: str
    generation_timestamp: datetime
    processing_time_ms: float
    confidence_score: float
    source_count: int
    completeness_score: float
    
    validation_status: str = "unvalidated"
    expert_review_needed: bool = False
    regulatory_disclaimer: bool = True


class GeneratedResponse(BaseModel):
    """A generated response to a compliance query."""
    
    # Core response data
    response_id: str = Field(..., description="Unique response identifier")
    query_id: str = Field(..., description="ID of the original query")
    
    # Response content
    response_type: ResponseType = Field(..., description="Type of response")
    response_format: ResponseFormat = Field(..., description="Format of response")
    response_tone: ResponseTone = Field(..., description="Tone of response")
    
    # Main content
    main_content: str = Field(..., description="Primary response content")
    sections: List[ResponseSection] = Field(default_factory=list, description="Structured response sections")
    
    # Supporting information
    executive_summary: Optional[str] = Field(None, description="Executive summary")
    key_points: List[str] = Field(default_factory=list, description="Key takeaway points")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    
    # References and sources
    regulatory_references: List[str] = Field(default_factory=list, description="Regulatory references cited")
    guidance_sources: List[str] = Field(default_factory=list, description="Guidance sources used")
    related_topics: List[str] = Field(default_factory=list, description="Related topics for further reading")
    
    # Quality indicators
    confidence_indicator: ConfidenceIndicator = Field(..., description="Overall confidence level")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Response completeness")
    accuracy_indicator: str = Field(..., description="Accuracy assessment")
    
    # Context and personalization
    context_relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance to user context")
    personalization_level: str = Field(..., description="Level of personalization applied")
    
    # Disclaimers and warnings
    regulatory_disclaimer: str = Field(..., description="Standard regulatory disclaimer")
    limitations: List[str] = Field(default_factory=list, description="Response limitations")
    update_frequency: Optional[str] = Field(None, description="How often this info should be updated")
    
    # Metadata
    metadata: ResponseMetadata = Field(..., description="Response generation metadata")
    
    def get_response_summary(self) -> Dict[str, Any]:
        """Get high-level response summary."""
        return {
            "response_type": self.response_type.value,
            "format": self.response_format.value,
            "confidence": self.confidence_indicator.value,
            "completeness": self.completeness_score,
            "section_count": len(self.sections),
            "key_points_count": len(self.key_points),
            "recommendations_count": len(self.recommendations),
            "source_count": self.metadata.source_count
        }
    
    def is_high_quality(self) -> bool:
        """Check if response meets high quality thresholds."""
        return (
            self.completeness_score >= 0.8 and
            self.context_relevance >= 0.7 and
            self.confidence_indicator in [ConfidenceIndicator.HIGH_CERTAINTY, ConfidenceIndicator.MEDIUM_CERTAINTY] and
            len(self.key_points) >= 2
        )


class BaseResponseGenerator(ABC):
    """Abstract base class for response generation algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation_stats = {
            "responses_generated": 0,
            "high_quality_responses": 0,
            "generation_errors": 0,
            "average_confidence": 0.0,
            "average_generation_time_ms": 0.0
        }
    
    @abstractmethod
    async def generate_response(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GeneratedResponse:
        """Generate a response to a processed query."""
        pass
    
    @abstractmethod
    def get_generator_name(self) -> str:
        """Get the name of the response generation algorithm."""
        pass
    
    def _generate_response_id(self, query_id: str) -> str:
        """Generate unique response identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return f"RESP_{query_id}_{timestamp}"


class ContextAwareResponseGenerator(BaseResponseGenerator):
    """Context-aware response generator with personalization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.response_templates = self._initialize_response_templates()
        self.tone_guidelines = self._initialize_tone_guidelines()
        self.domain_expertise = self._initialize_domain_expertise()
    
    def get_generator_name(self) -> str:
        return "context_aware_generator"
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize response templates for different query types."""
        
        return {
            "requirement_lookup": {
                "introduction": "Based on your query about {subject}, here are the relevant regulatory requirements:",
                "structure": ["Executive Summary", "Key Requirements", "Implementation Guidance", "Related References"],
                "conclusion": "These requirements apply to {firm_types} and should be implemented {urgency}."
            },
            "gap_analysis": {
                "introduction": "I've analyzed your current compliance position against {domain} requirements:",
                "structure": ["Current Status", "Identified Gaps", "Risk Assessment", "Remediation Plan"],
                "conclusion": "Addressing these gaps should be prioritized based on {risk_level} risk factors."
            },
            "compliance_check": {
                "introduction": "Here's an assessment of your compliance with {requirements}:",
                "structure": ["Compliance Summary", "Areas of Strength", "Areas for Improvement", "Next Steps"],
                "conclusion": "Overall compliance appears {status} with {priority} priority actions needed."
            },
            "guidance_request": {
                "introduction": "Here's practical guidance on {topic} based on regulatory expectations:",
                "structure": ["Key Principles", "Step-by-Step Guidance", "Best Practices", "Common Pitfalls"],
                "conclusion": "Following this guidance will help ensure compliance with {regulatory_framework}."
            },
            "interpretation": {
                "introduction": "Let me explain {concept} in the context of {regulatory_framework}:",
                "structure": ["Definition", "Regulatory Context", "Practical Application", "Examples"],
                "conclusion": "This interpretation is based on {sources} and current regulatory understanding."
            },
            "risk_assessment": {
                "introduction": "Here's a risk assessment for {scenario} under current regulations:",
                "structure": ["Risk Overview", "Key Risk Factors", "Likelihood Assessment", "Impact Analysis", "Mitigation Strategies"],
                "conclusion": "The overall risk level is assessed as {risk_level} with {mitigation} recommended."
            }
        }
    
    def _initialize_tone_guidelines(self) -> Dict[ResponseTone, Dict[str, str]]:
        """Initialize tone guidelines for different response styles."""
        
        return {
            ResponseTone.FORMAL: {
                "style": "Use formal regulatory language, precise terminology, and structured presentation",
                "vocabulary": "Technical regulatory terms, formal constructions",
                "disclaimer_level": "full"
            },
            ResponseTone.CONVERSATIONAL: {
                "style": "Use clear, accessible language while maintaining accuracy",
                "vocabulary": "Plain English with technical terms explained",
                "disclaimer_level": "standard"
            },
            ResponseTone.TECHNICAL: {
                "style": "Use precise technical language for expert audience",
                "vocabulary": "Technical regulatory and legal terminology",
                "disclaimer_level": "technical"
            },
            ResponseTone.EDUCATIONAL: {
                "style": "Structure as learning material with explanations and examples",
                "vocabulary": "Educational, building from basics to complexity",
                "disclaimer_level": "educational"
            },
            ResponseTone.ADVISORY: {
                "style": "Professional advice format with clear recommendations",
                "vocabulary": "Advisory language with action-oriented guidance",
                "disclaimer_level": "advisory"
            },
            ResponseTone.CAUTIONARY: {
                "style": "Emphasize risks, limitations, and need for professional advice",
                "vocabulary": "Risk-aware language with appropriate warnings",
                "disclaimer_level": "full_cautionary"
            }
        }
    
    def _initialize_domain_expertise(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific expertise and knowledge."""
        
        return {
            "governance": {
                "key_concepts": ["oversight", "accountability", "delegation", "structure"],
                "common_issues": ["unclear responsibilities", "inadequate oversight", "poor governance structure"],
                "best_practices": ["clear governance framework", "defined roles", "regular review"],
                "regulatory_focus": ["SYSC 4", "SYSC 5", "senior managers regime"]
            },
            "risk_management": {
                "key_concepts": ["identification", "assessment", "mitigation", "monitoring"],
                "common_issues": ["inadequate risk culture", "poor risk appetite", "insufficient controls"],
                "best_practices": ["three lines of defense", "risk appetite framework", "stress testing"],
                "regulatory_focus": ["SYSC 7", "ICAAP", "stress testing"]
            },
            "conduct": {
                "key_concepts": ["customer outcomes", "fair treatment", "suitability", "vulnerability"],
                "common_issues": ["poor customer outcomes", "unsuitable advice", "inadequate disclosure"],
                "best_practices": ["customer journey mapping", "outcome testing", "vulnerability identification"],
                "regulatory_focus": ["Consumer Duty", "COBS", "treating customers fairly"]
            }
        }
    
    async def generate_response(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GeneratedResponse:
        """Generate a context-aware response."""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Generating response for query {processed_query.query_id}")
            
            # Determine response characteristics
            response_type = self._determine_response_type(processed_query)
            response_format = self._determine_response_format(processed_query, context)
            response_tone = self._determine_response_tone(processed_query, context)
            
            # Generate main content
            main_content = await self._generate_main_content(
                processed_query, search_results, response_type, response_tone
            )
            
            # Generate structured sections
            sections = await self._generate_sections(
                processed_query, search_results, response_type
            )
            
            # Generate supporting elements
            executive_summary = self._generate_executive_summary(processed_query, main_content)
            key_points = self._extract_key_points(processed_query, main_content, sections)
            recommendations = self._generate_recommendations(processed_query, search_results)
            
            # Compile references and sources
            reg_references, guidance_sources = self._compile_sources(search_results)
            related_topics = self._identify_related_topics(processed_query)
            
            # Assess quality indicators
            confidence_indicator = self._assess_confidence(processed_query, search_results)
            completeness_score = self._assess_completeness(processed_query, sections, search_results)
            accuracy_indicator = self._assess_accuracy(search_results)
            
            # Calculate context relevance
            context_relevance = self._calculate_context_relevance(processed_query, context)
            personalization_level = self._determine_personalization_level(context)
            
            # Generate disclaimers and limitations
            disclaimer = self._generate_disclaimer(response_tone, confidence_indicator)
            limitations = self._identify_limitations(processed_query, search_results)
            
            # Create metadata
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            metadata = ResponseMetadata(
                generation_algorithm=self.get_generator_name(),
                generation_timestamp=end_time,
                processing_time_ms=processing_time,
                confidence_score=self._calculate_confidence_score(confidence_indicator, completeness_score),
                source_count=len(reg_references) + len(guidance_sources),
                completeness_score=completeness_score,
                expert_review_needed=self._needs_expert_review(processed_query, confidence_indicator)
            )
            
            # Create generated response
            response = GeneratedResponse(
                response_id=self._generate_response_id(processed_query.query_id),
                query_id=processed_query.query_id,
                response_type=response_type,
                response_format=response_format,
                response_tone=response_tone,
                main_content=main_content,
                sections=sections,
                executive_summary=executive_summary,
                key_points=key_points,
                recommendations=recommendations,
                regulatory_references=reg_references,
                guidance_sources=guidance_sources,
                related_topics=related_topics,
                confidence_indicator=confidence_indicator,
                completeness_score=completeness_score,
                accuracy_indicator=accuracy_indicator,
                context_relevance=context_relevance,
                personalization_level=personalization_level,
                regulatory_disclaimer=disclaimer,
                limitations=limitations,
                update_frequency=self._determine_update_frequency(processed_query),
                metadata=metadata
            )
            
            # Update statistics
            self.generation_stats["responses_generated"] += 1
            
            if response.is_high_quality():
                self.generation_stats["high_quality_responses"] += 1
            
            # Update averages
            self._update_average_stats(response)
            
            logger.info(f"Response generated: {response_type.value} ({confidence_indicator.value})")
            
            return response
            
        except Exception as e:
            self.generation_stats["generation_errors"] += 1
            logger.error(f"Error generating response: {e}")
            raise
    
    def _determine_response_type(self, processed_query: ProcessedQuery) -> ResponseType:
        """Determine the most appropriate response type."""
        
        query_type = processed_query.classified_query.query_type
        query_intent = processed_query.classified_query.query_intent
        
        # Map query characteristics to response types
        type_mapping = {
            (QueryType.REQUIREMENT_LOOKUP, QueryIntent.FIND): ResponseType.REFERENCE_COLLECTION,
            (QueryType.REQUIREMENT_LOOKUP, QueryIntent.EXPLAIN): ResponseType.GUIDED_EXPLANATION,
            (QueryType.GAP_ANALYSIS, QueryIntent.ASSESS): ResponseType.COMPLIANCE_STATUS,
            (QueryType.COMPLIANCE_CHECK, QueryIntent.CHECK): ResponseType.COMPLIANCE_STATUS,
            (QueryType.GUIDANCE_REQUEST, QueryIntent.RECOMMEND): ResponseType.PROCESS_GUIDE,
            (QueryType.INTERPRETATION, QueryIntent.EXPLAIN): ResponseType.INTERPRETIVE_GUIDANCE,
            (QueryType.COMPARISON, QueryIntent.COMPARE): ResponseType.COMPARATIVE_ANALYSIS,
            (QueryType.RISK_ASSESSMENT, QueryIntent.ASSESS): ResponseType.RISK_ASSESSMENT,
            (QueryType.REMEDIATION, QueryIntent.RECOMMEND): ResponseType.RECOMMENDATION_LIST
        }
        
        response_type = type_mapping.get((query_type, query_intent))
        
        if response_type:
            return response_type
        
        # Default mappings by query type
        type_defaults = {
            QueryType.REQUIREMENT_LOOKUP: ResponseType.REFERENCE_COLLECTION,
            QueryType.GAP_ANALYSIS: ResponseType.COMPLIANCE_STATUS,
            QueryType.COMPLIANCE_CHECK: ResponseType.COMPLIANCE_STATUS,
            QueryType.GUIDANCE_REQUEST: ResponseType.PROCESS_GUIDE,
            QueryType.INTERPRETATION: ResponseType.INTERPRETIVE_GUIDANCE,
            QueryType.RISK_ASSESSMENT: ResponseType.RISK_ASSESSMENT,
            QueryType.REMEDIATION: ResponseType.RECOMMENDATION_LIST
        }
        
        return type_defaults.get(query_type, ResponseType.CONTEXTUAL_SUMMARY)
    
    def _determine_response_format(
        self, 
        processed_query: ProcessedQuery, 
        context: Optional[Dict[str, Any]]
    ) -> ResponseFormat:
        """Determine the most appropriate response format."""
        
        query_type = processed_query.classified_query.query_type
        complexity = processed_query.classified_query.query_complexity
        
        # User preference from context
        if context and context.get("preferred_format"):
            return ResponseFormat(context["preferred_format"])
        
        # Format by query type and complexity
        if query_type in [QueryType.GUIDANCE_REQUEST, QueryType.REMEDIATION]:
            return ResponseFormat.NUMBERED_STEPS
        elif query_type == QueryType.REQUIREMENT_LOOKUP:
            return ResponseFormat.STRUCTURED
        elif query_type == QueryType.COMPARISON:
            return ResponseFormat.TABLE
        elif query_type == QueryType.GAP_ANALYSIS:
            return ResponseFormat.BULLET_POINTS
        elif complexity.value == "expert":
            return ResponseFormat.NARRATIVE
        else:
            return ResponseFormat.STRUCTURED
    
    def _determine_response_tone(
        self, 
        processed_query: ProcessedQuery, 
        context: Optional[Dict[str, Any]]
    ) -> ResponseTone:
        """Determine the most appropriate response tone."""
        
        # User role from context
        if context:
            user_role = context.get("user_role", "").lower()
            if "legal" in user_role or "compliance officer" in user_role:
                return ResponseTone.FORMAL
            elif "training" in user_role or "new" in user_role:
                return ResponseTone.EDUCATIONAL
            elif "risk" in user_role or "senior" in user_role:
                return ResponseTone.ADVISORY
        
        # Query characteristics
        complexity = processed_query.classified_query.query_complexity
        urgency = processed_query.classified_query.query_urgency
        
        if complexity.value == "expert":
            return ResponseTone.TECHNICAL
        elif urgency.value in ["high", "critical"]:
            return ResponseTone.CAUTIONARY
        else:
            return ResponseTone.CONVERSATIONAL
    
    async def _generate_main_content(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]],
        response_type: ResponseType,
        response_tone: ResponseTone
    ) -> str:
        """Generate the main response content."""
        
        query_type = processed_query.classified_query.query_type.value
        template = self.response_templates.get(query_type, {})
        
        # Extract key information for template
        subject = self._extract_subject_from_query(processed_query)
        domain = processed_query.classified_query.query_domain.value
        
        # Generate introduction
        introduction = template.get("introduction", "Here's information about your query:")
        introduction = introduction.format(
            subject=subject,
            domain=domain,
            topic=subject,
            requirements=subject
        )
        
        # Generate main body based on search results
        main_body = self._generate_content_from_results(search_results, response_type, response_tone)
        
        # Generate conclusion
        conclusion = template.get("conclusion", "Please review this information carefully.")
        conclusion = self._personalize_conclusion(conclusion, processed_query)
        
        return f"{introduction}\n\n{main_body}\n\n{conclusion}"
    
    async def _generate_sections(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]],
        response_type: ResponseType
    ) -> List[ResponseSection]:
        """Generate structured response sections."""
        
        sections = []
        query_type = processed_query.classified_query.query_type.value
        template = self.response_templates.get(query_type, {})
        section_names = template.get("structure", ["Overview", "Details", "Summary"])
        
        for i, section_name in enumerate(section_names):
            section_content = self._generate_section_content(
                section_name, processed_query, search_results, i
            )
            
            section = ResponseSection(
                section_id=f"section_{i+1}",
                title=section_name,
                content=section_content,
                section_type=self._determine_section_type(section_name),
                confidence=0.8,  # Default confidence
                sources=self._extract_section_sources(search_results, section_name)
            )
            
            sections.append(section)
        
        return sections
    
    def _generate_executive_summary(self, processed_query: ProcessedQuery, main_content: str) -> str:
        """Generate executive summary of the response."""
        
        query_type = processed_query.classified_query.query_type
        domain = processed_query.classified_query.query_domain.value
        
        # Extract key sentence from main content
        sentences = main_content.split('. ')
        key_sentence = sentences[1] if len(sentences) > 1 else sentences[0]
        
        if query_type == QueryType.REQUIREMENT_LOOKUP:
            return f"This response addresses regulatory requirements in {domain}. {key_sentence}."
        elif query_type == QueryType.GAP_ANALYSIS:
            return f"Compliance gap analysis has been performed for {domain}. {key_sentence}."
        elif query_type == QueryType.GUIDANCE_REQUEST:
            return f"Practical guidance provided for {domain} compliance. {key_sentence}."
        else:
            return f"Analysis completed for your {domain} query. {key_sentence}."
    
    def _extract_key_points(
        self, 
        processed_query: ProcessedQuery, 
        main_content: str, 
        sections: List[ResponseSection]
    ) -> List[str]:
        """Extract key points from the response."""
        
        key_points = []
        
        # Extract from sections
        for section in sections[:3]:  # Top 3 sections
            # Look for bullet points or numbered items
            content_lines = section.content.split('\n')
            for line in content_lines:
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or (line and line[0].isdigit()):
                    key_points.append(line.lstrip('•-0123456789. '))
        
        # If no structured points found, extract from main content
        if not key_points:
            sentences = main_content.split('. ')
            # Look for sentences with key indicators
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in ['must', 'should', 'required', 'important', 'key']):
                    key_points.append(sentence.strip())
        
        return key_points[:5]  # Limit to 5 key points
    
    def _generate_recommendations(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        query_type = processed_query.classified_query.query_type
        domain = processed_query.classified_query.query_domain.value
        
        # Domain-specific recommendations
        domain_info = self.domain_expertise.get(domain, {})
        best_practices = domain_info.get("best_practices", [])
        
        if query_type == QueryType.GAP_ANALYSIS:
            recommendations.extend([
                "Review and update existing policies and procedures",
                "Implement additional controls where gaps identified",
                "Establish regular monitoring and testing procedures"
            ])
        elif query_type == QueryType.COMPLIANCE_CHECK:
            recommendations.extend([
                "Document current compliance status",
                "Address any identified deficiencies promptly",
                "Establish ongoing compliance monitoring"
            ])
        elif query_type == QueryType.GUIDANCE_REQUEST:
            recommendations.extend(best_practices[:3])
        
        # Add search result-based recommendations
        if search_results:
            for result in search_results[:2]:
                if result.get("recommendations"):
                    recommendations.extend(result["recommendations"][:2])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _compile_sources(
        self, 
        search_results: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[str], List[str]]:
        """Compile regulatory references and guidance sources."""
        
        reg_references = []
        guidance_sources = []
        
        if search_results:
            for result in search_results:
                # Extract regulatory references
                if result.get("regulatory_source"):
                    reg_references.append(result["regulatory_source"])
                
                # Extract guidance sources
                if result.get("source_type") == "guidance" and result.get("source_title"):
                    guidance_sources.append(result["source_title"])
        
        return list(set(reg_references)), list(set(guidance_sources))
    
    def _identify_related_topics(self, processed_query: ProcessedQuery) -> List[str]:
        """Identify related topics for further reading."""
        
        domain = processed_query.classified_query.query_domain.value
        query_type = processed_query.classified_query.query_type
        
        domain_info = self.domain_expertise.get(domain, {})
        key_concepts = domain_info.get("key_concepts", [])
        
        related_topics = []
        
        # Add domain concepts
        related_topics.extend(key_concepts[:3])
        
        # Add query-type specific topics
        if query_type == QueryType.GOVERNANCE:
            related_topics.extend(["senior managers regime", "board effectiveness", "delegation frameworks"])
        elif query_type == QueryType.RISK_MANAGEMENT:
            related_topics.extend(["risk appetite", "stress testing", "operational resilience"])
        elif query_type == QueryType.CONDUCT:
            related_topics.extend(["consumer duty", "vulnerable customers", "product governance"])
        
        return related_topics[:5]
    
    def _assess_confidence(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]]
    ) -> ConfidenceIndicator:
        """Assess confidence in the response."""
        
        query_confidence = processed_query.classified_query.classification_confidence
        processing_confidence = processed_query.processing_confidence
        
        # Factor in search results quality
        search_quality = 0.7  # Default
        if search_results:
            result_count = len(search_results)
            if result_count >= 5:
                search_quality = 0.9
            elif result_count >= 3:
                search_quality = 0.8
            elif result_count >= 1:
                search_quality = 0.7
            else:
                search_quality = 0.4
        
        overall_confidence = (query_confidence + processing_confidence + search_quality) / 3
        
        if overall_confidence >= 0.9:
            return ConfidenceIndicator.HIGH_CERTAINTY
        elif overall_confidence >= 0.75:
            return ConfidenceIndicator.MEDIUM_CERTAINTY
        elif overall_confidence >= 0.6:
            return ConfidenceIndicator.LOW_CERTAINTY
        elif overall_confidence >= 0.4:
            return ConfidenceIndicator.REQUIRES_VALIDATION
        else:
            return ConfidenceIndicator.INTERPRETATION_NEEDED
    
    def _assess_completeness(
        self, 
        processed_query: ProcessedQuery,
        sections: List[ResponseSection],
        search_results: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Assess completeness of the response."""
        
        completeness_factors = []
        
        # Query completeness
        completeness_factors.append(processed_query.completeness_score)
        
        # Section completeness
        section_completeness = min(len(sections) / 4, 1.0)  # Expect ~4 sections
        completeness_factors.append(section_completeness)
        
        # Search results completeness
        if search_results:
            result_completeness = min(len(search_results) / 5, 1.0)  # Expect ~5 results
        else:
            result_completeness = 0.3  # Lower if no search results
        completeness_factors.append(result_completeness)
        
        # Content length completeness
        total_content_length = sum(len(section.content) for section in sections)
        content_completeness = min(total_content_length / 1000, 1.0)  # Expect ~1000 chars
        completeness_factors.append(content_completeness)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _assess_accuracy(self, search_results: Optional[List[Dict[str, Any]]]) -> str:
        """Assess accuracy indicator for the response."""
        
        if not search_results:
            return "Limited - No search results available"
        
        result_count = len(search_results)
        
        if result_count >= 5:
            return "High - Multiple authoritative sources"
        elif result_count >= 3:
            return "Good - Several relevant sources"
        elif result_count >= 1:
            return "Moderate - Limited sources available"
        else:
            return "Low - Insufficient source material"
    
    def _calculate_context_relevance(
        self, 
        processed_query: ProcessedQuery,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate relevance to user context."""
        
        if not context:
            return 0.6  # Default relevance without context
        
        relevance_factors = []
        
        # Role relevance
        user_role = context.get("user_role", "").lower()
        query_domain = processed_query.classified_query.query_domain.value
        
        role_domain_relevance = {
            "compliance": ["conduct", "governance", "reporting"],
            "risk": ["risk_management", "operational", "prudential"],
            "governance": ["governance", "risk_management"],
            "audit": ["governance", "risk_management", "operational"]
        }
        
        role_relevance = 0.5
        for role_key, relevant_domains in role_domain_relevance.items():
            if role_key in user_role and query_domain in relevant_domains:
                role_relevance = 0.9
                break
        
        relevance_factors.append(role_relevance)
        
        # Firm type relevance
        firm_type = context.get("firm_type", "").lower()
        query_entities = processed_query.semantic_representation.get("entities", {})
        
        if firm_type and query_entities.get("firm_types"):
            if any(firm_type in entity.lower() for entity in query_entities["firm_types"]):
                relevance_factors.append(0.9)
            else:
                relevance_factors.append(0.6)
        else:
            relevance_factors.append(0.7)
        
        return sum(relevance_factors) / len(relevance_factors)
    
    def _determine_personalization_level(self, context: Optional[Dict[str, Any]]) -> str:
        """Determine level of personalization applied."""
        
        if not context:
            return "Generic"
        
        personalization_indicators = [
            context.get("user_role"),
            context.get("firm_type"),
            context.get("business_function"),
            context.get("regulatory_scope")
        ]
        
        non_null_indicators = sum(1 for indicator in personalization_indicators if indicator)
        
        if non_null_indicators >= 3:
            return "Highly Personalized"
        elif non_null_indicators >= 2:
            return "Moderately Personalized"
        elif non_null_indicators >= 1:
            return "Lightly Personalized"
        else:
            return "Generic"
    
    def _generate_disclaimer(self, tone: ResponseTone, confidence: ConfidenceIndicator) -> str:
        """Generate appropriate regulatory disclaimer."""
        
        base_disclaimer = "This information is provided for general guidance only and should not be relied upon as legal or regulatory advice."
        
        if tone == ResponseTone.CAUTIONARY or confidence in [ConfidenceIndicator.LOW_CERTAINTY, ConfidenceIndicator.REQUIRES_VALIDATION]:
            return f"{base_disclaimer} Given the complexity of this area, we strongly recommend seeking professional regulatory advice before taking any action."
        elif confidence == ConfidenceIndicator.INTERPRETATION_NEEDED:
            return f"{base_disclaimer} This area may require specific regulatory interpretation. Please consult with your regulatory advisors."
        else:
            return f"{base_disclaimer} Always verify current regulatory requirements with official sources."
    
    def _identify_limitations(
        self, 
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Identify limitations of the response."""
        
        limitations = []
        
        # Query-based limitations
        if processed_query.processing_confidence < 0.8:
            limitations.append("Query interpretation may be incomplete")
        
        if not processed_query.is_valid:
            limitations.append("Query validation identified potential issues")
        
        # Search result limitations
        if not search_results:
            limitations.append("No search results available to support response")
        elif len(search_results) < 3:
            limitations.append("Limited source material available")
        
        # Complexity limitations
        if processed_query.classified_query.query_complexity.value == "expert":
            limitations.append("Complex regulatory area requiring expert interpretation")
        
        # Temporal limitations
        limitations.append("Regulatory requirements may change - verify current status")
        
        return limitations
    
    def _determine_update_frequency(self, processed_query: ProcessedQuery) -> str:
        """Determine how often this information should be updated."""
        
        domain = processed_query.classified_query.query_domain.value
        
        update_frequencies = {
            "prudential": "Quarterly - prudential requirements change frequently",
            "conduct": "Semi-annually - conduct regulations evolve regularly", 
            "governance": "Annually - governance requirements are relatively stable",
            "reporting": "Quarterly - reporting requirements change with regulatory cycles",
            "operational": "Annually - operational requirements evolve gradually"
        }
        
        return update_frequencies.get(domain, "Semi-annually - regulatory landscape evolves regularly")
    
    # Helper methods (simplified implementations)
    def _extract_subject_from_query(self, processed_query: ProcessedQuery) -> str:
        """Extract main subject from processed query."""
        subjects = [e for e in processed_query.query_elements if e.element_type.value == "subject"]
        return subjects[0].normalized_text if subjects else "regulatory requirements"
    
    def _generate_content_from_results(
        self, 
        search_results: Optional[List[Dict[str, Any]]],
        response_type: ResponseType,
        response_tone: ResponseTone
    ) -> str:
        """Generate main content from search results."""
        
        if not search_results:
            return "Based on available regulatory knowledge, here's relevant information for your query."
        
        # Combine relevant information from search results
        content_parts = []
        for result in search_results[:3]:  # Top 3 results
            if result.get("content"):
                content_parts.append(result["content"][:200])  # First 200 chars
        
        if content_parts:
            return " ".join(content_parts)
        else:
            return "Relevant regulatory information has been identified for your query."
    
    def _personalize_conclusion(self, conclusion: str, processed_query: ProcessedQuery) -> str:
        """Personalize conclusion based on query characteristics."""
        
        urgency = processed_query.classified_query.query_urgency.value
        
        if urgency == "critical":
            return conclusion.replace("review", "review urgently")
        elif urgency == "high":
            return conclusion.replace("review", "review promptly")
        
        return conclusion
    
    def _generate_section_content(
        self, 
        section_name: str,
        processed_query: ProcessedQuery,
        search_results: Optional[List[Dict[str, Any]]],
        section_index: int
    ) -> str:
        """Generate content for a specific section."""
        
        if section_name.lower() in ["overview", "summary", "executive summary"]:
            return f"This section provides an overview of {processed_query.classified_query.query_domain.value} requirements relevant to your query."
        elif section_name.lower() in ["requirements", "key requirements"]:
            return "Key regulatory requirements have been identified based on your query parameters."
        elif section_name.lower() in ["guidance", "implementation guidance"]:
            return "Implementation guidance is available to help meet these regulatory requirements."
        else:
            return f"Information about {section_name.lower()} relevant to your compliance query."
    
    def _determine_section_type(self, section_name: str) -> str:
        """Determine the type of section."""
        
        section_name_lower = section_name.lower()
        
        if any(word in section_name_lower for word in ["summary", "overview"]):
            return "summary"
        elif any(word in section_name_lower for word in ["recommendation", "action", "next steps"]):
            return "recommendation"
        elif any(word in section_name_lower for word in ["warning", "risk", "caution"]):
            return "warning"
        else:
            return "content"
    
    def _extract_section_sources(
        self, 
        search_results: Optional[List[Dict[str, Any]]],
        section_name: str
    ) -> List[str]:
        """Extract sources relevant to a section."""
        
        if not search_results:
            return []
        
        sources = []
        for result in search_results[:2]:  # Top 2 results per section
            if result.get("source"):
                sources.append(result["source"])
        
        return sources
    
    def _calculate_confidence_score(self, confidence_indicator: ConfidenceIndicator, completeness: float) -> float:
        """Calculate numeric confidence score."""
        
        confidence_scores = {
            ConfidenceIndicator.HIGH_CERTAINTY: 0.9,
            ConfidenceIndicator.MEDIUM_CERTAINTY: 0.75,
            ConfidenceIndicator.LOW_CERTAINTY: 0.6,
            ConfidenceIndicator.REQUIRES_VALIDATION: 0.4,
            ConfidenceIndicator.INTERPRETATION_NEEDED: 0.3
        }
        
        base_score = confidence_scores[confidence_indicator]
        return (base_score + completeness) / 2
    
    def _needs_expert_review(self, processed_query: ProcessedQuery, confidence: ConfidenceIndicator) -> bool:
        """Determine if expert review is needed."""
        
        return (
            processed_query.classified_query.query_complexity.value == "expert" or
            confidence in [ConfidenceIndicator.REQUIRES_VALIDATION, ConfidenceIndicator.INTERPRETATION_NEEDED] or
            processed_query.classified_query.query_urgency.value == "critical"
        )
    
    def _update_average_stats(self, response: GeneratedResponse):
        """Update average statistics."""
        
        # Update average confidence
        total_confidence = (
            self.generation_stats["average_confidence"] * (self.generation_stats["responses_generated"] - 1) +
            response.metadata.confidence_score
        )
        self.generation_stats["average_confidence"] = total_confidence / self.generation_stats["responses_generated"]
        
        # Update average generation time
        total_time = (
            self.generation_stats["average_generation_time_ms"] * (self.generation_stats["responses_generated"] - 1) +
            response.metadata.processing_time_ms
        )
        self.generation_stats["average_generation_time_ms"] = total_time / self.generation_stats["responses_generated"]


# Factory functions
def create_context_aware_generator(config: Optional[Dict[str, Any]] = None) -> ContextAwareResponseGenerator:
    """Create context-aware response generator."""
    if config is None:
        config = {}
    
    return ContextAwareResponseGenerator(config)