"""Entity extraction for regulatory content using LightRAG and custom NLP."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..models import FCAFirmType, FCASourcebook, RequirementType
from .lightrag_integration import RegulatoryEntityType, RegulatoryRelationType
from .parser import ExtractedRequirement, ParsedSection

logger = logging.getLogger(__name__)


class ExtractionMethod(str, Enum):
    """Methods for entity extraction."""
    REGEX_PATTERNS = "regex_patterns"
    KEYWORD_MATCHING = "keyword_matching"
    NLP_MODELS = "nlp_models"
    LIGHTRAG_LLM = "lightrag_llm"
    HYBRID = "hybrid"


class EntityConfidence(str, Enum):
    """Confidence levels for extracted entities."""
    HIGH = "high"       # 0.8+
    MEDIUM = "medium"   # 0.6-0.8
    LOW = "low"         # 0.4-0.6
    UNCERTAIN = "uncertain"  # <0.4


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from regulatory text."""
    
    entity_type: RegulatoryEntityType
    text: str
    normalized_value: str
    confidence: float
    extraction_method: ExtractionMethod
    source_text: str
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship between entities."""
    
    relationship_type: RegulatoryRelationType
    source_entity: ExtractedEntity
    target_entity: ExtractedEntity
    confidence: float
    extraction_method: ExtractionMethod
    source_text: str
    evidence_text: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EntityExtractionResult(BaseModel):
    """Result from entity extraction process."""
    
    source_id: str = Field(..., description="Source document/section ID")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="Extracted relationships")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_seconds: float = Field(..., description="Processing time")
    extraction_stats: Dict[str, int] = Field(default_factory=dict, description="Extraction statistics")
    
    class Config:
        arbitrary_types_allowed = True


class BaseEntityExtractor(ABC):
    """Abstract base class for entity extractors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extraction_stats = {
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "processing_errors": 0
        }
    
    @abstractmethod
    async def extract_entities(self, text: str, context: Dict[str, Any] = None) -> List[ExtractedEntity]:
        """Extract entities from text."""
        pass
    
    @abstractmethod
    async def extract_relationships(
        self, 
        text: str, 
        entities: List[ExtractedEntity], 
        context: Dict[str, Any] = None
    ) -> List[ExtractedRelationship]:
        """Extract relationships between entities."""
        pass
    
    async def extract_from_requirement(self, requirement: ExtractedRequirement) -> EntityExtractionResult:
        """Extract entities and relationships from a requirement."""
        start_time = datetime.utcnow()
        
        try:
            # Extract entities
            entities = await self.extract_entities(
                requirement.requirement_text,
                context={
                    "section_id": requirement.section_id,
                    "requirement_type": requirement.requirement_type,
                    "severity": requirement.severity
                }
            )
            
            # Extract relationships
            relationships = await self.extract_relationships(
                requirement.requirement_text,
                entities,
                context={
                    "section_id": requirement.section_id,
                    "cross_references": requirement.cross_references
                }
            )
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update stats
            self.extraction_stats["entities_extracted"] += len(entities)
            self.extraction_stats["relationships_extracted"] += len(relationships)
            
            return EntityExtractionResult(
                source_id=requirement.section_id,
                entities=entities,
                relationships=relationships,
                processing_time_seconds=processing_time,
                extraction_stats={
                    "entities_count": len(entities),
                    "relationships_count": len(relationships),
                    "unique_entity_types": len(set(e.entity_type for e in entities))
                }
            )
            
        except Exception as e:
            self.extraction_stats["processing_errors"] += 1
            logger.error(f"Error extracting from requirement {requirement.section_id}: {e}")
            raise
    
    async def extract_from_section(self, section: ParsedSection) -> EntityExtractionResult:
        """Extract entities and relationships from a section."""
        start_time = datetime.utcnow()
        
        try:
            # Extract entities
            entities = await self.extract_entities(
                section.content,
                context={
                    "section_id": section.section_id,
                    "sourcebook": section.sourcebook,
                    "chapter": section.chapter,
                    "level": section.level
                }
            )
            
            # Extract relationships
            relationships = await self.extract_relationships(
                section.content,
                entities,
                context={
                    "section_id": section.section_id,
                    "parent_section": section.parent_section,
                    "children_sections": section.children_sections
                }
            )
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update stats
            self.extraction_stats["entities_extracted"] += len(entities)
            self.extraction_stats["relationships_extracted"] += len(relationships)
            
            return EntityExtractionResult(
                source_id=section.section_id,
                entities=entities,
                relationships=relationships,
                processing_time_seconds=processing_time,
                extraction_stats={
                    "entities_count": len(entities),
                    "relationships_count": len(relationships),
                    "unique_entity_types": len(set(e.entity_type for e in entities))
                }
            )
            
        except Exception as e:
            self.extraction_stats["processing_errors"] += 1
            logger.error(f"Error extracting from section {section.section_id}: {e}")
            raise


class RegexEntityExtractor(BaseEntityExtractor):
    """Entity extractor using regex patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[RegulatoryEntityType, List[Tuple[str, float]]]:
        """Initialize regex patterns for entity extraction."""
        
        patterns = {
            RegulatoryEntityType.FIRM_TYPE: [
                (r"investment firm[s]?", 0.9),
                (r"authorised person[s]?", 0.8),
                (r"credit institution[s]?", 0.9),
                (r"bank[s]?", 0.8),
                (r"insurer[s]?", 0.9),
                (r"insurance compan(?:y|ies)", 0.9),
                (r"payment institution[s]?", 0.9),
                (r"electronic money institution[s]?", 0.9),
                (r"consumer credit firm[s]?", 0.9),
                (r"mortgage (?:lender[s]?|broker[s]?)", 0.9),
                (r"appointed representative[s]?", 0.9),
            ],
            
            RegulatoryEntityType.BUSINESS_FUNCTION: [
                (r"governance arrangement[s]?", 0.9),
                (r"risk management", 0.9),
                (r"compliance function", 0.9),
                (r"internal audit", 0.9),
                (r"client facing", 0.8),
                (r"trading activit(?:y|ies)", 0.8),
                (r"custody service[s]?", 0.8),
                (r"investment advice", 0.9),
                (r"portfolio management", 0.9),
                (r"payment service[s]?", 0.8),
                (r"lending activit(?:y|ies)", 0.8),
            ],
            
            RegulatoryEntityType.COMPLIANCE_OFFICER: [
                (r"compliance officer[s]?", 0.9),
                (r"money laundering reporting officer[s]?", 0.9),
                (r"MLRO[s]?", 0.9),
                (r"senior manager[s]?", 0.7),
                (r"approved person[s]?", 0.8),
                (r"responsible person[s]?", 0.7),
            ],
            
            RegulatoryEntityType.DEADLINE: [
                (r"within (\d+) (?:business )?days?", 0.9),
                (r"within (\d+) weeks?", 0.9),
                (r"within (\d+) months?", 0.9),
                (r"immediately", 0.9),
                (r"without delay", 0.8),
                (r"as soon as (?:reasonably )?practicable", 0.8),
                (r"before the end of", 0.7),
                (r"no later than", 0.8),
                (r"annually", 0.9),
                (r"quarterly", 0.9),
                (r"monthly", 0.9),
            ],
            
            RegulatoryEntityType.PENALTY: [
                (r"financial penalty", 0.9),
                (r"fine[s]?", 0.8),
                (r"sanction[s]?", 0.8),
                (r"enforcement action[s]?", 0.9),
                (r"prohibition order[s]?", 0.9),
                (r"suspension", 0.8),
                (r"disciplinary action[s]?", 0.8),
                (r"criminal offence[s]?", 0.9),
            ],
            
            RegulatoryEntityType.PROCEDURE: [
                (r"due diligence", 0.9),
                (r"know your customer", 0.9),
                (r"KYC", 0.9),
                (r"anti-money laundering", 0.9),
                (r"AML", 0.9),
                (r"risk assessment[s]?", 0.9),
                (r"suitability assessment[s]?", 0.9),
                (r"appropriateness test[s]?", 0.9),
                (r"best execution", 0.9),
                (r"client categorisation", 0.9),
            ],
            
            RegulatoryEntityType.DOCUMENT: [
                (r"policy document[s]?", 0.8),
                (r"procedure[s]?", 0.7),
                (r"record[s]?", 0.6),
                (r"register[s]?", 0.8),
                (r"agreement[s]?", 0.7),
                (r"contract[s]?", 0.7),
                (r"terms and conditions", 0.8),
                (r"disclosure[s]?", 0.8),
                (r"notification[s]?", 0.8),
                (r"report[s]?", 0.7),
            ],
            
            RegulatoryEntityType.SECTION: [
                (r"[A-Z]{3,5}\.?\s*\d+\.?\d*\.?\d*", 0.9),  # e.g., SYSC.4.1.1
                (r"(?:section|rule|guidance)\s+\d+", 0.8),
                (r"paragraph\s+\d+", 0.7),
                (r"chapter\s+\d+", 0.8),
            ],
        }
        
        return patterns
    
    async def extract_entities(self, text: str, context: Dict[str, Any] = None) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, pattern_list in self.patterns.items():
            for pattern, base_confidence in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    matched_text = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Normalize the entity value
                    normalized_value = self._normalize_entity_value(matched_text, entity_type)
                    
                    # Adjust confidence based on context
                    confidence = self._adjust_confidence(base_confidence, context, entity_type)
                    
                    entity = ExtractedEntity(
                        entity_type=entity_type,
                        text=matched_text,
                        normalized_value=normalized_value,
                        confidence=confidence,
                        extraction_method=ExtractionMethod.REGEX_PATTERNS,
                        source_text=text,
                        start_position=start_pos,
                        end_position=end_pos,
                        metadata={
                            "pattern": pattern,
                            "context": context or {}
                        }
                    )
                    
                    entities.append(entity)
        
        # Remove duplicate entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    async def extract_relationships(
        self, 
        text: str, 
        entities: List[ExtractedEntity], 
        context: Dict[str, Any] = None
    ) -> List[ExtractedRelationship]:
        """Extract relationships using pattern matching."""
        relationships = []
        
        # Define relationship patterns
        relationship_patterns = {
            RegulatoryRelationType.APPLIES_TO: [
                (r"applies? to", 0.9),
                (r"relevant to", 0.8),
                (r"concerns?", 0.7),
                (r"affects?", 0.7),
            ],
            RegulatoryRelationType.REQUIRES: [
                (r"must", 0.9),
                (r"shall", 0.9),
                (r"required? to", 0.9),
                (r"obliged? to", 0.8),
                (r"needs? to", 0.7),
            ],
            RegulatoryRelationType.REFERENCES: [
                (r"see", 0.8),
                (r"refer(?:s|ence)? to", 0.9),
                (r"as set out in", 0.9),
                (r"in accordance with", 0.9),
                (r"pursuant to", 0.8),
            ],
        }
        
        # Look for relationships between entities
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i == j:
                    continue
                
                # Check if entities are close enough to be related
                distance = abs(source_entity.start_position - target_entity.start_position)
                if distance > 200:  # Skip if entities are too far apart
                    continue
                
                # Extract text between entities
                start_pos = min(source_entity.end_position, target_entity.end_position)
                end_pos = max(source_entity.start_position, target_entity.start_position)
                between_text = text[start_pos:end_pos]
                
                # Check for relationship patterns
                for rel_type, pattern_list in relationship_patterns.items():
                    for pattern, confidence in pattern_list:
                        if re.search(pattern, between_text, re.IGNORECASE):
                            relationship = ExtractedRelationship(
                                relationship_type=rel_type,
                                source_entity=source_entity,
                                target_entity=target_entity,
                                confidence=confidence,
                                extraction_method=ExtractionMethod.REGEX_PATTERNS,
                                source_text=text,
                                evidence_text=between_text,
                                metadata={
                                    "pattern": pattern,
                                    "distance": distance
                                }
                            )
                            relationships.append(relationship)
                            break  # Only add one relationship per entity pair
        
        return relationships
    
    def _normalize_entity_value(self, text: str, entity_type: RegulatoryEntityType) -> str:
        """Normalize entity value for consistency."""
        normalized = text.lower().strip()
        
        # Entity-specific normalization
        if entity_type == RegulatoryEntityType.FIRM_TYPE:
            # Standardize firm type names
            firm_type_map = {
                "investment firm": "investment_firm",
                "investment firms": "investment_firm",
                "bank": "bank",
                "banks": "bank",
                "insurer": "insurer",
                "insurers": "insurer",
                "insurance company": "insurer",
                "insurance companies": "insurer",
            }
            normalized = firm_type_map.get(normalized, normalized)
        
        elif entity_type == RegulatoryEntityType.SECTION:
            # Standardize section references
            normalized = re.sub(r'\s+', '.', normalized.upper())
        
        elif entity_type == RegulatoryEntityType.DEADLINE:
            # Extract and normalize time periods
            if "day" in normalized:
                match = re.search(r'(\d+)', normalized)
                if match:
                    normalized = f"{match.group(1)}_days"
            elif "week" in normalized:
                match = re.search(r'(\d+)', normalized)
                if match:
                    normalized = f"{match.group(1)}_weeks"
            elif "month" in normalized:
                match = re.search(r'(\d+)', normalized)
                if match:
                    normalized = f"{match.group(1)}_months"
        
        return normalized
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        context: Dict[str, Any], 
        entity_type: RegulatoryEntityType
    ) -> float:
        """Adjust confidence score based on context."""
        confidence = base_confidence
        
        if not context:
            return confidence
        
        # Boost confidence if entity type matches requirement type
        if "requirement_type" in context:
            req_type = context["requirement_type"]
            
            if entity_type == RegulatoryEntityType.FIRM_TYPE and req_type == RequirementType.CONDUCT:
                confidence *= 1.1
            elif entity_type == RegulatoryEntityType.PROCEDURE and req_type == RequirementType.GOVERNANCE:
                confidence *= 1.1
            elif entity_type == RegulatoryEntityType.DEADLINE and req_type == RequirementType.REPORTING:
                confidence *= 1.1
        
        # Boost confidence if in specific sourcebooks
        if "sourcebook" in context:
            sourcebook = context["sourcebook"]
            
            if entity_type == RegulatoryEntityType.BUSINESS_FUNCTION and sourcebook == FCASourcebook.SYSC:
                confidence *= 1.1
            elif entity_type == RegulatoryEntityType.FIRM_TYPE and sourcebook == FCASourcebook.COBS:
                confidence *= 1.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on text and position overlap."""
        deduplicated = []
        
        for entity in entities:
            is_duplicate = False
            
            for existing in deduplicated:
                # Check for overlapping positions
                if (entity.start_position < existing.end_position and 
                    entity.end_position > existing.start_position):
                    
                    # Keep entity with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated


class EntityExtractionPipeline:
    """Pipeline for entity extraction from regulatory content."""
    
    def __init__(self, extractor: BaseEntityExtractor):
        self.extractor = extractor
        self.results: List[EntityExtractionResult] = []
    
    async def process_requirements(self, requirements: List[ExtractedRequirement]) -> List[EntityExtractionResult]:
        """Process multiple requirements for entity extraction."""
        results = []
        
        logger.info(f"Starting entity extraction for {len(requirements)} requirements")
        
        for requirement in requirements:
            try:
                result = await self.extractor.extract_from_requirement(requirement)
                results.append(result)
                
                logger.debug(f"Extracted {len(result.entities)} entities from {requirement.section_id}")
                
            except Exception as e:
                logger.error(f"Failed to extract entities from requirement {requirement.section_id}: {e}")
                continue
        
        self.results.extend(results)
        
        # Log summary statistics
        total_entities = sum(len(r.entities) for r in results)
        total_relationships = sum(len(r.relationships) for r in results)
        
        logger.info(f"Entity extraction completed: {total_entities} entities, {total_relationships} relationships")
        
        return results
    
    async def process_sections(self, sections: List[ParsedSection]) -> List[EntityExtractionResult]:
        """Process multiple sections for entity extraction."""
        results = []
        
        logger.info(f"Starting entity extraction for {len(sections)} sections")
        
        for section in sections:
            try:
                result = await self.extractor.extract_from_section(section)
                results.append(result)
                
                logger.debug(f"Extracted {len(result.entities)} entities from {section.section_id}")
                
            except Exception as e:
                logger.error(f"Failed to extract entities from section {section.section_id}: {e}")
                continue
        
        self.results.extend(results)
        
        # Log summary statistics
        total_entities = sum(len(r.entities) for r in results)
        total_relationships = sum(len(r.relationships) for r in results)
        
        logger.info(f"Section entity extraction completed: {total_entities} entities, {total_relationships} relationships")
        
        return results
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics."""
        if not self.results:
            return {}
        
        # Aggregate statistics
        total_entities = sum(len(r.entities) for r in self.results)
        total_relationships = sum(len(r.relationships) for r in self.results)
        total_processing_time = sum(r.processing_time_seconds for r in self.results)
        
        # Entity type distribution
        entity_type_counts = {}
        relationship_type_counts = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0, "uncertain": 0}
        
        for result in self.results:
            for entity in result.entities:
                entity_type = entity.entity_type.value
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                
                # Confidence distribution
                if entity.confidence >= 0.8:
                    confidence_distribution["high"] += 1
                elif entity.confidence >= 0.6:
                    confidence_distribution["medium"] += 1
                elif entity.confidence >= 0.4:
                    confidence_distribution["low"] += 1
                else:
                    confidence_distribution["uncertain"] += 1
            
            for relationship in result.relationships:
                rel_type = relationship.relationship_type.value
                relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        
        return {
            "total_documents_processed": len(self.results),
            "total_entities_extracted": total_entities,
            "total_relationships_extracted": total_relationships,
            "total_processing_time_seconds": total_processing_time,
            "average_entities_per_document": total_entities / len(self.results),
            "average_relationships_per_document": total_relationships / len(self.results),
            "entity_type_distribution": entity_type_counts,
            "relationship_type_distribution": relationship_type_counts,
            "confidence_distribution": confidence_distribution,
            "extractor_stats": self.extractor.extraction_stats
        }


# Factory functions
def create_regex_entity_extractor(config: Optional[Dict[str, Any]] = None) -> RegexEntityExtractor:
    """Create regex-based entity extractor."""
    if config is None:
        config = {}
    
    return RegexEntityExtractor(config)


def create_entity_extraction_pipeline(
    extraction_method: ExtractionMethod = ExtractionMethod.REGEX_PATTERNS
) -> EntityExtractionPipeline:
    """Create entity extraction pipeline with specified method."""
    
    if extraction_method == ExtractionMethod.REGEX_PATTERNS:
        extractor = create_regex_entity_extractor()
    else:
        raise NotImplementedError(f"Extraction method {extraction_method} not yet implemented")
    
    return EntityExtractionPipeline(extractor)