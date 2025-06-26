"""Knowledge base structure for FSCompliance regulatory content storage and retrieval."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import ConductRequirement, FCASourcebook
from .categorization import CategoryResult
from .parser import ExtractedRequirement, ParsedSection

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Available storage backends for knowledge base."""
    MEMORY = "memory"        # In-memory storage (development/testing)
    SQLITE = "sqlite"        # SQLite database
    POSTGRESQL = "postgresql"  # PostgreSQL database
    JSON_FILES = "json_files"  # JSON file-based storage


class QueryType(str, Enum):
    """Types of knowledge base queries."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    CATEGORY_FILTER = "category_filter"
    SECTION_LOOKUP = "section_lookup"
    CROSS_REFERENCE = "cross_reference"


class KnowledgeEntity(BaseModel):
    """Base entity in the knowledge base."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity ID")
    entity_type: str = Field(..., description="Type of entity")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RequirementEntity(KnowledgeEntity):
    """Knowledge base entity for regulatory requirements."""
    
    entity_type: str = Field(default="requirement", description="Always 'requirement'")
    section_id: str = Field(..., description="FCA section identifier")
    sourcebook: FCASourcebook = Field(..., description="Source sourcebook")
    title: str = Field(..., description="Requirement title")
    content: str = Field(..., description="Full requirement text")
    category_result: Optional[CategoryResult] = Field(None, description="Categorization result")
    conduct_requirement: Optional[ConductRequirement] = Field(None, description="Full requirement model")
    
    # Relationships
    parent_sections: List[str] = Field(default_factory=list, description="Parent section IDs")
    child_sections: List[str] = Field(default_factory=list, description="Child section IDs")
    cross_references: List[str] = Field(default_factory=list, description="Cross-referenced sections")
    related_requirements: List[str] = Field(default_factory=list, description="Related requirement IDs")
    
    # Search and retrieval fields
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    semantic_embedding: Optional[List[float]] = Field(None, description="Semantic embedding vector")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "req_123",
                "entity_type": "requirement",
                "section_id": "SYSC.4.1.1",
                "sourcebook": "sysc",
                "title": "General organisational requirements",
                "content": "A firm must have robust governance arrangements...",
                "keywords": ["governance", "arrangements", "robust"],
                "cross_references": ["SYSC.4.1.2", "SYSC.5.1.1"]
            }
        }


class SectionEntity(KnowledgeEntity):
    """Knowledge base entity for FCA Handbook sections."""
    
    entity_type: str = Field(default="section", description="Always 'section'")
    section_id: str = Field(..., description="Section identifier")
    sourcebook: FCASourcebook = Field(..., description="Source sourcebook")
    chapter: str = Field(..., description="Chapter number")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(..., description="Hierarchical level")
    
    # Hierarchical relationships
    parent_section: Optional[str] = Field(None, description="Parent section ID")
    child_sections: List[str] = Field(default_factory=list, description="Child section IDs")
    
    # Associated requirements
    requirements: List[str] = Field(default_factory=list, description="Requirement IDs in this section")


class KnowledgeGraph(BaseModel):
    """Represents the knowledge graph structure."""
    
    entities: Dict[str, KnowledgeEntity] = Field(default_factory=dict, description="All entities by ID")
    relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Entity relationships")
    indexes: Dict[str, Dict[str, Set[str]]] = Field(default_factory=dict, description="Search indexes")
    
    def add_entity(self, entity: KnowledgeEntity):
        """Add entity to knowledge graph."""
        self.entities[entity.id] = entity
        
        # Update indexes
        self._update_indexes(entity)
    
    def _update_indexes(self, entity: KnowledgeEntity):
        """Update search indexes for entity."""
        
        # Type index
        if "type" not in self.indexes:
            self.indexes["type"] = {}
        if entity.entity_type not in self.indexes["type"]:
            self.indexes["type"][entity.entity_type] = set()
        self.indexes["type"][entity.entity_type].add(entity.id)
        
        # Sourcebook index (for requirements and sections)
        if hasattr(entity, 'sourcebook'):
            if "sourcebook" not in self.indexes:
                self.indexes["sourcebook"] = {}
            sourcebook = entity.sourcebook.value
            if sourcebook not in self.indexes["sourcebook"]:
                self.indexes["sourcebook"][sourcebook] = set()
            self.indexes["sourcebook"][sourcebook].add(entity.id)
        
        # Keywords index (for requirements)
        if hasattr(entity, 'keywords') and entity.keywords:
            if "keywords" not in self.indexes:
                self.indexes["keywords"] = {}
            for keyword in entity.keywords:
                if keyword not in self.indexes["keywords"]:
                    self.indexes["keywords"][keyword] = set()
                self.indexes["keywords"][keyword].add(entity.id)


@dataclass
class SearchQuery:
    """Query for knowledge base search."""
    
    query_text: str
    query_type: QueryType = QueryType.SEMANTIC_SEARCH
    filters: Dict[str, Any] = None
    limit: int = 10
    include_metadata: bool = True
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class SearchResult(BaseModel):
    """Result from knowledge base search."""
    
    entity_id: str = Field(..., description="Entity ID")
    entity: KnowledgeEntity = Field(..., description="The entity")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_reason: str = Field(..., description="Reason for match")
    highlighted_text: Optional[str] = Field(None, description="Highlighted matching text")


class BaseKnowledgeBase(ABC):
    """Abstract base class for knowledge base implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_graph = KnowledgeGraph()
    
    @abstractmethod
    async def store_requirement(self, requirement: RequirementEntity) -> str:
        """Store a requirement entity."""
        pass
    
    @abstractmethod
    async def store_section(self, section: SectionEntity) -> str:
        """Store a section entity."""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search the knowledge base."""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Retrieve entity by ID."""
        pass
    
    @abstractmethod
    async def get_related_entities(self, entity_id: str, relationship_type: str) -> List[KnowledgeEntity]:
        """Get entities related to given entity."""
        pass
    
    async def bulk_store_requirements(self, requirements: List[RequirementEntity]) -> List[str]:
        """Store multiple requirements."""
        entity_ids = []
        for requirement in requirements:
            entity_id = await self.store_requirement(requirement)
            entity_ids.append(entity_id)
        return entity_ids
    
    async def bulk_store_sections(self, sections: List[SectionEntity]) -> List[str]:
        """Store multiple sections."""
        entity_ids = []
        for section in sections:
            entity_id = await self.store_section(section)
            entity_ids.append(entity_id)
        return entity_ids


class MemoryKnowledgeBase(BaseKnowledgeBase):
    """In-memory knowledge base implementation for development/testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("Initialized in-memory knowledge base")
    
    async def store_requirement(self, requirement: RequirementEntity) -> str:
        """Store requirement in memory."""
        self.knowledge_graph.add_entity(requirement)
        logger.debug(f"Stored requirement: {requirement.section_id}")
        return requirement.id
    
    async def store_section(self, section: SectionEntity) -> str:
        """Store section in memory."""
        self.knowledge_graph.add_entity(section)
        logger.debug(f"Stored section: {section.section_id}")
        return section.id
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search knowledge base using simple keyword matching."""
        results = []
        query_terms = set(query.query_text.lower().split())
        
        for entity_id, entity in self.knowledge_graph.entities.items():
            
            # Apply filters
            if not self._matches_filters(entity, query.filters):
                continue
            
            # Calculate relevance score
            score, reason = self._calculate_relevance(entity, query_terms, query.query_type)
            
            if score > 0:
                result = SearchResult(
                    entity_id=entity_id,
                    entity=entity,
                    relevance_score=score,
                    match_reason=reason
                )
                results.append(result)
        
        # Sort by relevance score and apply limit
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:query.limit]
    
    async def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get entity by ID."""
        return self.knowledge_graph.entities.get(entity_id)
    
    async def get_related_entities(self, entity_id: str, relationship_type: str) -> List[KnowledgeEntity]:
        """Get related entities."""
        entity = await self.get_entity(entity_id)
        if not entity:
            return []
        
        related_ids = []
        
        # Get related IDs based on relationship type
        if relationship_type == "cross_references" and hasattr(entity, 'cross_references'):
            related_ids.extend(entity.cross_references)
        elif relationship_type == "children" and hasattr(entity, 'child_sections'):
            related_ids.extend(entity.child_sections)
        elif relationship_type == "parent" and hasattr(entity, 'parent_section') and entity.parent_section:
            related_ids.append(entity.parent_section)
        
        # Return related entities
        related_entities = []
        for related_id in related_ids:
            related_entity = await self.get_entity(related_id)
            if related_entity:
                related_entities.append(related_entity)
        
        return related_entities
    
    def _matches_filters(self, entity: KnowledgeEntity, filters: Dict[str, Any]) -> bool:
        """Check if entity matches search filters."""
        if not filters:
            return True
        
        for filter_key, filter_value in filters.items():
            if filter_key == "entity_type":
                if entity.entity_type != filter_value:
                    return False
            elif filter_key == "sourcebook" and hasattr(entity, 'sourcebook'):
                if entity.sourcebook.value != filter_value:
                    return False
            elif filter_key == "category" and hasattr(entity, 'category_result'):
                if entity.category_result and entity.category_result.primary_category.value != filter_value:
                    return False
        
        return True
    
    def _calculate_relevance(self, entity: KnowledgeEntity, query_terms: Set[str], 
                           query_type: QueryType) -> tuple[float, str]:
        """Calculate relevance score for entity."""
        
        if query_type == QueryType.EXACT_MATCH:
            # Check for exact matches in content
            if hasattr(entity, 'content'):
                content_lower = entity.content.lower()
                if all(term in content_lower for term in query_terms):
                    return 1.0, "Exact match in content"
            return 0.0, "No exact match"
        
        elif query_type == QueryType.KEYWORD_SEARCH:
            # Keyword-based scoring
            score = 0.0
            reasons = []
            
            # Check title
            if hasattr(entity, 'title'):
                title_lower = entity.title.lower()
                title_matches = len(query_terms.intersection(set(title_lower.split())))
                if title_matches > 0:
                    score += 0.4 * (title_matches / len(query_terms))
                    reasons.append(f"{title_matches} keyword(s) in title")
            
            # Check content
            if hasattr(entity, 'content'):
                content_lower = entity.content.lower()
                content_words = set(content_lower.split())
                content_matches = len(query_terms.intersection(content_words))
                if content_matches > 0:
                    score += 0.3 * (content_matches / len(query_terms))
                    reasons.append(f"{content_matches} keyword(s) in content")
            
            # Check keywords
            if hasattr(entity, 'keywords') and entity.keywords:
                keyword_lower = {kw.lower() for kw in entity.keywords}
                keyword_matches = len(query_terms.intersection(keyword_lower))
                if keyword_matches > 0:
                    score += 0.3 * (keyword_matches / len(query_terms))
                    reasons.append(f"{keyword_matches} keyword(s) matched")
            
            return min(score, 1.0), "; ".join(reasons) if reasons else "No keyword matches"
        
        else:
            # Default semantic search (simplified)
            return self._calculate_relevance(entity, query_terms, QueryType.KEYWORD_SEARCH)


class JSONFileKnowledgeBase(BaseKnowledgeBase):
    """File-based knowledge base using JSON storage."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_path = Path(config.get("storage_path", "data/knowledge_base"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized JSON file knowledge base at {self.storage_path}")
    
    async def store_requirement(self, requirement: RequirementEntity) -> str:
        """Store requirement to JSON file."""
        file_path = self.storage_path / f"requirement_{requirement.id}.json"
        
        with open(file_path, 'w') as f:
            f.write(requirement.json(indent=2))
        
        # Also update in-memory graph
        self.knowledge_graph.add_entity(requirement)
        
        logger.debug(f"Stored requirement to {file_path}")
        return requirement.id
    
    async def store_section(self, section: SectionEntity) -> str:
        """Store section to JSON file."""
        file_path = self.storage_path / f"section_{section.id}.json"
        
        with open(file_path, 'w') as f:
            f.write(section.json(indent=2))
        
        # Also update in-memory graph
        self.knowledge_graph.add_entity(section)
        
        logger.debug(f"Stored section to {file_path}")
        return section.id
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using in-memory index with file loading."""
        # Load all entities into memory if not already loaded
        await self._ensure_entities_loaded()
        
        # Use memory-based search
        memory_kb = MemoryKnowledgeBase(self.config)
        memory_kb.knowledge_graph = self.knowledge_graph
        return await memory_kb.search(query)
    
    async def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get entity from file or memory."""
        # Check memory first
        if entity_id in self.knowledge_graph.entities:
            return self.knowledge_graph.entities[entity_id]
        
        # Try to load from file
        for file_pattern in ["requirement_*.json", "section_*.json"]:
            for file_path in self.storage_path.glob(file_pattern):
                if entity_id in file_path.name:
                    try:
                        with open(file_path, 'r') as f:
                            data = f.read()
                        
                        # Determine entity type and deserialize
                        if "requirement_" in file_path.name:
                            entity = RequirementEntity.parse_raw(data)
                        else:
                            entity = SectionEntity.parse_raw(data)
                        
                        # Add to memory
                        self.knowledge_graph.add_entity(entity)
                        return entity
                        
                    except Exception as e:
                        logger.error(f"Error loading entity from {file_path}: {e}")
        
        return None
    
    async def get_related_entities(self, entity_id: str, relationship_type: str) -> List[KnowledgeEntity]:
        """Get related entities."""
        # Ensure entities are loaded
        await self._ensure_entities_loaded()
        
        # Use memory-based implementation
        memory_kb = MemoryKnowledgeBase(self.config)
        memory_kb.knowledge_graph = self.knowledge_graph
        return await memory_kb.get_related_entities(entity_id, relationship_type)
    
    async def _ensure_entities_loaded(self):
        """Ensure all entities are loaded into memory."""
        # Load all JSON files
        for json_file in self.storage_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = f.read()
                
                # Determine entity type and load
                if "requirement_" in json_file.name:
                    entity = RequirementEntity.parse_raw(data)
                elif "section_" in json_file.name:
                    entity = SectionEntity.parse_raw(data)
                else:
                    continue
                
                # Add to memory if not already present
                if entity.id not in self.knowledge_graph.entities:
                    self.knowledge_graph.add_entity(entity)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")


class KnowledgeBaseFactory:
    """Factory for creating knowledge base instances."""
    
    @staticmethod
    def create_knowledge_base(backend: StorageBackend, config: Dict[str, Any]) -> BaseKnowledgeBase:
        """Create knowledge base instance based on backend type."""
        
        if backend == StorageBackend.MEMORY:
            return MemoryKnowledgeBase(config)
        elif backend == StorageBackend.JSON_FILES:
            return JSONFileKnowledgeBase(config)
        elif backend == StorageBackend.SQLITE:
            # TODO: Implement SQLite backend
            raise NotImplementedError("SQLite backend not yet implemented")
        elif backend == StorageBackend.POSTGRESQL:
            # TODO: Implement PostgreSQL backend
            raise NotImplementedError("PostgreSQL backend not yet implemented")
        else:
            raise ValueError(f"Unsupported storage backend: {backend}")


# Utility functions for knowledge base operations
async def build_knowledge_base_from_parsed_content(
    sections: List[ParsedSection],
    requirements: List[ExtractedRequirement],
    category_results: List[CategoryResult],
    storage_backend: StorageBackend = StorageBackend.MEMORY
) -> BaseKnowledgeBase:
    """Build knowledge base from parsed content."""
    
    # Create knowledge base
    config = {"storage_path": "data/knowledge_base"} if storage_backend == StorageBackend.JSON_FILES else {}
    kb = KnowledgeBaseFactory.create_knowledge_base(storage_backend, config)
    
    # Create section entities
    section_entities = []
    for section in sections:
        section_entity = SectionEntity(
            section_id=section.section_id,
            sourcebook=section.sourcebook,
            chapter=section.chapter,
            title=section.title,
            content=section.content,
            level=section.level,
            parent_section=section.parent_section,
            child_sections=section.children_sections
        )
        section_entities.append(section_entity)
    
    # Create requirement entities
    requirement_entities = []
    category_map = {cr.requirement_id: cr for cr in category_results}
    
    for requirement in requirements:
        # Find matching category result
        category_result = category_map.get(requirement.section_id)
        
        requirement_entity = RequirementEntity(
            section_id=requirement.section_id,
            sourcebook=requirement.section_id.split('.')[0].lower(),  # Extract from section ID
            title=f"Requirement {requirement.section_id}",
            content=requirement.requirement_text,
            category_result=category_result,
            cross_references=requirement.cross_references,
            keywords=requirement.keywords
        )
        requirement_entities.append(requirement_entity)
    
    # Store entities in knowledge base
    await kb.bulk_store_sections(section_entities)
    await kb.bulk_store_requirements(requirement_entities)
    
    logger.info(f"Built knowledge base with {len(section_entities)} sections and {len(requirement_entities)} requirements")
    
    return kb


async def create_default_knowledge_base() -> BaseKnowledgeBase:
    """Create default knowledge base for development."""
    config = {"storage_path": "data/knowledge_base"}
    return KnowledgeBaseFactory.create_knowledge_base(StorageBackend.JSON_FILES, config)