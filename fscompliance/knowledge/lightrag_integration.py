"""LightRAG integration for FSCompliance knowledge graph and retrieval."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ..models import ConductRequirement, FCASourcebook
from .parser import ExtractedRequirement, ParsedSection

logger = logging.getLogger(__name__)

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    logger.warning("LightRAG not available. Install with: pip install lightrag-hku[api]")
    LIGHTRAG_AVAILABLE = False


class RegulatoryEntityType(str, Enum):
    """Types of entities in regulatory knowledge graph."""
    REQUIREMENT = "requirement"
    FIRM_TYPE = "firm_type"
    BUSINESS_FUNCTION = "business_function"
    COMPLIANCE_OFFICER = "compliance_officer"
    REGULATORY_BODY = "regulatory_body"
    SOURCEBOOK = "sourcebook"
    CHAPTER = "chapter"
    SECTION = "section"
    DEADLINE = "deadline"
    PENALTY = "penalty"
    PROCEDURE = "procedure"
    DOCUMENT = "document"


class RegulatoryRelationType(str, Enum):
    """Types of relationships in regulatory knowledge graph."""
    APPLIES_TO = "applies_to"
    REQUIRES = "requires"
    REFERENCES = "references"
    SUPERSEDES = "supersedes"
    IMPLEMENTS = "implements"
    ENFORCED_BY = "enforced_by"
    CONTAINS = "contains"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    DEADLINE_FOR = "deadline_for"
    PENALTY_FOR = "penalty_for"


class KnowledgeGraphSchema(BaseModel):
    """Schema definition for regulatory knowledge graph."""
    
    entity_types: List[RegulatoryEntityType] = Field(
        default_factory=lambda: list(RegulatoryEntityType),
        description="Supported entity types"
    )
    relation_types: List[RegulatoryRelationType] = Field(
        default_factory=lambda: list(RegulatoryRelationType),
        description="Supported relationship types"
    )
    
    # Entity extraction patterns
    entity_patterns: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Regex patterns for entity extraction"
    )
    
    # Relationship extraction rules
    relationship_rules: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Rules for relationship extraction"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default extraction patterns."""
        
        self.entity_patterns = {
            RegulatoryEntityType.FIRM_TYPE: [
                r"investment firm[s]?",
                r"bank[s]?",
                r"insurer[s]?",
                r"payment institution[s]?",
                r"credit firm[s]?",
                r"mortgage broker[s]?",
            ],
            RegulatoryEntityType.BUSINESS_FUNCTION: [
                r"governance",
                r"risk management",
                r"compliance",
                r"trading",
                r"custody",
                r"client facing",
                r"reporting",
            ],
            RegulatoryEntityType.DEADLINE: [
                r"\d{1,2}\s+(?:days?|weeks?|months?|years?)",
                r"immediately",
                r"without delay",
                r"as soon as possible",
                r"quarterly",
                r"annually",
            ],
            RegulatoryEntityType.PENALTY: [
                r"fine[s]?",
                r"penalty",
                r"sanction[s]?",
                r"enforcement action",
                r"prohibition",
                r"suspension",
            ],
        }
        
        self.relationship_rules = {
            RegulatoryRelationType.APPLIES_TO: {
                "patterns": [
                    r"applies to",
                    r"relevant to",
                    r"concerns",
                    r"affects",
                ],
                "source_types": [RegulatoryEntityType.REQUIREMENT],
                "target_types": [RegulatoryEntityType.FIRM_TYPE, RegulatoryEntityType.BUSINESS_FUNCTION],
            },
            RegulatoryRelationType.REQUIRES: {
                "patterns": [
                    r"must",
                    r"shall",
                    r"required to",
                    r"obliged to",
                ],
                "source_types": [RegulatoryEntityType.REQUIREMENT],
                "target_types": [RegulatoryEntityType.PROCEDURE, RegulatoryEntityType.DOCUMENT],
            },
            RegulatoryRelationType.REFERENCES: {
                "patterns": [
                    r"see",
                    r"refer to",
                    r"as set out in",
                    r"in accordance with",
                ],
                "source_types": [RegulatoryEntityType.REQUIREMENT],
                "target_types": [RegulatoryEntityType.SECTION, RegulatoryEntityType.DOCUMENT],
            },
        }


class LightRAGConfig(BaseModel):
    """Configuration for LightRAG integration."""
    
    working_dir: Path = Field(default=Path("data/lightrag"), description="Working directory for LightRAG")
    
    # LLM Configuration
    llm_model_name: str = Field(default="gpt-4", description="LLM model for entity/relationship extraction")
    llm_api_key: Optional[str] = Field(None, description="API key for LLM")
    llm_base_url: Optional[str] = Field(None, description="Base URL for LLM API")
    
    # Embedding Configuration
    embedding_model_name: str = Field(default="text-embedding-ada-002", description="Embedding model")
    embedding_dim: int = Field(default=1536, description="Embedding dimension")
    max_token_size: int = Field(default=8192, description="Maximum token size")
    
    # Storage Configuration
    vector_storage: str = Field(default="NanoVectorDBStorage", description="Vector storage backend")
    kv_storage: str = Field(default="JsonKVStorage", description="Key-value storage backend")
    graph_storage: str = Field(default="NetworkXStorage", description="Graph storage backend")
    
    # Processing Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size for processing")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    class Config:
        arbitrary_types_allowed = True


class FSComplianceLightRAG:
    """FSCompliance-specific LightRAG integration."""
    
    def __init__(self, config: LightRAGConfig, schema: Optional[KnowledgeGraphSchema] = None):
        self.config = config
        self.schema = schema or KnowledgeGraphSchema()
        self.rag: Optional[LightRAG] = None
        self._initialized = False
        
        # Ensure working directory exists
        self.config.working_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize LightRAG instance."""
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError("LightRAG not available. Install with: pip install lightrag-hku[api]")
        
        try:
            # Initialize LightRAG with configuration
            self.rag = LightRAG(
                working_dir=str(self.config.working_dir),
                llm_model_func=self._create_llm_model_func(),
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.config.embedding_dim,
                    max_token_size=self.config.max_token_size,
                    func=self._create_embedding_func(),
                ),
                # Storage configuration would go here based on config
            )
            
            # Initialize storages
            await self.rag.initialize_storages()
            
            # Initialize pipeline status
            from lightrag.utils import initialize_pipeline_status
            await initialize_pipeline_status()
            
            self._initialized = True
            logger.info("LightRAG initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            raise
    
    def _create_llm_model_func(self):
        """Create LLM model function for LightRAG."""
        async def llm_model_func(
            prompt: str, 
            system_prompt: str = None, 
            history_messages: List = None, 
            **kwargs
        ) -> str:
            """Custom LLM model function for regulatory content."""
            
            # For now, return a placeholder response
            # In production, this would call your LLM provider
            logger.warning("Using placeholder LLM function - implement actual LLM integration")
            
            # Simulate entity/relationship extraction for regulatory content
            if "extract entities" in prompt.lower():
                return """
                Entities:
                - REQUIREMENT: General organisational requirements
                - FIRM_TYPE: Investment firms, banks
                - BUSINESS_FUNCTION: Governance, risk management
                
                Relationships:
                - REQUIREMENT applies_to FIRM_TYPE
                - REQUIREMENT requires BUSINESS_FUNCTION
                """
            
            return "Regulatory compliance analysis completed."
        
        return llm_model_func
    
    def _create_embedding_func(self):
        """Create embedding function for LightRAG."""
        async def embedding_func(texts: List[str]) -> List[List[float]]:
            """Custom embedding function for regulatory content."""
            
            # For now, return random embeddings
            # In production, this would call your embedding provider
            logger.warning("Using placeholder embedding function - implement actual embedding integration")
            
            import numpy as np
            embeddings = []
            for text in texts:
                # Generate random embedding of specified dimension
                embedding = np.random.rand(self.config.embedding_dim).tolist()
                embeddings.append(embedding)
            
            return embeddings
        
        return embedding_func
    
    async def process_requirements(self, requirements: List[ExtractedRequirement]) -> Dict[str, Any]:
        """Process extracted requirements through LightRAG."""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"Processing {len(requirements)} requirements through LightRAG")
        
        results = {
            "processed_count": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        for requirement in requirements:
            try:
                # Prepare text for LightRAG processing
                text = self._prepare_requirement_text(requirement)
                
                # Insert text into LightRAG
                await self.rag.ainsert(text)
                
                results["processed_count"] += 1
                
            except Exception as e:
                error_msg = f"Error processing requirement {requirement.section_id}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        logger.info(f"LightRAG processing complete: {results}")
        return results
    
    async def process_sections(self, sections: List[ParsedSection]) -> Dict[str, Any]:
        """Process parsed sections through LightRAG."""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"Processing {len(sections)} sections through LightRAG")
        
        results = {
            "processed_count": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        for section in sections:
            try:
                # Prepare text for LightRAG processing
                text = self._prepare_section_text(section)
                
                # Insert text into LightRAG
                await self.rag.ainsert(text)
                
                results["processed_count"] += 1
                
            except Exception as e:
                error_msg = f"Error processing section {section.section_id}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        logger.info(f"LightRAG section processing complete: {results}")
        return results
    
    def _prepare_requirement_text(self, requirement: ExtractedRequirement) -> str:
        """Prepare requirement text for LightRAG processing."""
        
        # Create structured text that helps LightRAG extract entities and relationships
        text_parts = [
            f"Section: {requirement.section_id}",
            f"Requirement Type: {requirement.requirement_type.value}",
            f"Severity: {requirement.severity.value}",
            f"Content: {requirement.requirement_text}",
        ]
        
        if requirement.cross_references:
            text_parts.append(f"Cross References: {', '.join(requirement.cross_references)}")
        
        if requirement.keywords:
            text_parts.append(f"Keywords: {', '.join(requirement.keywords)}")
        
        return "\n".join(text_parts)
    
    def _prepare_section_text(self, section: ParsedSection) -> str:
        """Prepare section text for LightRAG processing."""
        
        text_parts = [
            f"Section ID: {section.section_id}",
            f"Sourcebook: {section.sourcebook.value.upper()}",
            f"Chapter: {section.chapter}",
            f"Title: {section.title}",
            f"Level: {section.level}",
            f"Content: {section.content}",
        ]
        
        if section.parent_section:
            text_parts.append(f"Parent Section: {section.parent_section}")
        
        if section.children_sections:
            text_parts.append(f"Child Sections: {', '.join(section.children_sections)}")
        
        return "\n".join(text_parts)
    
    async def query_knowledge_graph(
        self, 
        query: str, 
        mode: str = "hybrid",
        only_need_context: bool = False
    ) -> str:
        """Query the knowledge graph using LightRAG."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create query parameters
            query_param = QueryParam(
                mode=mode,
                only_need_context=only_need_context
            )
            
            # Execute query
            result = await self.rag.aquery(query, param=query_param)
            
            logger.info(f"Knowledge graph query completed: {query[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            raise
    
    async def get_entities_and_relationships(self) -> Dict[str, Any]:
        """Extract entities and relationships from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Access the graph storage to get entities and relationships
            graph_storage = self.rag.graph_storage
            
            # Get all entities and relationships
            # This would depend on the specific storage implementation
            entities = []
            relationships = []
            
            # Placeholder implementation
            # In practice, you'd access the actual graph data structures
            
            return {
                "entities": entities,
                "relationships": relationships,
                "entity_count": len(entities),
                "relationship_count": len(relationships)
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            raise
    
    async def export_knowledge_graph(self, output_path: Path, format: str = "json") -> bool:
        """Export knowledge graph to file."""
        if not self._initialized:
            await self.initialize()
        
        try:
            graph_data = await self.get_entities_and_relationships()
            
            if format.lower() == "json":
                import json
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2, default=str)
            
            elif format.lower() == "graphml":
                # Export to GraphML format for visualization
                # This would require networkx or similar library
                logger.warning("GraphML export not yet implemented")
                return False
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Knowledge graph exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return False


# Factory functions
async def create_fscompliance_lightrag(
    working_dir: Optional[Path] = None,
    llm_model: str = "gpt-4",
    embedding_model: str = "text-embedding-ada-002"
) -> FSComplianceLightRAG:
    """Create and initialize FSCompliance LightRAG instance."""
    
    config = LightRAGConfig(
        working_dir=working_dir or Path("data/lightrag"),
        llm_model_name=llm_model,
        embedding_model_name=embedding_model
    )
    
    schema = KnowledgeGraphSchema()
    
    lightrag_instance = FSComplianceLightRAG(config, schema)
    await lightrag_instance.initialize()
    
    return lightrag_instance


async def build_regulatory_knowledge_graph(
    requirements: List[ExtractedRequirement],
    sections: List[ParsedSection],
    working_dir: Optional[Path] = None
) -> FSComplianceLightRAG:
    """Build complete regulatory knowledge graph from parsed content."""
    
    # Create LightRAG instance
    lightrag_instance = await create_fscompliance_lightrag(working_dir)
    
    # Process sections first (provides context)
    await lightrag_instance.process_sections(sections)
    
    # Process requirements (provides specific rules)
    await lightrag_instance.process_requirements(requirements)
    
    logger.info("Regulatory knowledge graph built successfully")
    
    return lightrag_instance