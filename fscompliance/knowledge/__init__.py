"""Knowledge Management Layer - FCA Handbook processing and knowledge base."""

from .categorization import (
    ApplicabilityScope,
    BusinessFunction,
    CategoryResult,
    CategorizationPipeline,
    CategoryConfidence,
    CategoryRules,
    KeywordBasedCategorizer,
    RegulatoryImpact,
    create_default_categorization_pipeline,
)
from .ingestion import (
    DocumentMetadata,
    FCARegisterAPIAdapter,
    IngestionConfig,
    IngestionPipeline,
    IngestionStatus,
    IngestionStrategy,
    PDFDownloadAdapter,
    WebScrapingAdapter,
    create_default_pipeline,
    validate_ingestion_prerequisites,
)
from .knowledge_base import (
    BaseKnowledgeBase,
    JSONFileKnowledgeBase,
    KnowledgeBaseFactory,
    KnowledgeEntity,
    KnowledgeGraph,
    MemoryKnowledgeBase,
    QueryType,
    RequirementEntity,
    SearchQuery,
    SearchResult,
    SectionEntity,
    StorageBackend,
    build_knowledge_base_from_parsed_content,
    create_default_knowledge_base,
)
from .parser import (
    DocumentParsingPipeline,
    ExtractedRequirement,
    HTMLParser,
    PDFParser,
    ParsedSection,
    ParsingStats,
    convert_to_conduct_requirements,
)
from .lightrag_integration import (
    FSComplianceLightRAG,
    KnowledgeGraphSchema,
    LightRAGConfig,
    RegulatoryEntityType,
    RegulatoryRelationType,
    build_regulatory_knowledge_graph,
    create_fscompliance_lightrag,
)
from .entity_extraction import (
    EntityExtractionPipeline,
    EntityExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    RegexEntityExtractor,
    create_entity_extraction_pipeline,
    create_regex_entity_extractor,
)
from .pipeline import (
    DocumentProcessingPipeline,
    PipelineConfig,
    ProcessingStage,
    StageResult,
    create_default_pipeline,
    run_fca_handbook_processing,
)
from .retrieval import (
    DualLevelRetrievalSystem,
    RetrievalContext,
    RetrievalLevel,
    RetrievalMode,
    RetrievalResult,
    StandardDualLevelRetriever,
    create_dual_level_retrieval_system,
)

__all__ = [
    # Ingestion
    "DocumentMetadata",
    "FCARegisterAPIAdapter", 
    "IngestionConfig",
    "IngestionPipeline",
    "IngestionStatus",
    "IngestionStrategy",
    "PDFDownloadAdapter",
    "WebScrapingAdapter",
    "create_default_pipeline",
    "validate_ingestion_prerequisites",
    # Parsing
    "DocumentParsingPipeline",
    "ExtractedRequirement",
    "HTMLParser", 
    "PDFParser",
    "ParsedSection",
    "ParsingStats",
    "convert_to_conduct_requirements",
    # Categorization
    "ApplicabilityScope",
    "BusinessFunction",
    "CategoryResult",
    "CategorizationPipeline",
    "CategoryConfidence", 
    "CategoryRules",
    "KeywordBasedCategorizer",
    "RegulatoryImpact",
    "create_default_categorization_pipeline",
    # Knowledge Base
    "BaseKnowledgeBase",
    "JSONFileKnowledgeBase",
    "KnowledgeBaseFactory",
    "KnowledgeEntity",
    "KnowledgeGraph",
    "MemoryKnowledgeBase",
    "QueryType",
    "RequirementEntity",
    "SearchQuery", 
    "SearchResult",
    "SectionEntity",
    "StorageBackend",
    "build_knowledge_base_from_parsed_content",
    "create_default_knowledge_base",
    # LightRAG Integration
    "FSComplianceLightRAG",
    "KnowledgeGraphSchema",
    "LightRAGConfig",
    "RegulatoryEntityType",
    "RegulatoryRelationType",
    "build_regulatory_knowledge_graph",
    "create_fscompliance_lightrag",
    # Entity Extraction
    "EntityExtractionPipeline",
    "EntityExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "RegexEntityExtractor",
    "create_entity_extraction_pipeline",
    "create_regex_entity_extractor",
    # Processing Pipeline
    "DocumentProcessingPipeline",
    "PipelineConfig",
    "ProcessingStage",
    "StageResult",
    "run_fca_handbook_processing",
    # Dual-Level Retrieval
    "DualLevelRetrievalSystem",
    "RetrievalContext",
    "RetrievalLevel",
    "RetrievalMode",
    "RetrievalResult",
    "StandardDualLevelRetriever",
    "create_dual_level_retrieval_system",
]