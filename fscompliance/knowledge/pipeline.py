"""Document processing pipeline integrating ingestion, parsing, categorization, and LightRAG."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..models import ConductRequirement, FCASourcebook
from .categorization import CategoryResult, CategorizationPipeline
from .ingestion import IngestionConfig, IngestionPipeline, IngestionStrategy
from .knowledge_base import (
    BaseKnowledgeBase,
    KnowledgeBaseFactory,
    RequirementEntity,
    SectionEntity,
    StorageBackend,
)
from .lightrag_integration import FSComplianceLightRAG, LightRAGConfig
from .parser import DocumentParsingPipeline, ExtractedRequirement, ParsedSection

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Stages in the document processing pipeline."""
    INGESTION = "ingestion"
    PARSING = "parsing"
    CATEGORIZATION = "categorization"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    STORAGE = "storage"
    INDEXING = "indexing"


class ProcessingStatus(str, Enum):
    """Status of processing stages."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a processing stage."""
    stage: ProcessingStage
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    items_processed: int = 0
    items_failed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True):
        """Mark stage as completed."""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
    
    def add_error(self, error: str):
        """Add error to stage result."""
        self.errors.append(error)
        self.items_failed += 1


class PipelineConfig(BaseModel):
    """Configuration for the complete processing pipeline."""
    
    # Ingestion configuration
    ingestion_strategy: IngestionStrategy = Field(
        default=IngestionStrategy.PDF_DOWNLOAD,
        description="Strategy for document ingestion"
    )
    ingestion_config: IngestionConfig = Field(
        default_factory=lambda: IngestionConfig(strategy=IngestionStrategy.PDF_DOWNLOAD),
        description="Detailed ingestion configuration"
    )
    
    # Storage configuration
    storage_backend: StorageBackend = Field(
        default=StorageBackend.JSON_FILES,
        description="Backend for knowledge base storage"
    )
    storage_config: Dict[str, Any] = Field(
        default_factory=lambda: {"storage_path": "data/knowledge_base"},
        description="Storage backend configuration"
    )
    
    # LightRAG configuration
    enable_lightrag: bool = Field(default=True, description="Enable LightRAG processing")
    lightrag_config: LightRAGConfig = Field(
        default_factory=LightRAGConfig,
        description="LightRAG configuration"
    )
    
    # Processing options
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    batch_size: int = Field(default=10, description="Batch size for processing")
    continue_on_errors: bool = Field(default=True, description="Continue processing if errors occur")
    
    # Output configuration
    output_directory: Path = Field(
        default=Path("data/pipeline_output"),
        description="Directory for pipeline output"
    )
    save_intermediate_results: bool = Field(
        default=True,
        description="Save intermediate results from each stage"
    )
    
    class Config:
        arbitrary_types_allowed = True


class DocumentProcessingPipeline:
    """Complete document processing pipeline for FSCompliance."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage_results: Dict[ProcessingStage, StageResult] = {}
        
        # Initialize components
        self.ingestion_pipeline: Optional[IngestionPipeline] = None
        self.parsing_pipeline: Optional[DocumentParsingPipeline] = None
        self.categorization_pipeline: Optional[CategorizationPipeline] = None
        self.knowledge_base: Optional[BaseKnowledgeBase] = None
        self.lightrag: Optional[FSComplianceLightRAG] = None
        
        # Create output directory
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
    
    async def run_complete_pipeline(
        self, 
        sourcebooks: List[FCASourcebook],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete document processing pipeline."""
        
        logger.info("Starting complete FSCompliance document processing pipeline")
        
        pipeline_start = datetime.utcnow()
        
        try:
            # Stage 1: Document Ingestion
            ingested_files = await self._run_ingestion_stage(api_key)
            
            # Stage 2: Document Parsing
            sections, requirements = await self._run_parsing_stage(ingested_files, sourcebooks)
            
            # Stage 3: Requirement Categorization
            category_results = await self._run_categorization_stage(requirements)
            
            # Stage 4: Knowledge Graph Construction (LightRAG)
            if self.config.enable_lightrag:
                await self._run_knowledge_graph_stage(sections, requirements)
            
            # Stage 5: Knowledge Base Storage
            await self._run_storage_stage(sections, requirements, category_results)
            
            # Stage 6: Indexing and Finalization
            await self._run_indexing_stage()
            
            pipeline_end = datetime.utcnow()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
            
            # Generate final report
            report = self._generate_pipeline_report(pipeline_duration)
            
            # Save final report
            await self._save_pipeline_report(report)
            
            logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f} seconds")
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            # Generate failure report
            pipeline_end = datetime.utcnow()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
            report = self._generate_pipeline_report(pipeline_duration, failed=True, error=str(e))
            
            await self._save_pipeline_report(report)
            
            raise
    
    async def _run_ingestion_stage(self, api_key: Optional[str] = None) -> List[Path]:
        """Run document ingestion stage."""
        
        stage_result = StageResult(
            stage=ProcessingStage.INGESTION,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.stage_results[ProcessingStage.INGESTION] = stage_result
        
        try:
            logger.info("Starting document ingestion stage")
            
            # Initialize ingestion pipeline
            self.ingestion_pipeline = IngestionPipeline(self.config.ingestion_config)
            
            # Run ingestion
            ingested_files = await self.ingestion_pipeline.run_ingestion(api_key)
            
            stage_result.items_processed = len(ingested_files)
            stage_result.metadata = {
                "strategy": self.config.ingestion_strategy.value,
                "files": [str(f) for f in ingested_files]
            }
            
            stage_result.complete(success=True)
            
            logger.info(f"Ingestion completed: {len(ingested_files)} files processed")
            
            return ingested_files
            
        except Exception as e:
            error_msg = f"Ingestion stage failed: {e}"
            stage_result.add_error(error_msg)
            stage_result.complete(success=False)
            
            if not self.config.continue_on_errors:
                raise
            
            logger.error(error_msg)
            return []
    
    async def _run_parsing_stage(
        self, 
        files: List[Path], 
        sourcebooks: List[FCASourcebook]
    ) -> Tuple[List[ParsedSection], List[ExtractedRequirement]]:
        """Run document parsing stage."""
        
        stage_result = StageResult(
            stage=ProcessingStage.PARSING,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.stage_results[ProcessingStage.PARSING] = stage_result
        
        try:
            logger.info(f"Starting document parsing stage for {len(files)} files")
            
            # Initialize parsing pipeline
            self.parsing_pipeline = DocumentParsingPipeline()
            
            all_sections = []
            all_requirements = []
            
            # Process each file
            for i, file_path in enumerate(files):
                try:
                    # Determine sourcebook from filename or use first one
                    sourcebook = sourcebooks[0] if sourcebooks else FCASourcebook.SYSC
                    
                    # Parse document
                    sections, requirements = await self.parsing_pipeline.parse_document(
                        file_path, sourcebook
                    )
                    
                    all_sections.extend(sections)
                    all_requirements.extend(requirements)
                    
                    stage_result.items_processed += 1
                    
                    logger.info(f"Parsed {file_path}: {len(sections)} sections, {len(requirements)} requirements")
                    
                except Exception as e:
                    error_msg = f"Failed to parse {file_path}: {e}"
                    stage_result.add_error(error_msg)
                    logger.error(error_msg)
            
            stage_result.metadata = {
                "total_sections": len(all_sections),
                "total_requirements": len(all_requirements),
                "parsing_stats": self.parsing_pipeline.get_parsing_stats()
            }
            
            # Save intermediate results
            if self.config.save_intermediate_results:
                await self._save_intermediate_results("sections", all_sections)
                await self._save_intermediate_results("requirements", all_requirements)
            
            stage_result.complete(success=True)
            
            logger.info(f"Parsing completed: {len(all_sections)} sections, {len(all_requirements)} requirements")
            
            return all_sections, all_requirements
            
        except Exception as e:
            error_msg = f"Parsing stage failed: {e}"
            stage_result.add_error(error_msg)
            stage_result.complete(success=False)
            
            if not self.config.continue_on_errors:
                raise
            
            logger.error(error_msg)
            return [], []
    
    async def _run_categorization_stage(
        self, 
        requirements: List[ExtractedRequirement]
    ) -> List[CategoryResult]:
        """Run requirement categorization stage."""
        
        stage_result = StageResult(
            stage=ProcessingStage.CATEGORIZATION,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.stage_results[ProcessingStage.CATEGORIZATION] = stage_result
        
        try:
            logger.info(f"Starting categorization stage for {len(requirements)} requirements")
            
            # Initialize categorization pipeline
            from .categorization import create_default_categorization_pipeline
            self.categorization_pipeline = await create_default_categorization_pipeline()
            
            # Categorize requirements
            category_results = await self.categorization_pipeline.categorize_requirements(requirements)
            
            stage_result.items_processed = len(category_results)
            stage_result.metadata = {
                "categorization_stats": self._analyze_categorization_results(category_results)
            }
            
            # Save intermediate results
            if self.config.save_intermediate_results:
                await self._save_intermediate_results("categories", category_results)
            
            stage_result.complete(success=True)
            
            logger.info(f"Categorization completed: {len(category_results)} requirements categorized")
            
            return category_results
            
        except Exception as e:
            error_msg = f"Categorization stage failed: {e}"
            stage_result.add_error(error_msg)
            stage_result.complete(success=False)
            
            if not self.config.continue_on_errors:
                raise
            
            logger.error(error_msg)
            return []
    
    async def _run_knowledge_graph_stage(
        self, 
        sections: List[ParsedSection], 
        requirements: List[ExtractedRequirement]
    ):
        """Run knowledge graph construction stage."""
        
        stage_result = StageResult(
            stage=ProcessingStage.KNOWLEDGE_GRAPH,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.stage_results[ProcessingStage.KNOWLEDGE_GRAPH] = stage_result
        
        try:
            logger.info("Starting knowledge graph construction stage")
            
            # Initialize LightRAG
            self.lightrag = FSComplianceLightRAG(self.config.lightrag_config)
            await self.lightrag.initialize()
            
            # Process sections and requirements
            section_results = await self.lightrag.process_sections(sections)
            requirement_results = await self.lightrag.process_requirements(requirements)
            
            stage_result.items_processed = section_results["processed_count"] + requirement_results["processed_count"]
            stage_result.items_failed = len(section_results["errors"]) + len(requirement_results["errors"])
            
            stage_result.metadata = {
                "section_processing": section_results,
                "requirement_processing": requirement_results
            }
            
            # Export knowledge graph
            if self.config.save_intermediate_results:
                graph_export_path = self.config.output_directory / "knowledge_graph.json"
                await self.lightrag.export_knowledge_graph(graph_export_path)
            
            stage_result.complete(success=True)
            
            logger.info("Knowledge graph construction completed")
            
        except Exception as e:
            error_msg = f"Knowledge graph stage failed: {e}"
            stage_result.add_error(error_msg)
            stage_result.complete(success=False)
            
            if not self.config.continue_on_errors:
                raise
            
            logger.error(error_msg)
    
    async def _run_storage_stage(
        self, 
        sections: List[ParsedSection], 
        requirements: List[ExtractedRequirement],
        category_results: List[CategoryResult]
    ):
        """Run knowledge base storage stage."""
        
        stage_result = StageResult(
            stage=ProcessingStage.STORAGE,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.stage_results[ProcessingStage.STORAGE] = stage_result
        
        try:
            logger.info("Starting knowledge base storage stage")
            
            # Initialize knowledge base
            self.knowledge_base = KnowledgeBaseFactory.create_knowledge_base(
                self.config.storage_backend, 
                self.config.storage_config
            )
            
            # Create entities
            section_entities = self._create_section_entities(sections)
            requirement_entities = self._create_requirement_entities(requirements, category_results)
            
            # Store in knowledge base
            section_ids = await self.knowledge_base.bulk_store_sections(section_entities)
            requirement_ids = await self.knowledge_base.bulk_store_requirements(requirement_entities)
            
            stage_result.items_processed = len(section_ids) + len(requirement_ids)
            stage_result.metadata = {
                "sections_stored": len(section_ids),
                "requirements_stored": len(requirement_ids),
                "storage_backend": self.config.storage_backend.value
            }
            
            stage_result.complete(success=True)
            
            logger.info(f"Storage completed: {len(section_ids)} sections, {len(requirement_ids)} requirements")
            
        except Exception as e:
            error_msg = f"Storage stage failed: {e}"
            stage_result.add_error(error_msg)
            stage_result.complete(success=False)
            
            if not self.config.continue_on_errors:
                raise
            
            logger.error(error_msg)
    
    async def _run_indexing_stage(self):
        """Run indexing and finalization stage."""
        
        stage_result = StageResult(
            stage=ProcessingStage.INDEXING,
            status=ProcessingStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        self.stage_results[ProcessingStage.INDEXING] = stage_result
        
        try:
            logger.info("Starting indexing and finalization stage")
            
            # Perform any final indexing operations
            # This could include building search indexes, caches, etc.
            
            stage_result.items_processed = 1  # Indexing operation
            stage_result.metadata = {
                "indexes_built": ["entity_type", "sourcebook", "keywords"]
            }
            
            stage_result.complete(success=True)
            
            logger.info("Indexing and finalization completed")
            
        except Exception as e:
            error_msg = f"Indexing stage failed: {e}"
            stage_result.add_error(error_msg)
            stage_result.complete(success=False)
            
            logger.error(error_msg)
    
    def _create_section_entities(self, sections: List[ParsedSection]) -> List[SectionEntity]:
        """Create section entities for knowledge base storage."""
        entities = []
        
        for section in sections:
            entity = SectionEntity(
                section_id=section.section_id,
                sourcebook=section.sourcebook,
                chapter=section.chapter,
                title=section.title,
                content=section.content,
                level=section.level,
                parent_section=section.parent_section,
                child_sections=section.children_sections
            )
            entities.append(entity)
        
        return entities
    
    def _create_requirement_entities(
        self, 
        requirements: List[ExtractedRequirement],
        category_results: List[CategoryResult]
    ) -> List[RequirementEntity]:
        """Create requirement entities for knowledge base storage."""
        entities = []
        category_map = {cr.requirement_id: cr for cr in category_results}
        
        for requirement in requirements:
            category_result = category_map.get(requirement.section_id)
            
            # Extract sourcebook from section ID
            section_parts = requirement.section_id.split('.')
            sourcebook_str = section_parts[0].lower() if section_parts else "unknown"
            
            # Map to FCASourcebook enum
            try:
                sourcebook = FCASourcebook(sourcebook_str)
            except ValueError:
                sourcebook = FCASourcebook.SYSC  # Default
            
            entity = RequirementEntity(
                section_id=requirement.section_id,
                sourcebook=sourcebook,
                title=f"Requirement {requirement.section_id}",
                content=requirement.requirement_text,
                category_result=category_result,
                cross_references=requirement.cross_references,
                keywords=requirement.keywords
            )
            entities.append(entity)
        
        return entities
    
    def _analyze_categorization_results(self, category_results: List[CategoryResult]) -> Dict[str, Any]:
        """Analyze categorization results for reporting."""
        if not category_results:
            return {}
        
        # Count by category
        category_counts = {}
        confidence_counts = {}
        
        for result in category_results:
            category = result.primary_category.value
            confidence = result.confidence.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence_score for r in category_results) / len(category_results)
        
        return {
            "category_distribution": category_counts,
            "confidence_distribution": confidence_counts,
            "average_confidence_score": avg_confidence,
            "total_categorized": len(category_results)
        }
    
    async def _save_intermediate_results(self, name: str, data: Any):
        """Save intermediate results to files."""
        try:
            import json
            
            output_file = self.config.output_directory / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert data to dict if it has a dict method
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                if hasattr(data[0], 'dict'):
                    json_data = [item.dict() for item in data]
                else:
                    json_data = data
            else:
                json_data = data
            
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.debug(f"Saved intermediate results to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results for {name}: {e}")
    
    def _generate_pipeline_report(
        self, 
        duration: float, 
        failed: bool = False, 
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        
        report = {
            "pipeline_id": f"fscompliance_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": duration,
            "overall_status": "FAILED" if failed else "COMPLETED",
            "config": self.config.dict(),
            "stages": {},
            "summary": {
                "total_stages": len(self.stage_results),
                "completed_stages": 0,
                "failed_stages": 0,
                "total_items_processed": 0,
                "total_errors": 0
            }
        }
        
        if error:
            report["error"] = error
        
        # Analyze stage results
        for stage, result in self.stage_results.items():
            report["stages"][stage.value] = {
                "status": result.status.value,
                "duration_seconds": result.duration_seconds,
                "items_processed": result.items_processed,
                "items_failed": result.items_failed,
                "errors": result.errors,
                "metadata": result.metadata
            }
            
            # Update summary
            if result.status == ProcessingStatus.COMPLETED:
                report["summary"]["completed_stages"] += 1
            elif result.status == ProcessingStatus.FAILED:
                report["summary"]["failed_stages"] += 1
            
            report["summary"]["total_items_processed"] += result.items_processed
            report["summary"]["total_errors"] += len(result.errors)
        
        return report
    
    async def _save_pipeline_report(self, report: Dict[str, Any]):
        """Save pipeline report to file."""
        try:
            import json
            
            report_file = self.config.output_directory / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Pipeline report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline report: {e}")


# Utility functions
async def create_default_pipeline() -> DocumentProcessingPipeline:
    """Create default document processing pipeline."""
    config = PipelineConfig()
    return DocumentProcessingPipeline(config)


async def run_fca_handbook_processing(
    sourcebooks: Optional[List[FCASourcebook]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run complete FCA Handbook processing pipeline."""
    
    if sourcebooks is None:
        sourcebooks = [FCASourcebook.SYSC, FCASourcebook.COBS, FCASourcebook.CONC]
    
    # Create pipeline configuration
    config = PipelineConfig()
    if output_dir:
        config.output_directory = output_dir
    
    # Create and run pipeline
    pipeline = DocumentProcessingPipeline(config)
    
    return await pipeline.run_complete_pipeline(sourcebooks)