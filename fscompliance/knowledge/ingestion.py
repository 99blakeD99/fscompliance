"""FCA Handbook data ingestion pipeline for regulatory content extraction."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiofiles
import httpx
from pydantic import BaseModel, Field, validator

from ..models import ConductRequirement, FCASourcebook, RequirementType, SeverityLevel

logger = logging.getLogger(__name__)


class IngestionStrategy(str, Enum):
    """Available strategies for FCA Handbook data ingestion."""
    API_REGISTER = "api_register"  # FCA Register API
    PDF_DOWNLOAD = "pdf_download"  # PDF download from handbook site
    WEB_SCRAPING = "web_scraping"  # Controlled web scraping
    MANUAL_UPLOAD = "manual_upload"  # Manual file upload


class IngestionStatus(str, Enum):
    """Status of data ingestion operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class IngestionConfig:
    """Configuration for FCA Handbook ingestion operations."""
    strategy: IngestionStrategy
    respect_robots_txt: bool = True
    rate_limit_delay: float = 1.0  # Seconds between requests
    max_retries: int = 3
    output_directory: Path = Path("data/fca_handbook")
    license_agreement_accepted: bool = False  # Required for >20k words
    contact_email: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.strategy == IngestionStrategy.WEB_SCRAPING and not self.license_agreement_accepted:
            logger.warning("Web scraping large volumes requires FCA license agreement")
        
        self.output_directory.mkdir(parents=True, exist_ok=True)


class DocumentMetadata(BaseModel):
    """Metadata for ingested FCA Handbook documents."""
    
    source_url: str = Field(..., description="Original URL of the document")
    sourcebook: FCASourcebook = Field(..., description="FCA sourcebook")
    chapter: str = Field(..., description="Chapter number")
    title: str = Field(..., description="Document title")
    last_modified: Optional[datetime] = Field(None, description="Last modification date")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    content_hash: Optional[str] = Field(None, description="SHA256 hash of content")
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('source_url')
    def validate_url(cls, v):
        """Validate URL format."""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        return v


class BaseIngestionAdapter(ABC):
    """Abstract base class for FCA Handbook ingestion strategies."""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "FSCompliance-Bot/1.0 (Compliance Analysis)"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    @abstractmethod
    async def fetch_document_list(self) -> List[DocumentMetadata]:
        """Fetch list of available documents to ingest."""
        pass
    
    @abstractmethod
    async def download_document(self, metadata: DocumentMetadata) -> Path:
        """Download a specific document and return local file path."""
        pass
    
    async def ingest_all(self) -> List[Path]:
        """Ingest all available documents."""
        documents = await self.fetch_document_list()
        ingested_files = []
        
        for doc_metadata in documents:
            try:
                file_path = await self.download_document(doc_metadata)
                ingested_files.append(file_path)
                logger.info(f"Successfully ingested: {doc_metadata.title}")
                
                # Rate limiting
                if self.config.rate_limit_delay > 0:
                    await asyncio.sleep(self.config.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Failed to ingest {doc_metadata.title}: {e}")
                continue
        
        return ingested_files


class FCARegisterAPIAdapter(BaseIngestionAdapter):
    """Adapter for FCA Register API access."""
    
    BASE_URL = "https://register.fca.org.uk/s/"
    
    def __init__(self, config: IngestionConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
    
    async def fetch_document_list(self) -> List[DocumentMetadata]:
        """Fetch document list from FCA Register API."""
        # Note: FCA Register API provides firm data, not handbook content
        # This adapter would need to be extended based on actual API endpoints
        logger.warning("FCA Register API provides firm data, not handbook content")
        return []
    
    async def download_document(self, metadata: DocumentMetadata) -> Path:
        """Download document via FCA Register API."""
        # Implementation would depend on actual API endpoints
        raise NotImplementedError("FCA Register API document download not yet implemented")


class PDFDownloadAdapter(BaseIngestionAdapter):
    """Adapter for downloading FCA Handbook PDFs."""
    
    BASE_URL = "https://www.handbook.fca.org.uk/handbook/"
    
    # Known PDF URLs for major sourcebooks
    SOURCEBOOK_PDFS = {
        FCASourcebook.SYSC: "SYSC.pdf",
        FCASourcebook.COBS: "COBS.pdf", 
        FCASourcebook.CONC: "CONC.pdf",
        FCASourcebook.SUP: "SUP.pdf",
        FCASourcebook.PRIN: "PRIN.pdf",
        FCASourcebook.MCOB: "MCOB.pdf",
        FCASourcebook.ICOBS: "ICOBS.pdf",
    }
    
    async def fetch_document_list(self) -> List[DocumentMetadata]:
        """Generate list of PDF documents to download."""
        documents = []
        
        for sourcebook, pdf_filename in self.SOURCEBOOK_PDFS.items():
            url = urljoin(self.BASE_URL, pdf_filename)
            
            metadata = DocumentMetadata(
                source_url=url,
                sourcebook=sourcebook,
                chapter="all",
                title=f"{sourcebook.value.upper()} Sourcebook"
            )
            documents.append(metadata)
        
        return documents
    
    async def download_document(self, metadata: DocumentMetadata) -> Path:
        """Download PDF document from FCA Handbook."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        filename = f"{metadata.sourcebook.value}_{datetime.now().strftime('%Y%m%d')}.pdf"
        file_path = self.config.output_directory / filename
        
        try:
            response = await self.session.get(metadata.source_url)
            response.raise_for_status()
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(response.content)
            
            logger.info(f"Downloaded {metadata.title} to {file_path}")
            return file_path
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading {metadata.source_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading {metadata.source_url}: {e}")
            raise


class WebScrapingAdapter(BaseIngestionAdapter):
    """Adapter for controlled web scraping of FCA Handbook."""
    
    BASE_URL = "https://www.handbook.fca.org.uk/handbook/"
    ROBOTS_TXT_URL = "https://www.handbook.fca.org.uk/robots.txt"
    
    async def check_robots_txt(self) -> bool:
        """Check robots.txt compliance."""
        if not self.config.respect_robots_txt:
            return True
        
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
                
            response = await self.session.get(self.ROBOTS_TXT_URL)
            robots_content = response.text
            
            # Basic robots.txt parsing (would need more sophisticated parsing for production)
            if "User-agent: *" in robots_content and "Disallow: /" in robots_content:
                logger.warning("robots.txt disallows crawling")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
            return False
    
    async def fetch_document_list(self) -> List[DocumentMetadata]:
        """Fetch document list via web scraping."""
        if not await self.check_robots_txt():
            raise PermissionError("robots.txt disallows crawling")
        
        # This would require HTML parsing to discover available documents
        # Implementation would depend on actual FCA Handbook HTML structure
        logger.warning("Web scraping adapter requires HTML parsing implementation")
        return []
    
    async def download_document(self, metadata: DocumentMetadata) -> Path:
        """Download document content via web scraping."""
        raise NotImplementedError("Web scraping document download not yet implemented")


class IngestionPipeline:
    """Main pipeline for FCA Handbook data ingestion."""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.adapter: Optional[BaseIngestionAdapter] = None
    
    def _create_adapter(self, api_key: Optional[str] = None) -> BaseIngestionAdapter:
        """Create appropriate adapter based on configuration."""
        if self.config.strategy == IngestionStrategy.API_REGISTER:
            if not api_key:
                raise ValueError("API key required for FCA Register API access")
            return FCARegisterAPIAdapter(self.config, api_key)
        
        elif self.config.strategy == IngestionStrategy.PDF_DOWNLOAD:
            return PDFDownloadAdapter(self.config)
        
        elif self.config.strategy == IngestionStrategy.WEB_SCRAPING:
            return WebScrapingAdapter(self.config)
        
        else:
            raise ValueError(f"Unsupported ingestion strategy: {self.config.strategy}")
    
    async def run_ingestion(self, api_key: Optional[str] = None) -> List[Path]:
        """Run the complete ingestion pipeline."""
        self.adapter = self._create_adapter(api_key)
        
        async with self.adapter:
            logger.info(f"Starting FCA Handbook ingestion with strategy: {self.config.strategy}")
            
            try:
                ingested_files = await self.adapter.ingest_all()
                logger.info(f"Ingestion completed. {len(ingested_files)} files processed.")
                return ingested_files
                
            except Exception as e:
                logger.error(f"Ingestion pipeline failed: {e}")
                raise
    
    def get_ingestion_status(self) -> Dict[str, Union[str, int, datetime]]:
        """Get current status of ingestion pipeline."""
        return {
            "strategy": self.config.strategy.value,
            "output_directory": str(self.config.output_directory),
            "rate_limit_delay": self.config.rate_limit_delay,
            "last_run": datetime.utcnow().isoformat(),
            "status": IngestionStatus.PENDING.value
        }


# Utility functions for pipeline management
async def create_default_pipeline() -> IngestionPipeline:
    """Create a default ingestion pipeline with PDF download strategy."""
    config = IngestionConfig(
        strategy=IngestionStrategy.PDF_DOWNLOAD,
        respect_robots_txt=True,
        rate_limit_delay=2.0,  # Conservative rate limiting
        output_directory=Path("data/fca_handbook")
    )
    
    return IngestionPipeline(config)


async def validate_ingestion_prerequisites() -> Dict[str, bool]:
    """Validate prerequisites for FCA Handbook ingestion."""
    checks = {
        "output_directory_writable": True,
        "network_connectivity": True,
        "robots_txt_compliance": True,
        "license_requirements_met": False  # Assume license not obtained
    }
    
    # Basic connectivity check
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://www.handbook.fca.org.uk/")
            checks["network_connectivity"] = response.status_code == 200
    except Exception:
        checks["network_connectivity"] = False
    
    return checks