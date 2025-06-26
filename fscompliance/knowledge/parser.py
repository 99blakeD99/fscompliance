"""Document parsing and text extraction for FCA Handbook content."""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ..models import ConductRequirement, FCASourcebook, RequirementType, SeverityLevel

logger = logging.getLogger(__name__)


class ParsedSection(BaseModel):
    """Represents a parsed section from FCA Handbook."""
    
    section_id: str = Field(..., description="Unique section identifier (e.g., SYSC.4.1.1)")
    sourcebook: FCASourcebook = Field(..., description="Source sourcebook")
    chapter: str = Field(..., description="Chapter number")
    subsection: Optional[str] = Field(None, description="Subsection identifier")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Raw section content")
    level: int = Field(..., description="Hierarchical level (1=chapter, 2=section, etc)")
    parent_section: Optional[str] = Field(None, description="Parent section ID")
    children_sections: List[str] = Field(default_factory=list, description="Child section IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "section_id": "SYSC.4.1.1",
                "sourcebook": "sysc",
                "chapter": "4",
                "subsection": "1.1",
                "title": "General organisational requirements",
                "content": "A firm must have robust governance arrangements...",
                "level": 3,
                "parent_section": "SYSC.4.1",
                "children_sections": ["SYSC.4.1.1R", "SYSC.4.1.1G"]
            }
        }


class ExtractedRequirement(BaseModel):
    """Represents an extracted regulatory requirement."""
    
    section_id: str = Field(..., description="Source section ID")
    requirement_text: str = Field(..., description="Requirement text")
    requirement_type: RequirementType = Field(..., description="Classified requirement type")
    severity: SeverityLevel = Field(..., description="Assessed severity level")
    rule_guidance_indicator: Optional[str] = Field(None, description="R/G/E indicator from FCA")
    cross_references: List[str] = Field(default_factory=list, description="Referenced sections")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")


@dataclass
class ParsingStats:
    """Statistics from document parsing operations."""
    total_sections: int = 0
    requirements_extracted: int = 0
    cross_references_found: int = 0
    parsing_errors: int = 0
    processing_time_seconds: float = 0.0
    file_size_bytes: int = 0


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    def __init__(self, sourcebook: FCASourcebook):
        self.sourcebook = sourcebook
        self.parsing_stats = ParsingStats()
    
    @abstractmethod
    async def parse_document(self, file_path: Path) -> List[ParsedSection]:
        """Parse document and extract structured sections."""
        pass
    
    @abstractmethod
    async def extract_requirements(self, sections: List[ParsedSection]) -> List[ExtractedRequirement]:
        """Extract regulatory requirements from parsed sections."""
        pass
    
    def _generate_section_id(self, sourcebook: str, chapter: str, section: str) -> str:
        """Generate standardized section ID."""
        return f"{sourcebook.upper()}.{chapter}.{section}"
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class PDFParser(BaseDocumentParser):
    """Parser for FCA Handbook PDF documents."""
    
    def __init__(self, sourcebook: FCASourcebook):
        super().__init__(sourcebook)
        try:
            import PyPDF2
            self.pdf_library = PyPDF2
        except ImportError:
            logger.error("PyPDF2 not available. Install with: pip install PyPDF2")
            raise
    
    async def parse_document(self, file_path: Path) -> List[ParsedSection]:
        """Parse PDF document and extract sections."""
        start_time = datetime.utcnow()
        sections = []
        
        try:
            self.parsing_stats.file_size_bytes = file_path.stat().st_size
            
            with open(file_path, 'rb') as file:
                pdf_reader = self.pdf_library.PdfReader(file)
                
                full_text = ""
                for page in pdf_reader.pages:
                    try:
                        full_text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page: {e}")
                        self.parsing_stats.parsing_errors += 1
                
                sections = await self._parse_text_into_sections(full_text)
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            self.parsing_stats.parsing_errors += 1
            raise
        
        finally:
            end_time = datetime.utcnow()
            self.parsing_stats.processing_time_seconds = (end_time - start_time).total_seconds()
            self.parsing_stats.total_sections = len(sections)
        
        return sections
    
    async def _parse_text_into_sections(self, text: str) -> List[ParsedSection]:
        """Parse extracted text into structured sections."""
        sections = []
        
        # FCA section patterns (e.g., "4.1.1", "4.1.1R", "4.1.1G")
        section_pattern = r'^(\d+)\.(\d+)\.(\d+)([RGE]?)\s+(.*?)(?=^\d+\.\d+\.\d+[RGE]?|\Z)'
        chapter_pattern = r'^(\d+)\s+([A-Z][^0-9]*?)(?=^\d+\s+[A-Z]|\d+\.\d+|\Z)'
        
        # Extract chapters first
        chapter_matches = re.finditer(chapter_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in chapter_matches:
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            
            section_id = f"{self.sourcebook.value.upper()}.{chapter_num}"
            
            section = ParsedSection(
                section_id=section_id,
                sourcebook=self.sourcebook,
                chapter=chapter_num,
                title=chapter_title,
                content=match.group(0),
                level=1
            )
            sections.append(section)
        
        # Extract detailed sections
        section_matches = re.finditer(section_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in section_matches:
            chapter_num = match.group(1)
            section_num = match.group(2)
            subsection_num = match.group(3)
            rule_indicator = match.group(4) or ""
            content = match.group(5).strip()
            
            # Extract title from first line of content
            lines = content.split('\n')
            title = lines[0].strip() if lines else "Untitled"
            
            section_id = f"{self.sourcebook.value.upper()}.{chapter_num}.{section_num}.{subsection_num}{rule_indicator}"
            
            section = ParsedSection(
                section_id=section_id,
                sourcebook=self.sourcebook,
                chapter=chapter_num,
                subsection=f"{section_num}.{subsection_num}",
                title=title,
                content=content,
                level=3,
                parent_section=f"{self.sourcebook.value.upper()}.{chapter_num}.{section_num}"
            )
            sections.append(section)
        
        return sections
    
    async def extract_requirements(self, sections: List[ParsedSection]) -> List[ExtractedRequirement]:
        """Extract regulatory requirements from parsed sections."""
        requirements = []
        
        for section in sections:
            try:
                extracted = await self._extract_requirements_from_section(section)
                requirements.extend(extracted)
            except Exception as e:
                logger.error(f"Error extracting requirements from {section.section_id}: {e}")
                self.parsing_stats.parsing_errors += 1
        
        self.parsing_stats.requirements_extracted = len(requirements)
        return requirements
    
    async def _extract_requirements_from_section(self, section: ParsedSection) -> List[ExtractedRequirement]:
        """Extract requirements from a single section."""
        requirements = []
        
        # Rule indicators in FCA Handbook
        rule_patterns = {
            'R': RequirementType.GOVERNANCE,  # Rule
            'G': RequirementType.GOVERNANCE,  # Guidance
            'E': RequirementType.GOVERNANCE,  # Evidential provision
        }
        
        # Requirement keywords for classification
        requirement_keywords = {
            RequirementType.GOVERNANCE: ['must', 'shall', 'required', 'obligation', 'governance'],
            RequirementType.CONDUCT: ['customer', 'client', 'fair', 'conduct', 'treating'],
            RequirementType.REPORTING: ['report', 'notification', 'disclosure', 'return'],
            RequirementType.RECORD_KEEPING: ['record', 'maintain', 'retain', 'documentation'],
            RequirementType.RISK_MANAGEMENT: ['risk', 'control', 'assessment', 'mitigation'],
            RequirementType.CLIENT_PROTECTION: ['protection', 'safeguard', 'segregation', 'security']
        }
        
        # Severity indicators
        severity_keywords = {
            SeverityLevel.HIGH: ['must not', 'prohibited', 'breach', 'criminal', 'immediately'],
            SeverityLevel.MEDIUM: ['should', 'expected', 'appropriate', 'reasonable'],
            SeverityLevel.LOW: ['may', 'consider', 'guidance', 'example']
        }
        
        # Split content into sentences for requirement extraction
        sentences = re.split(r'[.!?]+', section.content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence contains requirement language
            if any(keyword in sentence.lower() for keyword in ['must', 'shall', 'required', 'should']):
                
                # Classify requirement type
                req_type = RequirementType.GOVERNANCE  # Default
                for rtype, keywords in requirement_keywords.items():
                    if any(keyword in sentence.lower() for keyword in keywords):
                        req_type = rtype
                        break
                
                # Assess severity
                severity = SeverityLevel.MEDIUM  # Default
                for sev, keywords in severity_keywords.items():
                    if any(keyword in sentence.lower() for keyword in keywords):
                        severity = sev
                        break
                
                # Extract cross-references (e.g., "see SYSC 4.1.2")
                cross_refs = re.findall(r'[A-Z]{3,5}\s+\d+\.\d+(?:\.\d+)?', sentence)
                
                # Extract keywords
                keywords = []
                for word in sentence.lower().split():
                    if len(word) > 4 and word.isalpha():
                        keywords.append(word)
                
                # Calculate confidence score based on requirement indicators
                confidence = 0.5  # Base confidence
                if 'must' in sentence.lower():
                    confidence += 0.3
                if section.section_id.endswith('R'):  # Rule indicator
                    confidence += 0.2
                if any(ref in sentence for ref in cross_refs):
                    confidence += 0.1
                
                confidence = min(confidence, 1.0)
                
                requirement = ExtractedRequirement(
                    section_id=section.section_id,
                    requirement_text=sentence,
                    requirement_type=req_type,
                    severity=severity,
                    rule_guidance_indicator=section.section_id[-1] if section.section_id[-1] in 'RGE' else None,
                    cross_references=cross_refs,
                    keywords=keywords[:10],  # Limit keywords
                    confidence_score=confidence
                )
                
                requirements.append(requirement)
        
        return requirements


class HTMLParser(BaseDocumentParser):
    """Parser for FCA Handbook HTML content."""
    
    def __init__(self, sourcebook: FCASourcebook):
        super().__init__(sourcebook)
        try:
            from bs4 import BeautifulSoup
            self.BeautifulSoup = BeautifulSoup
        except ImportError:
            logger.error("BeautifulSoup not available. Install with: pip install beautifulsoup4")
            raise
    
    async def parse_document(self, file_path: Path) -> List[ParsedSection]:
        """Parse HTML document and extract sections."""
        start_time = datetime.utcnow()
        sections = []
        
        try:
            self.parsing_stats.file_size_bytes = file_path.stat().st_size
            
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                soup = self.BeautifulSoup(html_content, 'html.parser')
                
                sections = await self._parse_html_sections(soup)
            
        except Exception as e:
            logger.error(f"Error parsing HTML {file_path}: {e}")
            self.parsing_stats.parsing_errors += 1
            raise
        
        finally:
            end_time = datetime.utcnow()
            self.parsing_stats.processing_time_seconds = (end_time - start_time).total_seconds()
            self.parsing_stats.total_sections = len(sections)
        
        return sections
    
    async def _parse_html_sections(self, soup) -> List[ParsedSection]:
        """Parse HTML soup into structured sections."""
        sections = []
        
        # Find section headers and content
        # This would depend on actual FCA Handbook HTML structure
        section_headers = soup.find_all(['h1', 'h2', 'h3', 'h4'], class_=re.compile(r'section|rule|guidance'))
        
        for header in section_headers:
            # Extract section information from header
            header_text = header.get_text().strip()
            
            # Pattern matching for section IDs
            section_match = re.match(r'(\d+)\.(\d+)\.(\d+)([RGE]?)\s*(.*)', header_text)
            
            if section_match:
                chapter = section_match.group(1)
                section_num = section_match.group(2)
                subsection = section_match.group(3)
                indicator = section_match.group(4)
                title = section_match.group(5)
                
                section_id = f"{self.sourcebook.value.upper()}.{chapter}.{section_num}.{subsection}{indicator}"
                
                # Extract content following the header
                content_elements = []
                current = header.next_sibling
                while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                    if hasattr(current, 'get_text'):
                        content_elements.append(current.get_text())
                    current = current.next_sibling
                
                content = '\n'.join(content_elements).strip()
                
                section = ParsedSection(
                    section_id=section_id,
                    sourcebook=self.sourcebook,
                    chapter=chapter,
                    subsection=f"{section_num}.{subsection}",
                    title=title,
                    content=content,
                    level=len(section_id.split('.')),
                )
                
                sections.append(section)
        
        return sections
    
    async def extract_requirements(self, sections: List[ParsedSection]) -> List[ExtractedRequirement]:
        """Extract requirements from HTML sections."""
        # Reuse PDF parser logic for requirement extraction
        pdf_parser = PDFParser(self.sourcebook)
        return await pdf_parser.extract_requirements(sections)


class DocumentParsingPipeline:
    """Main pipeline for document parsing and requirement extraction."""
    
    def __init__(self):
        self.parsers: Dict[str, BaseDocumentParser] = {}
    
    def register_parser(self, file_extension: str, parser_class: type, sourcebook: FCASourcebook):
        """Register a parser for specific file extensions."""
        self.parsers[file_extension.lower()] = parser_class(sourcebook)
    
    async def parse_document(self, file_path: Path, sourcebook: FCASourcebook) -> Tuple[List[ParsedSection], List[ExtractedRequirement]]:
        """Parse document and extract requirements."""
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.parsers:
            # Auto-register parser based on extension
            if file_extension == '.pdf':
                self.register_parser('.pdf', PDFParser, sourcebook)
            elif file_extension in ['.html', '.htm']:
                self.register_parser(file_extension, HTMLParser, sourcebook)
            else:
                raise ValueError(f"No parser available for file extension: {file_extension}")
        
        parser = self.parsers[file_extension]
        
        # Parse document into sections
        sections = await parser.parse_document(file_path)
        
        # Extract requirements from sections
        requirements = await parser.extract_requirements(sections)
        
        logger.info(f"Parsed {len(sections)} sections and extracted {len(requirements)} requirements from {file_path}")
        
        return sections, requirements
    
    def get_parsing_stats(self) -> Dict[str, ParsingStats]:
        """Get parsing statistics for all registered parsers."""
        return {ext: parser.parsing_stats for ext, parser in self.parsers.items()}


# Utility functions
async def convert_to_conduct_requirements(
    extracted_requirements: List[ExtractedRequirement],
    sourcebook: FCASourcebook
) -> List[ConductRequirement]:
    """Convert extracted requirements to ConductRequirement models."""
    conduct_requirements = []
    
    for extracted in extracted_requirements:
        try:
            # Map section ID to proper format
            section_parts = extracted.section_id.split('.')
            if len(section_parts) >= 3:
                chapter = section_parts[1]
                section_ref = '.'.join(section_parts[1:])
            else:
                chapter = "unknown"
                section_ref = extracted.section_id
            
            requirement = ConductRequirement(
                source="FCA Handbook",
                sourcebook=sourcebook,
                section=extracted.section_id,
                chapter=chapter,
                title=f"Section {section_ref}",
                content=extracted.requirement_text,
                requirement_type=extracted.requirement_type,
                severity=extracted.severity,
                last_updated=datetime.utcnow(),
                effective_date=datetime.utcnow(),
                related_requirements=extracted.cross_references
            )
            
            conduct_requirements.append(requirement)
            
        except Exception as e:
            logger.error(f"Error converting extracted requirement to ConductRequirement: {e}")
            continue
    
    return conduct_requirements