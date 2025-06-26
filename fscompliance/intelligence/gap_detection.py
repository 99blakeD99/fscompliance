"""Compliance gap detection logic for identifying regulatory non-compliance."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..models import ComplianceStatus, FCASourcebook, RiskLevel, SeverityLevel
from .requirement_extraction import ExtractedRequirementEnhanced, RequirementScope

logger = logging.getLogger(__name__)


class GapType(str, Enum):
    """Types of compliance gaps."""
    MISSING_REQUIREMENT = "missing_requirement"      # Requirement not addressed
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"  # Partially addressed
    OUTDATED_IMPLEMENTATION = "outdated_implementation"     # Implementation outdated
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"         # Lack of documentation
    PROCESS_GAP = "process_gap"                             # Process deficiency
    POLICY_GAP = "policy_gap"                               # Policy deficiency
    TRAINING_GAP = "training_gap"                           # Training deficiency
    MONITORING_GAP = "monitoring_gap"                       # Monitoring deficiency


class GapSeverity(str, Enum):
    """Severity levels for compliance gaps."""
    CRITICAL = "critical"    # Immediate regulatory risk
    HIGH = "high"           # Significant compliance risk
    MEDIUM = "medium"       # Moderate compliance risk
    LOW = "low"             # Minor compliance risk
    NEGLIGIBLE = "negligible"  # Minimal compliance risk


class RemediationUrgency(str, Enum):
    """Urgency levels for gap remediation."""
    IMMEDIATE = "immediate"    # Within 24 hours
    URGENT = "urgent"         # Within 1 week
    HIGH = "high"             # Within 1 month
    MEDIUM = "medium"         # Within 3 months
    LOW = "low"               # Within 6 months


@dataclass
class PolicyDocument:
    """Represents a policy or procedure document for gap analysis."""
    
    document_id: str
    title: str
    content: str
    document_type: str = "policy"  # policy, procedure, manual, guideline
    last_updated: Optional[datetime] = None
    owner: Optional[str] = None
    applicable_functions: List[str] = None
    applicable_firms: List[str] = None
    
    def __post_init__(self):
        if self.applicable_functions is None:
            self.applicable_functions = []
        if self.applicable_firms is None:
            self.applicable_firms = []


class ComplianceGap(BaseModel):
    """Represents a specific compliance gap identified during analysis."""
    
    gap_id: str = Field(..., description="Unique gap identifier")
    requirement_id: str = Field(..., description="Related requirement ID")
    gap_type: GapType = Field(..., description="Type of compliance gap")
    gap_severity: GapSeverity = Field(..., description="Severity of the gap")
    
    # Gap description
    gap_title: str = Field(..., description="Brief title of the gap")
    gap_description: str = Field(..., description="Detailed description of the gap")
    evidence: str = Field(..., description="Evidence supporting the gap identification")
    
    # Requirement context
    requirement_text: str = Field(..., description="Text of the requirement")
    requirement_scope: RequirementScope = Field(..., description="Scope of the requirement")
    regulatory_source: str = Field(..., description="Regulatory source (e.g., SYSC.4.1.1)")
    
    # Policy/implementation context
    related_documents: List[str] = Field(default_factory=list, description="Related policy documents")
    current_implementation: Optional[str] = Field(None, description="Current implementation description")
    implementation_gaps: List[str] = Field(default_factory=list, description="Specific implementation gaps")
    
    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    potential_impact: str = Field(..., description="Potential impact description")
    likelihood: str = Field(..., description="Likelihood of regulatory scrutiny")
    
    # Remediation
    remediation_urgency: RemediationUrgency = Field(..., description="Urgency for remediation")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended remediation actions")
    estimated_effort: Optional[str] = Field(None, description="Estimated effort for remediation")
    responsible_function: Optional[str] = Field(None, description="Function responsible for remediation")
    
    # Detection metadata
    detection_algorithm: str = Field(..., description="Algorithm used for detection")
    detection_confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "gap_id": "GAP_SYSC_001",
                "requirement_id": "SYSC.4.1.1.001",
                "gap_type": "missing_requirement",
                "gap_severity": "high",
                "gap_title": "Missing governance framework documentation",
                "gap_description": "No documented governance framework found to satisfy SYSC 4.1.1 requirements",
                "evidence": "Document analysis found no references to governance arrangements",
                "requirement_text": "A firm must have robust governance arrangements...",
                "regulatory_source": "SYSC.4.1.1",
                "risk_level": "high",
                "remediation_urgency": "urgent",
                "recommended_actions": ["Create governance framework document", "Define board responsibilities"],
                "detection_confidence": 0.87
            }
        }


class BaseGapDetector(ABC):
    """Abstract base class for compliance gap detection algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_stats = {
            "gaps_detected": 0,
            "critical_gaps": 0,
            "high_severity_gaps": 0,
            "processing_errors": 0,
            "documents_analyzed": 0
        }
    
    @abstractmethod
    async def detect_gaps(
        self, 
        requirements: List[ExtractedRequirementEnhanced],
        policy_documents: List[PolicyDocument]
    ) -> List[ComplianceGap]:
        """Detect compliance gaps between requirements and policy documents."""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get the name of the detection algorithm."""
        pass
    
    def _generate_gap_id(self, requirement_id: str, gap_index: int) -> str:
        """Generate unique gap ID."""
        return f"GAP_{requirement_id.replace('.', '_')}_{gap_index:03d}"
    
    def _assess_gap_severity(
        self, 
        requirement: ExtractedRequirementEnhanced,
        gap_type: GapType,
        evidence_strength: float
    ) -> GapSeverity:
        """Assess the severity of a compliance gap."""
        
        # Base severity from requirement
        base_severity = requirement.severity
        
        # Adjust based on requirement scope
        if requirement.requirement_scope == RequirementScope.MANDATORY:
            if base_severity == SeverityLevel.HIGH:
                return GapSeverity.CRITICAL
            elif base_severity == SeverityLevel.MEDIUM:
                return GapSeverity.HIGH
            else:
                return GapSeverity.MEDIUM
        
        elif requirement.requirement_scope == RequirementScope.CONDITIONAL:
            if base_severity == SeverityLevel.HIGH:
                return GapSeverity.HIGH
            elif base_severity == SeverityLevel.MEDIUM:
                return GapSeverity.MEDIUM
            else:
                return GapSeverity.LOW
        
        else:  # GUIDANCE
            if base_severity == SeverityLevel.HIGH:
                return GapSeverity.MEDIUM
            else:
                return GapSeverity.LOW
    
    def _determine_remediation_urgency(
        self, 
        gap_severity: GapSeverity,
        requirement: ExtractedRequirementEnhanced
    ) -> RemediationUrgency:
        """Determine remediation urgency based on gap severity and requirement characteristics."""
        
        # Check for explicit deadlines
        if requirement.deadlines:
            for deadline in requirement.deadlines:
                if any(urgent_word in deadline.lower() for urgent_word in ["immediate", "without delay"]):
                    return RemediationUrgency.IMMEDIATE
                elif "days" in deadline.lower():
                    return RemediationUrgency.URGENT
        
        # Base urgency on severity
        if gap_severity == GapSeverity.CRITICAL:
            return RemediationUrgency.IMMEDIATE
        elif gap_severity == GapSeverity.HIGH:
            return RemediationUrgency.URGENT
        elif gap_severity == GapSeverity.MEDIUM:
            return RemediationUrgency.HIGH
        elif gap_severity == GapSeverity.LOW:
            return RemediationUrgency.MEDIUM
        else:
            return RemediationUrgency.LOW
    
    def _assess_risk_level(self, gap_severity: GapSeverity, requirement: ExtractedRequirementEnhanced) -> RiskLevel:
        """Assess overall risk level for the gap."""
        
        # Map gap severity to risk level
        severity_risk_map = {
            GapSeverity.CRITICAL: RiskLevel.CRITICAL,
            GapSeverity.HIGH: RiskLevel.HIGH,
            GapSeverity.MEDIUM: RiskLevel.MEDIUM,
            GapSeverity.LOW: RiskLevel.LOW,
            GapSeverity.NEGLIGIBLE: RiskLevel.NEGLIGIBLE
        }
        
        base_risk = severity_risk_map[gap_severity]
        
        # Adjust based on requirement characteristics
        if requirement.requirement_scope == RequirementScope.MANDATORY and \
           requirement.extraction_confidence >= 0.9:
            # Escalate risk for high-confidence mandatory requirements
            if base_risk == RiskLevel.HIGH:
                return RiskLevel.CRITICAL
            elif base_risk == RiskLevel.MEDIUM:
                return RiskLevel.HIGH
        
        return base_risk


class KeywordBasedGapDetector(BaseGapDetector):
    """Gap detector using keyword matching and content analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.keyword_mappings = self._initialize_keyword_mappings()
    
    def get_detector_name(self) -> str:
        return "keyword_based_detector"
    
    def _initialize_keyword_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize keyword mappings for different requirement types."""
        
        return {
            "governance": {
                "required_keywords": [
                    "governance", "board", "oversight", "responsibility", "accountability",
                    "framework", "structure", "arrangement", "committee", "delegation"
                ],
                "implementation_keywords": [
                    "policy", "procedure", "manual", "charter", "terms of reference",
                    "responsibility matrix", "escalation", "reporting line"
                ]
            },
            "risk_management": {
                "required_keywords": [
                    "risk", "control", "assessment", "mitigation", "monitoring",
                    "appetite", "tolerance", "framework", "identification", "measurement"
                ],
                "implementation_keywords": [
                    "risk register", "control testing", "risk assessment", "stress test",
                    "scenario analysis", "risk reporting", "risk committee"
                ]
            },
            "conduct": {
                "required_keywords": [
                    "customer", "client", "fair", "treating", "outcome", "suitability",
                    "appropriateness", "disclosure", "conflict", "best execution"
                ],
                "implementation_keywords": [
                    "customer journey", "suitability process", "disclosure document",
                    "conflicts register", "best execution policy", "complaints procedure"
                ]
            },
            "reporting": {
                "required_keywords": [
                    "report", "notification", "submission", "return", "disclosure",
                    "publication", "filing", "regulatory", "supervisory"
                ],
                "implementation_keywords": [
                    "reporting calendar", "data collection", "validation", "submission process",
                    "regulatory returns", "management information", "board reporting"
                ]
            },
            "record_keeping": {
                "required_keywords": [
                    "record", "documentation", "evidence", "audit trail", "retention",
                    "storage", "archive", "backup", "retrieval", "register"
                ],
                "implementation_keywords": [
                    "document management", "retention schedule", "archive policy",
                    "backup procedure", "access control", "data security"
                ]
            }
        }
    
    async def detect_gaps(
        self, 
        requirements: List[ExtractedRequirementEnhanced],
        policy_documents: List[PolicyDocument]
    ) -> List[ComplianceGap]:
        """Detect gaps using keyword-based analysis."""
        
        gaps = []
        gap_index = 0
        
        logger.info(f"Starting gap detection for {len(requirements)} requirements against {len(policy_documents)} documents")
        
        # Create document content index for faster searching
        document_content_index = self._build_document_index(policy_documents)
        
        for requirement in requirements:
            try:
                # Analyze requirement coverage
                coverage_analysis = await self._analyze_requirement_coverage(
                    requirement, 
                    policy_documents, 
                    document_content_index
                )
                
                # Identify gaps based on coverage analysis
                requirement_gaps = self._identify_gaps_from_coverage(
                    requirement, 
                    coverage_analysis, 
                    gap_index
                )
                
                gaps.extend(requirement_gaps)
                gap_index += len(requirement_gaps)
                
                # Update statistics
                self.detection_stats["gaps_detected"] += len(requirement_gaps)
                for gap in requirement_gaps:
                    if gap.gap_severity == GapSeverity.CRITICAL:
                        self.detection_stats["critical_gaps"] += 1
                    elif gap.gap_severity == GapSeverity.HIGH:
                        self.detection_stats["high_severity_gaps"] += 1
                
            except Exception as e:
                logger.error(f"Error detecting gaps for requirement {requirement.requirement_id}: {e}")
                self.detection_stats["processing_errors"] += 1
                continue
        
        self.detection_stats["documents_analyzed"] = len(policy_documents)
        
        logger.info(f"Gap detection completed: {len(gaps)} gaps identified")
        
        return gaps
    
    def _build_document_index(self, policy_documents: List[PolicyDocument]) -> Dict[str, str]:
        """Build searchable index of document content."""
        
        index = {}
        
        for doc in policy_documents:
            # Combine title and content for comprehensive searching
            searchable_content = f"{doc.title} {doc.content}".lower()
            index[doc.document_id] = searchable_content
        
        return index
    
    async def _analyze_requirement_coverage(
        self, 
        requirement: ExtractedRequirementEnhanced,
        policy_documents: List[PolicyDocument],
        document_index: Dict[str, str]
    ) -> Dict[str, Any]:
        """Analyze how well a requirement is covered by policy documents."""
        
        coverage_analysis = {
            "requirement_keywords_found": [],
            "implementation_keywords_found": [],
            "covering_documents": [],
            "coverage_score": 0.0,
            "evidence_strength": 0.0,
            "missing_elements": []
        }
        
        # Get relevant keyword mappings for requirement type
        req_type = requirement.requirement_type.value
        if req_type not in self.keyword_mappings:
            req_type = "governance"  # Default fallback
        
        keyword_mapping = self.keyword_mappings[req_type]
        required_keywords = keyword_mapping["required_keywords"]
        implementation_keywords = keyword_mapping["implementation_keywords"]
        
        # Search for keywords in documents
        for doc in policy_documents:
            if doc.document_id not in document_index:
                continue
            
            doc_content = document_index[doc.document_id]
            
            # Check for required keywords
            found_required = []
            for keyword in required_keywords:
                if keyword in doc_content:
                    found_required.append(keyword)
            
            # Check for implementation keywords
            found_implementation = []
            for keyword in implementation_keywords:
                if keyword in doc_content:
                    found_implementation.append(keyword)
            
            # If document has relevant content, include it
            if found_required or found_implementation:
                coverage_analysis["covering_documents"].append({
                    "document_id": doc.document_id,
                    "document_title": doc.title,
                    "required_keywords": found_required,
                    "implementation_keywords": found_implementation,
                    "relevance_score": (len(found_required) + len(found_implementation)) / 
                                     (len(required_keywords) + len(implementation_keywords))
                })
                
                coverage_analysis["requirement_keywords_found"].extend(found_required)
                coverage_analysis["implementation_keywords_found"].extend(found_implementation)
        
        # Remove duplicates
        coverage_analysis["requirement_keywords_found"] = list(set(coverage_analysis["requirement_keywords_found"]))
        coverage_analysis["implementation_keywords_found"] = list(set(coverage_analysis["implementation_keywords_found"]))
        
        # Calculate coverage score
        total_required_found = len(coverage_analysis["requirement_keywords_found"])
        total_implementation_found = len(coverage_analysis["implementation_keywords_found"])
        total_possible = len(required_keywords) + len(implementation_keywords)
        
        coverage_analysis["coverage_score"] = (total_required_found + total_implementation_found) / total_possible
        
        # Calculate evidence strength
        evidence_strength = 0.0
        if coverage_analysis["covering_documents"]:
            evidence_strength = max(doc["relevance_score"] for doc in coverage_analysis["covering_documents"])
        
        coverage_analysis["evidence_strength"] = evidence_strength
        
        # Identify missing elements
        missing_required = set(required_keywords) - set(coverage_analysis["requirement_keywords_found"])
        missing_implementation = set(implementation_keywords) - set(coverage_analysis["implementation_keywords_found"])
        
        coverage_analysis["missing_elements"] = {
            "missing_required_keywords": list(missing_required),
            "missing_implementation_keywords": list(missing_implementation)
        }
        
        return coverage_analysis
    
    def _identify_gaps_from_coverage(
        self, 
        requirement: ExtractedRequirementEnhanced,
        coverage_analysis: Dict[str, Any],
        gap_index: int
    ) -> List[ComplianceGap]:
        """Identify specific gaps based on coverage analysis."""
        
        gaps = []
        
        # Determine if there are gaps based on coverage score
        coverage_score = coverage_analysis["coverage_score"]
        evidence_strength = coverage_analysis["evidence_strength"]
        
        # Define gap detection thresholds
        if coverage_score < 0.3:
            gap_type = GapType.MISSING_REQUIREMENT
        elif coverage_score < 0.6:
            gap_type = GapType.INCOMPLETE_IMPLEMENTATION
        elif evidence_strength < 0.4:
            gap_type = GapType.INSUFFICIENT_EVIDENCE
        else:
            # No significant gap detected
            return gaps
        
        # Create gap
        gap_severity = self._assess_gap_severity(requirement, gap_type, evidence_strength)
        remediation_urgency = self._determine_remediation_urgency(gap_severity, requirement)
        risk_level = self._assess_risk_level(gap_severity, requirement)
        
        # Generate gap description and evidence
        gap_description, evidence = self._generate_gap_description(
            requirement, gap_type, coverage_analysis
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(
            requirement, gap_type, coverage_analysis
        )
        
        gap = ComplianceGap(
            gap_id=self._generate_gap_id(requirement.requirement_id, gap_index),
            requirement_id=requirement.requirement_id,
            gap_type=gap_type,
            gap_severity=gap_severity,
            gap_title=f"{gap_type.value.replace('_', ' ').title()} for {requirement.source_section}",
            gap_description=gap_description,
            evidence=evidence,
            requirement_text=requirement.requirement_text,
            requirement_scope=requirement.requirement_scope,
            regulatory_source=requirement.source_section,
            related_documents=[doc["document_id"] for doc in coverage_analysis["covering_documents"]],
            current_implementation=self._describe_current_implementation(coverage_analysis),
            implementation_gaps=coverage_analysis["missing_elements"]["missing_implementation_keywords"],
            risk_level=risk_level,
            potential_impact=self._describe_potential_impact(requirement, gap_severity),
            likelihood="High" if gap_severity in [GapSeverity.CRITICAL, GapSeverity.HIGH] else "Medium",
            remediation_urgency=remediation_urgency,
            recommended_actions=recommended_actions,
            estimated_effort=self._estimate_remediation_effort(gap_type, gap_severity),
            responsible_function=self._determine_responsible_function(requirement),
            detection_algorithm=self.get_detector_name(),
            detection_confidence=min(0.95, requirement.extraction_confidence + evidence_strength) / 2
        )
        
        gaps.append(gap)
        
        return gaps
    
    def _generate_gap_description(
        self, 
        requirement: ExtractedRequirementEnhanced,
        gap_type: GapType,
        coverage_analysis: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Generate gap description and evidence."""
        
        coverage_score = coverage_analysis["coverage_score"]
        missing_elements = coverage_analysis["missing_elements"]
        
        if gap_type == GapType.MISSING_REQUIREMENT:
            description = (
                f"The requirement from {requirement.source_section} appears to have minimal "
                f"coverage in existing policy documentation (coverage score: {coverage_score:.1%}). "
                f"Key regulatory concepts are not adequately addressed."
            )
            evidence = (
                f"Analysis found coverage score of {coverage_score:.1%}. "
                f"Missing required elements: {', '.join(missing_elements['missing_required_keywords'][:5])}. "
                f"Covering documents: {len(coverage_analysis['covering_documents'])}"
            )
        
        elif gap_type == GapType.INCOMPLETE_IMPLEMENTATION:
            description = (
                f"The requirement from {requirement.source_section} is partially addressed "
                f"but implementation appears incomplete (coverage score: {coverage_score:.1%}). "
                f"Additional implementation elements are needed."
            )
            evidence = (
                f"Partial coverage detected with score {coverage_score:.1%}. "
                f"Missing implementation elements: {', '.join(missing_elements['missing_implementation_keywords'][:5])}. "
                f"Found in {len(coverage_analysis['covering_documents'])} documents"
            )
        
        else:  # INSUFFICIENT_EVIDENCE
            description = (
                f"The requirement from {requirement.source_section} may be implemented "
                f"but evidence is insufficient for verification. Documentation appears "
                f"to lack detail or clarity."
            )
            evidence = (
                f"Evidence strength score: {coverage_analysis['evidence_strength']:.1%}. "
                f"Relevant documents found but with limited detail or specificity."
            )
        
        return description, evidence
    
    def _generate_recommended_actions(
        self, 
        requirement: ExtractedRequirementEnhanced,
        gap_type: GapType,
        coverage_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommended remediation actions."""
        
        actions = []
        
        if gap_type == GapType.MISSING_REQUIREMENT:
            actions.extend([
                f"Create policy documentation addressing {requirement.source_section} requirements",
                f"Develop procedures for implementing {requirement.requirement_type.value} controls",
                "Assign clear ownership and responsibilities"
            ])
        
        elif gap_type == GapType.INCOMPLETE_IMPLEMENTATION:
            missing_impl = coverage_analysis["missing_elements"]["missing_implementation_keywords"]
            if missing_impl:
                actions.append(f"Enhance existing documentation to address: {', '.join(missing_impl[:3])}")
            actions.extend([
                "Review and update existing policy documentation",
                "Ensure all required elements are comprehensively covered"
            ])
        
        else:  # INSUFFICIENT_EVIDENCE
            actions.extend([
                "Enhance documentation detail and specificity",
                "Add clear evidence of implementation",
                "Include specific procedures and controls"
            ])
        
        # Add requirement-specific actions
        if requirement.deadlines:
            actions.append(f"Ensure compliance with deadlines: {', '.join(requirement.deadlines)}")
        
        if requirement.documentation_required:
            actions.append("Implement required documentation and record-keeping")
        
        if requirement.reporting_obligations:
            actions.append("Establish required reporting processes")
        
        return actions[:5]  # Limit to 5 actions
    
    def _describe_current_implementation(self, coverage_analysis: Dict[str, Any]) -> Optional[str]:
        """Describe current implementation based on coverage analysis."""
        
        covering_docs = coverage_analysis["covering_documents"]
        if not covering_docs:
            return "No relevant implementation documentation identified"
        
        high_relevance_docs = [doc for doc in covering_docs if doc["relevance_score"] > 0.5]
        
        if high_relevance_docs:
            doc_titles = [doc["document_title"] for doc in high_relevance_docs[:3]]
            return f"Partially addressed in: {', '.join(doc_titles)}"
        else:
            return f"Limited coverage found in {len(covering_docs)} documents"
    
    def _describe_potential_impact(
        self, 
        requirement: ExtractedRequirementEnhanced, 
        gap_severity: GapSeverity
    ) -> str:
        """Describe potential impact of the compliance gap."""
        
        base_impacts = {
            GapSeverity.CRITICAL: "Immediate regulatory enforcement risk, potential sanctions",
            GapSeverity.HIGH: "Significant regulatory scrutiny, reputational damage risk",
            GapSeverity.MEDIUM: "Regulatory attention, compliance monitoring requirements",
            GapSeverity.LOW: "Minor regulatory concern, documentation requests",
            GapSeverity.NEGLIGIBLE: "Minimal regulatory impact"
        }
        
        base_impact = base_impacts[gap_severity]
        
        # Add requirement-specific impacts
        if requirement.requirement_type.value == "conduct":
            base_impact += ", customer detriment risk"
        elif requirement.requirement_type.value == "risk_management":
            base_impact += ", operational risk exposure"
        elif requirement.requirement_type.value == "governance":
            base_impact += ", governance framework deficiency"
        
        return base_impact
    
    def _estimate_remediation_effort(self, gap_type: GapType, gap_severity: GapSeverity) -> str:
        """Estimate effort required for remediation."""
        
        effort_matrix = {
            (GapType.MISSING_REQUIREMENT, GapSeverity.CRITICAL): "High - 2-4 weeks",
            (GapType.MISSING_REQUIREMENT, GapSeverity.HIGH): "Medium-High - 1-3 weeks",
            (GapType.INCOMPLETE_IMPLEMENTATION, GapSeverity.HIGH): "Medium - 1-2 weeks",
            (GapType.INSUFFICIENT_EVIDENCE, GapSeverity.MEDIUM): "Low-Medium - 3-7 days"
        }
        
        specific_effort = effort_matrix.get((gap_type, gap_severity))
        if specific_effort:
            return specific_effort
        
        # Default estimates
        if gap_severity in [GapSeverity.CRITICAL, GapSeverity.HIGH]:
            return "Medium-High - 1-3 weeks"
        elif gap_severity == GapSeverity.MEDIUM:
            return "Medium - 1-2 weeks"
        else:
            return "Low - 3-7 days"
    
    def _determine_responsible_function(self, requirement: ExtractedRequirementEnhanced) -> str:
        """Determine which business function should be responsible for remediation."""
        
        # Map requirement types to responsible functions
        responsibility_map = {
            RequirementType.GOVERNANCE: "Governance / Board Secretariat",
            RequirementType.RISK_MANAGEMENT: "Risk Management",
            RequirementType.CONDUCT: "Compliance / Conduct Risk",
            RequirementType.REPORTING: "Regulatory Reporting",
            RequirementType.RECORD_KEEPING: "Operations / Compliance",
            RequirementType.CLIENT_PROTECTION: "Compliance / Customer Protection"
        }
        
        return responsibility_map.get(requirement.requirement_type, "Compliance")


class GapDetectionPipeline:
    """Pipeline for orchestrating compliance gap detection."""
    
    def __init__(self, detectors: List[BaseGapDetector]):
        self.detectors = detectors
        self.pipeline_stats = {
            "total_gaps_detected": 0,
            "critical_gaps_detected": 0,
            "detection_sessions": 0,
            "requirements_analyzed": 0,
            "documents_analyzed": 0
        }
    
    async def detect_compliance_gaps(
        self, 
        requirements: List[ExtractedRequirementEnhanced],
        policy_documents: List[PolicyDocument]
    ) -> List[ComplianceGap]:
        """Detect compliance gaps using all configured detectors."""
        
        all_gaps = []
        
        logger.info(f"Starting gap detection pipeline for {len(requirements)} requirements using {len(self.detectors)} detectors")
        
        for detector in self.detectors:
            try:
                gaps = await detector.detect_gaps(requirements, policy_documents)
                all_gaps.extend(gaps)
                
                logger.info(f"Detector '{detector.get_detector_name()}' found {len(gaps)} gaps")
                
            except Exception as e:
                logger.error(f"Error in detector '{detector.get_detector_name()}': {e}")
                continue
        
        # Deduplicate gaps
        deduplicated_gaps = self._deduplicate_gaps(all_gaps)
        
        # Update statistics
        self.pipeline_stats["total_gaps_detected"] += len(deduplicated_gaps)
        self.pipeline_stats["critical_gaps_detected"] += sum(
            1 for gap in deduplicated_gaps if gap.gap_severity == GapSeverity.CRITICAL
        )
        self.pipeline_stats["detection_sessions"] += 1
        self.pipeline_stats["requirements_analyzed"] += len(requirements)
        self.pipeline_stats["documents_analyzed"] += len(policy_documents)
        
        logger.info(f"Gap detection pipeline completed: {len(deduplicated_gaps)} unique gaps identified")
        
        return deduplicated_gaps
    
    def _deduplicate_gaps(self, gaps: List[ComplianceGap]) -> List[ComplianceGap]:
        """Remove duplicate gaps based on requirement ID and gap type."""
        
        seen_combinations = set()
        deduplicated = []
        
        for gap in gaps:
            combination_key = (gap.requirement_id, gap.gap_type.value)
            
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                deduplicated.append(gap)
            else:
                # Keep gap with higher detection confidence
                for i, existing in enumerate(deduplicated):
                    if (existing.requirement_id == gap.requirement_id and 
                        existing.gap_type == gap.gap_type):
                        if gap.detection_confidence > existing.detection_confidence:
                            deduplicated[i] = gap
                        break
        
        return deduplicated
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        detector_stats = {}
        for detector in self.detectors:
            detector_stats[detector.get_detector_name()] = detector.detection_stats
        
        return {
            **self.pipeline_stats,
            "detector_statistics": detector_stats
        }


# Factory functions
def create_keyword_based_detector(config: Optional[Dict[str, Any]] = None) -> KeywordBasedGapDetector:
    """Create keyword-based gap detector."""
    if config is None:
        config = {}
    
    return KeywordBasedGapDetector(config)


def create_gap_detection_pipeline(
    detectors: Optional[List[str]] = None
) -> GapDetectionPipeline:
    """Create gap detection pipeline with specified detectors."""
    
    if detectors is None:
        detectors = ["keyword_based"]
    
    detector_instances = []
    
    for detector_name in detectors:
        if detector_name == "keyword_based":
            detector_instances.append(create_keyword_based_detector())
        else:
            logger.warning(f"Detector '{detector_name}' not implemented, skipping")
    
    return GapDetectionPipeline(detector_instances)