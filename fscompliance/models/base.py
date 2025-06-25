"""Base models for FSCompliance regulatory framework extensibility."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks."""
    FCA_HANDBOOK = "fca_handbook"
    PRA_HANDBOOK = "pra_handbook"
    GDPR = "gdpr"
    MIFID_II = "mifid_ii"


class RequirementType(str, Enum):
    """Types of regulatory requirements."""
    GOVERNANCE = "governance"
    CONDUCT = "conduct"
    REPORTING = "reporting"
    RECORD_KEEPING = "record_keeping"
    RISK_MANAGEMENT = "risk_management"
    CLIENT_PROTECTION = "client_protection"
    CAPITAL_ADEQUACY = "capital_adequacy"
    OPERATIONAL_RESILIENCE = "operational_resilience"


class SeverityLevel(str, Enum):
    """Severity levels for regulatory requirements."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UserRole(str, Enum):
    """User roles for compliance queries."""
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_MANAGER = "risk_manager"
    REGULATORY_INSPECTOR = "regulatory_inspector"
    PROFESSIONAL_ADVISER = "professional_adviser"
    SENIOR_MANAGER = "senior_manager"


class QueryType(str, Enum):
    """Types of compliance queries."""
    GAP_ANALYSIS = "gap_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    REGULATORY_REPORTING = "regulatory_reporting"
    POLICY_REVIEW = "policy_review"
    INCIDENT_ASSESSMENT = "incident_assessment"


class ComplianceStatus(str, Enum):
    """Compliance status outcomes."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANCE = "partial_compliance"
    UNCLEAR = "unclear"
    REQUIRES_REVIEW = "requires_review"


class RiskLevel(str, Enum):
    """Risk level assessments."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class BaseRegulatoryModel(BaseModel):
    """Base model for all regulatory data with common fields."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    framework: RegulatoryFramework = Field(..., description="Regulatory framework")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v, values):
        """Set updated_at to current time when model is modified."""
        return datetime.utcnow()


class BaseRequirement(BaseRegulatoryModel):
    """Base model for regulatory requirements across frameworks."""
    
    source: str = Field(..., description="Source document or regulation")
    section: str = Field(..., description="Section reference within source")
    title: str = Field(..., description="Requirement title")
    content: str = Field(..., description="Full requirement text")
    requirement_type: RequirementType = Field(..., description="Type of requirement")
    severity: SeverityLevel = Field(..., description="Severity level")
    applicability: List[str] = Field(default_factory=list, description="Applicable firm types")
    last_updated: datetime = Field(..., description="When requirement was last updated")
    related_requirements: List[str] = Field(default_factory=list, description="Related requirement IDs")
    
    @validator('content')
    def content_not_empty(cls, v):
        """Ensure requirement content is not empty."""
        if not v or not v.strip():
            raise ValueError("Requirement content cannot be empty")
        return v.strip()
    
    @validator('section')
    def section_format(cls, v):
        """Validate section reference format."""
        if not v or not v.strip():
            raise ValueError("Section reference cannot be empty")
        return v.strip().upper()


class BaseQuery(BaseRegulatoryModel):
    """Base model for compliance queries across frameworks."""
    
    user_role: UserRole = Field(..., description="Role of user making query")
    query_type: QueryType = Field(..., description="Type of compliance query")
    content: str = Field(..., description="Query content")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    privacy_mode: bool = Field(default=True, description="Enable privacy controls")
    firm_type: Optional[str] = Field(None, description="Type of firm making query")
    regulatory_scope: List[str] = Field(default_factory=list, description="Relevant regulatory areas")
    
    @validator('content')
    def content_not_empty(cls, v):
        """Ensure query content is not empty."""
        if not v or not v.strip():
            raise ValueError("Query content cannot be empty")
        return v.strip()


class BaseResponse(BaseRegulatoryModel):
    """Base model for compliance responses across frameworks."""
    
    query_id: str = Field(..., description="ID of the original query")
    compliance_status: ComplianceStatus = Field(..., description="Overall compliance status")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence score")
    risk_level: RiskLevel = Field(..., description="Overall risk assessment")
    action_required: bool = Field(..., description="Whether immediate action is needed")
    sources: List[str] = Field(default_factory=list, description="Source references")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Ensure confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v