"""FCA Handbook specific models for FSCompliance."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator

from .base import (
    BaseQuery,
    BaseRequirement,
    BaseResponse,
    ComplianceStatus,
    QueryType,
    RegulatoryFramework,
    RequirementType,
    RiskLevel,
    SeverityLevel,
    UserRole,
)


class FCASourcebook(str, Enum):
    """FCA Handbook sourcebooks."""
    SYSC = "sysc"  # Senior Management Arrangements, Systems and Controls
    COBS = "cobs"  # Conduct of Business
    CONC = "conc"  # Consumer Credit
    SUP = "sup"    # Supervision
    PRIN = "prin"  # Principles for Businesses
    APER = "aper"  # Approved Persons
    FIT = "fit"    # Fit and Proper test
    COMC = "comc"  # Compulsory Insurance
    MCOB = "mcob"  # Mortgages and Home Finance
    ICOBS = "icobs"  # Insurance Conduct of Business


class FCAFirmType(str, Enum):
    """FCA regulated firm types."""
    INVESTMENT_FIRM = "investment_firm"
    BANK = "bank"
    INSURER = "insurer"
    PAYMENT_INSTITUTION = "payment_institution"
    ELECTRONIC_MONEY_INSTITUTION = "electronic_money_institution"
    CONSUMER_CREDIT_FIRM = "consumer_credit_firm"
    MORTGAGE_BROKER = "mortgage_broker"
    APPOINTED_REPRESENTATIVE = "appointed_representative"


class ConductRequirement(BaseRequirement):
    """FCA Conduct Requirement model with FCA-specific fields."""
    
    framework: RegulatoryFramework = Field(
        default=RegulatoryFramework.FCA_HANDBOOK,
        description="Always FCA Handbook for this model"
    )
    sourcebook: FCASourcebook = Field(..., description="FCA Handbook sourcebook")
    chapter: str = Field(..., description="Chapter number within sourcebook")
    sub_section: Optional[str] = Field(None, description="Detailed subsection reference")
    application_scope: List[FCAFirmType] = Field(
        default_factory=list,
        description="Firm types this requirement applies to"
    )
    compliance_deadline: Optional[datetime] = Field(
        None,
        description="When compliance is required"
    )
    reporting_requirements: List[str] = Field(
        default_factory=list,
        description="Associated reporting obligations"
    )
    record_keeping_period: Optional[int] = Field(
        None,
        description="Record retention period in years"
    )
    effective_date: datetime = Field(..., description="When requirement became effective")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "fca-sysc-4-1-1",
                "framework": "fca_handbook",
                "source": "FCA Handbook",
                "sourcebook": "sysc",
                "section": "SYSC.4.1.1",
                "chapter": "4",
                "sub_section": "1(1)",
                "title": "General organisational requirements",
                "content": "A firm must have robust governance arrangements...",
                "requirement_type": "governance",
                "severity": "high",
                "applicability": ["investment_firm", "bank"],
                "application_scope": ["investment_firm", "bank"],
                "last_updated": "2024-01-01T00:00:00Z",
                "effective_date": "2024-01-01T00:00:00Z",
                "related_requirements": ["fca-sysc-4-1-2"],
                "reporting_requirements": ["Annual compliance report"],
                "record_keeping_period": 5
            }
        }
    
    @validator('section')
    def validate_fca_section_format(cls, v, values):
        """Validate FCA section reference format."""
        if 'sourcebook' in values:
            sourcebook = values['sourcebook'].upper()
            if not v.startswith(sourcebook):
                raise ValueError(f"Section reference must start with sourcebook {sourcebook}")
        return v
    
    @validator('record_keeping_period')
    def validate_record_keeping_period(cls, v):
        """Validate record keeping period is reasonable."""
        if v is not None and (v < 1 or v > 50):
            raise ValueError("Record keeping period must be between 1 and 50 years")
        return v


class ComplianceQuery(BaseQuery):
    """FCA Compliance Query model with FCA-specific context."""
    
    framework: RegulatoryFramework = Field(
        default=RegulatoryFramework.FCA_HANDBOOK,
        description="Always FCA Handbook for this model"
    )
    fca_firm_type: Optional[FCAFirmType] = Field(
        None,
        description="FCA regulated firm type making the query"
    )
    sourcebooks_in_scope: List[FCASourcebook] = Field(
        default_factory=list,
        description="FCA sourcebooks relevant to query"
    )
    customer_facing: bool = Field(
        default=False,
        description="Whether query involves customer-facing activities"
    )
    high_risk_customer: bool = Field(
        default=False,
        description="Whether query involves high-risk customers"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "id": "query-123",
                "framework": "fca_handbook",
                "user_role": "compliance_officer",
                "query_type": "gap_analysis",
                "content": "What are the conduct requirements for investment advice to retail clients?",
                "fca_firm_type": "investment_firm",
                "sourcebooks_in_scope": ["cobs", "sysc"],
                "customer_facing": True,
                "privacy_mode": True,
                "context": {
                    "client_type": "retail",
                    "investment_type": "portfolio_management"
                }
            }
        }


class ComplianceGap(BaseModel):
    """Represents a specific compliance gap identified during analysis."""
    
    requirement_id: str = Field(..., description="ID of the requirement with gap")
    gap_description: str = Field(..., description="Description of the compliance gap")
    severity: SeverityLevel = Field(..., description="Severity of the gap")
    recommended_action: str = Field(..., description="Recommended action to address gap")
    deadline: Optional[datetime] = Field(None, description="Deadline to address gap")
    
    @validator('gap_description')
    def gap_description_not_empty(cls, v):
        """Ensure gap description is not empty."""
        if not v or not v.strip():
            raise ValueError("Gap description cannot be empty")
        return v.strip()


class ComplianceResponse(BaseResponse):
    """FCA Compliance Response model with detailed analysis."""
    
    framework: RegulatoryFramework = Field(
        default=RegulatoryFramework.FCA_HANDBOOK,
        description="Always FCA Handbook for this model"
    )
    requirements: List[ConductRequirement] = Field(
        default_factory=list,
        description="Relevant FCA requirements"
    )
    gaps_identified: List[ComplianceGap] = Field(
        default_factory=list,
        description="Specific compliance gaps found"
    )
    fca_sourcebooks_referenced: List[FCASourcebook] = Field(
        default_factory=list,
        description="FCA sourcebooks referenced in response"
    )
    regulatory_impact: Optional[str] = Field(
        None,
        description="Assessment of regulatory impact"
    )
    escalation_required: bool = Field(
        default=False,
        description="Whether escalation to senior management is required"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "id": "response-123",
                "framework": "fca_handbook",
                "query_id": "query-123",
                "compliance_status": "partial_compliance",
                "confidence_score": 0.85,
                "risk_level": "medium",
                "action_required": True,
                "gaps_identified": [
                    {
                        "requirement_id": "fca-cobs-9-2-1",
                        "gap_description": "Missing suitability assessment documentation",
                        "severity": "high",
                        "recommended_action": "Implement comprehensive suitability assessment process",
                        "deadline": "2024-03-01T00:00:00Z"
                    }
                ],
                "fca_sourcebooks_referenced": ["cobs", "sysc"],
                "recommendations": [
                    "Update investment advice procedures",
                    "Enhance client suitability documentation"
                ],
                "regulatory_impact": "Medium impact - requires policy updates and staff training",
                "escalation_required": False
            }
        }
    
    @validator('requirements')
    def validate_requirements_not_empty_when_gaps_exist(cls, v, values):
        """Ensure requirements are provided when gaps are identified."""
        if 'gaps_identified' in values and values['gaps_identified'] and not v:
            raise ValueError("Requirements must be provided when compliance gaps are identified")
        return v