"""Unit tests for FSCompliance data models."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID

from fscompliance.models import (
    ComplianceGap,
    ComplianceQuery,
    ComplianceResponse,
    ComplianceStatus,
    ConductRequirement,
    FCAFirmType,
    FCASourcebook,
    QueryType,
    RegulatoryFramework,
    RequirementType,
    RiskLevel,
    SeverityLevel,
    UserRole,
)


class TestConductRequirement:
    """Test cases for ConductRequirement model."""
    
    def test_conduct_requirement_creation(self):
        """Test creating a valid ConductRequirement."""
        requirement = ConductRequirement(
            source="FCA Handbook",
            sourcebook=FCASourcebook.SYSC,
            section="SYSC.4.1.1",
            chapter="4",
            title="General organisational requirements",
            content="A firm must have robust governance arrangements...",
            requirement_type=RequirementType.GOVERNANCE,
            severity=SeverityLevel.HIGH,
            applicability=["investment_firm", "bank"],
            last_updated=datetime.utcnow(),
            effective_date=datetime.utcnow(),
            application_scope=[FCAFirmType.INVESTMENT_FIRM, FCAFirmType.BANK]
        )
        
        assert requirement.framework == RegulatoryFramework.FCA_HANDBOOK
        assert requirement.sourcebook == FCASourcebook.SYSC
        assert requirement.section == "SYSC.4.1.1"
        assert requirement.requirement_type == RequirementType.GOVERNANCE
        assert requirement.severity == SeverityLevel.HIGH
        assert len(requirement.application_scope) == 2
        assert isinstance(UUID(requirement.id), UUID)
    
    def test_conduct_requirement_section_validation(self):
        """Test section reference validation."""
        with pytest.raises(ValueError, match="Section reference must start with sourcebook"):
            ConductRequirement(
                source="FCA Handbook",
                sourcebook=FCASourcebook.SYSC,
                section="COBS.4.1.1",  # Wrong sourcebook in section
                chapter="4",
                title="Test requirement",
                content="Test content",
                requirement_type=RequirementType.GOVERNANCE,
                severity=SeverityLevel.HIGH,
                last_updated=datetime.utcnow(),
                effective_date=datetime.utcnow()
            )
    
    def test_conduct_requirement_empty_content_validation(self):
        """Test that empty content raises validation error."""
        with pytest.raises(ValueError, match="Requirement content cannot be empty"):
            ConductRequirement(
                source="FCA Handbook",
                sourcebook=FCASourcebook.SYSC,
                section="SYSC.4.1.1",
                chapter="4",
                title="Test requirement",
                content="",  # Empty content
                requirement_type=RequirementType.GOVERNANCE,
                severity=SeverityLevel.HIGH,
                last_updated=datetime.utcnow(),
                effective_date=datetime.utcnow()
            )
    
    def test_record_keeping_period_validation(self):
        """Test record keeping period validation."""
        # Valid period
        requirement = ConductRequirement(
            source="FCA Handbook",
            sourcebook=FCASourcebook.SYSC,
            section="SYSC.4.1.1",
            chapter="4",
            title="Test requirement",
            content="Test content",
            requirement_type=RequirementType.RECORD_KEEPING,
            severity=SeverityLevel.MEDIUM,
            last_updated=datetime.utcnow(),
            effective_date=datetime.utcnow(),
            record_keeping_period=5
        )
        assert requirement.record_keeping_period == 5
        
        # Invalid period (too high)
        with pytest.raises(ValueError, match="Record keeping period must be between 1 and 50 years"):
            ConductRequirement(
                source="FCA Handbook",
                sourcebook=FCASourcebook.SYSC,
                section="SYSC.4.1.1",
                chapter="4",
                title="Test requirement",
                content="Test content",
                requirement_type=RequirementType.RECORD_KEEPING,
                severity=SeverityLevel.MEDIUM,
                last_updated=datetime.utcnow(),
                effective_date=datetime.utcnow(),
                record_keeping_period=100  # Too high
            )


class TestComplianceQuery:
    """Test cases for ComplianceQuery model."""
    
    def test_compliance_query_creation(self):
        """Test creating a valid ComplianceQuery."""
        query = ComplianceQuery(
            user_role=UserRole.COMPLIANCE_OFFICER,
            query_type=QueryType.GAP_ANALYSIS,
            content="What are the conduct requirements for investment advice?",
            fca_firm_type=FCAFirmType.INVESTMENT_FIRM,
            sourcebooks_in_scope=[FCASourcebook.COBS, FCASourcebook.SYSC],
            customer_facing=True,
            privacy_mode=True
        )
        
        assert query.framework == RegulatoryFramework.FCA_HANDBOOK
        assert query.user_role == UserRole.COMPLIANCE_OFFICER
        assert query.query_type == QueryType.GAP_ANALYSIS
        assert query.fca_firm_type == FCAFirmType.INVESTMENT_FIRM
        assert len(query.sourcebooks_in_scope) == 2
        assert query.customer_facing is True
        assert query.privacy_mode is True
        assert isinstance(UUID(query.id), UUID)
    
    def test_query_empty_content_validation(self):
        """Test that empty query content raises validation error."""
        with pytest.raises(ValueError, match="Query content cannot be empty"):
            ComplianceQuery(
                user_role=UserRole.COMPLIANCE_OFFICER,
                query_type=QueryType.GAP_ANALYSIS,
                content="",  # Empty content
                fca_firm_type=FCAFirmType.INVESTMENT_FIRM
            )
    
    def test_query_default_values(self):
        """Test default values for ComplianceQuery."""
        query = ComplianceQuery(
            user_role=UserRole.COMPLIANCE_OFFICER,
            query_type=QueryType.GAP_ANALYSIS,
            content="Test query"
        )
        
        assert query.privacy_mode is True  # Default
        assert query.customer_facing is False  # Default
        assert query.high_risk_customer is False  # Default
        assert query.sourcebooks_in_scope == []  # Default empty list


class TestComplianceGap:
    """Test cases for ComplianceGap model."""
    
    def test_compliance_gap_creation(self):
        """Test creating a valid ComplianceGap."""
        gap = ComplianceGap(
            requirement_id="fca-cobs-9-2-1",
            gap_description="Missing suitability assessment documentation",
            severity=SeverityLevel.HIGH,
            recommended_action="Implement comprehensive suitability assessment process",
            deadline=datetime.utcnow() + timedelta(days=30)
        )
        
        assert gap.requirement_id == "fca-cobs-9-2-1"
        assert gap.severity == SeverityLevel.HIGH
        assert gap.deadline is not None
    
    def test_gap_empty_description_validation(self):
        """Test that empty gap description raises validation error."""
        with pytest.raises(ValueError, match="Gap description cannot be empty"):
            ComplianceGap(
                requirement_id="test-req-1",
                gap_description="",  # Empty description
                severity=SeverityLevel.MEDIUM,
                recommended_action="Test action"
            )


class TestComplianceResponse:
    """Test cases for ComplianceResponse model."""
    
    def test_compliance_response_creation(self):
        """Test creating a valid ComplianceResponse."""
        requirement = ConductRequirement(
            source="FCA Handbook",
            sourcebook=FCASourcebook.COBS,
            section="COBS.9.2.1",
            chapter="9",
            title="Suitability assessment",
            content="A firm must assess suitability...",
            requirement_type=RequirementType.CONDUCT,
            severity=SeverityLevel.HIGH,
            last_updated=datetime.utcnow(),
            effective_date=datetime.utcnow()
        )
        
        gap = ComplianceGap(
            requirement_id=requirement.id,
            gap_description="Missing suitability documentation",
            severity=SeverityLevel.HIGH,
            recommended_action="Implement documentation process"
        )
        
        response = ComplianceResponse(
            query_id="test-query-123",
            compliance_status=ComplianceStatus.PARTIAL_COMPLIANCE,
            confidence_score=0.85,
            risk_level=RiskLevel.MEDIUM,
            action_required=True,
            requirements=[requirement],
            gaps_identified=[gap],
            fca_sourcebooks_referenced=[FCASourcebook.COBS],
            recommendations=["Update procedures", "Train staff"],
            regulatory_impact="Medium impact requiring policy updates"
        )
        
        assert response.framework == RegulatoryFramework.FCA_HANDBOOK
        assert response.compliance_status == ComplianceStatus.PARTIAL_COMPLIANCE
        assert response.confidence_score == 0.85
        assert response.risk_level == RiskLevel.MEDIUM
        assert len(response.requirements) == 1
        assert len(response.gaps_identified) == 1
        assert len(response.fca_sourcebooks_referenced) == 1
        assert isinstance(UUID(response.id), UUID)
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid score
        response = ComplianceResponse(
            query_id="test-query",
            compliance_status=ComplianceStatus.COMPLIANT,
            confidence_score=0.95,
            risk_level=RiskLevel.LOW,
            action_required=False
        )
        assert response.confidence_score == 0.95
        
        # Invalid score (too high)
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            ComplianceResponse(
                query_id="test-query",
                compliance_status=ComplianceStatus.COMPLIANT,
                confidence_score=1.5,  # Too high
                risk_level=RiskLevel.LOW,
                action_required=False
            )
        
        # Invalid score (negative)
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            ComplianceResponse(
                query_id="test-query",
                compliance_status=ComplianceStatus.COMPLIANT,
                confidence_score=-0.1,  # Negative
                risk_level=RiskLevel.LOW,
                action_required=False
            )
    
    def test_requirements_validation_with_gaps(self):
        """Test that requirements are required when gaps exist."""
        gap = ComplianceGap(
            requirement_id="test-req",
            gap_description="Test gap",
            severity=SeverityLevel.MEDIUM,
            recommended_action="Test action"
        )
        
        with pytest.raises(ValueError, match="Requirements must be provided when compliance gaps are identified"):
            ComplianceResponse(
                query_id="test-query",
                compliance_status=ComplianceStatus.NON_COMPLIANT,
                confidence_score=0.8,
                risk_level=RiskLevel.HIGH,
                action_required=True,
                gaps_identified=[gap],
                requirements=[]  # Empty requirements with gaps
            )


class TestModelSerialization:
    """Test JSON serialization and deserialization of models."""
    
    def test_conduct_requirement_json_serialization(self):
        """Test ConductRequirement JSON serialization."""
        requirement = ConductRequirement(
            source="FCA Handbook",
            sourcebook=FCASourcebook.SYSC,
            section="SYSC.4.1.1",
            chapter="4",
            title="Test requirement",
            content="Test content",
            requirement_type=RequirementType.GOVERNANCE,
            severity=SeverityLevel.HIGH,
            last_updated=datetime.utcnow(),
            effective_date=datetime.utcnow()
        )
        
        # Test JSON serialization
        json_data = requirement.json()
        assert isinstance(json_data, str)
        
        # Test deserialization
        deserialized = ConductRequirement.parse_raw(json_data)
        assert deserialized.id == requirement.id
        assert deserialized.framework == requirement.framework
        assert deserialized.sourcebook == requirement.sourcebook
    
    def test_compliance_query_dict_conversion(self):
        """Test ComplianceQuery dict conversion."""
        query = ComplianceQuery(
            user_role=UserRole.COMPLIANCE_OFFICER,
            query_type=QueryType.RISK_ASSESSMENT,
            content="Test query content",
            fca_firm_type=FCAFirmType.BANK
        )
        
        # Test dict conversion
        query_dict = query.dict()
        assert isinstance(query_dict, dict)
        assert query_dict["user_role"] == "compliance_officer"
        assert query_dict["framework"] == "fca_handbook"
        assert query_dict["fca_firm_type"] == "bank"
        
        # Test creation from dict
        new_query = ComplianceQuery(**query_dict)
        assert new_query.id == query.id
        assert new_query.user_role == query.user_role