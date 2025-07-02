# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FSCompliance** is the open-source compliance intelligence platform built exclusively for financial services. We make regulatory intelligence accessible to any AI agent through innovative MCP integration while increasing the effectiveness of compliance professionals. Built as a universal Standards engine with the FCA Handbook as our first ingested Standard, FSCompliance rapidly ingests any well-articulated Standard, broadly defined to include regulatory frameworks, industry codes, statutory requirements, international standards, and jurisdictional regulations. FSCompliance is positioned as the first MCP-integrated compliance platform.

## Key Project Files

### Strategic & Planning Documents
- **`Planning.md`**: Complete project architecture, goals, technical specifications, and development roadmap
- **`Rules.md`**: Development guidelines, coding standards, and project-specific conventions
- **`Tasks.md`**: Current and completed development tasks organized by development phases
- **`ComplianceTools.md`**: Strategic market analysis and comprehensive MCP tool roadmap
- **`LLMChoice.md`**: LLM selection strategy and Claude 3.5 Sonnet decision rationale

### Documentation & Brand Materials
- **`FAQ.md`**: User-facing project information and comprehensive capability descriptions
- **`internal/Brand.md`**: Brand positioning, competitive differentiation, and market strategy
- **`internal/DatabaseStrategy.md`**: Database architecture evaluation and migration planning
- **`internal/FCAsandbox.md`**: FCA Sandbox application strategy and regulatory validation timeline
- **`internal/UserInterface.md`**: UI/UX design specifications and presentation prototypes

## Development Commands

### Setup
```bash
# Install Poetry (dependency management)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Development
```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=fscompliance
```

### MCP Server
```bash
# Start MCP server
poetry run python -m fscompliance.server

# Test MCP server
poetry run python -m fscompliance.test_client
```

## Architecture

FSCompliance follows a layered architecture as the first MCP-integrated compliance platform:

1. **MCP Server Layer**: Protocol-compliant JSON-RPC 2.0 server
2. **Knowledge Management Layer**: LightRAG-powered FCA Handbook processing
3. **Compliance Intelligence Layer**: AI-powered requirement analysis and gap detection
4. **Memory and Learning Layer**: Long-term memory with privacy controls
5. **LLM Abstraction Layer**: Multi-model support with Claude 3.5 Sonnet default (LLaMA 3, Falcon, Mistral Medium alternatives)

### Strategic Architecture Decisions
- **LLM Strategy**: Claude 3.5 Sonnet selected as default based on extensive real-world validation through FSCompliance development; no fine-tuning architectural decision per `LLMChoice.md`
- **Database Evolution**: Migrating to Supabase (PostgreSQL + PGVector) per `DatabaseStrategy.md` for simplified architecture and real-time capabilities
- **MCP Tool Priority**: 8 priority tools identified in `ComplianceTools.md` for Phase 3 implementation
- **Brand Positioning**: Positioned as first MCP-integrated compliance platform per `Brand.md` competitive analysis
- **UI/UX Design**: Professional financial services interface specifications detailed in `UserInterface.md`

### Key Technologies
- **Python 3.11+** with Poetry for dependency management
- **Pydantic v2** for data validation and serialization
- **LightRAG** for knowledge retrieval and graph processing
- **FastAPI** for web framework and MCP server implementation
- **Supabase** for unified database with real-time capabilities (migration planned Q3 2025)
- **OAuth 2.1** for authentication and security

## Universal Standards Definition

FSCompliance uses "Standard" widely to include:
- **Regulatory frameworks** (FCA Handbook, SEC rules, MiFID II, Basel III)
- **Industry codes** (conduct codes, best practice guidelines)  
- **Statutory requirements** (legislation, acts, laws)
- **International standards** (IFRS, SOX, ISO standards)
- **Jurisdictional regulations** (state, provincial, national requirements)

This universal approach enables FSCompliance to serve as a comprehensive compliance intelligence platform across multiple regulatory contexts.

## Development Guidelines

**Always consult `Rules.md` before starting development work.** Key principles:

- Follow layered architecture from `Planning.md`
- Maintain MCP protocol compliance
- Create comprehensive Pytest unit tests
- Never exceed 500 lines per file
- Include regulatory source citations in compliance logic
- Implement privacy controls for all memory features
- Validate all inputs, especially financial/customer data

## Target Users

- Compliance Officers
- Risk Managers  
- Regulatory Inspectors
- Professional Advisers

## Use Cases

Based on ComplianceTools.md analysis and current MCP tool roadmap:

### Primary Use Cases (Phase 2 Complete)
1. **Compliance Gap Analysis**: AI agents analyze policies to identify the ten most salient requirements and flag compliance gaps using `analyze_compliance` and `detect_gaps` tools
2. **Regulatory Intelligence**: AI agents extract relevant requirements from FCA Handbook using `extract_requirements` for specific policy or scenario analysis
3. **Compliance Assessment**: AI agents provide comprehensive policy analysis against regulatory requirements with confidence scoring

### Advanced Use Cases (Phase 3 Implementation)
4. **Regulatory Change Monitoring**: AI agents track FCA Handbook updates and assess impact on existing compliance frameworks using `monitor_regulatory_changes`
5. **Risk-Based Compliance**: AI agents score compliance risk across policies and business activities using `score_compliance_risk` with prioritization recommendations  
6. **Audit Trail Management**: AI agents collect and organize compliance evidence for regulatory examinations using `track_audit_evidence`
7. **Regulatory Relationship Mapping**: AI agents visualize connections between regulations and business activities using `map_regulatory_relationships`
8. **Customer Scenario Validation**: AI agents validate customer scenarios against FCA requirements (e.g., "For customers aged 60+ holding Bitcoin, did the risk warnings meet FCA requirements?") using `validate_customer_scenarios`

### Future Use Cases (Phase 4+)
9. **Automated Regulatory Reporting**: AI agents generate draft regulatory reports and submissions using `generate_compliance_reports`
10. **Remediation Planning**: AI agents suggest specific compliance remediation actions using `suggest_remediation`