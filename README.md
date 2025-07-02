# FSCompliance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![Claude 3.5 Sonnet](https://img.shields.io/badge/LLM-Claude%203.5%20Sonnet-blue.svg)](https://www.anthropic.com/claude)

**The open-source universal compliance intelligence platform built exclusively for financial services.** We rapidly ingest ANY well-articulated Standard to make compliance intelligence accessible to any AI agent through innovative MCP integration - slicing through red tape to enable compliance professionals to carry out senior executive tasks rather than get bogged down in small-print.

## 🎯 Overview

**FSCompliance is positioned as the first MCP-integrated universal compliance platform** for financial services. Unlike many FS AI approaches that treat AI as a chatbot add-on to existing processes, FSCompliance represents fundamental transformation - making it easier to bring the right product safely to consumers by enabling compliance capabilities that were not achievable before.

### Universal Standards Engine

In principle FSCompliance is designed to rapidly ingest many well-articulated Standards. We are using the word "Standard" widely, to include:
- **Regulatory frameworks** (FCA Handbook, SEC rules, MiFID II, Basel III)
- **Industry codes** (conduct codes, best practice guidelines)  
- **Statutory requirements** (legislation, acts, laws)
- **International standards** (IFRS, SOX, ISO standards)
- **Jurisdictional regulations** (state, provincial, national requirements)

Using the architectures developed for our first ingested Standard, the FCA Handbook, which serves as a proof of concept, we plan to expand to other Standards. Our goal is to develop an AI-enabled universal compliance intelligence that addresses many limitations of traditional RegTech approaches.

### Core Capabilities (Starting with the FCA Handbook as our first ingested Standard)

**Primary Use Cases (Phase 2 Complete):**
- **Compliance Gap Analysis**: AI agents analyze policies to identify salient requirements and flag compliance gaps
- **Regulatory Intelligence**: Extract relevant requirements from FCA Handbook for specific analysis
- **Compliance Assessment**: Comprehensive policy analysis with confidence scoring

**Advanced Use Cases (Phase 3 Implementation):**
- **Regulatory Change Monitoring**: Track FCA Handbook updates and assess impact
- **Risk-Based Compliance**: Score compliance risk with prioritization recommendations
- **Audit Trail Management**: Collect and organize compliance evidence for examinations
- **Regulatory Relationship Mapping**: Visualize connections between regulations and business activities
- **Customer Scenario Validation**: Real-time compliance checking (e.g., "For customers aged 60+ holding Bitcoin, did risk warnings meet FCA requirements?")


**📈 Project Status**: Phase 2 Complete - Core Intelligence & LLM Abstraction Layer implemented. **Claude 3.5 Sonnet selected as default LLM** based on extensive real-world validation through comprehensive FSCompliance development.

## 🏗️ Architecture

**FSCompliance follows a layered architecture as the first MCP-integrated compliance platform:**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agents     │    │   FSCompliance   │    │   Knowledge     │
│  (MCP Clients)  │◄──►│   MCP Server     │◄──►│     Store       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌──────────────┐         ┌─────────────┐
                       │ LLM Gateway  │         │  LightRAG   │
                       │(Claude 3.5)  │         │  Engine     │
                       └──────────────┘         └─────────────┘
```

### Core Layers
1. **MCP Server Layer**: Protocol-compliant JSON-RPC 2.0 server
2. **Knowledge Management Layer**: LightRAG-powered FCA Handbook processing with advanced RAG capabilities
3. **Compliance Intelligence Layer**: AI-powered requirement analysis and gap detection
4. **Memory and Learning Layer**: Long-term memory with privacy controls
5. **LLM Abstraction Layer**: Claude 3.5 Sonnet default with multi-model support (LLaMA 3, Falcon, Mistral Medium)

### Strategic Architecture Decisions
- **LLM Strategy**: Claude 3.5 Sonnet selected as default based on extensive real-world validation (see [LLMChoice.md](LLMChoice.md))
- **No Fine-Tuning**: Deliberate decision to use standard LLMs + advanced RAG for optimal flexibility and regulatory responsiveness
- **Database Evolution**: Migrating to Supabase (PostgreSQL + PGVector) for simplified architecture and real-time capabilities
- **MCP Tool Priority**: 8 priority tools identified for Phase 3 implementation (see [ComplianceTools.md](ComplianceTools.md))

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fscompliance.git
   cd fscompliance
   ```

2. **Install Poetry** (if not already installed)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Activate virtual environment**
   ```bash
   poetry shell
   ```

5. **Start the MCP server**
   ```bash
   poetry run python -m fscompliance.server
   ```

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint code
poetry run ruff check .
```

## 📖 Usage

### Basic Compliance Query

```python
from fscompliance import FSComplianceClient

client = FSComplianceClient()

# Analyze policy for compliance gaps
response = client.analyze_policy(
    policy_text="Our investment advisory process...",
    query_type="gap_analysis"
)

print(f"Identified {len(response.gaps_identified)} compliance gaps")
for gap in response.gaps_identified:
    print(f"- {gap.description} (FCA: {gap.regulation_reference})")
```

### Customer Risk Assessment

```python
# Validate customer scenario
response = client.assess_customer_scenario(
    scenario="60+ year old customers holding Bitcoin investments",
    requirements=["risk_warnings", "suitability_assessment"]
)

print(f"Compliance Status: {response.compliance_status}")
print(f"Required Actions: {response.recommendations}")
```

### MCP Integration

FSCompliance implements the Model Context Protocol, allowing AI agents to interact via standard MCP methods:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "analyze_compliance",
    "arguments": {
      "policy_text": "Investment advisory procedures...",
      "analysis_type": "gap_analysis"
    }
  }
}
```

## 👥 Target Users

- **Compliance Officers**: Policy analysis and gap identification
- **Risk Managers**: Customer scenario validation and risk assessment
- **Regulatory Inspectors**: Automated compliance checking and reporting
- **Professional Advisers**: Regulatory guidance and requirement clarification

> **Important Note**: FSCompliance is a technology platform that provides AI-powered compliance analysis tools. It does not provide legal or regulatory advice. Our expertise lies specifically in the intersection of AI technology and compliance processes, making regulatory intelligence accessible to AI systems. Users remain responsible for compliance decisions and should consult qualified legal and regulatory professionals for definitive guidance.

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
FSCOMPLIANCE_DEFAULT_LLM=claude-3.5-sonnet
FSCOMPLIANCE_CLAUDE_API_KEY=your_anthropic_api_key
FSCOMPLIANCE_ENABLE_MULTI_MODEL=true

# Database Configuration (Supabase Migration Planned)
FSCOMPLIANCE_DB_URL=postgresql://user:pass@localhost/fscompliance
FSCOMPLIANCE_VECTOR_STORE=pgvector

# Privacy Settings
FSCOMPLIANCE_MEMORY_ENABLED=true
FSCOMPLIANCE_ANONYMIZE_DATA=true
FSCOMPLIANCE_AUDIT_LOGGING=true

# MCP Server Settings
FSCOMPLIANCE_MCP_PORT=8000
FSCOMPLIANCE_MCP_HOST=localhost
```

### Privacy Controls

FSCompliance includes comprehensive privacy controls:

- **Memory Management**: Enable/disable long-term learning
- **Data Anonymization**: Automatic PII detection and masking
- **Local Processing**: On-premises deployment options
- **Audit Logging**: Complete compliance decision tracking

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=fscompliance

# Run specific test categories
poetry run pytest tests/compliance/
poetry run pytest tests/mcp/
poetry run pytest -m "not integration"
```

## 📚 Documentation

### Strategic & Planning Documents
- **[Planning.md](Planning.md)**: Complete project architecture, goals, and strategic direction
- **[Rules.md](Rules.md)**: Development guidelines, coding standards, and project conventions
- **[Tasks.md](Tasks.md)**: Development roadmap organized by phases with completion tracking
- **[LLMChoice.md](LLMChoice.md)**: LLM selection strategy and Claude 3.5 Sonnet decision rationale
- **[ComplianceTools.md](ComplianceTools.md)**: Strategic market analysis and comprehensive MCP tool roadmap
- **[CLAUDE.md](CLAUDE.md)**: Project context and guidance for AI-assisted development

### Documentation & Brand Materials
- **[FAQ.md](FAQ.md)**: User-facing project information and comprehensive capability descriptions
- **[internal/Brand.md](internal/Brand.md)**: Brand positioning, competitive differentiation, and market strategy
- **[internal/TakeToMarket.md](internal/TakeToMarket.md)**: Go-to-market strategy with "slice through red tape" messaging
- **[internal/FCAsandbox.md](internal/FCAsandbox.md)**: FCA Sandbox application strategy and regulatory validation timeline
- **[internal/UserInterface.md](internal/UserInterface.md)**: UI/UX design specifications and presentation prototypes

### API Documentation

Once running, visit:
- **OpenAPI Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **MCP Schema**: http://localhost:8000/mcp/schema

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Check [Tasks.md](Tasks.md)** for current development priorities
2. **Follow [Rules.md](Rules.md)** for coding standards and conventions
3. **Create comprehensive tests** for all new features
4. **Update documentation** when adding features or changing APIs

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes following Rules.md guidelines
# Add/update tests
poetry run pytest

# Format and lint
poetry run black .
poetry run ruff check .

# Update Tasks.md with completed work
# Submit pull request
```

## 🔒 Security

FSCompliance handles sensitive financial data and implements:

- **OAuth 2.1 Authentication**: Secure API access
- **TLS Encryption**: All communications encrypted
- **Input Validation**: Comprehensive data validation using Pydantic
- **Audit Logging**: Complete access and decision logging
- **Role-Based Access**: Granular permission system

### Security Reporting

Please report security vulnerabilities privately to: security@fscompliance.org

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

**Architecture & Vision:**
- **Blake Dempster, Founder & Principal Architect**: Strategic vision, expertise in regulatory-AI intersection, and technical architecture leadership

**Core Technologies:**
- **[Model Context Protocol](https://modelcontextprotocol.io/)**: Standard protocol specification
- **[LightRAG](https://github.com/HKUDS/LightRAG)**: Knowledge retrieval and graph processing
- **[FCA Handbook](https://www.handbook.fca.org.uk/)**: UK Financial Conduct Authority regulatory framework
- **[Pydantic](https://pydantic.dev/)**: Data validation and serialization

## 📞 Support

- **Documentation**: [Full documentation](https://fscompliance.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/fscompliance/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/fscompliance/discussions)
- **Email**: support@fscompliance.org

---

## About This Project

**Author**: Blake Dempster, Founder & Principal Architect  
**Co-Authored by**: Claude Code (claude.ai/code)  
**Created**: 2024-12-25  
**Last Updated**: 2024-12-25  

**FSCompliance** - The first MCP-integrated compliance platform for financial services. Slicing through red tape to make it easier to bring the right product safely to consumers.