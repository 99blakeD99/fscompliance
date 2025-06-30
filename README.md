# FSCompliance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

An open-source Model Context Protocol (MCP) service for financial services companies and institutions to manage compliance with regulatory "Conduct Requirements".

## 🎯 Overview

FSCompliance leverages AI-powered knowledge retrieval and graph processing to help financial organizations:

- **Identify compliance requirements** from regulatory frameworks
- **Detect compliance gaps** in policies and procedures  
- **Validate customer scenarios** against regulatory requirements
- **Generate regulatory reports** for inspections and audits

Initially focused on the [FCA Handbook](https://www.handbook.fca.org.uk/) with architecture designed to support additional regulatory frameworks.

**📈 Project Status**: Phase 2 Complete - Core Intelligence & LLM Abstraction Layer implemented with 15 major components including multi-model AI support, cost optimization, and enterprise-grade caching.

## 🏗️ Architecture

FSCompliance follows a layered MCP-compliant architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agents     │    │   FSCompliance   │    │   Knowledge     │
│  (MCP Clients)  │◄──►│   MCP Server     │◄──►│     Store       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌──────────────┐         ┌─────────────┐
                       │ LLM Gateway  │         │  LightRAG   │
                       │ (Multi-Model)│         │  Engine     │
                       └──────────────┘         └─────────────┘
```

### Core Layers
1. **MCP Server Layer**: Protocol-compliant JSON-RPC 2.0 server
2. **Knowledge Management Layer**: LightRAG-powered regulatory document processing
3. **Compliance Intelligence Layer**: AI-powered requirement analysis and gap detection
4. **Memory and Learning Layer**: Long-term memory with privacy controls
5. **LLM Abstraction Layer**: Multi-model support (LLaMA 3, Falcon, Mistral Medium)

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

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
FSCOMPLIANCE_DEFAULT_LLM=llama3
FSCOMPLIANCE_LLM_API_KEY=your_api_key

# Database Configuration
FSCOMPLIANCE_DB_URL=sqlite:///fscompliance.db

# Privacy Settings
FSCOMPLIANCE_MEMORY_ENABLED=true
FSCOMPLIANCE_ANONYMIZE_DATA=true

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

- **[Planning.md](Planning.md)**: Complete project architecture and specifications
- **[Rules.md](Rules.md)**: Development guidelines and coding standards
- **[Tasks.md](Tasks.md)**: Development roadmap and task tracking
- **[CLAUDE.md](CLAUDE.md)**: AI assistant guidance for development

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
- **Blake Dempster, Founder & Principal Architect**: Strategic vision, regulatory domain expertise, and technical architecture leadership

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

**FSCompliance** - Making financial regulatory compliance intelligent and accessible.