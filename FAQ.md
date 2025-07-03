# FSCompliance Frequently Asked Questions (FAQ)

**The first MCP-integrated compliance platform for financial services**

> **📝 Note:** This FAQ covers both current capabilities and planned functionality. For current implementation status, see [Tasks.md](Tasks.md).

---

## 🎯 Strategic Positioning

### What is FSCompliance?

**FSCompliance is the universal compliance intelligence platform for financial services** - the first platform designed specifically to make regulatory intelligence accessible to any AI agent through Model Context Protocol (MCP) integration.

We transform compliance from a regulatory burden into intelligence that scales with AI adoption, ultimately serving the goal of **making it easier to bring the right financial products safely to consumers**.

### What does "universal compliance platform" mean?

FSCompliance rapidly ingests any well-articulated **Standard**, using the term broadly to include:

- **Regulatory frameworks** (FCA Handbook, SEC rules, MiFID II, Basel III)
- **Industry codes** (conduct codes, best practice guidelines)  
- **Statutory requirements** (legislation, acts, laws)
- **International standards** (IFRS, SOX, ISO standards)
- **Jurisdictional regulations** (state, provincial, national requirements)

The **FCA Handbook is our first Standard** - a proof of concept. Other standards will be rapidly processed using the architectures established through the FCA Handbook implementation.

### How is FSCompliance different from other compliance tools?

FSCompliance is positioned as the **first MCP-integrated compliance platform**, providing unique advantages:

- **AI-Agent Native Design**: Built specifically for AI agent interaction, not retrofitted human interfaces
- **Open-Source Transparency**: Auditable compliance logic with enterprise-grade security
- **Universal Standards Engine**: Rapid expansion beyond single regulatory frameworks
- **LLM Independence**: Works with any enterprise AI agent regardless of underlying LLM choice

---

## 💡 Value Proposition

### How does FSCompliance "slice through red tape"?

Compliance with legal, regulatory, and industry requirements has become unmanageable for Financial Services. Requirements proliferate at an accelerating rate, creating mounting friction, costs, and reduced agility that holds back financial innovation.

**FSCompliance slices through this red tape by:**

- **Enabling senior executive focus**: Compliance professionals carry out strategic tasks rather than getting entangled in regulatory small-print
- **Transforming regulatory burden**: Converting compliance obstacles into intelligence that enables better decision-making
- **Scaling human effectiveness**: AI handles analysis complexity while humans retain final responsibility and decision authority
- **Accelerating innovation**: Making it easier to bring the right financial products safely to consumers

### What makes FSCompliance effective for compliance professionals?

FSCompliance **increases the effectiveness of compliance professionals** by:

- **Automated Analysis**: AI handles time-consuming document review and gap detection
- **Expert Intelligence**: Proven compliance analysis through Claude 3.5 Sonnet's regulatory expertise
- **Risk Prioritization**: Focus effort on highest-impact compliance issues
- **Audit Preparation**: Comprehensive evidence collection and compliance documentation
- **Regulatory Monitoring**: Real-time tracking of regulatory changes and impact assessment

**Humans retain complete responsibility** for all compliance decisions - FSCompliance provides intelligence to support professional decision-making, never to replace it.

### How does this benefit consumers?

All FSCompliance capabilities **ultimately serve the goal of making it easier to bring the right financial products safely to consumers**. By reducing regulatory friction and enabling faster, more confident compliance decisions, financial institutions can:

- **Innovate responsibly**: Develop new products with confidence in regulatory compliance
- **Reduce costs**: Lower compliance burden translates to better consumer pricing
- **Improve access**: Streamlined compliance enables broader product availability
- **Enhance protection**: Better compliance intelligence means stronger consumer safeguards

---

## 🤖 AI Agent Integration

### How do AI agents use FSCompliance?

AI agents discover and use FSCompliance through the standard **Model Context Protocol (MCP)** workflow:

1. **Resource Declaration**: AI agent configured to include FSCompliance MCP server as available resource
2. **Tool Discovery**: AI agent reads FSCompliance JSON manifest to understand available capabilities
3. **Intelligent Selection**: Based on user queries (e.g., "analyze this policy for FCA compliance"), AI agent chooses appropriate FSCompliance tools
4. **Tool Execution**: AI agent calls specific tools like `analyze_compliance`, `detect_gaps`, `extract_requirements`

**This happens automatically** - AI agents intelligently choose the right compliance tools without manual configuration.

### What MCP tools does FSCompliance provide?

**Current Tools (Phase 2 Complete):**
- `analyze_compliance` - Comprehensive policy analysis against FCA requirements
- `detect_gaps` - Identify specific compliance gaps in policies or procedures
- `extract_requirements` - Extract relevant regulatory requirements from FCA Handbook

**Phase 3 Tools (Implementation Priority):**
- `monitor_regulatory_changes` - Real-time tracking of FCA Handbook updates
- `score_compliance_risk` - Automated risk scoring for policies and activities
- `track_audit_evidence` - Automated collection and organization of compliance evidence
- `map_regulatory_relationships` - Visualize connections between regulations and business activities
- `validate_customer_scenarios` - Check customer scenarios against compliance requirements

**Advanced Tools (Future Phases):**
- `generate_compliance_reports` - Automated regulatory report generation
- `suggest_remediation` - AI-powered recommendations for addressing compliance gaps

### Can any AI agent use FSCompliance?

**Yes - FSCompliance works with any AI agent regardless of the enterprise's LLM choice.** This is a core architectural advantage:

- **Enterprise Flexibility**: Organizations can use Claude, ChatGPT, custom models, or any AI agent
- **No Vendor Lock-in**: FSCompliance MCP server operates independently from enterprise AI agent choices
- **Proven Intelligence**: FSCompliance uses Claude 3.5 Sonnet internally for optimal compliance accuracy
- **Zero Adoption Barriers**: No corporate LLM standardization decisions required

---

## 🏗️ Technical Architecture

### What technology architecture does FSCompliance use?

FSCompliance implements **layered architecture as the first MCP-integrated compliance platform**:

```
┌─────────────────────────────────────────┐
│     Any Enterprise AI Agent            │
│   (Claude, ChatGPT, Custom Models)     │
└─────────────────────────────────────────┘
                    │ MCP Protocol
┌─────────────────────────────────────────┐
│        FSCompliance MCP Server         │
│    (Runs Claude 3.5 Sonnet Default)    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      Compliance Intelligence Layer      │
│   Gap Detection • Risk Scoring • AI    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│     Knowledge Management Layer          │
│    LightRAG • FCA Handbook • Graph     │
└─────────────────────────────────────────┘
```

**Core Technologies:**
- **Python 3.11+** with FastAPI for MCP server implementation
- **LightRAG** for knowledge graph and regulatory document processing
- **Claude 3.5 Sonnet** as default LLM with multi-model support
- **PostgreSQL + Vector Storage** for data persistence
- **Docker & Kubernetes** for scalable deployment

### Why Claude 3.5 Sonnet as default?

Claude 3.5 Sonnet went through **an extreme and extensive testing process in the development of the MCP itself, in ways that are highly analogous to the way that the MCP will actually be used**. This amounted to a profound real-world validation, which furthermore is ongoing as the MCP's capabilities expand. This validation embraces complex regulatory analysis, multi-document synthesis, construction of tools that meet Compliance professionals' actual needs, and technical architecture tasks.

**Key advantages:**
- **Proven compliance accuracy** through extensive FSCompliance development validation
- **Enterprise data security** with Anthropic's Zero Data Retention and HIPAA-eligible services
- **Professional-grade output** consistently suitable for regulatory scrutiny
- **Cost-effectiveness** for compliance use cases where accuracy is critical

**Multi-model support** available: LLaMA 3, Mistral Medium, custom models, with intelligent routing based on complexity and requirements.

### Can FSCompliance be self-hosted?

**Yes - FSCompliance is designed for flexible deployment:**

**Self-Hosted Benefits:**
- **Complete data control** - Sensitive financial data never leaves your infrastructure
- **Regulatory compliance** - Easier to meet data residency and audit requirements
- **Custom security** - Apply organization-specific security policies
- **Cost predictability** - Infrastructure costs vs usage-based API pricing

**Cloud Deployment Benefits:**
- **Instant setup** - Ready to use without infrastructure planning
- **Automatic updates** - Latest features and regulatory updates
- **Enterprise support** - Professional assistance and monitoring
- **Proven reliability** - Managed infrastructure and scaling

**Recommendation**: Start with cloud for evaluation, move to self-hosted for production if you have high volume usage, strict data privacy requirements, and technical expertise.

### How does FSCompliance ensure quality and accuracy?

**Multi-Layer Quality Assurance:**

- **Ground Truth Validation**: Extensive corpus of expert-validated compliance scenarios and regulatory interpretations
- **Confidence Scoring**: Every response includes confidence levels with uncertainty quantification
- **Expert Review**: Regular validation by certified compliance professionals
- **Continuous Monitoring**: Real-time accuracy tracking and performance optimization
- **Human-in-the-Loop**: Automatic flagging of high-stakes decisions for expert review

**Accuracy Targets**: >95% for compliance requirement identification, >97% accuracy against ground truth dataset.

---

## 🔒 Security & Privacy

### How secure is FSCompliance for financial data?

FSCompliance implements **enterprise-grade security** designed for financial services:

**Data Protection:**
- **Zero Data Retention** options with automatic deletion capabilities
- **Encryption** - TLS 1.3 for transit, AES-256 for data at rest
- **Access Controls** - OAuth 2.1 with role-based permissions
- **Audit Logging** - Complete tracking of all compliance decisions

**Compliance Certifications:**
- **SOC 2 Type II** for hosted services
- **GDPR compliance** with data anonymization
- **Financial industry standards** (PCI DSS compatible)

**Privacy-First Design:**
- **Opt-in memory** - Users choose what information to retain
- **Data anonymization** - Automatic PII detection and protection
- **Local processing** - Self-hosted options for complete data sovereignty

### What about AI safety and guardrails?

FSCompliance implements comprehensive **AI safety measures**:

**Input Guardrails:**
- PII detection and filtering
- Query validation and authorization checks
- Rate limiting and abuse protection

**Output Guardrails:**
- Confidence thresholds with automatic flagging
- Professional disclaimer injection
- Human review escalation for high-stakes decisions
- Bias detection and monitoring

**Continuous Monitoring:**
- Real-time output quality assessment
- Escalation procedures for concerning outputs
- Regular safety framework updates

---

## 💼 Business & Deployment

### Should we self-host or use the hosted service?

**Choose Self-Hosted if:**
- High volume usage (>1M compliance queries/month)
- Strict data residency requirements
- Custom security policies required
- Technical expertise available for deployment

**Choose Hosted Service if:**
- Want instant setup without infrastructure planning
- Prefer professional support and automatic updates
- Small to medium usage volumes
- Focus on core business vs infrastructure management

**Hybrid Approach:**
- Sensitive analysis self-hosted, general queries via hosted service
- Gradual migration from hosted to self-hosted as usage grows

### What are the costs involved?

**Hosted Service:** Usage-based pricing ~£0.09 per complex query (cost negligible compared to compliance officer time at £150-500+ per hour)

**Self-Hosted:** Infrastructure costs vary by scale:
- **Small deployment**: 8+ cores, 32GB RAM, optional GPU (~£2,000-5,000 setup)
- **Production deployment**: 16+ cores, 64GB RAM, A100/H100 GPU (~£10,000-25,000 setup)
- **Enterprise deployment**: Multi-GPU clusters (~£50,000+ setup)

**ROI Consideration**: Compliance professional time is extremely valuable - the cost differential between LLM options is negligible compared to compliance failure risks or professional time savings.

### How do we get started?

**Quick Start:**
1. **Evaluation**: Start with hosted service for immediate access
2. **Integration**: Configure your AI agents to use FSCompliance MCP tools
3. **Pilot**: Run compliance analysis on sample policies
4. **Scale**: Expand usage based on results and requirements
5. **Deploy**: Consider self-hosted option for production if needed

**Support Available:**
- Comprehensive documentation and integration guides
- Professional services for enterprise deployment
- Community support for open-source implementation

---

## 🔮 Future & Expansion

### How will FSCompliance expand beyond FCA?

The **universal Standards engine** rapidly ingests any well-articulated Standard:

**Near-term expansion** (6-18 months):
- **MiFID II** (EU investment services)
- **SEC Rules** (US securities)
- **Basel III** (International banking)
- **CFTC Regulations** (Derivatives)

**Global frameworks**:
- **Asia-Pacific**: MAS Guidelines, JFSA Regulations, APRA Standards
- **Specialized areas**: AML/CTF, ESG Reporting, Cybersecurity

Each new Standard integration makes it easier to bring innovative financial products safely to consumers by reducing regulatory friction across multiple jurisdictions.

### How can organizations contribute?

FSCompliance embraces **community collaboration**:

- **Regulatory expertise**: Share interpretations and compliance strategies
- **Technical contributions**: Framework parsers and validation datasets
- **Expert review**: Validate AI outputs and compliance analysis
- **Open-source development**: Contribute code and improvements

**Recognition program** provides contributor status, priority support, and partnership opportunities for significant contributions.

---

## 👥 Human Oversight & Professional Responsibility

### What role do humans play in FSCompliance recommendations?

**FSCompliance provides compliance intelligence to support professional decision-making, never to replace it.**

**Human Requirements:**
- **Complete accountability** - Users retain full responsibility for all compliance decisions
- **Expert review required** - All AI recommendations require validation by qualified compliance professionals
- **Professional judgment essential** - Business context and implementation decisions remain human responsibility
- **Documentation required** - Human review and decision rationale must be documented

**AI as Decision Support:**
- FSCompliance handles complex analysis so professionals can focus on strategic decisions
- AI provides intelligence and recommendations, humans make final compliance choices
- Professional validation required before implementing any AI recommendations
- Complex interpretations require qualified legal and compliance professional consultation

### What disclaimers and limitations apply?

**Professional Responsibility:**
- No warranty for accuracy - professional validation required for all decisions
- Users must ensure adherence to all applicable regulations and current guidance
- FSCompliance analysis represents point-in-time assessment based on available data
- Regulatory requirements continuously evolve - maintain current awareness essential

**Technology Platform Disclaimer:**
FSCompliance is a technology platform providing AI-powered compliance analysis tools. It does not provide legal or regulatory advice. Our expertise lies specifically in the intersection of AI technology and compliance processes. Users remain responsible for compliance decisions and should consult qualified professionals for definitive guidance.

---

## About This Document

**Author**: Blake Dempster, Founder & Principal Architect  
**Co-Authored by**: Claude Code (claude.ai/code)  
**Created**: 25 December 2024  
**Last Updated**: 3 July 2025  
**Purpose**: Comprehensive FAQ for FSCompliance - the first MCP-integrated compliance platform for financial services, organized around core project touchstones.

*This FAQ provides essential information about FSCompliance's mission to slice through regulatory red tape and make it easier to bring the right financial products safely to consumers through AI-powered compliance intelligence.*

---