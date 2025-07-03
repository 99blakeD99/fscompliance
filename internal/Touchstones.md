# FSCompliance Project Touchstones

This document serves as a learning repository of important project and product underpinnings that provide coherence and consistency across all FSCompliance documentation and initiatives.

## Purpose

Touchstones.md serves to:
- **Give coherence to .md reviews** as outlined in ReviewRules.md
- **Provide source material** for marketing and communications
- **Serve as go-to background** when starting any new initiative

## Touchstone Quality Standards

When adding new touchstones, ensure they:
- **Are important** - Fundamental to project/product understanding
- **Are highly encapsulated** - Expressed concisely and precisely
- **Are reasonably distinct** - Not duplicating existing touchstones

---

## Core Touchstones

### Strategic Positioning
- The objective is to be the universal compliance intelligence platform in financial services
- Compliance refers to managing in accordance with identified Standards
- Standards refers to regulatory frameworks, industry codes, statutory requirements, international standards, jurisdictional regulations and any analogous sets of criteria which financial institutions have to meet
- FCA Handbook is the first Standard: a proof of concept
- Other standards will be rapidly processed using the architectures established through the FCA Handbook exercise

### Value Proposition  
- Using the MCP will achieve slicing through red tape
- Using the MCP will make it easier to bring the right product safely to consumers
- Using the MCP will enable compliance professionals to carry out senior executive tasks rather than get entangled in small-print
- Human effectiveness is transformationally enhanced
- Humans still have final responsibility, the MCP's role is to make that responsibility manageable
- All FSCompliance capabilities ultimately serve the goal of making it easier to bring the right financial products safely to consumers - transforming regulatory burden from a barrier to innovation into intelligence that enables better consumer outcomes

### Technical Architecture
- The MCP internally allows use of any LLM within the MCP
- The MCP recommends Claude as internal LLM, and adopts Claude by default, because of advanced domain capability and advanced data security
- The external LLM calling the MCP is a matter of user choice and has no bearing on use of the MCP
- The MCP is open-source and transparent for trust and auditability
- Architecture is self-hostable for maximum enterprise data control
- Complications of fine-tuning are bypassed through use of standard models and advanced RAG
- FSCompliance MCP server operates completely independently from whatever LLM the enterprise chooses for their AI agents - eliminating adoption barriers from corporate LLM standardization decisions while maintaining proven compliance intelligence through Claude 3.5 Sonnet default

### Product Strategy
- Model Context Protocol (MCP) integration makes the platform AI-agent native
- First MCP-integrated compliance platform positioning provides competitive differentiation
- Universal Standards engine approach enables rapid expansion beyond FCA Handbook
- Enterprise-grade data protection addresses financial services privacy requirements

### Project Management
- FSCompliance refers to the overall project, codebase, and development initiative
- The MCP refers to the product that enterprises deploy and use
- Documentation maintains professional tone suited to corporate financial services audience
- All strategic decisions are documented with clear rationale and cross-references
- FSCompliance provides compliance intelligence to support professional decision-making, never to replace it - users retain complete responsibility for all compliance decisions with expert review required for all AI recommendations

---

## About This Document

**Author**: Blake Dempster, Founder & Principal Architect  
**Co-Authored by**: Claude Code (claude.ai/code)  
**Created**: 3 July 2025  
**Purpose**: Central repository of fundamental project and product principles for consistency across FSCompliance documentation and strategic initiatives.

---