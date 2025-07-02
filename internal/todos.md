# FSCompliance Development Todos

*Comprehensive todo tracking for granular development work at a more detailed level than Tasks.md*

---

## Overview

This document tracks all todos identified during FSCompliance development, providing a granular view of work carried out beyond the phase-level tracking in Tasks.md. Todos capture specific improvements, refinements, and strategic decisions made during the project evolution.

---

## Completed Todos ✅

### Documentation & Content Strategy
- ✅ Add FAQ question about demo hosting options and why traditional web hosting (like one.com) won't work for FSCompliance
- ✅ Add FAQ question comparing FSCompliance's custom memory approach vs MCP-Mem0/OpenMemory solutions
- ✅ Review FAQ.md and add '(To be organised in due course)' notes to items requiring manual arrangement (like expert reviews, user acceptance testing) vs things Claude Code can implement or that happen automatically
- ✅ Add separate FAQ section/subsection on Human in the Loop requirements and add disclaimer about responsibility for correct results and their usage
- ✅ Add to FAQ that Claude uses Rules.md for development guidelines and coding standards
- ✅ Add FAQ section referencing GuardRails.md and explaining the comprehensive AI safety measures
- ✅ Add Vision & Mission section to FAQ.md after development disclaimer but before AI Agent Integration section

### Brand & Strategic Positioning
- ✅ Develop comprehensive brand values and positioning document covering: Focus (Financial Services), B2B, Professional, Trustworthy, Expert-backed, Innovative, Global, Open, Accessible. Create positioning statement and competitive differentiation
- ✅ Examine all .md files to make sure they square with the new ComplianceTools.md and add Blake Dempster, Founder & Principal Architect attribution where appropriate

### Strategic Planning & Analysis
- ✅ Review all pre-Phase 3 tasks to check whether they align with ComplianceTools.md, and make appropriate changes. Then add appropriate tasks to Phase 3 onwards
- ✅ Create internal/FCAsandbox.md capturing strategic opportunity and draft comprehensive answers to all FCA application questions (3.1-5.3) including solution overview, AI role, differentiation, impact, technical requirements, testing plan, and milestones
- ✅ Evaluate database architecture options: Current PostgreSQL + Qdrant vs Supabase (PostgreSQL + PGVector + real-time). Consider compliance requirements, self-hosting needs, vector search performance, and Cole Medin's recommendations for financial services architecture

### User Interface & Presentations
- ✅ Create FAQ.html version with nested structure similar to JBMD website format for professional presentation while keeping FAQ.md as source of truth
- ✅ Consider adding new category in Tasks.md after Phase 2, maybe 'Carry out detailed review of project so far' and shuffle existing tasks as necessary
- ✅ Create comprehensive UI mockup for demo version showing what FSCompliance looks like in practice. Create UserInterface.md as collection point for evolving design ideas, user workflows, personas, and interface concepts
- ✅ Create working HTML prototype (internal/UserInterface.html) to provide tangible first look at FSCompliance dashboard and key interface elements

### Quality Assurance & Consistency
- ✅ Review all content .md files (FAQ.md, Tasks.md, ComplianceTools.md, and internal files) for consistency in terminology, branding, attribution, and technical details
- ✅ Review Rules.md, Planning.md, and CLAUDE.md to reflect all documentation created and ensure they serve as accurate foundational guides (do this after content review)

### Development & Git Management
- ✅ Commit .gitignore changes and ComplianceTools.md to GitHub with appropriate commit message

---

## Pending Todos ⏳

### User Interface Updates
- ⏳ Update UserInterface.html landing page to reflect Claude 3.5 Sonnet as default LLM choice and premium positioning for compliance accuracy

### Strategic Content Enhancement
- ⏳ Consider adding points from Vision and Mission (in FAQ.md) to Project Overview (in Planning.md) to strengthen strategic context and regulatory problem articulation
- ⏳ Edit Vision and Mission section in FAQ.md to include reference to 'red tape' as part of regulatory burden language

### Go-to-Market Strategy
- ⏳ Create internal/TakeToMarket.md with primary pitch focused on 'slicing through red tape' messaging and go-to-market strategy
- ⏳ Ensure TakeToMarket.md messaging and positioning aligns with Brand.md core values and competitive differentiation strategy

### Documentation Formatting
- ⏳ Remove subtitle from Planning.md header and add proper About section at end consistent with other .md documents

---

## Todo Categories

### Documentation & Content Strategy
Items focused on improving content quality, consistency, and strategic messaging across all user-facing and internal documentation.

### Brand & Strategic Positioning  
Work related to establishing and maintaining consistent brand identity, competitive differentiation, and market positioning.

### Strategic Planning & Analysis
High-level strategic decisions, market analysis, competitive research, and architectural planning.

### User Interface & Presentations
Visual design, prototyping, and professional presentation materials for stakeholder engagement.

### Quality Assurance & Consistency
Cross-document validation, terminology alignment, and ensuring consistency across all project materials.

### Development & Git Management
Technical development tasks, version control, and project management activities.

### Go-to-Market Strategy
Sales, marketing, messaging, and business development activities.

---

## Usage Guidelines

### Adding New Todos
1. **Categorize appropriately** using the established categories above
2. **Use descriptive language** that clearly explains the work required  
3. **Include context** about why the todo is needed when not obvious
4. **Set appropriate status** (✅ Completed, ⏳ Pending, 🔄 In Progress)

### Updating Status
- Mark todos as ✅ **Completed** when fully finished
- Use 🔄 **In Progress** for active work
- Keep ⏳ **Pending** for items not yet started

### Relationship to Tasks.md
- **Tasks.md**: Phase-level implementation work and major deliverables
- **todos.md**: Granular improvements, refinements, and strategic decisions
- **Complementary tracking**: Both documents provide different levels of project visibility

---

## About This Document

**Author**: Blake Dempster, Founder & Principal Architect  
**Co-Authored by**: Claude Code (claude.ai/code)  
**Created**: 2024-12-25  
**Last Updated**: 2024-12-25  
**Purpose**: Granular todo tracking for FSCompliance development work, providing detailed visibility into project evolution and decision-making process.

*This document will be updated continuously as new todos are identified and completed throughout FSCompliance development.*

---