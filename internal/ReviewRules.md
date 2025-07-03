# Review Rules for FSCompliance Documentation

This document establishes a systematic methodology for reviewing .md files to ensure consistency, quality, and strategic alignment across FSCompliance project documentation.

**Important**: If reviewing content related to "Outline of Management Impacts" emailshots, use internal/MgmtImpactRules.md methodology instead of this ReviewRules.md process. MgmtImpactRules.md provides specialized guidance for the distinct audience, style, and objectives of Management Impacts communications.

## Review Sequence

When reviewing any .md file, follow this systematic approach:

### 1. Content Quality Check
- **Avoid repetition** - Don't repeat what has substantially been said before
- **Professional tone** - Avoid colloquialisms, avoid being bombastic, be succinct, and adopt a tone suited to the corporate world
- **Internal consistency** - Check the document's internal logic and consistency

### 2. Touchstones Alignment
- **Check against Touchstones.md** - Unless reviewing Touchstones.md itself, verify no inconsistencies with established project touchstones
- **Resolve inconsistencies** - If conflicts exist, resolve them or seek direction
- **Identify new touchstones** - If you identify potential new touchstones, raise them and seek direction on whether to add them
- **Explicit touchstone assessment** - Always conclude this step with either:
  - "No new touchstones proposed" (if document aligns with existing touchstones)
  - "New touchstone proposed: [description]" (if new fundamental principle identified)

### 3. Inter-Document Consistency
- **Use Touchstones method** - Don't try to check against every other .md file individually
- **Avoid complexity overload** - The Touchstones approach prevents overwhelming complexity from holding multiple documents in attention simultaneously

### 4. CLAUDE.md Impact Assessment
Determine if changes require updates to CLAUDE.md implementation guidance:
- **Technical architecture or configuration changes** - Does this affect system design, LLM selection, or deployment architecture?
- **Development practices or coding standards** - Does this impact how code should be written or structured?
- **Implementation tasks or requirements** - Does this require new development work or modify existing tasks?
- **User-facing elements** - Does this affect UI copy, messaging, or user workflows that need implementation?
- **If yes to any**, update CLAUDE.md with specific implementation guidance linking strategic decisions to code requirements

### 4.1. JBMD Website Update Reminder
**Special consideration for FAQ.md and UserInterface.md:**
- If reviewing FAQ.md, add reminder to update JBMDwebsite/fscompliance.html with finalized FAQ content
- If reviewing UserInterface.md, add reminder to update JBMD website with any new interface demonstrations or design changes
- Maintain sync between FSCompliance documentation and JBMD website presentation materials

### 5. Multi-Perspective Review
Read the .md being reviewed in complete sweeps from each relevant stakeholder viewpoint:

**Financial Institution Perspectives:**
1. **Chief Executive Officer (CEO)** - Strategic vision and business impact
2. **Chief Compliance Officer (CCO)** - Regulatory requirements and compliance strategy
3. **Chief Technology Officer (CTO)** - Strategic technology architecture and innovation
4. **Chief Information Security Officer (CISO)** - Data security, privacy, and risk management
5. **Chief Risk Officer (CRO)** - Enterprise risk management and regulatory risk

**Note:** Some documents may not be relevant to all perspectives - simply note this and move on to applicable viewpoints.

## Naming Conventions

**Project vs Product Distinction:**
- **FSCompliance** - refers to the overall project, codebase, and development initiative
- **The MCP** - refers to the product that enterprises deploy and use

This distinction should be maintained consistently across all documentation to clarify communications with different audiences.

## Review Documentation

- **Track review completion** - Mark documents as reviewed in todo systems
- **Note review date** - Include review date in document metadata when substantial reviews are completed
- **Document changes** - Significant changes should be noted and justified
- **Date format standard** - Use UK date format "DD Month YYYY" (e.g., "25 December 2024") for all Created and Last Updated dates

---

## About This Document

**Author**: Blake Dempster, Founder & Principal Architect  
**Co-Authored by**: Claude Code (claude.ai/code)  
**Created**: 3 July 2025  
**Purpose**: Systematic methodology for reviewing FSCompliance project documentation to ensure consistency, quality, and strategic alignment.

---