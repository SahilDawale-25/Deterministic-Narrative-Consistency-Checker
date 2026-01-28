# Deterministic Narrative Consistency Checker

## Overview
This project verifies whether a character’s hypothetical backstory is logically and causally consistent with events in a full-length novel (100,000+ words). The system focuses on deterministic, explainable reasoning rather than probabilistic language model judgments.

## Problem Statement
In long narratives, critical facts may be separated by hundreds of pages. Characters may perform actions that are impossible due to prior events such as death, imprisonment, or irreversible constraints. Traditional LLM-based approaches struggle with long context, reproducibility, and traceability.

## Solution
We propose a deterministic reasoning pipeline that:
- Preserves full temporal structure of the novel
- Converts narrative events into symbolic character states
- Enforces irreversible logical and causal constraints
- Uses LLMs only for perception-level tasks (optional)

Final decisions are made strictly through rule-based logic.

## Input
1. **Full Novel Text**
   - Entire novel ingested without truncation or summarization
   - Temporal order preserved across all events

2. **Hypothetical Character Backstory**
   - Written independently
   - Decomposed into atomic, time-aware claims

## Output
- **Binary Consistency Label**
  - `1` → Backstory is consistent
  - `0` → Backstory is contradictory
- **Optional Evidence-Based Rationale**
  - References violated constraints and supporting narrative evidence

## System Architecture
1. Long-context ingestion using Pathway
2. Ordered chunking with metadata
3. Symbolic character state extraction
4. Backstory claim decomposition
5. Constraint-based consistency checking
6. Deterministic final decision

## Reasoning Rules
- **Death Rule:** No actions allowed after death
- **Imprisonment Rule:** No free actions during confinement
- **Escape Rule:** Terminates imprisonment interval

Any violation results in a contradiction.

## Design Principles
- Deterministic and reproducible outputs
- Separation of perception and reasoning
- No probabilistic decision-making
- Full explainability and auditability

## Applications
- Legal and contract auditing
- Compliance and policy validation
- Enterprise knowledge consistency
- Financial due diligence
- Narrative analysis in literature

## Limitations
- Figurative language ambiguity
- Alias or identity resolution
- Implicit event detection

## Authors
- Sahil Dawale  
- Vyankat Rathod  
- Dhruvi Parekh  

## License
This project is intended for academic and research purposes.
