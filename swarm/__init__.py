"""
kb_secretion_swarm - an agent-based (LangGraph) knowledge-base system for
Plasmodium export / secretion biology and Type-VII / ESX secretion systems
(plus a standalone antimalarial drugs & resistance demo dataset).

A multi-agent extraction pipeline (small, specialized LLM agents wired as a LangGraph
state machine) builds the KB from PubMed, one paper at a time:
    PubMedExtractor -> EntityExtraction -> RelationshipExtraction
    -> Validation (the LLM judge) -> StructureEnrichment (UniProt/PDB/AlphaFold via MCP)

The result is a set of JSON files in data/, served by a Streamlit dashboard
(app.py) with a cited Q&A page (qa.py).
"""

__version__ = "1.0.0"
