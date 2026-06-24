"""Agents that build the KB (the per-paper pipeline):

    pubmed_extractor       - runs each query, fetches abstracts
    entity_extractor       - Entity schema (structured output)
    relationship_extractor - typed subject-predicate-object triples
    validator              - the LLM judge (confidence + accept/reject)
    structure_enricher     - UniProt / PDB / AlphaFold via MCP

(Q&A is handled separately by qa.py, used by the dashboard.)
"""
