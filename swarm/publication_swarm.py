"""
publication_swarm.py - wires the per-paper agents into one LangGraph.

    entity_extractor -> relationship_extractor -> validator -> structure_enricher
                                ^__________________|  (the judge can send it back once)

(pubmed_extractor - the 5th agent - runs before this, in run_swarm.py, to produce
the papers; it is not a graph node.)

get_graph() returns the compiled graph; run_swarm.py runs it one paper at a time.
assemble_record() turns a finished paper's state into one KB record.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .agents import (entity_extractor, relationship_extractor, structure_enricher,
                     validator)
from .config import FIELDS
from .state import PaperState

_compiled = None


def _after_judge(state: PaperState) -> str:
    return "relationships" if state.get("needs_reextract") else "structures"


def build_publication_graph():
    g = StateGraph(PaperState)
    g.add_node("entities", entity_extractor.node)
    g.add_node("relationships", relationship_extractor.node)
    g.add_node("judge", validator.node)
    g.add_node("structures", structure_enricher.node)

    g.add_edge(START, "entities")
    g.add_edge("entities", "relationships")
    g.add_edge("relationships", "judge")
    g.add_conditional_edges("judge", _after_judge,
                            {"relationships": "relationships", "structures": "structures"})
    g.add_edge("structures", END)
    return g.compile()


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_publication_graph()
    return _compiled


def assemble_record(paper: Dict[str, Any], final: Dict[str, Any], kb_id: int) -> Dict[str, Any]:
    """Turn the per-paper graph's final state into one KB record (entity fields
    flattened onto the record, plus relationships + structures)."""
    entities = final.get("entities") or {}
    rec: Dict[str, Any] = {
        "id": kb_id,
        "pmid": paper.get("pmid"), "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""), "journal": paper.get("journal", ""),
        "year": paper.get("year", ""),
        "url": paper.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/",
    }
    for k in FIELDS:
        rec[k] = entities.get(k, [] if k not in ("study_type", "sample_count")
                              else ("" if k == "study_type" else None))
    structs = final.get("structures") or []
    rec["relationships"] = final.get("relationships") or []
    rec["structures"] = structs
    rec["n_structures"] = len(structs)
    return rec
