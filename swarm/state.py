"""
state.py - the LangGraph state for the per-paper pipeline.

Each paper flows through its own subgraph: extraction -> relationships -> judge ->
structures. This dict is what gets passed from one step to the next.
"""

from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import TypedDict


class PaperState(TypedDict, total=False):
    """State threaded through the per-paper pipeline subgraph."""
    paper: Dict[str, Any]                 # {pmid,title,abstract,journal,year,url}
    entities: Dict[str, Any]              # PaperEntities as dict
    candidate_triples: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]   # judge-accepted, confidence-tagged triples
    structures: List[Dict[str, Any]]
    judge_retries: int
    needs_reextract: bool                 # set by the judge: route back to re-extract once
    errors: List[str]
    record: Dict[str, Any]                # final assembled KB record
