"""
relationship_extractor.py - turns a paper's entities into typed relationships.

Given the title + abstract and the entities already found in it, this agent adds
subject-predicate-object triples (e.g. EXP2 is_component_of PTEX), restricted to the
allowed predicates and entity types in config. clean_triples() keeps only well-formed,
in-vocabulary, de-duplicated triples. These are *candidates* - the judge checks them next.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from ..config import (ENTITY_FIELDS, ENTITY_TYPE_VOCABULARY, PREDICATE_VOCABULARY,
                      SCOPE_NOTE, make_llm)
from ..schemas import TripleList
from ..state import PaperState

_SYSTEM = (
    "You are a precise biomedical knowledge-graph builder. " + SCOPE_NOTE +
    " Only extract relationships explicitly supported by the abstract."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "From the paper below, extract a list of TYPED subject-predicate-object "
     "relationships that are explicitly stated in the abstract.\n\n"
     "TITLE: {title}\n\nABSTRACT:\n{abstract}\n\n"
     "PREFERRED VOCABULARY (reuse these entity names as subjects/objects where they fit):\n"
     "{hints}\n\n"
     "Rules:\n"
     "- predicate MUST be one of: {predicates}\n"
     "- subject_type and object_type MUST each be one of: {types}\n"
     "- Only include relationships actually supported by the abstract.\n"
     "- Return an empty list if no clear typed relationship is present."),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = make_llm(temperature=0.0)
        _chain = _PROMPT | llm.with_structured_output(TripleList)
    return _chain


def clean_triples(items: List[Any]) -> List[Dict[str, str]]:
    """Keep only well-formed triples whose predicate/types are in-vocabulary."""
    out: List[Dict[str, str]] = []
    seen = set()
    for t in items or []:
        t = t.model_dump() if hasattr(t, "model_dump") else t
        if not isinstance(t, dict):
            continue
        subj = str(t.get("subject", "")).strip()
        obj = str(t.get("object", "")).strip()
        pred = str(t.get("predicate", "")).strip()
        st_ = str(t.get("subject_type", "")).strip()
        ot_ = str(t.get("object_type", "")).strip()
        if not (subj and obj and pred) or pred not in PREDICATE_VOCABULARY:
            continue
        if st_ and st_ not in ENTITY_TYPE_VOCABULARY:
            st_ = ""
        if ot_ and ot_ not in ENTITY_TYPE_VOCABULARY:
            ot_ = ""
        key = (subj.lower(), pred, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"subject": subj, "subject_type": st_, "predicate": pred,
                    "object": obj, "object_type": ot_})
    return out


def node(state: PaperState) -> Dict[str, Any]:
    paper = state["paper"]
    entities = state.get("entities") or {}
    hints = {f: entities.get(f) for f in ENTITY_FIELDS if entities.get(f)}
    hint_block = json.dumps(hints, ensure_ascii=False, indent=2) if hints else "{}"
    try:
        result: TripleList = _get_chain().invoke({
            "title": paper.get("title", ""), "abstract": paper.get("abstract", ""),
            "hints": hint_block, "predicates": ", ".join(PREDICATE_VOCABULARY),
            "types": ", ".join(ENTITY_TYPE_VOCABULARY)})
        triples = clean_triples(result.relationships)
    except Exception as e:
        return {"candidate_triples": [],
                "errors": (state.get("errors") or []) + [f"rel:{paper.get('pmid')}:{e}"]}
    return {"candidate_triples": triples}
