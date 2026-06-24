"""
entity_extractor.py - the entity-extraction step (one paper at a time).

Reads a paper's title + abstract and asks the LLM to pull out the entities we care
about (genes, drugs, species, ... - the fields in config.FIELDS). The answer comes
back in a fixed shape (PaperEntities), so we always get clean, predictable JSON
instead of free text; a malformed reply is automatically retried.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from ..config import FIELDS, LIST_FIELDS, SCOPE_NOTE, make_llm
from ..schemas import PaperEntities
from ..state import PaperState

_SYSTEM = (
    "You are a precise biomedical information extractor for protein "
    "secretion/export systems. " + SCOPE_NOTE + " Return ONLY the structured fields."
)


def _schema_hint() -> str:
    return "\n".join(f"- {k}: {desc}" for k, desc in FIELDS.items())


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "Extract structured information about protein secretion/export systems from the "
     "paper below - either Plasmodium (malaria) export & secretion biology and "
     "antimalarial resistance, OR the bacterial ESX / Type-VII secretion system.\n\n"
     "Fields to populate (use [] for empty lists, null for a missing integer):\n"
     "{schema}\n\n"
     "TITLE: {title}\n\nABSTRACT:\n{abstract}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = make_llm(temperature=0.0)
        _chain = _PROMPT | llm.with_structured_output(PaperEntities)
    return _chain


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    for k in LIST_FIELDS:
        v = data.get(k)
        data[k] = v if isinstance(v, list) else ([] if v in (None, "") else [v])
    data.setdefault("study_type", "")
    if not isinstance(data.get("sample_count"), int):
        data["sample_count"] = None
    return data


def node(state: PaperState) -> Dict[str, Any]:
    paper = state["paper"]
    try:
        result: PaperEntities = _get_chain().invoke(
            {"schema": _schema_hint(), "title": paper.get("title", ""),
             "abstract": paper.get("abstract", "")})
        entities = _normalise(result.model_dump())
    except Exception as e:
        entities = _normalise({})
        return {"entities": entities,
                "errors": (state.get("errors") or []) + [f"entity:{paper.get('pmid')}:{e}"]}
    return {"entities": entities}
