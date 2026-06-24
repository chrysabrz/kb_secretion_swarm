"""
validator.py - the LLM judge (the quality gate).

This is the agent that makes the output trustworthy. First a deterministic **type gate**
(predicate_acceptance_types against PREDICATE_TYPE_CONSTRAINTS in config) drops candidates
that misuse a *constrained* predicate's entity types (currently only is_orthologous_to) -
e.g. species is_orthologous_to species. Then, for each surviving candidate triple it:
  - re-reads the abstract and decides if the triple is really supported, giving a
    0-1 confidence and flagging hallucinations / cross-system-homology / bad relations;
  - votes N times (SWARM_JUDGE_VOTES, default 3) and keeps a triple only if the
    median confidence >= threshold (SWARM_JUDGE_THRESHOLD, default 0.5) AND most
    votes mark it supported;
  - tags each surviving triple with its pmid and that confidence.

If every candidate is rejected, it sends the paper back to re-extract once. If the judge
call itself errors (e.g. API outage), it fails open: candidates are kept un-scored
(confidence=null) rather than dropped. See the README (Confidence section) for details.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from ..config import SCOPE_NOTE, make_llm, predicate_acceptance_types
from ..schemas import ValidationReport
from ..state import PaperState

JUDGE_VOTES = int(os.getenv("SWARM_JUDGE_VOTES", "3"))
CONF_THRESHOLD = float(os.getenv("SWARM_JUDGE_THRESHOLD", "0.5"))

_SYSTEM = (
    "You are a stringent biomedical fact-checker validating an automatically "
    "extracted knowledge graph. " + SCOPE_NOTE + " For each candidate triple, decide "
    "whether the ABSTRACT explicitly supports it. Be skeptical: reject triples that "
    "are plausible but not actually stated, that assert cross-system homology between "
    "Plasmodium export and bacterial ESX, or that misuse the relation type. Give a "
    "calibrated 0.0-1.0 confidence per triple."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "TITLE: {title}\n\nABSTRACT:\n{abstract}\n\n"
     "CANDIDATE TRIPLES (0-indexed):\n{triples}\n\n"
     "Return one verdict per triple: its index, whether it is supported, a 0-1 "
     "confidence, and a short issue note if you flag it."),
])

_chains: Dict[int, Any] = {}


def _get_chain(vote: int):
    if vote not in _chains:
        # small temperature spread across votes for diversity in the consensus
        temp = 0.0 if vote == 0 else min(0.6, 0.2 * vote)
        llm = make_llm(temperature=temp, strong=True)
        _chains[vote] = _PROMPT | llm.with_structured_output(ValidationReport)
    return _chains[vote]


def _judge_once(vote: int, title: str, abstract: str, triples: List[Dict]) -> Dict[int, Dict]:
    shown = json.dumps([{"index": i, **{k: t[k] for k in ("subject", "predicate", "object")}}
                        for i, t in enumerate(triples)], ensure_ascii=False, indent=2)
    report: ValidationReport = _get_chain(vote).invoke(
        {"title": title, "abstract": abstract, "triples": shown})
    return {v.index: {"supported": v.supported, "confidence": float(v.confidence),
                      "issue": v.issue} for v in report.verdicts}


def node(state: PaperState) -> Dict[str, Any]:
    paper = state["paper"]
    pmid = str(paper.get("pmid", ""))
    # Deterministic type gate before the LLM judge: drop triples that misuse a
    # predicate's entity types (e.g. species is_orthologous_to species).
    cands = [t for t in (state.get("candidate_triples") or [])
             if predicate_acceptance_types(t.get("predicate", ""),
                                           t.get("subject_type", ""), t.get("object_type", ""))]
    if not cands:
        return {"relationships": [], "needs_reextract": False}

    # collect votes
    per_triple: Dict[int, Dict[str, List]] = {i: {"sup": [], "conf": [], "issues": []}
                                              for i in range(len(cands))}
    try:
        for vote in range(max(1, JUDGE_VOTES)):
            verdicts = _judge_once(vote, paper.get("title", ""),
                                   paper.get("abstract", ""), cands)
            for i in range(len(cands)):
                v = verdicts.get(i)
                if v:
                    per_triple[i]["sup"].append(bool(v["supported"]))
                    per_triple[i]["conf"].append(v["confidence"])
                    if v["issue"]:
                        per_triple[i]["issues"].append(v["issue"])
    except Exception as e:
        # judge unavailable -> fail open with un-scored triples (still PMID-tagged)
        accepted = [{**t, "pmid": pmid, "confidence": None} for t in cands]
        return {"relationships": accepted, "needs_reextract": False,
                "errors": (state.get("errors") or []) + [f"judge:{pmid}:{e}"]}

    accepted: List[Dict] = []
    for i, t in enumerate(cands):
        votes = per_triple[i]
        if not votes["conf"]:
            continue
        conf = round(statistics.median(votes["conf"]), 3)
        majority_supported = sum(votes["sup"]) > (len(votes["sup"]) / 2.0)
        if conf >= CONF_THRESHOLD and majority_supported:
            accepted.append({**t, "pmid": pmid, "confidence": conf})

    # retry gate: candidates existed but the judge rejected all of them -> re-extract once.
    # needs_reextract is a declared PaperState channel the graph router reads; it MUST be
    # set on every return path (incl. False here) or it would persist True and loop forever.
    if cands and not accepted and state.get("judge_retries", 0) < 1:
        return {"relationships": [], "judge_retries": state.get("judge_retries", 0) + 1,
                "needs_reextract": True}
    return {"relationships": accepted, "needs_reextract": False}
