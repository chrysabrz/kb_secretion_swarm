"""
schemas.py - Pydantic schemas for structured agent outputs.

Every extraction / judgement agent emits a validated Pydantic object via
``llm.with_structured_output()``, so malformed JSON simply triggers a model
retry at the LangChain layer instead of crashing the graph. This is the "strong
typing" that keeps a multi-agent pipeline from drifting.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .config import ENTITY_TYPE_VOCABULARY, PREDICATE_VOCABULARY


# Stage 1 - entities
class PaperEntities(BaseModel):
    """One paper's extracted entities (the fields in config.FIELDS)."""
    species: List[str] = Field(default_factory=list)
    secretion_components: List[str] = Field(default_factory=list)
    exported_proteins: List[str] = Field(default_factory=list)
    genes: List[str] = Field(default_factory=list)
    drugs: List[str] = Field(default_factory=list)
    resistance_markers: List[str] = Field(default_factory=list)
    life_cycle_stages: List[str] = Field(default_factory=list)
    clinical_outcomes: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    study_type: str = ""
    sample_count: Optional[int] = None


# Stage 2 - relationships
class Triple(BaseModel):
    """A typed subject-predicate-object relationship from the controlled vocabulary."""
    subject: str
    subject_type: str = Field(default="", description=f"one of {ENTITY_TYPE_VOCABULARY}")
    predicate: str = Field(description=f"MUST be one of {PREDICATE_VOCABULARY}")
    object: str
    object_type: str = Field(default="", description=f"one of {ENTITY_TYPE_VOCABULARY}")


class TripleList(BaseModel):
    relationships: List[Triple] = Field(default_factory=list)


# Validation (the LLM judge)
class TripleVerdict(BaseModel):
    """The ValidationAgent's per-triple judgement."""
    index: int = Field(description="0-based index of the triple being judged")
    supported: bool = Field(description="Is this triple explicitly supported by the abstract?")
    confidence: float = Field(description="0.0-1.0 confidence the triple is correct and supported")
    issue: str = Field(default="", description="Short note if flagged (contradiction, hallucination, vocab misuse), else empty")


class ValidationReport(BaseModel):
    verdicts: List[TripleVerdict] = Field(default_factory=list)
