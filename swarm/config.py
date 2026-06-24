"""
config.py - shared settings for the whole pipeline.

Edit this file to retarget the project: the PubMed queries, the entity schema
(FIELDS), the relationship vocabulary, the scope note, file paths, and make_llm().

Two extraction stages: Stage 1 pulls entities from each abstract (FIELDS); Stage 2
links those entities into typed relationships (PREDICATE_VOCABULARY).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")   # avoid encoding errors on some consoles
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Generated data lives in data/ (committed); caches stay in the root (gitignored).
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_FILE = DATA_DIR / "secretion_systems.json"
RELATIONSHIPS_FILE = DATA_DIR / "secretion_systems_with_relationships.json"
STRUCTURES_FILE = DATA_DIR / "secretion_systems_with_structures.json"
PROTEINS_FILE = DATA_DIR / "proteins.json"
SYNONYMS_FILE = DATA_DIR / "synonyms.json"
STRUCT_CACHE_FILE = ROOT / "_structures_cache.json"

# Models (can be edited): fast = bulk extraction; strong = the
# LLM judge; embed = the QA page. Default gpt-4o-mini.
DEFAULT_FAST_MODEL = os.getenv("SWARM_FAST_MODEL", "gpt-4o-mini")
DEFAULT_STRONG_MODEL = os.getenv("SWARM_STRONG_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("SWARM_EMBED_MODEL", "text-embedding-3-small")


def make_llm(model: str | None = None, *, temperature: float = 0.0, strong: bool = False):
    """Build a model by id; uses OpenAI for gpt-* and Anthropic for claude-*."""
    model = model or (DEFAULT_STRONG_MODEL if strong else DEFAULT_FAST_MODEL)
    provider = "anthropic" if model.lower().startswith("claude") else "openai"
    from langchain.chat_models import init_chat_model
    return init_chat_model(model, model_provider=provider, temperature=temperature)


def have_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def have_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# Q1 (Plasmodium export) + Q2 (ESX / Type-VII) are the active corpus. Q3 (antimalarial
# resistance) is OFF here; the Q3 2026 demo was built separately by demo_q3_2026.py.
QUERIES = [
    # 1. Plasmodium export / secretion machinery + exported proteins
    '("Plasmodium"[Title/Abstract]) AND ("protein export"[Title/Abstract] '
    'OR "PTEX"[Title/Abstract] OR "PEXEL"[Title/Abstract] OR "exportome"[Title/Abstract] '
    'OR "Maurer\'s clefts"[Title/Abstract] OR "secretion"[Title/Abstract] '
    'OR "EXP2"[Title/Abstract] OR "EXP1"[Title/Abstract] OR "HSP101"[Title/Abstract] '
    'OR "PTEX150"[Title/Abstract] OR "PTEX88"[Title/Abstract] OR "plasmepsin V"[Title/Abstract] '
    'OR "PfEMP1"[Title/Abstract] OR "KAHRP"[Title/Abstract] OR "SBP1"[Title/Abstract] '
    'OR "skeleton-binding protein"[Title/Abstract] OR "MAHRP1"[Title/Abstract] '
    'OR "MAHRP2"[Title/Abstract] OR "REX1"[Title/Abstract] OR "REX2"[Title/Abstract] '
    'OR "PNEPs"[Title/Abstract] OR "Pf332"[Title/Abstract])',
    # 2. Type-VII / ESX secretion systems
    '("Type VII secretion"[Title/Abstract] OR "ESX-1"[Title/Abstract] '
    'OR "ESX-5"[Title/Abstract] OR "ESAT-6"[Title/Abstract] OR "T7SS"[Title/Abstract]) '
    'AND (Mycobacterium[Title/Abstract] OR Plasmodium[Title/Abstract])',
    # 3. Antimalarial drugs & resistance - OFF (see demo_q3_2026.py)
    # '("antimalarial"[Title/Abstract] OR "artemisinin"[Title/Abstract] '
    # 'OR "chloroquine resistance"[Title/Abstract] OR "kelch13"[Title/Abstract] '
    # 'OR "pfcrt"[Title/Abstract] OR "pfmdr1"[Title/Abstract])',
]

# Stage-1 schema: field -> instruction. The entity agent returns exactly these keys.
FIELDS: dict[str, str] = {
    "species": "Pathogen species / strains named (e.g. Plasmodium falciparum, "
               "Mycobacterium tuberculosis, Toxoplasma gondii).",
    "secretion_components": "Secretion/export machinery components (PTEX, EXP2, "
                            "HSP101, PTEX150, Type VII / ESX components, EsxA/EsxB, EccB/C/D).",
    "exported_proteins": "Exported / effector / surface proteins (PfEMP1, KAHRP, "
                         "rhoptry/microneme proteins, RON2, etc.).",
    "genes": "Gene names / loci (kelch13/k13, pfcrt, pfmdr1, exp2, ptex150, eccB3 ...).",
    "drugs": "Antimalarial drugs / compounds (artemisinin, chloroquine, lumefantrine, "
             "RTS,S, R21, artesunate).",
    "resistance_markers": "Resistance markers / mutations (k13 C580Y, pfcrt mutations, "
                          "pfmdr1 amplification).",
    "life_cycle_stages": "Parasite life-cycle stages (merozoite, sporozoite, ring, "
                         "trophozoite, schizont, gametocyte, liver stage).",
    "clinical_outcomes": "Clinical outcomes / phenotypes (severe malaria, parasite "
                         "clearance, treatment failure, cerebral malaria).",
    "methods": "Experimental / computational methods used (cryo-EM, CRISPR, knockdown, "
               "mass spectrometry, structure prediction).",
    "study_type": "One short label: experimental, structural, review, clinical, "
                  "computational, or genomic.",
    "sample_count": "Integer sample/isolate/structure count if stated, else null.",
}
LIST_FIELDS = [k for k in FIELDS if k not in ("study_type", "sample_count")]
ENTITY_FIELDS = LIST_FIELDS   # same set (the list-valued entity fields); alias avoids drift

# Stage-2 vocabulary: the only allowed predicates and entity types for triples.
PREDICATE_VOCABULARY = [
    "is_component_of", "is_orthologous_to", "secretes", "is_exported_via",
    "cleaves", "processes", "confers_resistance_to", "is_target_of", "inhibits",
    "required_for", "interacts_with", "localizes_to", "expressed_in",
    "causes_phenotype", "regulates",
]
ENTITY_TYPE_VOCABULARY = [
    "protein", "complex", "gene", "mutation_or_allele", "drug", "motif",
    "organelle_or_compartment", "life_stage", "process", "phenotype", "species",
    "disease",
]

# Some predicates are only meaningful between specific entity types. is_orthologous_to
# is a GENE/PROTEIN concept: two *species* (or processes, diseases, motifs ...) are
# never "orthologous" to each other - that is a category error the extractor sometimes
# makes ("Plasmodium spp. is_orthologous_to Toxoplasma gondii"). The validator drops
# any such triple whose (explicitly typed) ends are not molecular.
MOLECULAR_TYPES = {"gene", "protein"}
PREDICATE_TYPE_CONSTRAINTS = {
    "is_orthologous_to": (MOLECULAR_TYPES, MOLECULAR_TYPES),
}
#Orthologous is a term used for genes that evolved from a single common ancestral gene  via speciation; 
#hence, their derived proteins too, so those are the only types we allow for the relationship "is_orthologous"

def predicate_acceptance_types(predicate: str, subject_type: str, object_type: str) -> bool:
    """False if a typed triple violates a predicate's entity-type constraint.

    Empty/unknown types pass (benefit of the doubt - the LLM judge still scores
    them); only an explicitly wrong type is rejected here.
    """
    constraint = PREDICATE_TYPE_CONSTRAINTS.get(predicate)
    if not constraint:
        return True
    subj_ok, obj_ok = constraint
    if subject_type and subject_type not in subj_ok:
        return False
    if object_type and object_type not in obj_ok:
        return False
    return True

# Added to every prompt so the model never conflates the two systems.
SCOPE_NOTE = (
    "SCOPE: This corpus holds two functionally analogous but biologically DISTINCT "
    "systems side by side - (1) Plasmodium (malaria) protein export (PTEX translocon, "
    "PEXEL motif, plasmepsin V, Maurer's clefts) and (2) the bacterial ESX / Type-VII "
    "secretion system (Mycobacterium and other Gram-positives). They are FUNCTIONAL "
    "ANALOGUES ONLY: there is NO orthology and NO shared machinery BETWEEN THE TWO "
    "SYSTEMS - never assert homology that bridges Plasmodium export and bacterial ESX "
    "unless a paper explicitly states it. This does NOT forbid ordinary orthology WITHIN "
    "a lineage (e.g. between apicomplexans such as Plasmodium and Toxoplasma, or among "
    "mycobacteria); accept those when a paper states them."
)
