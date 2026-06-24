"""
structure_resolver.py - resolve an entity name to UniProt / PDB / AlphaFold via the MCP.

Given a messy name (e.g. "PTEX150", "k13 C580Y"), resolve_entity() tries an exact
gene match, then a validated free-text search; it keeps only apicomplexan/mycobacterial
hits, prefers reviewed (Swiss-Prot) entries, and rejects category phrases like
"microneme proteins" (not a single protein). The StructureEnrichmentAgent calls
resolve_entity and shares a disk cache across papers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

# NCBI taxonomy IDs used to restrict UniProt searches to the right organism.
MYCOBACTERIUM_TAXID = 1763   # genus Mycobacterium (ESX / Type-VII)
PLASMODIUM_TAXID = 5820      # genus Plasmodium (malaria)

PROT_FIELDS = ["genes", "exported_proteins", "secretion_components", "resistance_markers"]
ALLOWED_GENERA = ("mycobacterium", "plasmodium", "toxoplasma", "babesia",
                  "theileria", "cryptosporidium", "plasmodial")
MIN_LEN = 3
CATEGORY_WORDS = ("systems", "system", "proteins", "components", "complexes",
                  "clefts", "kinases", "proteases", "enzymes", "factors",
                  "antigens", "markers", "transporters", "channels", "pathways",
                  "genes", "motifs", "domains", "organelles", "vesicles", "family")


def norm(s: str) -> str:
    return " ".join(str(s).split())


def _norm_key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _is_category(entity: str) -> bool:
    e = entity.lower().strip()
    toks = e.split()
    if toks and toks[-1] in CATEGORY_WORDS:
        return True
    return bool(re.search(r"type\s+[ivx]+\s+secretion", e))


def _organism_ok(org: str) -> bool:
    o = (org or "").lower()
    return any(g in o for g in ALLOWED_GENERA)


def _validate_hit(entity: str, hit: dict) -> bool:
    k = _norm_key(entity)
    if len(k) < MIN_LEN:
        return False
    terms = {_norm_key(hit.get("gene", ""))}
    terms |= {_norm_key(s) for s in (hit.get("synonyms") or [])}
    terms.discard("")
    if k in terms:
        return True
    pn = _norm_key(hit.get("protein_name", ""))
    return len(k) >= 4 and k in pn


def _pick(hits, accept):
    fallback = None
    for h in hits:
        if h.get("accession") and _organism_ok(h.get("organism", "")) and accept(h):
            if h.get("reviewed"):
                return h
            if fallback is None:
                fallback = h
    return fallback


def candidate_entities(records) -> Counter:
    c: Counter = Counter()
    for r in records:
        for f in PROT_FIELDS:
            for v in r.get(f) or []:
                if isinstance(v, str):
                    t = norm(v)
                    if len(t) >= MIN_LEN and not _is_category(t):
                        c[t] += 1
    return c


def resolve_entity(mcp, entity: str) -> dict:
    """One entity -> {resolved, confidence, uniprot, pdb[], alphafold} via the MCP tools."""
    rec = {"entity": entity, "resolved": False, "confidence": None,
           "uniprot": None, "pdb": [], "alphafold": None}
    if _is_category(entity):
        return rec
    queries = [entity]
    if " " in entity:
        queries.append(entity.split()[0])
    taxids = (PLASMODIUM_TAXID, MYCOBACTERIUM_TAXID, 0)
    hit, confidence = None, None

    # Pass 1 - exact GENE match
    for q in queries:
        for taxid in taxids:
            try:
                hits = mcp.uniprot_search(f"gene:{q}", size=6, taxonomy_id=taxid)
            except Exception:
                hits = []
            h = _pick(hits, lambda x, _q=q: _norm_key(_q) and _norm_key(_q) == _norm_key(x.get("gene", "")))
            if h:
                hit, confidence = h, "gene_exact"
                break
        if hit:
            break

    # Pass 2 - validated free-text search
    if not hit:
        for q in queries:
            for taxid in taxids:
                try:
                    hits = mcp.uniprot_search(q, size=6, taxonomy_id=taxid)
                except Exception:
                    hits = []
                h = _pick(hits, lambda x, _q=q: _validate_hit(_q, x))
                if h:
                    hit, confidence = h, "name_match"
                    break
            if hit:
                break

    if not hit:
        return rec

    rec["resolved"] = True
    rec["confidence"] = confidence
    rec["uniprot"] = {
        "accession": hit.get("accession"), "protein_name": hit.get("protein_name"),
        "gene": hit.get("gene"), "organism": hit.get("organism"),
        "function_summary": hit.get("function_summary"), "url": hit.get("url"),
        "subcellular_locations": hit.get("subcellular_locations") or [],
        "go_component": hit.get("go_component") or [], "go_process": hit.get("go_process") or [],
        "go_function": hit.get("go_function") or [], "keywords": hit.get("keywords") or [],
        "pfam": hit.get("pfam") or [], "interpro": hit.get("interpro") or [],
        "length": hit.get("length"), "mass": hit.get("mass"),
        "families": hit.get("families") or [], "catalytic_activity": hit.get("catalytic_activity") or [],
        "ec_numbers": hit.get("ec_numbers") or [], "pathway": hit.get("pathway") or [],
        "features": hit.get("features") or {}, "kegg": hit.get("kegg") or [],
    }
    for pid in (hit.get("pdb_xrefs") or [])[:10]:
        rec["pdb"].append({"pdb_id": pid, "url": f"https://www.rcsb.org/structure/{pid}",
                           "view3d": f"https://www.rcsb.org/3d-view/{pid}"})
    if not rec["pdb"] and hit.get("accession"):
        try:
            af = mcp.alphafold_lookup(hit["accession"])
        except Exception:
            af = {}
        if af:
            rec["alphafold"] = af
    return rec
