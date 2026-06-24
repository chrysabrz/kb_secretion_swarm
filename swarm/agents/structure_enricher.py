"""
structure_enricher.py - attaches structures to each paper.

For every protein/gene the paper mentions, looks it up via resolve_entity (UniProt +
PDB + AlphaFold, through the MCP).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict

from ..config import STRUCT_CACHE_FILE
from ..state import PaperState
from ..structure_resolver import PROT_FIELDS, norm, resolve_entity
from ..tools import BioMCP

_lock = threading.Lock()
_cache: Dict[str, Any] | None = None


def _load_cache() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        if STRUCT_CACHE_FILE.exists():
            try:
                _cache = json.loads(STRUCT_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is not None:
        STRUCT_CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=2),
                                     encoding="utf-8")


def node(state: PaperState) -> Dict[str, Any]:
    paper = state["paper"]
    entities = state.get("entities") or {}
    cache = _load_cache()
    mcp = BioMCP.client()

    structs, seen = [], set()
    for f in PROT_FIELDS:
        for v in entities.get(f) or []:
            if not isinstance(v, str):
                continue
            k = norm(v).lower()
            if not k or k in seen:
                continue
            seen.add(k)
            with _lock:
                rec = cache.get(k)
                if rec is None:
                    try:
                        rec = resolve_entity(mcp, norm(v))
                    except Exception as e:
                        rec = {"entity": v, "resolved": False, "error": str(e)}
                    # Cache determinate results (resolved or genuine "not found"), but NOT
                    # transient errors - otherwise one MCP outage poisons the entity forever.
                    if "error" not in rec:
                        cache[k] = rec
            if rec and rec.get("resolved"):
                structs.append(rec)

    with _lock:
        _save_cache()
    return {"structures": structs}
