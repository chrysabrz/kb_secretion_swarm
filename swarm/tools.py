"""
tools.py - clients for the external data sources: PubMed (Entrez) and the bio-MCP server.

* PubMed (Bio.Entrez): pubmed_client / search_pubmed / fetch_pubmed - used by
  pubmed_extractor to find papers and download title/abstract/journal/year.
* BioMCP: one shared connection to the local MCP server (mcp_server.py via
  mcp_client.py) for UniProt / PDB / AlphaFold - used by structure_enricher.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# PubMed
def pubmed_client():
    from Bio import Entrez
    Entrez.email = os.getenv("NCBI_EMAIL", "research@example.com")
    if os.getenv("NCBI_API_KEY"):
        Entrez.api_key = os.getenv("NCBI_API_KEY")
    return Entrez


def search_pubmed(Entrez, query: str, max_results: int) -> List[str]:
    h = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    rec = Entrez.read(h); h.close()
    return list(rec.get("IdList", []))


def fetch_pubmed(Entrez, pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []
    h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    rec = Entrez.read(h); h.close()
    out = []
    for art in rec.get("PubmedArticle", []):
        try:
            mc = art["MedlineCitation"]
            pmid = str(mc["PMID"])
            article = mc["Article"]
            title = str(article.get("ArticleTitle", "") or "")
            ab = article.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(x) for x in ab) if isinstance(ab, list) else str(ab or "")
            journal = str(article.get("Journal", {}).get("Title", "") or "")
            year = ""
            try:
                year = str(article["Journal"]["JournalIssue"]["PubDate"].get("Year", ""))
            except Exception:
                pass
            out.append({"pmid": pmid, "title": title, "abstract": abstract,
                        "journal": journal, "year": year,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
        except Exception:
            continue
    return out


# Bio-MCP
# A single persistent MCP subprocess shared across the structure-enrichment run.
class BioMCP:
    """Thin singleton wrapper around BioMCPClient (UniProt / PDB / AlphaFold)."""

    _client = None

    @classmethod
    def client(cls):
        if cls._client is None:
            from mcp_client import BioMCPClient
            cls._client = BioMCPClient().start()
        return cls._client

    @classmethod
    def stop(cls):
        if cls._client is not None:
            try:
                cls._client.stop()
            finally:
                cls._client = None
