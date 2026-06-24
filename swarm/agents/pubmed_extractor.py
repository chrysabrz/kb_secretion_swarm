"""
pubmed_extractor.py - finds the papers to process.

Runs each configured PubMed query, drops duplicate PMIDs, skips any already in the
KB (so re-running doesn't redo work), and fetches title/abstract/journal/year for
the new ones. Returns the full list of new papers in one call (fetched in
batches); "incremental" across runs - already-seen PMIDs are skipped.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set

from ..config import QUERIES
from ..tools import fetch_pubmed, pubmed_client, search_pubmed


def run(max_results: int, queries_sel: Set[int] | None, done_pmids: Set[str],
        date_filter: str = "", batch: int = 50) -> List[Dict[str, Any]]:
    queries = [q for i, q in enumerate(QUERIES, 1)
               if not queries_sel or i in queries_sel] or QUERIES

    Entrez = pubmed_client()
    pmids: List[str] = []
    seen: Set[str] = set()
    for i, q in enumerate(queries, 1):
        ids = search_pubmed(Entrez, q + date_filter, max_results)
        new = [p for p in ids if p not in seen]
        seen.update(new)
        pmids.extend(new)
        print(f"[PubMedExtractorAgent] query {i}/{len(queries)}: {len(ids)} hits "
              f"({len(new)} new this run)")
        time.sleep(0.34)

    todo = [p for p in pmids if p not in done_pmids]
    print(f"[PubMedExtractorAgent] {len(pmids)} unique PMIDs; {len(todo)} new to process "
          f"(skipping {len(pmids) - len(todo)} already in KB).")

    papers: List[Dict[str, Any]] = []
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        for pub in fetch_pubmed(Entrez, chunk):
            if str(pub["pmid"]) in done_pmids:
                continue
            if not (pub.get("title") or pub.get("abstract")):
                continue
            papers.append(pub)
        time.sleep(0.34)
    return papers
