"""
run_swarm.py - build the knowledge base from PubMed.

PubMedExtractor -> [per-paper LangGraph: entities -> relationships -> judge ->
structures] -> writes the three stage files, then builds the protein view +
synonym maps:
    data/secretion_systems.json
    data/secretion_systems_with_relationships.json
    data/secretion_systems_with_structures.json
    data/proteins.json   (build_proteins.py)
    data/synonyms.json                      (normalize.py)

Incremental: papers already in the KB are skipped, so re-running only spends
budget on new papers.

Examples
    python run_swarm.py                                  # default corpus (queries 1 + 2)
    python run_swarm.py --queries 2                      # run only query 2 (the ESX / Type-VII search)
    python run_swarm.py --max-results 50 --limit 5       # quick smoke test (few papers)
    python run_swarm.py --year-from 2015 --year-to 2026  # restrict to a publication-year range
    python run_swarm.py --no-postprocess                 # skip building proteins + synonyms afterwards
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List

from swarm.config import (ENTITIES_FILE, FIELDS, QUERIES, RELATIONSHIPS_FILE, ROOT,
                          STRUCTURES_FILE, have_anthropic, have_openai)
from swarm import publication_swarm
from swarm.tools import BioMCP

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ENTITY_ONLY = {k: ([] if k not in ("study_type", "sample_count")
                   else ("" if k == "study_type" else None)) for k in FIELDS}


def _load(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _split_and_write(records: List[Dict[str, Any]]) -> None:
    """Write the three stage files from the fully-enriched record set."""
    stage1, stage2 = [], []
    for r in records:
        base = {k: r.get(k) for k in ("id", "pmid", "title", "abstract", "journal",
                                      "year", "url")}
        for k in FIELDS:
            base[k] = r.get(k, ENTITY_ONLY[k])
        stage1.append(dict(base))
        stage2.append({**base, "relationships": r.get("relationships", [])})
    ENTITIES_FILE.write_text(json.dumps(stage1, ensure_ascii=False, indent=2), encoding="utf-8")
    RELATIONSHIPS_FILE.write_text(json.dumps(stage2, ensure_ascii=False, indent=2), encoding="utf-8")
    STRUCTURES_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def build_kb(args) -> List[Dict[str, Any]]:
    from swarm.agents import pubmed_extractor

    existing = _load(STRUCTURES_FILE)
    by_pmid = {str(r.get("pmid")): r for r in existing if r.get("pmid")}
    next_id = max((r.get("id", 0) for r in existing), default=0) + 1
    print(f"[build] {len(existing)} papers already in KB (will be skipped).")

    # parse/validate --queries (1-based indices into QUERIES). Fail loudly on garbage or
    # out-of-range values instead of silently falling back to running the whole corpus.
    raw = [x.strip() for x in args.queries.split(",") if x.strip()]
    sel = {int(x) for x in raw if x.isdigit()}
    valid = set(range(1, len(QUERIES) + 1))
    if any(not x.isdigit() for x in raw) or not sel or not (sel <= valid):
        sys.exit(f"[ERROR] --queries must be comma-separated numbers in 1..{len(QUERIES)} "
                 f"(got {args.queries!r}).")
    date_filter = ""
    if args.year_from or args.year_to:
        lo, hi = args.year_from or 1800, args.year_to or 3000
        date_filter = f" AND {lo}:{hi}[pdat]"

    papers = pubmed_extractor.run(args.max_results, sel, set(by_pmid.keys()), date_filter)
    if args.limit:
        papers = papers[:args.limit]
    if not papers:
        print("[build] nothing new - KB is up to date.")
        return existing

    graph = publication_swarm.get_graph()
    records = list(existing)
    try:
        for i, paper in enumerate(papers, 1):
            final = graph.invoke({"paper": paper, "judge_retries": 0, "errors": []})
            rec = publication_swarm.assemble_record(paper, final, next_id)
            records.append(rec)
            by_pmid[str(paper["pmid"])] = rec
            next_id += 1
            n_ent = sum(len(rec[f]) for f in FIELDS if isinstance(rec.get(f), list))
            print(f"  [{i}/{len(papers)}] PMID {paper['pmid']}: "
                  f"{n_ent} entities | {len(rec['relationships'])} rels | "
                  f"{rec['n_structures']} structures")
            if i % 10 == 0:
                _split_and_write(records)
    finally:
        BioMCP.stop()

    _split_and_write(records)
    print(f"[build] done · {len(records)} papers total.")

    if not args.no_postprocess:
        _postprocess()
    return records


def _postprocess() -> None:
    """Build the protein-centric view + synonym maps."""
    for script in ("normalize.py", "build_proteins.py"):
        path = ROOT / script
        if path.exists():
            print(f"[build] running {script} …")
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit:
                pass
            except Exception as e:
                print(f"  [WARN] {script}: {e}")


def main():
    ap = argparse.ArgumentParser(description="kb_secretion_swarm - build the KB from PubMed")
    ap.add_argument("--max-results", type=int, default=200, help="PubMed results per query.")
    ap.add_argument("--queries", default="1,2",
                    help="Comma-separated query numbers to run (1=Plasmodium export, "
                         "2=ESX/Type-VII; query 3 antimalarial-resistance is disabled in config).")
    ap.add_argument("--limit", type=int, default=None, help="Cap new papers processed (smoke test).")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    ap.add_argument("--no-postprocess", action="store_true",
                    help="Skip normalize.py / build_proteins.py after the build.")
    args = ap.parse_args()

    if not (have_openai() or have_anthropic()):
        sys.exit("[ERROR] No LLM key set. Put OPENAI_API_KEY (and/or ANTHROPIC_API_KEY) in ./.env")
    print(f"[swarm] LLM keys - OpenAI: {have_openai()} · Anthropic: {have_anthropic()}")
    build_kb(args)


if __name__ == "__main__":
    main()
