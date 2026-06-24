"""
demo_q3_2026.py - standalone Q3 (antimalarial drugs & resistance) demo dataset.

Antimalarial drug resistance stood out as a prominent topic while studying the
literature for this project, so this demo was built. It runs the Q3 query (which is
OFF in the main config, so it never pollutes the Q1+Q2 KB) on its own, through the
SAME per-paper graph (entities -> relationships -> judge -> structures), and writes a
SEPARATE file. It was run for 2026, and that dataset is shown in the dashboard.

Output: data/q3_antimalarial_2026.json (same schema as the main KB; selectable in the
        dashboard's Dataset picker), plus its _proteins.json and _synonyms.json companions
        (built automatically at the end, the same way run_swarm builds the main ones).

Run:
    python demo_q3_2026.py                       # default 100 papers, year 2026 only
    python demo_q3_2026.py --max-results 50 --year-from 2025 --year-to 2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from swarm.config import DATA_DIR, FIELDS, ROOT
from swarm import publication_swarm
from swarm.tools import BioMCP, fetch_pubmed, pubmed_client, search_pubmed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Q3 - antimalarial drugs & resistance (verbatim from the disabled config entry)
Q3 = ('("antimalarial"[Title/Abstract] OR "artemisinin"[Title/Abstract] '
      'OR "chloroquine resistance"[Title/Abstract] OR "kelch13"[Title/Abstract] '
      'OR "pfcrt"[Title/Abstract] OR "pfmdr1"[Title/Abstract])')
OUT_FILE = DATA_DIR / "q3_antimalarial_2026.json"


def _postprocess_q3() -> None:
    """Build the Q3 protein view + synonyms so the demo is fully reproducible.

    run_swarm's _postprocess only targets the main KB, so we run the same two scripts
    here pointed at the Q3 files (both accept --in/--out and resolve names into data/).
    """
    base = OUT_FILE.name                                  # q3_antimalarial_2026.json
    for script, out in (("build_proteins.py", base.replace(".json", "_proteins.json")),
                        ("normalize.py", base.replace(".json", "_synonyms.json"))):
        path = ROOT / script
        if not path.exists():
            continue
        print(f"[Q3] running {script} -> {out}")
        r = subprocess.run([sys.executable, str(path), "--in", base, "--out", out])
        if r.returncode != 0:
            print(f"  [WARN] {script} exited with {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description="Q3 antimalarial-resistance demo dataset.")
    ap.add_argument("--max-results", type=int, default=100)
    ap.add_argument("--year-from", type=int, default=2026)
    ap.add_argument("--year-to", type=int, default=2026)
    args = ap.parse_args()

    term = f"{Q3} AND {args.year_from}:{args.year_to}[pdat]"
    E = pubmed_client()
    total = int(E.read(E.esearch(db="pubmed", term=term, retmax=0))["Count"])
    ids = search_pubmed(E, term, args.max_results)
    print(f"[Q3] {total} papers match {args.year_from}-{args.year_to}; "
          f"fetching top {len(ids)} by relevance.")

    # resume-friendly: skip anything already in this demo file
    existing = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else []
    done = {str(r.get("pmid")) for r in existing}
    next_id = max((r.get("id", 0) for r in existing), default=0) + 1

    papers = [p for p in fetch_pubmed(E, [i for i in ids if i not in done])
              if p.get("title") or p.get("abstract")]
    print(f"[Q3] {len(papers)} new papers to process (skipping {len(ids) - len(papers)}).")

    graph = publication_swarm.get_graph()
    records = list(existing)
    try:
        for i, paper in enumerate(papers, 1):
            final = graph.invoke({"paper": paper, "judge_retries": 0, "errors": []})
            rec = publication_swarm.assemble_record(paper, final, next_id)
            records.append(rec); next_id += 1
            n_ent = sum(len(rec[f]) for f in FIELDS if isinstance(rec.get(f), list))
            print(f"  [{i}/{len(papers)}] PMID {paper['pmid']}: {n_ent} entities | "
                  f"{len(rec['relationships'])} rels | {rec['n_structures']} structures")
            if i % 10 == 0:
                OUT_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        BioMCP.stop()

    OUT_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    n_rel = sum(len(r.get("relationships") or []) for r in records)
    n_str = sum(r.get("n_structures", 0) for r in records)
    print(f"\n[DONE] {len(records)} Q3 papers -> {OUT_FILE.name}")
    print(f"   relationships: {n_rel} | resolved structures: {n_str}")

    _postprocess_q3()


if __name__ == "__main__":
    main()
