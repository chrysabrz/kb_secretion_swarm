"""
score_confidence.py - attach a judge `confidence` to EVERY existing relationship.

The original corpus (papers imported from the scripted pipeline) carries typed
relationships with NO confidence, while papers added by the swarm carry a
judge-scored `confidence`. This utility makes the score uniform across the whole
KB WITHOUT changing the relationship set: it runs the ValidationAgent (LLM judge)
over each paper's *existing* triples, attaches the median-of-votes confidence to
each one, and keeps them all (no acceptance gate, nothing dropped).

It is idempotent: papers whose triples already carry `confidence` are skipped
(use --force to rescore). It updates both relationship-bearing KB files in lockstep
and rebuilds the protein view.

Run:
    python score_confidence.py                 # score every paper missing confidence
    python score_confidence.py --limit 3       # smoke test
    python score_confidence.py --force         # rescore everything
"""

from __future__ import annotations

import argparse
import json
import runpy
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

from swarm.agents import validator
from swarm.config import (RELATIONSHIPS_FILE, ROOT, STRUCTURES_FILE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _needs_scoring(rels: List[Dict[str, Any]]) -> bool:
    return bool(rels) and any("confidence" not in t for t in rels if isinstance(t, dict))


def _score_paper(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the paper's triples with a `confidence` attached to each (median over
    SWARM_JUDGE_VOTES judge passes). All triples are kept; none are dropped."""
    rels = paper.get("relationships") or []
    if not rels:
        return rels
    title, abstract = paper.get("title", ""), paper.get("abstract", "")
    per_idx: Dict[int, List[float]] = {i: [] for i in range(len(rels))}
    try:
        for vote in range(max(1, validator.JUDGE_VOTES)):
            verdicts = validator._judge_once(vote, title, abstract, rels)
            for i in range(len(rels)):
                v = verdicts.get(i)
                if v:
                    per_idx[i].append(float(v["confidence"]))
    except Exception as e:
        print(f"    [WARN] judge failed for PMID {paper.get('pmid')}: {e}")
        # Return triples UNTOUCHED: don't add a confidence key to ones that lack it, or
        # _needs_scoring would treat them as scored and skip them on every future run.
        return rels

    out = []
    for i, t in enumerate(rels):
        confs = per_idx[i]
        conf = round(statistics.median(confs), 3) if confs else None
        out.append({**t, "confidence": conf})
    return out


def main():
    ap = argparse.ArgumentParser(description="Attach judge confidence to every existing relationship.")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N papers needing scoring.")
    ap.add_argument("--force", action="store_true", help="Rescore even papers that already have confidence.")
    args = ap.parse_args()

    if not STRUCTURES_FILE.exists():
        sys.exit(f"[ERROR] {STRUCTURES_FILE.name} not found - run the pipeline first.")
    records = json.loads(STRUCTURES_FILE.read_text(encoding="utf-8"))

    todo = [r for r in records
            if (r.get("relationships") and (args.force or _needs_scoring(r["relationships"])))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[INFO] {len(records)} papers; scoring relationships for {len(todo)} "
          f"(votes={validator.JUDGE_VOTES}).")

    n_done = n_triples = 0
    for r in todo:
        r["relationships"] = _score_paper(r)
        n_done += 1
        n_triples += len(r["relationships"])
        if n_done % 25 == 0:
            STRUCTURES_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{n_done}/{len(todo)}] checkpoint ({n_triples} triples scored)")

    STRUCTURES_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # sync the relationships-stage file from the (authoritative) structures file
    if RELATIONSHIPS_FILE.exists():
        rel_recs = json.loads(RELATIONSHIPS_FILE.read_text(encoding="utf-8"))
        by_id = {r.get("id"): r for r in records}
        for p in rel_recs:
            src = by_id.get(p.get("id"))
            if src is not None:
                p["relationships"] = src.get("relationships", [])
        RELATIONSHIPS_FILE.write_text(json.dumps(rel_recs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] scored {n_done} papers, {n_triples} triples -> updated "
          f"{STRUCTURES_FILE.name} + {RELATIONSHIPS_FILE.name}")

    # rebuild the protein view so its attached relationships carry confidence too
    bp = ROOT / "build_proteins.py"
    if bp.exists() and not args.limit:
        print("[INFO] rebuilding protein view (build_proteins.py) …")
        try:
            runpy.run_path(str(bp), run_name="__main__")
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
