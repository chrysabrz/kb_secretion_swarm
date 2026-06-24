"""
build_proteins.py - turn the papers KB into a protein-centric view (proteins.json).

Reads secretion_systems_with_structures.json (a list of papers) and pivots it into
one entry per resolved protein: its UniProt info, PDB/AlphaFold links, every
relationship it appears in, and the papers that mention it.
So, it reorganizes the already-resolved structures: it reads each paper's structures and groups them by protein into proteins.json.

Run:
    python build_proteins.py                        # data/...structures.json -> data/proteins.json
    python build_proteins.py --in X.json --out Y.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
IN_FILE = DATA / "secretion_systems_with_structures.json"
OUT_FILE = DATA / "proteins.json"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Assemble a protein-centric view of a KB file.")
    ap.add_argument("--in", dest="in_file", default=str(IN_FILE),
                    help="structures JSON to read (default secretion_systems_with_structures.json).")
    ap.add_argument("--out", dest="out_file", default=str(OUT_FILE),
                    help="protein-view JSON to write (default proteins.json).")
    args = ap.parse_args()
    # resolve bare names into data/ (absolute paths are honored as-is)
    in_file = Path(args.in_file) if Path(args.in_file).is_absolute() else DATA / Path(args.in_file).name
    out_file = Path(args.out_file) if Path(args.out_file).is_absolute() else DATA / Path(args.out_file).name

    if not in_file.exists():
        sys.exit(f"[ERROR] {in_file.name} not found - run run_swarm.py first.")
    records = json.loads(in_file.read_text(encoding="utf-8"))
    print(f"[INFO] reading {in_file.name} ({len(records)} papers)")

    proteins: dict = {}              # accession -> aggregate entry
    name2acc: dict = {}             # lowercased entity name -> set of accessions

    # 1) gather resolved proteins + which papers mention them
    for r in records:
        for s in r.get("structures") or []:
            if not s.get("resolved"):
                continue
            up = s.get("uniprot") or {}
            acc = up.get("accession") or s.get("entity")
            if not acc:
                continue
            p = proteins.get(acc)
            if not p:
                p = {
                    "accession": acc,
                    "entity_names": set(),
                    "uniprot": up,
                    "pdb": s.get("pdb") or [],
                    "alphafold": s.get("alphafold"),
                    "confidence": s.get("confidence"),   # gene_exact | name_match
                    "_papers": {},          # pmid -> title
                    "_rels": {},            # (subj,pred,obj) -> set(pmid)
                    "_relconf": {},         # (subj,pred,obj) -> [confidence, ...]
                }
                proteins[acc] = p
            ent = str(s.get("entity", "")).strip()
            if ent:
                p["entity_names"].add(ent)
                name2acc.setdefault(ent.lower(), set()).add(acc)
            if r.get("pmid"):
                p["_papers"][str(r["pmid"])] = r.get("title", "")

    # 2) attach the extracted relationships that involve each protein's names
    for r in records:
        for t in r.get("relationships") or []:
            if not isinstance(t, dict):
                continue
            subj, obj = str(t.get("subject", "")), str(t.get("object", ""))
            pred = t.get("predicate", "")
            pmid = str(t.get("pmid") or r.get("pmid") or "")
            accs = name2acc.get(subj.lower(), set()) | name2acc.get(obj.lower(), set())
            for acc in accs:
                key = (subj, pred, obj)
                proteins[acc]["_rels"].setdefault(key, set())
                if pmid:
                    proteins[acc]["_rels"][key].add(pmid)
                c = t.get("confidence")
                if isinstance(c, (int, float)):
                    proteins[acc]["_relconf"].setdefault(key, []).append(float(c))

    # 3) finalise shapes
    out = []
    for acc, p in proteins.items():
        rels = [{"subject": k[0], "predicate": k[1], "object": k[2],
                 "pmids": sorted(v),
                 # aggregate confidence across the papers asserting this triple:
                 # the strongest support (max) any single paper gives it.
                 "confidence": (round(max(p["_relconf"][k]), 3)
                                if p["_relconf"].get(k) else None)}
                for k, v in p["_rels"].items()]
        rels.sort(key=lambda x: (-len(x["pmids"]), x["predicate"]))
        papers = [{"pmid": pid, "title": ttl} for pid, ttl in p["_papers"].items()]
        out.append({
            "accession": acc,
            "entity_names": sorted(p["entity_names"]),
            "uniprot": p["uniprot"],
            "pdb": p["pdb"],
            "alphafold": p["alphafold"],
            "confidence": p.get("confidence"),
            "n_papers": len(papers),
            "n_relationships": len(rels),
            "papers": papers,
            "relationships": rels,
        })
    out.sort(key=lambda x: (-x["n_papers"], -x["n_relationships"]))
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    n_pdb = sum(1 for p in out if p["pdb"])
    n_rel = sum(p["n_relationships"] for p in out)
    print(f"\n[DONE] {len(out)} proteins -> {out_file.name}")
    print(f"   with experimental PDB : {n_pdb}")
    print(f"   total relationships attached : {n_rel}")


if __name__ == "__main__":
    main()
