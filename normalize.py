"""
normalize.py - build synonyms.json: a raw -> canonical name map per field.

It does NOT change the data - it just produces a mapping the dashboard uses (via the
"Normalize names" toggle) to merge spelling variants (e.g. k13 / kelch13). Two rules:
  * species: expand abbreviations ("P." -> "Plasmodium") and drop strain suffixes
    ("Plasmodium berghei ANKA" -> "Plasmodium berghei").
  * other fields: group case/punctuation-insensitive variants and map each to the
    most common spelling (e.g. "esxA" / "EsxA" / "ESX-A" -> the commonest).

The map lists EVERY value the extractor produced, so most rows are identity
(key == value): those are names that are already canonical or have no variants
(e.g. "PTEX": "PTEX"). Only rows where key != value are real merges
(e.g. "ptex150" -> "PTEX150", "Hsp101" -> "HSP101"). Identity rows are harmless -
the dashboard falls back to the raw value anyway - they just make the file a
complete audit of every value seen.

Re-runnable and merge-preserving: existing synonyms.json entries are kept, so hand
edits survive a re-run.

Run:
    python normalize.py                        # data/...structures.json -> data/synonyms.json
    python normalize.py --in X.json --out Y.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
KB_FILES = ["secretion_systems_with_structures.json", "secretion_systems_with_relationships.json", "secretion_systems.json"]
OUT_FILE = DATA / "synonyms.json"

FIELDS = [
    "species", "secretion_components", "exported_proteins", "genes", "drugs",
    "resistance_markers", "life_cycle_stages", "clinical_outcomes", "methods",
]

GENUS_ABBR = {
    "p.": "Plasmodium", "m.": "Mycobacterium", "t.": "Toxoplasma", "b.": "Babesia",
    "s.": "Staphylococcus", "c.": "Cryptosporidium", "e.": "Escherichia",
}
_GENUS_QUALIFIER = {"spp.", "sp.", "spp", "sp", "species"}


def load_kb(kb_files):
    for name in kb_files:
        f = DATA / Path(name).name          # always resolve into data/ by basename
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8")), f.name
    sys.exit("[ERROR] No KB file found - run the extraction stages first.")


def canon_species(v: str) -> str:
    parts = v.strip().split()
    if not parts:
        return v.strip()
    first = parts[0].lower()
    if first in GENUS_ABBR:                       # "P." -> "Plasmodium"
        parts[0] = GENUS_ABBR[first]
    if len(parts) >= 2 and parts[1].lower() in _GENUS_QUALIFIER:
        return parts[0]                           # "Plasmodium spp." -> "Plasmodium"
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts[:2])                    # drop strain: keep genus + species


# For the genes field, a trailing "gene"/"genes" is a category qualifier, not part
# of the name ("var genes" == "var"). Stripping it before keying so those group together.
# Scoped to genes ONLY: in protein fields, "...protein" is usually part of the real
# name (e.g. "histidine-rich protein"), so we must NOT strip there.
_GENE_QUALIFIER = re.compile(r"\s+genes?$", re.IGNORECASE)


def key_generic(v: str, field: str | None = None) -> str:
    s = v.strip()
    if field == "genes":
        s = _GENE_QUALIFIER.sub("", s).strip() or s
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build(kb_files=None, out_file=None):
    kb_files = kb_files or KB_FILES
    out_file = DATA / Path(out_file).name if out_file else OUT_FILE   # output into data/
    records, src = load_kb(kb_files)
    print(f"[INFO] reading {src} ({len(records)} papers)")

    existing = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    syn = {f: dict(existing.get(f, {})) for f in FIELDS}

    for f in FIELDS:
        vals: Counter = Counter()
        for r in records:
            for v in r.get(f) or []:
                if str(v).strip():
                    vals[str(v).strip()] += 1
        if not vals:
            continue
        if f == "species":
            for v in vals:
                syn[f].setdefault(v, canon_species(v))
        else:
            groups: dict = defaultdict(list)
            for v in vals:
                groups[key_generic(v, f)].append(v)
            for members in groups.values():
                # canonical = most frequent, then shortest, then alphabetical (deterministic)
                canonical = sorted(members, key=lambda m: (-vals[m], len(m), m))[0]
                for m in members:
                    # every member is mapped, incl. the canonical itself -> identity rows
                    # (key == value) for variant-free / already-canonical names.
                    syn[f].setdefault(m, canonical)

    # Leading note so the JSON is self-documenting (ignored by the dashboard, which only
    # reads the per-field maps). See this file's module docstring for the full rules.
    out = {"_comment": ("raw name -> canonical spelling, per field. Rows where key == value "
                        "are already-canonical / variant-free names; only key != value rows "
                        "are real merges (e.g. 'ptex150' -> 'PTEX150'). Used by the dashboard's "
                        "'Normalize names' toggle; never alters source data. Regenerated by normalize.py."),
           **syn}
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] wrote {out_file.name}")
    for f in FIELDS:
        raw = len(syn[f])
        canon = len(set(syn[f].values()))
        if raw:
            print(f"   {f:22} {raw:4} raw -> {canon:4} canonical "
                  f"({raw - canon} merged)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build raw->canonical synonym dictionaries for a KB.")
    ap.add_argument("--in", dest="in_file", default=None,
                    help="KB JSON to read (default: most-enriched secretion_systems file).")
    ap.add_argument("--out", dest="out_file", default=None,
                    help="synonyms JSON to write (default synonyms.json).")
    a = ap.parse_args()
    build([a.in_file] if a.in_file else None, a.out_file)
