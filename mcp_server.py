"""
mcp_server.py - an MCP server that fetches protein structure data.

It uses only Python's standard library (nothing to install) and talks to the client
by exchanging JSON request/reply messages over stdin/stdout. It exposes three tools:

    * uniprot_search   - protein/gene -> UniProt accession, name, function, PDB cross-refs
                         (filtered to a taxonomy id; default 5820 = Plasmodium).
    * pdb_search       - full-text RCSB PDB search -> structures + 3D-viewer URL.
    * alphafold_lookup - UniProt accession -> AlphaFold predicted model.

You don't run this by hand - mcp_client.py starts it for you. To test it directly:
    python mcp_client.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"
ALPHAFOLD = "https://alphafold.ebi.ac.uk/api/prediction/{}"
PLASMODIUM_TAXID = 5820                      # genus Plasmodium
UA = "kb-secretion-swarm-mcp/1.0"
SERVER_INFO = {"name": "kb-secretion-swarm-bio-mcp", "version": "1.0.0"}
_LAST = {"t": 0.0}


def _log(*a):
    print("[kb-secretion-swarm-mcp]", *a, file=sys.stderr, flush=True)


def _throttle(gap=0.34):
    dt = time.time() - _LAST["t"]
    if dt < gap:
        time.sleep(gap - dt)
    _LAST["t"] = time.time()


def _get(url, timeout=20):
    _throttle()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw and raw.strip() else None


def _post(url, payload, timeout=25):
    _throttle()
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"User-Agent": UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw and raw.strip() else None


# tools
def _comment_texts(result, ctype):
    out = []
    for c in result.get("comments", []) or []:
        if c.get("commentType") == ctype:
            for t in c.get("texts") or []:
                if t.get("value"):
                    out.append(t["value"])
    return out


def _function_summary(result) -> str:
    t = _comment_texts(result, "FUNCTION")
    return t[0] if t else ""


def _subcellular_locations(result):
    locs = []
    for c in result.get("comments", []) or []:
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for sl in c.get("subcellularLocations", []) or []:
                v = (sl.get("location") or {}).get("value")
                if v:
                    locs.append(v)
    return locs


def _xrefs_by_db(result):
    """Group cross-references: returns {db: [{'id':..,'name':..}]}"""
    by = {}
    for x in result.get("uniProtKBCrossReferences", []) or []:
        db = x.get("database")
        nm = ""
        for p in x.get("properties", []) or []:
            if p.get("key") in ("GoTerm", "EntryName", "PathwayName", "GeneName"):
                nm = p.get("value", "")
                break
        by.setdefault(db, []).append({"id": x.get("id"), "name": nm})
    return by


def _features(result):
    """Summarise sequence features (transmembrane / signal peptide / topological)."""
    counts = {}
    for f in result.get("features", []) or []:
        t = f.get("type")
        if t:
            counts[t] = counts.get(t, 0) + 1
    return counts


def _protein_name(pd_):
    """Best protein name: recommendedName (Swiss-Prot), else submissionName
    (TrEMBL), else first alternativeName. Avoids empty names on TrEMBL entries."""
    rn = ((pd_.get("recommendedName") or {}).get("fullName") or {}).get("value")
    if rn:
        return rn
    for key in ("submissionNames", "alternativeNames"):
        for n in pd_.get(key, []) or []:
            if (n.get("fullName") or {}).get("value"):
                return n["fullName"]["value"]
    return ""


def _synonyms(result):
    """All names this protein is known by: recommended/alt protein names + short
    names + gene name + gene synonyms + ordered-locus names. Used to validate that
    a search hit actually IS the queried entity (avoids wrong-protein mappings)."""
    out = []
    pd_ = result.get("proteinDescription") or {}
    rn = pd_.get("recommendedName") or {}
    if (rn.get("fullName") or {}).get("value"):
        out.append(rn["fullName"]["value"])
    for sn in rn.get("shortNames", []) or []:
        if sn.get("value"):
            out.append(sn["value"])
    for alt in pd_.get("alternativeNames", []) or []:
        if (alt.get("fullName") or {}).get("value"):
            out.append(alt["fullName"]["value"])
        for sn in alt.get("shortNames", []) or []:
            if sn.get("value"):
                out.append(sn["value"])
    for g in result.get("genes", []) or []:
        if (g.get("geneName") or {}).get("value"):
            out.append(g["geneName"]["value"])
        for syn in g.get("synonyms", []) or []:
            if syn.get("value"):
                out.append(syn["value"])
        for on in g.get("orderedLocusNames", []) or []:
            if on.get("value"):
                out.append(on["value"])
    return out


def tool_uniprot_search(query, size=5, taxonomy_id=PLASMODIUM_TAXID):
    q = f"({query})"
    if taxonomy_id:
        q += f" AND (taxonomy_id:{int(taxonomy_id)})"
    params = {
        "query": q, "format": "json", "size": int(size),
        "fields": ("accession,id,protein_name,gene_names,organism_name,cc_function,xref_pdb,"
                   "cc_subcellular_location,go_c,go_f,go_p,keyword,xref_pfam,xref_interpro,"
                   "ft_transmem,ft_signal,length,mass,cc_similarity,cc_catalytic_activity,"
                   "ec,cc_pathway,xref_kegg"),
    }
    d = _get(UNIPROT + "?" + urllib.parse.urlencode(params)) or {}
    out = []
    for r in d.get("results", []):
        acc = r.get("primaryAccession")
        pd_ = r.get("proteinDescription") or {}
        name = _protein_name(pd_)
        genes = [g.get("geneName", {}).get("value", "") for g in (r.get("genes") or [])]
        org = ((r.get("organism") or {}).get("scientificName")) or ""
        xr = _xrefs_by_db(r)
        go_lists = {"C": [], "F": [], "P": []}
        for p in xr.get("GO", []):
            nm = p.get("name", "")
            if nm[1:2] == ":":
                go_lists.setdefault(nm[0], []).append(nm[2:])
        seq = r.get("sequence") or {}
        ecs = [e.get("value") for e in (pd_.get("recommendedName") or {}).get("ecNumbers", []) or []]
        feats = _features(r)
        out.append({
            "accession": acc,
            "protein_name": name,
            "gene": next((g for g in genes if g), ""),
            "organism": org,
            "function_summary": _function_summary(r),
            "pdb_xrefs": [x["id"] for x in xr.get("PDB", [])][:25],
            "url": f"https://www.uniprot.org/uniprotkb/{acc}/entry" if acc else "",
            # rich fields
            "subcellular_locations": _subcellular_locations(r),
            "go_component": go_lists.get("C", [])[:12],
            "go_process": go_lists.get("P", [])[:12],
            "go_function": go_lists.get("F", [])[:12],
            "keywords": [k.get("name") for k in r.get("keywords", []) or [] if k.get("name")],
            "pfam": [f"{p['id']} {p['name']}".strip() for p in xr.get("Pfam", [])],
            "interpro": [f"{p['id']} {p['name']}".strip() for p in xr.get("InterPro", [])][:10],
            "length": seq.get("length"),
            "mass": seq.get("molWeight"),
            "families": _comment_texts(r, "SIMILARITY"),
            "catalytic_activity": _comment_texts(r, "CATALYTIC ACTIVITY")[:3],
            "ec_numbers": ecs,
            "pathway": _comment_texts(r, "PATHWAY"),
            "features": feats,
            "kegg": [p["id"] for p in xr.get("KEGG", [])][:3],
            "synonyms": _synonyms(r),
            "reviewed": ("reviewed" in (r.get("entryType", "") or "").lower()
                         and "unreviewed" not in (r.get("entryType", "") or "").lower()),
        })
    return out


def _entry_detail(pdb_id):
    e = None
    try:
        e = _get(RCSB_ENTRY.format(pdb_id))
    except Exception:
        e = None
    base = {"pdb_id": pdb_id, "url": f"https://www.rcsb.org/structure/{pdb_id}",
            "view3d": f"https://www.rcsb.org/3d-view/{pdb_id}"}
    if not e:
        return base
    res = (e.get("rcsb_entry_info") or {}).get("resolution_combined")
    base.update({"title": (e.get("struct") or {}).get("title", ""),
                 "method": ((e.get("exptl") or [{}])[0]).get("method", ""),
                 "resolution": (res[0] if isinstance(res, list) and res else res)})
    return base


def tool_pdb_search(query, size=5, experimental_only=True, detail=3):
    body = {"query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": int(size)}}}
    if experimental_only:
        body["request_options"]["results_content_type"] = ["experimental"]
    try:
        d = _post(RCSB_SEARCH, body)
    except urllib.error.HTTPError as e:
        if e.code in (204, 404):
            return []
        raise
    if not d:
        return []
    ids = [x["identifier"] for x in d.get("result_set", [])]
    return [_entry_detail(p) if i < detail else
            {"pdb_id": p, "url": f"https://www.rcsb.org/structure/{p}",
             "view3d": f"https://www.rcsb.org/3d-view/{p}"} for i, p in enumerate(ids)]


def tool_alphafold_lookup(accession):
    try:
        a = _get(ALPHAFOLD.format(accession))
    except Exception:
        a = None
    if not a:
        return {}
    m = a[0]
    return {"accession": accession, "model_cif": m.get("cifUrl", ""), "model_pdb": m.get("pdbUrl", ""),
            "mean_plddt": m.get("globalMetricValue"),
            "viewer": f"https://alphafold.ebi.ac.uk/entry/{accession}"}


TOOLS = {
    "uniprot_search": {"fn": tool_uniprot_search, "schema": {
        "name": "uniprot_search",
        "description": "Search UniProtKB (default taxonomy 5820 = Plasmodium); returns Primary "
                       "Accession, Protein Name, Function Summary, organism and PDB cross-refs.",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string"}, "size": {"type": "integer", "default": 5},
            "taxonomy_id": {"type": "integer", "default": PLASMODIUM_TAXID}},
            "required": ["query"]}}},
    "pdb_search": {"fn": tool_pdb_search, "schema": {
        "name": "pdb_search",
        "description": "Full-text RCSB PDB search; returns structures with title/method/resolution "
                       "and a 3D-viewer URL.",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string"}, "size": {"type": "integer", "default": 5}},
            "required": ["query"]}}},
    "alphafold_lookup": {"fn": tool_alphafold_lookup, "schema": {
        "name": "alphafold_lookup",
        "description": "Look up an AlphaFold predicted model by UniProt accession.",
        "inputSchema": {"type": "object", "properties": {"accession": {"type": "string"}},
                        "required": ["accession"]}}},
}


def _content(obj):
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


def _run_tool(name, args):
    spec = TOOLS.get(name)
    if not spec:
        return {"content": [{"type": "text", "text": f"Error: unknown tool '{name}'"}], "isError": True}
    try:
        return _content(spec["fn"](**(args or {})))
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}], "isError": True}


def _handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [t["schema"] for t in TOOLS.values()]}}
    if method == "tools/call":
        p = msg.get("params", {})
        return {"jsonrpc": "2.0", "id": mid, "result": _run_tool(p.get("name"), p.get("arguments"))}
    if method == "call_tool":                          # legacy: top-level content
        out = dict(_run_tool(msg.get("tool_name"), msg.get("arguments")))
        if mid is not None:
            out["id"] = mid
        return out
    if method and method.startswith("notifications/"):
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main():
    _log("ready - tools:", ", ".join(TOOLS))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32700, "message": "Parse error"}}) + "\n")
            sys.stdout.flush()
            continue
        resp = _handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
