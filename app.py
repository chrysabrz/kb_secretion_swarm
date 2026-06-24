"""
app.py - Streamlit dashboard for the knowledge base.

A sidebar picker switches between datasets (Main KB, Q3 demo); a "Normalize names"
toggle merges spelling variants via synonyms.json. Seven pages:

    * Overview       - headline metrics + publications/year, journals, study mix
    * Field explorer - top values per entity field, with drill-down
    * Relationships  - typed subject-predicate-object triples, filterable
    * Structures     - UniProt / PDB / AlphaFold links
    * Proteins       - one card per protein (UniProt biology + its relationships + papers)
    * Papers         - searchable table; open a paper for its full record
    * QA             - retrieval-augmented Q&A (qa.py), cited by PMID

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from qa import QA

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"          # generated data files live here (committed)

# Selectable datasets (sidebar picker). Each entry: candidate KB files (most-enriched
# first), its protein-view file, and its synonyms file (None = no normalization).
# A dataset only appears in the picker if at least one of its KB files exists.
DATASETS = {
    "Main KB · Plasmodium export + ESX (Q1+Q2)": {
        "kb": ["secretion_systems_with_structures.json",
               "secretion_systems_with_relationships.json",
               "secretion_systems.json"],
        "proteins": "proteins.json",
        "synonyms": "synonyms.json",
    },
    "Q3 · antimalarial drugs & resistance (2026)": {
        "kb": ["q3_antimalarial_2026.json"],
        "proteins": "q3_antimalarial_2026_proteins.json",
        "synonyms": "q3_antimalarial_2026_synonyms.json",
    },
}

LIST_FIELDS = [
    "species", "secretion_components", "exported_proteins", "genes", "drugs",
    "resistance_markers", "life_cycle_stages", "clinical_outcomes", "methods",
]

st.set_page_config(page_title="Malaria · ESX knowledge base", page_icon="🧬", layout="wide")

# theme / palette
# Light + stylish: white background, gradient title, neon-accent metric cards,
# colourful charts. Scientific yet stylish (not a dark theme).
SCI_SEQ = ["#3b82f6", "#a855f7", "#10b981", "#ec4899", "#f59e0b",
           "#06b6d4", "#f43f5e", "#8b5cf6", "#84cc16"]
SCI_SCALE = ["#dbeafe", "#bfdbfe", "#93c5fd", "#3b82f6", "#2563eb"]
px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = SCI_SEQ

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
  :root{ --line:#e6e9f2; --txt:#0f172a; --mut:#64748b; }
  .stApp{ background:#fbfcff; }
  html,body,[class*="css"],.stMarkdown,button,input,textarea,select{ font-family:'Inter',sans-serif; }
  h1{ font-family:'Space Grotesk',sans-serif; font-weight:700 !important; letter-spacing:-.5px;
      background:linear-gradient(90deg,#1e3a8a 0%,#2563eb 45%,#38bdf8 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  h2,h3{ font-family:'Space Grotesk',sans-serif; color:#1e293b !important; font-weight:600 !important; }
  [data-testid="stSidebar"]{ background:#f5f7fc; border-right:1px solid var(--line); }
  [data-testid="stMetric"]{ background:#ffffff; border:1px solid var(--line); border-radius:12px;
      padding:18px 18px 14px; position:relative; overflow:hidden;
      box-shadow:0 2px 10px rgba(30,41,59,.06); }
  [data-testid="stMetric"]::before{ content:""; position:absolute; top:0; left:0; right:0;
      height:3px; background:linear-gradient(90deg,#3b82f6,#a855f7); }
  [data-testid="stHorizontalBlock"]>[data-testid="column"]:nth-child(4n+1) [data-testid="stMetric"]::before{ background:linear-gradient(90deg,#3b82f6,#06b6d4); }
  [data-testid="stHorizontalBlock"]>[data-testid="column"]:nth-child(4n+2) [data-testid="stMetric"]::before{ background:linear-gradient(90deg,#10b981,#06b6d4); }
  [data-testid="stHorizontalBlock"]>[data-testid="column"]:nth-child(4n+3) [data-testid="stMetric"]::before{ background:linear-gradient(90deg,#f43f5e,#ec4899); }
  [data-testid="stHorizontalBlock"]>[data-testid="column"]:nth-child(4n+4) [data-testid="stMetric"]::before{ background:linear-gradient(90deg,#f59e0b,#f43f5e); }
  [data-testid="stMetricValue"]{ font-family:'Space Grotesk',sans-serif; color:#0f172a !important; font-weight:700; }
  [data-testid="stMetricLabel"] *{ color:var(--mut) !important; text-transform:uppercase;
      font-size:11px; letter-spacing:.6px; }
  a,a:visited{ color:#7c3aed !important; }
  div.stButton>button,.stDownloadButton>button{ background:linear-gradient(90deg,#3b82f6,#a855f7);
      color:#fff; border:none; border-radius:8px; font-weight:600; }
  [data-testid="stExpander"]{ background:#ffffff; border:1px solid var(--line); border-radius:10px;
      box-shadow:0 1px 3px rgba(30,41,59,.04); }
  hr{ border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


# data
@st.cache_data(show_spinner=False)
def load_kb(kb_files: tuple):
    for name in kb_files:
        f = DATA / name
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8")), name
    return [], ""


@st.cache_resource(show_spinner="Building Q&A index…")
def get_qa(source_name: str, n: int, kb_files: tuple):
    records, _ = load_kb(kb_files)
    return QA(records)


@st.cache_data(show_spinner=False)
def load_synonyms(name) -> Dict[str, Dict[str, str]]:
    if not name:
        return {}
    f = DATA / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


SYN: Dict[str, Dict[str, str]] = {}   # set after the dataset is chosen (sidebar)
NORMALIZE = False                     # set from the sidebar toggle each run


def as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return [] if v in (None, "") else [str(v)]


def field_values(r, field) -> List[str]:
    """Field values for a record, mapped through synonyms when normalization is on."""
    vals = as_list(r.get(field))
    if NORMALIZE and SYN.get(field):
        m = SYN[field]
        vals = [m.get(v, v) for v in vals]
    return vals


def value_counts(records, field) -> Counter:
    c: Counter = Counter()
    for r in records:
        for v in field_values(r, field):
            c[v] += 1
    return c


def hbar(counts: Counter, title: str, n: int = 20):
    if not counts:
        st.caption("-")
        return
    top = counts.most_common(n)
    vals = [v for _, v in top]
    fig = px.bar(x=vals, y=[k for k, _ in top], orientation="h",
                 labels={"x": "count", "y": ""}, title=title,
                 color=vals, color_continuous_scale=SCI_SCALE)
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      height=max(320, 24 * len(top)), margin=dict(l=10, r=10, t=40, b=10),
                      coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    if len(counts) > n:
        st.caption(f"Showing top {n} of {len(counts)} unique values "
                   f"(remaining {len(counts) - n} are long-tail, mostly single mentions).")


st.sidebar.title("🦟 malaria · ESX KB")

# dataset picker (only datasets whose files exist are offered)
_available = {lbl: cfg for lbl, cfg in DATASETS.items()
              if any((DATA / f).exists() for f in cfg["kb"])}
if not _available:
    st.title("🦟 malaria · ESX - knowledge base")
    st.warning("No data found. Build the corpus first (`python run_swarm.py`), then reload.")
    st.stop()

_labels = list(_available)
ds_label = (st.sidebar.selectbox("Dataset", _labels, index=0)
            if len(_labels) > 1 else _labels[0])
DS = _available[ds_label]

records, source_name = load_kb(tuple(DS["kb"]))
SYN = load_synonyms(DS["synonyms"])
if not records:
    st.title("🦟 malaria · ESX - knowledge base")
    st.warning(f"`{ds_label}` has no readable records yet.")
    st.stop()

st.sidebar.caption(f"Loaded `{source_name}` · {len(records)} papers")
page = st.sidebar.radio(
    "Page",
    ["Overview", "Field explorer", "Relationships", "Structures", "Proteins", "Papers", "QA"],
)
NORMALIZE = st.sidebar.toggle(
    "Normalize names", value=False,
    help="Merge synonym/spelling variants (synonyms.json) into a canonical form. "
         "Raw values are always kept in the data; this only changes the view.") if SYN else False
if SYN:
    st.sidebar.caption("Names: **normalized**" if NORMALIZE else "Names: **raw** (as extracted)")


# pages
def page_overview():
    st.title("🦟 Malaria (Plasmodium) export & ESX secretion - knowledge base")
    st.caption("Plasmodium protein export/secretion · Type-VII (ESX) systems · antimalarial resistance")

    years = [int(r["year"]) for r in records if str(r.get("year") or "").isdigit()]
    n_triples = sum(len(r.get("relationships") or []) for r in records)
    n_struct = sum(r.get("n_structures", len(r.get("structures") or [])) for r in records)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Publications", len(records))
    c2.metric("Unique genes", len(value_counts(records, "genes")))
    c3.metric("Typed relationships", n_triples)
    c4.metric("Structure resolutions", n_struct)

    cc1, cc2 = st.columns(2)
    with cc1:
        yc = Counter(str(y) for y in years)
        if yc:
            ser = {y: yc[y] for y in sorted(yc)}
            yv = list(ser.values())
            fig = px.bar(x=list(ser), y=yv, labels={"x": "year", "y": "publications"},
                         title="Publications per year", color=yv,
                         color_continuous_scale=SCI_SCALE)
            fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No year data.")
    with cc2:
        st_mix = Counter((r.get("study_type") or "unspecified").strip() or "unspecified"
                         for r in records)
        hbar(st_mix, "Study-type mix", 12)

    jc = Counter(r.get("journal", "").strip() for r in records if r.get("journal"))
    hbar(jc, "Top journals", 15)

    with st.expander("ℹ️ About this data"):
        mal = sum(1 for r in records if any(
            t in (r.get("title", "") + " " + r.get("abstract", "")).lower()
            for t in ("plasmodium", "malaria", "falciparum")))
        pct_mal = round(100 * mal / len(records)) if records else 0
        # Shared tail (same for every dataset).
        shared = (
            "- **What the pages show:** the extracted **entities** (genes, drugs, ...), "
            "the typed **relationships** between them (each tagged with its source PMID "
            "and a **confidence** score), and the **structures** each protein resolves to "
            "(UniProt + PDB/AlphaFold; exact gene match vs ~ name match).\n"
            "- **Names:** raw values are kept; the **Normalize names** toggle only changes the view."
        )
        if "antimalarial" in ds_label.lower():
            st.markdown(f"""
- A knowledge base of **antimalarial drugs & resistance** - a theme that kept resurfacing
  while studying the main *Plasmodium* export literature, so it was spun out into its own
  focused KB by the same multi-agent extraction pipeline mining PubMed abstracts (query **Q3**:
  antimalarial / artemisinin / chloroquine resistance, *kelch13*, *pfcrt*, *pfmdr1*).
- A **standalone 2026 demo KB**, separate from the Main KB (*Plasmodium* export + ESX);
  switch between them with the **Dataset** picker in the sidebar.
- **Corpus:** {len(records)} papers; ~**{pct_mal}% mention malaria/Plasmodium**. (The rest are
  mostly **artemisinin**-focused papers.)
{shared}
""")
        else:
            st.markdown(f"""
- A knowledge base of **protein secretion/export systems**, built by a **multi-agent
  extraction pipeline** mining PubMed abstracts. Two **functionally analogous but biologically distinct**
  themes: *Plasmodium* protein export (PTEX translocon, PEXEL motif, Maurer's clefts)
  and the bacterial **ESX / Type-VII** secretion system.
- They are **functional analogues only: no orthology, no shared machinery** -
  independently-evolved machines that both drive effectors across membranes. Each has its
  own unrelated ATPase *motor* powering export: **HSP101** in PTEX, **EccC** in ESX (same
  role, different proteins).
- **Two datasets** - switch with the **Dataset** picker in the sidebar: the **Main KB ·
  Plasmodium export + ESX** (queries Q1+Q2, shown here) and a separate **Q3 · antimalarial
  drugs & resistance (2026)** demo KB built from its own dedicated PubMed query.
- **Corpus:** {len(records)} papers; ~**{pct_mal}% mention malaria/Plasmodium**, the rest mostly
  Mycobacterium/TB. The two literatures barely overlap - we treat them as **parallel KBs**.
{shared}
""")


def page_fields():
    st.title("Extracted entities")
    field = st.selectbox("Field", LIST_FIELDS,
                         format_func=lambda s: s.replace("_", " ").title())
    counts = value_counts(records, field)
    if not counts:
        st.info("Nothing extracted for this field yet.")
        return
    hbar(counts, f"Top values - {field.replace('_', ' ')}", 40)

    st.subheader("Drill-down")
    pick = st.selectbox("Value", [v for v, _ in counts.most_common(60)])
    if pick:
        hits = [r for r in records if pick in field_values(r, field)]
        st.caption(f"{len(hits)} papers mention “{pick}”")
        for p in hits[:25]:
            url = p.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{p.get('pmid')}/"
            st.markdown(f"- [{p.get('pmid')}]({url}) - {p.get('title','')}  ·  _{p.get('year','')}_")


def page_relationships():
    st.title("Typed relationships (PMID-tagged)")
    rows = []
    by_pmid: Dict[str, Dict[str, Any]] = {}
    for r in records:
        pm = str(r.get("pmid") or "")
        if pm:
            by_pmid[pm] = r
        for t in r.get("relationships") or []:
            if isinstance(t, dict):
                rows.append({
                    "pmid": str(t.get("pmid") or r.get("pmid") or ""),
                    "subject": t.get("subject", ""), "subject_type": t.get("subject_type", ""),
                    "predicate": t.get("predicate", ""),
                    "object": t.get("object", ""), "object_type": t.get("object_type", ""),
                    "confidence": t.get("confidence"),
                })
    if not rows:
        st.warning("No typed relationships yet - run `python run_swarm.py`.")
        return
    df = pd.DataFrame(rows)
    st.caption(f"{len(df)} triples · {df['predicate'].nunique()} predicates · "
               f"{df['subject'].nunique()} subjects · {df['object'].nunique()} objects")

    # filters: predicate / subject / object dropdowns + substring search
    c1, c2, c3 = st.columns(3)
    with c1:
        pick_pred = st.selectbox("Predicate", ["(all)"] + sorted(df["predicate"].unique()))
    with c2:
        pick_subj = st.selectbox("Subject", ["(all)"] + sorted(s for s in df["subject"].unique() if s))
    with c3:
        pick_obj = st.selectbox("Object", ["(all)"] + sorted(o for o in df["object"].unique() if o))
    q = st.text_input("Filter subject / object (substring)").strip().lower()

    view = df
    if pick_pred != "(all)":
        view = view[view["predicate"] == pick_pred]
    if pick_subj != "(all)":
        view = view[view["subject"] == pick_subj]
    if pick_obj != "(all)":
        view = view[view["object"] == pick_obj]
    if q:
        m = (view["subject"].str.lower().str.contains(q, na=False)
             | view["object"].str.lower().str.contains(q, na=False))
        view = view[m]

    st.caption(f"**{len(view)}** of {len(df)} triples match the current filters.")
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)

    # inspect one PMID: show the paper + all its relationships below
    st.divider()
    pmids = sorted(p for p in view["pmid"].unique() if p)
    sel = st.selectbox(f"Inspect a PMID ({len(pmids)} in the filtered rows)",
                       ["(none)"] + pmids)
    if sel and sel != "(none)":
        url = f"https://pubmed.ncbi.nlm.nih.gov/{sel}/"
        rec = by_pmid.get(sel)
        st.markdown(f"### [{sel}]({url}) — {rec.get('title','') if rec else ''}")
        if rec:
            meta = " · ".join(x for x in [rec.get("journal", ""), str(rec.get("year", ""))] if x)
            if meta:
                st.caption(meta)
            if rec.get("abstract"):
                with st.expander("Abstract"):
                    st.write(rec["abstract"])
        sub = df[df["pmid"] == sel]
        st.markdown(f"**{len(sub)} relationships in this paper**")
        st.dataframe(sub, use_container_width=True, hide_index=True,
                     height=min(420, 90 + 30 * len(sub)))


def render_3dmol(kind: str, ref: str, height: int = 460):
    """Embed an interactive 3D structure viewer (3Dmol.js) inline.

    kind='pdb'  -> ref is a PDB id, loaded from RCSB.
    kind='url'  -> ref is a direct .pdb model URL (e.g. an AlphaFold model).
    """
    if kind == "pdb":
        loader = (f'$3Dmol.download("pdb:{ref}", v, {{}}, function() {{'
                  f' v.setStyle({{}}, {{cartoon:{{color:"spectrum"}}}}); v.zoomTo(); v.render(); }});')
    else:
        loader = (f'fetch("{ref}").then(function(r){{if(!r.ok)throw 0;return r.text();}})'
                  f'.then(function(d){{ v.addModel(d,"pdb");'
                  f' v.setStyle({{}}, {{cartoon:{{color:"spectrum"}}}}); v.zoomTo(); v.render(); }})'
                  f'.catch(function(){{ document.getElementById("mol").innerHTML='
                  f'\'<p style="font:14px sans-serif">Could not load inline. '
                  f'<a href="{ref}" target="_blank">Open model file</a></p>\'; }});')
    html = f"""
    <div id="mol" style="height:{height}px;width:100%;position:relative;
         border:1px solid #e6e6e6;border-radius:8px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      var v = $3Dmol.createViewer(document.getElementById("mol"),
                                  {{backgroundColor:"white"}});
      {loader}
    </script>"""
    components.html(html, height=height + 12)


def _structure_viewer(s: Dict[str, Any], key: str):
    """Dropdown of a resolved entity's structures + an inline 3D viewer."""
    opts: Dict[str, tuple] = {}
    for pdb in s.get("pdb") or []:
        opts[f"PDB {pdb['pdb_id']} (experimental)"] = ("pdb", pdb["pdb_id"])
    af = s.get("alphafold") or {}
    # Only offer AlphaFold when the pipeline actually verified a model exists
    # (mcp_server returns {} for accessions AlphaFold DB doesn't cover, e.g. PfEMP1).
    # Never fabricate an AF-<acc>-F1-model_v4.pdb URL - many such files 404.
    if af.get("model_pdb"):
        opts["AlphaFold model (predicted)"] = ("url", af["model_pdb"])
    if not opts:
        return
    choice = st.selectbox("View 3D structure", ["(none)"] + list(opts), key=f"v_{key}")
    if choice != "(none)":
        kind, ref = opts[choice]
        render_3dmol(kind, ref)


def _struct_links(s: Dict[str, Any]) -> str:
    links = []
    up = s.get("uniprot") or {}
    if up.get("url"):
        links.append(f"[UniProt {up.get('accession','')}]({up['url']})")
    for pdb in (s.get("pdb") or [])[:6]:
        links.append(f"[PDB {pdb['pdb_id']}]({pdb.get('view3d') or pdb.get('url')})")
    af = s.get("alphafold") or {}
    # Only link AlphaFold when a model was verified upstream (af.viewer is set only
    # then). Avoids dead entry-page links for accessions AlphaFold DB doesn't cover.
    if af.get("viewer"):
        links.append(f"[AlphaFold model]({af['viewer']})")
    return " · ".join(links)


def _chips(label: str, items, color: str = "#ede9fe", text: str = "#5b21b6"):
    items = [str(x) for x in (items or []) if str(x).strip()]
    if not items:
        return
    seen, uniq = set(), []
    for it in items:                       # dedupe, preserve order
        if it.lower() not in seen:
            seen.add(it.lower()); uniq.append(it)
    spans = "".join(
        f'<span style="background:{color};color:{text};padding:2px 9px;margin:2px;'
        f'border-radius:11px;font-size:12px;display:inline-block;">{x}</span>'
        for x in uniq)
    st.markdown(f"**{label}**<br>{spans}", unsafe_allow_html=True)


def _conf_badge(conf: str) -> str:
    """Honest confidence badge for a UniProt resolution."""
    if conf == "gene_exact":
        bg, tx, label = "#dcfce7", "#166534", "✓ exact gene match (high confidence)"
    elif conf == "name_match":
        bg, tx, label = "#fef9c3", "#854d0e", "~ name match (verify)"
    else:
        bg, tx, label = "#f1f5f9", "#475569", "unverified"
    return (f'<span style="background:{bg};color:{tx};padding:2px 9px;border-radius:11px;'
            f'font-size:11px;font-weight:600;">{label}</span>')


def _uniprot_detail(up: Dict[str, Any]):
    """Render the rich UniProt fields for one protein."""
    if not up:
        return
    bits = []
    if up.get("length"):
        bits.append(f"**{up['length']}** aa")
    if up.get("mass"):
        bits.append(f"**{round(up['mass']/1000)}** kDa")
    if up.get("ec_numbers"):
        bits.append("EC " + ", ".join(up["ec_numbers"]))
    feats = up.get("features") or {}
    if feats.get("Transmembrane"):
        bits.append(f"{feats['Transmembrane']}× transmembrane")
    if feats.get("Signal"):
        bits.append("signal peptide")
    if bits:
        st.caption(" · ".join(bits))
    _chips("📍 Subcellular location", up.get("subcellular_locations"), "#fce7f3", "#9d174d")
    _chips("🏷️ Keywords", up.get("keywords"))
    _chips("🧩 Domains / families (Pfam)", up.get("pfam"), "#dbeafe", "#1e40af")
    _chips("🔬 GO · component", up.get("go_component"), "#e9d5ff", "#6b21a8")
    _chips("⚙️ GO · process", up.get("go_process"), "#e0e7ff", "#3730a3")
    if up.get("families"):
        st.caption("**Family:** " + " ".join(up["families"]))
    if up.get("pathway"):
        st.caption("**Pathway:** " + " ".join(up["pathway"]))
    if up.get("catalytic_activity"):
        st.caption("**Catalytic activity:** " + " | ".join(up["catalytic_activity"]))


def page_structures():
    st.title("Structures - UniProt / PDB / AlphaFold")
    # de-duplicate resolved entities across papers
    by_acc: Dict[str, Dict[str, Any]] = {}
    for r in records:
        for s in r.get("structures") or []:
            if not s.get("resolved"):
                continue
            acc = (s.get("uniprot") or {}).get("accession") or s.get("entity")
            by_acc.setdefault(acc, s)
    if not by_acc:
        st.warning("No structures yet - run `python run_swarm.py`.")
        return

    n_pdb = sum(1 for s in by_acc.values() if s.get("pdb"))
    n_af = sum(1 for s in by_acc.values() if s.get("alphafold") and not s.get("pdb"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Resolved entities", len(by_acc))
    c2.metric("With experimental PDB", n_pdb)
    c3.metric("AlphaFold-only", n_af)

    # common patterns across all resolved proteins (the "what do they share?" view)
    with st.expander("🔎 Common patterns across resolved proteins", expanded=True):
        def agg(field):
            c: Counter = Counter()
            for s in by_acc.values():
                for v in (s.get("uniprot") or {}).get(field) or []:
                    c[str(v)] += 1
            return c
        loc, kw = agg("subcellular_locations"), agg("keywords")
        pfam, goc = agg("pfam"), agg("go_component")
        if not (loc or kw or pfam or goc):
            st.caption("No rich annotation yet - re-run `python run_swarm.py`.")
        else:
            a, b = st.columns(2)
            with a:
                hbar(loc, "📍 Most common subcellular locations", 15)
                hbar(pfam, "🧩 Most common domains (Pfam)", 15)
            with b:
                hbar(kw, "🏷️ Most common keywords", 15)
                hbar(goc, "🔬 Most common GO components", 15)

    fc1, fc2 = st.columns([3, 1])
    q = fc1.text_input("Filter by entity / protein / accession").strip().lower()
    only_pdb = fc2.checkbox("Only experimental PDB", value=False)
    # show entities with experimental PDB first, then AlphaFold-only
    ordered = sorted(by_acc.items(), key=lambda kv: (0 if kv[1].get("pdb") else 1, kv[0]))
    for acc, s in ordered:
        up = s.get("uniprot") or {}
        hay = f"{s.get('entity','')} {up.get('protein_name','')} {acc}".lower()
        if q and q not in hay:
            continue
        if only_pdb and not s.get("pdb"):
            continue
        title = f"{s.get('entity','')} - {up.get('protein_name','') or '?'}  ·  `{acc}`"
        with st.expander(title):
            st.markdown(_conf_badge(s.get("confidence")), unsafe_allow_html=True)
            if up.get("organism"):
                st.caption(up["organism"])
            if up.get("function_summary"):
                st.write(up["function_summary"])
            _uniprot_detail(up)
            st.markdown(_struct_links(s) or "_no links_")
            _structure_viewer(s, key=acc)


def page_papers():
    st.title("Papers")
    rows = [{
        "id": r.get("id"), "pmid": r.get("pmid"), "title": (r.get("title") or "")[:160],
        "journal": r.get("journal", ""), "year": r.get("year", ""),
        "study_type": r.get("study_type", ""),
        "n_rel": len(r.get("relationships") or []),
        "n_struct": r.get("n_structures", len(r.get("structures") or [])),
    } for r in records]
    df = pd.DataFrame(rows)
    q = st.text_input("Search title / journal").strip().lower()
    view = df
    if q:
        m = (df["title"].str.lower().str.contains(q, na=False)
             | df["journal"].str.lower().str.contains(q, na=False))
        view = df[m]
    st.dataframe(view, use_container_width=True, hide_index=True, height=380)

    pmid = st.text_input("Open PMID").strip()
    if not pmid:
        return
    hit = next((r for r in records if str(r.get("pmid")) == pmid), None)
    if not hit:
        st.error("Not found.")
        return
    url = hit.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    st.markdown(f"### {hit.get('title','')}")
    st.caption(f"{hit.get('journal','')} · {hit.get('year','')} · [PubMed]({url})")
    if hit.get("abstract"):
        st.write(hit["abstract"])

    chips = {f: as_list(hit.get(f)) for f in LIST_FIELDS}
    chips = {f: v for f, v in chips.items() if v}
    if chips:
        st.subheader("Extracted entities")
        for f, vals in chips.items():
            st.markdown(f"**{f.replace('_', ' ').title()}**: " + ", ".join(vals))

    rels = [t for t in (hit.get("relationships") or []) if isinstance(t, dict)]
    if rels:
        st.subheader("Relationships")
        st.dataframe(pd.DataFrame([{"subject": t.get("subject"), "predicate": t.get("predicate"),
                                    "object": t.get("object")} for t in rels]),
                     use_container_width=True, hide_index=True)

    structs = [s for s in (hit.get("structures") or []) if s.get("resolved")]
    if structs:
        st.subheader("Structures")
        for i, s in enumerate(structs):
            up = s.get("uniprot") or {}
            st.markdown(f"**{s.get('entity','')}** - {up.get('protein_name','')}  ·  `{up.get('accession','')}`")
            _uniprot_detail(up)
            st.markdown(_struct_links(s) or "_no links_")
            _structure_viewer(s, key=f"{pmid}_{i}")


def page_ask():
    st.title("Ask the knowledge base")
    st.caption("Retrieval-augmented Q&A over the corpus - answers cite supporting PMIDs.")
    k = st.slider("Papers to retrieve", 3, 12, 6)
    question = st.text_input("Your question",
                             placeholder="e.g. How does kelch13 confer artemisinin resistance?")
    if not question:
        return
    qa = get_qa(source_name, len(records), tuple(DS["kb"]))
    with st.spinner("Thinking…"):
        res = qa.answer(question, k=k)
    st.markdown(res["answer"])
    if res["sources"]:
        st.subheader("Sources")
        for s in res["sources"]:
            st.markdown(f"- [{s['pmid']}]({s['url']}) - {s.get('title','')}")


@st.cache_data(show_spinner=False)
def load_proteins(name):
    f = DATA / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def page_proteins():
    st.title("Proteins")
    st.caption("One card per resolved protein: its UniProt biology + the "
               "relationships we extracted that involve it + the papers that mention it.")
    prot = load_proteins(DS["proteins"])
    if not prot:
        st.warning(f"No protein file for this dataset yet - run "
                   f"`python build_proteins.py --in {DS['kb'][0]} --out {DS['proteins']}`.")
        return

    n_pdb = sum(1 for p in prot if p.get("pdb"))
    n_rel = sum(p.get("n_relationships", 0) for p in prot)
    n_hi = sum(1 for p in prot if p.get("confidence") == "gene_exact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proteins", len(prot))
    c2.metric("High-confidence", f"{n_hi}/{len(prot)}")
    c3.metric("With experimental PDB", n_pdb)
    c4.metric("Relationships linked", n_rel)
    st.caption("Resolutions are UniProt name lookups, not curation. ✓ = exact gene "
               "match (high confidence); ~ = name match (verify before relying on it).")

    fc1, fc2, fc3 = st.columns([3, 1, 1])
    q = fc1.text_input("Filter by name / protein / accession").strip().lower()
    only_pdb = fc2.checkbox("Only experimental PDB", value=False)
    only_hi = fc3.checkbox("High-confidence only", value=False)

    # faceted filters built from the data
    def opts(field):
        c: Counter = Counter()
        for p in prot:
            vals = (p.get("uniprot") or {}).get(field) or []
            for v in (vals if isinstance(vals, list) else [vals]):
                if v:
                    c[v] += 1
        return [f"{k}  ({v})" for k, v in c.most_common()]

    def strip_count(s):           # "Secreted  (36)" -> "Secreted"
        return s.rsplit("  (", 1)[0]

    with st.expander("🔎 Filters", expanded=False):
        f_loc = st.multiselect("📍 Subcellular location", opts("subcellular_locations"))
        f_kw = st.multiselect("🏷️ Keyword", opts("keywords"))
        f_dom = st.multiselect("🧩 Domain (Pfam)", opts("pfam"))
        f_go = st.multiselect("🔬 GO component", opts("go_component"))
        org_opts = sorted({(p.get("uniprot") or {}).get("organism") for p in prot
                           if (p.get("uniprot") or {}).get("organism")})
        f_org = st.multiselect("🧬 Organism", org_opts)
    sel = {"subcellular_locations": [strip_count(x) for x in f_loc],
           "keywords": [strip_count(x) for x in f_kw],
           "pfam": [strip_count(x) for x in f_dom],
           "go_component": [strip_count(x) for x in f_go]}

    def passes_facets(up):
        for field, chosen in sel.items():
            if chosen and not (set(up.get(field) or []) & set(chosen)):
                return False
        if f_org and up.get("organism") not in f_org:
            return False
        return True

    matched = [p for p in prot
               if passes_facets(p.get("uniprot") or {})
               and (not only_pdb or p.get("pdb"))
               and (not only_hi or p.get("confidence") == "gene_exact")
               and (not q or q in f"{' '.join(p.get('entity_names') or [])} "
                    f"{(p.get('uniprot') or {}).get('protein_name','')} {p['accession']}".lower())]
    st.caption(f"**{len(matched)}** of {len(prot)} proteins match the current filters.")

    shown = 0
    for p in matched:
        up = p.get("uniprot") or {}
        names = ", ".join(p.get("entity_names") or [])
        shown += 1
        if shown > 80:                      # keep the page snappy
            st.caption("… narrow the filter to see more proteins.")
            break
        header = (f"{up.get('protein_name') or names or p['accession']}  ·  `{p['accession']}`"
                  f"   -   {p['n_papers']} papers · {p['n_relationships']} relationships")
        with st.expander(header):
            st.markdown(_conf_badge(p.get("confidence")), unsafe_allow_html=True)
            if names:
                st.caption("Extracted as: " + names)
            if up.get("organism"):
                st.caption(up["organism"])
            if up.get("function_summary"):
                st.write(up["function_summary"])
            _uniprot_detail(up)
            st.markdown(_struct_links(p) or "_no links_")
            _structure_viewer(p, key=f"prot_{p['accession']}")

            rels = p.get("relationships") or []
            if rels:
                st.markdown(f"**Relationships we extracted ({len(rels)})**")
                st.dataframe(
                    pd.DataFrame([{"subject": r["subject"], "predicate": r["predicate"],
                                   "object": r["object"], "papers": len(r["pmids"])}
                                  for r in rels]),
                    use_container_width=True, hide_index=True, height=min(320, 80 + 30 * len(rels)))

            papers = p.get("papers") or []
            if papers:
                with st.popover(f"📚 {len(papers)} papers"):
                    for pp in papers[:60]:
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pp['pmid']}/"
                        st.markdown(f"- [{pp['pmid']}]({url}) - {pp.get('title','')[:90]}")


PAGES = {
    "Overview": page_overview, "Field explorer": page_fields,
    "Relationships": page_relationships, "Structures": page_structures,
    "Proteins": page_proteins, "Papers": page_papers, "QA": page_ask,
}
PAGES[page]()
