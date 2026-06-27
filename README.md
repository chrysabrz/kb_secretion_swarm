# kb_secretion_swarm

An **agent-based knowledge-base builder** for **Plasmodium (malaria) protein export
& secretion biology, the bacterial Type-VII / ESX secretion system, and
antimalarial drugs & resistance**. A **multi-agent extraction pipeline** of small,
specialized LLM agents, wired together as a
[LangGraph](https://langchain-ai.github.io/langgraph/) state machine, reads ~1,619
PubMed abstracts (from 2015 to 2026) and turns them into a structured, queryable knowledge base:
extracted entities, judge-validated typed relationships, and protein structures
resolved via MCP against **UniProt / PDB / AlphaFold**. The result is served as a
multi-page **Streamlit dashboard** with a cited **Q&A** page.

>  The **Files** table below explains every file in one line; each file's
> own docstring has the detail.

> ## Scope note: malaria / ESX system
> Two functionally analogous but biologically distinct themes: Plasmodium protein export (PTEX translocon, PEXEL motif, Maurer's clefts) and the bacterial ESX / Type-VII secretion system.

> The parallel is **functional analogy, not homology**: ESX (a *Mycobacterium* prokaryote)
> and PTEX (a *Plasmodium* eukaryote) are independently-evolved machines that solve the same
> problem, driving effectors across a double membrane (e.g. the EccC FtsK/SpoIIIE ATPase
> mirrors PTEX's HSP101 AAA+ ATPase). They share no common ancestral blueprint.
>
> So this KB holds the two as **functional analogues only: no orthology, no shared
> machinery**, and every agent's prompt carries this rule. (Ordinary orthology *within* a
> lineage ,e.g. *Plasmodium* ↔ *Toxoplasma*  ->is real biology and is kept.)


## How it works

One paper at a time, through a LangGraph pipeline:

```text
PubMed
  │  find new papers, download abstracts
  ▼
pubmed_extractor
  │
  ▼  per paper, a LangGraph runs:
  │
  ├─► entity_extractor         genes, drugs, species, ...
  │       │
  │       ▼
  ├─► relationship_extractor   typed subject-predicate-object triples
  │       │
  │       ▼
  ├─► validator                LLM judge: scores + retry once
  │       │                    └─ if all rejected, back to
  │       │                       relationship_extractor
  │       ▼
  └─► structure_enricher       UniProt / PDB / AlphaFold (local MCP)
          │
          ▼
writes data/*.json, then builds the protein view + synonyms
          │
          ▼
app.py (Streamlit dashboard)  +  qa.py (cited Q&A)
```

Two ideas keep it trustworthy: every agent returns a **validated structured object**
(predicate / subject / object - not free text), and a separate **LLM judge** scores each relationship's
`confidence` (see [Confidence](#confidence) below).

## Files

**Run it**
| File | What it does |
|------|--------------|
| `run_swarm.py` | build the knowledge base from PubMed (the main entry point) |
| `app.py` | Streamlit dashboard for the knowledge base |
| `qa.py` | retrieval-augmented Q&A over the knowledge base (powers the QA page) |

**The extraction pipeline (`swarm/`)**
| File | What it does |
|------|--------------|
| `config.py` | shared settings: queries, entity schema, vocabularies, paths, model factory |
| `schemas.py` | Pydantic schemas for structured agent outputs |
| `state.py` | the LangGraph state for the per-paper pipeline |
| `tools.py` | clients for the external data sources: PubMed and the bio-MCP server |
| `structure_resolver.py` | resolve an entity name to UniProt / PDB / AlphaFold via the MCP |
| `publication_swarm.py` | wires the per-paper agents into one LangGraph |
| `agents/pubmed_extractor.py` | finds the papers to process |
| `agents/entity_extractor.py` | the entity-extraction step (one paper at a time) |
| `agents/relationship_extractor.py` | turn a paper's entities into typed relationships |
| `agents/validator.py` | the LLM judge (the quality gate) |
| `agents/structure_enricher.py` | attach structures to each paper |

**Structure lookups (MCP)**
| File | What it does |
|------|--------------|
| `mcp_server.py` | a small MCP server that fetches protein structure data (UniProt/PDB/AlphaFold) |
| `mcp_client.py` | the Python side that talks to mcp_server.py |

**Post-processing**
| File | What it does |
|------|--------------|
| `build_proteins.py` | turn the papers KB into a protein-centric view (`proteins.json`) |
| `normalize.py` | build `synonyms.json`: a raw → canonical name map per field |
| `score_confidence.py` | backfill a judge `confidence` onto relationships that lack one |

**Demo**
| File | What it does |
|------|--------------|
| `demo_q3_2026.py` | standalone Q3 (antimalarial resistance) demo dataset, selectable in the dashboard |

## Setup

> **Python 3.10+ required** (LangGraph/LangChain drop 3.8). On this machine that is
> `py -3.12`, **not** the bare `python` (3.8). Or use a venv:
> `py -3.12 -m venv .venv && .venv\Scripts\activate`.

```bash
py -3.12 -m pip install -r requirements-pipeline.txt   # full deps (run the swarm + dashboard)
cp .env.example .env        # then add an OpenAI (or Anthropic) key + NCBI email
```

> **Two requirements files:** `requirements.txt` is the **slim dashboard** set (what
> Streamlit Cloud installs); `requirements-pipeline.txt` adds the agent stack
> (LangGraph/LangChain/Biopython) needed to *run the swarm*. To just view the shipped
> data in the dashboard, `pip install -r requirements.txt` is enough.



## Usage

```bash
# Build the KB(already in data), then open the dashboard
py -3.12 run_swarm.py
py -3.12 -m streamlit run app.py

# Smoke tests
py -3.12 run_swarm.py --limit 5                      # 5 new papers only
py -3.12 run_swarm.py --queries 2 --max-results 50   # ESX query only(defined by query number)

# Ask a question on the command line
py -3.12 qa.py "How does kelch13 confer artemisinin resistance?"
```

The shipped `data/` files let the dashboard work immediately. Re-running
`run_swarm.py` only spends budget on genuinely new papers (it skips PMIDs already in
the KB), since the procedure is incremental.

## Outputs (in `data/`, committed)

- `secretion_systems.json` :entities per paper (stable `id` each).
- `secretion_systems_with_relationships.json` :+ judge-validated triples (PMID-tagged, with `confidence`).
- `secretion_systems_with_structures.json` :+ UniProt / PDB / AlphaFold (the main KB the dashboard reads).
- `proteins.json` :the same data pivoted to one entry per protein (`build_proteins.py`).
- `synonyms.json` :`raw → canonical` name maps (`normalize.py`).
- `q3_antimalarial_2026*.json` :the Q3 demo dataset (kb / proteins / synonyms), selectable in the dashboard.

Rebuildable caches (`_structures_cache.json`, `_embeddings.npy`, etc.) stay in the
repo root and are gitignored.

> For the exact record shape and the logic that writes each file, see the producing
> script's module docstring (`run_swarm.py`, `build_proteins.py`, `normalize.py`,
> `demo_q3_2026.py`) - the one-liners above are just the summary.

## Confidence

Every relationship carries a `confidence` from the **LLM judge** (`validator.py`).
For each candidate triple it re-reads the abstract and decides whether the abstract
*actually supports it*, giving a 0–1 score. It can **vote N times**
(`SWARM_JUDGE_VOTES`, default 3) and keeps a triple only if the **median** confidence
≥ `SWARM_JUDGE_THRESHOLD` (default 0.5) **and** most votes mark it supported; if it
rejects every candidate, the paper is re-extracted once. In `proteins.json`, a
relationship's confidence is the **max** across the papers asserting it. 

**Type gate (before the judge).** Some predicates are only meaningful between certain
entity types, so `validator.py` first drops any triple that misuses one - checked
deterministically by `predicate_acceptance_types()` against `PREDICATE_TYPE_CONSTRAINTS`
in `config.py`. For example, `is_orthologous_to` is a **gene/protein** relation, so a
`species → is_orthologous_to → species` triple (e.g. *Plasmodium* `is_orthologous_to`
*Toxoplasma gondii*) is rejected as a category error rather than handed to the judge.
Triples with empty/unknown types are left for the judge to score.

## Deployment (Streamlit Community Cloud)

The `data/` files are committed, so the dashboard deploys as-is - nothing to regenerate.

1. Push the repo (with `data/`) to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): **New app** → **Main file path** `app.py` (it auto-installs the slim `requirements.txt` - not the pipeline file) or pick deploy while running on localhost.
3. **Settings → Secrets**: add `OPENAI_API_KEY` (powers the QA page; other pages work without it) and `NCBI_EMAIL`.
4. Deploy. The **Dataset** picker switches between the Main KB and the Q3 demo.



## License

Copyright © 2026 Chrysa Bourtzinakou. Licensed for academic use to Prof. Dr. Dirk
Valkenborg and his group at Hasselt University (UHasselt) and the Institute of
Tropical Medicine (ITM), Antwerp. See [`LICENSE`](LICENSE).
