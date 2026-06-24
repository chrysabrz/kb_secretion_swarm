"""
qa.py - retrieval-augmented Q&A over the knowledge base.

A small, self-contained RAG layer: a retriever picks the most relevant papers, then
one LLM call answers using ONLY those papers, citing them by PMID. Two retrievers:
  * embeddings (default): OpenAI embeddings + cosine ranking, cached on disk.
  * tfidf: pure-Python, zero extra deps.
embeddings falls back to tfidf automatically if NumPy / the API key is unavailable.

Used by app.py (the QA page) and runnable on its own:
    python qa.py "How does kelch13 confer artemisinin resistance?"
    python qa.py --k 8 --retriever tfidf "What components make up PTEX?"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:                                # avoid UnicodeEncodeError on cp1253 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:                                # NumPy powers the embeddings retriever
    import numpy as np
except Exception:
    np = None

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

# Selectable datasets (mirrors app.py): most-enriched KB file first per dataset.
DATASETS = {
    "main": ["secretion_systems_with_structures.json",
             "secretion_systems_with_relationships.json", "secretion_systems.json"],
    "q3": ["q3_antimalarial_2026.json"],
}
KB_FILES = DATASETS["main"]   # default dataset (back-compat for load_kb() callers)
DEFAULT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMB_VECTORS = HERE / "_embeddings.npy"          # cached doc vectors (gitignored)
EMB_KEYS = HERE / "_embeddings_keys.json"       # parallel list of doc ids

ENTITY_FIELDS = [
    "species", "secretion_components", "exported_proteins", "genes", "drugs",
    "resistance_markers", "life_cycle_stages", "clinical_outcomes", "methods",
]

_STOP = set("a an the of and or to in on for with by is are was were be been being as at "
            "from into this that these those it its their his her our your my we you they "
            "which who whom whose what when where why how can could may might will would "
            "do does did has have had not no than then so such between within via using used "
            "study results showed show found also more most other both each".split())
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _warn(msg: str) -> None:
    print(f"[qa] {msg}", file=sys.stderr)


def load_kb(kb_files: List[str] | None = None) -> Tuple[List[Dict[str, Any]], str]:
    for name in (kb_files or KB_FILES):
        f = HERE / "data" / name
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8")), name
    return [], ""


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")
            if len(t) > 2 and t.lower() not in _STOP]


def paper_text(p: Dict[str, Any]) -> str:
    parts = [str(p.get("title", "")), str(p.get("abstract", ""))]
    for f in ENTITY_FIELDS:
        parts.extend(str(v) for v in (p.get(f) or []))
    for t in (p.get("relationships") or []):
        if isinstance(t, dict):
            parts.append(f"{t.get('subject','')} {t.get('predicate','')} {t.get('object','')}")
    return " ".join(parts)


def _doc_id(p: Dict[str, Any], i: int) -> str:
    """Stable id per record (PMID, else the KB id, else position)."""
    return str(p.get("pmid") or p.get("id") or f"rec-{i}").strip() or f"rec-{i}"


def _openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _embed_texts(client, texts: List[str]) -> List[List[float]]:
    """Embed a list of strings, batching the API calls."""
    out: List[List[float]] = []
    B = 100
    for s in range(0, len(texts), B):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[s:s + B])
        out.extend(d.embedding for d in resp.data)
    return out


class QA:
    """Pluggable retriever (embeddings or TF-IDF) + grounded LLM answerer over the KB."""

    def __init__(self, records: List[Dict[str, Any]], model: str = DEFAULT_MODEL,
                 retriever: str = "embeddings"):
        self.records = records
        self.model = model
        self._build_tfidf()                       # always built - the fallback retriever
        self.retriever = retriever
        self._emb = None                          # row-normalized doc matrix (NumPy)
        self._client = None
        if retriever == "embeddings" and not self._build_embeddings():
            self.retriever = "tfidf"              # graceful fallback

    # TF-IDF
    def _build_tfidf(self) -> None:
        self._docs = [tokenize(paper_text(p)) for p in self.records]
        self._tf = [Counter(d) for d in self._docs]
        df: Counter = Counter()
        for d in self._docs:
            df.update(set(d))
        n = max(1, len(self.records))
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
        self._norm = [math.sqrt(sum((tf[t] * self._idf.get(t, 0.0)) ** 2 for t in tf)) or 1.0
                      for tf in self._tf]

    def _retrieve_tfidf(self, query: str, k: int) -> List[Dict[str, Any]]:
        q = tokenize(query)
        if not q:
            return []
        qw = {t: self._idf.get(t, 0.0) for t in set(q)}
        scored = []
        for i, tf in enumerate(self._tf):
            s = sum(qw.get(t, 0.0) * tf.get(t, 0) * self._idf.get(t, 0.0) for t in qw)
            if s > 0:
                scored.append((s / self._norm[i], i))
        scored.sort(reverse=True)
        return [self.records[i] for _, i in scored[:k]]

    # embeddings
    def _load_emb_cache(self) -> Dict[str, Any]:
        if EMB_VECTORS.exists() and EMB_KEYS.exists():
            try:
                keys = json.loads(EMB_KEYS.read_text(encoding="utf-8"))
                mat = np.load(EMB_VECTORS)
                if len(keys) == len(mat):
                    return {k: mat[i] for i, k in enumerate(keys)}
            except Exception:
                pass
        return {}

    def _save_emb_cache(self, cache: Dict[str, Any]) -> None:
        keys = list(cache.keys())
        mat = np.asarray([cache[k] for k in keys], dtype="float32")
        np.save(EMB_VECTORS, mat)
        EMB_KEYS.write_text(json.dumps(keys), encoding="utf-8")

    def _build_embeddings(self) -> bool:
        """Embed any NEW docs, cache to disk, build the normalized matrix.
        Returns False to signal a fallback to TF-IDF."""
        if np is None:
            _warn("NumPy not available (`pip install numpy`) - falling back to TF-IDF.")
            return False
        self._client = _openai_client()
        if self._client is None:
            _warn("OPENAI_API_KEY not set - embeddings unavailable, using TF-IDF.")
            return False
        try:
            keys = [_doc_id(p, i) for i, p in enumerate(self.records)]
            cache = self._load_emb_cache()
            missing = [i for i, k in enumerate(keys) if k not in cache]
            if missing:
                _warn(f"embedding {len(missing)} new doc(s) with {EMBED_MODEL} "
                      "(one-time; cached in data/)…")
                vecs = _embed_texts(self._client, [paper_text(self.records[i])[:8000]
                                                   for i in missing])
                for i, v in zip(missing, vecs):
                    cache[keys[i]] = v
                self._save_emb_cache(cache)
            mat = np.asarray([cache[k] for k in keys], dtype="float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._emb = mat / norms               # row-normalized for cosine
            return True
        except Exception as e:
            _warn(f"embedding build failed ({type(e).__name__}: {e}) - using TF-IDF.")
            return False

    def _retrieve_embeddings(self, query: str, k: int) -> List[Dict[str, Any]]:
        try:
            qv = np.asarray(_embed_texts(self._client, [query])[0], dtype="float32")
            qv = qv / (np.linalg.norm(qv) or 1.0)
            scores = self._emb @ qv
            idx = np.argsort(-scores)[:k]
            return [self.records[i] for i in idx if scores[i] > 0]
        except Exception as e:
            _warn(f"embedding query failed ({type(e).__name__}: {e}) - using TF-IDF.")
            return self._retrieve_tfidf(query, k)

    # dispatch
    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.retriever == "embeddings" and self._emb is not None:
            hits = self._retrieve_embeddings(query, k)
            if hits:
                return hits                        # else fall through to TF-IDF
        return self._retrieve_tfidf(query, k)

    def _context(self, papers: List[Dict[str, Any]]) -> str:
        blocks = []
        for p in papers:
            ents = {f: p.get(f) for f in ENTITY_FIELDS if p.get(f)}
            rels = [f"{t.get('subject','')} {t.get('predicate','')} {t.get('object','')}"
                    for t in (p.get("relationships") or []) if isinstance(t, dict)][:12]
            blocks.append(
                f"[PMID {p.get('pmid','?')}] {p.get('title','')}\n"
                f"Abstract: {str(p.get('abstract',''))[:1200]}\n"
                f"Entities: {json.dumps(ents, ensure_ascii=False)}\n"
                f"Relationships: {'; '.join(rels) if rels else '(none)'}"
            )
        return "\n\n".join(blocks)

    def answer(self, query: str, k: int = 6) -> Dict[str, Any]:
        papers = self.retrieve(query, k)
        if not papers:
            return {"answer": "No relevant papers found in the knowledge base for that question.",
                    "sources": []}
        api_key = os.getenv("OPENAI_API_KEY")
        sources = [{"pmid": p.get("pmid"), "title": p.get("title"),
                    "url": p.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{p.get('pmid')}/"}
                   for p in papers]
        if not api_key:
            return {"answer": "(OPENAI_API_KEY not set - showing retrieved papers only.)",
                    "sources": sources}
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Answer the question using ONLY the secretion-systems knowledge-base context "
            "below (Plasmodium protein export AND bacterial ESX / Type-VII secretion). "
            "Cite supporting papers inline as [PMID xxxxx]. If the context does not "
            "contain the answer, say so plainly. Note: Plasmodium export (PTEX/PEXEL) and "
            "bacterial ESX are functional analogues, not the same system - do not assert "
            "orthology or shared machinery between them unless a paper explicitly states it.\n\n"
            f"QUESTION: {query}\n\n"
            f"CONTEXT:\n{self._context(papers)}\n"
        )
        resp = client.chat.completions.create(
            model=self.model, temperature=0.1, max_tokens=700,
            messages=[
                {"role": "system", "content": "You are a careful molecular-microbiology research "
                                              "assistant for protein secretion/export systems. "
                                              "Ground every claim in the provided context and "
                                              "cite PMIDs."},
                {"role": "user", "content": prompt},
            ],
        )
        return {"answer": resp.choices[0].message.content or "", "sources": sources}


def main():
    ap = argparse.ArgumentParser(description="Ask the knowledge base a question.")
    ap.add_argument("question", help="Natural-language question.")
    ap.add_argument("--k", type=int, default=6, help="Papers to retrieve (default 6).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--retriever", choices=["embeddings", "tfidf"], default="embeddings",
                    help="Retrieval backend: embeddings (default, OpenAI + NumPy, cached) "
                         "or tfidf (pure-Python, zero-dep). embeddings falls back to tfidf "
                         "if NumPy/API key are unavailable.")
    ap.add_argument("--dataset", choices=list(DATASETS), default="main",
                    help="Which KB to query: main (Plasmodium export + ESX) or "
                         "q3 (antimalarial drugs & resistance demo).")
    args = ap.parse_args()

    records, src = load_kb(DATASETS[args.dataset])
    if not records:
        sys.exit("[ERROR] No KB file found - run the extraction stages first.")
    qa = QA(records, model=args.model, retriever=args.retriever)
    print(f"[INFO] KB: {src} ({len(records)} papers) · retriever={qa.retriever}\n")
    res = qa.answer(args.question, k=args.k)
    print(res["answer"], "\n")
    print("Sources:")
    for s in res["sources"]:
        print(f"  - [PMID {s['pmid']}] {s['title']}  {s['url']}")


if __name__ == "__main__":
    main()
