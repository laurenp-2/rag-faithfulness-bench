"""
FAISS-based dense retriever.

Embeds a corpus of contexts with sentence-transformers and indexes them in
FAISS.  At query time, returns the top-k most similar contexts.

Usage:
    retriever = Retriever()
    retriever.build_index(contexts)          # list[str]
    top_contexts = retriever.retrieve(question, k=3)
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self._index: faiss.IndexFlatIP | None = None
        self._corpus: list[str] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, contexts: list[str]) -> None:
        """Encode *contexts* and build a FAISS inner-product index."""
        self._corpus = contexts
        embeddings = self._encode(contexts)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype="float32")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k contexts most similar to *query*."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_vec = self._encode([query])
        k = min(k, len(self._corpus))
        _distances, indices = self._index.search(q_vec, k)
        return [self._corpus[i] for i in indices[0] if i >= 0]

    def retrieve_with_scores(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Return (context, similarity_score) tuples for the top-k results."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_vec = self._encode([query])
        k = min(k, len(self._corpus))
        distances, indices = self._index.search(q_vec, k)
        return [
            (self._corpus[i], float(distances[0][rank]))
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, index_path: str, corpus_path: str) -> None:
        import json, pathlib
        faiss.write_index(self._index, index_path)
        pathlib.Path(corpus_path).write_text(json.dumps(self._corpus))

    def load(self, index_path: str, corpus_path: str) -> None:
        import json, pathlib
        self._index = faiss.read_index(index_path)
        self._corpus = json.loads(pathlib.Path(corpus_path).read_text())


if __name__ == "__main__":
    corpus = [
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "Apple was founded by Steve Jobs and Steve Wozniak in 1976.",
        "Google was founded by Larry Page and Sergey Brin in 1998.",
        "Amazon was founded by Jeff Bezos in 1994.",
    ]
    r = Retriever()
    r.build_index(corpus)
    results = r.retrieve("Who founded Microsoft?", k=2)
    for res in results:
        print(res)
