import config


class HybridRetriever:
    """Combine dense (Chroma) and sparse (BM25) retrieval, with optional reranking."""

    def __init__(self, chroma_store, bm25_store, embedding_model, reranker=None):
        self.chroma_store = chroma_store
        self.bm25_store = bm25_store
        self.embedding_model = embedding_model
        self.reranker = reranker  # optional cross-encoder, loaded lazily by caller
        self._corpus_chunks: list[dict] = []

    # ─── Corpus / BM25 maintenance ─────────────────────────────────────────

    def update_corpus(self, new_chunks: list[dict]) -> None:
        """Add newly ingested chunks to the in-memory BM25 corpus and rebuild it."""
        self._corpus_chunks.extend(new_chunks)
        self._build_bm25(self._corpus_chunks)

    def _build_bm25(self, chunks: list[dict]) -> None:
        self.bm25_store.build(chunks)

    # ─── Retrieval ──────────────────────────────────────────────────────────

    def retrieve_documents(
        self,
        query: str,
        top_k: int = None,
        rerank_top_k: int = None,
        use_reranking: bool = False,
    ) -> list[dict]:
        top_k = top_k or config.TOP_K
        rerank_top_k = rerank_top_k or config.RERANK_TOP_K

        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.chroma_store.query(query_embedding, k=top_k)
        bm25_results = self.bm25_store.search(query, k=top_k)

        merged = self._merge(vector_results, bm25_results, top_k)

        if use_reranking and self.reranker is not None and merged:
            merged = self.reranker.rerank(query, merged, top_k=rerank_top_k)
        elif use_reranking and rerank_top_k:
            merged = merged[:rerank_top_k]

        return merged

    @staticmethod
    def _merge(vector_results: list[dict], bm25_results: list[dict], k: int) -> list[dict]:
        """Reciprocal-style merge of two ranked lists, deduplicated by text."""
        scores: dict[str, float] = {}
        docs_by_text: dict[str, dict] = {}

        for rank, doc in enumerate(vector_results):
            s = 1.0 - (rank / max(len(vector_results), 1))
            scores[doc["text"]] = scores.get(doc["text"], 0) + config.HYBRID_ALPHA * s
            docs_by_text[doc["text"]] = doc

        for rank, doc in enumerate(bm25_results):
            s = 1.0 - (rank / max(len(bm25_results), 1))
            scores[doc["text"]] = scores.get(doc["text"], 0) + (1 - config.HYBRID_ALPHA) * s
            docs_by_text.setdefault(doc["text"], doc)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]

        results = []
        for text, score in ranked:
            doc = dict(docs_by_text[text])
            doc["score"] = score
            results.append(doc)

        return results
