from rank_bm25 import BM25Okapi


class BM25Store:
    """BM25-based sparse retriever over dict chunks ({"text", "metadata"})."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.bm25 = None

    def build(self, chunks: list[dict]) -> None:
        """Build (or rebuild) the BM25 index from a list of chunk dicts."""
        self.chunks = chunks
        corpus = [c["text"].split() for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, chunks: list[dict]) -> None:
        """Add new chunks and rebuild the index."""
        self.build(self.chunks + chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k most relevant chunks as {"text", "metadata", "score"}."""
        if self.bm25 is None:
            return []

        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )[:k]

        return [
            {"text": c["text"], "metadata": c["metadata"], "score": float(s)}
            for c, s in ranked
        ]
