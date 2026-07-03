import uuid

import chromadb

import config


class ChromaStore:
    """Persistent Chroma vector database (raw chromadb client, embeddings supplied externally)."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION
        )

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """
        Add chunks (each: {"text", "metadata"}) with their precomputed embeddings.
        Returns the number of vectors added.
        """
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        """Return top-k similar chunks as {"text", "metadata", "score"} dicts."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        docs = []
        if not results["ids"] or not results["ids"][0]:
            return docs

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert Chroma's L2/cosine distance to a similarity-style score.
            score = 1.0 / (1.0 + distance)
            docs.append({"text": text, "metadata": metadata, "score": score})

        return docs

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        sources = {m.get("source") for m in data.get("metadatas", []) if m}
        return sorted(s for s in sources if s)

    def delete_collection(self):
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION
        )
