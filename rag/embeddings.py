from sentence_transformers import SentenceTransformer

import config


class EmbeddingModel:
    """Generate embeddings for text using a SentenceTransformer model."""

    def __init__(self, model_key: str = None):
        self.model_key = model_key or config.DEFAULT_EMBEDDING_MODEL
        self.model_name = config.EMBEDDING_MODELS[self.model_key]
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).tolist()

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            convert_to_numpy=True,
        ).tolist()

    # Compatibility with langchain's Embeddings protocol
    def __call__(self, text: str):
        return self.embed_query(text)
