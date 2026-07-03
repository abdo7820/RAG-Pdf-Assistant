from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class TextChunker:
    """Split cleaned page text into smaller overlapping chunks."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, pages: list[dict]) -> list[dict]:
        """
        Split a list of page dicts ({"text", "metadata"}) into chunk dicts.

        Returns a list of dicts: {"text": ..., "metadata": {"source", "page", "chunk_index"}}
        """
        chunks = []

        for page in pages:
            pieces = self.splitter.split_text(page["text"])

            for i, piece in enumerate(pieces):
                chunks.append(
                    {
                        "text": piece,
                        "metadata": {
                            **page["metadata"],
                            "chunk_index": i,
                        },
                    }
                )

        return chunks

    # Kept for backwards compatibility with the old Document-based interface.
    def split(self, documents: list[dict]):
        from langchain_core.documents import Document

        pages = [
            {"text": d["page_content"], "metadata": d["metadata"]}
            for d in documents
        ]
        chunks = self.chunk_text(pages)
        return [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]
