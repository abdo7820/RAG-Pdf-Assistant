from pathlib import Path
import fitz  # PyMuPDF


class PDFIngestion:
    """Extract text from PDF documents."""

    def process_pdf(self, pdf_path: str | Path) -> list[dict]:
        """
        Extract per-page text from a PDF.

        Returns a list of dicts: {"text": ..., "metadata": {"source": ..., "page": ...}}
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} does not exist.")

        pages = []

        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    }
                )

        return pages

    # Kept for backwards compatibility with the old interface.
    def load(self, pdf_path: str | Path) -> list[dict]:
        pages = self.process_pdf(pdf_path)
        return [
            {"page_content": p["text"], "metadata": p["metadata"]}
            for p in pages
        ]
