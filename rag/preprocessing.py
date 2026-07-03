import re


class TextPreprocessor:
    """Clean extracted text before chunking."""

    @staticmethod
    def clean(text: str) -> str:
        # Remove extra spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove multiple blank lines
        text = re.sub(r"\n+", "\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @classmethod
    def clean_pages(cls, pages: list[dict]) -> list[dict]:
        """Clean the 'text' field of a list of page dicts in place and return them."""
        for page in pages:
            page["text"] = cls.clean(page["text"])
        return pages
