from groq import Groq

import config
from models.qwen_loader import QwenGenerator


class RAGGenerator:
    """Generate grounded answers using an LLM (Groq API or local Qwen) given retrieved context."""

    def __init__(self, registry):
        self.registry = registry

        self._groq_client = None
        if config.GROQ_API_KEY and config.GROQ_API_KEY != "your_groq_api_key":
            self._groq_client = Groq(api_key=config.GROQ_API_KEY)

        self._qwen_generator = None  # lazily instantiated + lazily loaded

    # ─── Model resolution ───────────────────────────────────────────────────

    def _resolve_model(self, model_name: str = None) -> str:
        if model_name and model_name not in config.LLM_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(config.LLM_MODELS)}"
            )
        return model_name or self.registry.current_model_name

    def _provider_for(self, model: str) -> str:
        return config.LLM_MODELS[model]["provider"]

    def _get_qwen(self) -> QwenGenerator:
        if self._qwen_generator is None:
            self._qwen_generator = QwenGenerator()
        return self._qwen_generator

    # ─── Context / sources helpers ──────────────────────────────────────────

    @staticmethod
    def _build_sources(documents: list[dict]) -> list[dict]:
        sources = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            text = doc.get("text", "")
            sources.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "page_number": metadata.get("page", 0),
                    "score": round(float(doc.get("score", 0.0)), 4),
                    "excerpt": text[:300] + ("…" if len(text) > 300 else ""),
                }
            )
        return sources

    @staticmethod
    def _build_context(documents: list[dict]) -> str:
        parts = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            tag = f"[{metadata.get('source', 'unknown')} p.{metadata.get('page', '?')}]"
            parts.append(f"{tag}\n{doc.get('text', '')}")
        return "\n\n---\n\n".join(parts)

    # ─── Provider dispatch ───────────────────────────────────────────────────

    def _call_llm(self, model: str, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        provider = self._provider_for(model)

        if provider == "groq":
            return self._call_groq(model, system_prompt, user_prompt)
        elif provider == "qwen":
            return self._call_qwen(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported provider '{provider}' for model '{model}'.")

    def _call_groq(self, model: str, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        if self._groq_client is None:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add a valid key to your .env file."
            )

        response = self._groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return answer, usage

    def _call_qwen(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        qwen = self._get_qwen()
        answer = qwen.generate(system_prompt, user_prompt)
        # Local inference: no token accounting from an API, report what we can.
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        return answer, usage

    # ─── Public API ──────────────────────────────────────────────────────────

    def generate_answer(self, question: str, documents: list[dict], model_name: str = None) -> dict:
        model = self._resolve_model(model_name)
        context = self._build_context(documents)

        system_prompt = (
            "You are a precise RAG assistant. Answer strictly using the provided "
            "context. If the answer is not contained in the context, say you "
            "don't know rather than guessing."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        answer, usage = self._call_llm(model, system_prompt, user_prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": self._build_sources(documents),
            "model": model,
            "retrieved_count": len(documents),
            "usage": usage,
        }

    def answer_without_context(self, question: str, model_name: str = None) -> dict:
        model = self._resolve_model(model_name)

        system_prompt = (
            "You are a helpful assistant. No relevant context was found in the "
            "knowledge base, so answer from general knowledge and make clear "
            "that this answer is not grounded in the uploaded documents."
        )
        user_prompt = f"Question: {question}\n\nAnswer:"

        answer, usage = self._call_llm(model, system_prompt, user_prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "model": model,
            "retrieved_count": 0,
            "usage": usage,
        }
