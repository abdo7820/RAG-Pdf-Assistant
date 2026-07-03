import config


class ModelRegistry:
    """Tracks the active LLM and exposes metadata about available models."""

    def __init__(self, default_model: str = None):
        self.current_model_name = default_model or config.DEFAULT_MODEL

        if self.current_model_name not in config.LLM_MODELS:
            raise ValueError(
                f"Unknown default model '{self.current_model_name}'. "
                f"Available: {list(config.LLM_MODELS)}"
            )

    def list_models(self, qwen_loaded: bool = False) -> list[dict]:
        """
        List all registered models.

        Groq models are API-backed, so they're always considered "loaded".
        Qwen runs locally and is only loaded into memory on first use —
        pass `qwen_loaded` (e.g. from the generator's live state) to reflect that.
        """
        return [
            {
                "name": name,
                "provider": meta["provider"],
                "description": meta["description"],
                "loaded": True if meta["provider"] == "groq" else qwen_loaded,
                "active": name == self.current_model_name,
                "max_tokens": meta["max_tokens"],
                "context_window": meta["context_window"],
            }
            for name, meta in config.LLM_MODELS.items()
        ]

    def switch_model(self, model_name: str) -> dict:
        if model_name not in config.LLM_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(config.LLM_MODELS)}"
            )

        previous_model = self.current_model_name
        self.current_model_name = model_name

        return {
            "previous_model": previous_model,
            "current_model": self.current_model_name,
        }
