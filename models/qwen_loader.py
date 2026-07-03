"""
Local Qwen inference via Hugging Face `transformers`.

The model is loaded lazily on first use (loading a multi-GB checkpoint at
import time would make every route — including /health — slow to boot).
"""

import logging

import config

logger = logging.getLogger(__name__)


def detect_available_gpu() -> dict:
    """Report whether a CUDA (or Apple MPS) device is available for Qwen."""
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}

    info = {"torch_installed": True, "cuda_available": torch.cuda.is_available()}

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["mps_available"] = True

    return info


class QwenGenerator:
    """Loads Qwen once and answers chat-style prompts locally."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.QWEN_MODEL
        self.device_preference = device or config.QWEN_DEVICE
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading Qwen model '{self.model_name}' — this may take a while…")

        if self.device_preference == "auto":
            device_map = "auto" if torch.cuda.is_available() else None
        else:
            device_map = self.device_preference

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=device_map,
        )

        logger.info(f"Qwen model '{self.model_name}' loaded.")

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = None) -> str:
        self._load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or config.QWEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.2,
        )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
