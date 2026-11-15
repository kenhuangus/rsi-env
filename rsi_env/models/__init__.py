"""Model backend registry: turn a JSON model spec into a backend instance."""
from __future__ import annotations


def load_model(spec: dict):
    backend = spec.get("backend", "toy")
    if backend == "toy":
        from .toy import ToyModel
        return ToyModel(spec)
    if backend in ("openai", "vllm"):
        from .openai_compat import OpenAICompatModel
        return OpenAICompatModel(spec)
    if backend == "anthropic":
        from .anthropic_backend import AnthropicModel
        return AnthropicModel(spec)
    raise ValueError(f"unknown model backend {backend!r}")
