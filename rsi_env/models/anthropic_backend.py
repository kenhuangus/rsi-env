"""Claude backend (Anthropic Python SDK), usable as an external Teacher Improver.

Spec example::

    {"backend": "anthropic", "name": "claude-opus-5-5", "effort": "high"}

Sampling parameters (temperature/top_p/top_k) are not sent: current Claude models
reject them. Depth is controlled with ``output_config.effort``; thinking runs
adaptively by default. Requests stream and use ``get_final_message()`` so long
agent turns do not hit HTTP timeouts. Server-side refusal fallbacks are enabled
by default (beta ``server-side-fallback-2026-07-01``, ``fallbacks: "default"``).
"""
from __future__ import annotations

from .base import split_system


class AnthropicModel:
    def __init__(self, spec: dict):
        import anthropic  # lazy import

        self.spec = spec
        self.client = anthropic.Anthropic()

    def generate(self, messages: list[dict], sampling: dict | None = None) -> str:
        system, rest = split_system(messages)
        kwargs = dict(
            model=self.spec.get("name", "claude-opus-5-5"),
            max_tokens=int(self.spec.get("max_tokens", 64000)),
            messages=rest,
            output_config={"effort": self.spec.get("effort", "high")},
        )
        if system:
            kwargs["system"] = system
        if self.spec.get("fallbacks", True):
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["extra_body"] = {"fallbacks": "default"}
        with self.client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            return f"[model refused: {cat}]"
        return "".join(b.text for b in msg.content if b.type == "text")
