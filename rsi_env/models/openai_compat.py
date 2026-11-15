"""OpenAI-compatible chat backend (e.g. vLLM / SGLang serving Qwen3.8-27B).

This is how the paper-scale target model is served: start an inference server,
then use a spec such as::

    {"backend": "openai", "name": "Qwen/Qwen3.8-27B",
     "base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY"}

A model produced by training is served the same way with a new ``name`` (for
example a LoRA adapter name registered with the inference server).
"""
from __future__ import annotations

import os


class OpenAICompatModel:
    def __init__(self, spec: dict):
        from openai import OpenAI  # imported lazily so offline use needs no SDK

        self.spec = spec
        key = os.environ.get(spec.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
        self.client = OpenAI(base_url=spec.get("base_url"), api_key=key)

    def generate(self, messages: list[dict], sampling: dict | None = None) -> str:
        s = dict(sampling or {})
        extra = {}
        if "top_k" in s:  # vLLM accepts top_k through extra_body
            extra["top_k"] = s.pop("top_k")
        resp = self.client.chat.completions.create(
            model=self.spec["name"],
            messages=messages,
            temperature=s.get("temperature", 1.0),
            top_p=s.get("top_p", 0.95),
            max_tokens=min(int(s.get("max_tokens", 8192)), int(self.spec.get("max_tokens", 81920))),
            seed=s.get("seed"),
            extra_body=extra or None,
        )
        return resp.choices[0].message.content or ""
