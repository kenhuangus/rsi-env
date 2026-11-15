"""Model backend protocol.

Every backend exposes ``generate(messages, sampling) -> str`` where ``messages`` is a
list of ``{"role": "system"|"user"|"assistant", "content": str}``. Harnesses only use
this text interface, so a harness written for one backend runs on any other.
"""
from __future__ import annotations

from typing import Protocol


class ModelBackend(Protocol):
    spec: dict

    def generate(self, messages: list[dict], sampling: dict | None = None) -> str: ...


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest
