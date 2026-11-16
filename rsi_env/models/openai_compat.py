"""OpenAI-compatible chat backend (vLLM / SGLang / OpenRouter / Gemini OpenAI endpoint ...).

This is how the paper-scale target model is served: start an inference server,
then use a spec such as::

    {"backend": "openai", "name": "Qwen/Qwen3.8-27B",
     "base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY"}

Providers differ in which sampling fields they accept. ``supports`` lists the
optional fields to forward (default: temperature, top_p). Add "seed" and "top_k"
for vLLM. Transient errors (429/5xx/timeouts) are retried with backoff.
"""
from __future__ import annotations

import os
import random
import time

DEFAULT_SUPPORTS = ("temperature", "top_p")

# Native tool calling ("native_tools": true in the spec). Harnesses keep their text protocol
# (<action>{json}</action> + "[tool:bash]" results); this adapter maps it onto the server's
# tools API in both directions, which is how open models are normally served with a tool parser.
NATIVE_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the working directory and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "compact", "description": "Summarize the interaction history to free context.",
        "parameters": {"type": "object", "properties": {}}}},
]
_ACTION_RE = __import__("re").compile(r"<action>\s*(\{.*?\})\s*</action>", __import__("re").S)


NATIVE_NOTE = ("\n\n[Tool interface] Act by calling the provided `bash` (and `compact`) function tools "
               "directly as native tool calls. Do not write <action> tags or tool-call markup in your text.")


def to_native(messages: list[dict]) -> list[dict]:
    import json

    out: list[dict] = []
    pending: str | None = None
    for i, m in enumerate(messages):
        if m["role"] == "system":
            out.append({"role": "system", "content": m["content"] + NATIVE_NOTE})
            continue
        if m["role"] == "user" and m["content"].startswith("[harness]") and "<action>" in m["content"]:
            m = {"role": "user", "content": "[harness] your tool call was malformed; call the `bash` tool natively."}
        if m["role"] == "assistant":
            match = _ACTION_RE.search(m["content"])
            if match:
                try:
                    act = json.loads(match.group(1))
                except json.JSONDecodeError:
                    act = None
                if act and act.get("tool") in ("bash", "compact"):
                    pending = f"call_{i}"
                    args = {k: v for k, v in act.items() if k != "tool"}
                    text = m["content"][:match.start()].strip()
                    out.append({"role": "assistant", "content": text or None, "tool_calls": [
                        {"id": pending, "type": "function",
                         "function": {"name": act["tool"], "arguments": json.dumps(args)}}]})
                    continue
            out.append({"role": "assistant", "content": m["content"]})
            pending = None
        elif m["role"] == "user" and pending and m["content"].startswith("[tool:bash]"):
            out.append({"role": "tool", "tool_call_id": pending, "content": m["content"][len("[tool:bash]"):].lstrip("\n")})
            pending = None
        else:
            if pending:  # tool call without a result (e.g. compact): close it
                out.append({"role": "tool", "tool_call_id": pending, "content": "(done)"})
                pending = None
            out.append(dict(m))
    if pending:
        out.append({"role": "tool", "tool_call_id": pending, "content": "(done)"})
    return out


def from_native(message) -> str:
    import json

    text = message.content or ""
    calls = message.tool_calls or []
    if not calls:
        return text
    c = calls[0]
    try:
        args = json.loads(c.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {"command": c.function.arguments}
    return (text + "\n" if text else "") + "<action>" + json.dumps({"tool": c.function.name, **args}) + "</action>"


class OpenAICompatModel:
    def __init__(self, spec: dict):
        from openai import OpenAI  # imported lazily so offline use needs no SDK

        self.spec = spec
        key = os.environ.get(spec.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
        base_url = os.environ.get(spec.get("base_url_env", ""), "") or spec.get("base_url")
        self.client = OpenAI(base_url=base_url, api_key=key,
                             timeout=float(spec.get("timeout", 300)), max_retries=0)
        self.supports = set(spec.get("supports", DEFAULT_SUPPORTS))
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def generate(self, messages: list[dict], sampling: dict | None = None) -> str:
        import openai

        s = dict(sampling or {})
        kwargs: dict = {"model": self.spec["name"], "messages": messages,
                        "max_tokens": min(int(s.get("max_tokens", 8192)), int(self.spec.get("max_tokens", 8192)))}
        for k in ("temperature", "top_p", "seed"):
            if k in self.supports and s.get(k) is not None:
                kwargs[k] = s[k]
        if "top_k" in self.supports and s.get("top_k") is not None:
            kwargs["extra_body"] = {"top_k": s["top_k"]}
        if self.spec.get("reasoning_effort"):
            kwargs["reasoning_effort"] = self.spec["reasoning_effort"]
        native = bool(self.spec.get("native_tools")) and not (
            messages and str(messages[-1].get("content", "")).startswith("[compact]"))
        if native:
            kwargs["messages"] = to_native(messages)
            kwargs["tools"] = NATIVE_TOOLS
        delay = 2.0
        empty_retries = 0
        for attempt in range(int(self.spec.get("retries", 8)) + 2):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self.usage["calls"] += 1
                if resp.usage:
                    self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
                text = ""
                if resp.choices:
                    msg = resp.choices[0].message
                    text = from_native(msg) if native else (msg.content or "")
                if not text.strip() and empty_retries < 2:  # transient empty generation: sample again
                    empty_retries += 1
                    continue
                return text
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                if attempt == int(self.spec.get("retries", 8)) - 1:
                    raise
                wait = delay + random.uniform(0, delay / 2)
                retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
                if hasattr(retry_after, "get") and retry_after.get("retry-after"):
                    try:
                        wait = max(wait, float(retry_after.get("retry-after")))
                    except ValueError:
                        pass
                time.sleep(min(wait, 90))
                delay = min(delay * 2, 60)
        return ""
