"""H0 - the minimal harness (paper Appendix A.1). This directory is editable by the Improver.

The harness calls the model with the task and interaction history, executes one
tool action, and appends its result. The two tools are ``bash`` (environment
interaction) and ``compact`` (model-generated history summary). Automatic
compaction starts near 90% context usage and always retains the original task.

Interface contract checked by the Host Controller (must be preserved by edits):

    build(model, runtime, settings, harness_dir) -> object with
        .run(task, sampling=None, messages=None) -> dict(final, messages, steps, tool_calls)
        .solve(task, sampling=None) -> str

Extensions: every name in ``harness.json["extensions"]`` is loaded from
``<harness_dir>/<name>.py`` and must define ``wrap_solve(solve, harness) -> solve``.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.S)
# Many open models emit their native XML tool-call syntax instead (e.g. Qwen-style
# <function=bash><parameter=command>...</parameter>). Accept it as the same action.
XML_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*(?=</parameter>|<parameter=|</function>|</tool_call>|$)", re.S)
TOOL_HINT_RE = re.compile(r"<action|</?tool_call|<function[=>]|<parameter[=>]", re.I)


def parse_action(reply: str):
    """Return (action_dict | None, error | None)."""
    m = ACTION_RE.search(reply)
    if m:
        try:
            act = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            return None, f"invalid action JSON: {e}"
        act.setdefault("tool", "bash" if "command" in act else None)
        return act, None
    params = dict(XML_PARAM_RE.findall(reply))
    if params:
        fn = re.search(r"<function=(\w+)>", reply)
        tool = params.get("tool") or (fn.group(1) if fn else ("bash" if "command" in params else None))
        return {"tool": tool, **{k: v for k, v in params.items() if k != "tool"}}, None
    if TOOL_HINT_RE.search(reply):
        return None, "could not parse your tool call; use <action>{\"tool\": \"bash\", \"command\": \"...\"}</action>"
    return None, None


def build(model, runtime, settings, harness_dir):
    return Harness(model, runtime, settings, Path(harness_dir))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"rsi_ext_{name}_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Harness:
    def __init__(self, model, runtime, settings: dict, harness_dir: Path):
        self.model, self.runtime, self.settings, self.dir = model, runtime, settings, harness_dir
        self.config = json.loads((harness_dir / "harness.json").read_text(encoding="utf-8"))
        self.solve = self._solve
        for name in self.config.get("extensions", []):
            self.solve = _load(harness_dir / f"{name}.py", name).wrap_solve(self.solve, self)

    # -------------------------------------------------------------- context
    @staticmethod
    def _tokens(messages) -> int:
        return sum(len(m["content"]) for m in messages) // 4

    def _compact(self, messages, task, sampling):
        history = "\n".join(f"{m['role']}: {m['content']}" for m in messages[2:])
        summary = self.model.generate(
            [{"role": "system", "content": "Summarize the agent history faithfully and concisely."},
             {"role": "user", "content": "[compact] " + history[-200_000:]}], sampling)
        return [messages[0], {"role": "user", "content": task},
                {"role": "user", "content": "[history summary]\n" + summary}]

    # -------------------------------------------------------------- loop
    def run(self, task: str, sampling: dict | None = None, messages: list | None = None) -> dict:
        msgs = messages or [{"role": "system", "content": self.config["system_prompt"]},
                            {"role": "user", "content": task}]
        threshold = int(self.settings.get("compaction_threshold", 235_929))
        steps = tool_calls = 0
        reply = ""
        while steps < int(self.config.get("max_steps", 60)) and not self.runtime.should_stop():
            if self._tokens(msgs) > threshold:
                msgs = self._compact(msgs, task, sampling)
            reply = self.model.generate(msgs, sampling)
            msgs.append({"role": "assistant", "content": reply})
            act, err = parse_action(reply)
            if err:
                msgs.append({"role": "user", "content": f"[harness] {err}"})
                steps += 1
                continue
            if act is None:
                return {"final": reply, "messages": msgs, "steps": steps, "tool_calls": tool_calls}
            if act.get("tool") == "compact":
                msgs = self._compact(msgs, task, sampling)
            elif act.get("tool") == "bash":
                out = self.runtime.bash(str(act.get("command", "")))
                tool_calls += 1
                msgs.append({"role": "user", "content": "[tool:bash]\n" + out})
            else:
                msgs.append({"role": "user", "content": f"[harness] unknown tool {act.get('tool')!r}"})
            steps += 1
        return {"final": reply, "messages": msgs, "steps": steps, "tool_calls": tool_calls, "stopped": True}

    def _solve(self, task: str, sampling: dict | None = None) -> str:
        return self.run(task, sampling)["final"]
