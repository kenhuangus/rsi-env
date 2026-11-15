"""Deterministic offline "toy" model.

The toy model is a *simulator* that lets the whole RSI-Env pipeline (harness loop,
controller services, training, validity checks, versioning, recursive takeover and
independent evaluation) run end to end on a laptop with no GPU or API key. Its
capabilities are probabilistic and seeded, so the effect of a harness or model
update is measurable with avg@k / pass@k exactly like a real LLM:

* factual QA  - correct with p=0.85 if the fact is known (pretraining or SFT), else p=0.15
* math        - correct with p=0.55
* science MC  - correct with p=0.65 (minus 0.10 after any SFT: simulated interference,
                mirroring the paper's observation that gains coexist with regressions)
* shell tasks - emits the correct command with p=0.50; when a harness asks it to check
                postconditions it repairs a wrong result with p=0.70

SFT really changes the model: training pairs are written into the checkpoint and
answered from it (garbage training data -> garbage answers).

When it is asked to act as an Improver it follows :mod:`rsi_env.models.toy_policy`,
a scripted research policy that drives the environment through bash.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..benchmarks.world import fact_index, parse_fact_question, stable_unit
from ..tasks import IMPROVER_MARKER

P_KNOWN, P_UNKNOWN, P_MATH, P_MC, P_SHELL, P_REPAIR = 0.85, 0.15, 0.55, 0.65, 0.50, 0.70
MC_INTERFERENCE = 0.10
BASE_KNOWN_FRACTION = 0.60

SHELL_SKILLS = {
    # instruction prefix: (correct command, buggy command)
    "Write the sum of the integers in numbers.txt": (
        "python -c \"print(sum(int(x) for x in open('numbers.txt') if x.strip()))\" > answer.txt",
        "wc -l < numbers.txt | tr -d ' ' > answer.txt"),
    "Count the lines in app.log that contain ERROR": (
        "grep -c ERROR app.log > answer.txt", "grep -c error app.log > answer.txt"),
    "Sort the names in names.txt alphabetically": (
        "sort names.txt > sorted.txt", "sort -r names.txt > sorted.txt"),
    "Create a directory named out and copy every .csv file": (
        "mkdir -p out && cp *.csv out/", "mkdir -p out && cp * out/ 2>/dev/null; true"),
}


def _wrong_value(relation: str, rng: random.Random) -> str:
    if relation == "founded":
        return str(rng.randint(1601, 1999))
    if relation == "length":
        return f"{rng.randint(40, 4200)} km"
    if relation == "inventor":
        return "Unknown Person"
    return "Unknown"


class ToyModel:
    def __init__(self, spec: dict):
        self.spec = spec
        self.profile = spec.get("profile", "self")
        self.seed = int(spec.get("seed", 0))
        self.world_seed = int(spec.get("world_seed", 7))
        self.learned: dict[str, str] = {}
        ckpt = spec.get("checkpoint")
        if ckpt:
            self.learned = json.loads(Path(ckpt).read_text(encoding="utf-8")).get("learned", {})
        self._calls = 0

    # ------------------------------------------------------------------ public API
    def generate(self, messages: list[dict], sampling: dict | None = None) -> str:
        self._calls += 1
        sampling = sampling or {}
        users = [m["content"] for m in messages if m["role"] == "user"]
        task = users[0] if users else ""
        last = messages[-1]["content"] if messages else ""
        rng = random.Random(f"{self.seed}|{sampling.get('seed', 0)}|{self._calls}|{task}|{len(messages)}")

        if last.startswith("[compact]"):
            return "Summary of earlier work: " + task[:200]
        if IMPROVER_MARKER in task:
            from .toy_policy import ResearchPolicy
            return ResearchPolicy(self).step(messages)
        if task.startswith("[terminal-task]"):
            return self._shell(task, messages, rng)
        return "Final answer: " + self._qa(task, rng)

    # ------------------------------------------------------------------ skills
    def knows(self, key: str) -> bool:
        return key in self.learned or stable_unit("known", self.world_seed, key) < BASE_KNOWN_FRACTION

    def _qa(self, q: str, rng: random.Random) -> str:
        key = parse_fact_question(q)
        if key:
            rel = key.split(":")[0]
            fi = fact_index(self.world_seed)
            truth = self.learned.get(key) or (fi[key].value if key in fi else None)
            if truth is None:
                return "I don't know"
            p = P_KNOWN if self.knows(key) else P_UNKNOWN
            return truth if rng.random() < p else _wrong_value(rel, rng)
        m = re.search(r"Compute:\s*\((\d+)\*(\d+) \+ (\d+)\) mod (\d+)", q)
        if m:
            a, b, c, d = map(int, m.groups())
            v = (a * b + c) % d
            return str(v) if rng.random() < P_MATH else str(v + rng.choice([-3, -1, 1, 2, 5]))
        if re.search(r"\(A\).*\(B\).*\(C\).*\(D\)", q, re.S):
            return self._mc(q, rng)
        return "I don't know"

    def _mc(self, q: str, rng: random.Random) -> str:
        opts = dict(re.findall(r"\(([A-D])\)\s*([\d.]+)", q))
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:kg|m/s|A|ohm)", q)]
        val = None
        if "kinetic energy" in q and len(nums) >= 2:
            val = nums[0] * nums[1] ** 2 / 2
        elif "momentum" in q and len(nums) >= 2:
            val = nums[0] * nums[1]
        elif "voltage" in q and len(nums) >= 2:
            val = nums[0] * nums[1]
        correct = next((k for k, v in opts.items() if val is not None and float(v) == val), "A")
        p = P_MC - (MC_INTERFERENCE if self.learned else 0.0)
        if rng.random() < p:
            return correct
        return rng.choice([k for k in "ABCD" if k != correct])

    def _shell(self, task: str, messages: list[dict], rng: random.Random) -> str:
        skill = next((v for k, v in SHELL_SKILLS.items() if k in task), None)
        n_actions = sum(1 for m in messages if m["role"] == "assistant" and "<action>" in m["content"])
        last = messages[-1]["content"]
        if skill is None:
            return "Final answer: unable to complete"
        good, bad = skill
        if "[postcondition-check]" in last:
            if rng.random() < P_REPAIR:
                return ("Re-checking the postconditions and redoing the step.\n"
                        + action(good))
            return "Final answer: postconditions look satisfied"
        if n_actions == 0:
            return "Planning: one shell command should do it.\n" + action(good if rng.random() < P_SHELL else bad)
        return "Final answer: done"


def action(command: str) -> str:
    return "<action>" + json.dumps({"tool": "bash", "command": command}) + "</action>"
