"""Controller-side (private) benchmark suites and scoring.

Four offline suites mirror the paper's four complementary benchmarks:

=================  ===============================  ======  ==========
suite              stands in for                    metric  kind
=================  ===============================  ======  ==========
mini-gpqa          GPQA Diamond (science MC)        avg@k   qa
mini-aime          AIME 2026 (exact-match math)     avg@k   qa
mini-terminal      Terminal-Bench 2.1 (shell tasks) pass@k  terminal
mini-simpleqa      SimpleQA (short factual QA)      avg@k   qa
=================  ===============================  ======  ==========

Real benchmark data can be plugged in with :func:`load_jsonl_suite` (JSONL with
``prompt`` / ``answer`` fields), e.g. a local copy of GPQA Diamond.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .world import world_facts

ANSWER_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.I)


@dataclass
class Item:
    id: str
    prompt: str
    answer: str
    kind: str = "qa"                       # qa | mc | terminal
    meta: dict = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    metric: str                            # avg@k | pass@k
    kind: str                              # qa | terminal
    items: list[Item]


# ----------------------------------------------------------------------------- scoring
def extract_final(text: str) -> str:
    if text is None:
        return ""
    m = None
    for m in ANSWER_RE.finditer(text):
        pass
    ans = m.group(1) if m else text.strip().splitlines()[-1] if text.strip() else ""
    return ans.strip().strip(".").strip()


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = re.sub(r"[\s,]+", " ", s)
    s = s.replace("kilometres", "km").replace("kilometers", "km")
    return s.strip(" .\"'")


def grade(item: Item, response: str) -> bool:
    got = normalize(extract_final(response))
    want = normalize(item.answer)
    if item.kind == "mc":
        m = re.search(r"\(?([a-d])\)?", got)
        return bool(m) and m.group(1) == want
    if re.fullmatch(r"-?\d+(\.\d+)?", want):
        nums = re.findall(r"-?\d+(?:\.\d+)?", got)
        return bool(nums) and float(nums[0]) == float(want)
    return got == want


def avg_at_k(per_item_trials: list[list[bool]]) -> float:
    """Average accuracy over k trials (paper: avg@3)."""
    if not per_item_trials:
        return 0.0
    k = len(per_item_trials[0])
    return 100.0 * sum(sum(t) for t in per_item_trials) / (len(per_item_trials) * k)


def pass_at_k(per_item_trials: list[list[bool]]) -> float:
    """Proportion of items passed at least once in k trials (paper: pass@3)."""
    if not per_item_trials:
        return 0.0
    return 100.0 * sum(any(t) for t in per_item_trials) / len(per_item_trials)


def all_k(per_item_trials: list[list[bool]]) -> float:
    """Proportion passed in *all* trials (reported in parentheses for Terminal-Bench)."""
    if not per_item_trials:
        return 0.0
    return 100.0 * sum(all(t) for t in per_item_trials) / len(per_item_trials)


# ----------------------------------------------------------------------------- suites
def mini_simpleqa(seed: int = 7, n: int = 60) -> Suite:
    rng = random.Random(1000 + seed)
    facts = rng.sample(list(world_facts(seed)), n)
    items = [Item(f"sqa-{i:03d}", f.question("C") + " Respond with the answer only.", f.value,
                  meta={"fact": f.key}) for i, f in enumerate(facts)]
    return Suite("mini-simpleqa", "avg@k", "qa", items)


def aime_problem(rng: random.Random) -> tuple[str, str]:
    a, b, c, d = rng.randint(11, 99), rng.randint(11, 99), rng.randint(10, 999), rng.randint(17, 997)
    expr = f"({a}*{b} + {c}) mod {d}"
    value = (a * b + c) % d
    return f"Compute: {expr}. Give the integer answer.", str(value)


def mini_aime(seed: int = 7, n: int = 30) -> Suite:
    rng = random.Random(2000 + seed)
    items = []
    for i in range(n):
        p, a = aime_problem(rng)
        items.append(Item(f"aime-{i:02d}", p, a))
    return Suite("mini-aime", "avg@k", "qa", items)


def gpqa_problem(rng: random.Random) -> tuple[str, str]:
    kind = rng.choice(["ke", "p", "v"])
    if kind == "ke":
        m, v = rng.randint(2, 20), rng.randint(2, 15)
        q, val = f"A {m} kg object moves at {v} m/s. What is its kinetic energy in joules?", m * v * v / 2
    elif kind == "p":
        m, v = rng.randint(2, 30), rng.randint(2, 25)
        q, val = f"A {m} kg cart travels at {v} m/s. What is its momentum in kg*m/s?", m * v
    else:
        i_, r = rng.randint(2, 12), rng.randint(3, 60)
        q, val = f"A current of {i_} A flows through a {r} ohm resistor. What is the voltage in volts?", i_ * r
    opts = {val, val * 2, val / 2 if val % 2 == 0 else val + 3, val + 7}
    while len(opts) < 4:
        opts.add(val + rng.randint(8, 40))
    opts = list(opts)
    rng.shuffle(opts)
    letters = "ABCD"
    fmt = lambda x: str(int(x)) if float(x).is_integer() else f"{x:.1f}"  # noqa: E731
    body = q + "\n" + " ".join(f"({letters[j]}) {fmt(o)}" for j, o in enumerate(opts))
    return body + "\nAnswer with the letter.", letters[opts.index(val)].lower()


def mini_gpqa(seed: int = 7, n: int = 40) -> Suite:
    rng = random.Random(3000 + seed)
    items = []
    for i in range(n):
        p, a = gpqa_problem(rng)
        items.append(Item(f"gpqa-{i:02d}", p, a, kind="mc"))
    return Suite("mini-gpqa", "avg@k", "qa", items)


# --- terminal tasks: (instruction, files, verifier, reference solution) -------------
TerminalSpec = tuple[str, dict, Callable[[Path], bool], str]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _terminal_specs(rng: random.Random) -> list[TerminalSpec]:
    specs: list[TerminalSpec] = []
    for _ in range(3):
        nums = [rng.randint(1, 500) for _ in range(rng.randint(8, 20))]
        specs.append((
            "Write the sum of the integers in numbers.txt (one per line) to answer.txt.",
            {"numbers.txt": "\n".join(map(str, nums)) + "\n"},
            lambda d, s=sum(nums): _read(d / "answer.txt") == str(s),
            "python -c \"print(sum(int(x) for x in open('numbers.txt') if x.strip()))\" > answer.txt",
        ))
    for _ in range(3):
        lines = [rng.choice(["INFO ok", "ERROR disk", "WARN slow", "ERROR net", "INFO up"])
                 for _ in range(rng.randint(10, 30))]
        specs.append((
            "Count the lines in app.log that contain ERROR and write the count to answer.txt.",
            {"app.log": "\n".join(lines) + "\n"},
            lambda d, c=sum("ERROR" in x for x in lines): _read(d / "answer.txt") == str(c),
            "grep -c ERROR app.log > answer.txt",
        ))
    for _ in range(3):
        names = rng.sample(["delta", "alpha", "kilo", "echo", "bravo", "lima", "golf", "hotel",
                            "india", "juliet", "charlie", "foxtrot"], 7)
        specs.append((
            "Sort the names in names.txt alphabetically and write them, one per line, to sorted.txt.",
            {"names.txt": "\n".join(names) + "\n"},
            lambda d, n=sorted(names): _read(d / "sorted.txt").split() == n,
            "sort names.txt > sorted.txt",
        ))
    for _ in range(3):
        k = rng.randint(2, 5)
        files = {f"data{i}.csv": "a,b\n1,2\n" for i in range(k)}
        files["notes.txt"] = "not csv\n"
        specs.append((
            "Create a directory named out and copy every .csv file in the current directory into it.",
            files,
            lambda d, k=k: (d / "out").is_dir() and len(list((d / "out").glob("*.csv"))) == k
            and not (d / "out" / "notes.txt").exists(),
            "mkdir -p out && cp *.csv out/",
        ))
    return specs


def mini_terminal(seed: int = 7) -> Suite:
    rng = random.Random(4000 + seed)
    items = []
    for i, (instr, files, _verify, sol) in enumerate(_terminal_specs(rng)):
        items.append(Item(f"tb-{i:02d}", f"[terminal-task] {instr}", "", kind="terminal",
                          meta={"files": files, "solution": sol, "index": i}))
    return Suite("mini-terminal", "pass@k", "terminal", items)


def terminal_verifier(seed: int, index: int) -> Callable[[Path], bool]:
    rng = random.Random(4000 + seed)
    return _terminal_specs(rng)[index][2]


def load_jsonl_suite(name: str, path: str | Path, metric: str = "avg@k", kind: str = "qa") -> Suite:
    items = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            row = json.loads(line)
            items.append(Item(str(row.get("id", i)), row["prompt"], str(row["answer"]),
                              kind=row.get("kind", "qa")))
    return Suite(name, metric, kind, items)


SUITES: dict[str, Callable[..., Suite]] = {
    "mini-gpqa": mini_gpqa,
    "mini-aime": mini_aime,
    "mini-terminal": mini_terminal,
    "mini-simpleqa": mini_simpleqa,
}


def domain_suite(name: str, salt: str = "") -> Suite:
    """Private suites for the real-world domains; ``salt`` is a controller-only secret."""
    if name == "redteam-ctf":
        from ..domains import redteam
        rows = redteam.items("eval", salt, per_level=2)
    elif name == "research-qa":
        from ..domains import research
        rows = research.items("eval", salt, n=28)
    else:
        raise KeyError(name)
    kind = rows[0]["kind"]
    return Suite(name, "avg@k", kind, [Item(r["id"], r["prompt"], r.get("answer", ""), kind=kind,
                                            meta={"row": r}) for r in rows])


DOMAIN_SUITES = ("redteam-ctf", "research-qa")


def get_suite(name: str, seed: int = 7, salt: str = "") -> Suite:
    if name in DOMAIN_SUITES:
        return domain_suite(name, salt)
    if name in SUITES:
        return SUITES[name](seed)
    p = Path(name)
    if p.exists():
        return load_jsonl_suite(p.stem, p)
    raise KeyError(f"unknown benchmark {name!r}")


def public_corpus(seed: int = 7) -> list[dict]:
    """Public reference corpus placed in the RSI Environment's resources."""
    return [{"id": f"doc-{i:04d}", "text": f.sentence()} for i, f in enumerate(world_facts(seed))]
