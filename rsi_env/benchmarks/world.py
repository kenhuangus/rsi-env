"""A deterministic synthetic world used by the offline mini-benchmarks.

Facts are published to the Improver as a *public reference corpus* (like an
encyclopedia dump that is part of the resources R). Private benchmark items are
phrased with a template the Improver never sees, so legitimate training on the
corpus can be told apart from contamination with benchmark prompts.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from functools import lru_cache

SYL = ["vel", "mor", "ost", "rin", "cal", "dra", "ven", "tal", "is", "qua", "bel", "nor",
       "zan", "thi", "ul", "pex", "ka", "sor", "lun", "eth", "ar", "gol", "mi", "ra"]

RELATIONS = {
    # relation: (corpus sentence, template A (train-style), template B (paraphrase), template C (private))
    "capital": (
        "The capital of the country {e} is {v}.",
        "What is the capital of the country {e}?",
        "Name the capital city of {e}.",
        "Which city serves as the seat of government of the country {e}?",
    ),
    "founded": (
        "The {e} Institute was founded in {v}.",
        "In what year was the {e} Institute founded?",
        "Give the founding year of the {e} Institute.",
        "The {e} Institute was established in which year?",
    ),
    "inventor": (
        "The {e} engine was invented by {v}.",
        "Who invented the {e} engine?",
        "Name the inventor of the {e} engine.",
        "Which person is credited with creating the {e} engine?",
    ),
    "length": (
        "The {e} River is {v} long.",
        "How long is the {e} River?",
        "Give the length of the {e} River, with units.",
        "What is the total length of the {e} River (include units)?",
    ),
}


@dataclass(frozen=True)
class Fact:
    relation: str
    entity: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.relation}:{self.entity}"

    def sentence(self) -> str:
        return RELATIONS[self.relation][0].format(e=self.entity, v=self.value)

    def question(self, template: str) -> str:
        idx = {"A": 1, "B": 2, "C": 3}[template]
        return RELATIONS[self.relation][idx].format(e=self.entity)


def _name(rng: random.Random, parts: int) -> str:
    return "".join(rng.choice(SYL) for _ in range(parts)).capitalize()


@lru_cache(maxsize=4)
def world_facts(seed: int = 7, per_relation: int = 50) -> tuple[Fact, ...]:
    rng = random.Random(seed)
    used: set[str] = set()
    facts: list[Fact] = []

    def fresh(parts: int) -> str:
        while True:
            n = _name(rng, parts)
            if n not in used:
                used.add(n)
                return n

    for rel in RELATIONS:
        for _ in range(per_relation):
            e = fresh(3)
            if rel == "capital":
                v = fresh(2)
            elif rel == "founded":
                v = str(rng.randint(1601, 1999))
            elif rel == "inventor":
                v = f"{fresh(2)} {fresh(3)}"
            else:
                v = f"{rng.randint(40, 4200)} km"
            facts.append(Fact(rel, e, v))
    return tuple(facts)


@lru_cache(maxsize=4)
def fact_index(seed: int = 7) -> dict[str, Fact]:
    return {f.key: f for f in world_facts(seed)}


def _template_regex(tpl: str) -> re.Pattern:
    pat = re.escape(tpl).replace(re.escape("{e}"), r"(?P<e>[A-Za-z]+)")
    return re.compile(pat, re.I)


_PARSERS = [(rel, _template_regex(t)) for rel, tpls in RELATIONS.items() for t in tpls[1:]]


def parse_fact_question(text: str) -> str | None:
    """Return the fact key a question asks about (any template), or None."""
    for rel, rx in _PARSERS:
        m = rx.search(text)
        if m:
            return f"{rel}:{m.group('e')}"
    return None


def stable_unit(*parts: object) -> float:
    """Deterministic pseudo-random number in [0,1) from arbitrary parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)
