"""Loaders for top published benchmarks, mapped onto RSI-Env's tool-based agents.

The synthetic ``research`` / ``research-pro`` domains are reproducible and leak-free,
but the agents should also be measured on, and improve against, the standard
benchmarks the field reports. These are the published multi-hop research benchmarks
whose items ship with their own supporting documents, so they map directly onto the
agent's ``search.py`` / ``fetch.py`` tools (no live web needed, evaluation stays fair
and offline):

* **HotpotQA** (distractor setting) - Yang et al., EMNLP 2018. 10 paragraphs per
  question (2 gold + 8 distractors); 2-hop. CC BY-SA 4.0.
* **2WikiMultiHopQA** - Ho et al., COLING 2020. Multi-hop with evidence; Apache-2.0.
* **MuSiQue** (answerable) - Trivedi et al., TACL 2022. 2-4 hop, 20 paragraphs; CC BY 4.0.

Each loader converts the dataset's *official* released format into RSI-Env's item
schema (``kind: "research_ext"`` carries the per-question context documents), and the
worker grades answers with the standard SQuAD exact-match / token-F1 metric.

Datasets are read from a LOCAL file you downloaded from the official source - nothing
is fetched or sent anywhere by this module. See ``docs/BENCHMARKS.md`` for the
one-line download per dataset. Related red-team benchmarks (Gandalf, TensorTrust,
AgentDojo, InjecAgent) plug into the same adapter pattern via LLM-backed guarded
targets; that loader is tracked separately.
"""
from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------- SQuAD-style grading
def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def grade_squad(item: dict, response: str, f1_threshold: float = 0.6) -> bool:
    """Correct if the extracted final answer exactly matches any gold answer (normalized),
    or reaches the SQuAD token-F1 threshold against any gold answer."""
    m = re.findall(r"final answer\s*[:\-]\s*(.+)", response or "", re.I | re.S)
    pred = (m[-1] if m else (response or "")).strip()
    golds = [item["answer"], *item.get("aliases", [])]
    np = _normalize(pred)
    for g in golds:
        if np == _normalize(g) or squad_f1(pred, g) >= f1_threshold:
            return True
    return False


# ---------------------------------------------------------------- shared item builder
def task_prompt(question: str, budget: int = 25) -> str:
    return ("[research-ext-task] Answer the question using ONLY the local document collection in the current "
            "directory (do not rely on memory; the gold answer is supported by these documents). Tools: "
            "python search.py \"query\" [-k N] and python fetch.py DOC_ID. You may use at most "
            f"{budget} tool calls. Decompose multi-hop questions and track intermediate facts. Give a SHORT, "
            "exact answer (a name, entity, number or yes/no), not a sentence. "
            f"Question: {question}\nEnd your reply with 'Final answer: <answer>'.")


def _item(bench: str, i: int, qid: str, question: str, answer: str, context: list[dict],
          aliases: list[str] | None = None) -> dict:
    return {"id": f"{bench}-{i:04d}", "ext_id": str(qid), "kind": "research_ext", "benchmark": bench,
            "prompt": task_prompt(question), "question": question, "answer": str(answer),
            "aliases": [str(a) for a in (aliases or []) if str(a).strip()], "context": context}


def _ctx_from_title_sentences(context) -> list[dict]:
    """HotpotQA / 2Wiki format: context = [[title, [sent, sent, ...]], ...]."""
    out = []
    for entry in context:
        title, sents = entry[0], entry[1]
        text = " ".join(s.strip() for s in sents).strip()
        out.append({"title": str(title), "text": text})
    return out


# ---------------------------------------------------------------- per-benchmark loaders
def _read_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_hotpotqa(path: str | Path) -> list[dict]:
    rows = _read_json_or_jsonl(Path(path))
    out = []
    for i, r in enumerate(rows):
        ctx = _ctx_from_title_sentences(r["context"])
        out.append(_item("hotpotqa", i, r.get("_id", i), r["question"], r["answer"], ctx))
    return out


def load_2wiki(path: str | Path) -> list[dict]:
    rows = _read_json_or_jsonl(Path(path))
    out = []
    for i, r in enumerate(rows):
        ctx = _ctx_from_title_sentences(r["context"])
        out.append(_item("2wiki", i, r.get("_id", i), r["question"], r["answer"], ctx))
    return out


def load_musique(path: str | Path) -> list[dict]:
    rows = _read_json_or_jsonl(Path(path))
    out = []
    for i, r in enumerate(rows):
        ctx = [{"title": p.get("title", f"p{j}"), "text": p["paragraph_text"]}
               for j, p in enumerate(r["paragraphs"])]
        out.append(_item("musique", i, r.get("id", i), r["question"], r["answer"],
                         ctx, aliases=r.get("answer_aliases")))
    return out


LOADERS = {"hotpotqa": load_hotpotqa, "2wiki": load_2wiki, "musique": load_musique}


def load_benchmark(name: str, path: str | Path, n: int | None = None, seed: int = 7) -> list[dict]:
    if name not in LOADERS:
        raise KeyError(f"unknown published benchmark {name!r}; choose from {sorted(LOADERS)}")
    items = LOADERS[name](path)
    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
        for j, it in enumerate(items):
            it["id"] = f"{name}-{j:04d}"
    return items


def context_corpus(item: dict) -> list[dict]:
    """Turn one item's context into the corpus.json the agent's tools read."""
    return [{"id": f"doc-{j:03d}", "title": c["title"], "text": c["text"]}
            for j, c in enumerate(item["context"])]
