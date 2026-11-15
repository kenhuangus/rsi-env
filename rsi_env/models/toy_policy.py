"""Scripted research policy for the toy model acting as an Improver.

This deterministic policy is a stand-in for an LLM Improver. Every move is an
ordinary bash command run inside the RSI Environment sandbox, and every service
call goes through the controlled ``rsi`` interface, so the policy exercises the
real infrastructure:

1. inspect the active system, lineage, journal and experiment records;
2. pick the next untried improvement *operation* (journal markers ``[op-done] ...``);
3. build independent local evidence (or request benchmark feedback when allowed);
4. build a candidate (harness edit or SFT model) and compare it with the active one;
5. record the outcome in the journal, then submit only when the candidate is better;
6. return ``TASK_COMPLETE`` when no justified improvement is left.

Because the policy is stateless (it re-derives its plan from the transcript and the
journal), it naturally continues across rounds and across *system versions*, which
is exactly the recursive continuation that RSI-Env is built to support.
"""
from __future__ import annotations

import json
import re

from .toy import action

OPS = {
    "self": {
        "general": ["harness:self_consistency", "harness:postcondition"],
        "reasoning": ["harness:self_consistency"],
        "knowledge": ["harness:self_consistency", "sft:knowledge"],
        "agentic": ["harness:postcondition"],
    },
    "teacher": {  # a conservative external Teacher: harness edits only, never trains
        "general": ["harness:self_consistency"],
        "reasoning": ["harness:self_consistency"],
        "knowledge": ["harness:self_consistency"],
        "agentic": ["harness:postcondition"],
    },
}

SC_EXT = r'''"""Extension written by the Improver: self-consistency voting for tool-free answers."""
import re
from collections import Counter

def _key(text):
    m = re.findall(r"final answer\s*[:\-]\s*(.+)", text or "", re.I)
    return (m[-1] if m else (text or "").strip()).strip().strip(".").lower()

def wrap_solve(solve, harness):
    k = int(harness.config.get("self_consistency_k", 3))
    def solve_vote(task, sampling=None):
        before = harness.runtime.tool_calls
        first = solve(task, sampling)
        if harness.runtime.tool_calls > before:   # only vote on tool-free (reasoning / QA) turns
            return first
        finals = [first] + [solve(task, sampling) for _ in range(k - 1)]
        best = Counter(_key(f) for f in finals).most_common(1)[0][0]
        return next(f for f in finals if _key(f) == best)
    return solve_vote
'''

PC_EXT = r'''"""Extension written by the Improver: verify task postconditions after acting."""
CHECK = ("[postcondition-check] Inspect the workspace and verify every postcondition of the task. "
         "Fix anything that is wrong, then give the final answer.")

def wrap_solve(solve, harness):
    def solve_checked(task, sampling=None):
        r = harness.run(task, sampling)
        if not r["tool_calls"]:          # nothing to check: defer to the inner chain
            return r["final"] if solve == harness._solve else solve(task, sampling)
        if not harness.runtime.should_stop():
            r = harness.run(task, sampling, messages=r["messages"] + [{"role": "user", "content": CHECK}])
        return r["final"]
    return solve_checked
'''

KNOBS = {"harness:self_consistency": ("self_consistency", SC_EXT),
         "harness:postcondition": ("postcondition", PC_EXT)}

LOCAL_VAL_PY = r'''"""Independent local validation set built from public resources only."""
import json, random, re
random.seed(11)
PATS = [
    (r"The capital of the country (\w+) is (\w+)\.", "Name the capital city of {}."),
    (r"The (\w+) Institute was founded in (\d+)\.", "Give the founding year of the {} Institute."),
    (r"The (\w+) engine was invented by (\w+ \w+)\.", "Name the inventor of the {} engine."),
    (r"The (\w+) River is (\d+ km) long\.", "Give the length of the {} River, with units."),
]
rows = []
for line in open("resources/reference_corpus.jsonl", encoding="utf-8"):
    d = json.loads(line)
    for pat, q in PATS:
        m = re.fullmatch(pat, d["text"])
        if m:
            rows.append({"id": d["id"], "prompt": q.format(m.group(1)) + " Respond with the answer only.",
                         "answer": m.group(2), "source": d["id"]})
know = random.sample(rows, 40)
math = []
for i in range(20):
    a, b, c, d = (random.randint(11, 99), random.randint(11, 99), random.randint(10, 999), random.randint(17, 997))
    math.append({"id": f"m{i}", "prompt": f"Compute: ({a}*{b} + {c}) mod {d}. Give the integer answer.",
                 "answer": str((a * b + c) % d), "source": "synthetic"})
with open("data/local_val.jsonl", "w", encoding="utf-8") as f:
    for r in know + math:
        f.write(json.dumps(r) + "\n")
print(json.dumps({"local_val": len(know) + len(math), "knowledge": len(know), "synthetic_math": len(math)}))
'''

LOCAL_SHELL_PY = r'''"""Self-constructed shell tasks with postcondition checks (no benchmark access)."""
import json, random
random.seed(5)
rows = []
for i in range(3):
    nums = [random.randint(1, 900) for _ in range(12)]
    rows.append({"prompt": "[terminal-task] Write the sum of the integers in numbers.txt (one per line) to answer.txt.",
                 "files": {"numbers.txt": "\n".join(map(str, nums)) + "\n"},
                 "check": f'test "$(cat answer.txt)" = "{sum(nums)}"'})
    lines = [random.choice(["ERROR a", "INFO b", "WARN c"]) for _ in range(15)]
    rows.append({"prompt": "[terminal-task] Count the lines in app.log that contain ERROR and write the count to answer.txt.",
                 "files": {"app.log": "\n".join(lines) + "\n"},
                 "check": f'test "$(cat answer.txt)" = "{sum("ERROR" in x for x in lines)}"'})
    names = random.sample(["oak", "elm", "ash", "fir", "yew", "box", "bay"], 6)
    rows.append({"prompt": "[terminal-task] Sort the names in names.txt alphabetically and write them, one per line, to sorted.txt.",
                 "files": {"names.txt": "\n".join(names) + "\n"},
                 "check": 'sort names.txt | diff -q - sorted.txt'})
    rows.append({"prompt": "[terminal-task] Create a directory named out and copy every .csv file in the current directory into it.",
                 "files": {"x.csv": "a\n", "y.csv": "b\n", "readme.txt": "r\n"},
                 "check": 'test "$(ls out/*.csv | wc -l)" -eq 2 && test ! -e out/readme.txt'})
with open("data/local_shell.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps({"local_shell_tasks": len(rows)}))
'''

EDIT_HARNESS_PY = r'''"""Register an extension module in a candidate harness's harness.json."""
import json, sys
path, name = sys.argv[1] + "/harness.json", sys.argv[2]
cfg = json.load(open(path, encoding="utf-8"))
exts = cfg.setdefault("extensions", [])
if name not in exts:
    exts.append(name)
json.dump(cfg, open(path, "w", encoding="utf-8"), indent=2)
print(json.dumps({"extensions": exts}))
'''

SFT_PY = r'''"""Targeted SFT data: probe the live model on public-corpus facts; keep the ones it misses."""
import json, re
from rsi_env.envcli import Client
from rsi_env.benchmarks.suites import extract_final, normalize
c = Client.from_env()
PATS = [
    (r"The capital of the country (\w+) is (\w+)\.", "What is the capital of the country {}?"),
    (r"The (\w+) Institute was founded in (\d+)\.", "In what year was the {} Institute founded?"),
    (r"The (\w+) engine was invented by (\w+ \w+)\.", "Who invented the {} engine?"),
    (r"The (\w+) River is (\d+ km) long\.", "How long is the {} River?"),
]
facts = []
for line in open("resources/reference_corpus.jsonl", encoding="utf-8"):
    d = json.loads(line)
    for pat, q in PATS:
        m = re.fullmatch(pat, d["text"])
        if m:
            facts.append((d["id"], q.format(m.group(1)) + " Respond with the answer only.", m.group(2)))
missed = []
for doc_id, q, a in facts:
    outs = [c.infer(q, model="active", seed=s) for s in (1, 2)]
    if any(normalize(extract_final(o)) != normalize(a) for o in outs):
        missed.append((doc_id, q, a))
sel = missed[:48]
with open("data/sft.jsonl", "w", encoding="utf-8") as f:
    for _, q, a in sel:
        f.write(json.dumps({"prompt": q, "completion": "Final answer: " + a}) + "\n")
prov = {"source": "resources/reference_corpus.jsonl", "method": "probe active model x2, keep misses",
        "probed": len(facts), "missed": len(missed), "selected": len(sel), "doc_ids": [d for d, _, _ in sel]}
json.dump(prov, open("data/sft_provenance.json", "w", encoding="utf-8"), indent=2)
print(json.dumps({k: v for k, v in prov.items() if k != "doc_ids"}))
'''


def heredoc(path: str, body: str) -> str:
    return f"mkdir -p $(dirname {path}) && cat > {path} <<'PYEOF'\n{body}PYEOF\n"


def score_of(text: str) -> float | None:
    m = re.findall(r'"score":\s*(-?[\d.]+)', text or "")
    return float(m[-1]) if m else None


class ResearchPolicy:
    def __init__(self, model):
        self.model = model

    def step(self, messages: list[dict]) -> str:
        users = [m["content"] for m in messages if m["role"] == "user"]
        task = users[0]
        obs = users[1:]
        k = sum(1 for m in messages if m["role"] == "assistant")
        meta = dict(re.findall(r"(\w+)=([\w,\-]+)", task.splitlines()[0]))
        if k == 0:
            return "Inspect the active system and its lineage first.\n" + action("rsi status && rsi lineage")
        if k == 1:
            return "Read retained research state.\n" + action(
                "cat journal.md 2>/dev/null; echo '--- experiments:'; ls experiments 2>/dev/null")
        status = self._status(obs[0])
        journal = obs[1] if len(obs) > 1 else ""
        done = set(re.findall(r"\[op-done\] (\S+)", journal))
        ops = OPS[self.model.profile].get(meta.get("capability", "general"), [])
        op = next((o for o in ops if o not in done), None)
        steps = self._conclude(status) if op is None else self._op_steps(op, meta, status)
        i = k - 2
        if i >= len(steps):
            return "Final answer: round finished."
        return steps[i](obs)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _status(text: str) -> dict:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {}

    def _eval(self, meta: dict, harness: str, model: str, data: str, tag: str, rnd) -> str:
        fb = meta.get("feedback")
        out = f"experiments/r{rnd}_{tag}.json"
        if fb:
            cmd = f"rsi feedback --benchmark {fb.split(',')[0]} --harness {harness} --model {model}"
        else:
            cmd = f"rsi local-eval --harness {harness} --model {model} --data {data} --trials 2"
        return f"mkdir -p experiments && {cmd} | tee {out}"

    def _op_steps(self, op: str, meta: dict, status: dict):
        rnd = status.get("round", 0)
        agentic = op == "harness:postcondition"
        data = "data/local_shell.jsonl" if agentic else "data/local_val.jsonl"
        name = op.replace(":", "_")
        build = (heredoc("scripts/build_local_val.py", LOCAL_VAL_PY)
                 + heredoc("scripts/build_local_shell.py", LOCAL_SHELL_PY)
                 + heredoc("scripts/edit_harness.py", EDIT_HARNESS_PY)
                 + ("" if not op.startswith("sft") else heredoc("scripts/build_sft.py", SFT_PY))
                 + "mkdir -p data && \"$RSI_PYTHON\" scripts/build_local_val.py && "
                   "\"$RSI_PYTHON\" scripts/build_local_shell.py")
        steps = [
            lambda o: f"Hypothesis for {op}. Build independent local evidence first.\n" + action(build),
            lambda o: "Measure the active system as the baseline.\n"
                      + action(self._eval(meta, "active", "active", data, f"{name}_baseline", rnd)),
        ]
        if op.startswith("harness"):
            ext, code = KNOBS[op]
            steps.append(lambda o: f"Write a candidate harness with a new '{ext}' extension.\n" + action(
                "rm -rf candidate/harness && mkdir -p candidate && cp -r active/harness candidate/harness && "
                + heredoc(f"candidate/harness/{ext}.py", code)
                + f"\"$RSI_PYTHON\" scripts/edit_harness.py candidate/harness {ext}"))
            steps.append(lambda o: "Evaluate the candidate under the same protocol.\n" + action(
                self._eval(meta, "candidate/harness", "active", data, f"{name}_candidate", rnd)))
            cand_h, cand_m, changed = "candidate/harness", "active", f"candidate/harness/{ext}.py (+harness.json)"
        else:
            steps.append(lambda o: "Construct targeted SFT data from the public corpus and train.\n" + action(
                "\"$RSI_PYTHON\" scripts/build_sft.py && "
                f"rsi train-sft --data data/sft.jsonl --epochs 2 --lr 1e-4 | tee experiments/r{rnd}_{name}_train.json"))
            steps.append(lambda o: "Evaluate the trained model with the active harness.\n" + action(
                self._eval(meta, "active", self._cand(o[-1]), data, f"{name}_candidate", rnd)))
            cand_h, cand_m, changed = "active", None, "model weights (SFT, see data/sft_provenance.json)"

        def decide(o):
            base, cand = score_of(o[3]), score_of(o[5])
            model = cand_m or self._cand(o[4])
            better = base is not None and cand is not None and cand > base
            entry = (f"## Round {rnd} - {op} (active {status.get('active', '?')})\n"
                     f"- Hypothesis: {op} improves {meta.get('capability')} without regressions.\n"
                     f"- Evidence: baseline={base} candidate={cand} ({'feedback' if meta.get('feedback') else data})\n"
                     f"- Changed: {changed}\n"
                     f"- Decision: {'submit candidate' if better else 'reject candidate'}\n"
                     f"[op-done] {op}\n")
            cmd = f"cat >> journal.md <<'JEOF'\n{entry}JEOF\n"
            if better:
                cmd += f"rsi submit --harness {cand_h} --model {model}"
                return "Candidate is better on local evidence; recording and submitting.\n" + action(cmd)
            return "Candidate not better; recording the negative result.\n" + action(cmd)

        steps.append(decide)
        steps.append(lambda o: f"Final answer: {op} rejected on local evidence; recorded in the journal.")
        return steps

    def _conclude(self, status: dict):
        rnd = status.get("round", 0)
        entry = (f"## Round {rnd} - conclusion (active {status.get('active', '?')})\n"
                 "- All planned operations tried; no further justified improvement identified.\n")
        return [
            lambda o: "No untried operation remains; recording the conclusion.\n"
                      + action(f"cat >> journal.md <<'JEOF'\n{entry}JEOF\necho recorded"),
            lambda o: "TASK_COMPLETE",
        ]

    @staticmethod
    def _cand(text: str) -> str:
        m = re.search(r'"candidate_model":\s*"([^"]+)"', text or "")
        return m.group(1) if m else "active"
